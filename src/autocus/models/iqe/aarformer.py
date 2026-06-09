"""AARFormer: Annotation Artifact Removal Transformer.

A dual-branch architecture for medical ultrasound annotation artifact removal
that decomposes the task into explicit artifact localization and content-aware
residual estimation, connected by an artifact-guided gating mechanism.

Architecture::

    Input → ContrastEnhancedStem → Encoder (4 stages)
          → Bottleneck (self-attention for global context)
          → Artifact Detection Decoder → multi-scale masks {M}
          → Restoration Decoder (with AAFM guided by M) → residual features
          → Artifact-Guided Gating → Boundary Refinement → residual
          → Output = clamp(Input − residual, 0, 1)

Key innovations:
    1. **Contrast-Enhanced Stem (CES)**: Learns high-pass features to exploit
       the high-contrast nature of annotation overlays.
    2. **Dual-Branch Decoder**: Artifact Detection Branch (ADB) produces soft
       masks; Restoration Branch uses them via AAFM.
    3. **Artifact-Aware Feature Modulation (AAFM)**: SPADE-inspired spatially-
       varying normalization guided by artifact probability maps.
    4. **Artifact-Guided Gating (AGG)**: Mask-based gating prevents residual
       leakage into clean tissue regions.
    5. **Boundary Refinement Module (BRM)**: Dilated convolutions at artifact
       edges for smooth transitions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AARFormer"]


# ======================================================================
# Building Blocks
# ======================================================================


class _DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x / keep * mask


class _ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x).flatten(1)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class _ConvBlock(nn.Module):
    """ConvNeXt V2-style residual block with channel attention.

    Structure: DWConv7×7 → GN → PW(expand 4×) → GELU → PW(compress) → CA → residual
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)
        self.ca = _ChannelAttention(dim)
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.ca(x)
        return shortcut + self.drop_path(x)


class _Downsample(nn.Module):
    """2× spatial downsample with channel expansion."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class _SelfAttentionBlock(nn.Module):
    """Global self-attention block (for bottleneck at low resolution).

    Uses standard multi-head self-attention + FFN in a pre-norm residual
    layout. Designed for feature maps ≤ 32×32 where global attention is
    computationally feasible.
    """

    def __init__(self, dim: int, num_heads: int, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.drop_path1 = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, 1),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, 1),
        )
        self.drop_path2 = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Self-attention
        x_norm = self.norm1(x).reshape(B, C, H * W).permute(0, 2, 1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out.permute(0, 2, 1).reshape(B, C, H, W))
        # FFN
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


# ======================================================================
# Novel Components
# ======================================================================


class ContrastEnhancedStem(nn.Module):
    """Learns contrast-aware features from the input image.

    Motivation: Annotation artifacts exhibit significantly higher local
    contrast than tissue. This stem explicitly computes a high-pass
    representation (image − blur) and learns features from it, fusing
    them with the raw intensity features.

    The fixed Gaussian kernel provides a physics-informed prior, while
    the learnable convolutions adapt to the specific contrast patterns
    of different artifact types.
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        hp_dim = max(embed_dim // 4, 8)

        # Fixed Gaussian blur kernel for high-pass computation
        self.register_buffer("_blur_kernel", self._make_gaussian(5, 1.5))

        # Learnable high-pass feature extractor
        self.hp_conv = nn.Sequential(
            nn.Conv2d(in_channels, hp_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hp_dim, hp_dim, 3, padding=1),
        )

        # Fusion: raw + high-pass features → embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels + hp_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
        self.norm = nn.GroupNorm(1, embed_dim)

    @staticmethod
    def _make_gaussian(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords.pow(2) / (2 * sigma**2))
        kernel = g.outer(g)
        kernel = kernel / kernel.sum()
        return kernel.reshape(1, 1, size, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute high-pass: image − blur(image)
        pad = self._blur_kernel.shape[-1] // 2
        blur_k = self._blur_kernel.expand(x.shape[1], -1, -1, -1)
        blurred = F.conv2d(x, blur_k, padding=pad, groups=x.shape[1])
        high_pass = x - blurred

        # Learnable high-pass features
        hp_feat = self.hp_conv(high_pass)

        # Fuse raw + high-pass
        fused = torch.cat([x, hp_feat], dim=1)
        return self.norm(self.proj(fused))


class ArtifactAwareModulation(nn.Module):
    """SPADE-inspired feature modulation guided by artifact mask.

    Generates spatially-varying affine transformation parameters (γ, β)
    for feature normalization, conditioned on the artifact probability map.
    This enables the network to apply different processing strategies for
    artifact regions (aggressive restoration) vs clean regions (preservation).

    Unlike standard SPADE which uses semantic label maps, AAFM conditions
    on a continuous artifact probability, allowing gradient flow through
    the mask prediction.
    """

    def __init__(self, feat_dim: int, mask_channels: int = 1):
        super().__init__()
        hidden = max(feat_dim // 4, 16)
        self.norm = nn.GroupNorm(1, feat_dim, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(mask_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma_conv = nn.Conv2d(hidden, feat_dim, 3, padding=1)
        self.beta_conv = nn.Conv2d(hidden, feat_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        m = self.shared(mask)
        gamma = self.gamma_conv(m)
        beta = self.beta_conv(m)
        return normalized * (1.0 + gamma) + beta


class ArtifactGuidedGating(nn.Module):
    """Mask-based gating to suppress residual in clean regions.

    Learns a soft gate from the concatenation of residual features and
    the artifact mask. In non-artifact regions (mask ≈ 0), the gate
    suppresses the residual towards zero. In artifact regions (mask ≈ 1),
    the gate allows the full residual through.

    This is the key mechanism preventing clean tissue modification.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([feat, mask], dim=1))
        return feat * g


class BoundaryRefinement(nn.Module):
    """Edge-aware refinement for smooth artifact boundary transitions.

    Detects artifact edges from the mask gradient, then applies dilated
    convolutions (receptive field = 1 + 4 + 8 = 13 pixels) to gather
    wide context around boundary pixels. The edge mask ensures refinement
    is focused on boundaries only.

    This directly addresses the "ghosting" problem at artifact edges that
    plagues standard residual-learning architectures.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Learnable edge detector on mask
        self.edge_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        # Multi-scale dilated convolutions for wide boundary context
        self.refine = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Edge detection on artifact mask
        edge = torch.abs(self.edge_conv(mask))
        edge = torch.clamp(edge, 0.0, 1.0)

        # Dilated processing around boundaries
        boundary_feat = self.refine(torch.cat([feat, edge], dim=1))

        # Apply refinement only at boundary pixels
        return feat + edge * boundary_feat


# ======================================================================
# Encoder / Decoder
# ======================================================================


class _Encoder(nn.Module):
    """Hierarchical ConvNeXt-style encoder with 4 stages.

    Produces multi-scale features for skip connections:
        skip0 (C0, H, W), skip1 (C1, H/2, W/2), skip2 (C2, H/4, W/4)
    and a bottleneck output at (C3, H/8, W/8).
    """

    def __init__(
        self,
        embed_dims: List[int],
        depths: List[int],
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert len(embed_dims) == 4 and len(depths) == 4

        # Stochastic depth schedule
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        idx = 0

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(_ConvBlock(embed_dims[i], drop_path=dpr[idx]))
                idx += 1
            self.stages.append(nn.Sequential(*blocks))
            if i < 3:
                self.downsamples.append(_Downsample(embed_dims[i], embed_dims[i + 1]))

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns (skips=[f0, f1, f2], bottleneck=f3)."""
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < 3:
                skips.append(x)
                x = self.downsamples[i](x)
        return skips, x


class _Bottleneck(nn.Module):
    """Global self-attention bottleneck.

    Applies self-attention at the lowest resolution (e.g. 32×32) where
    global attention is computationally feasible (~1024 tokens).
    Captures long-range context for understanding full image layout.
    """

    def __init__(self, dim: int, num_heads: int, depth: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.blocks = nn.Sequential(
            *[_SelfAttentionBlock(dim, num_heads, drop_path) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _ArtifactDetectionDecoder(nn.Module):
    """Lightweight decoder for multi-scale artifact mask prediction.

    Produces artifact probability maps at 3 resolutions (H/4, H/2, H)
    with deep supervision. Each mask head is initialized with negative
    bias so the model starts from a safe "no artifact" assumption.
    """

    def __init__(self, embed_dims: List[int]):
        super().__init__()
        # embed_dims = [C0, C1, C2, C3]
        # Decode: C3 → C2 (H/4), C2 → C1 (H/2), C1 → C0 (H)
        self.levels = nn.ModuleList()
        self.mask_heads = nn.ModuleList()

        decode_pairs = [
            (embed_dims[3], embed_dims[2]),  # bottleneck → H/4
            (embed_dims[2], embed_dims[1]),  # H/4 → H/2
            (embed_dims[1], embed_dims[0]),  # H/2 → H
        ]

        for in_dim, out_dim in decode_pairs:
            self.levels.append(nn.Sequential(
                nn.Conv2d(in_dim + out_dim, out_dim, 3, padding=1),
                nn.GroupNorm(1, out_dim),
                nn.GELU(),
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.GroupNorm(1, out_dim),
                nn.GELU(),
            ))
            head = nn.Conv2d(out_dim, 1, 1)
            # Initialize to predict "no artifact" (sigmoid(-3) ≈ 0.05)
            nn.init.constant_(head.bias, -3.0)
            self.mask_heads.append(head)

    def forward(
        self, bottleneck: torch.Tensor, skips: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Returns (masks, logits) at [H/4, H/2, H] resolutions.

        Masks are sigmoid-activated probabilities. Logits are raw outputs
        (used for BCE-with-logits loss which is AMP-safe).
        """
        masks = []
        logits = []
        x = bottleneck  # (C3, H/8, W/8)

        # skips = [skip0 (C0,H), skip1 (C1,H/2), skip2 (C2,H/4)]
        for i, (level, head) in enumerate(zip(self.levels, self.mask_heads)):
            skip_idx = 2 - i  # 2, 1, 0
            skip = skips[skip_idx]
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = level(torch.cat([x, skip], dim=1))
            logit = head(x)
            logits.append(logit)
            masks.append(torch.sigmoid(logit))

        return masks, logits  # [H/4, H/2, H]


class _RestorationDecoder(nn.Module):
    """Restoration decoder with Artifact-Aware Feature Modulation.

    At each level, features are upsampled, fused with encoder skips,
    processed by ConvBlocks, then modulated by AAFM using the
    corresponding artifact mask from the detection decoder.
    """

    def __init__(
        self,
        embed_dims: List[int],
        decoder_depths: List[int],
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # Decoder levels: C3→C2 (H/4), C2→C1 (H/2), C1→C0 (H)
        self.up_projs = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.block_groups = nn.ModuleList()
        self.aafms = nn.ModuleList()

        decode_pairs = [
            (embed_dims[3], embed_dims[2]),
            (embed_dims[2], embed_dims[1]),
            (embed_dims[1], embed_dims[0]),
        ]

        # Simple linear drop path for decoder
        total = sum(decoder_depths)
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, total)]
        idx = 0

        for (in_dim, out_dim), depth in zip(decode_pairs, decoder_depths):
            self.up_projs.append(nn.Conv2d(in_dim, out_dim, 1))
            self.fusions.append(nn.Conv2d(out_dim * 2, out_dim, 1))
            blocks = []
            for _ in range(depth):
                blocks.append(_ConvBlock(out_dim, drop_path=dpr[idx]))
                idx += 1
            self.block_groups.append(nn.Sequential(*blocks))
            self.aafms.append(ArtifactAwareModulation(out_dim))

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            bottleneck: (B, C3, H/8, W/8)
            skips: [skip0 (C0,H), skip1 (C1,H/2), skip2 (C2,H/4)]
            masks: [mask_H/4, mask_H/2, mask_H] from ADB
        """
        x = bottleneck

        for i, (up, fuse, blocks, aafm) in enumerate(
            zip(self.up_projs, self.fusions, self.block_groups, self.aafms)
        ):
            skip_idx = 2 - i  # 2, 1, 0
            skip = skips[skip_idx]
            x = F.interpolate(
                up(x), size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            x = fuse(torch.cat([x, skip], dim=1))
            x = blocks(x)
            x = aafm(x, masks[i])

        return x  # (B, C0, H, W)


# ======================================================================
# Main Model
# ======================================================================


class AARFormer(nn.Module):
    """Annotation Artifact Removal Transformer.

    Dual-branch architecture that decomposes annotation artifact removal
    into artifact localization and content-aware residual estimation.

    During training, returns a dict::

        {"output": clean_image,
         "residual": artifact_residual,
         "artifact_mask": final_mask,
         "multi_scale_masks": [mask_H/4, mask_H/2, mask_H]}

    During evaluation, returns just the clean image tensor.

    Args:
        spatial_dims: Must be 2 (3D not supported).
        in_channels: Input channels (1 for grayscale ultrasound).
        out_channels: Output channels (must equal in_channels).
        embed_dims: Channel dimensions for the 4 encoder stages.
        depths: Number of ConvBlocks per encoder stage.
        decoder_depths: Number of ConvBlocks per decoder level.
        bottleneck_depth: Number of self-attention blocks in bottleneck.
        num_heads: Number of attention heads in bottleneck.
        drop_path_rate: Stochastic depth rate.
        clamp_output: Clamp output to [0, 1].
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        decoder_depths: Optional[List[int]] = None,
        bottleneck_depth: int = 2,
        num_heads: int = 8,
        drop_path_rate: float = 0.1,
        clamp_output: bool = True,
        **kwargs,  # absorb extra keys from config
    ):
        super().__init__()
        if spatial_dims != 2:
            raise ValueError("AARFormer only supports spatial_dims=2")

        if embed_dims is None:
            embed_dims = [48, 96, 192, 384]
        if depths is None:
            depths = [2, 2, 6, 2]
        if decoder_depths is None:
            decoder_depths = [2, 2, 2]

        self.clamp_output = clamp_output

        # 1. Contrast-Enhanced Stem
        self.stem = ContrastEnhancedStem(in_channels, embed_dims[0])

        # 2. Hierarchical Encoder
        self.encoder = _Encoder(embed_dims, depths, drop_path_rate)

        # 3. Global Self-Attention Bottleneck
        self.bottleneck = _Bottleneck(
            embed_dims[3], num_heads, bottleneck_depth, drop_path_rate
        )

        # 4. Artifact Detection Decoder (lightweight)
        self.artifact_decoder = _ArtifactDetectionDecoder(embed_dims)

        # 5. Restoration Decoder (with AAFM)
        self.restoration_decoder = _RestorationDecoder(
            embed_dims, decoder_depths, drop_path_rate
        )

        # 6. Artifact-Guided Gating
        self.gating = ArtifactGuidedGating(embed_dims[0])

        # 7. Boundary Refinement
        self.boundary_refine = BoundaryRefinement(embed_dims[0])

        # 8. Final residual projection
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], out_channels, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ConvNeXt conventions."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Re-initialize mask heads to predict "no artifact"
        for head in self.artifact_decoder.mask_heads:
            nn.init.constant_(head.bias, -3.0)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        identity = x

        # 1. Contrast-enhanced feature extraction
        f0 = self.stem(x)

        # 2. Hierarchical encoding
        skips, f_deep = self.encoder(f0)  # skips=[f0,f1,f2], f_deep=(C3,H/8,W/8)

        # 3. Global context via self-attention
        f_deep = self.bottleneck(f_deep)

        # 4. Artifact detection → multi-scale masks and logits
        masks, mask_logits = self.artifact_decoder(f_deep, skips)

        # 5. Restoration with artifact-aware modulation
        r_feat = self.restoration_decoder(f_deep, skips, masks)  # (B, C0, H, W)

        # 6. Artifact-guided gating
        r_gated = self.gating(r_feat, masks[-1])  # masks[-1] = full-res mask

        # 7. Boundary refinement
        r_refined = self.boundary_refine(r_gated, masks[-1])

        # 8. Final residual
        residual = self.final_conv(r_refined)

        # 9. Subtract residual from input
        output = identity - residual
        if self.clamp_output:
            output = torch.clamp(output, 0.0, 1.0)

        if self.training:
            return {
                "output": output,
                "residual": residual,
                "artifact_mask": masks[-1],
                "multi_scale_masks": masks,
                "mask_logits": mask_logits,
            }
        return output
