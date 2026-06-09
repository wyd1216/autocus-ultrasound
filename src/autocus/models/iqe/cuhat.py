"""CU-HAT: Carotid Ultrasound Hybrid Attention Transformer for Super-Resolution.

A SOTA 2D super-resolution network tailored to the physical and statistical
characteristics of carotid B-mode ultrasound images, operating on
pre-upsampled low-resolution inputs (ILR strategy, input and output share
the same spatial size).

Architecture overview::

    Input (B, 1, H, W)
        |
        v  Shallow Feature Extraction (3x3 Conv)
        |
        v  Deep Feature Extraction
        |    +-- N x Residual Hybrid Attention Group (RHAG)
        |    |      |
        |    |      +-- M x Hybrid Attention Block (HAB)
        |    |      |      +-- Anisotropic Window Self-Attention (AO-WSA)
        |    |      |      |     (lateral / axial rectangular windows
        |    |      |      |      alternated between blocks)
        |    |      |      +-- Channel Attention Block (CAB, parallel branch)
        |    |      +-- Overlapping Cross-Attention Block (OCAB)
        |    |      +-- Speckle-Statistics Side Branch (SSB)
        |    |      +-- 3x3 Conv + residual
        |    +-- 3x3 Conv (deep-feature tail)
        |
        v  Laminar Boundary Enhancement Module (LBEM)
        |
        v  Reconstruction Head (3x3 Conv -> 1 channel residual)
        |
        v  Output = clamp(Input + residual, 0, 1)

Key innovations for the carotid ultrasound super-resolution task:

    1. Anisotropic Orientation-aware Window Attention (AO-WSA): rectangular
       windows alternated between lateral- and axial-oriented shapes to
       match the physical anisotropy of B-mode ultrasound resolution.
    2. Hybrid Attention (Window-SA || CAB): HAT-style parallel window
       attention + channel attention per block, activating more input
       pixels than pure spatial attention.
    3. Speckle-Statistics Side Branch (SSB): depthwise local mean/variance
       estimator that modulates features channel-wise, giving the network
       an explicit speckle prior to discourage hallucination.
    4. Laminar Boundary Enhancement Module (LBEM): parallel dilated convs
       (d=1,2,4) with Sobel-biased init to strengthen laminar vessel-wall
       boundaries that dictate IMT accuracy.
    5. Overlapping Cross-Attention Block (OCAB): overlapping windows
       suppress window-boundary artifacts, preserving smooth boundary
       reconstruction across the vessel wall.
    6. Global residual with output clamping: hard fidelity constraint
       (output = clamp(input + residual, 0, 1)), essential for medical SR.

References:
    - HAT (Chen et al., CVPR 2023) — hybrid attention & OCAB.
    - SwinIR (Liang et al., ICCVW 2021) — cascaded residual groups.
    - Restormer (Zamir et al., CVPR 2022) — channel attention design.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CUHAT"]


# ======================================================================
# Common utilities
# ======================================================================


def _window_partition(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    """Partition a (B, H, W, C) tensor into non-overlapping (wh, ww) windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, wh * ww, C)
    return x


def _window_reverse(x: torch.Tensor, wh: int, ww: int, H: int, W: int) -> torch.Tensor:
    """Inverse of :func:`_window_partition`. (nW*B, wh*ww, C) -> (B, H, W, C)."""
    B = x.shape[0] // ((H // wh) * (W // ww))
    x = x.view(B, H // wh, W // ww, wh, ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


def _pad_to_multiple(x: torch.Tensor, wh: int, ww: int) -> Tuple[torch.Tensor, int, int]:
    """Zero-pad (B, C, H, W) so H%wh==0 and W%ww==0."""
    _, _, H, W = x.shape
    pad_h = (wh - H % wh) % wh
    pad_w = (ww - W % ww) % ww
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


class _DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x / keep * mask


class _LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


# ======================================================================
# Channel Attention Block (CAB) -- HAT-style parallel branch
# ======================================================================


class _ChannelAttentionBlock(nn.Module):
    """Channel Attention Block: 3x3 Conv -> GELU -> 3x3 Conv -> SE."""

    def __init__(self, dim: int, squeeze: int = 16) -> None:
        super().__init__()
        mid = max(dim // 4, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, dim, 3, padding=1),
        )
        reduced = max(dim // squeeze, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, reduced, 1),
            nn.GELU(),
            nn.Conv2d(reduced, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y * self.se(y)


# ======================================================================
# Speckle-Statistics Side Branch (SSB)
# ======================================================================


class _SpeckleStatisticsBranch(nn.Module):
    """Learnable local-statistics modulator for speckle-aware processing.

    Estimates a local mean and variance per channel via depthwise box
    filtering and produces a channel/spatial (gain, bias) pair via a
    small MLP, implementing a speckle-aware spatially-adaptive modulation.
    """

    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.k = kernel_size
        self.mean_conv = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2,
            groups=dim, bias=False,
        )
        nn.init.constant_(self.mean_conv.weight, 1.0 / (kernel_size * kernel_size))
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mean_conv(x)
        var = self.mean_conv(x * x) - mu * mu
        var = torch.clamp(var, min=1e-6)
        sig = torch.sqrt(var)
        gb = self.gate(torch.cat([mu, sig], dim=1))
        gain, bias = gb.chunk(2, dim=1)
        return x * (1.0 + gain) + bias


# ======================================================================
# Anisotropic Orientation-aware Window Self-Attention (AO-WSA)
# ======================================================================


class _RelativePositionBias(nn.Module):
    """Learnable relative position bias table for rectangular windows."""

    def __init__(self, wh: int, ww: int, num_heads: int) -> None:
        super().__init__()
        self.wh, self.ww = wh, ww
        self.num_heads = num_heads
        self.table = nn.Parameter(torch.zeros((2 * wh - 1) * (2 * ww - 1), num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords = coords.flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += wh - 1
        rel[:, :, 1] += ww - 1
        rel[:, :, 0] *= 2 * ww - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

    def forward(self) -> torch.Tensor:
        bias = self.table[self.relative_position_index.view(-1)].view(
            self.wh * self.ww, self.wh * self.ww, -1
        )
        return bias.permute(2, 0, 1).contiguous()


class _WindowAttention(nn.Module):
    """Window multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        wh: int,
        ww: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.wh, self.ww = dim, wh, ww
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel_bias = _RelativePositionBias(wh, ww, num_heads)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B_, N, C = tokens.shape
        qkv = self.qkv(tokens).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Memory-efficient SDPA (fused kernel). Relative position bias is
        # broadcast across batch as an additive attention mask. This avoids
        # materialising the (B_, heads, N, N) attention matrix for backward.
        bias = self.rel_bias().unsqueeze(0)  # (1, heads, N, N)
        # Run attention in fp32 for numerical stability under AMP; the
        # projection layers outside still benefit from half precision.
        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            out = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                attn_mask=bias.float(),
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        out = out.to(tokens.dtype)
        out = out.transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


# ======================================================================
# Hybrid Attention Block (HAB)
# ======================================================================


class _Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _HybridAttentionBlock(nn.Module):
    """Parallel Window-SA || CAB + MLP, with rectangular (anisotropic) windows."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_h: int,
        window_w: int,
        mlp_ratio: float = 2.0,
        cab_weight: float = 0.01,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_h = window_h
        self.window_w = window_w
        self.cab_weight = cab_weight
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_h, window_w, num_heads)
        self.cab = _ChannelAttentionBlock(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, mlp_ratio)
        self.drop_path = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        wh, ww = self.window_h, self.window_w
        x_pad, pad_h, pad_w = _pad_to_multiple(x, wh, ww)
        Hp, Wp = x_pad.shape[-2:]

        tokens = x_pad.permute(0, 2, 3, 1)
        normed = self.norm1(tokens)
        windows = _window_partition(normed, wh, ww)
        attn_out = self.attn(windows)
        attn_out = _window_reverse(attn_out, wh, ww, Hp, Wp)
        attn_out = attn_out.permute(0, 3, 1, 2).contiguous()

        cab_out = self.cab(x_pad)

        merged = self.drop_path(attn_out + self.cab_weight * cab_out)
        x_pad = x_pad + merged

        tokens = x_pad.permute(0, 2, 3, 1)
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        x_pad = tokens.permute(0, 3, 1, 2).contiguous()

        if pad_h or pad_w:
            x_pad = x_pad[:, :, :H, :W]
        return x_pad


# ======================================================================
# Overlapping Cross-Attention Block (OCAB)
# ======================================================================


class _OverlappingCrossAttention(nn.Module):
    """Overlapping cross-window attention block (HAT-style OCAB).

    Query tokens come from a regular (wh, ww) window; key/value tokens from
    a larger (wh+2o, ww+2o) window with stride (wh, ww), giving each query
    token a slightly broader receptive field than plain W-MSA.
    """

    def __init__(
        self,
        dim: int,
        wh: int,
        ww: int,
        overlap: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim, self.wh, self.ww, self.overlap = dim, wh, ww, overlap
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.norm1 = _LayerNorm2d(dim)
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm2 = _LayerNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wh, ww, o = self.wh, self.ww, self.overlap
        B, C, H, W = x.shape
        x_pad, pad_h, pad_w = _pad_to_multiple(x, wh, ww)
        Hp, Wp = x_pad.shape[-2:]

        n = self.norm1(x_pad)
        q = self.q(n)
        kv = self.kv(n)

        q_win = q.unfold(2, wh, wh).unfold(3, ww, ww)
        nH, nW = q_win.shape[2], q_win.shape[3]
        q_win = q_win.permute(0, 2, 3, 1, 4, 5).reshape(B * nH * nW, C, wh, ww)

        kv_pad = F.pad(kv, (o, o, o, o), mode="reflect")
        kv_win = kv_pad.unfold(2, wh + 2 * o, wh).unfold(3, ww + 2 * o, ww)
        kv_win = kv_win.permute(0, 2, 3, 1, 4, 5).reshape(
            B * nH * nW, 2 * C, wh + 2 * o, ww + 2 * o
        )
        k_win, v_win = kv_win.chunk(2, dim=1)

        Nq = wh * ww
        Nk = (wh + 2 * o) * (ww + 2 * o)
        hd = C // self.num_heads

        q_tok = q_win.flatten(2).transpose(1, 2).reshape(-1, Nq, self.num_heads, hd).transpose(1, 2)
        k_tok = k_win.flatten(2).transpose(1, 2).reshape(-1, Nk, self.num_heads, hd).transpose(1, 2)
        v_tok = v_win.flatten(2).transpose(1, 2).reshape(-1, Nk, self.num_heads, hd).transpose(1, 2)

        with torch.amp.autocast(device_type=q_tok.device.type, enabled=False):
            out = F.scaled_dot_product_attention(
                q_tok.float(), k_tok.float(), v_tok.float(),
            )
        out = out.to(x.dtype)
        out = out.transpose(1, 2).reshape(-1, Nq, C).transpose(1, 2).reshape(-1, C, wh, ww)
        out = out.view(B, nH, nW, C, wh, ww).permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(B, C, Hp, Wp)
        out = self.proj(out)

        x_pad = x_pad + out
        x_pad = x_pad + self.ffn(self.norm2(x_pad))

        if pad_h or pad_w:
            x_pad = x_pad[:, :, :H, :W]
        return x_pad


# ======================================================================
# Residual Hybrid Attention Group (RHAG)
# ======================================================================


class _RHAG(nn.Module):
    """Residual Hybrid Attention Group: M HABs + 1 OCAB + SSB + conv (+ outer residual)."""

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        num_heads: int,
        window_lateral: Tuple[int, int],
        window_axial: Tuple[int, int],
        overlap: int,
        mlp_ratio: float = 2.0,
        drop_path: Sequence[float] = (0.0,),
        use_ocab: bool = True,
        use_ssb: bool = True,
    ) -> None:
        super().__init__()
        assert len(drop_path) == num_blocks
        blocks: List[nn.Module] = []
        for j in range(num_blocks):
            if j % 2 == 0:
                wh, ww = window_lateral
            else:
                wh, ww = window_axial
            blocks.append(_HybridAttentionBlock(
                dim=dim, num_heads=num_heads, window_h=wh, window_w=ww,
                mlp_ratio=mlp_ratio, drop_path=drop_path[j],
            ))
        self.blocks = nn.Sequential(*blocks)
        self.ocab = _OverlappingCrossAttention(
            dim=dim, wh=window_lateral[0], ww=window_lateral[1],
            overlap=overlap, num_heads=num_heads, mlp_ratio=mlp_ratio,
        ) if use_ocab else None
        self.ssb = _SpeckleStatisticsBranch(dim, kernel_size=7) if use_ssb else None
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        if self.ocab is not None:
            y = self.ocab(y)
        if self.ssb is not None:
            y = self.ssb(y)
        y = self.conv(y)
        return x + y


# ======================================================================
# Laminar Boundary Enhancement Module (LBEM)
# ======================================================================


class _LaminarBoundaryEnhancement(nn.Module):
    """Parallel dilated depthwise convs (d=1,2,4) with Sobel-biased init."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw1 = nn.Conv2d(dim, dim, 3, padding=1, dilation=1, groups=dim, bias=False)
        self.dw2 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False)
        self.dw4 = nn.Conv2d(dim, dim, 3, padding=4, dilation=4, groups=dim, bias=False)
        self._init_sobel()
        self.pw = nn.Conv2d(dim * 3, dim, 1)
        self.act = nn.GELU()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )

    def _init_sobel(self) -> None:
        dim = self.dw1.weight.shape[0]
        sobel_h = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_v = sobel_h.t().contiguous()
        with torch.no_grad():
            w1 = self.dw1.weight
            w2 = self.dw2.weight
            for c in range(dim):
                base1 = w1[c, 0].clone()
                base2 = w2[c, 0].clone()
                if c % 2 == 0:
                    w1[c, 0] = sobel_h * 0.1 + base1 * 0.1
                    w2[c, 0] = sobel_h * 0.1 + base2 * 0.1
                else:
                    w1[c, 0] = sobel_v * 0.1 + base1 * 0.1
                    w2[c, 0] = sobel_v * 0.1 + base2 * 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.dw1(x)
        y2 = self.dw2(x)
        y4 = self.dw4(x)
        y = self.act(self.pw(torch.cat([y1, y2, y4], dim=1)))
        g = self.gate(y)
        return x + y * g


# ======================================================================
# CU-HAT
# ======================================================================


class CUHAT(nn.Module):
    """Carotid Ultrasound Hybrid Attention Transformer.

    Args:
        spatial_dims: Must be 2 (2D SR only; kept for API consistency).
        in_channels: Input channels (grayscale -> 1).
        out_channels: Output channels (grayscale -> 1).
        embed_dim: Feature dimension used throughout the transformer body.
        num_groups: Number of Residual Hybrid Attention Groups (N).
        num_blocks_per_group: Number of HABs per RHAG before OCAB (M).
        num_heads: Attention heads (must divide embed_dim).
        window_lateral: (Wh, Ww) for lateral-oriented windows (e.g. (8, 16)).
        window_axial: (Wh, Ww) for axial-oriented windows (e.g. (16, 8)).
        overlap_ratio: Overlap size (pixels) for OCAB windows.
        mlp_ratio: FFN expansion ratio.
        drop_path_rate: Maximum stochastic-depth rate (linearly scaled).
        img_range: Scalar applied to input/output for mean-centred training.
        use_boundary_aux: If True, forward returns a dict containing an
            auxiliary boundary map for deep supervision.

    Forward IO:
        Input : (B, in_channels, H, W) in [0, 1].
        Output: (B, out_channels, H, W) in [0, 1]
                (or dict when ``use_boundary_aux=True``).
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 180,
        num_groups: int = 6,
        num_blocks_per_group: int = 6,
        num_heads: int = 6,
        window_lateral: Tuple[int, int] = (8, 16),
        window_axial: Tuple[int, int] = (16, 8),
        overlap_ratio: int = 4,
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
        img_range: float = 1.0,
        use_boundary_aux: bool = False,
        use_checkpoint: bool = False,
        use_ocab: bool = True,
        use_ssb: bool = True,
    ) -> None:
        super().__init__()
        if spatial_dims != 2:
            raise ValueError(f"CUHAT supports spatial_dims=2 only, got {spatial_dims}.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if isinstance(window_lateral, list):
            window_lateral = tuple(window_lateral)
        if isinstance(window_axial, list):
            window_axial = tuple(window_axial)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.img_range = img_range
        self.use_boundary_aux = use_boundary_aux
        self.use_checkpoint = use_checkpoint

        self.register_buffer("mean", torch.full((1, in_channels, 1, 1), 0.5))

        self.shallow = nn.Conv2d(in_channels, embed_dim, 3, padding=1)

        total_blocks = num_groups * num_blocks_per_group
        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, total_blocks)]
        self.groups = nn.ModuleList()
        for g in range(num_groups):
            start = g * num_blocks_per_group
            self.groups.append(_RHAG(
                dim=embed_dim,
                num_blocks=num_blocks_per_group,
                num_heads=num_heads,
                window_lateral=window_lateral,
                window_axial=window_axial,
                overlap=overlap_ratio,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[start:start + num_blocks_per_group],
                use_ocab=use_ocab,
                use_ssb=use_ssb,
            ))
        self.deep_tail = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        self.lbem = _LaminarBoundaryEnhancement(embed_dim)

        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_channels, 3, padding=1),
        )
        # Zero-init head so initial output ~= input (fidelity prior).
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        if use_boundary_aux:
            self.boundary_head = nn.Conv2d(embed_dim, 1, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"CUHAT expects (B, C, H, W); got shape {tuple(x.shape)}.")

        identity = x
        x = (x - self.mean) * self.img_range

        f0 = self.shallow(x)
        f = f0
        for group in self.groups:
            if self.use_checkpoint and self.training:
                f = torch.utils.checkpoint.checkpoint(group, f, use_reentrant=False)
            else:
                f = group(f)
        f = self.deep_tail(f) + f0
        f = self.lbem(f)

        residual = self.head(f) / self.img_range
        output = torch.clamp(identity + residual, 0.0, 1.0)

        if self.use_boundary_aux:
            boundary = torch.sigmoid(self.boundary_head(f))
            return {"output": output, "residual": residual, "boundary": boundary}
        return output
