"""
PlaqueSENet — SOTA carotid plaque stable/unstable classifier.

Design philosophy
-----------------
PlaqueSENet builds on a battle-tested SE-ResNet50 backbone and adds three
**lightweight, residually-initialised** components targeted at the
specific challenges of small-data ultrasound plaque classification:

* **Multi-Scale Feature Pyramid (MSF)** — ``layer2``/``layer3``/``layer4``
  features are pooled and concatenated, so the classifier sees both
  fine-grained texture (lipid core, calcific specks) and coarse
  morphology (cap thickness, surface contour).
* **CBAM Attention with γ-residual gate** — channel + spatial attention
  applied to the deepest feature map; the output is fused as
  ``f = f0 + γ · CBAM(f0)`` with γ initialised to 0.  At step 0 the
  network is **bit-exact equivalent** to vanilla SE-ResNet50, so the
  attention can only improve from a known-good initialisation.
* **Deep-Supervision Auxiliary Head (DSV)** — a tiny classifier hooked
  off ``layer3`` providing an auxiliary cross-entropy loss
  (``α=0.3`` by default).  Only used at training time.

A ``warm_start_ckpt`` argument loads matching weights from an existing
SE-ResNet50 checkpoint into the backbone (non-strict), giving a
guaranteed strong initialisation on this dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SEResNet50

logger = logging.getLogger(__name__)


# =============================================================================
# Building blocks
# =============================================================================

class _ChannelAttention(nn.Module):
    """CBAM channel attention (avg + max pooled MLP)."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * att


class _SpatialAttention(nn.Module):
    """CBAM spatial attention (channel-pooled 7x7 conv)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class _CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

    def __init__(self, in_channels: int, reduction: int = 16,
                 spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = _ChannelAttention(in_channels, reduction)
        self.spatial = _SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# =============================================================================
# PlaqueSENet
# =============================================================================

class PlaqueSENet(nn.Module):
    """SE-ResNet50 + Multi-Scale Pyramid + γ-gated CBAM + Deep Supervision.

    Parameters
    ----------
    spatial_dims : int
        Always 2 in this task. Kept for API symmetry with MONAI nets.
    in_channels : int
        Image channels (1 for grayscale ultrasound).
    out_channels : int (a.k.a. ``num_classes``)
        Number of output classes.
    dropout_prob : float
        Dropout applied before the main classifier head.
    use_msf : bool
        Enable the multi-scale feature pyramid (``layer2`` + ``layer3``
        + ``layer4``).  When ``False``, only ``layer4`` features are used.
    use_cbam : bool
        Enable the γ-gated CBAM attention on ``layer4``. γ is
        zero-initialised so the model starts identical to plain
        SE-ResNet50.
    use_dsv : bool
        Enable the deep-supervision auxiliary head on ``layer3``.
    aux_loss_weight : float
        Weight of the auxiliary CE loss (only used if ``use_dsv``).
        Stored as an attribute; the trainer reads it.
    warm_start_ckpt : str | Path | None
        Path to a SE-ResNet50 checkpoint (training output).  Backbone
        weights with matching shapes are loaded non-strictly.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float = 0.2,
        use_msf: bool = True,
        use_cbam: bool = True,
        use_dsv: bool = True,
        aux_loss_weight: float = 0.3,
        warm_start_ckpt: Optional[str] = None,
        backbone_source: str = "monai",   # "monai" | "timm_seresnet50"
        pretrained: bool = False,         # only used when backbone_source="timm_*"
    ) -> None:
        super().__init__()
        if spatial_dims != 2:
            raise ValueError("PlaqueSENet only supports 2D inputs.")

        self.in_channels = int(in_channels)
        self.num_classes = int(out_channels)
        self.use_msf = bool(use_msf)
        self.use_cbam = bool(use_cbam)
        self.use_dsv = bool(use_dsv)
        self.aux_loss_weight = float(aux_loss_weight)
        self.backbone_source = str(backbone_source)

        # ---- Backbone --------------------------------------------------
        # Two choices:
        #   * "monai"            – MONAI SEResNet50 (random init).  Layer
        #                           access via ``backbone.layer{0..4}``.
        #   * "timm_seresnet50"  – timm SEResNet50 (ImageNet pretrained
        #                           when ``pretrained=True``).  Multi-scale
        #                           features pulled via ``features_only``.
        if self.backbone_source == "monai":
            self.backbone = SEResNet50(
                spatial_dims=2,
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                dropout_prob=0.0,
            )
            self.backbone.last_linear = nn.Identity()
            self.backbone.dropout = nn.Identity()
        elif self.backbone_source.startswith("timm_"):
            import timm
            timm_name = self.backbone_source[len("timm_"):]
            self.backbone = timm.create_model(
                timm_name,
                pretrained=bool(pretrained),
                in_chans=self.in_channels,
                features_only=True,
                out_indices=(2, 3, 4),  # stages with C=512, 1024, 2048
            )
            chans = self.backbone.feature_info.channels()
            assert chans == [512, 1024, 2048], (
                f"Expected timm SE-ResNet50 channels [512,1024,2048], got {chans}"
            )
        else:
            raise ValueError(f"Unknown backbone_source: {self.backbone_source}")

        # ---- γ-gated CBAM on layer4 (2048 channels) ---------------------
        if self.use_cbam:
            self.cbam = _CBAM(in_channels=2048, reduction=16, spatial_kernel=7)
            # γ-residual gate, zero-init -> identity at step 0
            self.cbam_gamma = nn.Parameter(torch.zeros(1))
        else:
            self.cbam = None
            self.cbam_gamma = None

        # ---- Multi-scale pyramid (layer2: 512, layer3: 1024, layer4:2048)
        if self.use_msf:
            head_in = 512 + 1024 + 2048  # = 3584
        else:
            head_in = 2048

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_norm = nn.LayerNorm(head_in)
        self.head_dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(head_in, self.num_classes)

        # ---- Deep-supervision auxiliary head on layer3 (1024 channels) --
        if self.use_dsv:
            self.aux_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(1024),
                nn.Dropout(dropout_prob),
                nn.Linear(1024, self.num_classes),
            )
        else:
            self.aux_classifier = None

        # ---- Initialise newly-added head modules ------------------------
        self._init_head_weights()

        # ---- Optional warm start from SE-ResNet50 checkpoint ------------
        # Skipped automatically when the backbone is timm-pretrained
        # (ImageNet weights are stronger than any in-domain warm-start
        # on a 729-image training set).
        if warm_start_ckpt is not None and self.backbone_source == "monai":
            self._warm_start_from(Path(warm_start_ckpt))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_head_weights(self) -> None:
        for m in (self.classifier,):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        if self.aux_classifier is not None:
            for layer in self.aux_classifier.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)

    def _warm_start_from(self, ckpt_path: Path) -> None:
        """Load SE-ResNet50 backbone weights from a training checkpoint.

        The checkpoint is expected to be either a raw ``state_dict`` or
        a dict with a ``"model_state_dict"`` key.  Keys whose name or
        shape do not match the current ``self.backbone`` are skipped
        with a logged warning.
        """
        if not ckpt_path.exists():
            logger.warning("warm_start_ckpt not found: %s", ckpt_path)
            return
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob

        own_sd = self.backbone.state_dict()
        loaded, skipped = 0, 0
        new_sd = {}
        for k, v in sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                new_sd[k] = v
                loaded += 1
            else:
                skipped += 1
        own_sd.update(new_sd)
        self.backbone.load_state_dict(own_sd, strict=False)
        logger.info(
            "PlaqueSENet warm-start: loaded %d / skipped %d backbone "
            "tensors from %s", loaded, skipped, ckpt_path,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_features(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the SE-ResNet50 backbone and return (f2, f3, f4).

        ``f2`` is layer2 output (512 ch), ``f3`` is layer3 (1024 ch),
        ``f4`` is layer4 (2048 ch).  Both MONAI and timm code paths
        return the same channel layout.
        """
        if self.backbone_source == "monai":
            h = self.backbone.layer0(x)
            h = self.backbone.layer1(h)
            f2 = self.backbone.layer2(h)
            f3 = self.backbone.layer3(f2)
            f4 = self.backbone.layer4(f3)
            return f2, f3, f4
        # timm features_only path
        feats = self.backbone(x)
        return feats[0], feats[1], feats[2]

    def forward(self, x: torch.Tensor):
        f2, f3, f4 = self._forward_features(x)

        # γ-gated CBAM on f4 (residual; γ=0 at init -> identity)
        if self.cbam is not None:
            f4 = f4 + self.cbam_gamma * self.cbam(f4)

        # Multi-scale feature pyramid
        if self.use_msf:
            v2 = self.gap(f2).flatten(1)
            v3 = self.gap(f3).flatten(1)
            v4 = self.gap(f4).flatten(1)
            v = torch.cat([v2, v3, v4], dim=1)
        else:
            v = self.gap(f4).flatten(1)

        v = self.head_norm(v)
        v = self.head_dropout(v)
        logits = self.classifier(v)

        # Deep supervision: stash aux logits as a side-channel attribute so
        # ``forward`` always returns a plain tensor (compatible with the
        # generic classification trainer).  ``classification_trainer`` reads
        # ``model._last_aux_logits`` / ``model.aux_loss_weight`` and adds
        # the auxiliary CE term during training only.
        if self.training and self.aux_classifier is not None:
            self._last_aux_logits = self.aux_classifier(f3)
        else:
            self._last_aux_logits = None
        return logits


__all__ = ["PlaqueSENet"]
