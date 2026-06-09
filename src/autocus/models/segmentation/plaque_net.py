"""
PlaqueNetV1 — task-tailored network for carotid plaque segmentation
on long-axis B-mode ultrasound (CPLS).

Design philosophy (extending LARSNet v5):

* **Strong backbone, flexible head** — reuse a single ``FlexibleUNet``
  (EfficientNet-B0, ImageNet-pretrained) and place task-specific heads
  on top.
* **Empty-mask aware** — the dataset contains a sizeable subset
  of *plaque-negative* samples whose
  ground-truth mask is all-background. A pixel-only network with
  Dice/CE supervision tends to (a) NaN-divide on empty masks and
  (b) hallucinate plaque on smooth normal vessel walls.

The single architectural innovation is the
**Plaque Presence Gate (PPG)**:

* A lightweight image-level binary head taps the encoder bottleneck
  (``GAP -> MLP -> sigmoid -> p in [0, 1]``).
* Trained with image-level BCE against
  ``y = 1 if mask.sum() > 0 else 0`` — a label that is consistent
  across all subsets and robust to the public-source classification-label
  noise documented during curation.
* At inference, the foreground logit is shifted by ``log(p)`` so the
  predicted plaque probability is multiplicatively gated by the
  image-level presence score, suppressing false positives on normal
  vessels.

Outputs of ``forward_multitask`` (training):
* ``seg``       — segmentation logits ``[B, num_classes, H, W]`` (un-gated).
* ``sdt``       — signed-distance regression in (-1, 1), ``[B, 1, H, W]``.
* ``skel``      — centerline heatmap logits, ``[B, 1, H, W]``.
* ``ppg_logit`` — image-level plaque-presence logit, ``[B, 1]``.

Output of ``forward`` (inference / eval / TTA): segmentation logits
*after* PPG gating.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from monai.networks.nets import FlexibleUNet



class PlaqueNetV1(nn.Module):
    """FlexibleUNet + multi-task heads + image-level Plaque Presence Gate."""

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "efficientnet-b0",
        pretrained: bool = True,
        in_channels: int = 1,
        spatial_dims: int = 2,
        ppg_hidden: int = 256,
        ppg_dropout: float = 0.2,
        gate_at_inference: bool = True,
    ) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        self.num_classes = int(num_classes)
        self.spatial_dims = int(spatial_dims)
        self.gate_at_inference = bool(gate_at_inference)

        self.unet = FlexibleUNet(
            in_channels=in_channels,
            out_channels=self.num_classes + 2,
            backbone=backbone,
            pretrained=pretrained,
            spatial_dims=spatial_dims,
        )

        self._ppg_hidden = int(ppg_hidden)
        self._ppg_dropout = float(ppg_dropout)
        self.ppg_head: Optional[nn.Module] = None
        self._gap = (
            nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)
        )

        # Eagerly materialise the PPG head so ``state_dict`` keys are
        # available before checkpoint loading (otherwise a lazy build at
        # first forward would surface "Unexpected key(s)" during eval).
        with torch.no_grad():
            dummy_shape = [1, in_channels] + [64] * spatial_dims
            dummy = torch.zeros(*dummy_shape)
            try:
                bottleneck = self._encoder_bottleneck(dummy)
                pooled = self._gap(bottleneck).flatten(1)
                self._build_ppg_head(pooled.shape[1], pooled.device, pooled.dtype)
            except Exception:
                # Fall back to lazy initialisation when encoder requires GPU.
                self.ppg_head = None

    # -- Internals -----------------------------------------------------
    def _build_ppg_head(self, in_features: int, device, dtype) -> None:
        head = nn.Sequential(
            nn.Linear(in_features, self._ppg_hidden),
            nn.GELU(),
            nn.Dropout(self._ppg_dropout),
            nn.Linear(self._ppg_hidden, 1),
        )
        self.ppg_head = head.to(device=device, dtype=dtype)

    def _encoder_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.unet.encoder(x)
        if isinstance(feats, (list, tuple)):
            return feats[-1]
        return feats

    def _ppg_logit(self, bottleneck: torch.Tensor) -> torch.Tensor:
        pooled = self._gap(bottleneck).flatten(1)
        if self.ppg_head is None:
            self._build_ppg_head(pooled.shape[1], pooled.device, pooled.dtype)
        return self.ppg_head(pooled)

    # -- Public API ----------------------------------------------------
    def forward_multitask(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        seg_full = self.unet(x)
        bottleneck = self._encoder_bottleneck(x)
        ppg_logit = self._ppg_logit(bottleneck)
        nc = self.num_classes
        return {
            "seg": seg_full[:, :nc],
            "sdt": torch.tanh(seg_full[:, nc : nc + 1]),
            "skel": seg_full[:, nc + 1 : nc + 2],
            "ppg_logit": ppg_logit,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seg_full = self.unet(x)
        nc = self.num_classes
        seg = seg_full[:, :nc]
        if not self.gate_at_inference or nc < 2:
            return seg
        bottleneck = self._encoder_bottleneck(x)
        ppg_logit = self._ppg_logit(bottleneck)
        view_shape = [seg.shape[0], 1] + [1] * (seg.ndim - 2)
        log_p = torch.nn.functional.logsigmoid(ppg_logit).view(view_shape)
        bg = seg[:, :1]
        fg = seg[:, 1:] + log_p
        return torch.cat([bg, fg], dim=1)



__all__ = ["PlaqueNetV1"]
