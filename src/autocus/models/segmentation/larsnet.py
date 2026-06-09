"""
LARSNet — task-tailored multi-task segmentation network.

Designed for the LARS (Long-Axis Real-time Segmentation) family of
2-D ultrasound carotid-artery segmentation tasks where the labelled
set is small (~150 images), boundaries are speckled, and the target
is a single connected tubular structure.

A single ``FlexibleUNet`` (EfficientNet-B0 ImageNet-pretrained
encoder) shares features across three task heads read off the final
4-channel decoder output:

* ``ch[0:num_classes]`` — segmentation logits (DiceCE region loss)
* ``ch[num_classes:num_classes+1]`` — signed distance, ``tanh`` in [-1, 1]
  (L1 boundary loss)
* ``ch[num_classes+1:num_classes+2]`` — centerline heatmap, raw logits
  (BCE-with-logits topology loss)

``forward()`` returns segmentation logits only so EMA / TTA / inference
pipelines work unchanged. ``forward_multitask()`` returns all three
heads for training.

References:
    - Cellpose (Stringer et al., 2021) — multi-task flow regression for
      cell segmentation in low-data regimes.
    - SDF-net (Xue et al., 2020) — signed distance regression as dense
      boundary supervision.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from monai.networks.nets import FlexibleUNet



class LARSNetV5(nn.Module):
    """Multi-task FlexibleUNet with seg + SDT + skeleton heads.

    Args:
        num_classes: Number of segmentation classes (incl. background).
        backbone: EfficientNet variant (default ``"efficientnet-b0"``).
        pretrained: Use ImageNet-pretrained encoder weights.
        in_channels: Input image channels (default ``1`` for grayscale US).
        spatial_dims: ``2`` for 2-D, ``3`` for 3-D.

    Notes:
        Inference path (``forward``) returns segmentation logits only;
        the SDT and SKEL heads are training-only auxiliary outputs.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "efficientnet-b0",
        pretrained: bool = True,
        in_channels: int = 1,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = FlexibleUNet(
            in_channels=in_channels,
            out_channels=num_classes + 2,
            backbone=backbone,
            pretrained=pretrained,
            spatial_dims=spatial_dims,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return segmentation logits only ``[B, num_classes, ...]``."""
        out = self.backbone(x)
        return out[:, : self.num_classes]

    def forward_multitask(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return all three heads' outputs for training.

        Returns:
            Dict with keys:
                * ``seg`` — raw logits, ``[B, num_classes, ...]``
                * ``sdt`` — ``tanh``-bounded SDT, ``[B, 1, ...]`` in (-1, 1)
                * ``skel`` — raw logits (BCE applied later), ``[B, 1, ...]``
        """
        out = self.backbone(x)
        nc = self.num_classes
        return {
            "seg": out[:, :nc],
            "sdt": torch.tanh(out[:, nc : nc + 1]),
            "skel": out[:, nc + 1 : nc + 2],
        }



__all__ = ["LARSNetV5"]
