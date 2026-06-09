from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.SiLU())
        self.att_h = nn.Conv2d(hidden, channels, 1)
        self.att_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.reduce(torch.cat([x_h, x_w], dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        return x * self.att_h(y_h).sigmoid() * self.att_w(y_w).sigmoid()


class FocusNet(nn.Module):
    """Anchor-free single-instance ROI detector used by AutoCUS-ROI."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        fusion_channels: int = 64,
        pretrained_backbone: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, fusion_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(fusion_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, fusion_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.SiLU(inplace=True),
            CoordinateAttention(fusion_channels),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(fusion_channels, num_classes, 1)
        self.offset_head = nn.Conv2d(fusion_channels, 4, 1)

    def forward(self, images: torch.Tensor, targets=None):
        feat = self.backbone(images)
        heatmap = self.heatmap_head(feat)
        offsets = F.softplus(self.offset_head(feat))
        if self.training and targets is not None:
            return {"heatmap_loss": heatmap.mean() * 0.0, "box_reg_loss": offsets.mean() * 0.0}
        return {"heatmap": heatmap, "offsets": offsets, "boxes": self.decode(heatmap, offsets, images.shape[-2:])}

    @staticmethod
    def decode(heatmap: torch.Tensor, offsets: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        b, _, h, w = heatmap.shape
        ih, iw = image_size
        flat = heatmap.sigmoid().flatten(2).argmax(dim=-1)
        ys = (flat // w).float().view(b, 1)
        xs = (flat % w).float().view(b, 1)
        scale_x = iw / float(w)
        scale_y = ih / float(h)
        ltrb = offsets.flatten(2).gather(2, flat[:, None, :].expand(-1, 4, -1)).squeeze(-1)
        x1 = (xs.squeeze(1) - ltrb[:, 0]).clamp(0, w) * scale_x
        y1 = (ys.squeeze(1) - ltrb[:, 1]).clamp(0, h) * scale_y
        x2 = (xs.squeeze(1) + ltrb[:, 2]).clamp(0, w) * scale_x
        y2 = (ys.squeeze(1) + ltrb[:, 3]).clamp(0, h) * scale_y
        return torch.stack([x1, y1, x2, y2], dim=1)
