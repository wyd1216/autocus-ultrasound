from __future__ import annotations

import numpy as np


def foreground_bbox(image: np.ndarray, margin: int = 4) -> list[int]:
    """Return a deterministic fallback ROI box around nonzero ultrasound content."""
    arr = np.asarray(image)
    mask = arr > max(float(arr.max()) * 0.02, 1e-6)
    if not mask.any():
        h, w = arr.shape[-2:]
        return [0, 0, w, h]
    ys, xs = np.where(mask)
    h, w = arr.shape[-2:]
    x1 = max(int(xs.min()) - margin, 0)
    y1 = max(int(ys.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin + 1, w)
    y2 = min(int(ys.max()) + margin + 1, h)
    return [x1, y1, x2, y2]


def crop_box(image: np.ndarray, box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    return image[y1:y2, x1:x2]
