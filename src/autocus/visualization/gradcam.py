from __future__ import annotations

import numpy as np


def simple_saliency(image: np.ndarray) -> np.ndarray:
    """Lightweight saliency fallback for demo outputs when Grad-CAM is unavailable."""
    arr = np.asarray(image, dtype=np.float32)
    gy, gx = np.gradient(arr)
    sal = np.sqrt(gx * gx + gy * gy)
    maxv = float(sal.max())
    return sal / maxv if maxv > 0 else sal
