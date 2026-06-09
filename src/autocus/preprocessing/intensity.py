from __future__ import annotations

import numpy as np
from PIL import Image


def load_grayscale(path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def robust_minmax(image: np.ndarray, lower: float = 0.5, upper: float = 99.5, eps: float = 1e-6) -> np.ndarray:
    """Foreground-aware ultrasound robust min-max normalization."""
    arr = np.asarray(image, dtype=np.float32)
    mask = arr > eps
    values = arr[mask] if int(mask.sum()) >= 10 else arr.reshape(-1)
    lo, hi = np.percentile(values, [lower, upper])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~mask] = 0.0
    return out


def save_grayscale(image: np.ndarray, path) -> None:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out)
