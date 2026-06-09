from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from skimage import exposure


@dataclass(frozen=True)
class IQENormConfig:
    """Configuration for the deterministic IQE-NORM pipeline."""

    method: str = "iqe_norm_traditional"
    background_threshold: float = 5 / 255
    use_foreground_mask: bool = True
    mask_feather: int = 8
    pclip_low: float = 0.5
    pclip_high: float = 99.5
    denoise_method: str = "srad"
    srad_iter: int = 30
    srad_dt: float = 0.05
    srad_window: int = 3
    guided_radius: int = 4
    guided_eps: float = 1000.0
    clahe_clip: float = 1.5
    clahe_tile_grid: int = 8
    clahe_dynamic_grid: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "IQENormConfig":
        if not data:
            return cls()
        aliases = {
            "background_eps": "background_threshold",
            "percentile_range": ("pclip_low", "pclip_high"),
            "output_range": None,
        }
        known = {field.name for field in fields(cls)}
        payload: dict[str, Any] = {}
        for key, value in data.items():
            mapped = aliases.get(key, key)
            if mapped is None:
                continue
            if isinstance(mapped, tuple):
                payload[mapped[0]] = value[0]
                payload[mapped[1]] = value[1]
            elif mapped in known:
                payload[mapped] = value
        return cls(**payload)


def _as_float01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype.kind in {"u", "i"}:
        max_value = np.iinfo(arr.dtype).max if arr.dtype.kind == "u" else max(float(arr.max()), 1.0)
        return np.clip(arr.astype(np.float32) / float(max_value), 0.0, 1.0)
    arr = arr.astype(np.float32, copy=False)
    if arr.size and float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _pad_replicate(image: np.ndarray) -> np.ndarray:
    return np.pad(image, 1, mode="edge")


def _laplacian(image: np.ndarray) -> np.ndarray:
    padded = _pad_replicate(image)
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * image
    )


def _gradient_magnitude_sq(image: np.ndarray) -> np.ndarray:
    padded = _pad_replicate(image)
    dy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    dx = padded[1:-1, 2:] - padded[1:-1, :-2]
    return (dx * dx + dy * dy) / 4.0


def _edge_trim_from_dark_fraction(dark_fraction: np.ndarray, threshold: float, max_gap: int) -> int:
    last_dark = -1
    gap = 0
    for index, fraction in enumerate(dark_fraction):
        if fraction >= threshold:
            last_dark = index
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return last_dark + 1


def detect_foreground(
    image: np.ndarray,
    background_threshold: float = 5 / 255,
) -> np.ndarray | None:
    """Detect the acquisition window while preserving dark in-frame anatomy."""
    arr = _as_float01(image)
    if arr.size == 0:
        return None

    height, width = arr.shape[:2]
    near_black = arr <= float(background_threshold)
    row_dark_fraction = near_black.mean(axis=1)
    col_dark_fraction = near_black.mean(axis=0)
    bezel_fraction = 0.98
    row_gap = max(8, height // 100)
    col_gap = max(8, width // 100)

    top = _edge_trim_from_dark_fraction(row_dark_fraction, bezel_fraction, row_gap)
    bottom = height - _edge_trim_from_dark_fraction(row_dark_fraction[::-1], bezel_fraction, row_gap)
    left = _edge_trim_from_dark_fraction(col_dark_fraction, bezel_fraction, col_gap)
    right = width - _edge_trim_from_dark_fraction(col_dark_fraction[::-1], bezel_fraction, col_gap)

    if top >= bottom or left >= right:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask if mask.any() else None


def percentile_clip(
    image: np.ndarray,
    low: float = 0.5,
    high: float = 99.5,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Foreground-aware percentile clipping rescaled to [0, 1]."""
    arr = _as_float01(image)
    values = arr[mask] if mask is not None and mask.any() else arr.reshape(-1)
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo:
        out = arr.astype(np.float32, copy=True)
    else:
        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if mask is not None:
        out = np.where(mask, out, np.float32(0.0))
    return out.astype(np.float32, copy=False)


def _prefill_background(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None or not mask.any():
        return image
    foreground_mean = float(image[mask].mean())
    return np.where(mask, image, foreground_mean)


def srad(
    image: np.ndarray,
    n_iter: int = 30,
    dt: float = 0.05,
    window: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Speckle reducing anisotropic diffusion in the normalized float domain."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")

    current = _as_float01(image).astype(np.float64)
    current = _prefill_background(current, mask)
    current = np.clip(current, 1 / 255, 1.0)

    for index in range(int(n_iter)):
        local_mean = current if window == 1 else uniform_filter(current, size=window, mode="nearest")
        local_mean = np.clip(local_mean, 1 / 255, None)
        lap = _laplacian(current)
        grad_sq = _gradient_magnitude_sq(current)
        q_sq = (grad_sq / (local_mean**2 + 1e-12)) - (lap / (local_mean + 1e-12)) ** 2
        q_sq = np.clip(q_sq, 0.0, None)
        q0_sq = 1.0 / (1.0 + index)
        coeff = 1.0 / (1.0 + (q_sq - q0_sq) / (q0_sq * (1.0 + q0_sq) + 1e-12))
        coeff = np.clip(coeff, 0.0, 1.0)
        current = np.clip(current + float(dt) * (coeff * lap), 0.0, 1.0)

    if mask is not None:
        current = np.where(mask, current, 0.0)
    return current.astype(np.float32)


def guided_filter(
    image: np.ndarray,
    radius: int = 4,
    eps: float = 1000.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fast self-guided edge-preserving smoothing fallback."""
    arr = _prefill_background(_as_float01(image).astype(np.float64), mask)
    size = 2 * int(radius) + 1
    eps_norm = float(eps) / (255.0 * 255.0)
    mean_i = uniform_filter(arr, size=size, mode="nearest")
    mean_ii = uniform_filter(arr * arr, size=size, mode="nearest")
    var_i = mean_ii - mean_i * mean_i
    a = var_i / (var_i + eps_norm)
    b = mean_i - a * mean_i
    mean_a = uniform_filter(a, size=size, mode="nearest")
    mean_b = uniform_filter(b, size=size, mode="nearest")
    out = np.clip(mean_a * arr + mean_b, 0.0, 1.0)
    if mask is not None:
        out = np.where(mask, out, 0.0)
    return out.astype(np.float32)


def denoise(
    image: np.ndarray,
    method: str = "srad",
    mask: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray:
    if method == "srad":
        return srad(
            image,
            n_iter=int(kwargs.get("n_iter", 30)),
            dt=float(kwargs.get("dt", 0.05)),
            window=int(kwargs.get("window", 3)),
            mask=mask,
        )
    if method == "guided":
        return guided_filter(
            image,
            radius=int(kwargs.get("radius", 4)),
            eps=float(kwargs.get("eps", 1000.0)),
            mask=mask,
        )
    raise ValueError(f"Unknown denoise method: {method!r}. Use 'srad' or 'guided'.")


def _resolve_tile_grid(image: np.ndarray, tile_grid: int, dynamic: bool) -> int:
    if not dynamic:
        return max(1, int(tile_grid))
    short_side = min(image.shape[:2])
    return int(max(4, min(16, short_side // 64)))


def clahe(
    image: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid: int = 8,
    mask: np.ndarray | None = None,
    dynamic_grid: bool = False,
) -> np.ndarray:
    """Apply conservative CLAHE with foreground-aware background fill."""
    arr = _as_float01(image)
    grid_n = _resolve_tile_grid(arr, tile_grid, dynamic_grid)
    kernel_size = (max(1, int(np.ceil(arr.shape[0] / grid_n))), max(1, int(np.ceil(arr.shape[1] / grid_n))))
    sk_clip = max(0.001, float(clip_limit) / 100.0)

    if mask is not None and mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        roi = arr[y_min : y_max + 1, x_min : x_max + 1].copy()
        roi_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
        if roi_mask.any():
            roi[~roi_mask] = float(roi[roi_mask].mean())
        enhanced_roi = exposure.equalize_adapthist(roi, kernel_size=kernel_size, clip_limit=sk_clip)
        enhanced_roi = np.where(roi_mask, enhanced_roi, 0.0)
        out = arr.copy()
        out[y_min : y_max + 1, x_min : x_max + 1] = enhanced_roi
    else:
        out = exposure.equalize_adapthist(arr, kernel_size=kernel_size, clip_limit=sk_clip)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _blend_mask_boundary(
    processed: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray | None,
    feather: int,
) -> np.ndarray:
    if mask is None or not mask.any() or feather <= 0:
        return processed.astype(np.float32, copy=False)
    distance = distance_transform_edt(mask)
    alpha = np.clip(distance / float(feather), 0.0, 1.0).astype(np.float32)
    blended = processed.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def apply_iqe_norm(
    image: np.ndarray,
    config: IQENormConfig | None = None,
    **overrides: Any,
) -> np.ndarray:
    """Run IQE-NORM: percentile clipping, SRAD/guided denoise, and CLAHE."""
    cfg = config or IQENormConfig()
    if overrides:
        allowed = {field.name for field in fields(IQENormConfig)}
        unknown = set(overrides) - allowed
        if unknown:
            raise TypeError(f"Unknown IQE-NORM overrides: {sorted(unknown)}")
        cfg = replace(cfg, **overrides)

    original = _as_float01(image)
    mask = detect_foreground(original, cfg.background_threshold) if cfg.use_foreground_mask else None
    out = percentile_clip(original, low=cfg.pclip_low, high=cfg.pclip_high, mask=mask)
    out = denoise(
        out,
        method=cfg.denoise_method,
        mask=mask,
        n_iter=cfg.srad_iter,
        dt=cfg.srad_dt,
        window=cfg.srad_window,
        radius=cfg.guided_radius,
        eps=cfg.guided_eps,
    )
    out = clahe(
        out,
        clip_limit=cfg.clahe_clip,
        tile_grid=cfg.clahe_tile_grid,
        mask=mask,
        dynamic_grid=cfg.clahe_dynamic_grid,
    )
    out = _blend_mask_boundary(out, original, mask, cfg.mask_feather)
    if mask is not None:
        out = np.where(mask, out, np.float32(0.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)
