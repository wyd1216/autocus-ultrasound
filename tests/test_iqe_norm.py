from __future__ import annotations

import numpy as np

from autocus.config import load_config
from autocus.preprocessing.norm import (
    IQENormConfig,
    apply_iqe_norm,
    detect_foreground,
    percentile_clip,
)


def _synth_ultrasound(seed: int = 0, shape: tuple[int, int] = (96, 96), border: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros(shape, dtype=np.float32)
    h, w = shape
    tissue = rng.uniform(0.15, 0.75, size=(h - 2 * border, w - 2 * border)).astype(np.float32)
    yy, xx = np.ogrid[: tissue.shape[0], : tissue.shape[1]]
    lumen = ((yy - tissue.shape[0] // 2) ** 2 + (xx - tissue.shape[1] // 2) ** 2) < 10**2
    tissue[lumen] *= 0.25
    image[border : h - border, border : w - border] = tissue
    return image


def test_detect_foreground_excludes_border_but_keeps_dark_lumen():
    image = _synth_ultrasound()
    mask = detect_foreground(image, background_threshold=5 / 255)

    assert mask is not None
    assert not mask[:10, :].any()
    assert not mask[:, :10].any()
    assert mask[48, 48]


def test_percentile_clip_is_foreground_mask_aware():
    image = _synth_ultrasound(border=28)
    mask = detect_foreground(image, background_threshold=5 / 255)

    masked = percentile_clip(image, low=1.0, high=99.0, mask=mask)
    unmasked = percentile_clip(image, low=1.0, high=99.0)

    assert masked.dtype == np.float32
    assert masked.min() >= 0.0
    assert masked.max() <= 1.0
    assert np.ptp(masked[mask]) >= np.ptp(unmasked[mask])
    assert np.all(masked[~mask] == 0.0)


def test_apply_iqe_norm_runs_traditional_three_stage_pipeline():
    image = _synth_ultrasound()
    cfg = IQENormConfig(srad_iter=2, clahe_tile_grid=4, mask_feather=2)

    first = apply_iqe_norm(image, cfg)
    second = apply_iqe_norm(image, cfg)

    assert first.shape == image.shape
    assert first.dtype == np.float32
    assert first.min() >= 0.0
    assert first.max() <= 1.0
    np.testing.assert_array_equal(first, second)


def test_paper_norm_config_uses_traditional_iqe_norm():
    cfg = load_config("configs/paper/norm.yaml")

    assert cfg["normalization"]["method"] == "iqe_norm_traditional"
    norm_cfg = IQENormConfig.from_mapping(cfg["normalization"])
    output = apply_iqe_norm(_synth_ultrasound(), norm_cfg)
    assert output.shape == (96, 96)
