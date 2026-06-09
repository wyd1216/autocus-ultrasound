# IQE-NORM

IQE-NORM is the deterministic traditional image-quality normalization stage used between ROI extraction and downstream anatomy/plaque inference.

The public implementation is a focused extraction of the normalization logic from the separate UltraEnhance project. Only the paper-relevant algorithmic core is included here. Private processing scripts, local dataset paths, generated experiment reports, figures, and clinical output tables are intentionally excluded.

## Pipeline

The implementation is in `src/autocus/preprocessing/norm.py`.

1. Foreground detection identifies the ultrasound acquisition window by trimming contiguous near-black outer bands. This avoids treating dark in-frame anatomy as background.
2. Foreground-aware percentile clipping rescales tissue intensities using `pclip_low` and `pclip_high`.
3. SRAD denoising suppresses speckle while preserving local structure. A guided-filter fallback is available for fast smoke workflows, but the paper config uses SRAD.
4. CLAHE enhances local contrast with conservative clipping and foreground-aware background filling.
5. Optional boundary feathering blends the processed foreground into the original near the detected acquisition-window edge.

## Paper Defaults

The default paper config is `configs/paper/norm.yaml`:

```yaml
normalization:
  method: iqe_norm_traditional
  background_threshold: 0.0196078431372549
  use_foreground_mask: true
  mask_feather: 8
  pclip_low: 0.5
  pclip_high: 99.5
  denoise_method: srad
  srad_iter: 30
  srad_dt: 0.05
  srad_window: 3
  clahe_clip: 1.5
  clahe_tile_grid: 8
  clahe_dynamic_grid: false
```

## Public Demo

The bundled demo uses the same IQE-NORM config through `configs/paper/autocus_pipeline.yaml`:

```bash
uv run autocus demo --output outputs/demo --device cpu
```

The stage output is written as `outputs/demo/stage_outputs/*_iqe_norm.png`, and `pipeline_result.json` records `norm_method` and `norm_stages`.

## Reproducibility Boundary

The public repository provides deterministic algorithm code and tests for shape, value range, mask behavior, deterministic output, and pipeline integration. Paper-level numerical results still depend on the restricted clinical cohorts and frozen checkpoints described in `docs/reproducibility.md` and `weights/README.md`.
