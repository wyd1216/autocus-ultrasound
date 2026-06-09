# AutoCUS Ultrasound

Public research framework for quality-gated carotid-ultrasound plaque profiling.

This repository is a paper-facing extraction of the AutoCUS research code. It contains model definitions, sanitized configuration templates, toy inputs, public-data recipes, inference utilities, training/evaluation scaffolds, and external weight-management utilities. It is not a medical device and is not intended for clinical diagnosis.

## Install

```bash
uv sync --extra dev
uv run autocus --help
```

## Demo

```bash
uv run autocus infer       --config configs/paper/autocus_pipeline.yaml       --input examples/sample_input       --output outputs/demo       --device cpu
```

Outputs include `pipeline_result.json`, `predictions.csv`, `metrics.json`, and stage images under `stage_outputs/`.

## Reproducibility Scope

The public repository supports code inspection, model instantiation, toy inference, public-data recipe construction, and re-training on user-provided or public data. Internal clinical images and linked clinical metadata are not included. Frozen paper weights are expected to be hosted externally and referenced through `weights/registry.json`.

## Main Modules

- ROI localization: FocusNet.
- Image-quality enhancement: AARFormer, deterministic masked robust min-max normalization, CU-HAT.
- Anatomy and plaque inference: LARSNetV5 and PlaqueNetV1.
- Plaque profiling: PlaqueSENet plus saliency helpers.

## Release Checks

```bash
uv run python scripts/audit_release.py
uv run pytest
uv run ruff check
```
