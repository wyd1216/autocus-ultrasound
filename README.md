# AutoCUS Ultrasound

[![CI](https://github.com/wyd1216/autocus-ultrasound/actions/workflows/ci.yml/badge.svg)](https://github.com/wyd1216/autocus-ultrasound/actions/workflows/ci.yml)

Public research framework for quality-gated carotid-ultrasound plaque profiling.

This repository is a paper-facing extraction of the AutoCUS research code. It contains model definitions, sanitized configuration templates, toy inputs, public-data recipes, inference utilities, and training/evaluation scaffolds. It is not a medical device and is not intended for clinical diagnosis.

## Install

```bash
uv sync --extra dev
uv run autocus --help
```

## Demo

Run the bundled CPU demo:

```bash
uv run autocus demo --output outputs/demo --device cpu
```

The equivalent explicit pipeline command is:

```bash
uv run autocus infer       --config configs/paper/autocus_pipeline.yaml       --input examples/sample_input       --output outputs/demo       --device cpu
```

Outputs include `pipeline_result.json`, `predictions.csv`, `metrics.json`, and stage images under `stage_outputs/`.

See `docs/quickstart.md` for a reviewer-oriented walkthrough.

## Optional Checkpoint Smoke Test

The public repository runs without the paper checkpoints. If you have an approved local checkpoint, verify that it matches a sanitized config before running larger jobs:

```bash
uv run autocus model-smoke \
  --config configs/paper/focusnet_roi.yaml \
  --checkpoint weights/autocus_focusnet_roi_v1.pth \
  --device cpu
```

This command instantiates the model, loads common checkpoint formats (`state_dict`, `model_state_dict`, raw state dicts, and `module.`-prefixed state dicts), and reports missing or unexpected keys. The default `weights/registry.json` records paper checkpoint names as `not_released`, so `autocus weights verify` is informational rather than a required download step.

## Reproducibility Scope

The public repository supports code inspection, model instantiation, toy inference, public-data recipe construction, and re-training on user-provided or public data. Internal clinical images and linked clinical metadata are not included. The frozen checkpoints used for the paper experiments are not required for the public code demo and are not released by default because they were trained on restricted clinical data.

## Main Modules

- ROI localization: FocusNet.
- Image-quality enhancement: AARFormer, deterministic masked robust min-max normalization, CU-HAT.
- Anatomy and plaque inference: LARSNetV5 and PlaqueNetV1.
- Plaque profiling: PlaqueSENet plus saliency helpers.

## Documentation

- `docs/quickstart.md`: reviewer-oriented install and demo walkthrough.
- `docs/paper_alignment.md`: map from paper terminology to public code paths and configs.
- `docs/reproducibility.md`: reproducibility levels and limits.
- `docs/data_recipes.md`: public-data recipe notes.
- `docs/privacy_and_limitations.md`: data, metadata, checkpoint, and clinical-use boundaries.
- `docs/release_checklist.md`: checks to run before tagging a public release.

## Release Checks

```bash
uv run python scripts/audit_release.py
uv run pytest
uv run ruff check
```
