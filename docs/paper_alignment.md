# Paper-Code Alignment

This document maps the paper-facing AutoCUS terminology to public repository files. It is intended for reviewers who want to inspect where each named component is implemented.

## Scope Boundary

The repository exposes the research framework, model definitions, sanitized configs, public-data recipes, toy examples, and runnable smoke workflows. It does not include private clinical images, linked clinical metadata, annotation records, production deployment code, or paper checkpoints.

The bundled `autocus demo` command is a deterministic execution smoke test. It verifies I/O contracts and stage outputs without claiming paper-level clinical performance.

## Module Map

| Paper term | Public implementation | Config or command | Notes |
| --- | --- | --- | --- |
| AutoCUS-ROI / FocusNet | `src/autocus/models/roi/focusnet.py` | `configs/paper/focusnet_roi.yaml` | Anchor-free ROI detector definition and checkpoint smoke-loading support. |
| IQE-AAR / AARFormer | `src/autocus/models/iqe/aarformer.py` | `configs/paper/aarformer.yaml` | Annotation artifact removal network definition. |
| IQE-NORM | `src/autocus/preprocessing/norm.py` | `configs/paper/norm.yaml` | Traditional deterministic pipeline: foreground-aware percentile clipping, SRAD denoising, and CLAHE contrast enhancement. |
| IQE-SR / CU-HAT | `src/autocus/models/iqe/cuhat.py` | `configs/paper/cuhat.yaml` | Super-resolution network definition. |
| LARSNet for artery-region segmentation | `src/autocus/models/segmentation/larsnet.py` | `configs/paper/larsnet_long.yaml`, `configs/paper/larsnet_trans.yaml` | Shared architecture with long-axis and transverse-view configs. |
| PlaqueNet | `src/autocus/models/segmentation/plaque_net.py` | `configs/paper/plaquenet.yaml` | Plaque segmentation network definition. |
| PlaqueSENet | `src/autocus/models/classification/plaque_senet.py` | `configs/paper/plaquesenet.yaml` | Stable/unstable plaque classification model plus Grad-CAM helper support. |
| P0-P3 IQE ablation | `src/autocus/paper/ablation.py` | `uv run autocus paper ablate-iqe --manifest examples/toy_manifest.json` | Template writer for paper-style ablation manifests. |
| End-to-end public smoke workflow | `src/autocus/pipelines/autocus.py` | `uv run autocus demo --output outputs/demo --device cpu` | Toy-data workflow that writes JSON, CSV, metrics, and stage images. |

## Review Checklist

1. Run `uv run autocus demo --output outputs/demo --device cpu`.
2. Confirm `outputs/demo/pipeline_result.json`, `predictions.csv`, `metrics.json`, and `stage_outputs/` exist.
3. Run `uv run pytest -q` to instantiate all paper-facing model families with small tensors.
4. Run `uv run autocus weights verify`; the default paper checkpoints should report `not-released`.
5. Inspect `docs/privacy_and_limitations.md` for data and checkpoint boundaries.
