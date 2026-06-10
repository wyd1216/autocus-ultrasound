# Quickstart

This guide verifies the public repository without private data or paper checkpoints.

## 1. Install

```bash
uv sync --extra dev
```

## 2. Run the Bundled Demo

```bash
uv run autocus demo --output outputs/demo --device cpu
```

The command uses:

- `configs/paper/autocus_pipeline.yaml`
- `examples/sample_input/demo_ultrasound.png`

It writes:

- `outputs/demo/pipeline_result.json`
- `outputs/demo/predictions.csv`
- `outputs/demo/metrics.json`
- `outputs/demo/stage_outputs/demo_ultrasound_iqe_norm.png`
- `outputs/demo/stage_outputs/demo_ultrasound_artery_mask.png`
- `outputs/demo/stage_outputs/demo_ultrasound_plaque_mask.png`

## 3. Inspect the Weight Registry

```bash
uv run autocus weights verify
```

The default paper checkpoint entries should report `not-released`. The demo does not require these checkpoints.

## 4. Run Validation Checks

```bash
uv run pytest -q
uv run ruff check
uv run python scripts/audit_release.py
```

The release audit scans text files for private absolute paths, internal framework imports, and protected metadata terms.
