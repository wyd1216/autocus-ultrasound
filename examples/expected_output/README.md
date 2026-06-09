# Expected Demo Output

Running `uv run autocus demo --output outputs/demo --device cpu` writes:

- `pipeline_result.json`: one entry per input image with ROI box, IQE-NORM method/stages and output paths, plaque score, unstable probability, label, device, and configured model-version paths.
- `predictions.csv`: one row per image with plaque score, unstable probability, and label.
- `metrics.json`: lightweight demo metadata, currently the number of processed images.
- `stage_outputs/*_iqe_norm.png`: deterministic normalized ROI.
- `stage_outputs/*_artery_mask.png`: deterministic foreground-style artery mask.
- `stage_outputs/*_plaque_mask.png`: deterministic threshold-based plaque mask.

Numeric values can vary slightly by dependency versions because the toy pipeline uses image statistics. The expected stable contract is the file set and JSON/CSV field names, not paper-level clinical performance.
