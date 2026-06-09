# Weights

Large frozen model weights are not stored in Git. Use `weights/registry.json` plus `autocus weights download` and `autocus weights verify` after the external hosting records are finalized.

The code license does not automatically apply to model weights. Weight-use terms are recorded per release.

## Registry Workflow

1. Upload each frozen paper checkpoint to an external archive such as Zenodo or Hugging Face.
2. Fill in the final `url`, `sha256`, `version`, and hosting metadata in `weights/registry.json`.
3. Run `uv run autocus weights verify` after download.
4. Run `uv run autocus model-smoke --config <config> --checkpoint <weight> --device cpu` for each model.

Supported checkpoint payloads are raw PyTorch state dicts or dictionaries containing `model_state_dict`, `state_dict`, or `model`. DataParallel-style `module.` prefixes are stripped automatically.
