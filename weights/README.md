# Weights

The public code repository does not require paper checkpoints. The default `weights/registry.json` records the expected checkpoint filenames, but marks them as `not_released`.

This preserves a clear place to document approved local or future checkpoint releases. The Apache-2.0 code license does not automatically apply to model weights.

## Registry Workflow

For the current public release:

1. Leave paper checkpoints out of Git.
2. Run `uv run autocus weights verify` to inspect the registry. Unreleased entries should report `not-released`.
3. Use `uv run autocus model-smoke --config <config> --checkpoint <weight> --device cpu` only when you have an approved local checkpoint.

If a checkpoint is approved for later public release, update its registry entry with `release_status`, `url`, `sha256`, `version`, and weight-use terms, then run `uv run autocus weights verify` after download.

Supported checkpoint payloads are raw PyTorch state dicts or dictionaries containing `model_state_dict`, `state_dict`, or `model`. DataParallel-style `module.` prefixes are stripped automatically.
