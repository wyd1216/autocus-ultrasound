# Release Checklist

Use this checklist before tagging a public paper-facing release.

## Local Checks

```bash
uv sync --extra dev
uv run ruff check
uv run pytest -q
uv run python scripts/audit_release.py
uv run autocus demo --output outputs/demo --device cpu
uv run autocus weights verify
```

Expected default weight status:

```text
not-released
```

## Repository Checks

- `README.md` includes install, demo, reproducibility scope, and documentation links.
- `CITATION.cff` points to the GitHub repository.
- `pyproject.toml` includes `Homepage`, `Repository`, and `Issues` URLs.
- `.github/workflows/ci.yml` runs lint, tests, release audit, and the demo smoke test.
- `docs/paper_alignment.md` maps paper terminology to code paths and configs.
- `docs/privacy_and_limitations.md` states that private clinical data, linked metadata, and paper checkpoints are excluded.

## Privacy Checks

- No private absolute paths.
- No protected identifiers or linked clinical metadata.
- No raw clinical images or annotations.
- No private checkpoint files in Git.
- No internal deployment or product-UI code.

## Manuscript Checks

- Code availability text points to `https://github.com/wyd1216/autocus-ultrasound`.
- Data availability text states that private clinical cohorts are restricted.
- Checkpoint availability text matches the `not-released` registry status.
- Claimed public reproducibility is code-level and workflow-level unless approved paper checkpoints and data are separately released.

## Tagging

When the checks above are satisfied and the manuscript text has been updated, create an annotated release tag such as:

```bash
git tag -a v0.1.0-review -m "AutoCUS public research framework review release"
git push origin v0.1.0-review
```
