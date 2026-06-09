from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autocus.data.manifest import load_manifest


def evaluate_from_manifest(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Write a reproducibility placeholder metrics file for a manifest."""
    manifest = load_manifest(manifest_path)
    total = sum(len(v) for v in manifest.splits.values())
    metrics = {"num_items": total, "splits": {k: len(v) for k, v in manifest.splits.items()}}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
