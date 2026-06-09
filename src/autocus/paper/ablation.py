from __future__ import annotations

import json
from pathlib import Path

from autocus.data.manifest import load_manifest


IQE_STAGES = ["p0_roi", "p1_aar", "p2_norm", "p3_sr"]


def write_iqe_ablation_template(manifest: str | Path, output_dir: str | Path) -> dict[str, object]:
    data = load_manifest(manifest)
    payload = {"stages": IQE_STAGES, "splits": {k: len(v) for k, v in data.splits.items()}}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "iqe_ablation_template.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
