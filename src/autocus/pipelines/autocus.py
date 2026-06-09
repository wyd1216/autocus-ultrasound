from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from autocus.preprocessing.intensity import load_grayscale, save_grayscale
from autocus.preprocessing.norm import IQENormConfig, apply_iqe_norm
from autocus.preprocessing.roi import crop_box, foreground_bbox


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def norm_config_from_public_config(config: dict[str, Any]) -> IQENormConfig:
    payload = config.get("normalization", {})
    return IQENormConfig.from_mapping(payload)


def run_pipeline(config: dict[str, Any], input_path: str | Path, output_dir: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Run a deterministic public demo pipeline with optional model hooks."""
    in_path = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage_dir = out / "stage_outputs"
    stage_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    norm_config = norm_config_from_public_config(config)
    for image_path in iter_images(in_path):
        image = load_grayscale(image_path)
        roi_box = foreground_bbox(image)
        roi = crop_box(image, roi_box)
        normalized = apply_iqe_norm(roi, norm_config)
        threshold = float(config.get("pipeline", {}).get("plaque_threshold", 0.62))
        mask = normalized > threshold
        plaque_presence_score = float(mask.mean())
        unstable_probability = float(np.clip(normalized.mean() + normalized.std(), 0.0, 1.0))
        label = "unstable" if unstable_probability >= 0.5 else "stable"
        stem = image_path.stem
        norm_path = stage_dir / f"{stem}_iqe_norm.png"
        mask_path = stage_dir / f"{stem}_plaque_mask.png"
        artery_path = stage_dir / f"{stem}_artery_mask.png"
        save_grayscale(normalized, norm_path)
        save_grayscale(mask.astype(np.float32), mask_path)
        save_grayscale((normalized > 0.05).astype(np.float32), artery_path)
        item = {
            "input": str(image_path),
            "roi_box_xyxy": roi_box,
            "iqe_outputs": {
                "norm": str(norm_path),
                "norm_method": norm_config.method,
                "norm_stages": ["percentile_clip", norm_config.denoise_method, "clahe"],
            },
            "artery_mask": str(artery_path),
            "plaque_mask": str(mask_path),
            "plaque_presence_score": plaque_presence_score,
            "unstable_probability": unstable_probability,
            "label": label,
            "device": device,
            "model_versions": config.get("weights", {}),
        }
        results.append(item)
        rows.append({
            "image": str(image_path),
            "plaque_presence_score": plaque_presence_score,
            "unstable_probability": unstable_probability,
            "label": label,
        })
    payload = {"framework_version": "0.1.0", "images": results}
    (out / "pipeline_result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "plaque_presence_score", "unstable_probability", "label"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "metrics.json").write_text(json.dumps({"num_images": len(results)}, indent=2) + "\n", encoding="utf-8")
    return payload
