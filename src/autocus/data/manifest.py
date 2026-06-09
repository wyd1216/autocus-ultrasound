from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_SPLITS = {"train", "val", "test", "external"}


@dataclass(frozen=True)
class ManifestEntry:
    image: str
    mask: str | None = None
    bbox: list[float] | None = None
    label: int | str | None = None
    target: str | None = None
    source: str | None = None
    case_id: str | None = None

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "ManifestEntry":
        if "image" not in item:
            raise ValueError("Manifest entries require an 'image' field")
        image = str(item["image"])
        if Path(image).is_absolute():
            raise ValueError("Manifest image paths must be relative")
        for key in ("mask", "target"):
            value = item.get(key)
            if value is not None and Path(str(value)).is_absolute():
                raise ValueError(f"Manifest {key} paths must be relative")
        bbox = item.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("bbox must be [x1, y1, x2, y2]")
            bbox = [float(v) for v in bbox]
        return cls(
            image=image,
            mask=item.get("mask"),
            bbox=bbox,
            label=item.get("label"),
            target=item.get("target"),
            source=item.get("source"),
            case_id=item.get("case_id"),
        )


@dataclass(frozen=True)
class Manifest:
    path: Path
    splits: dict[str, list[ManifestEntry]]


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be a JSON object")
    unknown = set(payload) - VALID_SPLITS
    if unknown:
        raise ValueError(f"Unknown manifest split(s): {sorted(unknown)}")
    splits: dict[str, list[ManifestEntry]] = {}
    for split, entries in payload.items():
        if not isinstance(entries, list):
            raise ValueError(f"Manifest split '{split}' must be a list")
        splits[split] = [ManifestEntry.from_mapping(item) for item in entries]
    if not splits:
        raise ValueError("Manifest must contain at least one split")
    return Manifest(path=manifest_path, splits=splits)


def write_manifest(path: str | Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
