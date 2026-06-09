from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

UNRELEASED_STATUSES = {"not_released", "not-released", "optional"}


def load_registry(path: str | Path = "weights/registry.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_unreleased_weight(item: dict) -> bool:
    status = str(item.get("release_status", "")).lower()
    return status in UNRELEASED_STATUSES


def download_registered_weights(registry_path: str | Path = "weights/registry.json", output_dir: str | Path = "weights") -> list[Path]:
    registry = load_registry(registry_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for item in registry.get("weights", []):
        url = item.get("url")
        filename = item.get("filename")
        if is_unreleased_weight(item) or not url or url.startswith("https://example.org/"):
            continue
        target = out / filename
        urlretrieve(url, target)
        downloaded.append(target)
    return downloaded


def verify_registered_weights(registry_path: str | Path = "weights/registry.json", weight_dir: str | Path = "weights") -> dict[str, str]:
    registry = load_registry(registry_path)
    statuses: dict[str, str] = {}
    for item in registry.get("weights", []):
        filename = item["filename"]
        expected = item.get("sha256", "")
        path = Path(weight_dir) / filename
        if not path.exists():
            statuses[filename] = "not-released" if is_unreleased_weight(item) else "missing"
        elif expected and expected != "PENDING" and sha256_file(path) != expected:
            statuses[filename] = "sha256-mismatch"
        else:
            statuses[filename] = "ok"
    return statuses
