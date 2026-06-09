from __future__ import annotations

import sys
from pathlib import Path

SKIP_DIRS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "outputs"}
SKIP_FILES = {Path("scripts/audit_release.py")}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".cff"}
PRIVATE_PATTERNS = [
    "/data/workdir",
    "/Users/yudongwang117/CascadeProjects/Monai",
    "IntelliMedAI",
    "accession_number",
    "patient_id",
    "medical_record_number",
    "BaYuan",
    "Xinhua",
    "Kunshan",
    "secret_key",
    "api_token",
]


def audit_paths(root: Path) -> list[str]:
    offenders: list[str] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root)
        if rel in SKIP_FILES or path.is_dir() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PRIVATE_PATTERNS:
            if pattern in text:
                offenders.append(f"{rel}: {pattern}")
    return offenders


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    offenders = audit_paths(root)
    if offenders:
        print("Private release-audit patterns found:")
        print("\n".join(offenders))
        return 1
    print("Release audit passed: no private path patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
