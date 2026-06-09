from __future__ import annotations

from pathlib import Path
import tomllib


REPOSITORY_URL = "https://github.com/wyd1216/autocus-ultrasound"


def test_public_repository_metadata_and_release_docs_are_present():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    urls = pyproject["project"]["urls"]

    assert urls["Repository"] == REPOSITORY_URL
    assert urls["Homepage"] == REPOSITORY_URL
    assert Path(".github/workflows/ci.yml").exists()
    assert Path("docs/paper_alignment.md").exists()
    assert Path("docs/release_checklist.md").exists()

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/paper_alignment.md" in readme
    assert "docs/release_checklist.md" in readme
