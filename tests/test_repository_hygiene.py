"""Audit tracked research files for strings the authors want kept out of the review snapshot.

The terms themselves are not stored in the repository. Provide them locally, one per line, in
``tests/.hygiene_terms.txt`` (ignored by Git) or as a comma-separated ``HYGIENE_TERMS`` environment
variable. A term written as ``word:<term>`` matches only as a whole word. The test is skipped when
no terms are configured.
"""

import os
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
TERMS_FILE = Path(__file__).with_name(".hygiene_terms.txt")
TEXT_EXTENSIONS = {".py", ".ipynb", ".md", ".txt", ".json", ".csv", ".log", ".toml", ".yml", ".yaml", ".rst"}
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}


def _load_terms() -> list[str]:
    terms = os.environ.get("HYGIENE_TERMS", "").split(",")
    if TERMS_FILE.is_file():
        terms += TERMS_FILE.read_text(encoding="utf-8").splitlines()
    return sorted({t.strip().casefold() for t in terms if t.strip() and not t.strip().startswith("#")})


def test_research_files_do_not_embed_listed_terms() -> None:
    """Check current repository files, including notebooks and archived output logs."""
    terms = _load_terms()
    if not terms:
        pytest.skip("no local hygiene terms configured")
    patterns = [
        (term, re.compile(r"\b" + re.escape(term[5:]) + r"\b") if term.startswith("word:") else None)
        for term in terms
    ]
    matches = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS or path == TERMS_FILE:
            continue
        if EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace").casefold()
        for term, pattern in patterns:
            if (pattern.search(text) if pattern else term in text):
                matches.append((str(path.relative_to(ROOT)), term))
    assert not matches, f"Listed terms found in tracked research files: {matches}"
