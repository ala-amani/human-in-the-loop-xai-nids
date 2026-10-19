"""Audit tracked research files for inadvertent attribution and identity leaks."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".ipynb", ".md", ".txt", ".json", ".csv", ".log", ".toml", ".yml", ".yaml", ".rst"}
EXACT_TERMS = (
    "".join(("co", "dex")),
    "".join(("cl", "aude")),
    "".join(("chat", "g", "p", "t")),
    "".join(("open", "ai")),
    "".join(("anth", "ropic")),
    "".join(("co", "pilot")),
    "".join(("ala", "-", "amani")),
    "".join(("shah", "rekord")),
)
STANDALONE_TOKENS = ("".join(("g", "p", "t")),)
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}


def test_research_files_do_not_embed_assistant_branding_or_identifying_names() -> None:
    """Check current repository files, including notebooks and archived output logs."""
    matches = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace").casefold()
        for term in EXACT_TERMS:
            if term in text:
                matches.append((str(path.relative_to(ROOT)), term))
        for token in STANDALONE_TOKENS:
            if re.search(r"\b" + re.escape(token) + r"\b", text):
                matches.append((str(path.relative_to(ROOT)), token))
    assert not matches, f"Unexpected attribution or identity strings in tracked research files: {matches}"
