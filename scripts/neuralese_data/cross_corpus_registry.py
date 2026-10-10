"""Resolve cross-corpus index references: a directory, or a corpus registry id (training/neuralese_corpora.json,
kind "cross-corpus-index") whose snapshot lives at <repo>/<entry path> on every machine that synced it."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = "training/neuralese_corpora.json"


def resolve(ref, repo: Path = REPO) -> tuple[Path, str | None]:
    """(index directory, registry id or None). An existing directory wins; otherwise `ref` must be a registered
    cross-corpus index id."""
    path = Path(ref)
    if path.is_dir():
        return path, None
    entries = {e["id"]: e for e in json.loads((repo / REGISTRY).read_text())["corpora"]}
    entry = entries.get(str(ref))
    if entry is None or entry.get("kind") != "cross-corpus-index":
        raise FileNotFoundError(f"cross-corpus index {ref!r}: neither a directory nor a registered index id")
    root = repo / entry["path"]
    if not (root / "index.json").is_file():
        raise FileNotFoundError(f"cross-corpus index {ref!r} is registered but not synced here: {root} "
                                f"(scripts/sync_training_corpora.py sync --id {ref})")
    return root, entry["id"]
