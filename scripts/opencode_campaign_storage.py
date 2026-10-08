"""Small filesystem helpers for immutable OpenCode campaign launch records."""

from pathlib import Path


def ensure_parent(path: str | Path) -> Path:
    """Create and return the parent directory for a fresh campaign artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.parent
