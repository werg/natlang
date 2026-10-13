"""On-disk cache of what a pass over a large text-data file computes (validation, dedup, prefix-merge decisions).

A direct stage reads its multi-gigabyte text file several times at every start and handoff (each plan rebuild and the
real load repeat the same validation, duplicate filtering and prefix-merge passes). Those results are pure functions of the
file bytes, the tokenizer and the windowing policy, so they are keyed by the file's sha256 (``sha256_file_hex_cached``),
the tokenizer fingerprint and a policy tag, and reused across processes (plan-only and the real handoff). A cache miss
or an unreadable entry recomputes exactly as before; nothing is ever trusted without its key.

Location: ``$NATLANG_ROW_INDEX_CACHE`` or ``~/.cache/natlang/row-index``. Entries are small (line indexes and merge maps).
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path


def cache_dir() -> Path:
    return Path(os.environ.get('NATLANG_ROW_INDEX_CACHE') or Path.home() / '.cache' / 'natlang' / 'row-index')


def entry_path(kind: str, *key_parts) -> Path:
    digest = hashlib.sha256('\x1f'.join(str(part) for part in key_parts).encode()).hexdigest()[:40]
    return cache_dir() / f'{kind}-{digest}.pkl'


def load(path: Path):
    """The stored value, or None (missing, unreadable or from another schema)."""
    try:
        with open(path, 'rb') as handle:
            value = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get('schema') == 'natlang.row-index/1' else None


def save(path: Path, value: dict) -> None:
    """Atomic write; a failure to cache never fails the run."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        with open(temporary, 'wb') as handle:
            pickle.dump({**value, 'schema': 'natlang.row-index/1'}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except OSError:
        pass
