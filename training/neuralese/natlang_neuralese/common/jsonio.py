"""Canonical JSON serialisation shared by manifest and identity hashing.

Dominant repo convention: sorted keys, compact separators, ensure_ascii=False,
UTF-8 bytes, no trailing newline. Variants that differ (no sort_keys, indent,
trailing newline, ASCII escaping) are deliberately NOT provided here.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def canonical_json_str(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json_str(value).encode("utf-8")


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp, `datetime.isoformat()` form (+00:00 suffix)."""
    return datetime.now(timezone.utc).isoformat()


def write_canonical_jsonl(path, rows) -> int:
    """Write rows as canonical JSON lines (one `canonical_json_str` per line) to a new file; refuses to overwrite,
    because published outputs are immutable. Returns the row count."""
    count = 0
    with open(path, "x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json_str(row) + "\n")
            count += 1
    return count


def write_json_durable(path, value) -> None:
    """Replace `path` with indented, key-sorted JSON (a state file, not a canonical form): write a `.pending`
    sibling, fsync it and rename it over `path`, so a crash leaves the old or the new file, never a torn one."""
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
