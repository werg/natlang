"""Canonical JSON serialisation shared by manifest and identity hashing.

Dominant repo convention: sorted keys, compact separators, ensure_ascii=False,
UTF-8 bytes, no trailing newline. Variants that differ (no sort_keys, indent,
trailing newline, ASCII escaping) are deliberately NOT provided here.
"""
from __future__ import annotations

import json
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
