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
