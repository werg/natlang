"""Bounded-memory digest for canonical JSON arrays of ordered metadata rows."""
from __future__ import annotations

import hashlib
from typing import Any

from .jsonio import canonical_json_bytes


def canonical_json_sequence_sha256_hex(values: list[Any] | tuple[Any, ...]) -> str:
    """Hash the exact canonical JSON encoding of a list/tuple without joining it.

    JSON array encoding is `[` followed by each element's canonical encoding,
    separated by commas, and `]`. Hashing those exact byte pieces preserves the
    output of `canonical_json_sha256_hex(values)` while keeping temporary memory
    bounded by the largest individual element.
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError('canonical JSON sequence digest requires a list or tuple')
    digest = hashlib.sha256()
    digest.update(b'[')
    for index, value in enumerate(values):
        if index:
            digest.update(b',')
        digest.update(canonical_json_bytes(value))
    digest.update(b']')
    return digest.hexdigest()
