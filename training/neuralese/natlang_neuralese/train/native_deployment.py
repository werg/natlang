"""Exact path/SHA-256 artifact bindings for native deployment declarations."""
from __future__ import annotations
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached


def binding(path, expected=None):
    path = Path(path).resolve(strict=True)
    value={'path': str(path), 'sha256': sha256_file_hex_cached(path)}
    if expected is not None and (not isinstance(expected,dict)
            or set(expected) not in ({'path','sha256'},{'path','sha256','bytes'})
            or value!={key:expected.get(key)for key in ('path','sha256')}
            or ('bytes'in expected and(type(expected['bytes'])is not int
                or expected['bytes']<0 or expected['bytes']!=path.stat().st_size))):
        raise ValueError('native deployment artifact binding changed')
    return value
