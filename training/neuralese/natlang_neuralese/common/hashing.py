"""SHA-256 helpers whose output feeds immutable corpus manifests.

Digests are lowercase hex. Changing any behaviour here changes provenance
hashes, so the golden digests in tests/test_common_hashing_jsonio.py pin it.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from .jsonio import canonical_json_bytes

FILE_CHUNK_BYTES = 1024 * 1024


def sha256_hex(data: bytes) -> str:
    """Hex SHA-256 of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file_hex(path: str | os.PathLike[str], chunk_size: int = FILE_CHUNK_BYTES, *, drop_cache: bool = False) -> str:
    """Hex SHA-256 of a file's bytes, read in bounded chunks (chunk size never changes the digest).

    ``drop_cache``: drop the file's cached pages first, so the hash reads what reached the disk.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        if drop_cache and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256_hex(value: Any) -> str:
    """Hex SHA-256 of `canonical_json_bytes(value)`."""
    return sha256_hex(canonical_json_bytes(value))
