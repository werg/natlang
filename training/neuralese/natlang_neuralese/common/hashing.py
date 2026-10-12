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


def _hash_cache_path():
    from pathlib import Path
    return Path(os.environ.get("NATLANG_HASH_CACHE") or Path.home() / ".cache" / "natlang" / "sha256-cache.jsonl")


def _archive_receipt_digest(path, real: str, size: int) -> str | None:
    """The digest an archive receipt (``<path>.archived.json``, scripts/archive_to_external.py) recorded for the file a
    symlink now points to: the mover verified the copy against it."""
    import json
    from pathlib import Path
    records = []
    try:
        records.append(json.loads(Path(str(path) + ".archived.json").read_text()))
    except (OSError, ValueError):
        pass
    # A path given already resolved to the archive: the archive's manifest (archive root/manifest.jsonl) holds it.
    from .paths import resolve
    manifest = resolve("data_hdd", "natlang-development-data", "archive", "manifest.jsonl")
    if not records and real.startswith(str(manifest.parent) + os.sep):
        try:
            with open(manifest) as handle:
                for line in handle:
                    if real in line:
                        try:
                            records.append(json.loads(line))
                        except ValueError:
                            continue
        except OSError:
            pass
    for record in reversed(records):
        if os.path.realpath(record.get("dest", "")) == real and record.get("bytes") == size and record.get("sha256"):
            return record["sha256"]
    return None


def sha256_file_hex_cached(path: str | os.PathLike[str]) -> str:
    """``sha256_file_hex`` without re-reading a file already hashed: verified digests are cached by (realpath, size,
    mtime_ns) in ``~/.cache/natlang/sha256-cache.jsonl`` (NATLANG_HASH_CACHE), and a symlink left by the archive mover
    takes the digest its receipt recorded. A resume must never re-read a 47 GB predecessor state from the HDD
    (v4 resume 2026-10-12: a 2 h stall at 7 MB/s)."""
    import json
    real = os.path.realpath(path)
    status = os.stat(real)
    key = f"{real}|{status.st_size}|{status.st_mtime_ns}"
    cache = _hash_cache_path()
    try:
        with open(cache) as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("key") == key:
                    return entry["sha256"]
    except OSError:
        pass
    digest = _archive_receipt_digest(path, real, status.st_size) or sha256_file_hex(real)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "a") as handle:
            handle.write(json.dumps({"key": key, "sha256": digest}) + "\n")
    except OSError:
        pass
    return digest
