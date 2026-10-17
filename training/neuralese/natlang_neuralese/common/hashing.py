"""SHA-256 helpers whose output feeds immutable corpus manifests.

Digests are lowercase hex. Changing any behaviour here changes provenance
hashes, so the golden digests in tests/test_common_hashing_jsonio.py pin it.
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

from .jsonio import canonical_json_bytes

FILE_CHUNK_BYTES = 1024 * 1024
_CACHE_INDEX_LOCK = threading.RLock()
_CACHE_INDEX = {'path': None, 'signature': None, 'position': 0, 'entries': {}}


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


def _cached_file_digest(cache, key):
    """Index the existing append log once, then consume only complete appends."""
    import json
    cache_path = os.path.abspath(cache)
    with _CACHE_INDEX_LOCK:
        try:
            with open(cache, 'rb') as handle:
                status = os.fstat(handle.fileno())
                signature = (status.st_dev, status.st_ino, status.st_size,
                             status.st_mtime_ns, status.st_ctime_ns)
                previous = _CACHE_INDEX['signature']
                reset = (_CACHE_INDEX['path'] != cache_path or previous is None
                         or previous[:2] != signature[:2]
                         or signature[2] < previous[2]
                         or (signature[2] == previous[2] and signature != previous))
                if reset:
                    _CACHE_INDEX.update(path=cache_path, signature=None, position=0, entries={})
                if _CACHE_INDEX['signature'] != signature:
                    handle.seek(_CACHE_INDEX['position'])
                    while True:
                        line = handle.readline()
                        if not line or not line.endswith(b'\n'):
                            break
                        _CACHE_INDEX['position'] = handle.tell()
                        try:
                            entry = json.loads(line)
                        except (ValueError, UnicodeError):
                            continue
                        if isinstance(entry, dict) and isinstance(entry.get('key'), str) and entry.get('sha256'):
                            _CACHE_INDEX['entries'].setdefault(entry['key'], entry['sha256'])
                    _CACHE_INDEX['signature'] = signature
                return _CACHE_INDEX['entries'].get(key)
        except OSError:
            _CACHE_INDEX.update(path=cache_path, signature=None, position=0, entries={})
            return None


def sha256_file_hex_cached(path: str | os.PathLike[str], *, trust_archive_receipt: bool = True) -> str:
    """Return a cached digest, with an opt-in content-verified cache identity.

    Existing callers retain the archive-receipt and legacy cache behavior. A
    caller that sets ``trust_archive_receipt=False`` gets a cache keyed by the
    exact device/inode/size/mtime/ctime tuple; cache misses hash the bytes through
    one descriptor and reject changes observed during the read. This mode catches
    same-size rewrites even when mtime is restored, without turning prior
    checkpoint cache entries into expensive cache misses.
    """
    import json
    real = os.path.realpath(path)
    status = os.stat(real)
    signature = (status.st_dev, status.st_ino, status.st_size,
                 status.st_mtime_ns, status.st_ctime_ns)
    key = (f"content|{real}|{status.st_dev}|{status.st_ino}|{status.st_size}|{status.st_mtime_ns}|{status.st_ctime_ns}"
           if not trust_archive_receipt else f"{real}|{status.st_size}|{status.st_mtime_ns}")
    cache = _hash_cache_path()
    cached = _cached_file_digest(cache, key)
    if cached is not None and not trust_archive_receipt:
        current = os.stat(real)
        if (current.st_dev, current.st_ino, current.st_size,
                current.st_mtime_ns, current.st_ctime_ns) != signature:
            raise RuntimeError(f"file changed while reusing cached digest: {real}")
        return cached
    if cached is not None:
        return cached
    digest = _archive_receipt_digest(path, real, status.st_size) if trust_archive_receipt else None
    if digest is None:
        hasher = hashlib.sha256()
        with open(real, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns) != signature:
                raise RuntimeError(f"file changed before hashing: {real}")
            for chunk in iter(lambda: stream.read(FILE_CHUNK_BYTES), b""):
                hasher.update(chunk)
            after = os.fstat(stream.fileno())
        current = os.stat(real)
        if any((item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns) != signature
               for item in (after, current)):
            raise RuntimeError(f"file changed while hashing: {real}")
        digest = hasher.hexdigest()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "a") as handle:
            handle.write(json.dumps({"key": key, "sha256": digest}) + "\n")
    except OSError:
        pass
    return digest
