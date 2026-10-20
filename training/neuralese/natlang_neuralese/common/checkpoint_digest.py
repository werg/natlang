"""Checkpoint digest publication and alias reuse; intrinsic source hashing is unchanged."""
import os
from .hashing import _hash_cache_path, _cached_file_digest, sha256_file_hex_cached


def _signature(path):
    status = os.stat(path)
    return (status.st_dev, status.st_ino, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)


def _physical_key(signature):
    return 'checkpoint-content|' + '|'.join(str(value) for value in signature)


def _record(keys, digest):
    import json
    cache = _hash_cache_path()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, 'a') as handle:
            handle.write(''.join(json.dumps({'key': key, 'sha256': digest}) + '\n' for key in keys))
    except OSError:
        # Failure to cache never invalidates authenticated bytes.
        pass


def verified_checkpoint_digest(path):
    """Share one content-verified digest across stable checkpoint hard links.

    Every logical binding retains its own path. Archive-trusting cache entries
    cannot establish this full physical identity; the existing strict hasher
    authenticates a miss, including any exact older path-scoped content entry.
    """
    real = os.path.realpath(path)
    signature = _signature(real)
    key = _physical_key(signature)
    digest = _cached_file_digest(_hash_cache_path(), key)
    if digest is None:
        digest = sha256_file_hex_cached(real, trust_archive_receipt=False)
        if _signature(real) != signature:
            raise RuntimeError('checkpoint changed while authenticating its digest')
        _record((key,), digest)
    if _signature(real) != signature:
        raise RuntimeError('checkpoint changed while reusing its digest')
    return digest


def remember_written_file_digest(path, digest, *, signature):
    """Cache a full sequential writer's digest for its exact committed inode.

    This is a writer API: callers must have hashed every successfully written
    byte. It does not authenticate an externally supplied digest receipt.
    """
    real = os.path.realpath(path)
    def current():
        return _signature(real)
    if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
            or current() != signature):
        raise ValueError('written digest lost its exact committed file identity')
    dev, inode, size, mtime, ctime = signature
    keys = (f'content|{real}|{dev}|{inode}|{size}|{mtime}|{ctime}', f'{real}|{size}|{mtime}',
            _physical_key(signature))
    _record(keys, digest)
    if current() != signature:
        raise RuntimeError('committed checkpoint changed while recording its writer digest')
