"""Writer-only checkpoint digest publication; intrinsic source hashing is unchanged."""
import os
from .hashing import _hash_cache_path


def remember_written_file_digest(path, digest, *, signature):
    """Cache a full sequential writer's digest for its exact committed inode.

    This is a writer API: callers must have hashed every successfully written
    byte. It does not authenticate an externally supplied digest receipt.
    """
    import json
    real = os.path.realpath(path)
    def current():
        status = os.stat(real)
        return (status.st_dev, status.st_ino, status.st_size,
                status.st_mtime_ns, status.st_ctime_ns)
    if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
            or current() != signature):
        raise ValueError('written digest lost its exact committed file identity')
    dev, inode, size, mtime, ctime = signature
    keys = (f'content|{real}|{dev}|{inode}|{size}|{mtime}|{ctime}', f'{real}|{size}|{mtime}')
    cache = _hash_cache_path()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, 'a') as handle:
            handle.write(''.join(json.dumps({'key': key, 'sha256': digest}) + '\n' for key in keys))
    except OSError:
        # A cache failure must never invalidate an already durable checkpoint.
        pass
    if current() != signature:
        raise RuntimeError('committed checkpoint changed while recording its writer digest')
