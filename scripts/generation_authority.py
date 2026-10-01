"""Short shared lock for generation authority read/modify/write operations."""
from contextlib import contextmanager
import fcntl


@contextmanager
def authority_lock(path):
    with path.with_name(path.name + '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield
