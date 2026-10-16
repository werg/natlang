"""Exclusive ownership for resumable data-conversion output directories."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path


def acquire_output_lease(output: Path):
    """Acquire the existing durable nonblocking lease before touching output state.

    The returned open file owns the flock; callers must use it as a context
    manager so success, exceptions, and interruption all release ownership.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lease_path = output / '.conversion.lock'
    lease = lease_path.open('a+', encoding='utf-8')
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lease.close()
        raise RuntimeError(f'a data conversion already owns {lease_path}') from exc
    lease.seek(0)
    lease.truncate()
    lease.write(json.dumps({'pid': os.getpid(),
                            'started_at': datetime.now(timezone.utc).isoformat()}) + '\n')
    lease.flush()
    os.fsync(lease.fileno())
    return lease
