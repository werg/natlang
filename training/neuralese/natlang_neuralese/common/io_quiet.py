"""io-quiet markers: ask background data jobs to pause while a trainer writes a checkpoint.

The NVMe has no I/O scheduler or cgroup controller, so ionice cannot protect a multi-GB checkpoint write from the
data generators. A trainer holds ``io_quiet(reason)`` around the write; ``scripts/disk_guard.sh`` and the Python
disk guards pause while any live marker exists. Standard library only (disk_guard.sh runs this file directly).

Marker dir: $NATLANG_IO_QUIET_DIR, default ~/.local/state/natlang/io-quiet. One ``<unit-or-pid>.json`` per holder:
{pid, started, reason}. Live = younger than MAX_AGE_S and its pid alive. A holder in a container has a pid that is
meaningless to the host, so its marker records ``container: true`` and is live while younger than CONTAINER_AGE_S.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

MAX_AGE_S = 2 * 3600
CONTAINER_AGE_S = 30 * 60


def marker_dir() -> Path:
    return Path(os.environ.get('NATLANG_IO_QUIET_DIR') or os.path.expanduser('~/.local/state/natlang/io-quiet'))


@contextlib.contextmanager
def io_quiet(reason: str):
    """Hold a marker for the duration of the block; best effort (an unwritable dir never fails the save)."""
    unit = os.environ.get('NATLANG_LEDGER_UNIT') or str(os.getpid())
    path = marker_dir() / f'{unit}-{os.getpid()}.json'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'pid': os.getpid(), 'started': time.time(), 'reason': reason,
                                   'container': os.path.exists('/.dockerenv')}))
        tmp.replace(path)
    except OSError:
        path = None
    try:
        yield
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


def live_markers(now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    found = []
    for path in sorted(marker_dir().glob('*.json')):
        try:
            record = json.loads(path.read_text())
            age = now - float(record['started'])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        limit = CONTAINER_AGE_S if record.get('container') else MAX_AGE_S
        if age < limit and (record.get('container') or _alive(record.get('pid'))):
            found.append(dict(record, file=str(path)))
    return found


def wait_while_quiet(poll_s: float = 10.0, say=print) -> None:
    """Block while any live marker exists (the Python generators' between-batch check)."""
    said = False
    while (live := live_markers()):
        if not said:
            say(json.dumps({'io_quiet': 'pausing', 'reason': live[0].get('reason')}), flush=True)
            said = True
        time.sleep(poll_s)


if __name__ == '__main__':
    # exit 0 and print the first reason when a live marker exists, else exit 1
    live = live_markers()
    if live:
        print(live[0].get('reason', ''))
    sys.exit(0 if live else 1)
