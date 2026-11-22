"""Scratch directories owned by one process (an ``owner.pid`` file inside), swept when their owner is gone.

A process stopped by SIGKILL (or a unit's stop timeout) never runs its cleanup, and on the GB10 a tmpfs or spill
directory left behind is memory outside the ledger: a 7.5 GB /dev/shm weight-slot set kept a 100 GB SFT from being
admitted (2026-11-22), and paused operator fits leaked 7.6 GB spill stores each. Every creator claims its directory
(``claim``), removes it on exit (``remove_on_exit``: atexit and a default SIGTERM), and sweeps siblings of dead owners
when it starts (``sweep_dead``). A directory without an owner file is never touched."""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import threading
from pathlib import Path

OWNER_FILE = 'owner.pid'


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def owner(path) -> int | None:
    try:
        return int((Path(path) / OWNER_FILE).read_text().strip())
    except (OSError, ValueError):
        return None


def sweep_dead(root, pattern) -> list[str]:
    """Remove the directories under ``root`` matching ``pattern`` whose recorded owner process is gone."""
    removed = []
    for path in Path(root).glob(pattern):
        pid = owner(path)
        if pid is not None and pid != os.getpid() and not alive(pid):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
    return removed


def claim(path) -> Path:
    """Create ``path`` (if needed) and record this process as its owner."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / OWNER_FILE).write_text(f'{os.getpid()}\n')
    return path


_REMOVE: list[Path] = []
_LOCK = threading.Lock()
_INSTALLED = {'atexit': False, 'sigterm': False}


def _remove_all():
    with _LOCK:
        paths, _REMOVE[:] = list(_REMOVE), []
    for path in paths:
        if owner(path) == os.getpid():
            shutil.rmtree(path, ignore_errors=True)


def _on_sigterm(signum, frame):
    _remove_all()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)


def remove_on_exit(path) -> None:
    """Remove ``path`` when this process exits: at interpreter exit (also after an exception) and on SIGTERM when no
    other SIGTERM handler is installed (a trainer that installs its own handler exits through atexit)."""
    with _LOCK:
        _REMOVE.append(Path(path))
        if not _INSTALLED['atexit']:
            atexit.register(_remove_all)
            _INSTALLED['atexit'] = True
    if (not _INSTALLED['sigterm'] and threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL):
        signal.signal(signal.SIGTERM, _on_sigterm)
        _INSTALLED['sigterm'] = True


def release(path) -> None:
    """Remove ``path`` now (and forget it)."""
    path = Path(path)
    with _LOCK:
        _REMOVE[:] = [p for p in _REMOVE if p != path]
    if owner(path) == os.getpid():
        shutil.rmtree(path, ignore_errors=True)
