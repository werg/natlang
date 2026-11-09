"""Collector workers and the managed actor outlive a controller restart (restart in place, owner 2026-11-09).

A code change must not kill or redo work. ``restart_controller.py --in-place`` sends the controller SIGUSR1; the
controller (run under ``scripts/neuralese_data/controller_supervisor.py``, which relaunches exit code ``RESTART_EXIT``)
then

* during collection: stops dispatching, leaves every running collector worker and the managed actor running, records the
  workers in ``workers/worker-<tag>-detached.json`` and the actor in ``actor.pid.json``, and exits ``RESTART_EXIT``;
* during an update or SFT window: lets it finish and exits ``RESTART_EXIT`` at the next round boundary.

The relaunched controller (current main) re-attaches: an actor that still attests the round's policy is adopted instead of
started, each detached worker finishes its own episode on the code it started with (its journal row is the result; its
slot stays busy until it exits) and only entries nobody is running are dispatched. Workers and actor stay inside the
controller's own ledger unit and cgroup (the supervisor is the unit's main process), so no claim changes.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

from ..common.jsonio import write_json_durable

RESTART_EXIT = 75
RESTART = threading.Event()   # SIGUSR1: restart in place (detach at once during collection, else at the round boundary)


def install_restart_signal():
    signal.signal(signal.SIGUSR1, lambda *_: RESTART.set())


def restart_requested():
    return RESTART.is_set()


def identity(pid):
    """(pid, start ticks) of a live process, else None: the start time tells a reused pid from the recorded process."""
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text()
    except (OSError, ValueError):
        return None
    fields = stat[stat.rindex(')') + 2:].split()
    if fields[0] == 'Z':   # a zombie is finished
        return None
    return int(pid), int(fields[19])


def alive(record):
    return record is not None and identity(record['pid']) == (record['pid'], record['starttime'])


class AdoptedProcess:
    """The ``Popen`` surface the controller uses (poll, terminate, wait, pid) for a process another controller started."""

    def __init__(self, record):
        self.pid, self.record = record['pid'], record

    def poll(self):
        return None if alive(self.record) else 0

    def terminate(self):
        if alive(self.record):
            try:
                os.kill(self.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def wait(self, timeout=None):
        import subprocess
        deadline = None if timeout is None else time.monotonic() + timeout
        while alive(self.record):
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.1)
        return 0


def record_process(path, pid, **fields):
    value = {'schema': 'natlang.rl-detached-process/1', 'pid': pid, 'at': time.time(), **fields}
    ident = identity(pid)
    value['starttime'] = ident[1] if ident else None
    write_json_durable(path, value)
    return value


def load_record(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def detached_workers(directory):
    """[record] of the recorded detached collector workers of ``directory`` (workers/) that are still running."""
    found = []
    for path in sorted(Path(directory).glob('worker-*-detached.json')):
        record = load_record(path)
        if record and alive(record):
            found.append({**record, 'path': str(path)})
    return found


def entry_in_flight(record):
    """The key of the entry a detached worker is running: the last entry of its queue without a finish row."""
    from ..common.jsonio import read_jsonl
    queue, journal = Path(record['queue']), Path(record['journal'])
    finished = ({row['key'] for row in read_jsonl(journal) if row.get('event') == 'finish'} if journal.exists() else set())
    keys = [entry['key'] for entry in read_jsonl(queue)] if queue.exists() else []
    return next((key for key in reversed(keys) if key not in finished), None)
