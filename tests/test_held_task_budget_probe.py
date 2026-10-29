import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import held_task_budget_probe as probe


def queues(tmp_path, count):
    result = []
    for i in range(count):
        directory = tmp_path / str(i)
        directory.mkdir()
        queue = directory / 'queue.jsonl'
        queue.write_text(json.dumps({'key': str(i)}) + '\n')
        result.append((queue, directory / 'journal.jsonl', [{'key': str(i)}]))
    return result


def test_probe_bounds_real_collector_processes_and_refills_slots(tmp_path, monkeypatch):
    events = tmp_path / 'events'
    script = tmp_path / 'collector.py'
    script.write_text('''import fcntl,os,sys,time
def record(delta):
    with open(sys.argv[1], 'a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.write(str(delta)+'\\n');stream.flush()
record(1);time.sleep(.08);record(-1)
''')
    popen = subprocess.Popen
    monkeypatch.setattr(probe.subprocess, 'Popen',
        lambda command, **options: popen([sys.executable, str(script), str(events)], **options))
    args = SimpleNamespace(server='http://localhost:1234', model_concurrency=1, case_seconds=1200)
    assert probe.collect_queues(queues(tmp_path, 5), {'workers': 2}, args, dict(os.environ)) == [0] * 5
    running = peak = 0
    for event in events.read_text().splitlines():
        running += int(event)
        peak = max(peak, running)
    assert peak == 2 and running == 0


def test_probe_failure_stops_the_exact_collector_descendant_group(tmp_path, monkeypatch):
    script = tmp_path / 'collector.py'
    child_pid = tmp_path / 'child.pid'
    script.write_text('''import subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
open(sys.argv[1],'w').write(str(child.pid))
time.sleep(60)
''')
    popen = subprocess.Popen
    monkeypatch.setattr(probe.subprocess, 'Popen',
        lambda command, **options: popen([sys.executable, str(script), str(child_pid)], **options))
    def fail(children, pending, launch, **options):
        launch(*pending[0])
        deadline = time.monotonic() + 5
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert child_pid.exists()
        raise RuntimeError('dispatcher failed')
    monkeypatch.setattr(probe.task_cadence, 'wait_collectors', fail)
    handlers = {value: signal.getsignal(value) for value in (signal.SIGTERM, signal.SIGINT)}
    args = SimpleNamespace(server='http://localhost:1234', model_concurrency=1, case_seconds=1200)
    with pytest.raises(RuntimeError, match='dispatcher failed'):
        probe.collect_queues(queues(tmp_path, 1), {'workers': 1}, args, dict(os.environ))
    assert all(signal.getsignal(value) == handler for value, handler in handlers.items())
    path = Path('/proc') / child_pid.read_text() / 'stat'
    deadline = time.monotonic() + 5
    while path.exists() and path.read_text().split()[2] != 'Z' and time.monotonic() < deadline:
        time.sleep(.01)
    assert not path.exists() or path.read_text().split()[2] == 'Z'
