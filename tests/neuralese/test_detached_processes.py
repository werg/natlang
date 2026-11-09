"""Restart in place: workers and the actor outlive the controller and the relaunched controller re-attaches."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from natlang_neuralese.rl import detached_processes as dp, production
from natlang_neuralese.common.jsonio import read_jsonl

from test_collection_generations import Harness, finishes  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clear_restart():
    dp.RESTART.clear()
    yield
    dp.RESTART.clear()


def sleeper(seconds):
    return subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({seconds})'], start_new_session=True)


def test_adopted_process_follows_a_process_it_did_not_start(tmp_path):
    child = sleeper(30)
    try:
        record = dp.record_process(tmp_path / 'actor.pid.json', child.pid, policy_id='p')
        adopted = dp.AdoptedProcess(dp.load_record(tmp_path / 'actor.pid.json'))
        assert record['starttime'] and adopted.poll() is None and dp.alive(record)
        with pytest.raises(subprocess.TimeoutExpired):
            adopted.wait(timeout=.3)
        adopted.terminate()
        child.wait(timeout=10)
        assert adopted.poll() == 0 and not dp.alive(record) and adopted.wait(timeout=1) == 0
        # a recycled pid is not the recorded process: the start time differs
        assert not dp.alive({**record, 'pid': os.getpid()})
    finally:
        if child.poll() is None:
            child.kill()


def test_a_restart_request_detaches_the_running_collector_instead_of_terminating_it(tmp_path):
    script = tmp_path / 'worker.sh'
    script.write_text('#!/bin/sh\nsleep 30\n')
    queue, journal = tmp_path / 'queue-g1-0.jsonl', tmp_path / 'journal-g1-0.jsonl'
    queue.write_text(json.dumps({'key': 'k1'}) + '\n')
    command = ['/bin/sh', str(script), str(queue), str(journal), '--server', 'http://a']
    stop = type('Stop', (), {'requested': False})()
    control = production.RunControl(stop)
    dp.RESTART.set()
    detached = tmp_path / 'worker-g1-0-detached.json'
    with (tmp_path / 'log').open('a') as log:
        result = production._run_collector(command, log, control, detach_path=detached)
    assert result is None
    record = dp.load_record(detached)
    try:
        assert dp.alive(record) and record['queue'] == str(queue) and record['endpoint'] == 'http://a'
        assert dp.entry_in_flight(record) == 'k1'
        assert [r['pid'] for r in dp.detached_workers(tmp_path)] == [record['pid']]
        journal.write_text(json.dumps({'event': 'finish', 'key': 'k1'}) + '\n')
        assert dp.entry_in_flight(record) is None
    finally:
        os.kill(record['pid'], signal.SIGKILL)


def test_without_a_restart_request_a_stop_still_terminates_the_collector(tmp_path):
    script = tmp_path / 'worker.sh'
    script.write_text('#!/bin/sh\nsleep 30\n')
    command = ['/bin/sh', str(script), 'q', 'j', '--server', 'http://a']
    stop = type('Stop', (), {'requested': True})()
    started = time.monotonic()
    with (tmp_path / 'log').open('a') as log:
        code = production._run_collector(command, log, production.RunControl(stop), shutdown_grace=2,
                                         detach_path=tmp_path / 'd.json')
    assert code is not None and time.monotonic() - started < 20 and not (tmp_path / 'd.json').exists()


def test_a_relaunched_controller_waits_on_a_detached_worker_and_never_dispatches_its_entry_twice(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, entries=3)
    h.budget = 99
    workers = h.out / 'workers'
    workers.mkdir(parents=True)
    queue, journal = workers / 'queue-g1-0.jsonl', workers / 'journal-g1-0.jsonl'
    queue.write_text(json.dumps({'key': 's7-task-000', 'rl_endpoint': 'http://a'}) + '\n')
    journal.write_text(json.dumps({'event': 'start', 'key': 's7-task-000'}) + '\n')
    child = sleeper(2)
    dp.record_process(workers / 'worker-g1-0-detached.json', child.pid, queue=str(queue), journal=str(journal),
                      endpoint='http://a')
    try:
        # the detached worker finishes its episode while the new controller runs: its finish row is the result
        def finish_later():
            child.wait()
            with journal.open('a') as stream:
                stream.write(json.dumps({'event': 'finish', 'key': 's7-task-000', 'status': 'complete'}) + '\n')
        import threading
        threading.Thread(target=finish_later, daemon=True).start()
        report = h.collect(slot_poll_seconds=.1)
    finally:
        child.kill()
    assert report['collection_complete'] is True
    assert sorted(set(finishes(h.out))) == ['s7-task-000', 's7-task-001', 's7-task-002']
    assert 's7-task-000' not in [key for key, *_ in h.ran]          # nobody redid the detached worker's episode
    events = [row['event'] for row in read_jsonl(h.out / 'generations.jsonl')]
    assert 'attach' in events


def test_a_detached_worker_that_died_unfinished_is_dispatched_again(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, entries=2)
    h.budget = 99
    workers = h.out / 'workers'
    workers.mkdir(parents=True)
    queue, journal = workers / 'queue-g1-0.jsonl', workers / 'journal-g1-0.jsonl'
    queue.write_text(json.dumps({'key': 's7-task-000'}) + '\n')
    journal.write_text(json.dumps({'event': 'start', 'key': 's7-task-000'}) + '\n')
    child = sleeper(1)
    dp.record_process(workers / 'worker-g1-0-detached.json', child.pid, queue=str(queue), journal=str(journal),
                      endpoint='http://a')
    report = h.collect(slot_poll_seconds=.1)
    child.kill()
    assert report['collection_complete'] is True
    assert sorted(set(finishes(h.out))) == ['s7-task-000', 's7-task-001']


def test_a_restart_request_halts_dispatch_without_sealing_or_retiring_the_actor(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, entries=4)
    h.budget = 99
    retired = []
    monkeypatch.setattr(production, '_run_collector', lambda *a, **k: (dp.RESTART.set(), None)[1])
    report = h.collect(workers=1)
    assert report['collection_complete'] is False and report['collection_keys']['unfinished']
    assert not (h.out / 'blocks').exists() and retired == []


def test_the_supervisor_relaunches_on_75_forwards_signals_and_passes_other_codes_through(tmp_path):
    count = tmp_path / 'count'
    script = tmp_path / 'controller.py'
    script.write_text(f'import sys, pathlib\np = pathlib.Path({str(count)!r})\nn = int(p.read_text()) if p.exists() else 0\n'
                      f'p.write_text(str(n + 1))\nsys.exit(75 if n < 2 else 3)\n')
    done = subprocess.run([sys.executable, str(REPO / 'scripts/neuralese_data/controller_supervisor.py'), '--',
                           sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 3 and count.read_text() == '3' and done.stdout.count('restarting') == 2
    # SIGTERM is forwarded and ends the supervisor without a relaunch even when the child answers 75
    stubborn = tmp_path / 'stubborn.py'
    stubborn.write_text('import signal, sys, time\nsignal.signal(signal.SIGTERM, lambda *_: sys.exit(75))\n'
                        'print("ready", flush=True)\ntime.sleep(60)\n')
    proc = subprocess.Popen([sys.executable, str(REPO / 'scripts/neuralese_data/controller_supervisor.py'), '--',
                             sys.executable, str(stubborn)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == 'ready'
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 75 and 'restarting' not in proc.stdout.read()
