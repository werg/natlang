"""Exact provider artifacts and graceful child draining, with no cloud allocation."""
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from natlang_neuralese.rl import native_bridge as bridge, production
from natlang_neuralese.common.jsonio import write_json_durable


def request_fixture(tmp_path):
    parent = tmp_path/'parent.pt'; parent.write_bytes(b'control fixture parent')
    adapter = tmp_path/'adapter.json'; adapter.write_text('{}')
    config = tmp_path/'config.json'; out = tmp_path/'round'
    config.write_text(json.dumps({'native_state': True, 'native_participant': 'lfm',
        'checkpoint': str(parent), 'semantic_adapter': str(adapter), 'out': str(out)}))
    request = bridge.write_request(tmp_path/'requests', kind='round', out=out, config=config)
    return request, out


def test_worker_result_carries_exact_request_and_all_pending_bytes(tmp_path, monkeypatch):
    request, out = request_fixture(tmp_path)
    parent = json.loads(request.read_text())['parent_checkpoint']['path']
    def run(config, **kwargs):
        assert kwargs['defer_native_refresh'] is True and kwargs['stop'].requested is False
        checkpoint = out/'learner/checkpoint.pt'; checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'completed control fixture update')
        state = {'checkpoint': parent, 'pending_native_checkpoint': str(checkpoint),
                 'phase': 'awaiting_semantic_refresh'}
        write_json_durable(out/'state.json', state)
        return state
    monkeypatch.setattr(production, 'run', run)
    result_path = tmp_path/'result.json'
    result = bridge.execute(request, result_path)
    assert result['status'] == 'completed'
    assert bridge.verify_result(request, result_path)['checkpoint']['path'] == str(out/'learner/checkpoint.pt')
    (out/'learner/checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        bridge.verify_result(request, result_path)


def test_result_from_another_attempt_and_incomplete_fetch_are_rejected(tmp_path, monkeypatch):
    request, out = request_fixture(tmp_path)
    def run(config, **kwargs):
        state = {'checkpoint': json.loads(request.read_text())['parent_checkpoint']['path'], 'phase': 'requested_stop'}
        write_json_durable(out/'state.json', state)
        return state
    monkeypatch.setattr(production, 'run', run)
    result_path = tmp_path/'result.json'; bridge.execute(request, result_path)
    another = bridge.write_request(tmp_path/'requests', kind='round', out=out,
                                   config=tmp_path/'config.json')
    assert request != another
    with pytest.raises(ValueError, match='exact request'):
        bridge.verify_result(another, result_path)
    result = json.loads(result_path.read_text()); result['artifacts'] = []
    write_json_durable(result_path, result)
    with pytest.raises(ValueError, match='absent from fetched'):
        bridge.verify_result(request, result_path)


def test_stop_terminates_exact_child_and_waits_for_its_checkpoint(tmp_path):
    ready, saved = tmp_path/'ready', tmp_path/'saved'
    program = ('import signal,time,pathlib,sys\n'
        'def stop(sig,frame):\n'
        ' time.sleep(.05)\n'
        ' pathlib.Path(sys.argv[2]).write_text("checkpoint drained")\n'
        ' raise SystemExit(0)\n'
        'signal.signal(signal.SIGTERM,stop)\n'
        'pathlib.Path(sys.argv[1]).touch()\n'
        'while True: time.sleep(.02)\n')
    child = subprocess.Popen([sys.executable, '-c', program, str(ready), str(saved)])
    try:
        deadline = time.monotonic()+10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        assert production.RunControl(SimpleNamespace(requested=True)).wait(child) == 0
        assert saved.read_text() == 'checkpoint drained'
    finally:
        if child.poll() is None:
            child.terminate(); child.wait()


def test_provider_stop_is_cooperative_and_keeps_bridge_alive_to_drain(tmp_path):
    stop_file, drained = tmp_path/'stop.json', tmp_path/'drained'
    program = ('import pathlib,time,sys\n'
        'while not pathlib.Path(sys.argv[1]).exists(): time.sleep(.02)\n'
        'pathlib.Path(sys.argv[2]).write_text("remote checkpoint fetched")\n')
    child = subprocess.Popen([sys.executable, '-c', program, str(stop_file), str(drained)])
    try:
        control = production.RunControl(SimpleNamespace(requested=True))
        control.wait(child, stop_file=stop_file, stop_request={'sha256': 'exact-request'})
        assert drained.read_text() == 'remote checkpoint fetched'
        assert json.loads(stop_file.read_text())['request']['sha256'] == 'exact-request'
    finally:
        if child.poll() is None:
            child.terminate(); child.wait()


def test_cloud_queue_uses_only_files_and_drains_stopped_result(tmp_path, monkeypatch):
    import threading
    request, out = request_fixture(tmp_path)
    result, stop_file, queue = tmp_path/'result.json', tmp_path/'stop.json', tmp_path/'ipc'
    parent = json.loads(request.read_text())['parent_checkpoint']['path']
    def run(config, **kwargs):
        state = {'checkpoint': parent, 'phase': 'requested_stop'}
        write_json_durable(out/'state.json', state)
        return state
    monkeypatch.setattr(production, 'run', run)
    observed = []
    def provider():
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            jobs = bridge.queued_requests(queue)
            if jobs and stop_file.exists():
                observed.append(jobs[0])
                bridge.execute(request, result, stop=SimpleNamespace(requested=True))
                return
            time.sleep(.01)
        pytest.fail('queue client never published its exact request/stop')
    thread = threading.Thread(target=provider); thread.start()
    value = bridge.enqueue(request, result, queue, stop_file, stop=SimpleNamespace(requested=True))
    thread.join(timeout=10)
    assert not thread.is_alive() and value['status'] == 'stopped'
    assert observed[0]['request']['sha256'] == bridge.binding(request)['sha256']
    assert bridge.queued_requests(queue) == []
    assert json.loads(stop_file.read_text())['request']['sha256'] == bridge.binding(request)['sha256']
