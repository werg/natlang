from types import SimpleNamespace

import pytest

from scripts.coreweave import rl_cloud_autopilot as autopilot


def test_entitlement_does_not_start_stale_recipe_while_launch_is_held(monkeypatch, tmp_path):
    monkeypatch.setattr(autopilot.rl_cloud, 'hub', lambda run: tmp_path)
    monkeypatch.setattr(autopilot, 'lineage_state', lambda *a: pytest.fail('held launch inspected stale smoke'))
    monkeypatch.setattr(autopilot, 'launch_on_entitled', lambda *a, **k: pytest.fail('held launch started GPU'))
    run = {'launch_hold': 'replace obsolete recipe with qualified current source'}
    autopilot.one_pass(run, SimpleNamespace(interval=3600, smoke='smoke', lineages=['production']),
                       {'entitled': ['account']})


def test_actual_launch_rereads_hold_before_spending(monkeypatch):
    monkeypatch.setattr(autopilot.rl_cloud, 'load_run', lambda path: {'launch_hold': 'current operator hold'})
    with pytest.raises(ValueError, match='current operator hold'):
        autopilot.rl_cloud.launch({'_path': 'run.json'}, 'production')


@pytest.mark.parametrize('stop_code,status,get_code,confirmed', [
    (0, 'running', 0, False),
    (1, 'running', 0, False),
    (0, 'terminated', 1, False),
    (1, 'completed', 0, True),
    (0, 'terminated', 0, True),
])
def test_probe_shutdown_requires_terminal_provider_observation(monkeypatch, stop_code, status, get_code, confirmed):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 30
        return SimpleNamespace(returncode=stop_code if command[2] == 'stop' else get_code,
                               stdout='' if command[2] == 'stop' else 'STATUS '+status+'\n')
    monkeypatch.setattr(autopilot.subprocess, 'run', run)
    assert autopilot.stop_probe('probe-1')[0] is confirmed
    assert calls[0][-2:] == ['--graceful-shutdown-seconds', '1']


def test_unresolved_probe_is_durable_before_shutdown_and_blocks_new_allocations(monkeypatch, tmp_path):
    monkeypatch.setattr(autopilot.rl_cloud, 'hub', lambda run: tmp_path)
    monkeypatch.setattr(autopilot, 'accounts', lambda: ['account'])
    monkeypatch.setattr(autopilot.rl_cloud.jobs, 'validate', lambda job: job)
    launches = []
    def launch(*args, **kwargs):
        assert args[0]['resources'] == {'gpu':1}
        launches.append(kwargs['account'])
        return {'sandbox_id':'probe-1'}
    monkeypatch.setattr(autopilot.rl_cloud.cwctl, 'launch', launch)
    def unresolved(sandbox_id):
        saved = autopilot.rl_cloud.read_json(tmp_path/'autopilot.json', {})
        assert saved['probe_cleanup']['sandbox_id'] == sandbox_id
        return False, 'still running'
    monkeypatch.setattr(autopilot, 'stop_probe', unresolved)
    auto = {}
    args = SimpleNamespace(interval=3600, smoke='smoke', lineages=['production'])
    autopilot.one_pass({'launch_hold':'stale recipe'}, args, auto)
    assert auto['entitled'] == [] and auto['probe_cleanup']['sandbox_id'] == 'probe-1'
    # Simulate monitor restart: cleanup survives independently of live process state.
    restored = autopilot.rl_cloud.read_json(tmp_path/'autopilot.json', {})
    autopilot.one_pass({'launch_hold':'stale recipe'}, args, restored)
    assert launches == ['account']
    monkeypatch.setattr(autopilot, 'stop_probe', lambda ident:(True, 'verified terminated'))
    autopilot.one_pass({'launch_hold':'stale recipe'}, args, restored)
    assert 'probe_cleanup' not in restored and restored['entitled'] == ['account']
    assert launches == ['account']


def test_probe_cleanup_timeout_remains_unconfirmed(monkeypatch):
    import subprocess
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 30)
    monkeypatch.setattr(autopilot.subprocess, 'run', timeout)
    assert autopilot.stop_probe('probe-1')[0] is False


def test_monitor_restart_preserves_probe_deadline_and_retries_cleanup_promptly():
    assert autopilot.next_delay({'probes':[{'at':1000}]},3600,now=4000) == 600
    assert autopilot.next_delay({'probes':[{'at':1000}]},3600,now=5000) == 1
    assert autopilot.next_delay({'probe_cleanup':{'sandbox_id':'probe-1'}},3600) == 30
    assert autopilot.next_delay({'entitled':['account']},3600) == 300


def test_launch_fails_and_records_a_sandbox_that_never_started(monkeypatch, tmp_path):
    rl_cloud = autopilot.rl_cloud
    monkeypatch.setattr(rl_cloud, 'hub', lambda run: tmp_path)
    (tmp_path / 'run-state.json').write_text('{"bundle": "/b.tar"}')
    monkeypatch.setattr(rl_cloud, 'newest_verified', lambda run, name: {'seq': 1, 'trained_rounds': 0})
    monkeypatch.setattr(rl_cloud, 'render_job', lambda *a: {'id': 'job', 'resources': {}, 'lifetime_hours': 1})
    monkeypatch.setattr(rl_cloud, 'lease_blobs', lambda *a, **k: None)
    monkeypatch.setattr(rl_cloud, 'record_digests', lambda *a: set())
    monkeypatch.setattr(rl_cloud, 'parent_capability', lambda *a: {})
    monkeypatch.setattr(rl_cloud, 'diloco_secrets', lambda *a: {})
    monkeypatch.setattr(rl_cloud, 'follow_logs', lambda *a: None)
    monkeypatch.setattr(rl_cloud, 'note', lambda *a: None)
    released, stopped = [], []
    monkeypatch.setattr(rl_cloud, 'release_consumer', lambda run, consumer: released.append(consumer))
    monkeypatch.setattr(rl_cloud.subprocess, 'run', lambda argv, check: stopped.append(argv[-1]))
    monkeypatch.setattr(rl_cloud, 'started', lambda sandbox_id: False)

    def launch(job, *a, on_created, account, **k):
        on_created('sbx-1')
        return {'sandbox_id': 'sbx-1', 'wait_error': 'StartupTimeout: no node was assigned'}
    monkeypatch.setattr(rl_cloud.cwctl, 'launch', launch)
    run = {'lineages': {'d1': {'account': 'cw2'}}}
    with pytest.raises(SystemExit, match='never started'):
        rl_cloud.launch(run, 'd1')
    import json
    live = json.loads((tmp_path / 'run-state.json').read_text())['lineages']['d1']
    assert live['status'] == 'failed' and 'StartupTimeout' in live['error']
    assert stopped == ['sbx-1'] and released == ['launch:job']


def test_add_runtime_makes_relaunches_restore_an_in_place_adopted_runtime(monkeypatch, tmp_path):
    rl_cloud = autopilot.rl_cloud
    monkeypatch.setattr(rl_cloud, 'hub', lambda run: tmp_path / 'hub')
    monkeypatch.setattr(rl_cloud, 'locate', lambda run, digest: None)
    put = []
    monkeypatch.setattr(rl_cloud, 'put_local', lambda run, path: put.append(path))
    seed = {'kind': 'runtime', 'path': '/rt/v44', 'sha256': 'a' * 64}
    rl_cloud.store_json(tmp_path / 'hub/exports/lin/seed.json', {'inputs': [seed]})
    runtime = tmp_path / 'v45'
    runtime.mkdir()
    (runtime / 'runtime-input-closure.json').write_text('{}')
    archive = tmp_path / 'v45.tar.zst'
    archive.write_bytes(b'runtime')
    item = rl_cloud.add_runtime({}, 'lin', runtime, archive)
    assert put == [archive] and item['path'] == str(runtime)
    assert rl_cloud.lineage_seed_inputs({}, 'lin') == [seed, item]
    rl_cloud.add_runtime({}, 'lin', runtime, archive)  # idempotent per runtime path
    assert rl_cloud.lineage_seed_inputs({}, 'lin') == [seed, item]
