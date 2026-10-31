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
