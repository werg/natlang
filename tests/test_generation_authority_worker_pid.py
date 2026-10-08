import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from generation_authority import observe_additional_teacher_processes, verify_worker_process
from monitor_generation import additional_teacher_alerts


def test_worker_process_resolves_successor_pid_aliases_against_exact_argv():
    expected = ['/usr/bin/timeout', '--signal=TERM', '1800s', '/usr/bin/node',
                '/opt/runtime/cli.js', '/data/source.jsonl']
    worker = {'worker_pid': 321, 'process_pid': 322, 'argv': expected}
    live = {321: expected, 322: expected[3:]}

    assert verify_worker_process(worker, process_argv=live.get) == {
        'pid': 322, 'field': 'process_pid', 'argv': expected[3:],
    }


def test_pid_reuse_or_missing_argv_does_not_mark_worker_live():
    worker = {'worker_pid': 321, 'argv': ['/usr/bin/timeout', '1800s', '/usr/bin/node', '/expected/cli.js']}
    assert verify_worker_process(worker, process_argv=lambda _pid: ['/usr/bin/python', 'unrelated.py']) is None
    assert verify_worker_process({'worker_pid': 321}, process_argv=lambda _pid: ['/usr/bin/node']) is None


def test_additional_teacher_observation_keeps_lifecycle_status_and_counts_verified_workers():
    expected = ['/usr/bin/timeout', '--signal=TERM', '1800s', '/usr/bin/node', '/runtime/cli.js']
    teacher = {'status': 'running', 'workers': [
        {'worker_pid': 41, 'argv': expected, 'status': 'running'},
        {'process_pid': 42, 'argv': expected, 'status': 'running'},
        {'pid': 43, 'argv': expected, 'status': 'completed'},
    ]}
    live = {41: expected, 42: ['/usr/bin/node', '/wrong/cli.js']}

    observed = observe_additional_teacher_processes(
        teacher, observed_at='fixed-time', process_argv=live.get)

    assert observed == {'actual_live_workers': 1, 'active_processes_unverified': 1}
    assert teacher['workers'][0]['observed_process_pid_field'] == 'worker_pid'
    assert teacher['workers'][1]['process_running'] is False
    assert teacher['workers'][2]['status'] == 'completed'
    assert teacher['status'] == 'running'


def test_monitor_warns_when_running_additional_worker_has_unverified_pid_binding():
    authority = {'additional_teachers': {'bunny': {
        'state': 'running',
        'workers': [{'worker_pid': 51, 'status': 'running',
                     'argv': ['/usr/bin/timeout', '1800s', '/usr/bin/node', '/expected/cli.js']}],
    }}}

    alerts = additional_teacher_alerts(authority, process_argv=lambda _pid: ['/usr/bin/python', 'unrelated.py'])

    assert any('bunny: 1 active worker binding(s) have no verified live PID/argv' in alert
               for alert in alerts)


def test_monitor_detects_idle_lifecycle_with_verified_live_child():
    authority = {'additional_teachers': {'bunny': {
        'state': 'completed', 'status': 'idle',
        'workers': [{'process_pid': 61, 'status': 'running',
                     'argv': ['/usr/bin/node', '/runtime/cli.js']}],
    }}}

    alerts = additional_teacher_alerts(authority, process_argv=lambda _pid: ['/usr/bin/node', '/runtime/cli.js'])

    assert any('bunny: lifecycle state' in alert and 'verified live worker processes' in alert
               for alert in alerts)
