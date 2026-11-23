import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('ops_watchdog', Path(__file__).resolve().parents[1] / 'scripts/ops_watchdog.py')
wd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wd)

NOW = 1_800_000_000.0
EXP = {'name': 'aei1', 'units': ['aei1-*'], 'gpus': ['dgx', 'colab-g4a'], 'drive_log': 'ei/aei1/drive.log',
       'stall_hours': 3, 'health_glob': 'ei/aei1/r*/health.json'}


class Machine:
    """Fake command outputs keyed by a substring of the joined command; records every command run."""

    def __init__(self, **outputs):
        self.outputs = {'list-units': (0, 'aei1-collect-v0000.service loaded active running x\n', '')}
        self.outputs.update(outputs)
        self.commands = []

    def __call__(self, command, timeout=60):
        self.commands.append(command)
        joined = ' '.join(command)
        for needle, result in self.outputs.items():
            if needle in joined:
                return result
        return 1, '', f'no fake for {joined}'


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, 'OPS', tmp_path)
    machine = Machine()
    monkeypatch.setattr(wd, 'run', machine)
    monkeypatch.setattr(wd, 'is_mount', lambda m: True)
    config = wd.load_config(tmp_path / 'absent.json')
    config['experiments'] = [EXP]
    return machine, config, tmp_path


def checks(config, state, now, tmp_path, only):
    return wd.run_checks(config, state, now, tmp_path / 'alerts.jsonl', set(only))


def test_unit_prefix_drops_ids_counters_and_versions():
    assert wd.unit_prefix('colab-g4a-fill-semantic-1795406085366328809-nores2.service') == 'colab-g4a-fill-semantic'
    assert wd.unit_prefix('aei1-drive-1795417829.service') == 'aei1-drive'
    assert wd.unit_prefix('b16-conf-10.service') == 'b16-conf'
    assert wd.unit_prefix('colab-g4a-tunnel.service') == 'colab-g4a-tunnel'


def test_disk_floor_and_fast_fall(env, monkeypatch):
    machine, config, tmp = env
    free = {'/': 14.0}
    monkeypatch.setattr(wd, 'disk_free_gb', lambda m: free.get(m, 500.0))
    state = {'disk': {'/': [[NOW - 3600, 30.0]]}}
    out = checks(config, state, NOW, tmp, ['disk'])
    keys = {a['key']: a for a in out}
    assert set(keys) == {'disk:floor:/', 'disk:fall:/'}
    assert keys['disk:fall:/']['detail']['rate_gb_per_hour'] == pytest.approx(16.0)
    assert len((tmp / 'alerts.jsonl').read_text().splitlines()) == 2


def test_dedup_realert_and_clear(env, monkeypatch):
    machine, config, tmp = env
    free = {'/': 10.0}
    monkeypatch.setattr(wd, 'disk_free_gb', lambda m: free.get(m, 500.0))
    state = {}
    assert [a['key'] for a in checks(config, state, NOW, tmp, ['disk'])] == ['disk:floor:/']
    assert checks(config, state, NOW + 300, tmp, ['disk']) == []            # deduplicated
    again = checks(config, state, NOW + 3700, tmp, ['disk'])                # re-alert interval passed
    assert [(a['key'], a['repeat']) for a in again] == [('disk:floor:/', 1)]
    free['/'] = 100.0
    checks(config, state, NOW + 3800, tmp, ['disk'])                        # cleared: key forgotten
    assert 'disk:floor:/' not in state['alerts']
    free['/'] = 10.0
    assert [a['key'] for a in checks(config, state, NOW + 3900, tmp, ['disk'])] == ['disk:floor:/']
    # a check that did not run keeps its keys
    checks(config, state, NOW + 4000, tmp, ['ledger'])
    assert 'disk:floor:/' in state['alerts']


def test_colab_session_not_gpu_stray_tunnel_and_idle_burn(env):
    machine, config, tmp = env
    machine.outputs.update({
        'sessions': (0, '[g4a] m-s-ass1c1-10rp | Hardware: CPU | Shape: Standard | Variant: DEFAULT\n'
                        '[?] gpu-a100-xyz | Hardware: A100 | Shape: Standard | Variant: GPU\n', ''),
        'show colab-g4a-tunnel': (0, 'NRestarts=50\nActiveState=activating\nSubState=auto-restart\n', ''),
        'usage': (0, 'Current balance: 234.31 compute units\nUsage rate: 2.02/hr\nActive assignments: 2\n', ''),
        'journalctl': (0, 'ssh: 429\n', '')})
    state = {'tunnels': {'colab-g4a-tunnel': {'restarts': 40, 'down': 0}}, 'gpu': {'colab-g4a': {'idle_s': 1800}}}
    keys = {a['key']: a for a in checks(config, state, NOW, tmp, ['colab'])}
    assert 'colab:not_gpu:g4a' in keys and keys['colab:not_gpu:g4a']['severity'] == 'critical'
    assert 'colab:stray:gpu-a100-xyz' in keys
    assert keys['colab:tunnel:colab-g4a-tunnel']['detail']['rose'] == 10
    assert 'colab:burn_idle' in keys


def test_colab_missing_session_and_quiet_when_healthy(env):
    machine, config, tmp = env
    machine.outputs.update({'sessions': (0, '', ''),
                            'show colab-g4a-tunnel': (0, 'NRestarts=3\nActiveState=active\n', ''),
                            'usage': (0, 'Current balance: 234\nUsage rate: 2.0/hr\n', '')})
    keys = [a['key'] for a in checks(config, {}, NOW, tmp, ['colab'])]
    assert keys == ['colab:missing:g4a']
    machine.outputs['sessions'] = (0, '[g4a] gpu-l4-s | Hardware: L4 | Shape: Standard | Variant: GPU\n', '')
    assert checks(config, {}, NOW, tmp, ['colab']) == []


def test_gpu_idle_alerts_only_when_an_experiment_expects_it_and_colab_uses_only_the_master(env):
    machine, config, tmp = env
    machine.outputs.update({'bash -c': (0, '0\n0\n1\n', ''),
                            'BatchMode=yes -o ConnectTimeout=10 pop': (0, '0\n0\n0\n', ''),
                            '-O check': (255, '', 'Control socket connect: No such file')})
    state = {'last_run': NOW - 300}
    assert checks(config, state, NOW, tmp, ['gpu']) == []                    # no history yet
    state['last_run'] = NOW + 900
    out = {a['key']: a for a in checks(config, state, NOW + 1500, tmp, ['gpu'])}
    assert set(out) == {'gpu:dgx', 'gpu:colab-g4a'}                          # Pop is expected by no experiment
    assert out['gpu:colab-g4a']['severity'] == 'critical' and 'ControlMaster' in out['gpu:colab-g4a']['message']
    colab = [c for c in machine.commands if 'colab-g4' in c]
    assert colab and all(c[:3] == ['ssh', '-O', 'check'] for c in colab)   # never a second Colab connection
    machine.outputs['bash -c'] = (0, '95\n', '')
    state['last_run'] = NOW + 1500
    checks(config, state, NOW + 1800, tmp, ['gpu'])
    assert state['gpu']['dgx']['last_busy'] == NOW + 1800 and 'gpu:dgx' not in state['alerts']


def test_colab_gpu_sample_goes_through_the_existing_master(env):
    machine, config, tmp = env
    machine.outputs.update({'-O check': (0, '', 'Master running'), 'ProxyCommand=/bin/false': (0, '87\n', '')})
    assert wd.gpu_sample({'kind': 'colab', 'host': 'colab-g4'}) == (87, None)
    assert ['ssh', '-o', 'ControlMaster=no', '-o', 'ProxyCommand=/bin/false'] == machine.commands[-1][:5]


def journal_line(ts, unit, result='exit-code', cursor='c'):
    return json.dumps({'__REALTIME_TIMESTAMP': str(int(ts * 1e6)), 'USER_UNIT': unit, 'UNIT_RESULT': result,
                       '__CURSOR': cursor})


def test_unit_failure_storm_and_experiment_failure(env):
    machine, config, tmp = env
    lines = [journal_line(NOW - 60 * i, f'colab-g4a-fill-semantic-17954{i}.service', cursor=f'c{i}') for i in range(4)]
    lines += [journal_line(NOW - 10, 'aei1-r0001-build.service', cursor='c9'),
              *[journal_line(NOW - 5 - i, 'snap.firmware-updater.firmware-notifier.service') for i in range(5)]]
    machine.outputs.update({'MESSAGE_ID': (0, '\n'.join(lines) + '\n', ''), '-n 4': (0, 'boom\n', '')})
    state = {}
    keys = {a['key']: a for a in checks(config, state, NOW, tmp, ['units'])}
    assert set(keys) == {'units:storm:colab-g4a-fill-semantic', 'units:failed:aei1-r0001-build.service'}
    assert keys['units:failed:aei1-r0001-build.service']['detail']['tail'] == ['boom']
    assert state['journal_cursor'] == 'c'
    assert any('--since' in c for c in machine.commands)
    machine.outputs['MESSAGE_ID'] = (0, '', '')
    assert checks(config, state, NOW + 300, tmp, ['units']) == []            # storm still in window: deduplicated
    assert any(c[-2:] == ['--after-cursor', 'c'] for c in machine.commands)
    checks(config, state, NOW + 3600, tmp, ['units'])                       # window passed: storm cleared
    assert state['failures'] == [] and not any(k.startswith('units:storm') for k in state['alerts'])


def test_ledger_headroom_and_stuck_queue_head(env):
    machine, config, tmp = env
    status = {'headroom_gb': 10.0, 'claims': {'aei1-collect-v0000.service': {'budget_gb': 74.0}},
              'waiting': [{'unit': 'heldv4-measure.service', 'budget_gb': 32.0, 'waiting_s': 5400,
                           'blocked_by': [{'unit': 'aei1-collect-v0000.service'}]},
                          {'unit': 'other.service', 'waiting_s': 9000}]}
    machine.outputs['memory_ledger.py status'] = (0, json.dumps(status), '')
    keys = [a['key'] for a in checks(config, {}, NOW, tmp, ['ledger'])]
    assert keys == ['ledger:headroom', 'ledger:queue:heldv4-measure.service']
    status.update(headroom_gb=20.0)
    status['waiting'][0]['waiting_s'] = 600
    machine.outputs['memory_ledger.py status'] = (0, json.dumps(status), '')
    assert checks(config, {}, NOW, tmp, ['ledger']) == []


def test_driver_stall_only_while_units_are_active(env):
    machine, config, tmp = env
    log = tmp / 'ei/aei1/drive.log'
    log.parent.mkdir(parents=True)
    stamp = wd.datetime.fromtimestamp(NOW - 4 * 3600).strftime('%Y-%m-%dT%H:%M:%S')
    log.write_text(json.dumps({'time': stamp, 'stage': 'async'}) + '\nnot json\n')
    assert [a['key'] for a in checks(config, {}, NOW, tmp, ['driver'])] == ['driver:aei1']
    fresh = wd.datetime.fromtimestamp(NOW - 600).strftime('%Y-%m-%dT%H:%M:%S')
    log.write_text(log.read_text() + json.dumps({'time': fresh, 'stage': 'async'}) + '\n')
    assert checks(config, {}, NOW, tmp, ['driver']) == []
    machine.outputs['list-units'] = (0, '', '')
    log.write_text(json.dumps({'time': stamp}) + '\n')
    assert checks(config, {}, NOW, tmp, ['driver']) == []


def test_health_flags_once_per_report_and_new_log_lines(env):
    machine, config, tmp = env
    health = tmp / 'ei/aei1/r0001/health.json'
    health.parent.mkdir(parents=True)
    health.write_text(json.dumps({'time': 1.0, 'flags': [{'flag': 'construct_collapse', 'construct': 'memetic',
                                                          'scored': 40, 'sfp': True}, {'flag': 'no_episodes'}]}))
    alerts_log = tmp / 'storage-alerts.log'
    alerts_log.write_text('old COLLECTION HEALTH line\n')
    state = {}
    out = checks(config, state, NOW, tmp, ['health'])
    assert len(out) == 2 and 'construct_collapse memetic [sfp]' in out[0]['message']
    assert checks(config, state, NOW + 7200, tmp, ['health']) == []          # same report: never again
    alerts_log.write_text(alerts_log.read_text() + 'x DISK FLOOR /\nx COLLECTION HEALTH aei1 r0002: infrastructure\npartial COLLECTION HEALTH')
    out = checks(config, state, NOW + 7300, tmp, ['health'])
    assert [a['message'] for a in out] == ['x COLLECTION HEALTH aei1 r0002: infrastructure']
    health.write_text(json.dumps({'time': 2.0, 'flags': [{'flag': 'no_episodes'}]}))
    assert len(checks(config, state, NOW + 7400, tmp, ['health'])) == 1     # a new report alerts again


def test_a_failing_check_is_an_alert_and_the_others_run(env, monkeypatch):
    machine, config, tmp = env
    monkeypatch.setattr(wd, 'disk_free_gb', lambda m: 1 / 0)
    machine.outputs['memory_ledger.py status'] = (0, json.dumps({'headroom_gb': 1.0}), '')
    keys = [a['key'] for a in checks(config, {}, NOW, tmp, ['disk', 'ledger'])]
    assert keys == ['watchdog:error:disk', 'ledger:headroom']


def test_wait_is_gapless_skips_partial_lines_and_reports_a_stale_watchdog(tmp_path):
    alerts = tmp_path / 'alerts.jsonl'
    a1 = {'time': '2026-11-23T13:00:00+01:00', 'ts': NOW - 7200, 'severity': 'warning', 'key': 'disk:fall:/',
          'message': 'old'}
    a2 = dict(a1, ts=NOW - 60, message='recent')
    alerts.write_text(json.dumps(a1) + '\n' + json.dumps(a2) + '\n')
    state = {'cursor': wd.start_cursor(alerts, NOW - 3600)}
    lines = wd.wait_once(alerts, state, {'last_run': NOW - 60}, NOW, 15)
    assert len(lines) == 1 and lines[0].endswith('disk:fall:/: recent')
    assert wd.wait_once(alerts, state, {'last_run': NOW - 60}, NOW, 15) == []
    with open(alerts, 'a') as handle:
        handle.write(json.dumps(dict(a1, message='third')) + '\n' + '{"half": ')
    assert [line.split(': ', 1)[1] for line in wd.wait_once(alerts, state, {'last_run': NOW}, NOW, 15)] == ['third']
    with open(alerts, 'a') as handle:
        handle.write('1}\n')
    assert len(wd.wait_once(alerts, state, {'last_run': NOW}, NOW, 15)) == 1   # completed later, read once
    stale = wd.wait_once(alerts, state, {'last_run': NOW - 3600}, NOW, 15)
    assert any('watchdog:stale' in line for line in stale)
    assert wd.wait_once(alerts, state, {'last_run': NOW - 3600}, NOW + 60, 15) == []   # reported once per stale run


def test_wait_command_prints_and_exits(tmp_path, monkeypatch, capsys):
    alerts = tmp_path / 'alerts.jsonl'
    alerts.write_text(json.dumps({'time': '2026-11-23T13:00', 'ts': 9e12, 'severity': 'critical', 'key': 'k',
                                  'message': 'm'}) + '\n')
    monkeypatch.setattr(wd, 'STATE', tmp_path / 'wd-state.json')
    (tmp_path / 'wd-state.json').write_text(json.dumps({'last_run': 9e12}))
    wd.main(['--alerts', str(alerts), 'wait', '--state', str(tmp_path / 'cur.json'), '--interval', '0.01'])
    assert 'CRITICAL k: m' in capsys.readouterr().out
    wd.main(['--alerts', str(alerts), 'wait', '--state', str(tmp_path / 'cur.json'), '--interval', '0.01',
             '--timeout', '0.05'])
    assert capsys.readouterr().out.strip() == 'quiet'
