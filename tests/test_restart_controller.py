"""The restart script's pure parts: launch-spec derivation from a running unit, unit naming, ledger command."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location('restart_controller', Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/restart_controller.py')
rc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc)

CAT = '''# /run/user/1000/systemd/transient/bf95-lowlr-w3.service
[Service]
ExecStart=
ExecStart=:"/bin/sh" "-c" "cd /d/runtime-native-corrected-mix-v57-1106 && exec env A=1 python -u -m x run --config /o/w3/config.json >> /o/w3/controller.log 2>&1"
'''


def test_launch_spec_is_derived_from_the_running_unit_and_relaunched_under_a_fresh_name(monkeypatch):
    def systemctl(*args):
        if args[0] == 'cat':
            return SimpleNamespace(stdout=CAT, returncode=0)
        return SimpleNamespace(stdout='MemoryMax=112527409561\nWorkingDirectory=/d/runtime-native-corrected-mix-v57-1106\n', returncode=0)
    monkeypatch.setattr(rc, 'systemctl', systemctl)
    monkeypatch.setattr(rc, 'run', lambda argv, **kw: SimpleNamespace(returncode=0, stdout='{"claims": {"bf95-lowlr-w3.service": '
                                                                    '{"priority": "high", "fixed_claim": true, "budget_gb": 100}}}'))
    spec = rc.derive_launch('bf95-lowlr-w3')
    assert spec['budget_gb'] == 104.8 and spec['priority'] == 'high' and spec['fixed_claim'] is True
    assert spec['payload'].startswith('cd /d/runtime-native-corrected-mix-v57-1106 && exec env')
    assert rc.current_runtime(spec) == 'v57-1106'
    command = rc.ledger_command(spec, 'bf95-lowlr-w3-r1')
    assert command[command.index('--unit') + 1] == 'bf95-lowlr-w3-r1' and '--fixed-claim' in command
    assert command[-3:-1] == ['/bin/sh', '-c'] and command[-1] == spec['payload']


def test_unit_names_count_up_from_every_known_restart(monkeypatch):
    monkeypatch.setattr(rc, 'units_named', lambda prefix: ['bf95-lowlr-w3', 'bf95-lowlr-w3-r2'])
    assert rc.base_unit('bf95-lowlr-w3-r7') == 'bf95-lowlr-w3'
    assert rc.next_unit('bf95-lowlr-w3-r2', ['bf95-lowlr-w3-r4']) == 'bf95-lowlr-w3-r5'
    assert rc.next_unit('bf95-lowlr-w3', []) == 'bf95-lowlr-w3-r3'
