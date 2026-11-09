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


# ---- run from the checkout (owner, 2026-11-09) ---------------------------------------------------------------------

import json

FROZEN = f'{rc.DATA}/runtime-native-corrected-mix-v71-1109'


def w5_like_config(tmp_path):
    wrapper = tmp_path / 'control-w5' / 'admit-learner.py'
    wrapper.parent.mkdir()
    wrapper.write_text(f"import os\nruntime=Path('{FROZEN}')\nprint('x')\n")
    return {
        'actor_command': ['/py', f'{FROZEN}/scripts/neuralese_data/serve_native_checkpoint.py', '--port', '8200'],
        'actor_environment': {'NATLANG_REPO': FROZEN, 'PYTHONPATH': f'{FROZEN}/training/neuralese:{FROZEN}'},
        'native_actor_python_runtime': {'schema': 'natlang.python-runtime/1', 'sha256': 'a' * 64, 'files': 403},
        'native_actor_assets': {'training/prompt-pieces/pieces.json': 'b' * 64},
        'runtime': f'{FROZEN}/ts-host',
        'branch_origin': {'runtime': FROZEN},
        'trainer_parameters': {'task_eval': {'runtime': f'{rc.DATA}/runtime-native-corrected-mix-v44-1103/ts-host'}},
        'learner_command': ['/py', str(wrapper), '--', '/py', '-m', 'natlang_neuralese.rl.native_stage'],
        'after_actor_ready': ['/py', '/natlang/scripts/neuralese_data/pop_rl_actor.py', 'start', '--policy-id', '{policy_id}',
                              '--runtime', FROZEN, '--defer-to-eval'],
        'workers': 4}


def test_a_checkout_launch_has_no_frozen_runtime_and_keeps_the_checkout_workdir():
    spec = {'payload': f'cd {rc.REPO} && exec env NATLANG_REPO={rc.REPO} python -m natlang_neuralese.rl.production run',
            'workdir': str(rc.REPO), 'budget_gb': 100, 'priority': 'high', 'fixed_claim': True}
    assert rc.current_runtime(spec) is None
    command = rc.ledger_command(spec, 'bf95-lowlr-w5-r5')
    assert command[command.index('--workdir') + 1] == str(rc.REPO)


def test_migration_moves_code_paths_to_the_checkout_and_keeps_provenance_and_the_dev_eval_runtime(tmp_path):
    config = w5_like_config(tmp_path)
    new, changed = rc.migrate_config(config)
    checkout = str(rc.REPO)
    assert new['actor_command'][1] == f'{checkout}/scripts/neuralese_data/serve_native_checkpoint.py'
    assert new['actor_environment'] == {'NATLANG_REPO': checkout, 'PYTHONPATH': f'{checkout}/training/neuralese:{checkout}'}
    assert new['runtime'] == f'{checkout}/ts-host'
    assert 'native_actor_python_runtime' not in new and 'native_actor_assets' not in new
    assert new['after_actor_ready'] == ['/py', '/natlang/scripts/neuralese_data/pop_rl_actor.py', 'start', '--policy-id',
                                        '{policy_id}', '--defer-to-eval']
    assert new['branch_origin'] == config['branch_origin']                      # branch provenance as recorded
    assert new['trainer_parameters'] == config['trainer_parameters']            # hash-bound dev-eval runtime untouched
    assert new['workers'] == 4 and new['learner_command'] == config['learner_command']
    assert changed == ['actor_command', 'actor_environment', 'after_actor_ready', 'native_actor_assets',
                       'native_actor_python_runtime', 'runtime']
    assert rc.migrate_config(new)[1] == []                                      # idempotent


def test_migration_apply_writes_backups_a_state_record_and_restore_undoes_it(tmp_path, monkeypatch):
    out = tmp_path / 'w5'
    out.mkdir()
    config = w5_like_config(tmp_path)
    wrapper = Path(config['learner_command'][1])
    env_sh = tmp_path / 'env.sh'
    env_sh.write_text(f'V={FROZEN}\n')
    monkeypatch.setattr(rc, 'OPS', tmp_path)
    (tmp_path / 'interleave-v49').mkdir()
    (tmp_path / 'interleave-v49/env.sh').write_text(f'V={FROZEN}\n')
    rc.save_json(out / 'config.json', config)
    rc.save_json(out / 'state.json', {'phase': 'collect', 'round': 3})
    spec = {'unit': 'u', 'payload': f'cd {FROZEN} && exec env NATLANG_REPO={FROZEN} PYTHONPATH={FROZEN}/training/neuralese:{FROZEN} py',
            'workdir': FROZEN, 'budget_gb': 100, 'priority': 'high', 'fixed_claim': True}
    rc.save_json(out / 'launch.json', spec)
    original = {p: p.read_bytes() for p in (out / 'config.json', out / 'state.json', out / 'launch.json', wrapper,
                                            tmp_path / 'interleave-v49/env.sh')}
    plan, _, launch = rc.plan_migration(out, spec)
    assert all(p.read_bytes() == b for p, b in original.items())                 # planning writes nothing
    assert launch['payload'].startswith(f'cd {rc.REPO} && exec env NATLANG_REPO={rc.REPO} PYTHONPATH={rc.REPO}/training')
    restore, launch = rc.apply_migration(out, spec, '20261109-000000')
    assert FROZEN not in wrapper.read_text() and str(rc.REPO) in wrapper.read_text()
    assert FROZEN not in (tmp_path / 'interleave-v49/env.sh').read_text()
    state = json.loads((out / 'state.json').read_text())
    record = state['code_transitions'][-1]
    assert record['schema'] == 'natlang.rl-code-checkout-transition/1' and record['to']['checkout'] == str(rc.REPO)
    assert record['from']['runtime'] == FROZEN and state['round'] == 3
    assert json.loads((out / 'launch.json').read_text())['workdir'] == str(rc.REPO)
    rc.restore_files(restore)
    assert all(p.read_bytes() == b for p, b in original.items())


def test_dry_run_to_checkout_prints_the_plan_and_changes_nothing(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'w5'
    out.mkdir()
    config = w5_like_config(tmp_path)
    rc.save_json(out / 'config.json', config)
    spec = {'unit': 'bf95-lowlr-w5', 'payload': f'cd {FROZEN} && exec env NATLANG_REPO={FROZEN} py', 'workdir': FROZEN,
            'budget_gb': 100, 'priority': 'high', 'fixed_claim': True}
    rc.save_json(out / 'launch.json', spec)
    monkeypatch.setattr(rc, 'pick_unit', lambda o, e=None: 'bf95-lowlr-w5')
    monkeypatch.setattr(rc, 'units_named', lambda prefix: ['bf95-lowlr-w5'])
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert rc.main([str(out), '--to-checkout', '--dry-run']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['to'] == 'checkout' and report['next_unit'] == 'bf95-lowlr-w5-r1'
    assert report['migration']['actor_environment']['NATLANG_REPO'] == str(rc.REPO)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
