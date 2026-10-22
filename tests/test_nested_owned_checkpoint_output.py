"""An exact nested recovery keeps the first role-owned control namespace."""
import hashlib
import json

import pytest

from scripts.coreweave.cursor_checkpoint_export import bound_output_checkpoint


def reference(tmp_path, output, allowed):
    path = tmp_path / 'effective-argv.json'
    raw = json.dumps(['python', '-m', 'trainer', '--out', str(output)]).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'allowed_output_root': str(allowed)}


@pytest.mark.parametrize('role', ['lfm', 'mellum'])
def test_actual_nested_recovery_output_is_bound_to_its_role(tmp_path, role):
    control = tmp_path / 'control'
    recovery = control / (role + '-shared-current-production') / 'oom-recovery-v8' / 'operation'
    output = recovery / 'learner'
    ref = reference(tmp_path, output, recovery)
    assert bound_output_checkpoint(ref, role, production_root=str(tmp_path), control_root=str(control)) == output / 'checkpoint.pt'


@pytest.mark.parametrize('role', ['lfm', 'mellum'])
@pytest.mark.parametrize('fault', ['other-role', 'unscoped', 'symlink', 'argv-escape', 'changed-argv'])
def test_nested_recovery_does_not_borrow_or_escape_namespace(tmp_path, role, fault):
    control = tmp_path / 'control';control.mkdir()
    other = 'mellum' if role == 'lfm' else 'lfm'
    owner = control / (role + '-owner');owner.mkdir()
    allowed = owner / 'recovery' / 'operation';output = allowed / 'learner'
    if fault == 'other-role':allowed = control / (other + '-owner') / 'nested';output = allowed / 'learner'
    elif fault == 'unscoped':allowed = control / 'unscoped' / (role + '-owner');output = allowed / 'learner'
    elif fault == 'symlink':
        foreign = control / (other + '-owner');foreign.mkdir();(owner / 'recovery').symlink_to(foreign, target_is_directory=True)
    elif fault == 'argv-escape':output = owner / 'another-operation' / 'learner'
    ref = reference(tmp_path, output, allowed)
    if fault == 'changed-argv':(tmp_path / 'effective-argv.json').write_text('[]')
    with pytest.raises(ValueError):bound_output_checkpoint(ref, role, production_root=str(tmp_path), control_root=str(control))
