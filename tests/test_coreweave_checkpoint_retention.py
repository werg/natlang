import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'checkpoint_retention', Path(__file__).resolve().parents[1] / 'scripts/coreweave/checkpoint_retention.py')
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)


def setup_export(tmp_path, role):
    name = lambda step: f'{role}/{"a" * 64}/{role}-native-checkpoint-step-{step}.pt'
    state = {'last_verified_step': 3, 'receipt': {'private': True, 'run': 'owner/project/run',
             'identity': {'sha256': 'a' * 64, 'bytes': 40},
             'independent_verification': {'sha256': 'a' * 64, 'bytes': 40}, 'name': name(3)}}
    files = {n: SimpleNamespace(name=n, size=40, md5='md5') for n in
             [name(1), name(2), name(3), name(4), f'{role}/source.tar', f'{role}/inference.pt', 'metadata.json']}
    same_step = f'{role}/{"b" * 64}/{role}-native-checkpoint-step-3.pt'
    files[same_step] = SimpleNamespace(name=same_step, size=40, md5='other')
    deleted = []
    for n, f in files.items():
        def remove(n=n):
            assert list((tmp_path / 'journal').glob('*-intent-*.json'))
            deleted.append(n)
            del files[n]
        f.delete = remove
    run = SimpleNamespace(files=lambda **kwargs: list(files.values()))
    api = SimpleNamespace(run=lambda path: run)
    state_path = tmp_path / 'state.json'
    state_path.write_text(json.dumps(state))
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text(json.dumps({'models': {role: {'run': 'owner/project/run',
        'state': str(state_path), 'protected_files': [name(2)]}}}))
    return name, state, files, deleted, api, state_path, policy_path


@pytest.mark.parametrize('role', ['lfm', 'mellum'])
def test_verified_replacement_prunes_only_superseded_unprotected_full_state(tmp_path, role):
    name, state, files, deleted, api, _, policy = setup_export(tmp_path, role)
    invoke = lambda **kwargs: retention.tick(policy, tmp_path / 'journal', api=api,
                                            ownership_directory=tmp_path / 'owners', **kwargs)
    assert invoke()[0]['candidates'] == 1
    assert not deleted
    intent = json.loads(next((tmp_path / 'journal').glob('*-intent-*.json')).read_text())
    assert intent['verified_export_state'] == state
    assert intent['protected_files'] == [name(2)]
    assert intent['policy']['path'] == str(policy)
    invoke(apply=True)
    assert deleted == [name(1)]
    assert {name(2), name(3), name(4), f'{role}/source.tar', f'{role}/inference.pt', 'metadata.json',
            f'{role}/{"b" * 64}/{role}-native-checkpoint-step-3.pt'} == set(files)
    # A resumed sweep does not delete again or reinterpret the live latest.
    assert invoke(apply=True)[0]['candidates'] == 0


@pytest.mark.parametrize('damage', ['readback', 'private', 'step', 'missing', 'size', 'run'])
def test_invalid_recovery_authority_never_deletes(tmp_path, damage):
    name, state, files, deleted, api, state_path, policy = setup_export(tmp_path, 'mellum')
    if damage == 'readback': state['receipt']['independent_verification']['sha256'] = 'b' * 64
    if damage == 'private': state['receipt']['private'] = False
    if damage == 'step': state['last_verified_step'] = 4
    if damage == 'missing': del files[name(3)]
    if damage == 'size': files[name(3)].size += 1
    if damage == 'run': state['receipt']['run'] = 'owner/other/run'
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        retention.tick(policy, tmp_path / 'journal', apply=True, api=api, ownership_directory=tmp_path / 'owners')
    assert not deleted


def test_conflicting_policy_cannot_take_over_same_run_through_another_journal(tmp_path):
    name, _, _, deleted, api, _, policy = setup_export(tmp_path, 'mellum')
    owners = tmp_path / 'owners'
    retention.tick(policy, tmp_path / 'journal', apply=True, api=api, ownership_directory=owners)
    other = tmp_path / 'other-policy.json'
    data = json.loads(policy.read_text())
    data['models']['mellum']['protected_files'] = []
    other.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='another explicit retention policy'):
        retention.tick(other, tmp_path / 'another-journal', apply=True, api=api, ownership_directory=owners)
    assert deleted == [name(1)]
