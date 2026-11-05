import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('sft_window', Path(__file__).parents[1] / 'scripts/neuralese_data/sft_window.py')
sft_window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft_window)


def _runtime(root, commit):
    root.mkdir()
    (root / 'runtime-input-closure.json').write_text(json.dumps({'commit': commit}))
    return root


def test_rebind_runtime_moves_controller_actor_and_environment_and_records_both_closures(tmp_path):
    old, new = _runtime(tmp_path / 'v44', 'a' * 40), _runtime(tmp_path / 'v45', 'b' * 40)
    c = {'runtime': f'{old}/ts-host',
         'actor_command': ['/venv/python', f'{old}/scripts/neuralese_data/serve_native_checkpoint.py', '--port', '8200'],
         'actor_environment': {'NATLANG_REPO': str(old), 'PYTHONPATH': f'{old}/training/neuralese:{old}', 'X': '1'},
         'seed_command': ['/venv/python', '-m', 'natlang_neuralese.train.native_state_sft'],
         'learner_command': ['/venv/python', '-m', 'natlang_neuralese.rl.native_stage']}
    record = sft_window.rebind_runtime(c, old, new)
    assert c['runtime'] == f'{new}/ts-host'
    assert c['actor_command'][1] == f'{new}/scripts/neuralese_data/serve_native_checkpoint.py'
    assert c['actor_environment'] == {'NATLANG_REPO': str(new), 'PYTHONPATH': f'{new}/training/neuralese:{new}', 'X': '1'}
    assert record['from']['commit'] == 'a' * 40 and record['to']['commit'] == 'b' * 40
    assert str(old) not in json.dumps(c)


def test_rebind_runtime_refuses_learner_commands_naming_the_old_runtime(tmp_path):
    old, new = _runtime(tmp_path / 'v44', 'a' * 40), _runtime(tmp_path / 'v45', 'b' * 40)
    c = {'runtime': f'{old}/ts-host', 'learner_command': ['/venv/python', f'{old}/wrapper.py']}
    with pytest.raises(SystemExit):
        sft_window.rebind_runtime(c, old, new)
    with pytest.raises(SystemExit):
        sft_window.rebind_runtime({'runtime': f'{old}/ts-host'}, old, tmp_path / 'not-frozen')


def _memory_request(**kw):
    return {'schema': sft_window.MEMORY_REQUEST_SCHEMA, **kw}


def test_memory_request_turns_offload_off_and_records_each_change():
    c = {'trainer_parameters': {'checkpoint_input_offload': True, 'checkpoint_input_offload_directory': '/scratch',
                                'chunk_tokens': 1024, 'ffn_chunk_tokens': 1024, 'learning_rate': 1e-5}}
    record = sft_window.apply_memory_parameters(
        c, _memory_request(set={'checkpoint_input_offload': False, 'chunk_tokens': 4096, 'ffn_chunk_tokens': 4096}),
        {'path': 'req', 'sha256': 'x'})
    assert c['trainer_parameters'] == {'checkpoint_input_offload': False, 'chunk_tokens': 4096,
                                       'ffn_chunk_tokens': 4096, 'learning_rate': 1e-5}
    assert record['changed'] == {'checkpoint_input_offload': [True, False], 'chunk_tokens': [1024, 4096],
                                 'ffn_chunk_tokens': [1024, 4096], 'checkpoint_input_offload_directory': ['/scratch', None]}
    assert record['math_changed'] is False


def test_memory_request_refuses_other_keys_and_offload_without_directory():
    c = {'trainer_parameters': {'checkpoint_input_offload': False, 'learning_rate': 1e-5}}
    with pytest.raises(SystemExit):
        sft_window.apply_memory_parameters(c, _memory_request(set={'learning_rate': 2e-5}), {})
    with pytest.raises(SystemExit):
        sft_window.apply_memory_parameters(c, _memory_request(set={'checkpoint_input_offload': True}), {})
    with pytest.raises(SystemExit):
        sft_window.apply_memory_parameters(c, {'set': {'chunk_tokens': 4096}}, {})


def test_composition_request_is_recorded_separately_from_memory_changes():
    c = {'trainer_parameters': {'batch_tokens': 16384, 'accumulate_batches': 16, 'chunk_tokens': 1024}}
    record = sft_window.apply_memory_parameters(
        c, _memory_request(set={'batch_tokens': 65536, 'accumulate_batches': 4, 'chunk_tokens': 4096}), {})
    assert c['trainer_parameters'] == {'batch_tokens': 65536, 'accumulate_batches': 4, 'chunk_tokens': 4096}
    assert record['composition_changed'] == ['accumulate_batches', 'batch_tokens']
    with pytest.raises(SystemExit):
        sft_window.apply_memory_parameters(c, _memory_request(remove=['batch_tokens']), {})
    with pytest.raises(SystemExit):
        sft_window.apply_memory_parameters(c, _memory_request(set={'accumulate_batches': 0}), {})


def test_next_cli_accepts_an_explicit_trainer_parameter_request(monkeypatch):
    seen = {}
    monkeypatch.setattr(sft_window, 'next_window', lambda *a, **k: seen.update(k))
    sft_window.main(['next', '--previous-config', 'p', '--out', 'o', '--config-out', 'c',
                     '--trainer-parameter-request', '/r.json'])
    assert seen['parameter_request'] == '/r.json'


def _focus(tmp_path, name, *, supersedes=None):
    import hashlib
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}')
    reference = {'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}
    doc = {'schema': 'natlang.native-pending-focus/1',
           'rows': [{'id': 'r0', 'binding': 'a' * 64, 'source_sha256': 'b' * 64, 'split': 'train', 'evidence': [reference]}]}
    if supersedes:
        doc['supersedes'] = {'request_sha256': supersedes, 'reason': 'core focus'}
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_pending_focus_adoption_rebinds_the_stage_input_and_names_the_active_request(tmp_path):
    old = 'c' * 64
    history = [{'request': {'path': '/old/focus.json', 'sha256': old}, 'state': 'active'}]
    recipe = {'input_bindings': {'old_focus': {'path': '/old/focus.json', 'sha256': old}}}
    stage = {'inputs': {'pending_focus': 'old_focus'}}
    path, sha = _focus(tmp_path, 'core.json', supersedes=old)
    record = sft_window.adopt_pending_focus(recipe, stage, path, {'pending_focus_history': history})
    assert recipe['input_bindings']['old_focus'] == {'path': str(path), 'sha256': sha}
    assert stage['inputs']['pending_focus'] == 'old_focus'
    assert record['supersedes'] == old and record['previous']['sha256'] == old and record['rows'] == 1
    # the request the checkpoint already runs is a no-op
    assert sft_window.adopt_pending_focus(recipe, stage, path, {'pending_focus_history': [
        {'request': {'path': str(path), 'sha256': sha}, 'state': 'active'}]}) is None


def test_pending_focus_adoption_refuses_a_supersession_that_the_trainer_would_reject(tmp_path):
    active = [{'request': {'path': '/old/focus.json', 'sha256': 'c' * 64}, 'state': 'active'}]
    finished = [{'request': {'path': '/old/focus.json', 'sha256': 'c' * 64}, 'state': 'complete'}]
    recipe, stage = {'input_bindings': {}}, {'inputs': {}}
    plain, _ = _focus(tmp_path, 'plain.json')
    wrong, _ = _focus(tmp_path, 'wrong.json', supersedes='d' * 64)
    named, _ = _focus(tmp_path, 'named.json', supersedes='c' * 64)
    with pytest.raises(SystemExit, match='must declare supersedes'):
        sft_window.adopt_pending_focus(recipe, stage, plain, {'pending_focus_history': active})
    with pytest.raises(SystemExit, match='must declare supersedes'):
        sft_window.adopt_pending_focus(recipe, stage, wrong, {'pending_focus_history': active})
    with pytest.raises(SystemExit, match='no active focus'):
        sft_window.adopt_pending_focus(recipe, stage, named, {'pending_focus_history': finished})
    assert sft_window.adopt_pending_focus(recipe, stage, plain, {'pending_focus_history': finished})['supersedes'] is None
    assert stage['inputs']['pending_focus'] == 'pending_focus_request'
