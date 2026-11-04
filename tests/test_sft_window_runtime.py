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
