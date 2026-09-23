import hashlib
import json

import pytest

from scripts.create_improvement_round import improvement_recipe
from scripts.create_training_pipeline import recipe


def fixture(tmp_path, image=None):
    base_recipe = tmp_path / 'base-recipe.json'
    config = recipe(tmp_path, image=image, python='python')
    base_recipe.write_text(json.dumps(config))
    base_run = tmp_path / 'base-run'
    checkpoint = base_run / 'train-teacher/checkpoint'
    (checkpoint / 'weights').mkdir(parents=True)
    (checkpoint / 'weights/adapter_model.safetensors').write_bytes(b'weights')
    (checkpoint / 'state.json').write_text(json.dumps({'trained_examples': 3,
                                                       'corpus': {'target_examples': 3}}))
    (base_run / 'prepared').mkdir()
    (base_run / 'prepared/splits.json').write_text(json.dumps({'groups': {}}))
    (base_run / 'training-readiness.json').write_text('{}')
    (base_run / 'pipeline-state.json').write_text(json.dumps({
        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'stages': {'train-teacher': {'status': 'complete'}}}))
    turns = tmp_path / 'corrections.jsonl'
    turns.write_text(json.dumps({'version': 'natlang.teacher_training_turn.native/1',
                                 'id': 'task:decision:0001', 'program_id': 'task',
                                 'source_groups': ['task'], 'source_ref': {'source_row_sha256': 'a' * 64},
                                 'messages': [{'role': 'system', 'content': 'test'}],
                                 'target': {'role': 'assistant', 'content': 'done'},
                                 'outcome': {'accepted': True},
                                 'training_admission': {'kind': 'exact-native-runtime-oracle', 'approved': True},
                                 'trace_admission': {'admitted': True}}) + '\n')
    return base_recipe, base_run, turns


def test_continuation_uses_frozen_splits_and_last_adapter(tmp_path):
    base_recipe, base_run, turns = fixture(tmp_path)
    result = improvement_recipe(base_recipe, base_run, turns)
    stages = {stage['id']: stage for stage in result['stages']}
    assert list(stages) == ['validate-corrections', 'prepare-correction', 'render-correction',
                            'audit-correction', 'train-correction']
    assert str(base_run / 'prepared/splits.json') in stages['prepare-correction']['inputs']
    assert str(turns) in stages['prepare-correction']['inputs']
    assert '--require-audit' in stages['train-correction']['command']
    assert stages['train-correction']['command'][
        stages['train-correction']['command'].index('--init-adapter') + 1] == str(base_run / 'train-teacher/checkpoint/weights')
    assert str(base_run / 'training-readiness.json') in stages['train-correction']['inputs']
    assert stages['train-correction']['training_state'] == '${run}/train-correction/checkpoint/state.json'
    assert all('${run}/train-teacher/' not in str(stage) for stage in stages.values())


def test_continuation_rejects_partial_or_unrelated_run(tmp_path):
    base_recipe, base_run, turns = fixture(tmp_path)
    state_path = base_run / 'pipeline-state.json'
    state = json.loads(state_path.read_text())
    state['stages']['train-teacher']['status'] = 'stopped'
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='must be complete'):
        improvement_recipe(base_recipe, base_run, turns)
    state['stages']['train-teacher']['status'] = 'complete'
    state['config_sha256'] = 'different'
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='does not match'):
        improvement_recipe(base_recipe, base_run, turns)


def test_docker_continuation_keeps_inputs_mounted(tmp_path):
    base_recipe, base_run, turns = fixture(tmp_path, image='natlang-train-kernels')
    result = improvement_recipe(base_recipe, base_run, turns)
    assert result['stages'][0]['command'][:3] == ['docker', 'run', '--rm']
    outside = tmp_path.parent / 'external-corrections.jsonl'
    outside.write_text(turns.read_text())
    try:
        with pytest.raises(ValueError, match='inside the repository'):
            improvement_recipe(base_recipe, base_run, outside)
    finally:
        outside.unlink()


def test_second_round_chains_previous_correction_and_original_splits(tmp_path):
    base_recipe, base_run, turns = fixture(tmp_path)
    first = improvement_recipe(base_recipe, base_run, turns)
    first_recipe = tmp_path / 'round-1-recipe.json'
    first_recipe.write_text(json.dumps(first))
    first_run = tmp_path / 'round-1'
    checkpoint = first_run / 'train-correction/checkpoint'
    (checkpoint / 'weights').mkdir(parents=True)
    (checkpoint / 'weights/adapter_model.safetensors').write_bytes(b'round-1')
    (checkpoint / 'state.json').write_text(json.dumps({'trained_examples': 2,
                                                       'corpus': {'target_examples': 2}}))
    (first_run / 'pipeline-state.json').write_text(json.dumps({
        'config_sha256': hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest(),
        'stages': {'train-correction': {'status': 'complete'}}}))
    second = improvement_recipe(first_recipe, first_run, turns)
    stages = {stage['id']: stage for stage in second['stages']}
    assert second['continuation']['split_registry'] == str(base_run / 'prepared/splits.json')
    assert stages['train-correction']['command'][
        stages['train-correction']['command'].index('--init-adapter') + 1] == str(checkpoint / 'weights')
    assert str(checkpoint / 'weights/adapter_model.safetensors') in stages['train-correction']['inputs']


@pytest.mark.parametrize('change', [
    {'outcome': {'accepted': False}},
    {'training_admission': {'kind': 'unverified', 'approved': True}},
    {'training_admission': {'kind': 'exact-native-runtime-oracle', 'approved': False}},
    {'trace_admission': {'admitted': False}},
])
def test_continuation_rejects_unverified_correction_turns(tmp_path, change):
    base_recipe, base_run, turns = fixture(tmp_path)
    row = json.loads(turns.read_text())
    row.update(change)
    turns.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='correction turn|no approved'):
        improvement_recipe(base_recipe, base_run, turns)
