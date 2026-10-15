"""Readiness is exact trained-state evidence; the second model triggers work."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_readiness import ready_participants, run, validate_ready
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA


def ready(tmp_path, name):
    root = tmp_path/name; root.mkdir()
    def artifact(filename, value):
        path = root/filename
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': sha256_file_hex(path)}
    checkpoint = artifact('checkpoint', {'test': 'artifact-byte binding, not a model qualification'})
    checkpoint['schema'] = CHECKPOINT_SCHEMA
    identity = {'checkpoint_sha256': checkpoint['sha256'], 'tokenizer_fingerprint': '2'*64,
                'embedding_weight_sha256': '3'*64, 'architecture': name}
    recipe = artifact('recipe', {'native_stages': ['full', 'graph']})
    prior = artifact('prior-exposure', {'completed': True})
    report = artifact('report', {'completed': True, 'checkpoint_sha256': checkpoint['sha256']})
    runtime = artifact('runtime', {'schema': 'natlang.native-runtime-certificate/1', 'passed': True,
        'native_identity': identity, 'backbone_optimizer_tested': True, 'backbone_optimizer_updates': 20,
        'forward_atol': 1e-4, 'gradient_atol': 1e-5,
        'forward_max_abs': 0., 'incremental_max_abs': 0., 'consumer_ce_abs': 0., 'consumer_gradient_max_abs': 1e-6})
    task = artifact('tasks', {'schema': 'natlang.native-task-execution-receipt/1', 'passed': True,
        'native_identity': identity, 'complete': True, 'eligible': True, 'task_successes': 0})
    receipt = {'schema': 'natlang.native-training-readiness/1', 'model_id': name,
        'completed': True, 'stage_id': 'graph', 'native_identity': identity, 'checkpoint': checkpoint,
        'recipe': recipe, 'prerequisite_stages': [{'id': 'full', 'completed': True, 'exposure_receipt': prior}],
        'training_receipt': report, 'runtime_certificate': runtime, 'task_execution_receipt': task,
        'exposure': {'consumed_updates': 20, 'supervised_token_visits': 800}}
    path = root/'readiness.json'; path.write_text(json.dumps(receipt))
    requirement = {'model_id': name, 'readiness': str(path), 'recipe_sha256': recipe['sha256'],
                   'stage_ids': ['full', 'graph'], 'forward_atol': 1e-4, 'gradient_atol': 1e-5}
    return path, receipt, requirement


def test_second_ready_model_proceeds_without_waiting_for_all_models(tmp_path):
    a, _, first = ready(tmp_path, 'lfm')
    b, _, second = ready(tmp_path, 'mellum')
    third = {**first, 'model_id': 'later', 'readiness': str(tmp_path/'not-ready')}
    config = {'schema': 'natlang.native-shared-stage-controller/1', 'participants': [first, third, second],
              'command': ['python', 'actual-shared-stage.py', '--participants', '{participants}'],
              'out': str(tmp_path/'shared')}
    selected, pending = ready_participants(config)
    assert [item['model_id'] for item in selected] == ['lfm', 'mellum']
    assert len(pending) == 1
    path = tmp_path/'controller.json'; path.write_text(json.dumps(config))
    result = run(path, once=True)
    assert result['ready_models'] == 2 and result['child_pid'] is None
    assert len(json.loads(Path(result['participants']).read_text())['models']) == 2
    # The once inspection must never accidentally start a GPU process.




def test_changed_checkpoint_or_missing_stage_does_not_count_ready(tmp_path):
    path, receipt, requirement = ready(tmp_path, 'lfm')
    assert validate_ready(path, requirement)['model_id'] == 'lfm'
    with pytest.raises(ValueError, match='missing required'):
        validate_ready(path, {**requirement, 'stage_ids': ['full', 'graph', 'whole']})
    Path(receipt['checkpoint']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        validate_ready(path, requirement)


def test_inherited_recipe_readiness_binds_file_bytes_and_resolved_recipe(tmp_path):
    from natlang_neuralese.train.recipe import recipe_identity_sha256
    path, receipt, requirement = ready(tmp_path, 'lfm')
    recipe_dir = tmp_path/'recipes'; recipe_dir.mkdir()
    parent = recipe_dir/'base.json'
    child = recipe_dir/'line.json'
    parent.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1',
        'id':'base', 'stages':[], 'runtime':{'precision':'bf16'}, 'description':'base policy'}))
    child.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1',
        'id':'line', 'extends':'base', 'description':'line policy', 'stages':[]}))
    identity = recipe_identity_sha256(child)
    file_sha = sha256_file_hex(child)
    receipt['recipe'] = {'path':str(child), 'sha256':identity, 'file_sha256':file_sha}
    path.write_text(json.dumps(receipt))
    requirement['recipe_sha256'] = identity
    assert validate_ready(path, requirement)['model_id'] == 'lfm'
    parent.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1',
        'id':'base', 'stages':[], 'runtime':{'precision':'fp32'}, 'description':'changed inherited policy'}))
    with pytest.raises(ValueError, match='production recipe inheritance changed'):
        validate_ready(path, requirement)


def test_corrupt_third_model_is_a_repair_without_blocking_two_ready_models(tmp_path):
    _, _, first = ready(tmp_path, 'lfm')
    _, _, second = ready(tmp_path, 'mellum')
    _, receipt, third = ready(tmp_path, 'broken')
    Path(receipt['runtime_certificate']['path']).write_text('{}')
    selected, pending = ready_participants({'participants': [third, first, second]})
    assert len(selected) == 2 and pending[0]['repair_required'] is True
