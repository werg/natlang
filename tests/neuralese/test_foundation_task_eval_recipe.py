import copy
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.train.foundation_task_eval_recipe import prepare_foundation_boundary_recipe


def _recipe(path, task_eval=None, cadence=0, *, foundation=True):
    value = {'schema': 'natlang.neuralese-training-recipe/1', 'id': path.stem,
        'input_bindings': {'native_train_graphs': {'path': '/train', 'sha256': 'a' * 64},
                           'native_held_graphs': {'path': '/held', 'sha256': 'b' * 64}},
        'stages': [{'id': 'native_full_sft', 'kind': 'native_state_sft', 'requires': [],
            'inputs': {'native_graphs': 'native_train_graphs', 'held_native_graphs': 'native_held_graphs'},
            'parameters': {'communication_foundation': foundation, 'task_eval': task_eval or '',
                           'task_eval_every': cadence, 'epochs': 1, 'learning_rate': 1e-5}}]}
    path.write_text(json.dumps(value, indent=2))
    return value


def _spec_recipe(tmp_path):
    packet = tmp_path / 'held-cases.jsonl'
    packet.write_text('{"id":"held-0","split":"test"}\n')
    packet_sha = hashlib.sha256(packet.read_bytes()).hexdigest()
    spec = {'packet': {'path': str(packet), 'sha256': packet_sha}, 'cases': [0],
            'workers': 1, 'neuralese_server': True, 'min_successes': 0,
            'runtime_failure_policy': 'report'}
    source_path = tmp_path / 'registered-task-recipe.json'
    _recipe(source_path, spec)
    return source_path, spec, packet


def _prepare(parent, source, output, proof):
    return prepare_foundation_boundary_recipe(
        parent_recipe=parent, parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        task_eval_recipe=source, task_eval_recipe_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        stage_id='native_full_sft', task_eval_stage_id='native_full_sft',
        output_recipe=output, proof_path=proof)


def test_successor_adds_only_registered_task_eval_and_keeps_boundary_cadence_zero(tmp_path):
    parent = tmp_path / 'current-foundation.json'
    parent_value = _recipe(parent)
    parent_before = parent.read_bytes()
    source, expected_spec, packet = _spec_recipe(tmp_path)
    output, proof_path = tmp_path / 'successor.json', tmp_path / 'proof.json'

    proof = _prepare(parent, source, output, proof_path)

    successor = json.loads(output.read_bytes())
    before_stage = parent_value['stages'][0]
    after_stage = successor['stages'][0]
    expected = copy.deepcopy(before_stage)
    expected['parameters']['task_eval'] = expected_spec
    expected['parameters']['task_eval_every'] = 0
    assert after_stage == expected
    assert parent.read_bytes() == parent_before
    assert proof['changed_parameters'] == ['task_eval', 'task_eval_every']
    assert proof['task_eval_every'] == 0
    assert proof['success_threshold_is_not_foundation_qualification'] is True
    assert proof['held_packet'] == {'path': str(packet.resolve()),
        'sha256': hashlib.sha256(packet.read_bytes()).hexdigest(), 'bytes': packet.stat().st_size}
    assert proof['output_recipe']['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(proof_path.read_bytes()) == proof


@pytest.mark.parametrize('mutation', ['stale_parent', 'stale_source', 'not_foundation', 'nonzero_cadence',
                                      'conflicting_eval', 'packet_bytes', 'wrong_split', 'non_native_transport'])
def test_successor_rejects_unbound_or_conflicting_recipe_inputs(tmp_path, mutation):
    parent = tmp_path / 'current-foundation.json'
    _recipe(parent)
    source, spec, packet = _spec_recipe(tmp_path)
    if mutation == 'not_foundation':
        _recipe(parent, foundation=False)
    elif mutation == 'nonzero_cadence':
        _recipe(parent, cadence=250)
    elif mutation == 'conflicting_eval':
        _recipe(parent, task_eval={'packet': {'path': '/wrong', 'sha256': 'f' * 64}})
    elif mutation == 'packet_bytes':
        packet.write_text('{"id":"changed"}\n')
    elif mutation in ('wrong_split', 'non_native_transport'):
        packet.write_text(json.dumps({'id': 'held-0', 'split': 'train'}) + '\n'
                          if mutation == 'wrong_split' else
                          json.dumps({'id': 'held-0', 'split': 'test'}) + '\n')
        spec['packet']['sha256'] = hashlib.sha256(packet.read_bytes()).hexdigest()
        if mutation == 'non_native_transport':
            spec['neuralese_server'] = False
        _recipe(source, spec)
    output, proof = tmp_path / 'next.json', tmp_path / 'proof.json'
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if mutation == 'stale_parent':
        parent_sha = '0' * 64
    elif mutation == 'stale_source':
        source_sha = '0' * 64
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_foundation_boundary_recipe(parent_recipe=parent, parent_sha256=parent_sha,
            task_eval_recipe=source, task_eval_recipe_sha256=source_sha,
            stage_id='native_full_sft', task_eval_stage_id='native_full_sft',
            output_recipe=output, proof_path=proof)
    assert not output.exists() and not proof.exists()


def test_successor_does_not_overwrite_existing_immutable_output(tmp_path):
    parent = tmp_path / 'current-foundation.json'
    _recipe(parent)
    source, _, _ = _spec_recipe(tmp_path)
    output, proof = tmp_path / 'next.json', tmp_path / 'proof.json'
    output.write_text('reserved')
    with pytest.raises(FileExistsError):
        _prepare(parent, source, output, proof)
    assert output.read_text() == 'reserved'
    assert not proof.exists()


def test_recipe_and_proof_outputs_cannot_alias(tmp_path):
    parent = tmp_path / 'current-foundation.json'
    _recipe(parent)
    source, _, _ = _spec_recipe(tmp_path)
    shared = tmp_path / 'shared-output.json'
    with pytest.raises(ValueError, match='distinct paths'):
        _prepare(parent, source, shared, shared)
    assert not shared.exists()
