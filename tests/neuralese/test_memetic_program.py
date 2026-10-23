import json
from pathlib import Path

import pytest

from natlang_neuralese.rl import memetic_program as memetic


def case(ident, program='p', split='train'):
    return {'id': ident, 'program_id': program, 'split': split,
        'curriculum': {'reference': {'root': [['eval', {'code': 'return 1'}]]}}}


def test_repeated_program_slices_are_disjoint_train_only(tmp_path):
    dev, confirmation = tmp_path / 'dev.jsonl', tmp_path / 'confirm.jsonl'
    dev.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('a', 'b')))
    confirmation.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('c', 'd')))
    left, right = memetic.prepare_slices(dev, confirmation, 'p')
    assert [item['id'] for item in left + right] == ['a', 'b', 'c', 'd']
    confirmation.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('a', 'd')))
    with pytest.raises(ValueError, match='overlap'):
        memetic.prepare_slices(dev, confirmation, 'p')
    confirmation.write_text(''.join(json.dumps(case(i, split='test')) + '\n' for i in ('c', 'd')))
    with pytest.raises(ValueError, match='non-train'):
        memetic.prepare_slices(dev, confirmation, 'p')


def test_binding_requires_actual_native_id_and_oracle_gain(tmp_path):
    ident = 'nzs1_' + 'a' * 52
    assert memetic.state_guidance('Check the contract.', ident) == '\ue002' + ident + '\ue003'
    with pytest.raises(ValueError):
        memetic.state_guidance('Check.', 'nz1_' + 'a' * 52)
    assert memetic.acceptance({'scored': 4, 'successes': 1}, {'scored': 4, 'successes': 2})
    assert not memetic.acceptance({'scored': 4, 'successes': 1}, {'scored': 3, 'successes': 3})
    assert not memetic.acceptance({'scored': 4, 'successes': 1}, {'scored': 4, 'successes': 1})
    output = tmp_path / 'row-000' / 'sample-00'
    output.mkdir(parents=True)
    (output / 'results.jsonl').write_text(json.dumps({'trajectory': [
        {'model_response': {'raw_response': {'native_state': {'imports': [{'id': ident}]}}}}]}) + '\n')
    assert memetic.native_import_reached(tmp_path, ident)
    assert not memetic.native_import_reached(tmp_path, 'nzs1_' + 'b' * 52)


def test_author_requires_exact_self_hosted_source_receipt(tmp_path):
    path = tmp_path / 'teacher.json'
    receipt = {'schema': 'natlang.memetic-author-provenance/1', 'model_id': 'Qwen/Qwen3.6-35B-A3B',
        'hosting': 'self-hosted-coreweave', 'checkpoint_sha256': 'a' * 64,
        'tokenizer_fingerprint': 'b' * 64, 'runtime_sha256': 'c' * 64}
    path.write_text(json.dumps(receipt))
    assert memetic.author_provenance(path, receipt['model_id'])['receipt'] == receipt
    receipt['hosting'] = 'provider'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='self-hosted'):
        memetic.author_provenance(path, receipt['model_id'])


def test_writer_rejects_wrong_student_or_missing_producer(monkeypatch):
    ident = 'nzs1_' + 'a' * 52
    actor = {key: 'f' * 64 for key in memetic.IDENTITY}
    actor['architecture'] = 'lfm'
    good = {'id': ident, 'native_state': actor,
        'source_capture': {'policy': actor, 'segments': [
            {'producing': [False]}, {'producing': [True]}, {'producing': [False]}]},
        'source_text_sha256': __import__('hashlib').sha256(b'guide').hexdigest()}
    monkeypatch.setattr(memetic, 'request_json', lambda *a, **k: good)
    assert memetic.write_state('http://localhost', 'guide', actor) == good
    bad = {**good, 'native_state': {**actor, 'checkpoint_sha256': '0' * 64}}
    monkeypatch.setattr(memetic, 'request_json', lambda *a, **k: bad)
    with pytest.raises(ValueError, match='identity'):
        memetic.write_state('http://localhost', 'guide', actor)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_guidance_graph_backpropagates_through_local_producer(family):
    from natlang_neuralese.data.native_graph import graph_loss, parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from test_shared_state import tiny
    actor = {key: 'f' * 64 for key in memetic.IDENTITY}
    written = {'id': 'nzs1_' + 'a' * 52,
        'source_capture': {'policy': actor, 'segments': [
            {'token_ids': [58], 'producing': [False]},
            {'token_ids': [3, 4], 'producing': [True, True]},
            {'token_ids': [59], 'producing': [False]}], 'payload_start': 1, 'payload_stop': 3}}
    source = {'schema': 'natlang.neuralese-native-graph/1', 'id': 'case-1',
        'program_id': 'p', 'split': 'train', 'source_groups': ['teacher-case-1'],
        'tokenizer_fingerprint': 'f' * 64, 'nodes': [
            {'name': 'call', 'token_ids': [6, 7, 8], 'target_weights': [0, 1, 1]}]}
    row = memetic.guidance_training_graph(source, program_id='p', written=written, actor=actor)
    assert row['nodes'][1]['imports'] == [{'at': 0, 'source': 'memetic-guidance-producer', 'start': 1, 'stop': 3}]
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    example = parse_graph(row, tokenizer_fingerprint='f' * 64)
    loss, _ = graph_loss(model, example, checkpoint_layers=True, chunk_tokens=1)
    loss.backward()
    assert model.adapter.up.weight.grad is not None
    assert model.adapter.up.weight.grad.norm().item() > 0


def test_recipe_continuation_uses_native_earned_parent(tmp_path):
    base = Path(__file__).resolve().parents[2] / 'training/neuralese/recipes/native-state-production-base-20261015-v1.json'
    output = tmp_path / 'recipe.json'
    binding = {'path': '/work/memetic/train.jsonl', 'sha256': 'a' * 64}
    receipt = memetic.emit_recipe(base, output, binding)
    recipe = json.loads(output.read_text())
    stage = recipe['stages'][-1]
    assert receipt['stage_id'] == stage['id']
    assert stage['requires'] == ['native_full_sft']
    assert stage['inputs']['native_graphs'] == 'memetic_native_train_graphs'
    assert stage['inputs']['held_native_graphs'] == 'native_held_graphs'
    assert recipe['input_bindings']['memetic_native_train_graphs'] == binding
