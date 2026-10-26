import json
from pathlib import Path

import pytest

from test_native_serving_adapters import engine_fixture

SNAPSHOTS = sorted(Path.home().glob('.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/*/config.json'))
_needs_lfm = pytest.mark.skipif(not SNAPSHOTS, reason='needs the LiquidAI/LFM2.5-350M snapshot in the local HF cache')


def real_lfm(test):
    """Real LFM2.5-350M in float32 on the CPU: a multi-GB peak, so check-main runs these in the separate heavy process
    (a full run's suite process was killed in the soft-state build test, 2026-10-26)."""
    return pytest.mark.gpu_heavy(_needs_lfm(test))


def real_engine(tmp_path):
    """The real LFM2.5-350M with its own chat template (the fixture's toy template cannot render ChatML)."""
    import torch
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
    from natlang_neuralese.serve import load_native_engine
    model, _, _, config = load_native_model(SNAPSHOTS[0].parent, device='cpu', dtype=torch.float32)
    checkpoint = tmp_path / 'native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, checkpoint)
    return load_native_engine(str(checkpoint), device='cpu', dtype=torch.float32), checkpoint

from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.rows import lower_rows, turn_graph


def turn(**changes):
    row = {'schema': 'natlang.tune-turn/v1', 'root': 'r1', 'component': 'c1', 'partition': 'train', 'binding': 'b1',
           'definition': 'outer.nl', 'call': 'task-1/c2', 'turn': 1,
           'messages': [{'role': 'system', 'content': 'You run natlang functions.'}, {'role': 'user', 'content': 'Upper-case hi.'}],
           'tools': [{'type': 'function', 'function': {'name': 'eval', 'description': 'Run code.',
                                                       'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}}}}],
           'reply': {'calls': [['eval', {'code': 'return "HI"'}]]}, 'teacher': 'teacher:test', 'source': 'real',
           'evidence': {'rootOracle': True, 'enclosingSuccess': True, 'callOutcome': 'done'}}
    row.update(changes)
    return row


@real_lfm
def test_turn_renders_with_the_student_template_and_targets_only_the_reply(tmp_path):
    engine, _ = real_engine(tmp_path)
    lowered = turn_graph(engine, turn(), prompt_weight=.1)
    graph = lowered['graph']
    example = parse_graph(graph, tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
    weights = graph['nodes'][0]['target_weights']
    assert weights[0] == 0 and set(weights) == {0., .1, 1.}
    assert example.plan.token_count == len(graph['nodes'][0]['token_ids'])
    target = [token for token, weight in zip(graph['nodes'][0]['token_ids'], weights) if weight == 1.]
    assert 'eval' in engine.tokenizer.decode(target)
    assert lowered['binding'] == 'b1' and lowered['evidence_weight'] == 1. and lowered['occurrence'] == 'r1:task-1/c2:1'
    assert lower_rows(engine, [graph_row := lowered])[0] is graph_row  # already-lowered rows pass through


def test_non_training_and_failed_roots_never_become_positive_rows(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    with pytest.raises(TuneMathError, match='training-partition'):
        turn_graph(engine, turn(partition='development'), prompt_weight=.1)
    with pytest.raises(TuneMathError, match='not a positive target'):
        turn_graph(engine, turn(evidence={'rootOracle': False, 'enclosingSuccess': False, 'callOutcome': 'failed'}), prompt_weight=.1)
    from natlang_neuralese.tune.rows import evidence_weight
    assert evidence_weight(turn(evidence={'rootOracle': None, 'enclosingSuccess': True, 'callOutcome': 'done'})) == .5


@real_lfm
def test_build_trains_from_collected_turns(tmp_path):
    engine, checkpoint = real_engine(tmp_path)
    from natlang_neuralese.tune.build import main
    manifest = {'schema': 'natlang.adapters/v1', 'program': {'id': 'p', 'buildHash': 'h'},
                'base': {key: engine.native_identity[key] for key in ('architecture', 'checkpoint_sha256', 'embedding_weight_sha256', 'tokenizer_fingerprint')},
                'tree': [{'id': 'tree:', 'level': 'tree', 'path': '', 'parent': None},
                         {'id': 'lambda:outer', 'level': 'lambda', 'path': 'outer.nl', 'parent': 'tree:'}],
                'bindings': [{'id': 'b1', 'component': 'outer', 'definitionId': 'outer.nl', 'contextId': None, 'origin': 'named',
                              'ancestry': ['tree:', 'lambda:outer'], 'semanticHash': 's'}],
                'artifacts': [], 'families': [], 'memberships': [], 'scales': {}, 'validatedReductions': [], 'fallback': 'base',
                'receipts': {}, 'digest': 'd'}
    rows = tmp_path / 'rows.jsonl'
    rows.write_text('\n'.join(json.dumps(turn(root=f'r{index}', turn=1)) for index in range(2)) + '\n')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'manifest': manifest, 'rows': str(rows), 'student': {'name': 'lfm', 'checkpoint': str(checkpoint)},
                                'recipe': {'steps': 1, 'batch_rows': 2}}))
    assert main(['--plan', str(plan), '--out', str(tmp_path / 'out')]) == 0
    result = json.loads((tmp_path / 'out' / 'result.json').read_text())
    assert {item['node'] for item in result['artifacts']} == {'tree:', 'lambda:outer'}
    assert result['receipt']['exposure']['rows_seen'] == 2


@real_lfm
def test_distillation_export_lowers_training_turns_and_binds_hashes(tmp_path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.tune import distill
    engine, checkpoint = real_engine(tmp_path)
    export = tmp_path / 'export'
    export.mkdir()
    (export / 'turns.jsonl').write_text(json.dumps(turn()) + '\n')
    (export / 'manifest.json').write_text(json.dumps({'turns': {'sha256': sha256_file_hex(export / 'turns.jsonl')}}))
    assert distill.main(['--export', str(export), '--checkpoint', str(checkpoint)]) == 0
    lowered = json.loads((export / 'lowered.json').read_text())
    graphs = [json.loads(line) for line in (export / 'native-graphs.jsonl').read_text().splitlines()]
    assert lowered['native_graphs']['rows'] == len(graphs) == 1
    assert lowered['native_graphs']['sha256'] == sha256_file_hex(export / 'native-graphs.jsonl')
    parse_graph(graphs[0], tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
    # Only training-split tune turns are accepted, and the manifest must bind the exact turns.
    (export / 'turns.jsonl').write_text(json.dumps(turn(partition='development')) + '\n')
    with pytest.raises(TuneMathError, match='does not match'):
        distill.main(['--export', str(export), '--checkpoint', str(checkpoint)])
    (export / 'manifest.json').write_text(json.dumps({'turns': {'sha256': sha256_file_hex(export / 'turns.jsonl')}}))
    with pytest.raises(TuneMathError, match='training-split'):
        distill.main(['--export', str(export), '--checkpoint', str(checkpoint)])
