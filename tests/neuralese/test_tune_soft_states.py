"""Soft instructions and soft skills: directly tuned native state jointly with program residuals (plan §3.3)."""
import copy
import json
from concurrent.futures import Future

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for, tensors
from natlang_neuralese.model.native_adapters import run_native
from natlang_neuralese.model.native_state_derivation import state_values, verify_state_derivation
from natlang_neuralese.model.shared_state import ImportSpan, Invocation, compile_graph
from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.soft import (initialize_soft_states, place_states, soft_items_to_node, soft_key,
                                         text_sha256)
from natlang_neuralese.tune.trainer import JointTrainer, Recipe

INSTRUCTIONS = 't20 t21 t22'


def turn(binding='bx', root='r1', **changes):
    row = {'schema': 'natlang.tune-turn/v1', 'root': root, 'component': 'c', 'partition': 'train', 'binding': binding,
           'call': 'call-1', 'turn': 1, 'tools': [], 'teacher': 'teacher:test', 'source': 'real',
           'messages': [{'role': 'system', 'content': 't10 t11'}, {'role': 'user', 'content': 't12 ' + INSTRUCTIONS + ' t13'},
                        {'role': 'tool', 'content': 't30 t31'}],
           'texts': {'instructions': INSTRUCTIONS, 'skills': {'skills.s': 't30 t31', 'skills.unread': 't40'}},
           'reply': {'text': 't14'}, 'evidence': {'rootOracle': True, 'enclosingSuccess': True, 'callOutcome': 'done',
           'actionPolicy': 'natlang.tune-action-outcome/2', 'actionDisposition': 'admissible', 'failedHistory': False,
           'actionEvents': []}}
    row.update(changes)
    return row


def test_states_are_placed_exactly_where_their_texts_were():
    states = {soft_key('bx', 'instructions', None): 'nzs1_' + 'a' * 52, soft_key('bx', 'skill', 'skills.s'): 'nzs1_' + 'b' * 52}
    messages, placed = place_states(turn(), states)
    assert messages[1]['content'] == [{'type': 'text', 'text': 't12 '}, {'type': 'native_state', 'id': 'nzs1_' + 'a' * 52},
                                      {'type': 'text', 'text': ' t13'}]
    assert messages[2]['content'] == [{'type': 'native_state', 'id': 'nzs1_' + 'b' * 52}]
    assert placed == [soft_key('bx', 'instructions', None), soft_key('bx', 'skill', 'skills.s')]
    missing = turn(messages=[{'role': 'user', 'content': 't12'}])
    with pytest.raises(TuneMathError, match='occurs 0 times'):
        place_states(missing, states)
    twice = turn(messages=[{'role': 'user', 'content': INSTRUCTIONS + ' ' + INSTRUCTIONS}])
    with pytest.raises(TuneMathError, match='occurs 2 times'):
        place_states(twice, states)


def soft_engine(tmp_path):
    return engine_fixture(tmp_path, 'lfm', service_template=True)


def test_initialization_writes_in_context_and_training_graph_equals_serving_consumption(tmp_path):
    engine = soft_engine(tmp_path)
    records = initialize_soft_states(engine, [turn(), turn(root='r2')])
    assert set(records) == {soft_key('bx', 'instructions', None), soft_key('bx', 'skill', 'skills.s')}, 'unread skills stay text'
    record = records[soft_key('bx', 'instructions', None)]
    block = engine.store.get_native_state(record['state']).block
    assert block.text == INSTRUCTIONS and record['textSha256'] == text_sha256(INSTRUCTIONS)
    assert record['init']['writer'] == 'native_init_body' and block.adapter_binding is None
    assert initialize_soft_states(engine, [turn(), turn(root='r2')]) == records, 'deterministic and content addressed'
    # The training graph's consumer import equals the serving engine's own native-state consumption.
    tokens, weights, imports, external = soft_items_to_node(engine, [('tok', 4), ('block', record['state']), ('tok', 7)],
                                                            [], prompt_weight=.1, state_names={record['state']: 'soft:instructions'})
    assert tokens[1:3] == [engine.role_tokens.consume_open, engine.role_tokens.consume_close] and weights[2] == 0
    plan = compile_graph([Invocation('turn', tuple(tokens), imports=tuple(imports))],
                         external_sizes={'soft:instructions': external['soft:instructions']['token_count']})
    identity = {key: getattr(block, key) for key in ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')}
    graph = engine.shared_model(plan, external_blocks={'soft:instructions': block}, external_identity={'soft:instructions': identity})
    from natlang_neuralese.serve.engine import GenerationRequest, Sequence
    seq = Sequence(GenerationRequest(messages=[]), Future())
    seq.cache = engine._native_forward([4], None)['cache']; seq.native_own_position = 1
    engine._native_consume(seq, record['state'])
    served = engine._native_forward([7], seq.cache)['h_final'][:, -1]
    torch.testing.assert_close(graph['h_final'][:, -1], served, rtol=2e-5, atol=3e-6)


def soft_rows(engine, records, binding='bx', count=3):
    """Hand-rendered rows (the fixture template has no ChatML generation prefix) importing the binding's states."""
    fingerprint = engine.native_identity['tokenizer_fingerprint']
    key = soft_key(binding, 'instructions', None)
    state = records[key]['state']
    rows = []
    for index in range(count):
        tokens, weights, imports, external = soft_items_to_node(engine, [('tok', 4), ('block', state), ('tok', 5 + index % 3)],
            [('tok', 8), ('tok', 9 + index % 2)], prompt_weight=.1, state_names={state: 'soft:instructions'})
        import dataclasses
        rows.append({'binding': binding, 'evidence_weight': 1., 'occurrence': f'soft-{binding}-{index}',
                     'soft': {'soft:instructions': key},
                     'graph': {'schema': 'natlang.neuralese-native-graph/1', 'id': f'soft-{binding}-{index}', 'split': 'train',
                               'source_groups': ['g'], 'tokenizer_fingerprint': fingerprint, 'external_states': external,
                               'nodes': [{'name': 'turn', 'token_ids': tokens, 'target_weights': weights,
                                          'imports': [dataclasses.asdict(edge) for edge in imports]}]}})
    return rows


def test_joint_step_updates_the_soft_state_with_lineage_and_the_coefficients(tmp_path):
    engine = soft_engine(tmp_path)
    records = initialize_soft_states(engine, [turn()], skills=False)
    rows = soft_rows(engine, records) + rows_for(engine, 'by', 2)
    trainer = JointTrainer(engine, manifest_for(engine), rows, Recipe(steps=2, batch_rows=5,
        optimizer={'name': 'sgd', 'learning_rate': 1.}, state_optimizer={'name': 'adam', 'learning_rate': .05}))
    key = soft_key('bx', 'instructions', None)
    initial = trainer.state['soft_states'][key]
    before = {node: trainer.state['node_artifacts'][node] for node in ('lambda:x', 'lambda:y')}
    record = trainer.step()
    assert record['updated_soft_states'] == [key]
    tuned = trainer.state['soft_states'][key]
    assert tuned != initial
    old, new = (state_values(engine.store.get_native_state(ident).block) for ident in (initial, tuned))
    assert any(not torch.equal(old[name], new[name]) for name in old), 'the state values actually changed'
    block = engine.store.get_native_state(tuned).block
    verify_state_derivation(engine.store, block)
    assert block.derivation['parent_state_id'] == initial and block.text == INSTRUCTIONS
    assert all(trainer.state['node_artifacts'][node] != before[node] for node in before), 'coefficients moved in the same step'
    trainer.step()
    assert engine.store.get_native_state(trainer.state['soft_states'][key]).block.derivation['step'] == 2


def test_text_rows_are_unaffected_by_soft_state_training(tmp_path):
    engine = soft_engine(tmp_path)
    alone = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'by', 2), Recipe(steps=1, batch_rows=2))
    assert alone.state['soft_states'] == {}
    assert alone.step()['updated_soft_states'] == []


def test_resume_reproduces_soft_and_coefficient_updates(tmp_path):
    engine = soft_engine(tmp_path)
    records = initialize_soft_states(engine, [turn()], skills=False)
    rows = soft_rows(engine, records) + rows_for(engine, 'by', 2)
    recipe = Recipe(steps=3, batch_rows=2, state_optimizer={'name': 'adam', 'learning_rate': .05})
    straight = JointTrainer(engine, manifest_for(engine), rows, recipe)
    straight.run()
    first = JointTrainer(engine, manifest_for(engine), rows, recipe)
    first.step()
    saved = json.loads(json.dumps(first.state))
    resumed = JointTrainer(engine, manifest_for(engine), rows, recipe, state=saved)
    resumed.run(2)
    for field in ('soft_states', 'soft_optimizer_states', 'node_artifacts', 'optimizer_states', 'exposure'):
        assert resumed.state[field] == straight.state[field], field


def test_rows_importing_foreign_state_are_still_refused(tmp_path):
    engine = soft_engine(tmp_path)
    records = initialize_soft_states(engine, [turn()], skills=False)
    rows = soft_rows(engine, records)
    foreign = copy.deepcopy(rows[0]); foreign.pop('soft')
    with pytest.raises(NotImplementedError):
        JointTrainer(engine, manifest_for(engine), [foreign], Recipe(steps=0))
    other = copy.deepcopy(rows[0]); other['binding'] = 'by'
    with pytest.raises(TuneMathError, match='another binding'):
        JointTrainer(engine, manifest_for(engine), [other], Recipe(steps=0))


from test_tune_rows import real_engine, real_lfm


@real_lfm
def test_real_lfm_build_initializes_trains_and_persists_soft_states(tmp_path):
    """End to end on LFM2.5-350M: collected turns -> in-context soft states -> joint fit -> release fragment."""
    from pathlib import Path
    from natlang_neuralese.tune.build import main
    from natlang_neuralese.tune.rows import turn_graph
    engine, checkpoint = real_engine(tmp_path)
    instructions = 'Return the value in uppercase letters.'
    def real_turn(root, value):
        return turn(root=root, messages=[{'role': 'system', 'content': 'You execute natlang functions.'},
                                         {'role': 'user', 'content': f'Instructions: {instructions}\nvalue: {value}'}],
                    texts={'instructions': instructions}, reply={'text': value.upper()})
    turns = [real_turn('r1', 'alpha'), real_turn('r2', 'beta')]
    records = initialize_soft_states(engine, turns)
    key = soft_key('bx', 'instructions', None)
    lowered = turn_graph(engine, turns[0], prompt_weight=.1, soft={key: records[key]['state']})
    assert lowered['soft'] == {'soft:instructions': key}
    assert lowered['graph']['external_states']['soft:instructions']['id'] == records[key]['state']
    manifest = manifest_for(engine)
    rows = tmp_path / 'rows.jsonl'
    rows.write_text('\n'.join(json.dumps(item) for item in turns) + '\n')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'manifest': manifest, 'rows': str(rows), 'softStates': True,
                                'student': {'name': 'lfm', 'checkpoint': str(checkpoint)},
                                'recipe': {'steps': 1, 'batch_rows': 2, 'state_optimizer': {'name': 'adam', 'learning_rate': .01}}}))
    out = tmp_path / 'out'
    assert main(['--plan', str(plan), '--out', str(out)]) == 0
    result = json.loads((out / 'result.json').read_text())
    [soft] = result['softStates']
    assert soft['binding'] == 'bx' and soft['kind'] == 'instructions' and soft['textSha256'] == text_sha256(instructions)
    assert soft['state'] != records[key]['state'], 'the released state is the tuned one'
    assert (out / result['softStateFiles'][soft['state']]).exists()
    # main loads a fresh engine: resuming restores states from the archive, reinitializes nothing, reports the same state.
    assert json.loads((out / 'soft-init.json').read_text())[key]['state'] == records[key]['state']
    assert main(['--plan', str(plan), '--out', str(out)]) == 0
    assert json.loads((out / 'result.json').read_text())['softStates'] == result['softStates']
