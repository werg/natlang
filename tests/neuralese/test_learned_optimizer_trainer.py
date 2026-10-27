"""The outer learner-training step end to end on the native engine: writer, head, keys, bank composition, executor
distillation through generated states and composed adapters, exact cotangent replay, resume."""
import dataclasses
import json

import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for
from test_learned_optimizer_bank import _bank, _component
from test_learned_optimizer_writer import GRID
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.tune.learned import numbers as N
from natlang_neuralese.tune.learned import head as H
from natlang_neuralese.tune.learned.assemble import training_target
from natlang_neuralese.tune.learned.evidence import build_view
from natlang_neuralese.tune.learned.fit import placeholder_adapters
from natlang_neuralese.tune.learned.trainer import OuterRecipe, OuterTrainer
from natlang_neuralese.tune.learned.views import select_view
from natlang_neuralese.tune.soft import soft_key


def soft_rows(engine, state, count):
    roles, fingerprint = engine.role_tokens, engine.native_identity['tokenizer_fingerprint']
    tokens = len(engine.store.get_native_state(state).block.token_ids)
    rows = []
    for index in range(count):
        ids = [4, roles.consume_open, roles.consume_close, 5 + index % 3, 6, 7, 8]
        rows.append({'binding': 'bx', 'evidence_weight': 1., 'occurrence': f'bx-{index}',
                     'soft': {'soft:instructions': soft_key('bx', 'instructions', None)},
                     'graph': {'schema': 'natlang.neuralese-native-graph/1', 'id': f'bx-{index}', 'split': 'train',
                               'source_groups': ['bx-group'], 'tokenizer_fingerprint': fingerprint,
                               'external_states': {'soft:instructions': {'id': state, 'token_count': tokens}},
                               'nodes': [{'name': 'answer', 'token_ids': ids, 'target_weights': [0, 0, 0, 0, 1, 1, 1],
                                          'imports': [{'at': 2, 'source': 'soft:instructions', 'start': 0, 'stop': tokens}],
                                          'producing': []}]}})
    return rows


def setup(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    bank = _bank(engine)
    components = []
    for seed in (1, 2, 3):
        component = _component(bank, seed)
        components.append(dataclasses.replace(component, key=engine.native_write_text(f't{30 + seed} t{40 + seed}')['id']))
    bank = dataclasses.replace(bank.with_components(components, reason='planted'), number_format=N.NumberFormat(.08))
    manifest = manifest_for(engine)
    incoming = engine.native_write_text('t8 t9')['id']
    block = engine.store.get_native_state(incoming).block
    planted = dataclasses.replace(block, layers=tuple({k: (v + .05 * torch.randn_like(v) if k != 'kind' else v) for k, v in layer.items()}
                                                      for layer in block.layers))
    target_state = engine.store.put_native_state(planted).id
    key = soft_key('bx', 'instructions', None)
    view = select_view(bank, 'lambda:x', 'lambda', 3, policy='full-bank-v1')
    integers = [3, 0, -2]
    shares = H.shares_from_norms(bank.group_norms())
    executed = H.executed(integers, torch.zeros(3), torch.zeros(3, len(bank.group_labels)), shares, step=.08, beta=.5)
    target_adapter = bank.compose_adapter(engine, bank.composed_cores([0, 1, 2], executed)).id
    placeholders = placeholder_adapters(engine, bank, ['lambda:x'])
    target = training_target(engine, bank, manifest=manifest, rows=soft_rows(engine, incoming, 2), contract='route tickets',
                             view=build_view('route tickets', 'incoming', []), soft_texts={key: 't8 t9'},
                             incoming_states={key: incoming}, bank_views={'lambda:x': view},
                             target_integers={'lambda:x': integers}, target_adapters={'lambda:x': target_adapter},
                             target_states={key: target_state}, target_rho={'lambda:x': [0., 0., 0.]},
                             target_delta={'lambda:x': torch.zeros(3, len(bank.group_labels)).tolist()},
                             placeholders=placeholders)
    return engine, bank, target


def test_outer_step_updates_writer_head_and_keys_from_one_snapshot(tmp_path):
    engine, bank, target = setup(tmp_path)
    recipe = OuterRecipe(steps=2, batch_targets=1, writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.},
                         writer_optimizer={'name': 'adam', 'learning_rate': 1e-2}, key_optimizer={'name': 'adam', 'learning_rate': 1e-2},
                         head_optimizer={'learning_rate': 1e-2})
    trainer = OuterTrainer(engine, bank, GRID, recipe, [target])
    writer_before, keys_before = trainer.state['writer'], dict(trainer.state['keys'])
    record = trainer.step()
    assert record['updated'].get('writer') and record['updated'].get('head') and record['updated'].get('keys')
    terms = record['targets'][0]
    assert terms['distill'] > 0 and terms['mixture_tokens'] > 0 and terms['instruction_state'] > 0
    # Realized route: parsed integers carry no gradient to the sign logits. The zero-initialized readout passes no
    # cotangent to the hidden states at step 0 (dL/dh = W^T dL/du with W = 0); the head's own weights move first.
    assert terms['cotangents']['sign'] == 0 and terms['cotangents']['hidden'] == 0
    assert trainer.state['writer'] != writer_before and trainer.state['keys'] != keys_before
    assert trainer.state['exposure'] == {target['id']: 1}
    # Resume restores the head and continues the same cursor and lineage.
    resumed = OuterTrainer(engine, bank, GRID, recipe, [target], state=json.loads(json.dumps(trainer.state)))
    assert resumed.head.identity() == trainer.head.identity()
    second = resumed.step()
    assert second['targets'][0]['cotangents']['hidden'] > 0
    assert resumed.state['step'] == 2 and resumed.state['writer_history'][:2] == trainer.state['writer_history'][:2]


def test_token_expectation_route_trains_the_sign_and_digit_logits(tmp_path):
    engine, bank, target = setup(tmp_path)
    recipe = OuterRecipe(steps=1, batch_targets=1, writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.},
                         coefficients='token-expectation-v1', train_keys=False,
                         weights={'distill': 1.}, writer_optimizer={'name': 'adam', 'learning_rate': 1e-2})
    trainer = OuterTrainer(engine, bank, GRID, recipe, [target])
    record = trainer.step()
    # With only the distillation term, the writer still moves: behavior reached the integer choice through the
    # declared expectation estimator (no parsed-number gradient exists).
    assert record['updated'].get('writer')
    routes = record['targets'][0]['cotangents']
    assert routes['sign'] > 0 and routes['digit'] > 0 and routes['other'] > 0 and routes['states'] > 0
