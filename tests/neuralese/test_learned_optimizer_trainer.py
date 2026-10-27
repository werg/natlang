"""The outer learner-training step end to end on the native engine: writer, head, keys, bank composition, executor
distillation through generated states and composed adapters, exact cotangent replay, resume."""
import dataclasses
import json
import pytest

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


def setup(tmp_path, family='lfm'):
    engine = engine_fixture(tmp_path, family)
    if family == 'lfm':
        bank = _bank(engine)
    else:
        from natlang_neuralese.model.native_adapters import target_parameters
        from natlang_neuralese.tune.targets import module_map
        from natlang_neuralese.tune.learned.bank import new_bank, weight_svd_basis
        names, _ = module_map(sorted(target_parameters(engine.shared_model)))
        bank = new_bank(engine, weight_svd_basis(engine, sorted(names)[:6], 4), N.NumberFormat(.5))
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


def test_proposal_replays_at_ratio_one_and_joint_ratio_updates(tmp_path):
    from natlang_neuralese.tune.learned.episode import action_logprobs, propose, validate_proposal
    from natlang_neuralese.tune.learned.bundle import verify_bundle
    from natlang_neuralese.tune.learned.views import BankView
    engine, bank, target = setup(tmp_path)
    recipe = OuterRecipe(steps=1, batch_targets=1, writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.},
                         writer_optimizer={'name': 'adam', 'learning_rate': 1e-2})
    trainer = OuterTrainer(engine, bank, GRID, recipe, [target])
    trainer.step()  # a nonzero head, so the regenerated mixture depends on live parameters
    context = {key: target[key] for key in ('messages', 'writes', 'arrays', 'manifest', 'rows', 'placeholders')}
    context['program'] = {'id': 'p', 'buildHash': 'b'}
    bundle, proposal, behavior = propose(trainer, context, sample=True, seed=5)
    verify_bundle(engine, bundle, bank)
    view = target['arrays'][0]['view']
    validate_proposal(bank, bundle, {'lambda:x': BankView(view['snapshot'], view['scope'], view['level'], tuple(view['slots']),
                                     tuple(view['componentIds']), tuple(view['eligible']), tuple(view['dropoutMask']), None,
                                     view['selection'], view['mask'])})
    assert proposal['next'] == bundle['id'] and behavior['arrays']['lambda:x']['tokens'][0] == GRID.open
    old = sum(action_logprobs(engine, bundle, target['manifest'], target['rows']))
    record = {'context': context, 'behavior': behavior, 'old_executor_logprob': old, 'advantage': 1.}
    result = trainer.policy_step([record])
    policy = result['targets'][0]['policy']
    assert abs(policy['optimizerLogRatio']) < 1e-4 and abs(policy['executorLogRatio']) < 1e-3
    assert not policy['clipped'] and result['updated'].get('writer')


def test_policy_update_executes_and_trains_the_live_basis(tmp_path):
    from natlang_neuralese.tune.learned.episode import action_logprobs, propose
    engine, bank, target = setup(tmp_path)
    recipe = OuterRecipe(steps=1, batch_targets=1, train_keys=False, train_bank=True, train_basis=True, publish_every=99,
                         writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.}, bank_optimizer={'learning_rate': .01, 'weight_decay': 0.})
    trainer = OuterTrainer(engine, bank, GRID, recipe, [target])
    context = {key: target[key] for key in ('messages', 'writes', 'arrays', 'manifest', 'rows', 'placeholders')}
    context['program'] = {'id': 'p', 'buildHash': 'b'}
    bundle, _, behavior = propose(trainer, context, sample=True, seed=5)
    old = sum(action_logprobs(engine, bundle, target['manifest'], target['rows']))
    basis = {k: v.detach().clone() for k, v in trainer.bank_leaves['basis'].items()}
    result = trainer.policy_step([{'context': context, 'behavior': behavior, 'old_executor_logprob': old, 'advantage': 1.}])
    policy = result['targets'][0]['policy']
    # Live leaves equal the proposal's bank snapshot: the executor ratio starts at one.
    assert abs(policy['executorLogRatio']) < 1e-3 and not policy['clipped']
    assert any(not torch.equal(basis[k], v) for k, v in trainer.bank_leaves['basis'].items())
    engine.store.close()


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_bank_and_optimizer_resume_matches_uninterrupted_updates(tmp_path, family):
    from natlang_neuralese.tune.learned.trainer import _bank_tensor_map
    engine, bank, target = setup(tmp_path, family)
    recipe = OuterRecipe(steps=3, batch_targets=1, train_keys=False, train_bank=True, train_basis=True,
        publish_every=99, writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.},
        bank_optimizer={'learning_rate': .001, 'weight_decay': 0.})
    trainer = OuterTrainer(engine, bank, GRID, recipe, [target])
    initial = {k: v.detach().clone() for k, v in _bank_tensor_map(trainer.bank_leaves).items()}
    trainer.step()
    saved = json.loads(json.dumps(trainer.state))
    assert saved['bank_tensors'] and saved['bank_optimizer']
    assert any(not torch.equal(initial[k], v) for k, v in _bank_tensor_map(trainer.bank_leaves).items())
    trainer.step()
    expected = {k: v.detach().clone() for k, v in _bank_tensor_map(trainer.bank_leaves).items()}
    resumed = OuterTrainer(engine, bank, GRID, recipe, [target], state=saved)
    resumed.step()
    for name, tensor in _bank_tensor_map(resumed.bank_leaves).items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert resumed.state['bank_optimizer'] == trainer.state['bank_optimizer']
    assert resumed.state['writer'] == trainer.state['writer']
    assert resumed.state['head'] == trainer.state['head']
    assert resumed.state['exposure'] == trainer.state['exposure']
    engine.store.close()
