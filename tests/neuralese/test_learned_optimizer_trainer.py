"""The outer learner-training step end to end on the native engine: writer, component-token head, keys, bank
composition, executor distillation through generated states and composed adapters, exact cotangent replay, the
expected-score update, resume."""
import dataclasses
import json
import pytest

import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for
from test_learned_optimizer_bank import _bank, _component
from natlang_neuralese.tune.learned import mixture as M
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
        bank = new_bank(engine, weight_svd_basis(engine, sorted(names)[:6], 4), M.MixtureFormat())
    components = []
    for seed in (1, 2, 3):
        component = _component(bank, seed)
        components.append(dataclasses.replace(component, key=engine.native_write_text(f't{30 + seed} t{40 + seed}')['id']))
    bank = bank.with_components(components, reason='planted')
    manifest = manifest_for(engine)
    incoming = engine.native_write_text('t8 t9')['id']
    block = engine.store.get_native_state(incoming).block
    planted = dataclasses.replace(block, layers=tuple({k: (v + .05 * torch.randn_like(v) if k != 'kind' else v) for k, v in layer.items()}
                                                      for layer in block.layers))
    target_state = engine.store.put_native_state(planted).id
    key = soft_key('bx', 'instructions', None)
    view = select_view(bank, 'lambda:x', 'lambda', 3, policy='full-bank-v1')
    label = {'distribution': [.1, .6, 0., .3], 'gain': [1.2] * len(bank.group_labels)}
    executed = M.coefficients(torch.tensor(label['distribution']), torch.tensor(label['gain']))
    target_adapter = bank.compose_adapter(engine, bank.composed_cores([0, 1, 2], executed)).id
    placeholders = placeholder_adapters(engine, bank, ['lambda:x'])
    target = training_target(engine, bank, manifest=manifest, rows=soft_rows(engine, incoming, 2), contract='route tickets',
                             view=build_view('route tickets', 'incoming', []), soft_texts={key: 't8 t9'},
                             incoming_states={key: incoming}, bank_views={'lambda:x': view},
                             target_mixtures={'lambda:x': label}, target_adapters={'lambda:x': target_adapter},
                             target_states={key: target_state}, placeholders=placeholders)
    return engine, bank, target


def recipe_for(**extra):
    return OuterRecipe(**{'steps': 2, 'batch_targets': 1, 'writer': {'kind': 'free-lora', 'rank': 2, 'alpha': 2.},
                          'writer_optimizer': {'name': 'adam', 'learning_rate': 1e-2},
                          'key_optimizer': {'name': 'adam', 'learning_rate': 1e-2}, 'head_optimizer': {'learning_rate': 1e-2},
                          **extra})


def context_of(target):
    context = {key: target[key] for key in ('id', 'messages', 'writes', 'arrays', 'manifest', 'rows', 'placeholders')}
    context['program'] = {'id': 'p', 'buildHash': 'b'}
    return context


def test_outer_step_updates_writer_head_and_keys_from_one_snapshot(tmp_path):
    engine, bank, target = setup(tmp_path)
    recipe = recipe_for(dropout=0.)
    trainer = OuterTrainer(engine, bank, recipe, [target])
    writer_before, keys_before = trainer.state['writer'], dict(trainer.state['keys'])
    record = trainer.step()
    assert record['updated'].get('writer') and record['updated'].get('head') and record['updated'].get('keys')
    terms = record['targets'][0]
    assert terms['distill'] > 0 and terms['mixture'] > 0 and terms['gain'] > 0 and terms['instruction_state'] > 0
    # Rows were initialized from the keys in this request, so behavior reaches the decision state at step 0.
    assert terms['cotangents']['hidden'] > 0 and terms['cotangents']['states'] > 0
    assert bool(trainer.head.initialized.all())
    assert trainer.state['writer'] != writer_before and trainer.state['keys'] != keys_before
    assert trainer.state['exposure'] == {target['id']: 1}
    # Resume restores the head (BF16 rows, AdamWSR moments) and continues the same cursor and lineage.
    resumed = OuterTrainer(engine, bank, recipe, [target], state=json.loads(json.dumps(trainer.state)))
    assert resumed.head.identity() == trainer.head.identity()
    assert resumed.head.rows.dtype == torch.bfloat16
    resumed.step()
    assert resumed.state['step'] == 2 and resumed.state['writer_history'][:2] == trainer.state['writer_history'][:2]


def test_dropout_masks_components_and_the_label(tmp_path):
    engine, bank, target = setup(tmp_path)
    trainer = OuterTrainer(engine, bank, recipe_for(dropout=.5), [target])
    views = trainer._views(target['arrays'], 3)
    assert views == trainer._views(target['arrays'], 3)  # recorded per step: the same seed gives the same mask
    masks = {tuple(trainer._views(target['arrays'], seed)['lambda:x']['available']) for seed in range(20)}
    assert len(masks) > 1 and all(len(mask) == 3 for mask in masks)
    trainer.step()  # a dropped labeled component renormalizes the label or drops the mixture term, never fails


def test_expected_score_update_moves_mass_toward_the_better_token(tmp_path):
    from natlang_neuralese.tune.learned.bundle import verify_bundle
    from natlang_neuralese.tune.learned.episode import candidates, propose
    engine, bank, target = setup(tmp_path)
    trainer = OuterTrainer(engine, bank, recipe_for(train_keys=False, head_optimizer={'learning_rate': .05}), [target])
    trainer.step()
    context = context_of(target)
    bundle, proposal, behavior = propose(trainer, context, seed=5)
    verify_bundle(engine, bundle, bank)
    assert proposal['next'] == bundle['id'] and behavior['arrays']['lambda:x']['chosen'] is None
    tokens = candidates(behavior)['lambda:x']
    assert tokens and tokens[0] in (M.NULL, 0, 1, 2)
    for token in tokens:  # every candidate executes exactly one kept token at the scope's gains
        single, _, _ = propose(trainer, context, choose={'lambda:x': token}, seed=5)
        verify_bundle(engine, single, bank)
        assert single['loraMixtures']['lambda:x']['mixture']['chosen'] == token
    best = tokens[-1]
    before = [float.fromhex(x) for x in behavior['arrays']['lambda:x']['distribution']]
    records = [{'context': context, 'behavior': behavior, 'candidate': {'scope': 'lambda:x', 'token': token},
                'reward': {'total': 1. if token == best else 0.}} for token in tokens]
    records.append({'context': context, 'behavior': behavior, 'candidate': None, 'reward': {'total': .5}})
    result = trainer.policy_step(records)
    assert result['estimator'] == 'expected-score-v1' and result['updated'].get('writer') and result['updated'].get('head')
    _, _, after = propose(trainer, context, seed=5)
    position = 0 if best == M.NULL else 1 + best
    later = [float.fromhex(x) for x in after['arrays']['lambda:x']['distribution']]
    if len(tokens) > 1:
        assert later[position] > before[position]


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_bank_and_optimizer_resume_matches_uninterrupted_updates(tmp_path, family):
    from natlang_neuralese.tune.learned.trainer import _bank_tensor_map
    engine, bank, target = setup(tmp_path, family)
    recipe = OuterRecipe(steps=3, batch_targets=1, train_keys=False, train_bank=True, train_basis=True,
        publish_every=99, writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.},
        bank_optimizer={'learning_rate': .001, 'weight_decay': 0.})
    trainer = OuterTrainer(engine, bank, recipe, [target])
    initial = {k: v.detach().clone() for k, v in _bank_tensor_map(trainer.bank_leaves).items()}
    trainer.step()
    saved = json.loads(json.dumps(trainer.state))
    assert saved['bank_tensors'] and saved['bank_optimizer']
    assert any(not torch.equal(initial[k], v) for k, v in _bank_tensor_map(trainer.bank_leaves).items())
    trainer.step()
    expected = {k: v.detach().clone() for k, v in _bank_tensor_map(trainer.bank_leaves).items()}
    resumed = OuterTrainer(engine, bank, recipe, [target], state=saved)
    resumed.step()
    for name, tensor in _bank_tensor_map(resumed.bank_leaves).items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert resumed.state['bank_optimizer'] == trainer.state['bank_optimizer']
    assert resumed.state['writer'] == trainer.state['writer']
    assert resumed.state['head'] == trainer.state['head']
    assert resumed.state['exposure'] == trainer.state['exposure']
    engine.store.close()
