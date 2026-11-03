"""Continuous bank meta-learning (tune/learned/meta_bank.py) on the tiny native fixture: pool splits, exact
publication invariants, first-order outer steps into the bank and basis, resume, held-family arms and the gate."""
import json

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import rows_for
from natlang_neuralese.train import adaptation_stages
from natlang_neuralese.tune.learned import meta_bank as MB
from natlang_neuralese.tune.learned.bank import load_bank


def pool_rows(engine, families=6, worlds=6):
    rows = []
    for f in range(families):
        for w, row in enumerate(rows_for(engine, f'fam{f}', worlds, offset=f)):
            row['graph']['nodes'][0]['token_ids'][3] = 10 + w  # every world supervises a distinct target
            rows.append({'family': f'fam{f}', 'world': f'w{w}', 'graph': row['graph'], 'actor': {'kind': 'fixture'}})
    return rows


def small_recipe(**overrides):
    return MB.MetaRecipe(**{'rank': 4, 'components': 3, 'component_rank': 2, 'inner_steps': 2, 'init': 'random',
                            'layer_groups': 2, 'residual_rank': 1, 'meta_batch': 2, 'lora_rank': 2,
                            'outer_lr_bank': 5e-2, 'outer_lr_basis': 1e-2, **overrides})


def test_pool_holds_out_whole_families_and_splits_their_worlds(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    pool = MB.build_pool(pool_rows(engine, families=12), held_fraction=.4, seed=3)
    roles = {f['role'] for f in pool['families'].values()}
    assert roles == {'held', 'meta-train'}
    for family in pool['families'].values():
        splits = [w['split'] for w in family['worlds']]
        if family['role'] == 'held':
            assert splits == ['train'] * 3 + ['dev'] * 2 + ['confirmation']
        else:
            assert set(splits) == {'train'}
    receipt = MB.write_pool(pool, tmp_path / 'pool')
    again = MB.read_pool(tmp_path / 'pool')
    assert receipt['held'] + receipt['meta-train'] == len(again['families']) == 12
    assert again['families'] == json.loads(json.dumps(pool['families']))
    engine.store.close()


def test_invariants_keep_composed_updates_and_unit_energy(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    learner = MB.BankMetaLearner(engine, MB.build_pool(pool_rows(engine)), small_recipe(), tmp_path / 'op', parent_sha256='p')
    bank, leaves = learner.bank0, learner.leaves
    with torch.no_grad():  # perturb off the invariants as an optimizer step would
        for value in leaves['basis'].values():
            value.add_(.05 * torch.randn_like(value))
        for factors in leaves['factors_by_component'].values():
            for P, Q in factors.values():
                P.mul_(1.3)
    def dense(k):
        factors, basis = MB._detached(leaves)
        one = torch.zeros(3, len(bank.group_labels))
        one[k] = 1.
        cores = bank.composed_cores([0, 1, 2], one, factors=factors)
        return {name: basis['left.' + name] @ cores[name] @ basis['right.' + name].transpose(-2, -1) for name in cores}
    before = [dense(k) for k in range(3)]
    scales = MB.restore_invariants(bank, leaves, basis=True)
    # Orthonormalization re-expresses factors exactly; renormalization scales each component's update by its scale.
    for k in range(3):
        for name, value in dense(k).items():
            torch.testing.assert_close(value, scales[k] * before[k][name], atol=1e-5, rtol=1e-4)
    factors, basis = MB._detached(leaves)
    for value in basis.values():
        gram = value.transpose(-2, -1) @ value
        torch.testing.assert_close(gram, torch.eye(gram.shape[-1]).expand_as(gram), atol=1e-5, rtol=0)
    for k in range(3):
        energy = sum(float(bank.q[n] * (P @ Q.transpose(-2, -1)).pow(2).sum()) for n, (P, Q) in factors[k].items())
        assert abs(energy - 1) < 1e-4
    engine.store.close()


def test_outer_steps_train_bank_and_basis_resume_exactly_and_evaluate_four_arms(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    pool = MB.build_pool(pool_rows(engine, families=10), held_fraction=.5, seed=1)
    assert {f['role'] for f in pool['families'].values()} == {'held', 'meta-train'}
    recipe = small_recipe()
    learner = MB.BankMetaLearner(engine, pool, recipe, tmp_path / 'op', parent_sha256='p')
    factors0, basis0 = MB._detached(learner.leaves)
    start = ({k: {n: (P.clone(), Q.clone()) for n, (P, Q) in v.items()} for k, v in factors0.items()},
             {k: v.clone() for k, v in basis0.items()})
    first = learner.step()
    assert first['grad_norm']['bank'] > 0 and first['grad_norm']['basis'] > 0 and first['update_norm'] > 0
    assert first['grad_norm']['meta'] > 0 and first['regression_unexplained'] is None  # random init: no LoRA fits
    assert len(first['families']) == 2 and all(len(set(r['worlds'])) == 5 for r in first['families'])
    moved = MB._detached(learner.leaves)
    assert any(not torch.equal(start[1][k], moved[1][k]) for k in start[1])
    assert any(not torch.equal(start[0][0][n][0], moved[0][0][n][0]) for n in start[0][0])
    learner.save()
    second = learner.step()
    learner.save()
    # Resume replays the saved bank, optimizer and episode RNG: the next step is identical.
    expected = learner.step()
    resumed = MB.BankMetaLearner(engine, pool, recipe, tmp_path / 'op', parent_sha256='p')
    assert resumed.state['step'] == 2
    replay = resumed.step()
    assert [r['family'] for r in replay['families']] == [r['family'] for r in expected['families']]
    assert abs(replay['families'][0]['query_fitted'] - expected['families'][0]['query_fitted']) < 1e-5
    evaluation = learner.evaluate('dev')
    assert evaluation['families'] and set(evaluation['mean']) == set(MB.ARMS) == set(learner.arms)
    assert all(0 <= f['similarity'] <= 1 for f in evaluation['families'].values()) and 'mean_distinct' in evaluation
    parent = evaluation['mean']['parent']
    assert parent is not None and set(evaluation['meta_wins']) == {'initial', 'lora'}
    # Each finished family is saved: a preempted evaluation resumes from the saved families.
    interrupted = MB.BankMetaLearner(engine, pool, recipe, tmp_path / 'op', parent_sha256='p')
    saved = interrupted.state['partial_evaluations'][f"dev:{learner.state['step']}"]
    assert set(saved) == set(evaluation['families'])
    assert interrupted.evaluate('dev')['mean'] == evaluation['mean']
    manifest = learner.publish()
    bank = load_bank(engine, manifest)
    assert bank.parent == learner.bank0.snapshot and len(bank.components) == 3
    # A new promoted parent: the live bank continues from its saved tensors with a recorded transition.
    follow = MB.BankMetaLearner(engine, pool, recipe, tmp_path / 'op', parent_sha256='q')
    assert follow.state['parents'][-1]['sha256'] == 'q' and follow.state['step'] == learner.state['step']
    engine.store.close()


def test_pca_initialization_spans_the_meta_train_lora_fits_and_trains_with_the_regression(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    pool = MB.build_pool(pool_rows(engine, families=10), held_fraction=.5, seed=1)
    recipe = small_recipe(init='pca', components=8, variance=.95, aux_families=3)
    names = MB.select_targets(engine, recipe)
    fits = MB.family_lora_fits(engine, pool, recipe, names, tmp_path / 'fits')
    meta_train = {n for n, f in pool['families'].items() if f['role'] == 'meta-train'}
    assert set(fits) == meta_train and sorted(p.stem for p in (tmp_path / 'fits').glob('*.pt')) == sorted(meta_train)
    assert MB.family_lora_fits(engine, pool, recipe, names, tmp_path / 'fits').keys() == fits.keys()  # resumes
    learner = MB.BankMetaLearner(engine, pool, recipe, tmp_path / 'op', parent_sha256='p', fits=fits)
    spectrum = learner.state['spectrum']
    assert 1 <= learner.state['components'] <= min(8, len(fits))
    assert spectrum['explained'] >= .95 or learner.state['components'] == min(8, len(fits))
    initial = float(learner.regression(sorted(fits)))
    assert 0 <= initial <= 1
    # The un-meta-trained PCA bank's initial coefficients already reproduce part of the average update.
    assert float(learner.meta['init'].abs().sum()) > 0
    entry = learner.step()
    assert entry['regression_unexplained'] is not None and entry['grad_norm']['meta'] > 0
    # Renormalization rescales coefficients and steps with the components: composed updates are unchanged.
    with torch.no_grad():  # an off-invariant component, as after an optimizer step
        for f in learner.leaves['factors_by_component'].values():
            for P, Q in f.values():
                P.mul_(1.7)
    factors, _ = MB._detached(learner.leaves)
    slots = list(range(len(learner.bank0.components)))
    before = learner.bank0.composed_cores(slots, learner.meta['init'].detach().clone(), factors=factors)
    before = {name: value.clone() for name, value in before.items()}
    learner._rescale_meta(MB.restore_invariants(learner.bank0, learner.leaves, basis=False))
    factors, _ = MB._detached(learner.leaves)
    after = learner.bank0.composed_cores(slots, learner.meta['init'].detach(), factors=factors)
    for name in before:
        torch.testing.assert_close(after[name], before[name], atol=1e-5, rtol=1e-4)
    engine.store.close()


def test_promotion_needs_all_arms_and_a_strict_win():
    good = {'families': {'a': {}}, 'mean': {'parent': 2., 'meta': 1., 'initial': 1.5, 'lora': 1.2}}
    assert MB.promotion(good)['met']
    assert not MB.promotion({**good, 'mean': {**good['mean'], 'lora': .9}})['met']
    assert not MB.promotion({'families': {}, 'mean': {}})['met']


def test_stage_kind_validates_and_gates():
    adaptation_stages.validate_parameters('bank_metalearning', {'outer_steps': 10, 'train_bank': True})
    with pytest.raises(ValueError):
        adaptation_stages.validate_parameters('bank_metalearning', {'train_basis': False})
    report = {'schema': adaptation_stages.BANK_METALEARNING_REPORT, 'completed': True,
              'production_exposure_advanced': False,
              'outer_training': {'steps': 3, 'train_bank': True, 'train_basis': True},
              'held': {'met': True}, 'confirmation': {'met': True}}
    adaptation_stages.require_gate(report, 'bank_metalearning')
    with pytest.raises(ValueError):
        adaptation_stages.require_gate({**report, 'confirmation': {'met': False}}, 'bank_metalearning')


def test_whole_trajectory_sources_are_reduced_to_their_declared_assistant_targets():
    graph = {'id': 'g', 'conversion': {'supervision_policy': {'assistant_target_tokens': 3}},
             'nodes': [{'token_ids': [1, 2, 3, 4, 5], 'target_weights': [.05, 28.9, .25, 9.1, 1.25]}]}
    out = MB.assistant_targets(graph)
    assert out['nodes'][0]['target_weights'] == [0., 1., 0., 1., 1.] and out['supervision'] == 'assistant-output-only-v1'
    graph['conversion']['supervision_policy']['assistant_target_tokens'] = 4
    assert MB.assistant_targets(graph) is None  # a count that disagrees with the declaration is refused
    plain = {'id': 'h', 'nodes': [{'token_ids': [1], 'target_weights': [.5]}]}
    assert MB.assistant_targets(plain) is plain


def test_pool_drops_worlds_that_repeat_an_earlier_worlds_targets(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    rows = pool_rows(engine, families=1, worlds=8)
    duplicate = json.loads(json.dumps(rows[0]))
    duplicate['world'], duplicate['graph']['id'] = 'w9', 'dup'
    pool = MB.build_pool(rows + [duplicate], held_fraction=.01, seed=0)
    worlds = [w['world'] for f in pool['families'].values() for w in f['worlds']]
    assert 'w9' not in worlds and len(worlds) == len(set(worlds))
    # The cheap index pass (graph ids plus target digests) chooses exactly the families and worlds of the full pass.
    index = [{**row, 'graph': {'id': row['graph']['id'], 'target_digest': MB.target_digest(row['graph'])}}
             for row in rows + [duplicate]]
    chosen = MB.build_pool(index, held_fraction=.01, seed=0)
    assert [w['world'] for f in chosen['families'].values() for w in f['worlds']] == worlds
    engine.store.close()
