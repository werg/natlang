"""The nested family as an in-stage objective and the objective continuation of an active stage (owner 2026-10-12,
REVIEW_20261012 rec. 4): members route over the N0 order by index, the family is drawn per update like a precision
point, and a running neuralese_direct stage takes a new curriculum and the family without a jump."""
import copy
import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.maple.family import Family, stage_family, validate_family
from natlang_neuralese.maple.model import MapleConfig, MapleForCausalLM
from natlang_neuralese.train import direct_sft as trainer
from natlang_neuralese.train import recipe as runner

RECIPES = Path(__file__).parents[2] / 'training' / 'neuralese' / 'recipes'


def tiny(experts=8, layers=2):
    torch.manual_seed(0)
    config = MapleConfig(vocab_size=64, hidden_size=32, num_hidden_layers=layers, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, moe_intermediate_size=16, num_experts=experts,
                         num_experts_per_tok=2, layer_types=['full_attention'] * layers, dense_experts=True,
                         model_type='mellum', swiglu_clamp=None, partial_rotary_factor=1.0)
    model = MapleForCausalLM(config)
    for tensor in list(model.parameters()) + [b for b in model.buffers() if b.is_floating_point()]:
        torch.nn.init.normal_(tensor, std=0.2)  # dense experts are buffers until made latent
    return model.eval()


def test_a_member_of_an_unpermuted_model_equals_the_prefix_member_of_the_permuted_model():
    model = tiny()
    orders = [torch.randperm(8, generator=torch.Generator().manual_seed(layer)) for layer in range(2)]
    ids = torch.randint(0, 64, (1, 12))
    model.set_member_order(orders)
    model.set_member('2x4', experts=4)
    by_index = model(ids).logits
    model.set_member(None)
    full = model(ids).logits
    permuted = copy.deepcopy(model)
    permuted.set_member_order(None)
    permuted.order_experts(orders)
    assert torch.allclose(permuted(ids).logits, full, atol=1e-5)  # ordering is function-preserving
    permuted.set_member('2x4', experts=4)
    assert torch.allclose(permuted(ids).logits, by_index, atol=1e-5)
    assert not torch.allclose(by_index, full, atol=1e-3)
    # The member's routing only ever selects its experts, by their own ids.
    model.set_member('2x4', experts=4)
    index, _ = model.model.layers[0].mlp.route(torch.randn(5, 32))
    assert set(index.flatten().tolist()) <= set(orders[0][:4].tolist())
    model.set_member(None)
    with pytest.raises(ValueError, match='permutation'):
        model.set_member_order([torch.zeros(8, dtype=torch.long)] * 2)


def family_spec(**overrides):
    return {'schema': 'natlang.neuralese-family/1', 'order': {'path': 'order.pt', 'sha256': '0' * 64},
            'members': {'2x4': 0.125, '2x2': 0.125}, 'evaluate': ['2x4', '2x3', '2x2'], 'kl_weight': 1.0,
            'ramp': {'start': 0.03, 'end': 0.08}, 'stages': ['sft'], **overrides}


def test_the_family_is_drawn_per_update_by_weight_and_ramps_in_from_zero(tmp_path):
    validate_family(family_spec(), {'sft': 'neuralese_direct'})
    with pytest.raises(ValueError, match='neuralese_direct'):
        validate_family(family_spec(), {'sft': 'core_text_warmup'})
    with pytest.raises(ValueError, match='LxE'):
        validate_family(family_spec(members={'wide': 1.0}))
    with pytest.raises(ValueError, match='order'):
        validate_family(family_spec(order=None))
    validate_family(family_spec(order=None, members={'1xall': 0.2}, evaluate=['1xall']))  # dense: depth only
    family = Family(stage_family(family_spec(), 'sft'))
    assert stage_family(family_spec(), 'other') is None
    assert all(family.sampled(step, 0.02) is None for step in range(200))  # before the ramp: the full model only
    draws = [family.sampled(step, 0.5) for step in range(4000)]
    share = sum(d is not None for d in draws) / len(draws)
    assert share == pytest.approx(0.25 / 1.25, abs=0.03) and {'2x4', '2x2'} <= set(draws)
    assert draws == [family.sampled(step, 0.5) for step in range(4000)]  # deterministic per (seed, stage, step)
    from natlang_neuralese.maple.maple_port import MaplePortBackbone
    from natlang_neuralese.model.lfm2_port import ControlTokens
    model = tiny()
    backbone = MaplePortBackbone(model, controls=ControlTokens(open_id=1, close_id=2))
    order = tmp_path / 'order.pt'
    torch.save({'orders': [torch.randperm(8) for _ in range(2)]}, order)
    pinned = {'path': str(order), 'sha256': runner.sha(order)}
    family = Family(stage_family(family_spec(order=pinned, members={'2x4': 0.1, '1x2': 0.1}), 'sft'))
    assert family.install(backbone)['members'] == ['2x4', '1x2', '2x3', '2x2']
    ids = torch.randint(3, 64, (1, 9))  # no control tokens (the port embeds those from its control rows)
    full = backbone.forward_ids(ids, logits=False)['h_final']
    with family.selected(backbone, '1x2'):
        assert model.model.layers[1].mlp.member_experts.tolist() == model.member_orders[1][:2].tolist()
        early = backbone.forward_ids(ids, logits=False)['h_final']
    model.set_member('1x2', experts=2, layers=1)
    reference = model.model(input_ids=ids).last_hidden_state  # MapleModel's own early exit (final-normed)
    model.set_member(None)
    assert torch.allclose(backbone.final_norm(early), reference, atol=1e-5)
    assert model.model.layers[1].mlp.active_experts is None and backbone.exit_layers is None
    assert torch.allclose(backbone.forward_ids(ids, logits=False)['h_final'], full)
    with pytest.raises(ValueError, match='proper member'):
        Family(stage_family(family_spec(order=pinned, members={'2xall': 0.1}), 'sft')).install(backbone)
    stale = Family(stage_family(family_spec(order={'path': str(order), 'sha256': '1' * 64}), 'sft'))
    with pytest.raises(ValueError, match='changed'):
        stale.install(backbone)


def quantization(progress=(0.0, 0.35), q4=(0.15, 0.3)):
    return {'schema': 'natlang.neuralese-quantization/1', 'loss': {'mode': 'sample', 'bf16_weight': 1.0, 'seed': 0},
            'stage_progress': {'sft': list(progress)},
            'points': [{'name': 'q4', 'gate': 'required', 'weight': 0.5, 'ramp': {'start': q4[0], 'end': q4[1]},
                        'modules': {'experts': {'format': 'int4', 'group': 32}}},
                       {'name': 'ternary-experts', 'gate': 'required', 'weight': 1.0, 'ramp': {'start': 0.3, 'end': 0.7},
                        'modules': {'experts': {'format': 'ternary'}}}]}


def option(component, progress):
    return json.dumps({**component, 'stage': {'id': 'sft', 'progress': list(progress), 'start': 0, 'end': 'derived'}},
                      sort_keys=True, separators=(',', ':'))


def test_an_objective_continuation_rebases_the_curriculum_without_a_jump():
    old_q, new_q = quantization(), quantization(progress=(0.0, 0.75))
    old = {'quantization': option(old_q, (0.0, 0.35))}
    new = {'quantization': option(new_q, (0.0, 0.75)), 'family': json.dumps(stage_family(family_spec(), 'sft'))}
    declared = {'stage': 'sft', 'previous': {'quantization': canonical_json_sha256_hex(old_q), 'family': None}}
    # v4: a data continuation at update 400 already rebased the curriculum to [p_400, 0.35].
    continuation = {'quantization_rebase': {'start': 400, 'end': 31314, 'progress': [0.008, 0.35]}}
    with pytest.raises(ValueError, match='declared objective continuation'):
        trainer.objective_continuation(old, new, None, step=1000, steps=31314, continuation=continuation,
                                       previous_record=None)
    record = trainer.objective_continuation(old, new, json.dumps(declared), step=1000, steps=31314,
                                            continuation=continuation, previous_record=None)
    p_k = 0.008 + (0.35 - 0.008) * 600 / (31314 - 400)
    assert record['progress'] == [pytest.approx(p_k), 0.75] and record['start'] == 1000
    assert record['changed'] == ['quantization', 'family'] and record['family_weights_at_step'] == {'2x4': 0.0, '2x2': 0.0}
    stage = trainer.effective_quantization_stage(json.loads(new['quantization']), 31314, continuation, record)
    from natlang_neuralese.train.quantization import Quantization
    quant = Quantization({**new_q, 'stage': stage})
    assert quant.progress(1000) == pytest.approx(p_k) and quant.progress(31314) == pytest.approx(0.75)
    assert quant.mixes(31314)['ternary-experts'] == 1.0  # ternary experts complete inside the stage
    # Nothing changed: no record. A later resume reapplies the stored one.
    assert trainer.objective_continuation(new, new, None, step=2000, steps=31314, continuation=continuation,
                                          previous_record=record) is None
    # A moved ramp that changes λ at the handoff update, a family already ramped in, or a wrong pin: refused.
    moved = {**new, 'quantization': option(quantization(progress=(0.0, 0.75), q4=(0.0, 0.01)), (0.0, 0.75))}
    with pytest.raises(ValueError, match='jumps'):
        trainer.objective_continuation(old, moved, declared, step=1000, steps=31314, continuation=continuation,
                                       previous_record=None)
    hot = {**new, 'family': json.dumps(stage_family(family_spec(ramp={'start': 0.0, 'end': 0.01}), 'sft'))}
    with pytest.raises(ValueError, match='family:2x4'):
        trainer.objective_continuation(old, hot, declared, step=1000, steps=31314, continuation=continuation,
                                       previous_record=None)
    wrong = {'stage': 'sft', 'previous': {'quantization': '0' * 64, 'family': None}}
    with pytest.raises(ValueError, match='not the declared previous'):
        trainer.objective_continuation(old, new, wrong, step=1000, steps=31314, continuation=continuation,
                                       previous_record=None)
    assert 'objective_continuation' in trainer.OPERATIONAL and trainer.LATER_DEFAULTS['family'] is None


def _plan(tmp_path, recipe):
    return {'recipe': recipe, 'recipe_sha256': 'new', 'inputs': {'/heads': 'H'}, 'device': 'cuda'}


def test_the_handoff_takes_objective_changes_only_by_a_declaration_pinning_the_running_components(tmp_path):
    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('x = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('x = 2\n')
    (out / 'sft').mkdir()
    stages = [{'id': 'sft', 'kind': 'neuralese_direct', 'parameters': {'epochs': 1.0}},
              {'id': 'traj', 'kind': 'neuralese_direct', 'parameters': {'epochs': 1.0}}]
    old_q = quantization()
    old_q['stage_progress'] = {'sft': [0.0, 0.35], 'traj': [0.35, 0.5]}
    old = {'id': 'r', 'stages': stages, 'quantization': old_q}
    new_q = copy.deepcopy(old_q)
    new_q['stage_progress'] = {'sft': [0.0, 0.75], 'traj': [0.75, 1.0]}
    new = {'id': 'r', 'stages': stages, 'quantization': new_q, 'family': family_spec(stages=['sft', 'traj'])}
    existing = {'recipe': old, 'recipe_sha256': 'lineage', 'inputs': {'/heads': 'H'}, 'device': 'cuda',
                'code': {'a.py': runner.sha(frozen / 'a.py')}}
    plan_path = out / 'recipe-plan.json'
    plan_path.write_text(json.dumps(existing))
    with pytest.raises(ValueError, match='without a declared objective_continuation'):
        runner.lineage_handoff(copy.deepcopy(existing), _plan(tmp_path, new), frozen, package, plan_path, out,
                               dry_run=True)
    declared = {'stage': 'sft', 'previous': {'quantization': runner.objective_component_sha256(old_q), 'family': None}}
    with pytest.raises(ValueError, match='not the running'):
        bad = dict(new, objective_continuation={'stage': 'sft', 'previous': {'quantization': '0' * 64,
                                                                             'family': None}})
        runner.lineage_handoff(copy.deepcopy(existing), _plan(tmp_path, bad), frozen, package, plan_path, out,
                               dry_run=True)
    with pytest.raises(ValueError, match='not the active stage'):
        runner.lineage_handoff(copy.deepcopy(existing), _plan(tmp_path, dict(new, objective_continuation=dict(
            declared, stage='traj'))), frozen, package, plan_path, out, dry_run=True)
    updated = runner.lineage_handoff(copy.deepcopy(existing), _plan(tmp_path, dict(new, objective_continuation=declared)),
                                     frozen, package, plan_path, out, dry_run=True)
    assert updated['handoffs'][-1]['objective_continuation']['changed'] == ['quantization', 'family']
    # The trainer receives the family and the declaration on the declared stage only.
    args = runner.stage_objective_args(dict(new, objective_continuation=declared), stages[0])
    assert json.loads(args[args.index('--family') + 1])['stage'] == 'sft'
    assert json.loads(args[args.index('--objective-continuation') + 1])['previous'] == declared['previous']
    assert '--objective-continuation' not in runner.stage_objective_args(new, stages[1])
    # Any other change still refuses.
    other = dict(new, objective_continuation=declared, stages=[dict(stages[0], parameters={'epochs': 1.0, 'lr': 1}),
                                                               stages[1]])
    with pytest.raises(ValueError, match='more than operational'):
        runner.lineage_handoff(copy.deepcopy(existing), _plan(tmp_path, other), frozen, package, plan_path, out,
                               dry_run=True)


def test_the_mellum_lineage_declares_the_continuation_of_its_running_quantization_component():
    lineage = runner.load_recipe(RECIPES / 'neuralese-direct-mellum-v4-from-v3-warmup.json')
    assert lineage['quantization']['stage_progress'] == {'direct_sft': [0.0, 0.75], 'direct_trajectory': [0.75, 1.0]}
    assert lineage['objective_continuation']['stage'] == 'direct_sft'
    assert set(lineage['family']['members']) == {'28x32', '28x16', '21x32'}
    lfm = runner.load_recipe(RECIPES / 'neuralese-direct-lfm-v3.json')
    assert [p['name'] for p in lfm['quantization']['points']] == ['q4'] and set(lfm['family']['members']) == {
        '12xall', '8xall'}
    shared = ('kl_weight', 'ramp', 'stages', 'seed')
    assert {k: lfm['family'][k] for k in shared} == {k: lineage['family'][k] for k in shared}
    assert sum(lfm['family']['members'].values()) == pytest.approx(sum(lineage['family']['members'].values()))
    lq, mq = lfm['quantization']['points'][0], lineage['quantization']['points'][0]
    assert {k: v for k, v in lq.items() if k != 'ramp'} == {k: v for k, v in mq.items() if k != 'ramp'}
    assert lq['ramp'] == {'start': 0.25, 'end': 0.3}  # QAD base weights: later, shorter ramp
    assert lfm['backbone']['model'] == 'LiquidAI/LFM2.5-350M-GGUF/qad' and 'backbone' in lineage
    from natlang_neuralese.train import line_parity
    assert line_parity.check() == []


def test_a_handoff_ignores_the_fresh_lineage_backbone_source():
    a = {'id': 'a', 'stages': [], 'backbone': {'model': 'x'}}
    assert runner._without_operational(a) == runner._without_operational({'id': 'b', 'stages': []})
