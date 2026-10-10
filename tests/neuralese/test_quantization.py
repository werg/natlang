"""The recipe quantization component (train/quantization.py): QAT woven into the training stages (owner 2026-10-10)."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from natlang_neuralese.maple import ternary
from natlang_neuralese.maple.ternary import PRECISION, active_precision, precision_value, q4_0, quantized_value
from natlang_neuralese.train import quantization as q

RECIPES = Path(__file__).resolve().parents[2] / 'training' / 'neuralese' / 'recipes'


def spec(**loss):
    return {'schema': q.SCHEMA, 'loss': {'mode': 'sum', **loss},
            'stage_progress': {'warm': [0.0, 0.5], 'rec': [0.5, 1.0]},
            'points': [
                {'name': 'q4', 'gate': 'required', 'weight': 0.5, 'ramp': {'start': 0.2, 'end': 0.4},
                 'modules': {'mlp': {'format': 'int4', 'group': 32}, 'attention': {'format': 'int4'}}},
                {'name': 'tern', 'gate': 'report', 'weight': 1.0, 'ramp': {'start': 0.4, 'end': 0.8},
                 'modules': {'mlp': {'format': 'ternary', 'nested_group': 32}}}]}


def component(stage='warm', start=0, end=100, **loss):
    return q.stage_component(spec(**loss), stage, start=start, end=end)


def q4_0_reference(row, group=32):
    """llama.cpp quantize_row_q4_0_ref + dequantize_row_q4_0, written out per block (scalar loop)."""
    out = []
    for b in range(0, len(row), group):
        block = [float(x) for x in row[b:b + group]]
        amax, mx = 0.0, 0.0
        for v in block:
            if abs(v) > amax:
                amax, mx = abs(v), v
        d = mx / -8
        idv = 1.0 / d if d else 0.0
        stored = float(torch.tensor(d, dtype=torch.float16))
        for v in block:
            code = min(15, int(torch.floor(torch.tensor(v * idv + 8.5, dtype=torch.float32))))
            out.append((code - 8) * stored)
    return torch.tensor(out)


# Schedule ---------------------------------------------------------------------------------------------------------
def test_progress_spans_the_stage_range_and_ramps_are_linear_and_clipped():
    run = q.Quantization(component())
    assert run.progress(0) == 0.0 and run.progress(50) == pytest.approx(0.25) and run.progress(100) == 0.5
    assert run.progress(500) == 0.5  # clipped at the stage end
    assert q.ramp_mix({'start': 0.2, 'end': 0.4}, 0.1) == 0.0
    assert q.ramp_mix({'start': 0.2, 'end': 0.4}, 0.3) == pytest.approx(0.5)
    assert q.ramp_mix({'start': 0.2, 'end': 0.4}, 0.9) == 1.0
    assert run.mixes(60) == {'q4': pytest.approx(0.5), 'tern': 0.0}
    later = q.Quantization(component('rec', start=1000, end=1100))  # an update range not starting at 0 (AR fixup)
    assert later.progress(1000) == 0.5 and later.progress(1050) == pytest.approx(0.75)
    assert later.mixes(1050) == {'q4': 1.0, 'tern': pytest.approx(0.875)}


def test_required_points_are_those_whose_ramp_ends_inside_the_stage():
    assert q.Quantization(component('warm')).required_points() == ['q4']  # q4 ends at 0.4 <= 0.5
    assert q.Quantization(component('rec')).required_points() == ['q4']
    assert q.Quantization(component('warm')).gated_points() == ['q4', 'tern']
    assert q.stage_component(spec(), 'other', start=0, end=10) is None  # a stage not named trains at BF16


def test_sum_plan_lists_bf16_and_active_points_with_their_weights():
    run = q.Quantization(component(bf16_weight=1.0))
    assert run.plan(10) == [('bf16', 0.0, 1.0)]  # progress 0.05: nothing active yet
    assert run.plan(60) == [('bf16', 0.0, 1.0), ('q4', pytest.approx(0.5), 0.5)]


def test_sample_plan_draws_one_pass_by_weight_deterministically_scaled_by_the_total():
    run = q.Quantization(component(stage='rec', mode='sample', bf16_weight=1.0))
    draws = [run.plan(step) for step in range(100)]
    assert all(len(d) == 1 and d[0][2] == pytest.approx(2.5) for d in draws)  # weights 1 + 0.5 + 1
    assert draws == [run.plan(step) for step in range(100)]  # a function of (seed, stage, step): resumable
    counts = {}
    for step in range(2000):
        counts[run.sampled(step)[0]] = counts.get(run.sampled(step)[0], 0) + 1
    assert counts['bf16'] / 2000 == pytest.approx(0.4, abs=0.05)
    assert counts['q4'] / 2000 == pytest.approx(0.2, abs=0.05)


# Quantizers -------------------------------------------------------------------------------------------------------
def test_q4_0_is_llama_cpp_bit_exact_and_on_its_grid():
    torch.manual_seed(0)
    weight = torch.randn(3, 64) * 0.05
    expected = torch.stack([q4_0_reference(row) for row in weight])
    assert torch.equal(q4_0(weight), expected)
    blocks = q4_0(weight).reshape(3, 2, 32)
    scale = weight.reshape(3, 2, 32).gather(-1, weight.reshape(3, 2, 32).abs().argmax(-1, keepdim=True)) / -8
    codes = blocks / scale.half().float()
    assert torch.allclose(codes, codes.round()) and codes.min() >= -8 and codes.max() <= 7
    with pytest.raises(ValueError, match='multiple of the Q4_0 block'):
        q4_0(torch.randn(2, 40))


def test_nested_ternary_is_the_ternary_rule_on_the_int4_values():
    torch.manual_seed(1)
    weight = torch.randn(4, 64)
    nested = quantized_value(weight, 'ternary', {'nested_group': 32})
    assert torch.equal(nested, ternary.ternarize(q4_0(weight)).float())
    # A function of the int4 values only: weights with the same Q4_0 values give the same ternary weights.
    nudged = weight + (q4_0(weight) - weight) * 0.001
    assert torch.equal(q4_0(nudged), q4_0(weight))
    assert torch.equal(quantized_value(nudged, 'ternary', {'nested_group': 32}), nested)
    assert torch.equal(quantized_value(weight, 'ternary'), ternary.ternarize(weight).float())


def test_precision_value_ramps_straight_through_and_respects_groups_and_layers():
    torch.manual_seed(2)
    latent = torch.randn(4, 32, requires_grad=True)
    assert precision_value(latent, 'mlp') is latent  # no active point
    target = ternary.ternarize(latent.detach())
    with active_precision({'mlp': ('ternary', 0.25, {'layers': {3}})}, 'p'):
        assert precision_value(latent, 'attention') is latent  # another group
        assert precision_value(latent, 'mlp@1') is latent  # a layer the point excludes
        value = precision_value(latent, 'mlp@3')
    assert PRECISION['groups'] is None  # restored
    assert torch.allclose(value, latent.detach() + 0.25 * (target - latent.detach()), atol=1e-7)
    value.sum().backward()
    assert torch.equal(latent.grad, torch.ones_like(latent))  # identity gradient to the latent


# Selection and installation -----------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(32, 32, bias=False)
        self.feed_forward = nn.Module()
        self.feed_forward.w1 = nn.Linear(32, 64, bias=False)
        self.conv = nn.Module()
        self.conv.in_proj = nn.Linear(32, 96, bias=False)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 32)
        self.model.layers = nn.ModuleList([Block(), Block()])
        self.lm_head = nn.Linear(32, 16, bias=False)


def test_classify_names_the_module_groups_of_both_lines():
    from natlang_neuralese.maple.model import DenseExperts
    assert q.classify('model.layers.0.self_attn.q_proj', nn.Linear(2, 2)) == 'attention'
    assert q.classify('model.layers.0.self_attn.out_proj', nn.Linear(2, 2)) == 'attention'
    assert q.classify('model.layers.0.conv.in_proj', nn.Linear(2, 2)) == 'conv'
    assert q.classify('model.layers.0.feed_forward.w2', nn.Linear(2, 2)) == 'mlp'
    assert q.classify('model.layers.0.mlp.gate', nn.Linear(2, 2)) == 'router'
    assert q.classify('model.layers.0.mlp.experts', DenseExperts(2, 4, 4)) == 'experts'
    assert q.classify('model.embed_tokens', nn.Embedding(2, 2)) == 'embeddings'
    assert q.classify('lm_head', nn.Linear(2, 2)) == 'heads'
    assert q.classify('model.norm', nn.LayerNorm(2)) is None


def test_install_routes_selected_weights_through_the_active_point_and_keeps_latents():
    torch.manual_seed(3)
    model = Tiny()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    run = q.Quantization(component())
    installed = run.install(model, layer_count=2)
    assert installed == {'attention': 2, 'mlp': 2}  # conv/embeddings/heads not selected by any point
    w1 = model.model.layers[0].feed_forward.w1
    assert torch.equal(w1.weight, before['model.layers.0.feed_forward.w1.weight'])  # BF16 outside any point
    with run.context('q4', 1.0):
        assert torch.equal(w1.weight, q4_0(before['model.layers.0.feed_forward.w1.weight']))
        assert torch.equal(model.model.layers[0].conv.in_proj.weight, before['model.layers.0.conv.in_proj.weight'])
    with run.context('tern', 1.0):
        assert torch.equal(model.model.layers[0].self_attn.q_proj.weight,
                           before['model.layers.0.self_attn.q_proj.weight'])  # tern selects mlp only
    # The latent is the same trainable parameter, saved and restored under its plain name.
    from natlang_neuralese.train.backbone_policy import plain_named_tensors
    plain = plain_named_tensors(model)
    assert plain['model.layers.0.feed_forward.w1.weight'] is w1.parametrizations.weight.original


def test_layer_selection_quantizes_only_the_chosen_layers(tmp_path):
    probe = tmp_path / 'probe.json'
    probe.write_text(json.dumps({'layers': {'0': 5.0, '1': 0.1}}))
    import hashlib
    point = {'name': 'p', 'ramp': {'start': 0.0, 'end': 0.1}, 'modules': {'mlp': {'format': 'int4'}},
             'layers': {'least_sensitive': 0.5, 'probe': {'path': str(probe),
                                                          'sha256': hashlib.sha256(probe.read_bytes()).hexdigest()}}}
    run = q.Quantization({'schema': q.SCHEMA, 'points': [point],
                          'stage': {'id': 's', 'progress': [0.0, 1.0], 'start': 0, 'end': 10}})
    model = Tiny()
    run.install(model, layer_count=2)
    assert run.layers['p'] == {1}  # the less sensitive layer
    with run.context('p', 1.0):
        first = model.model.layers[0].feed_forward.w1
        second = model.model.layers[1].feed_forward.w1
        assert torch.equal(first.weight, first.parametrizations.weight.original)
        assert torch.equal(second.weight, q4_0(second.parametrizations.weight.original))
    probe.write_text('{"layers": {}}')
    with pytest.raises(ValueError, match='sensitivity probe changed'):
        q.selected_layers(point, 2)


def test_dense_experts_follow_the_precision_point():
    from natlang_neuralese.maple.model import DenseExperts
    torch.manual_seed(4)
    experts = DenseExperts(2, 32, 32)
    with torch.no_grad():
        experts.gate_up.normal_(0, 0.05)
        experts.down.normal_(0, 0.05)
    holder = nn.Module()
    holder.model = nn.Module()
    holder.model.layers = nn.ModuleList([nn.Module()])
    holder.model.layers[0].mlp = nn.Module()
    holder.model.layers[0].mlp.experts = experts
    run = q.Quantization({'schema': q.SCHEMA, 'points': [
        {'name': 't', 'ramp': {'start': 0.0, 'end': 0.5}, 'modules': {'experts': {'format': 'ternary'}}}],
        'stage': {'id': 's', 'progress': [0.0, 1.0], 'start': 0, 'end': 10}})
    run.install(holder, layer_count=1)
    assert isinstance(experts.gate_up, nn.Parameter) and not experts.quantize  # a BF16 latent
    gate_up, _ = experts.unbound(torch.float32)
    assert torch.equal(gate_up[0], experts.gate_up[0].float())
    with run.context('t', 1.0):
        gate_up, _ = experts.unbound(torch.float32)
    assert torch.equal(gate_up[0], ternary.ternarize(experts.gate_up.detach()).float()[0])


# Multi-precision objective and gates --------------------------------------------------------------------------------
def test_multi_precision_backward_weights_each_point_and_learns_quantization_robustness():
    torch.manual_seed(5)
    model = Tiny()
    run = q.Quantization(component(bf16_weight=1.0))
    run.install(model, layer_count=2)
    layer = model.model.layers[0].feed_forward.w1
    x = torch.randn(8, 32)
    target = torch.randn(8, 64)

    def passes(_name):
        yield ((layer(x) - target) ** 2).mean(), {}
    results, losses = q.multi_precision_backward(run, 60, passes)
    assert set(losses) == {'bf16', 'q4'}
    # The gradient is bf16 + 0.5 x q4 (straight-through): recompute it by hand.
    latent = layer.parametrizations.weight.original
    expected = torch.zeros_like(latent)
    for value in (latent.detach(), latent.detach() + 0.5 * (q4_0(latent.detach()) - latent.detach())):
        w = value.clone().requires_grad_(True)
        loss = ((x @ w.T - target) ** 2).mean()
        weight = 1.0 if torch.equal(value, latent.detach()) else 0.5
        (weight * loss).backward()
        expected += w.grad
    assert torch.allclose(latent.grad, expected, atol=1e-6)


def test_gate_columns_evaluate_each_gated_point_at_deploy_precision_and_name_what_passed():
    run = q.Quantization(component('rec'))
    seen = []

    def evaluate():
        seen.append((PRECISION['point'], {g: v[1] for g, v in (PRECISION['groups'] or {}).items()}))
        return {'ok': PRECISION['point'] == 'q4'}
    columns = q.gate_columns(run, evaluate, lambda report: report['ok'])
    assert seen == [('q4', {'mlp': 1.0, 'attention': 1.0}), ('tern', {'mlp': 1.0})]
    verdict = q.precision_verdict(run, columns, bf16_passed=True)
    assert verdict == {'precisions_passed': ['bf16', 'q4'], 'precisions_required': ['bf16', 'q4'],
                       'precisions_qualified': True}
    assert not q.precision_verdict(run, {**columns, 'q4': {'passed': False}}, True)['precisions_qualified']
    assert not q.precision_verdict(run, columns, bf16_passed=False)['precisions_qualified']
    # A point not required in this stage whose ramp has not started is skipped (never counted as passed); a required
    # point is evaluated anyway. Each evaluated column records its wall time.
    seen.clear()
    run.mixes = lambda step: {'q4': 0.0, 'tern': 0.0}  # before either ramp starts
    early = q.gate_columns(run, evaluate, lambda report: report['ok'], step=0)
    assert 'tern' not in run.required_points()
    assert early['tern']['skipped'] and early['tern']['passed'] is None and [p for p, _ in seen] == ['q4']
    assert early['q4']['seconds'] >= 0
    assert 'tern' not in q.precision_verdict(run, early, True)['precisions_passed']


def test_recipe_gate_refuses_a_stage_that_missed_a_required_precision():
    from natlang_neuralese.train.recipe import require_gate
    with pytest.raises(ValueError, match='required precision: q4'):
        require_gate({'qualified': True, 'quantization': {'precisions_qualified': False,
                      'precisions_required': ['bf16', 'q4'], 'precisions_passed': ['bf16']}}, 'core_text_warmup')
    require_gate({'qualified': True, 'quantization': {'precisions_qualified': True}}, 'core_text_warmup')


def test_bf16_teacher_context_clears_the_active_point():
    from natlang_neuralese.serve.grad import _bf16_teacher
    with active_precision({'mlp': ('int4', 1.0, {})}, 'q4'):
        with _bf16_teacher():
            assert PRECISION['groups'] is None
        assert PRECISION['point'] == 'q4'


# Validation and recipes -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('change, message', [
    (lambda s: s['points'][0].update(name='bf16'), 'not bf16'),
    (lambda s: s['points'][0].update(ramp={'start': 0.5, 'end': 0.5}), 'ramp'),
    (lambda s: s['points'][0]['modules'].update(router={'format': 'int3'}), 'format'),
    (lambda s: s['points'][0]['modules'].update(mlp={'format': 'int4', 'nested_group': 32}), 'nested_group'),
    (lambda s: s['stage_progress'].update(warm=[0.6, 0.5]), 'stage progress'),
    (lambda s: s['loss'].update(mode='mean'), 'mode'),
])
def test_validate_spec_rejects_malformed_components(change, message):
    value = spec()
    change(value)
    with pytest.raises(ValueError, match=message):
        q.validate_spec(value)


def test_validate_spec_rejects_stages_without_a_trained_backbone():
    with pytest.raises(ValueError, match='trained-backbone stages'):
        q.validate_spec(spec(), {'warm': 'causal_embedding_distillation', 'rec': 'raw_recurrence_training'})
    with pytest.raises(ValueError, match='unknown stage'):
        q.validate_spec(spec(), {'warm': 'core_text_warmup'})


def test_shared_and_mellum_recipes_carry_one_component_and_inject_it_per_stage():
    from natlang_neuralese.train.recipe import load_recipe, stage_quantization, effective_stage_parameters
    shared = load_recipe(RECIPES / 'raw-recurrence-v6.json')
    mellum = load_recipe(RECIPES / 'raw-recurrence-mellum-v5.json')
    for key in ('loss', 'stage_progress', 'points'):  # unify: one component for both lines
        assert shared['quantization'][key] == mellum['quantization'][key]
    # Mellum-tokenized reuse of conversion v3 is declared backbone-inherent: preservation stream, gate, init.
    assert set(mellum['quantization']) - set(shared['quantization']) == {'preserve', 'gate'}
    assert mellum['init']['source'] == {'artifact': 'mellum21-convert-v3-latents-20261010', 'file': 'best-weights.pt'}
    stages = {s['id']: s for s in mellum['stages']}
    warm = stages['core_text_warmup']
    parameters = effective_stage_parameters(warm, [], mellum)
    assert parameters['backbone_training'] == 'latent' and parameters['lr'] == 7.5e-6
    injected = stage_quantization(mellum, warm, parameters, [])
    assert injected['stage'] == {'id': 'core_text_warmup', 'progress': [0.0, 0.35], 'start': 0, 'end': 4096}
    fixup = stages['autoregressive_text_fixup']
    reports = [{'id': 'core_text_warmup', 'kind': 'core_text_warmup', 'gate': {'step': 3000}}]
    fixup_parameters = effective_stage_parameters(fixup, reports, mellum)
    assert stage_quantization(mellum, fixup, fixup_parameters, reports)['stage'] == {
        'id': 'autoregressive_text_fixup', 'progress': [0.35, 0.45], 'start': 3000, 'end': 4024}
    assert stage_quantization(mellum, stages['embedding_distillation'], {}, []) is None
    recurrence = stages['recurrence_warmup']['parameters']
    assert recurrence['backbone_training'] == 'latent' and recurrence['backbone_lr'] == 7.5e-6
    assert 'member_weight' not in recurrence and 'qat_latent_lr' not in recurrence
    lfm = {s['id']: s for s in shared['stages']}
    assert lfm['recurrence_warmup']['parameters']['backbone_training'] == 'auto'  # LFM: auto resolves to full
    run = q.Quantization(injected)
    assert run.required_points() == ['q4']
    assert q.Quantization(stage_quantization(mellum, stages['recurrence_warmup'],
                                             stages['recurrence_warmup']['parameters'], [])).required_points() \
        == ['q4', 'ternary-experts']
    assert load_recipe(RECIPES / 'foundation-mellum-v2.json')['stages'] == \
        load_recipe(RECIPES / 'foundation-mellum-v1.json')['stages']


# End to end: the text warm-up on a tiny LFM2 with the component --------------------------------------------------------
def tiny_student():
    try:
        from transformers import Lfm2Config, Lfm2ForCausalLM
    except ImportError:
        pytest.skip('installed transformers does not provide the tiny LFM2 model')
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone
    torch.manual_seed(21)
    config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, block_multiple_of=8,
                        block_auto_adjust_ff_dim=False,
                        layer_types=['conv', 'full_attention', 'conv', 'full_attention'])
    model = Lfm2ForCausalLM(config).eval()
    backbone = PortBackbone(model, ControlTokens(62, 63), fast=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = PortHeads(backbone, cutoff=2, max_length=8, profile='latent-sketch-v2').eval()
    with torch.no_grad():
        heads.feedback.correction.weight.normal_(0, .02)
    return backbone, heads


def test_text_warmup_trains_every_planned_precision_and_reports_gate_columns(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.train import text_warmup

    def load(*_args):
        backbone, heads = tiny_student()
        return SimpleNamespace(backbone=backbone, heads=heads, tokenizer=None,
                               _tokens=lambda _text: [9, 3, 5, 8] * 8), None
    monkeypatch.setattr(text_warmup, 'load_initial', load)
    heads_path = tmp_path / 'heads.pt'
    torch.save({}, heads_path)
    records = tmp_path / 'records.jsonl'
    records.write_text('')
    text = tmp_path / 'text.jsonl'
    text.write_text('\n'.join(json.dumps({'text': s, 'split': split, 'source_groups': [s]})
                              for s, split in [('train', 'train'), ('held', 'test')]) + '\n')
    value = spec()
    value['points'][0]['ramp'] = {'start': 0.0, 'end': 0.25}  # q4 active after the first update
    value['points'][1]['ramp'] = {'start': 0.1, 'end': 0.5}
    value['points'][0]['modules'] = {'mlp': {'format': 'int4'}, 'conv': {'format': 'int4'},
                                     'attention': {'format': 'int4'}}
    component = q.stage_component(value, 'warm', start=0, end=4)
    out = tmp_path / 'run'
    text_warmup.main(['--heads', str(heads_path), '--records', str(records), '--text-data', str(text),
                      '--out', str(out), '--device', 'cpu', '--steps', '4', '--tokens', '16', '--prefix-tokens', '2',
                      '--batch', '1', '--eval-batch', '1', '--held-documents', '1', '--eval-every', '2',
                      '--checkpoint-every', '2', '--optimizer', 'adamw', '--backbone-training', 'full',
                      '--projection-patience', '1', '--projection-min-evals', '1', '--backbone-ramp-evals', '1',
                      '--pass-ramp-evals', '1', '--quantization', json.dumps(component)])
    rows = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]
    planned = [set(r['pass_metrics'][0]['precision']) for r in rows]
    assert planned[0] == {'bf16'}  # progress 0: every ramp at λ = 0
    assert planned[1:] == [{'bf16', 'q4', 'tern'}] * 3  # the sum objective: BF16 plus every active point
    assert rows[1]['pass_metrics'][0]['precision']['q4']['mix'] == pytest.approx(0.5)
    report = json.loads((out / 'report.json').read_text())
    columns = report['quantization']['columns']
    assert set(columns) == {'q4', 'tern'} and all('alignment_gate_passed' in c for c in columns.values())
    assert report['quantization']['precisions_required'] == ['bf16', 'q4']
    assert report['quantization']['mixes']['q4'] == 1.0
    assert all('seconds' in c for c in columns.values())
    evaluations = (out / 'eval.jsonl').read_text().splitlines()
    assert all('quantization' not in json.loads(line) or 'columns' in json.loads(line)['quantization']
               for line in evaluations)  # gate-column evaluations are not logged as their own rows
    # Checkpoints hold the latents under their plain names.
    saved = torch.load(out / 'checkpoint.pt', weights_only=False)
    names = [n for n in saved['student_parameters'] if n.startswith('backbone.')]
    assert names and not any('parametrizations' in n for n in names)


# Tiny Mellum smoke: BF16 latents trained under the Mellum recipe's own precision points -------------------------------
def test_tiny_mellum_latent_policy_trains_toward_its_deploy_precisions_and_restores_into_bf16(tmp_path):
    test_mellum_port = pytest.importorskip('test_mellum_port')
    from types import SimpleNamespace
    from natlang_neuralese.maple.model import load_maple
    from natlang_neuralese.train.backbone_policy import (backbone_trainable_state, configure_backbone_training,
                                                         restore_backbone_trainables)
    from natlang_neuralese.train.recipe import load_recipe, stage_quantization

    test_mellum_port.tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device='cpu', dtype=torch.float32, ternary_attention=False)
    backbone = SimpleNamespace(hf=model, ternary=True, parameters=model.parameters)
    with pytest.raises(ValueError, match='precision bf16'):
        configure_backbone_training(backbone, 'latent')
    model.natlang_precision = 'bf16'
    named = configure_backbone_training(backbone, 'latent')
    names = {n for n, _ in named}
    assert 'model.layers.0.mlp.experts.gate_up' in names and 'model.layers.0.self_attn.q_proj.weight' in names
    mellum = load_recipe(RECIPES / 'raw-recurrence-mellum-v5.json')
    stage = next(s for s in mellum['stages'] if s['id'] == 'recurrence_warmup')
    component = stage_quantization(mellum, stage, {'steps': 40}, [])
    component['loss']['mode'] = 'sum'
    run = q.Quantization(component)
    assert run.install(model, layer_count=4) == {'attention': 16, 'experts': 4}
    torch.manual_seed(0)
    ids = torch.randint(0, 128, (2, 24))

    def loss():
        logits = model(ids).logits[:, :-1]
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))

    def deploy(point):
        with torch.no_grad(), run.context(point, 1.0):
            return float(loss())
    before = {p: deploy(p) for p in ('q4', 'ternary-experts', 'ternary')}
    optimizer = torch.optim.AdamW([p for _, p in named], lr=3e-3)
    for step in range(40):
        optimizer.zero_grad()
        q.multi_precision_backward(run, step, lambda _point: iter([(loss(), {})]))
        assert model.model.layers[0].mlp.experts.gate_up.grad is not None
        optimizer.step()
    after = {p: deploy(p) for p in before}
    assert all(after[p] < before[p] for p in before), (before, after)
    # The latents restore by plain names into a fresh BF16 model (experts as buffers, no parametrizations).
    state = backbone_trainable_state(named)
    assert not any('parametrizations' in n for n in state)
    fresh = load_maple(tmp_path, device='cpu', dtype=torch.float32, ternary_attention=False)
    restore_backbone_trainables(SimpleNamespace(hf=fresh), state, expected_names=set(state))
    with torch.no_grad():
        assert torch.allclose(fresh(ids).logits, model(ids).logits, atol=1e-5)



# Reuse of conversion v3: init decision, ramp floor, preservation stream, behaviour gate --------------------------------
def test_init_ramp_floor_keeps_a_point_at_least_at_the_latents_lambda():
    run = q.Quantization({**component(), 'init_mix': {'tern': 0.6}})
    assert run.mixes(0) == {'q4': 0.0, 'tern': 0.6}
    assert run.active(0) == [('tern', 0.6)]
    with pytest.raises(ValueError, match='init_mix'):
        q.Quantization({**component(), 'init_mix': {'other': 0.5}})


def test_validate_init_requires_a_gate_a_bf16_fallback_and_a_fresh_optimizer():
    value = {**spec(), 'gate': {'probes': {'path': 'p', 'sha256': '0' * 64}}}
    init = {'source': {'artifact': 'a', 'file': 'f'}, 'fallback': 'bf16', 'ramp_floor': 'tern', 'optimizer': 'fresh'}
    assert q.validate_init(init, value) == init
    assert q.validate_init({'source': 'bf16'}, None)
    with pytest.raises(ValueError, match='fresh optimizer'):
        q.validate_init({**init, 'optimizer': 'restore'}, value)
    with pytest.raises(ValueError, match='gated'):
        q.validate_init(init, spec())
    with pytest.raises(ValueError, match='ramp_floor'):
        q.validate_init({**init, 'ramp_floor': 'nope'}, value)


def test_init_mix_follows_the_recorded_decision_and_its_receipt(tmp_path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    recipe = {'init': {'source': {'artifact': 'v3', 'file': 'best.pt'}, 'fallback': 'bf16', 'ramp_floor': 'tern'}}

    def heads(passed, source):
        receipt = tmp_path / f'receipt-{passed}.json'
        receipt.write_text(json.dumps({'artifact': 'v3', 'file': 'best.pt', 'sha256': 'abc', 'passed': passed}))
        init = {'source': source, 'receipt': str(receipt), 'receipt_sha256': sha256_file_hex(receipt)}
        if source == 'artifact':
            init.update(artifact='v3', file='best.pt', sha256='abc', step=900, mix=0.75)
        return {'init': init}
    assert q.init_mix_for(recipe, heads(True, 'artifact')) == {'tern': 0.75}
    assert q.init_mix_for(recipe, heads(False, 'bf16')) == {}  # the declared fallback
    with pytest.raises(ValueError, match='did not pass'):
        q.init_mix_for(recipe, heads(False, 'artifact'))
    with pytest.raises(ValueError, match='must start from the declared latents'):
        q.init_mix_for(recipe, heads(True, 'bf16'))
    with pytest.raises(ValueError, match='--init-receipt'):
        q.init_mix_for(recipe, {})
    with pytest.raises(ValueError, match='does not declare'):
        q.init_mix_for({}, heads(True, 'artifact'))


def test_behaviour_gate_compares_with_the_declared_bf16_reference(tmp_path):
    probes = tmp_path / 'probes.jsonl'
    probes.write_text('{"id": "p", "messages": []}\n')
    from natlang_neuralese.common.hashing import sha256_file_hex
    from types import SimpleNamespace
    config = {'probes': {'path': str(probes), 'sha256': sha256_file_hex(probes)}, 'max_gate_drop': 0.1,
              'max_held_kl': 0.25, 'reference': {'gate_pass': 0.75}}
    gate = q.Behaviour(config, SimpleNamespace(convert_tokens_to_ids=lambda token: 7))
    assert gate.passed({'gate_pass': 0.66, 'held_kl': 0.2})
    assert not gate.passed({'gate_pass': 0.6, 'held_kl': 0.2})
    assert not gate.passed({'gate_pass': 0.9, 'held_kl': 0.3})
    assert gate.passed({'gate_pass': 0.5, 'held_kl': 0.0}, {'gate_pass': 0.55})  # a live BF16 report instead
    probes.write_text('{}\n')
    with pytest.raises(ValueError, match='probes changed'):
        q.Behaviour(config, SimpleNamespace(convert_tokens_to_ids=lambda token: 7))


def test_preservation_stream_is_zero_at_bf16_and_trains_the_quantized_model_toward_bf16(tmp_path):
    test_mellum_port = pytest.importorskip('test_mellum_port')
    from natlang_neuralese.maple.model import load_maple
    test_mellum_port.tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device='cpu', dtype=torch.float32, ternary_attention=False)
    torch.manual_seed(0)
    records = {'ids': [], 'top_ids': [], 'top_logp': [], 'prompt_len': []}
    with torch.no_grad():
        for _ in range(4):
            ids = torch.randint(0, 128, (20,))
            logp = torch.log_softmax(model(ids[None]).logits[0, :-1].float(), -1)
            top = logp.topk(128, -1)  # the full vocabulary: KL to itself is zero
            records['ids'].append(ids)
            records['top_ids'].append(top.indices)
            records['top_logp'].append(top.values)
            records['prompt_len'].append(5)
    preserve = q.Preserve({'teacher': {'artifact': 'unused'}, 'weight': 0.5, 'max_tokens': 16}, 'warm',
                          shards=(records, records, {}))
    loss, info = preserve.loss(model, 3)
    assert info['preserve_record'] == preserve.record(3) and abs(info['preserve_kl']) < 1e-3  # BF16: its own teacher
    # A frozen-backbone phase (projection first): no gradient path, so the stream is skipped and logged, not run.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert not preserve.applies(model, 0) and not preserve.loss(model, 0)[0].requires_grad
    run = q.Quantization({'schema': q.SCHEMA, 'points': [
        {'name': 't', 'ramp': {'start': 0.0, 'end': 0.01}, 'modules': {'experts': {'format': 'ternary'},
                                                                        'attention': {'format': 'ternary'}}}],
        'stage': {'id': 'warm', 'progress': [0.0, 1.0], 'start': 0, 'end': 10}})
    run.install(model, layer_count=4)
    named = [(n, p) for n, p in model.named_parameters() if 'original' in n or 'experts' in n]
    for _, parameter in named:
        parameter.requires_grad_(True)
    assert preserve.applies(model, 1)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=3e-3)
    with run.context('t', 1.0):
        start = float(preserve.loss(model, 0)[0])
        for step in range(30):
            optimizer.zero_grad()
            preserve.loss(model, step)[0].backward()
            optimizer.step()
        end = sum(float(preserve.loss(model, s)[0]) for s in range(4)) / 4
    assert start > 0.01 and end < start
