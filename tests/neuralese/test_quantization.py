"""The recipe quantization component (train/quantization.py): QAT woven into the training stages (owner 2026-10-10)."""
import json

import pytest
import torch
from torch import nn

from natlang_neuralese.maple import ternary
from natlang_neuralese.maple.ternary import PRECISION, active_precision, precision_value, q4_0, quantized_value
from natlang_neuralese.train import quantization as q



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


# Initial mix ------------------------------------------------------------------------------------------------------
def test_init_ramp_floor_keeps_a_point_at_least_at_the_latents_lambda():
    run = q.Quantization({**component(), 'init_mix': {'tern': 0.6}})
    assert run.mixes(0) == {'q4': 0.0, 'tern': 0.6}
    assert run.active(0) == [('tern', 0.6)]
    with pytest.raises(ValueError, match='init_mix'):
        q.Quantization({**component(), 'init_mix': {'other': 0.5}})
