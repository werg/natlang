"""NVFP4 / FP8_BLOCK formats of the quantization component (train/qformats.py) and the LFM3 deploy exports
(export/lfm3_quantized.py), owner 2026-10-28: exact codecs, straight-through ramps, activation quantizers, static
calibration state, gates at deploy precision, and exports in Liquid's layouts."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from natlang_neuralese.export import lfm3_quantized as X
from natlang_neuralese.maple.ternary import PRECISION, quantized_value
from natlang_neuralese.train import qformats as F
from natlang_neuralese.train import quantization as q

HUB = Path.home() / '.cache' / 'huggingface' / 'hub'
BASE = HUB / 'models--LiquidAI--LFM3-1.7B' / 'snapshots' / '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
NVFP4 = HUB / 'models--LiquidAI--LFM3-1.7B-NVFP4' / 'snapshots' / 'b30025bbbbcbf62f6c110b40d51d36af6579f6a2'
FP8 = HUB / 'models--LiquidAI--LFM3-1.7B-FP8' / 'snapshots' / '243955da95b1cd68026f293b29de4b4c13f3638f'
LIQUID = all((p / 'model.safetensors').exists() for p in (BASE, NVFP4, FP8))
LAYER_TYPES = ['conv', 'full_attention', 'conv', 'sliding_attention']


# Scalar references (written out per element, independent of the vectorized codecs) ------------------------------
def e2m1_reference(value):
    """Nearest E2M1 code, ties to the even code, saturating; sign bit from value < 0 (ModelOpt ``_cast_fp4``)."""
    magnitude = abs(value)
    best = None
    for code, grid in enumerate(F.E2M1_GRID):
        distance = abs(magnitude - grid)
        if best is None or distance < best[0] or (distance == best[0] and code % 2 == 0):
            best = (distance, code)
    return best[1] + (8 if value < 0 else 0)


def nvfp4_reference(row, global_amax=None):
    """ModelOpt NVFP4 of one row (FP32 scalar arithmetic): codes, E4M3 scales and the global scale."""
    row = [float(v) for v in row]
    amax = max(abs(v) for v in row) if global_amax is None else float(global_amax)
    scale_2 = float(torch.tensor(amax, dtype=torch.float32) / torch.tensor(6.0 * 448.0, dtype=torch.float32))
    codes, scales = [], []
    for start in range(0, len(row), 16):
        block = row[start:start + 16]
        per = torch.tensor(max(abs(v) for v in block), dtype=torch.float32) / (
            torch.tensor(6.0, dtype=torch.float32) * torch.tensor(scale_2, dtype=torch.float32))
        per = torch.tensor(1.0) if float(per) == 0 else per
        stored = float(per.to(torch.float8_e4m3fn).float())
        scales.append(stored)
        divisor = torch.tensor(stored, dtype=torch.float32) * torch.tensor(scale_2, dtype=torch.float32)
        codes.extend(e2m1_reference(float(torch.tensor(v, dtype=torch.float32) / divisor)) for v in block)
    return codes, scales, scale_2


# Codecs ----------------------------------------------------------------------------------------------------------
def test_e2m1_codes_round_to_nearest_even_saturate_and_keep_negative_zero():
    values = torch.tensor([0.0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 7.0, 1e9, -0.1, -0.0, -6.0,
                           -2.75, 0.74])
    codes = F.e2m1_codes(values).tolist()
    assert codes == [e2m1_reference(float(v)) for v in values]
    assert codes[:12] == [0, 0, 1, 2, 2, 4, 4, 6, 6, 7, 7, 7]  # ties: .25->0 .75->1.0 1.25->1.0 1.75->2 2.5->2 3.5->4 5->4
    assert codes[12] == 8 and codes[13] == 0 and codes[14] == 15  # -0.1 keeps the sign bit, -0.0 does not
    grid = torch.linspace(-8, 8, 4001)
    assert F.e2m1_codes(grid).tolist() == [e2m1_reference(float(v)) for v in grid]
    assert torch.equal(F.e2m1_values(F.e2m1_codes(torch.tensor(F.E2M1_GRID))), torch.tensor(F.E2M1_GRID))


def test_fp4_packing_is_modelopt_low_nibble_first():
    codes = torch.tensor([[1, 2, 15, 8]], dtype=torch.uint8)
    packed = F.pack_fp4(codes)
    assert packed.tolist() == [[0x21, 0x8F]]
    assert torch.equal(F.unpack_fp4(packed), codes)


def test_nvfp4_weight_matches_the_scalar_modelopt_rule():
    torch.manual_seed(0)
    weight = (torch.randn(4, 64) * torch.tensor([1e-3, 1.0, 30.0, 0.02])[:, None]).to(torch.bfloat16)
    weight[0, :16] = 0  # an all-zero group: scale stored as 1.0, codes 0
    packed, scales, scale_2 = F.nvfp4_weight_encode(weight)
    codes = F.unpack_fp4(packed)
    reference = [nvfp4_reference(row, weight.float().abs().max()) for row in weight.float()]
    assert float(scale_2) == reference[0][2]
    for i, (row_codes, row_scales, _) in enumerate(reference):
        assert codes[i].tolist() == row_codes
        assert scales[i].float().tolist() == row_scales
    assert scales[0, 0].float() == 1.0 and codes[0, :16].eq(0).all()
    value = F.nvfp4_weight_value(weight)
    assert torch.equal(value, F.nvfp4_weight_decode(packed, scales, scale_2))
    assert torch.equal(quantized_value(weight, 'nvfp4', {}), value)


def test_nvfp4_shared_global_scale_uses_the_larger_amax():
    torch.manual_seed(1)
    small, large = torch.randn(8, 32).to(torch.bfloat16), (torch.randn(8, 32) * 4).to(torch.bfloat16)
    shared = torch.stack([small.abs().amax(), large.abs().amax()]).amax()
    _, _, own = F.nvfp4_weight_encode(small)
    _, _, joint = F.nvfp4_weight_encode(small, global_amax=shared)
    assert float(joint) == float(F.nvfp4_global_scale(large.abs().amax())) and float(joint) > float(own)
    assert torch.equal(F.nvfp4_weight_value(small, global_amax=shared),
                       quantized_value(small, 'nvfp4', {'global_amax': shared}))


def test_fp8_block_matches_compressed_tensors_rule():
    torch.manual_seed(2)
    weight = (torch.randn(256, 384) * 0.05).to(torch.bfloat16)
    weight[:128, :128] = 0  # a zero block: scale clamps to FP32 eps (then BF16)
    codes, scales = F.fp8_block_encode(weight)
    assert scales.dtype == torch.bfloat16 and scales.shape == (2, 3) and codes.dtype == torch.float8_e4m3fn
    for i in range(2):
        for j in range(3):
            block = weight[128 * i:128 * (i + 1), 128 * j:128 * (j + 1)]
            amax = max(float(block.float().abs().max()), 0.0)
            expected = torch.tensor(max(amax / 448.0, torch.finfo(torch.float32).eps), dtype=torch.float32)
            expected = expected.to(torch.bfloat16)
            assert scales[i, j] == expected
            # The division runs in the weight dtype (BF16), then E4M3: compressed-tensors' _quantize.
            ref = torch.clamp(block / expected, -448, 448).to(torch.float8_e4m3fn)
            assert torch.equal(codes[128 * i:128 * (i + 1), 128 * j:128 * (j + 1)].view(torch.uint8),
                               ref.view(torch.uint8))
    assert torch.equal(quantized_value(weight, 'fp8_block', {}), F.fp8_block_decode(codes, scales))


def test_activation_quantizers():
    torch.manual_seed(3)
    x = (torch.randn(3, 5, 256) * 3).to(torch.bfloat16)
    value = F.fp8_group_activation_value(x)
    blocks = x.float().reshape(3, 5, 2, 128)
    scale = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448
    assert torch.equal(value, ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape))
    # NVFP4 at a static scale: groups inside the calibrated range keep 16 values on the E2M1 grid times their scale;
    # a group beyond it saturates (scale clamps at 448, values at 6) as deployed.
    amax = x.float().abs().max()
    input_scale = F.nvfp4_global_scale(amax)
    value = F.nvfp4_activation_value(x, input_scale)
    groups = value.reshape(-1, 16)
    assert torch.allclose(groups.abs().amax(-1), x.float().reshape(-1, 16).abs().amax(-1), rtol=0.3)
    saturated = F.nvfp4_activation_value(x * 10, input_scale)
    assert float(saturated.abs().max()) == pytest.approx(float(6 * 448 * input_scale), rel=1e-6)
    assert torch.equal(F.activation_value(x, {'format': 'nvfp4'}, input_scale), value)
    with pytest.raises(ValueError, match='calibrated'):
        F.activation_value(x, {'format': 'nvfp4'})


# Component -------------------------------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, hidden, inner):
        super().__init__()
        self.w1, self.w3 = nn.Linear(hidden, inner, bias=False), nn.Linear(hidden, inner, bias=False)
        self.w2 = nn.Linear(inner, hidden, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, nn.Linear(hidden, hidden, bias=False))
        self.gate_proj = nn.Linear(hidden, 2, bias=False)

    def forward(self, x):
        y = self.o_proj(self.v_proj(x) + self.q_proj(x) * 0.1 + self.k_proj(x) * 0.1)
        return y * torch.sigmoid(self.gate_proj(x)).mean(-1, keepdim=True)


class Conv(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.in_proj, self.out_proj = nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.out_proj(self.in_proj(x))


class Layer(nn.Module):
    def __init__(self, kind, hidden, inner):
        super().__init__()
        if kind == 'conv':
            self.conv = Conv(hidden)
        else:
            self.self_attn = Attention(hidden)
        self.feed_forward = FeedForward(hidden, inner)

    def forward(self, x):
        x = x + (self.conv(x) if hasattr(self, 'conv') else self.self_attn(x))
        return x + self.feed_forward(x)


class Tiny(nn.Module):
    """An LFM3-shaped module tree with the native port's names (model.layers.N.{conv,self_attn,feed_forward})."""

    def __init__(self, hidden=128, inner=256):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Layer(kind, hidden, inner) for kind in LAYER_TYPES)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def lfm3(stage='export'):
    run = q.Quantization(X.lfm3_component(stage))
    torch.manual_seed(4)
    model = Tiny().to(torch.bfloat16)
    reference = Tiny().to(torch.bfloat16)
    reference.load_state_dict({q.plain_parameter_name(k): v for k, v in model.state_dict().items()})
    run.install(model)
    return run, model, reference


def test_lfm3_component_validates_and_rejects_bad_deploy_options():
    q.validate_spec({k: v for k, v in X.lfm3_component().items() if k != 'stage'})
    base = {'schema': q.SCHEMA, 'stage_progress': {}, 'points': [
        {'name': 'p', 'ramp': {'start': 0.0, 'end': 1.0}, 'modules': {'mlp': None}}]}
    bad = [{'format': 'int4', 'shared_global': [['w1', 'w3']]},
           {'format': 'nvfp4', 'block': 128},
           {'format': 'fp8_block', 'group': 16},
           {'format': 'int4', 'activations': {'format': 'fp8_group'}},
           {'format': 'nvfp4', 'activations': {'format': 'nvfp4'}},  # static scale without calibration
           {'format': 'fp8_block', 'activations': {'format': 'fp8_group', 'calibration': {'refresh': 1, 'batches': 1}}},
           {'format': 'nvfp4', 'names': ['feed_forward.w1']},
           {'format': 'nvfp4', 'shared_global': [['w1', 'w3'], ['w3', 'w2']]}]
    for config in bad:
        spec = json.loads(json.dumps(base))
        spec['points'][0]['modules']['mlp'] = config
        with pytest.raises(ValueError):
            q.validate_spec(spec)


def test_install_selects_liquids_modules_per_point():
    run, model, _ = lfm3()
    keys = {n: m.parametrizations.weight[0].group for n, m in model.named_modules()
            if hasattr(m, 'parametrizations')}
    attention = {n for n in keys if '.self_attn.' in n}
    assert attention == {f'model.layers.{i}.self_attn.{p}' for i in (1, 3) for p in ('q_proj', 'k_proj', 'v_proj', 'o_proj')}
    assert not any('.conv.' in n for n in keys)  # no point quantizes conv: not parametrized
    assert keys['model.layers.0.feed_forward.w1'] == 'mlp@0:w1'
    assert set(run.inputs) == set(keys)  # every quantized linear quantizes its input in some point
    assert {n for n, h in run.inputs.items() if h.static} == {n for n in keys if '.feed_forward.' in n}
    ste = model.model.layers[0].feed_forward.w1.parametrizations.weight[0]
    assert [leaf for leaf, _ in ste.peers] == ['w3']
    assert model.model.layers[0].feed_forward.w2.parametrizations.weight[0].peers == []


def test_points_at_deploy_precision_match_manual_quantization_with_identity_gradients():
    run, model, reference = lfm3()
    windows = [torch.randn(2, 7, 128, dtype=torch.bfloat16) for _ in range(3)]
    with pytest.raises(RuntimeError, match='not calibrated'):
        with run.context('nvfp4', 1.0):
            model(windows[0])
    assert run.calibration_due(0)
    amax = run.calibrate(model, windows, step=0, source='test')
    assert not run.calibration_due(249) and run.calibration_due(250)
    # max calibration at BF16: the largest |input| of each module over the windows
    seen = {}
    hooks = [m.register_forward_pre_hook(lambda mod, args, n=n: seen.__setitem__(n, max(seen.get(n, 0.0), float(args[0].abs().max()))))
             for n, m in reference.named_modules() if n in amax]
    for w in windows:
        reference(w)
    for h in hooks:
        h.remove()
    assert amax == pytest.approx(seen, rel=0, abs=0)
    # λ = 1 forward equals the reference model run with quantized weights and inputs, by hand.
    x = torch.randn(2, 7, 128, dtype=torch.bfloat16)
    for point in ('fp8', 'nvfp4'):
        manual = {}
        for name, module in reference.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf, parent = name.rsplit('.', 1)[1], name.rsplit('.', 1)[0]
            ffn, attn = '.feed_forward.' in name, '.self_attn.' in name and leaf != 'gate_proj'
            if point == 'fp8' and (ffn or attn):
                manual[name] = (F.fp8_block_value(module.weight).to(torch.bfloat16),
                                lambda t: F.fp8_group_activation_value(t).to(t.dtype))
            elif point == 'nvfp4' and ffn:
                peers = [reference.get_submodule(f'{parent}.{n}').weight for n in ('w1', 'w3')] if leaf in ('w1', 'w3') \
                    else [module.weight]
                shared = torch.stack([p.abs().amax() for p in peers]).amax()
                scale = F.nvfp4_global_scale(torch.tensor(amax[name]))
                manual[name] = (F.nvfp4_weight_value(module.weight, global_amax=shared).to(torch.bfloat16),
                                lambda t, s=scale: F.nvfp4_activation_value(t, s).to(t.dtype))
        handles = []
        for name, (weight, act) in manual.items():
            module = reference.get_submodule(name)
            handles.append(module.register_forward_pre_hook(lambda mod, args, a=act: (a(args[0]),)))
            module.weight.data, module.original = weight, module.weight.data.clone()
        expected = reference(x)
        for name in manual:
            module = reference.get_submodule(name)
            module.weight.data = module.original
        for h in handles:
            h.remove()
        with run.context(point, 1.0):
            got = model(x)
        assert torch.equal(got, expected), point
    # Straight-through: at λ = 1 the gradient reaches the latent and the input as identity through the quantizers,
    # i.e. it is the gradient of y = Q(x) Q(W)^T taken with respect to Q(W) and Q(x).
    layer = model.model.layers[0].feed_forward.w2
    x = torch.randn(1, 3, 256, dtype=torch.bfloat16, requires_grad=True)
    with run.context('nvfp4', 1.0):
        layer(x).sum().backward()
        weight_q = layer.weight.detach()
    scale = F.nvfp4_global_scale(torch.tensor(amax['model.layers.0.feed_forward.w2']))
    x_q = F.nvfp4_activation_value(x.detach(), scale).to(torch.bfloat16)
    latent = layer.parametrizations.weight.original
    expected_w = torch.ones(1, 3, 128, dtype=torch.bfloat16).reshape(3, 128).t() @ x_q.reshape(3, 256)
    assert torch.equal(latent.grad, expected_w)
    assert torch.equal(x.grad, (torch.ones(1, 3, 128, dtype=torch.bfloat16) @ weight_q))


def test_ramp_mixes_activations_and_weights_and_bf16_is_untouched():
    run, model, reference = lfm3()
    run.calibrate(model, [torch.randn(1, 4, 128, dtype=torch.bfloat16)], step=0)
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    with run.context('bf16', 0.0):
        assert torch.equal(model(x), reference(x))
    with run.context('fp8', 0.0):
        assert torch.equal(model(x), reference(x))
    with run.context('fp8', 0.5):
        half = model(x)
    with run.context('fp8', 1.0):
        full = model(x)
    assert not torch.equal(half, full) and not torch.equal(half, reference(x))
    assert PRECISION['groups'] is None


def test_calibration_state_round_trips_and_gates_require_it():
    run, model, _ = lfm3()
    with pytest.raises(ValueError, match='calibrate'):
        q.gate_columns(run, lambda: {}, lambda report: True)
    run.calibrate(model, [torch.randn(1, 4, 128, dtype=torch.bfloat16)], step=7, source={'cursor': 3})
    state = json.loads(json.dumps(run.state_dict()))
    assert state['step'] == 7 and state['source'] == {'cursor': 3} and len(state['amax']) == 12
    other, model2, _ = lfm3()
    other.load_state_dict(state)
    assert other.input_amax() == run.input_amax()
    x = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    with run.context('nvfp4', 1.0):
        a = model(x)
    with other.context('nvfp4', 1.0):
        b = model2(x)
    assert torch.equal(a, b)
    seen = []

    def evaluate():
        seen.append((PRECISION['point'], {g: v[1] for g, v in PRECISION['groups'].items()}))
        return {}
    columns = q.gate_columns(run, evaluate, lambda report: True)
    assert set(columns) == {'fp8', 'nvfp4'} and all(c['passed'] for c in columns.values())
    assert seen == [('fp8', {'attention': 1.0, 'mlp': 1.0}), ('nvfp4', {'mlp': 1.0})]
    assert q.Quantization(X.lfm3_component()).required_points() == ['fp8', 'nvfp4']
    with pytest.raises(ValueError):
        other.load_state_dict({**state, 'amax': {'model.layers.0.feed_forward.w1': 1.0}})


def test_old_formats_keep_their_keys_and_values():
    """int4/ternary components install as before (leaf suffix only) and give the same forward."""
    spec = {'schema': q.SCHEMA, 'stage_progress': {'s': [0.0, 1.0]}, 'points': [
        {'name': 'q4', 'ramp': {'start': 0.0, 'end': 0.5}, 'modules': {'mlp': {'format': 'int4', 'group': 32},
                                                                       'attention': {'format': 'int4'}},
         'layers': [1, 2]}]}
    run = q.Quantization(q.stage_component(spec, 's', start=0, end=10))
    torch.manual_seed(5)
    model = Tiny().to(torch.bfloat16)
    reference = Tiny().to(torch.bfloat16)
    reference.load_state_dict(model.state_dict())
    run.install(model)
    assert run.inputs == {} and run.calibration_config() is None and not run.calibration_due(0)
    x = torch.randn(1, 3, 128, dtype=torch.bfloat16)
    with run.context('q4', 1.0):
        got = model(x)
    for i in (1, 2):
        for name, module in reference.model.layers[i].named_modules():
            if isinstance(module, nn.Linear) and not name.startswith('conv'):  # mlp + attention (gate_proj included)
                module.weight.data = quantized_value(module.weight, 'int4', {'group': 32}).to(torch.bfloat16)
    assert torch.equal(got, reference(x))


# Exports ---------------------------------------------------------------------------------------------------------
def tiny_checkpoint(directory):
    from safetensors.torch import save_file
    torch.manual_seed(6)
    tensors = {'model.embed_tokens.weight': torch.randn(64, 128), 'model.embedding_norm.weight': torch.ones(128)}
    for i, kind in enumerate(LAYER_TYPES):
        prefix = f'model.layers.{i}.'
        tensors[prefix + 'operator_norm.weight'] = torch.ones(128)
        for leaf, shape in (('w1', (256, 128)), ('w3', (256, 128)), ('w2', (128, 256))):
            tensors[prefix + f'feed_forward.{leaf}.weight'] = torch.randn(shape) * 0.02
        if kind == 'conv':
            tensors[prefix + 'conv.conv.weight'] = torch.randn(128, 1, 7)
            tensors[prefix + 'conv.in_proj.weight'] = torch.randn(384, 128)
            tensors[prefix + 'conv.out_proj.weight'] = torch.randn(128, 128)
        else:
            for leaf in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                tensors[prefix + f'self_attn.{leaf}.weight'] = torch.randn(128, 128) * 0.02
            tensors[prefix + 'self_attn.gate_proj.weight'] = torch.randn(16, 128)
            tensors[prefix + 'self_attn.q_norm.weight'] = torch.ones(128)
    tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}
    directory.mkdir()
    save_file(tensors, str(directory / 'model.safetensors'), metadata={'format': 'pt'})
    (directory / 'config.json').write_text(json.dumps({'model_type': 'lfm3', 'layer_types': LAYER_TYPES}))
    (directory / 'tokenizer.json').write_text('{}')
    return tensors


def test_exports_write_liquids_layouts(tmp_path):
    tensors = tiny_checkpoint(tmp_path / 'bf16')
    amax = {f'model.layers.{i}.feed_forward.{leaf}': 2.5 + i for i in range(4) for leaf in ('w1', 'w2', 'w3')}
    X.export(tmp_path / 'bf16', tmp_path / 'nvfp4', 'nvfp4', input_amax=amax)
    X.export(tmp_path / 'bf16', tmp_path / 'fp8', 'fp8_block')
    nv, fp = X._read(tmp_path / 'nvfp4'), X._read(tmp_path / 'fp8')
    assert (tmp_path / 'nvfp4' / 'tokenizer.json').exists()
    config = json.loads((tmp_path / 'nvfp4' / 'config.json').read_text())
    assert config['quantization_config']['ignore'] == [
        'lm_head', 'model.embed_tokens', 'model.layers.0.conv*', 'model.layers.1.self_attn*', 'model.layers.2.conv*',
        'model.layers.3.self_attn*']
    w1, w3 = 'model.layers.1.feed_forward.w1', 'model.layers.1.feed_forward.w3'
    assert nv[w1 + '.weight'].dtype == torch.uint8 and nv[w1 + '.weight'].shape == (256, 64)
    assert nv[w1 + '.weight_scale'].dtype == torch.float8_e4m3fn and nv[w1 + '.weight_scale'].shape == (256, 8)
    assert torch.equal(nv[w1 + '.weight_scale_2'], nv[w3 + '.weight_scale_2'])
    assert float(nv[w1 + '.input_scale']) == float(F.nvfp4_global_scale(torch.tensor(3.5)))
    assert torch.equal(nv['model.layers.1.self_attn.q_proj.weight'], tensors['model.layers.1.self_attn.q_proj.weight'])
    quantized = {k[:-len('.weight_scale')] for k in fp if k.endswith('.weight_scale')}
    assert quantized == {f'model.layers.{i}.feed_forward.{l}' for i in range(4) for l in ('w1', 'w2', 'w3')} | {
        f'model.layers.{i}.self_attn.{l}' for i in (1, 3) for l in ('q_proj', 'k_proj', 'v_proj', 'o_proj')}
    assert fp['model.layers.1.self_attn.gate_proj.weight'].dtype == torch.bfloat16
    assert json.loads((tmp_path / 'fp8' / 'config.json').read_text())['quantization_config'] == X.FP8_BLOCK_CONFIG
    report = X.compare_states('fp8_block', fp, fp, LAYER_TYPES)
    assert report['summary']['tensors_bit_identical'] == len(fp) and report['summary']['modules'] == 20
    # The QAT forward's deployed values are the export's dequantized values.
    weight = tensors[w1 + '.weight']
    shared = torch.stack([weight.abs().amax(), tensors[w3 + '.weight'].abs().amax()]).amax()
    assert torch.equal(F.nvfp4_weight_value(weight, global_amax=shared),
                       F.nvfp4_weight_decode(nv[w1 + '.weight'], nv[w1 + '.weight_scale'], nv[w1 + '.weight_scale_2']))
    with pytest.raises(ValueError, match='input amax'):
        X.export(tmp_path / 'bf16', tmp_path / 'missing', 'nvfp4', input_amax={})


@pytest.mark.skipif(not LIQUID, reason="Liquid's LFM3 checkpoints are not in the HF cache")
def test_liquid_configs_match_ours():
    config = json.loads((BASE / 'config.json').read_text())
    quantization_config, hf_quant_config = X.nvfp4_configs(config['layer_types'])
    assert quantization_config == json.loads((NVFP4 / 'config.json').read_text())['quantization_config']
    assert hf_quant_config == json.loads((NVFP4 / 'hf_quant_config.json').read_text())
    assert X.FP8_BLOCK_CONFIG == json.loads((FP8 / 'config.json').read_text())['quantization_config']


@pytest.mark.skipif(not LIQUID, reason="Liquid's LFM3 checkpoints are not in the HF cache")
def test_liquid_codes_and_scales_reproduce_exactly():
    """Our codecs on the BF16 base reproduce Liquid's stored codes and scales bit for bit (a sample of modules; the
    full comparison is ``lfm3_quantized compare``), and their static input scales are BF16 maxima / (6·448)."""
    from safetensors import safe_open
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with safe_open(str(BASE / 'model.safetensors'), 'pt', device) as base, \
            safe_open(str(NVFP4 / 'model.safetensors'), 'pt', device) as nv, \
            safe_open(str(FP8 / 'model.safetensors'), 'pt', device) as fp:
        for layer in (0, 13, 23):
            prefix = f'model.layers.{layer}.feed_forward.'
            shared = torch.stack([base.get_tensor(prefix + f'{l}.weight').abs().amax() for l in ('w1', 'w3')]).amax()
            for leaf in ('w1', 'w2', 'w3'):
                weight = base.get_tensor(prefix + f'{leaf}.weight')
                packed, scales, scale_2 = F.nvfp4_weight_encode(weight, global_amax=shared if leaf != 'w2' else None)
                assert torch.equal(packed, nv.get_tensor(prefix + f'{leaf}.weight'))
                assert torch.equal(scales.view(torch.uint8), nv.get_tensor(prefix + f'{leaf}.weight_scale').view(torch.uint8))
                assert torch.equal(scale_2, nv.get_tensor(prefix + f'{leaf}.weight_scale_2'))
                codes, block = F.fp8_block_encode(weight)
                assert torch.equal(codes.view(torch.uint8), fp.get_tensor(prefix + f'{leaf}.weight').view(torch.uint8))
                assert torch.equal(block, fp.get_tensor(prefix + f'{leaf}.weight_scale'))
            input_scale = nv.get_tensor(prefix + 'w1.input_scale')
            amax = (input_scale * (6 * 448)).to(torch.bfloat16)
            assert torch.equal(F.nvfp4_global_scale(amax), input_scale)
            assert torch.equal(input_scale, nv.get_tensor(prefix + 'w3.input_scale'))
        q_proj = 'model.layers.2.self_attn.q_proj.'
        codes, block = F.fp8_block_encode(base.get_tensor(q_proj + 'weight'))
        assert torch.equal(codes.view(torch.uint8), fp.get_tensor(q_proj + 'weight').view(torch.uint8))
        assert torch.equal(block, fp.get_tensor(q_proj + 'weight_scale'))
