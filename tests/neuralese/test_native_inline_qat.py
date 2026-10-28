"""Inline QAT in the native trainer (``--quantization``): the LFM3 FP8/NVFP4 points on a tiny BF16 LFM3.

Covers: plain parameter names after installation; λ = 0 equals the unquantized model exactly and λ > 0 differs, on
both the plain path and the BF16 cotangent-carrier path of a real update; gradients reach the latents through the
straight-through route; the trainer's calibration window and pass choice; calibration state round-trip.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lfm3_port import LFM3_TINY_TEMPLATE, tiny_lfm3_hf  # noqa: E402


def lfm3_points(block=16):
    return {'schema': 'natlang.neuralese-quantization/1',
            'loss': {'mode': 'sample', 'bf16_weight': 1.0, 'seed': 0},
            'stage_progress': {'joint_sft': [0.0, 1.0]},
            'points': [
                {'name': 'fp8', 'gate': 'required', 'weight': 0.5, 'ramp': {'start': 0.0, 'end': 0.15},
                 'modules': {
                     'attention': {'format': 'fp8_block', 'block': block, 'names': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                                   'activations': {'format': 'fp8_group', 'group': 16}},
                     'mlp': {'format': 'fp8_block', 'block': block, 'names': ['w1', 'w2', 'w3'],
                             'activations': {'format': 'fp8_group', 'group': 16}}}},
                {'name': 'nvfp4', 'gate': 'required', 'weight': 1.0, 'ramp': {'start': 0.1, 'end': 0.5},
                 'modules': {
                     'mlp': {'format': 'nvfp4', 'group': 16, 'names': ['w1', 'w2', 'w3'], 'shared_global': [['w1', 'w3']],
                             'activations': {'format': 'nvfp4', 'group': 16,
                                             'calibration': {'method': 'max', 'refresh': 5, 'batches': 2}}}}}]}


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    path = tmp_path_factory.mktemp('tiny-lfm3')
    tiny_lfm3_hf().save_pretrained(path, safe_serialization=True)
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'sample': 1, '<think>': 2, '</think>': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    hf_tokenizer.chat_template = LFM3_TINY_TEMPLATE
    hf_tokenizer.save_pretrained(path)
    return path


def graph():
    from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
    raw = {'schema': SCHEMA, 'id': 'lfm3-qat', 'split': 'train', 'source_groups': ['lfm3-qat'],
           'tokenizer_fingerprint': '1' * 64,
           'nodes': [{'name': 'producer', 'token_ids': [4, 9, 10, 11, 5, 12], 'producing': [[0, 5]],
                      'target_weights': [0., 1., 1., 1., 1., 1.]},
                     {'name': 'consumer', 'token_ids': [6, 13, 14, 7, 15, 16, 17, 18, 19, 20, 21, 22],
                      'target_weights': [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      'imports': [{'at': 1, 'source': 'producer', 'start': 1, 'stop': 4}]}]}
    return parse_graph(raw, tokenizer_fingerprint='1' * 64)


def model_with_points(base, end=10):
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.quantization import Quantization, parse_native_component, stage_component
    torch.manual_seed(5)
    model, _, _, _ = load_native_model(base, rank=4, dtype=torch.bfloat16)
    names = [name for name, _ in model.named_parameters()]
    spec, bind = parse_native_component({**lfm3_points(), 'bind': {'stage': 'joint_sft', 'start': 0, 'end': end}})
    quantization = Quantization(stage_component(spec, bind['stage'], start=bind['start'], end=bind['end']))
    installed = quantization.install(model, inline=True)
    assert installed == {'attention': 12, 'mlp': 18}  # 3 attention layers x q/k/v/o (gate_proj excluded), 6 x w1-w3
    assert [name for name, _ in model.named_parameters()] == names  # plain names: no parametrization
    return model, quantization


def loss_of(model, example):
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    with torch.no_grad():
        return float(packed_graph_loss(model, [example], checkpoint_layers=False)[0])


def test_inline_points_are_exact_at_zero_and_quantize_above(base):
    from natlang_neuralese.train.native_state_sft import _quantization_pass
    from natlang_neuralese.train.quantization import Quantization
    example = graph()
    model, quantization = model_with_points(base)
    plain = loss_of(model, example)
    with quantization.context('fp8', 0.0):
        assert loss_of(model, example) == plain
    with quantization.context('fp8', 1.0):
        fp8 = loss_of(model, example)
    assert fp8 != plain
    with pytest.raises(RuntimeError, match='not calibrated'):
        with quantization.context('nvfp4', 1.0):
            loss_of(model, example)
    # Trainer pass choice: the calibration window opens at update 0 and commits after 2 batches.
    forward = lambda batch: loss_of(model, batch[0])  # noqa: E731
    passes = [_quantization_pass(quantization, global_step=step, batch=[example], forward=forward, cursor=step)
              for step in range(3)]
    assert quantization.calibration['step'] == 1 and quantization.calibration['batches'] == 2
    assert passes[0] == ('bf16', 0.0, 1.0)  # progress 0: no point ramped in yet
    amax = quantization.input_amax()
    assert len(amax) == 18 and all(value > 0 for value in amax.values())
    with quantization.context('nvfp4', 1.0):
        assert loss_of(model, example) != plain
    # Calibration state round-trip into a freshly installed component.
    fresh_model, fresh = model_with_points(base)
    fresh.load_state_dict(copy.deepcopy(quantization.state_dict()))
    assert fresh.input_amax() == amax and fresh.calibration == quantization.calibration
    assert isinstance(fresh, Quantization) and not fresh.calibration_due(3)


def test_resume_inside_calibration_window_continues_it_exactly(base):
    """A checkpoint taken inside an open NVFP4 calibration window carries it (start, batches, running maxima): the
    resumed component commits at the same update with the same ranges as an uninterrupted run, and the window's
    quantized training passes never enter the ranges."""
    from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
    from natlang_neuralese.train.native_state_sft import _quantization_pass
    first = graph()
    other = parse_graph({'schema': SCHEMA, 'id': 'second', 'split': 'train', 'source_groups': ['lfm3-qat'],
                         'tokenizer_fingerprint': '1' * 64,
                         'nodes': [{'name': 'only', 'token_ids': [9 + (i * 7) % 20 for i in range(26)],
                                    'target_weights': [0.] + [1.] * 25}]}, tokenizer_fingerprint='1' * 64)
    model, quantization = model_with_points(base)
    forward = lambda batch: loss_of(model, batch[0])  # noqa: E731
    _quantization_pass(quantization, global_step=0, batch=[first], forward=forward, cursor=0)
    saved = copy.deepcopy(quantization.state_dict())
    assert saved['window']['start'] == 0 and saved['window']['batches'] == 1 and saved['step'] is None
    with quantization.context('fp8', 1.0):  # a training pass inside the window observes nothing
        loss_of(model, other)
    _quantization_pass(quantization, global_step=1, batch=[other], forward=forward, cursor=1)
    resumed_model, resumed = model_with_points(base)
    resumed.load_state_dict(saved)
    resumed_forward = lambda batch: loss_of(resumed_model, batch[0])  # noqa: E731
    _quantization_pass(resumed, global_step=1, batch=[other], forward=resumed_forward, cursor=1)
    assert resumed.calibration == quantization.calibration == {
        'step': 1, 'batches': 2, 'source': {'start_global_step': 0, 'cursor': 1}}
    assert resumed.input_amax() == quantization.input_amax()
    assert 'window' not in resumed.state_dict()


def test_quantized_update_routes_straight_through_gradients_to_bf16_latents(base):
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, native_training_update
    example = graph()
    deltas = {}
    for point, mix in (('bf16', 0.0), ('fp8', 0.0), ('fp8', 1.0)):
        model, quantization = model_with_points(base)
        optimizer, _ = make_native_optimizer(model, lr=1e-2, weight_decay=0., learned_row_lr=1e-2)
        weights = {name: parameter for name, parameter in model.named_parameters()
                   if name.endswith(('feed_forward.w1.weight', 'self_attn.q_proj.weight'))}
        assert weights and all(parameter.dtype == torch.bfloat16 for parameter in weights.values())
        before = {name: value.detach().float().clone() for name, value in weights.items()}
        torch.manual_seed(11)
        loss, _ = native_training_update(model, optimizer, [example], checkpoint_layers=True,
                                         precision=(quantization, point, mix, 1.0))
        deltas[(point, mix)] = (float(loss.detach()), {name: weights[name].detach().float() - before[name] for name in weights})
        assert all(delta.abs().sum() > 0 for delta in deltas[(point, mix)][1].values())
    # λ = 0 is the unquantized update bit for bit; the quantized forward changes the loss and the update.
    assert deltas[('fp8', 0.0)][0] == deltas[('bf16', 0.0)][0]
    assert all(torch.equal(deltas[('fp8', 0.0)][1][n], deltas[('bf16', 0.0)][1][n]) for n in deltas[('bf16', 0.0)][1])
    assert deltas[('fp8', 1.0)][0] != deltas[('bf16', 0.0)][0]
    assert any(not torch.equal(deltas[('fp8', 1.0)][1][n], deltas[('bf16', 0.0)][1][n]) for n in deltas[('bf16', 0.0)][1])


def test_native_component_declaration_forms():
    from natlang_neuralese.train.quantization import parse_native_component
    assert parse_native_component(None) is None and parse_native_component('off') == 'off'
    declared = {**lfm3_points(), 'bind': {'stage': 'joint_sft', 'start': 0, 'end': 100}}
    spec, bind = parse_native_component(declared)
    recorded = {'schema': 'natlang.native-inline-qat/1', 'spec': spec, 'bind': bind}
    assert parse_native_component(recorded) == (spec, bind)
    with pytest.raises(ValueError, match='loss mode "sample"'):
        parse_native_component({**declared, 'loss': {'mode': 'sum'}})
    with pytest.raises(ValueError, match='bind'):
        parse_native_component(lfm3_points())


def test_lfm3_default_component_is_adopted_recorded_and_continued(base):
    """Omitted --quantization: an LFM3 SFT run adopts the recipe's FP8/NVFP4 component over its remaining planned
    updates (recorded at that update); a resume continues the recorded declaration; `off` opts out explicitly."""
    from types import SimpleNamespace
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.native_state_sft import _setup_quantization
    args = SimpleNamespace(quantization=None, fp8_training='off')
    bind = {'stage': 'joint_sft', 'start': 40, 'end': 140}
    model, _, _, _ = load_native_model(base, rank=4, dtype=torch.bfloat16)
    options = {}
    quantization = _setup_quantization(args, model, None, options, saved_options={}, default_bind=bind)
    declaration = options['quantization']
    assert quantization is not None and declaration['bind'] == bind
    assert [p['name'] for p in declaration['spec']['points']] == ['fp8', 'nvfp4']
    assert options['quantization_history'][0] == {'event': 'default-adopted', 'at_global_step': 40,
                                                  'weights_optimizer_rng_exposure': 'preserved'}
    assert options['quantization_history'][1]['sha256'] == declaration['sha256']
    assert quantization.progress(90) == 0.5
    resumed, resumed_options = load_native_model(base, rank=4, dtype=torch.bfloat16)[0], {}
    _setup_quantization(args, resumed, quantization.state_dict(), resumed_options,
                        saved_options={'quantization': declaration, 'quantization_history': options['quantization_history']},
                        default_bind={'stage': 'joint_sft', 'start': 90, 'end': 200})
    assert resumed_options['quantization']['bind'] == bind  # the recorded ramp continues
    assert resumed_options['quantization_history'] == options['quantization_history']
    off = {}
    assert _setup_quantization(SimpleNamespace(quantization='off', fp8_training='off'),
                               load_native_model(base, rank=4, dtype=torch.bfloat16)[0], None, off,
                               saved_options={}, default_bind=bind) is None
    assert off['quantization'] is None


@pytest.mark.gpu_heavy
@pytest.mark.skipif(not torch.cuda.is_available(), reason='the fused pass ramp runs on CUDA')
def test_in_training_deploy_weights_equal_liquid_exports():
    """Export parity: the λ = 1 weights an inline QAT pass trains through (the compiled per-pass ramp, with the w1/w3
    shared NVFP4 global scale as the pass resolves it) are the BF16 values of the deployed codes: the dequantized
    `export.lfm3_quantized` state, which reproduces Liquid's NVFP4/FP8 checkpoints bit for bit
    (test_qat_deploy_formats)."""
    from test_qat_deploy_formats import BASE, LIQUID, NVFP4
    if not LIQUID:
        pytest.skip("Liquid's LFM3 checkpoints are not in the HF cache")
    from safetensors import safe_open
    from natlang_neuralese.export import lfm3_quantized as X
    from natlang_neuralese.maple.ternary import _ramped_weight, quantized_value
    one = torch.ones((), device='cuda')
    with safe_open(str(BASE / 'model.safetensors'), 'pt', 'cuda') as base, \
            safe_open(str(NVFP4 / 'model.safetensors'), 'pt', 'cuda') as nv:
        for layer in (0, 13, 23):
            prefix = f'model.layers.{layer}.feed_forward.'
            tensors = {prefix + f'{leaf}.weight': base.get_tensor(prefix + f'{leaf}.weight')
                       for leaf in ('w1', 'w2', 'w3')}
            scales = {prefix + leaf: nv.get_tensor(prefix + f'{leaf}.input_scale') for leaf in ('w1', 'w2', 'w3')}
            for fmt in ('nvfp4', 'fp8_block'):
                exported = X.quantize_state(fmt, tensors, ['conv'] * 24, device='cuda',
                                            input_scales=scales if fmt == 'nvfp4' else None)
                exported = {k: v.to('cuda') for k, v in exported.items()}
                for leaf in ('w1', 'w2', 'w3'):
                    weight = tensors[prefix + f'{leaf}.weight']
                    options = {'group': 16} if fmt == 'nvfp4' else {'block': 128}
                    if fmt == 'nvfp4' and leaf != 'w2':
                        options['global_amax'] = torch.stack([tensors[prefix + f'{n}.weight'].abs().amax()
                                                              for n in ('w1', 'w3')]).amax()
                    trained = _ramped_weight(weight, fmt, options, one)
                    deployed = X._dequantize(fmt, exported, prefix + leaf)
                    assert torch.equal(trained, deployed.to(torch.bfloat16)), (layer, fmt, leaf)
                    assert torch.equal(trained, quantized_value(weight, fmt, options).to(torch.bfloat16))
