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


def test_quantized_update_routes_straight_through_gradients_to_bf16_latents(base):
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, native_training_update
    example = graph()
    deltas = {}
    for point, mix in (('bf16', 0.0), ('fp8', 0.0), ('fp8', 1.0)):
        model, quantization = model_with_points(base)
        optimizer, _ = make_native_optimizer(model, lr=1e-2, weight_decay=0., in_backward=False, learned_row_lr=1e-2)
        weights = {name: parameter for name, parameter in model.named_parameters()
                   if name.endswith(('feed_forward.w1.weight', 'self_attn.q_proj.weight'))}
        assert weights and all(parameter.dtype == torch.bfloat16 for parameter in weights.values())
        before = {name: value.detach().float().clone() for name, value in weights.items()}
        torch.manual_seed(11)
        loss, _ = native_training_update(model, optimizer, [example], checkpoint_layers=True,
                                         precision=(quantization, point, mix, 1.0))
        deltas[(point, mix)] = (float(loss), {name: weights[name].detach().float() - before[name] for name in weights})
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
