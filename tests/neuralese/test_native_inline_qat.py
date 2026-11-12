"""Inline QAT in the native trainer (``--quantization``): the LFM3 FP8/NVFP4 points on a tiny BF16 LFM3.

Covers: plain parameter names after installation; λ = 0 equals the unquantized model exactly and λ > 0 differs, on
both the plain path and the BF16 cotangent-carrier path of a real update; gradients reach the latents through the
straight-through route; the trainer's calibration window and pass choice; calibration state round-trip.
"""
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


def model_with_points(base, end=10, profile='source-v1',device='cpu'):
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.quantization import Quantization, parse_native_component, stage_component
    torch.manual_seed(5)
    model, _, _, _ = load_native_model(base, rank=4, dtype=torch.bfloat16,device=device)
    if profile!='source-v1':
        from natlang_neuralese.model.shared_execution import configure_shared_execution
        configure_shared_execution(model,profile)
    names = [name for name, _ in model.named_parameters()]
    spec, bind = parse_native_component({**lfm3_points(), 'bind': {'stage': 'joint_sft', 'start': 0, 'end': end}})
    quantization = Quantization(stage_component(spec, bind['stage'], start=bind['start'], end=bind['end']))
    installed = quantization.install(model, inline=True)
    assert installed == {'attention': 12, 'mlp': 18}  # 3 attention layers x q/k/v/o (gate_proj excluded), 6 x w1-w3
    assert [name for name, _ in model.named_parameters()] == names  # plain names: no parametrization
    return model, quantization


def test_inline_precision_rejects_unknown_forward_even_with_native_flags(base):
    from types import MethodType
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.model.shared_execution import configure_shared_execution
    from natlang_neuralese.train.quantization import Quantization,stage_component
    model,*_=load_native_model(base,rank=4,dtype=torch.bfloat16)
    configure_shared_execution(model,'ffn-stable-v1')
    projection=next(m for n,m in model.named_modules() if n.endswith('feed_forward.w1'))
    projection.forward=MethodType(lambda self,x: x,projection)
    quantization=Quantization(stage_component(lfm3_points(),'joint_sft',start=0,end=10))
    with pytest.raises(ValueError,match='plain Linear'):
        quantization.install(model,inline=True)


def loss_of(model, example):
    from natlang_neuralese.train.native_graph_loss import packed_graph_loss
    with torch.no_grad():
        return float(packed_graph_loss(model, [example], checkpoint_layers=False)[0])


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
