"""LFM3-1.7B real-weight qualification of the native port against the vendored pinned reference (owner 2026-10-28).

(a) logits parity of ``Lfm3PortBackbone`` vs the reference forward, FP32 (IEEE) and BF16, on a sequence several
sliding windows (256) long; (b) gradient parity for parameters of every layer form; (c) the one-pass shared native
graph vs explicit cached execution (producer step, captured native state, splice, consumer steps) with a producer
span imported into a consumer whose later queries leave the sliding window of the imported slots, so the import
crosses conv, full-attention and sliding-attention layers differently; (d) incremental decode with the cache equals
the full forward across the window boundary. Needs the pinned local snapshot and CUDA.
"""
import copy

import pytest
import torch

from natlang_neuralese.model.lfm2_port import ControlTokens, PortCache

pytestmark = [pytest.mark.gpu_heavy,
              pytest.mark.skipif(not torch.cuda.is_available(), reason='LFM3 real-weight parity runs on CUDA')]


@pytest.fixture(scope='module')
def lfm3():
    from natlang_neuralese.model.lfm3_port import load_lfm3_backbone, resolve_base
    try:
        resolve_base()
    except FileNotFoundError:
        pytest.skip('pinned LFM3-1.7B snapshot is not in the local HF cache')
    model, tokenizer = load_lfm3_backbone(dtype=torch.float32, device='cuda')
    return model, tokenizer


def _text_ids(tokenizer, count):
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / 'AGENTS.md').read_text()
    ids = tokenizer(text, add_special_tokens=False).input_ids[:count - 1]
    assert len(ids) == count - 1
    return torch.tensor([[tokenizer.bos_token_id, *ids]], device='cuda')


def _port(model, **kwargs):
    from natlang_neuralese.model.lfm3_port import Lfm3PortBackbone
    return Lfm3PortBackbone(model, ControlTokens(125033, 125034), markers=False, **kwargs)


def _relative(actual, expected, label=None):
    value = float((actual - expected).abs().max()) / float(expected.abs().max())
    if label is not None:  # measured values for the record (pytest -s)
        print(f'LFM3-PARITY {label}: max|diff| {float((actual - expected).abs().max()):.3g} '
              f'scale {float(expected.abs().max()):.3g} relative {value:.3g}')
    return value


def test_role_tokens_land_in_unused_rows_without_resize(lfm3):
    from natlang_neuralese.model.shared_state import provision_role_tokens
    from types import SimpleNamespace
    model, tokenizer = lfm3
    tokenizer = copy.deepcopy(tokenizer)
    assert max(tokenizer.get_vocab().values()) == 125032
    assert '<|reserved_7|>' not in tokenizer.get_vocab()  # the LFM2 marker path cannot apply
    weight = model.get_input_embeddings().weight
    roles = provision_role_tokens(SimpleNamespace(hf=model, config=model.config, embedding_weight=weight), tokenizer)
    assert roles.ids == (125033, 125034, 125035, 125036)
    assert model.get_input_embeddings().weight is weight and weight.shape[0] == 128000


@pytest.mark.parametrize('fast', [True, False])
def test_fp32_logits_match_reference_across_sliding_windows(lfm3, fast):
    model, tokenizer = lfm3
    ids = _text_ids(tokenizer, 701)
    with torch.no_grad():
        expected = model(ids).logits
        actual = _port(model, fast=fast).forward_ids(ids)['logits']
    # Measured 2026-10-28 (IEEE FP32, GB10): max |diff| 4.8e-5 on logits of max magnitude 27.9.
    assert _relative(actual, expected, f'fp32 logits fast={fast}') < 1e-5
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))


def test_bf16_logits_match_reference_at_its_precision(lfm3):
    model, tokenizer = lfm3
    ids = _text_ids(tokenizer, 701)
    half = copy.deepcopy(model).to(torch.bfloat16)
    with torch.no_grad():
        exact = torch.log_softmax(model(ids).logits.float(), -1)
        reference = torch.log_softmax(half(ids).logits.float(), -1)
        actual = torch.log_softmax(_port(half).forward_ids(ids)['logits'].float(), -1)
    del half
    kl = lambda logp: (exact.exp() * (exact - logp)).sum(-1)
    reference_kl, actual_kl = kl(reference), kl(actual)
    print(f'LFM3-PARITY bf16 KL vs fp32: port mean {float(actual_kl.mean()):.3g} max {float(actual_kl.max()):.3g}; '
          f'reference mean {float(reference_kl.mean()):.3g} max {float(reference_kl.max()):.3g}; '
          f'port vs reference bf16 logprob max|diff| {float((actual - reference).abs().max()):.3g}')
    # The port's BF16 kernels (FlexAttention windows, its conv/RoPE order) differ from the reference's SDPA path in
    # rounding only: both are equally far from FP32. Measured: mean KL 6.8e-4 (both), max 6.3e-3 (port) vs
    # 7.4e-3 (reference).
    assert float(actual_kl.mean()) <= 1.1 * float(reference_kl.mean())
    assert float(actual_kl.max()) <= 2 * float(reference_kl.max())
    agree = (actual.argmax(-1) == exact.argmax(-1)).float().mean()
    reference_agree = (reference.argmax(-1) == exact.argmax(-1)).float().mean()
    assert agree >= reference_agree - 0.01


def test_fp32_gradients_match_reference(lfm3):
    model, tokenizer = lfm3
    names = ['model.layers.0.conv.conv.weight', 'model.layers.1.conv.in_proj.weight',
             'model.layers.2.self_attn.gate_proj.weight', 'model.layers.2.self_attn.q_norm.weight',
             'model.layers.4.self_attn.k_proj.weight', 'model.layers.4.self_attn.gate_proj.weight',
             'model.layers.8.post_operator_norm.weight', 'model.layers.21.post_ffn_norm.weight',
             'model.layers.22.self_attn.o_proj.weight', 'model.embedding_norm.weight']
    parameters = dict(model.named_parameters())
    watched = [parameters[name] for name in names]
    ids = _text_ids(tokenizer, 321)
    port = _port(model)
    try:
        for parameter in watched:
            parameter.requires_grad_(True)
        def loss(logits):
            return torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
        expected = torch.autograd.grad(loss(model(ids).logits), watched)
        actual = torch.autograd.grad(loss(port.forward_ids(ids)['logits']), watched)
    finally:
        for parameter in watched:
            parameter.requires_grad_(False)
    for name, left, right in zip(names, actual, expected):
        assert right.abs().sum() > 0, name
        # Measured 2026-10-28 (IEEE FP32, GB10): largest relative max-difference 2.0e-6 (sliding k_proj).
        assert _relative(left, right, f'fp32 grad {name}') < 1e-4, name


def _shared_model(model, tokenizer):
    from natlang_neuralese.model.shared_state import SharedStateModel, provision_role_tokens
    from types import SimpleNamespace
    tokenizer = copy.deepcopy(tokenizer)
    weight = model.get_input_embeddings().weight
    roles = provision_role_tokens(SimpleNamespace(hf=model, config=model.config, embedding_weight=weight), tokenizer)
    from natlang_neuralese.model.lfm3_port import Lfm3PortBackbone
    port = Lfm3PortBackbone(model, ControlTokens(roles.produce_open, roles.produce_close), markers=False)
    shared = SharedStateModel(port, roles.ids, rank=8).to('cuda')
    torch.manual_seed(7)
    torch.nn.init.normal_(shared.adapter.up.weight, std=0.03)
    return shared


def test_one_pass_shared_graph_matches_explicit_cached_execution_with_gradients(lfm3):
    from types import SimpleNamespace
    from natlang_neuralese.model.shared_state import (ImportSpan, Invocation, compile_graph, native_step,
                                                      splice_block)
    model, tokenizer = lfm3
    shared = _shared_model(model, tokenizer)
    ids = _text_ids(tokenizer, 601)[0].tolist()
    source, target = tuple(ids[:300]), tuple(ids[300:590])
    # 60 produced positions imported at consumer position 20: the consumer's last ~70 queries are more than 256
    # logical positions past the imported slots, so sliding layers no longer see them while full layers do.
    plan = compile_graph([
        Invocation('source', source, producing=((100, 160),)),
        Invocation('target', target, imports=(ImportSpan(20, 'source', 100, 160),)),
    ])
    names = ['model.layers.1.conv.in_proj.weight', 'model.layers.2.self_attn.k_proj.weight',
             'model.layers.4.self_attn.gate_proj.weight', 'model.layers.4.self_attn.v_proj.weight',
             'model.layers.23.post_operator_norm.weight']
    parameters = dict(model.named_parameters())
    watched = [shared.adapter.up.weight, *(parameters[name] for name in names)]
    try:
        for parameter in watched[1:]:
            parameter.requires_grad_(True)
        flat = shared(plan)
        producing = torch.zeros(1, len(source), dtype=torch.bool, device='cuda')
        producing[:, 100:160] = True
        producer = native_step(shared, torch.tensor([source], device='cuda'), producing=producing)
        identity = dict(checkpoint_sha256='a' * 64, tokenizer_fingerprint='b' * 64, embedding_weight_sha256='c' * 64)
        block = SimpleNamespace(
            positions=torch.arange(100, 160, device='cuda'), token_ids=producer['token_ids'][100:160],
            embeddings=producer['embeddings'][100:160], architecture='lfm3', validate=lambda: None,
            layers=tuple({key: value[100:160] if key != 'kind' else value for key, value in layer.items()}
                         for layer in producer['layers']), **identity)
        prefix = native_step(shared, torch.tensor([target[:20]], device='cuda'))
        cache = splice_block(shared, prefix['cache'], block, **identity)
        suffix = native_step(shared, torch.tensor([target[20:]], device='cuda'), cache)
        explicit = torch.cat([producer['h_final'], prefix['h_final'], suffix['h_final']], 1)
        # Measured 2026-10-28 (IEEE FP32): h 8.2e-7, logits 2.0e-6, gradients at most 6.7e-6 relative.
        assert _relative(flat['h_final'], explicit, 'graph h_final') < 2e-5
        logits_flat = shared.logits(flat['h_final'][:, -40:])
        logits_explicit = shared.logits(explicit[:, -40:])
        assert _relative(logits_flat, logits_explicit, 'graph logits') < 2e-5
        generator = torch.Generator(device='cuda').manual_seed(11)
        weight = torch.randn(explicit.shape, device='cuda', generator=generator)
        expected = torch.autograd.grad((explicit * weight).sum(), watched)
        actual = torch.autograd.grad((flat['h_final'] * weight).sum(), watched)
    finally:
        for parameter in watched[1:]:
            parameter.requires_grad_(False)
    for name, left, right in zip(['adapter.up', *names], actual, expected):
        assert right.abs().sum() > 0, name
        assert _relative(left, right, f'graph grad {name}') < 2e-4, name


def test_incremental_decode_equals_full_forward_across_window(lfm3):
    from natlang_neuralese.model.shared_state import Invocation, compile_graph, native_step
    model, tokenizer = lfm3
    ids = _text_ids(tokenizer, 280)
    port = _port(model)
    with torch.no_grad():
        full = port.forward_ids(ids)['logits']
        result = port.forward_ids(ids[:, :250])
        pieces, cache = [result['logits']], result['cache']
        for index in range(250, 280):  # decode steps cross the 256-position sliding window
            result = port.forward_ids(ids[:, index:index + 1], cache=cache)
            pieces.append(result['logits'])
            cache = result['cache']
        assert _relative(torch.cat(pieces, 1), full, 'port decode logits') < 1e-5
        shared = _shared_model(model, tokenizer)
        graph = shared(compile_graph([Invocation('a', tuple(ids[0].tolist()))]))['h_final']
        step = native_step(shared, ids[:, :250])
        steps, cache = [step['h_final']], step['cache']
        for index in range(250, 280):
            step = native_step(shared, ids[:, index:index + 1], cache)
            steps.append(step['h_final'])
            cache = step['cache']
        assert _relative(torch.cat(steps, 1), graph, 'native_step decode h') < 1e-5
        torch.testing.assert_close(shared.logits(graph), full, atol=1e-3, rtol=1e-4)
