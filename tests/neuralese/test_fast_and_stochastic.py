import copy
import math

import pytest
import torch

from natlang_neuralese.model.heads import payload_kl, payload_log_prob, sample_payload
from natlang_neuralese.model.lfm2_port import AttentionState, PortBackbone, append_kv
from natlang_neuralese.train.execution import prefill, unroll_write
from natlang_neuralese.train.losses import policy_gradient_surrogate
from natlang_neuralese.write import greedy_continue, open_block, read_back, write_block

ATOL = 2e-4


def ids_for(tokenizer, backbone, text="The harbour master logged every ship that arrived after dark"):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    return torch.cat([ids, torch.tensor([[backbone.controls.open_id]])], 1)


@pytest.fixture(scope="module")
def reference(loaded):
    """The same backbone on the original reference path."""
    _, _, backbone = loaded
    ref = copy.copy(backbone)
    ref.fast = False
    return ref


@torch.no_grad()
def test_fast_path_matches_reference(loaded, reference, heads):
    _, tokenizer, backbone = loaded
    assert backbone.fast and not reference.fast
    ids = ids_for(tokenizer, backbone)
    for chunks in ((ids,), (ids[:, :4], ids[:, 4:5], ids[:, 5:])):
        caches = [None, None]
        for chunk in chunks:
            a = backbone.forward_ids(chunk, cache=caches[0], cutoff=6)
            b = reference.forward_ids(chunk, cache=caches[1], cutoff=6)
            torch.testing.assert_close(a["logits"], b["logits"], atol=ATOL, rtol=1e-4)
            caches = [a["cache"], b["cache"]]
    fast = write_block(backbone, heads, open_block(backbone, heads, ids), max_length=5)
    ref = write_block(reference, heads, open_block(reference, heads, ids), max_length=5)
    torch.testing.assert_close(fast.payload, ref.payload, atol=ATOL, rtol=1e-4)
    back_fast = read_back(backbone, heads, fast.block_start, fast.payload)
    back_ref = read_back(reference, heads, ref.block_start, ref.payload)
    torch.testing.assert_close(back_fast["logits"], back_ref["logits"], atol=ATOL, rtol=1e-4)


@torch.no_grad()
def test_static_kv_snapshots_survive_branching(loaded, heads):
    _, tokenizer, backbone = loaded
    ids = ids_for(tokenizer, backbone)
    opened = open_block(backbone, heads, ids)
    written = write_block(backbone, heads, opened, max_length=4)
    assert any(isinstance(s, AttentionState) and s.buffer is not None for s in opened.cache.states)
    first = read_back(backbone, heads, opened.cache, written.payload)
    # A second branch from the same snapshot (different payload) must not disturb the first.
    other = read_back(backbone, heads, opened.cache, torch.zeros_like(written.payload))
    again = read_back(backbone, heads, opened.cache, written.payload)
    torch.testing.assert_close(first["logits"], again["logits"], rtol=0, atol=0)
    assert not torch.allclose(first["logits"], other["logits"])
    tokens_a, _ = greedy_continue(backbone, first["cache"], first["logits"], steps=4)
    tokens_b, _ = greedy_continue(backbone, again["cache"], again["logits"], steps=4)
    assert tokens_a == tokens_b


def test_append_kv_copy_on_write():
    k = torch.arange(8.0).view(1, 1, 8, 1)
    with torch.no_grad():
        base = append_kv(None, k[:, :, :4], k[:, :, :4], static=True)
        longer = append_kv(base, k[:, :, 4:6], k[:, :, 4:6], static=True)
        assert longer.buffer is base.buffer  # in place at the frontier
        branch = append_kv(base, -k[:, :, 4:5], -k[:, :, 4:5], static=True)
        assert branch.buffer is not base.buffer  # base is behind the frontier: copy first
    torch.testing.assert_close(longer.k[0, 0, :, 0], torch.arange(6.0))
    torch.testing.assert_close(branch.k[0, 0, :, 0], torch.tensor([0.0, 1, 2, 3, -4]))
    grown = append_kv(base, k, k, static=False)
    assert grown.buffer is None and grown.k.shape[2] == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.no_grad()
def test_gpu_fast_path(loaded):
    """On the GPU: fast (flash/cuDNN SDPA, static KV, conv kernel) equals the reference in fp32,
    and in bf16 the fast path's cached write/readback agrees with a from-scratch recompute.
    (Fast and reference paths are not compared token-for-token in bf16: different attention
    backends round differently, and an untrained payload makes near-ties common.)"""
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.model.lfm2_port import ControlTokens, load_backbone, load_conv_kernel

    if torch.cuda.mem_get_info()[0] < 4 * 2**30:
        pytest.skip("not enough free GPU memory on this shared machine")
    _, tokenizer, _ = loaded
    kernel = load_conv_kernel()
    long_text = " ".join(["The harbour master logged every ship that arrived after dark."] * 8)
    for dtype in (torch.float32, torch.bfloat16):
        model, _ = load_backbone(dtype=dtype, device="cpu")
        model.to("cuda")
        fast = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer), conv_kernel=kernel).cuda()
        ref = copy.copy(fast)
        ref.fast = False
        torch.manual_seed(1)
        heads = PortHeads(fast, cutoff=6, max_length=8).cuda().eval()
        ids = ids_for(tokenizer, fast, long_text).cuda()
        if dtype == torch.float32:
            torch.testing.assert_close(fast.forward_ids(ids)["logits"], ref.forward_ids(ids)["logits"], atol=1e-3, rtol=1e-4)
            wa = write_block(fast, heads, open_block(fast, heads, ids), max_length=6)
            wb = write_block(ref, heads, open_block(ref, heads, ids), max_length=6)
            torch.testing.assert_close(wa.payload, wb.payload, atol=1e-3, rtol=1e-4)
        else:
            opened = open_block(fast, heads, ids)
            written = write_block(fast, heads, opened, max_length=6)
            back = read_back(fast, heads, written.block_start, written.payload)
            tokens, cached = greedy_continue(fast, back["cache"], back["logits"], steps=6)
            close = torch.tensor([[fast.controls.close_id]], device="cuda")
            sequence = torch.cat([fast.embed(ids), heads.interface(written.payload), fast.embed(close),
                                  fast.embed(torch.tensor([tokens], device="cuda"))], 1)
            full = fast.forward_embeds(sequence)["logits"][0]
            start = ids.shape[1] + written.payload.shape[1]
            assert full[start: start + len(tokens)].argmax(-1).tolist() == tokens
        del model, fast, ref, heads
        torch.cuda.empty_cache()


@torch.no_grad()
def test_temperature_zero_is_deterministic_mean(loaded, heads, device):
    _, tokenizer, backbone = loaded
    opened = open_block(backbone, heads, ids_for(tokenizer, backbone))
    cold = write_block(backbone, heads, opened, max_length=4)
    torch.testing.assert_close(cold.payload, cold.mean, rtol=0, atol=0)
    assert cold.log_prob is None and cold.temperature == 0.0
    hot1 = write_block(backbone, heads, opened, max_length=4, temperature=1.0, generator=torch.Generator(device=device).manual_seed(5))
    hot2 = write_block(backbone, heads, opened, max_length=4, temperature=1.0, generator=torch.Generator(device=device).manual_seed(5))
    torch.testing.assert_close(hot1.payload, hot2.payload, rtol=0, atol=0)
    torch.testing.assert_close(hot1.mean, cold.mean, atol=ATOL, rtol=1e-4)
    assert not torch.allclose(hot1.payload, hot1.mean)
    # Noise is small at initialisation: sigma starts at 0.05 of the vector scale.
    relative = (hot1.payload - hot1.mean).float().pow(2).mean().sqrt() / hot1.mean.float().pow(2).mean().sqrt()
    assert 0.01 < float(relative) < 0.2
    assert hot1.log_prob.shape == (1,) and torch.isfinite(hot1.log_prob).all()


def test_payload_log_prob_and_kl_formulas():
    torch.manual_seed(0)
    mu = torch.randn(2, 3, 8)
    log_sigma = torch.full((2, 3, 8), math.log(0.3))
    sample = sample_payload(mu, log_sigma, temperature=0.5, generator=torch.Generator().manual_seed(1))
    scale = mu.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    u_z, u_mu = sample.payload / scale, mu / scale
    dist = torch.distributions.Normal(u_mu, 0.5 * 0.3)
    torch.testing.assert_close(payload_log_prob(sample), dist.log_prob(u_z).sum(-1), atol=1e-4, rtol=1e-4)
    kl = payload_kl(sample)
    expected = torch.distributions.kl_divergence(torch.distributions.Normal(u_mu, 0.3),
                                                 torch.distributions.Normal(0.0, 1.0)).mean(-1)
    torch.testing.assert_close(kl, expected, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        payload_log_prob(sample_payload(mu, log_sigma, temperature=0.0))


def test_policy_gradient_reaches_mean_and_scale(loaded, device):
    from natlang_neuralese.model.heads import PortHeads

    _, tokenizer, backbone = loaded
    torch.manual_seed(2)
    heads = PortHeads(backbone, cutoff=6, max_length=4)
    pre = prefill(backbone, heads, ids_for(tokenizer, backbone))
    written = unroll_write(backbone, heads, pre, length=3, temperature=1.0, generator=torch.Generator(device=device).manual_seed(0))
    loss = policy_gradient_surrogate(written, torch.tensor([1.5]))
    loss = loss + 0.1 * written.kl()
    loss.backward()
    assert heads.content.log_sigma.bias.grad.abs().sum() > 0
    assert heads.content.proj.weight.grad.abs().sum() > 0
    assert heads.feedback.mlp_out.weight.grad.abs().sum() > 0  # through mu into the sketch recurrence
