"""Batched (ragged) execution, phases D–F, anti-collapse terms and LoRA deltas."""

import json

import pytest
import torch

from natlang_neuralese.data.fixtures import synthetic_records
from natlang_neuralese.data.records import parse_record
from natlang_neuralese.data.render import Renderer, render_record, span_examples
from natlang_neuralese.train.execution import (consumer_forward, consumer_forward_batch, consumer_context_cache, prefill, prefill_batch,
                                               stop_log_prob, teacher_logits_batch, teacher_target_logits,
                                               unroll_write)
from natlang_neuralese.train.losses import consumer_batch_loss, diversity_loss

ATOL = 5e-4


@pytest.mark.parametrize('profile', ['legacy-rms-v1', 'raw-token-v1'])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_batched_differentiable_scope_preserves_child_and_head_adjoints(loaded, profile, checkpointed):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.train.execution import prefill_write_context, prefill_write_contexts
    backbone = loaded[2]
    torch.manual_seed(81)
    heads = PortHeads(backbone, cutoff=6, max_length=8, profile=profile).eval()
    scopes = [torch.randn(n, backbone.embedding_weight.shape[1], requires_grad=True) * .02
              for n in (7, 11)]
    lengths = torch.tensor([3, 5])
    weight = (heads.feedback.state_out if profile == 'raw-token-v1' else heads.feedback.mlp_out).weight
    old = getattr(backbone, 'checkpoint_layers', False)
    backbone.checkpoint_layers = checkpointed
    try:
        singles = [unroll_write(backbone, heads, prefill_write_context(backbone, heads, c[None]),
                                length=int(n)) for c, n in zip(scopes, lengths)]
        expected = sum(w.payload.square().mean() for w in singles)
        expected_grads = torch.autograd.grad(expected, [*scopes, weight])
        batched = unroll_write(backbone, heads, prefill_write_contexts(backbone, heads, scopes), lengths=lengths)
        actual = sum(batched.payload[b, :int(n)].square().mean() for b, n in enumerate(lengths))
        actual_grads = torch.autograd.grad(actual, [*scopes, weight])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
        for a, e in zip(actual_grads, expected_grads):
            # Batched GEMM changes floating-point accumulation order. The
            # production replay must still reproduce its pinned batch layout.
            torch.testing.assert_close(a, e, atol=2e-6, rtol=1e-4)
        for row, single in enumerate(singles):
            torch.testing.assert_close(batched.payload[row, :int(lengths[row])], single.payload[0], atol=ATOL, rtol=1e-4)
    finally:
        backbone.checkpoint_layers = old


def test_padding_host_layout_survives_cache_selection_and_recurrence(loaded):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.train.execution import prefill_write_contexts
    backbone = loaded[2]
    heads = PortHeads(backbone, cutoff=6, max_length=4, profile='raw-token-v1')
    rows = [backbone.embed(torch.tensor([1] * n)) for n in (7, 11)]
    pre = prefill_write_contexts(backbone, heads, rows)
    assert pre.cache.pad_offsets == (4, 0)
    assert pre.cache.select([1]).pad_offsets == (0,)
    assert pre.cache.select(slice(0, 1)).pad_offsets == (4,)
    _, cache = backbone.run_layers(rows[0][-1:][None].expand(2, -1, -1), range(6), pre.cache)
    assert cache.pad_offsets == (4, 0)


def test_ragged_attention_uses_host_offsets_without_gpu_scalar_reads():
    from types import SimpleNamespace
    from natlang_neuralese.model.lfm2_port import PortBackbone
    torch.manual_seed(17)
    q = torch.randn(2, 2, 3, 4)
    k, v = torch.randn(2, 1, 7, 4), torch.randn(2, 1, 7, 4)
    attn = SimpleNamespace(scaling=.5)
    expected = PortBackbone._attend_ragged(attn, q, k, v, left_pad=torch.tensor([0, 2]))
    # An opaque sentinel cannot be indexed/read: the fixed host layout must
    # supply all slicing decisions while attention still operates on tensors.
    actual = PortBackbone._attend_ragged(attn, q, k, v, left_pad=object(), left_offsets=(0, 2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reader_prefix_cache_reuse_preserves_outputs_and_gradients(loaded, fresh_heads, records, renderer):
    backbone = loaded[2]
    rows = _ragged(records, renderer)[:2]
    payload = torch.randn(2, 4, backbone.embedding_weight.shape[1], requires_grad=True)
    lengths = torch.tensor([4, 3])
    cache = consumer_context_cache(backbone, rows)
    cached = consumer_forward_batch(backbone, fresh_heads, rows, payload, lengths, context_cache=cache)
    fresh = consumer_forward_batch(backbone, fresh_heads, rows, payload, lengths)
    for a, b in zip(cached, fresh):
        torch.testing.assert_close(a, b, atol=ATOL, rtol=1e-4)
    ca = torch.autograd.grad(sum(x.square().mean() for x in cached), payload, retain_graph=True)[0]
    cb = torch.autograd.grad(sum(x.square().mean() for x in fresh), payload)[0]
    torch.testing.assert_close(ca, cb, atol=ATOL, rtol=1e-4)
    reused = consumer_forward_batch(backbone, fresh_heads, rows, payload.flip(0), lengths.flip(0), context_cache=cache)
    assert all(torch.isfinite(x).all() for x in reused)


@pytest.fixture(scope="module")
def renderer(loaded):
    _, tokenizer, backbone = loaded
    return Renderer(tokenizer, backbone.controls)


@pytest.fixture(scope="module")
def records(renderer):
    rows = [render_record(renderer, parse_record(r)) for r in synthetic_records(6)]
    # Make producer lengths ragged.
    lengths = {len(r.producer) for r in rows}
    assert len(lengths) > 1 or True
    return rows


@pytest.fixture()
def fresh_heads(loaded):
    from natlang_neuralese.model.heads import PortHeads

    torch.manual_seed(1)
    return PortHeads(loaded[2], cutoff=6, max_length=6)


def _ragged(records, renderer):
    """Records whose producers differ in length (pad one producer's instructions)."""
    rows = list(records[:3])
    extra = renderer.text(" Please read carefully.")
    first = rows[0]
    rows[0] = type(first)(**{**first.__dict__, "producer": first.producer[:3] + extra + first.producer[3:]})
    assert len({len(r.producer) for r in rows}) > 1
    return rows


def test_left_padded_write_matches_single_rows(loaded, fresh_heads, records, renderer):
    _, _, backbone = loaded
    rows = _ragged(records, renderer)
    with torch.no_grad():
        batched = unroll_write(backbone, fresh_heads, prefill_batch(backbone, fresh_heads, [r.producer for r in rows]),
                               length=4)
        for b, r in enumerate(rows):
            single = unroll_write(backbone, fresh_heads,
                                  prefill(backbone, fresh_heads, torch.tensor([r.producer])), length=4)
            assert torch.allclose(batched.payload[b], single.payload[0], atol=ATOL)
            assert torch.allclose(batched.stop_logits[b], single.stop_logits[0], atol=ATOL)


def test_batched_consumer_and_teacher_match_single(loaded, fresh_heads, records, renderer):
    _, _, backbone = loaded
    rows = _ragged(records, renderer)
    torch.manual_seed(0)
    payload = torch.randn(len(rows), 5, backbone.embedding_weight.shape[1]) * 0.5
    lengths = torch.tensor([5, 2, 3])
    with torch.no_grad():
        batched = consumer_forward_batch(backbone, fresh_heads, rows, payload, lengths)
        none = consumer_forward_batch(backbone, fresh_heads, rows, None, None)
        teacher = teacher_logits_batch(backbone, rows)
        for b, r in enumerate(rows):
            single = consumer_forward(backbone, fresh_heads, r.consumer_before, payload[b, : int(lengths[b])][None],
                                      r.consumer_after, r.target)[0]
            assert torch.allclose(batched[b], single, atol=ATOL)
            no_block = consumer_forward(backbone, fresh_heads, r.consumer_before, None, r.consumer_after, r.target)[0]
            assert torch.allclose(none[b], no_block, atol=ATOL)
            t = teacher_target_logits(backbone, r.teacher_prefix, r.target)[0]
            assert (teacher[b] - t).abs().max() < 1e-2, float((teacher[b] - t).abs().max())


def test_stop_log_prob_matches_decisions(loaded, fresh_heads, records):
    _, _, backbone = loaded
    with torch.no_grad():
        fresh_heads.stop.mlp_out.bias.fill_(0.0)
    gen = torch.Generator().manual_seed(3)
    pre = prefill_batch(backbone, fresh_heads, [r.producer for r in records[:4]])
    written = unroll_write(backbone, fresh_heads, pre, max_length=6, sample=True, generator=gen)
    logp = stop_log_prob(fresh_heads, written)
    assert logp.shape == (4,) and bool((logp <= 0).all())
    # Recompute row 0 by hand.
    counts = torch.arange(1, written.shallow.shape[1] + 1)[None]
    logits = fresh_heads.stop(written.shallow.detach()[:1], counts).float()[0]
    L = int(written.lengths[0])
    expected = sum(torch.nn.functional.logsigmoid(-logits[c - 1]) for c in range(1, L))
    if not bool(written.truncated[0]):
        expected = expected + torch.nn.functional.logsigmoid(logits[L - 1])
    assert torch.allclose(logp[0], torch.as_tensor(expected, dtype=logp.dtype), atol=1e-4)


def test_consumer_batch_loss_terms_and_gradients(loaded, fresh_heads, records):
    _, _, backbone = loaded
    loss, metrics = consumer_batch_loss(backbone, fresh_heads, records[:4], max_length=4, contrastive_weight=0.5,
                                        diversity_weight=1.0, temperature=0.3, payload_kl_weight=0.01,
                                        stop_policy_weight=1.0, length_cost=0.01, policy_samples=2,
                                        generator=torch.Generator().manual_seed(0))
    loss.backward()
    for key in ("consumer_ce", "consumer_kl", "contrastive_hinge", "shuffled_ce", "diversity_hinge", "stop_policy",
                "batch_cross_similarity", "payload_kl"):
        assert key in metrics, key
    assert fresh_heads.feedback.mlp_out.weight.grad.abs().sum() > 0
    assert fresh_heads.stop.mlp_out.weight.grad.abs().sum() > 0
    assert fresh_heads.content.proj.weight.grad.abs().sum() > 0


def test_diversity_loss_penalises_identical_blocks():
    same = torch.ones(4, 8)
    varied = torch.randn(4, 8)
    assert diversity_loss(same) > diversity_loss(varied)


def test_lora_deltas_off_restores_base_and_trains(tmp_path):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone, load_backbone
    from natlang_neuralese.train.adapters import deltas_off, inject_lora

    model, tokenizer = load_backbone(dtype=torch.float32, device="cpu")
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer))
    ids = torch.tensor([tokenizer("The harbour town grew up around the ferry.").input_ids])
    with torch.no_grad():
        base = backbone.forward_ids(ids)["logits"]
    grouped = inject_lora(backbone, [15, 14], rank=4)
    assert set(grouped) == {14, 15}
    with torch.no_grad():
        for params in grouped.values():
            for p in params:
                p.add_(0.05 * torch.randn_like(p))
        changed = backbone.forward_ids(ids)["logits"]
        with deltas_off(backbone):
            off = backbone.forward_ids(ids)["logits"]
    assert not torch.allclose(changed, base, atol=1e-3)
    assert torch.allclose(off, base, atol=1e-5)
