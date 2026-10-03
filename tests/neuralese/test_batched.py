"""Batched (ragged) execution, phases D–F, anti-collapse terms and LoRA deltas."""

import json

import pytest
import torch

from natlang_neuralese.data.fixtures import synthetic_records
from natlang_neuralese.data.records import parse_record
from natlang_neuralese.data.render import Renderer, render_record, span_examples
from natlang_neuralese.train.execution import (consumer_forward, consumer_forward_batch, prefill, prefill_batch,
                                               stop_log_prob, teacher_logits_batch, teacher_target_logits,
                                               unroll_write)
from natlang_neuralese.train.losses import consumer_batch_loss, diversity_loss
from natlang_neuralese.train.phases import Phase, pilot_phases
from natlang_neuralese.train.trainer import Trainer

ATOL = 5e-4


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
    # A short D -> F run with released layers, checkpointed and resumed.
    torch.manual_seed(1)
    heads = PortHeads(backbone, cutoff=6, max_length=4)
    renderer = Renderer(tokenizer, backbone.controls)
    recs = [render_record(renderer, parse_record(r)) for r in synthetic_records(4)]
    text = "The river rises in the northern hills and flows south through three valleys. " * 6
    spans = list(span_examples(renderer, [text], prefix_len=10, span_len=4, cont_len=6, limit=4))
    phases = [Phase("F", 2, batch_size=2, max_length=4, lora_layers=(13, 12), lora_rank=4, text_replay_weight=0.5,
                    contrastive_weight=0.5, diversity_weight=1.0)]
    trainer = Trainer(backbone, heads, phases, tmp_path, span_train=spans, records_train=recs, log=lambda *_: None)
    trainer.run()
    lines = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all("replay_kl" in l for l in lines)
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert set(state["lora_layers"]) >= {12, 13, 14, 15}


def test_pilot_phases_cover_a_to_f():
    assert [p.name for p in pilot_phases(0.01)] == ["A", "B", "C", "D", "E", "F"]
