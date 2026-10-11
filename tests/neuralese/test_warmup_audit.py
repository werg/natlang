"""Warm-up audit 2026-10-11 (plans/neuralese/WARMUP_AUDIT_20261011.md): properties of core_text_warmup that were
checked and hold. Target alignment of both sequence passes and the preserve KL's direction and top-k semantics."""
import math

import pytest
import torch

from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone


def tiny_student():
    try:
        from transformers import Lfm2Config, Lfm2ForCausalLM
    except ImportError:
        pytest.skip("installed transformers does not provide the tiny LFM2 model")
    torch.manual_seed(21)
    config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, block_multiple_of=8,
                        block_auto_adjust_ff_dim=False, layer_types=['conv', 'full_attention', 'conv', 'full_attention'])
    model = Lfm2ForCausalLM(config).eval()
    backbone = PortBackbone(model, ControlTokens(62, 63), fast=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = PortHeads(backbone, cutoff=2, max_length=8, profile='latent-sketch-v2').eval()
    return backbone, heads


def test_both_sequence_passes_predict_each_target_without_reading_it():
    """top[:, j] is the predictor of span[:, j] in the gold pass and in the mapped-history pass: changing span[:, k]
    leaves top[:, :k+1] unchanged (no target leak through the causal input map or the history shift)."""
    from natlang_neuralese.model.input_map import NeuraleseInputMap
    from natlang_neuralese.train.text_warmup import mapped_completions
    backbone, heads = tiny_student()
    heads.add_module('input_map', NeuraleseInputMap(backbone.embedding_weight.shape[1], kernel=3, rank=4))
    with torch.no_grad():
        heads.input_map.conv.weight.normal_(0, .3)
        heads.input_map.up.weight.normal_(0, .3)
    prefix = torch.tensor([[1, 4, 7]])
    span = torch.tensor([[9, 3, 5, 8, 11, 2]])
    with torch.no_grad():
        base = [out['top'] for out in mapped_completions(backbone, heads, prefix, span, passes=2)]
        for k in range(span.shape[1]):
            changed = span.clone()
            changed[0, k] = 40
            tops = [out['top'] for out in mapped_completions(backbone, heads, prefix, changed, passes=2)]
            for before, after in zip(base, tops):
                assert torch.allclose(before[:, :k + 1], after[:, :k + 1], atol=1e-5)
                if k + 1 < span.shape[1]:
                    assert not torch.allclose(before[:, k + 1:], after[:, k + 1:], atol=1e-5)


class _TinyLM(torch.nn.Module):
    """The interface weighted_topk_kl reads: model.model(input_ids=...).last_hidden_state and model.lm_head."""

    def __init__(self, vocab=16, width=8):
        super().__init__()
        torch.manual_seed(3)
        self.embed = torch.nn.Embedding(vocab, width)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)
        outer = self

        class Inner(torch.nn.Module):
            def forward(self, input_ids):
                from types import SimpleNamespace
                return SimpleNamespace(last_hidden_state=outer.embed(input_ids))
        self.model = Inner()


def test_preserve_kl_is_forward_kl_to_the_renormalised_teacher_top_k():
    """KL(teacher || student) over the teacher's top-k, the teacher renormalised over its k and the student's full
    log-softmax: equal to the student at k = V gives zero; at k < V the term adds the student's mass outside the
    teacher's top-k (-log of its top-k mass), so it sharpens the student towards the teacher's k tokens."""
    from natlang_neuralese.maple.qat_convert import weighted_topk_kl
    model = _TinyLM()
    ids = torch.tensor([[1, 5, 7, 2, 9, 4]])
    with torch.no_grad():
        logp = torch.log_softmax(model.lm_head(model.embed(ids[0, :-1])).float(), -1)
    full = logp.argsort(-1, descending=True)
    same = weighted_topk_kl(model, ids, full, logp.gather(-1, full), 0, 1.0)
    assert same['loss_kl'].item() == pytest.approx(0.0, abs=1e-5)
    k = 4
    top = full[:, :k]
    partial = weighted_topk_kl(model, ids, top, logp.gather(-1, top), 0, 1.0)
    residual = -torch.logsumexp(logp.gather(-1, top), -1)  # -log(student mass on the teacher's top-k)
    assert partial['loss_kl'].item() == pytest.approx(residual.mean().item(), rel=1e-4)
    # Direction: the teacher's probabilities weight the log-ratio (forward KL), not the student's.
    teacher = torch.log_softmax(torch.randn_like(logp), -1)
    tk = teacher.argsort(-1, descending=True)
    value = weighted_topk_kl(model, ids, tk, teacher.gather(-1, tk), 0, 1.0)['loss_kl'].item()
    forward = (teacher.exp() * (teacher - logp)).sum(-1).mean().item()
    assert value == pytest.approx(forward, rel=1e-4) and math.isfinite(value)
