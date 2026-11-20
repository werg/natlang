"""Ambient SFT's Temperature Matching in exp0_sft (arXiv 2611.23229, Algorithm 1; DECISIONS 2026-11-20)."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese')]
import exp0_sft  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self, vocab=11, width=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(vocab, width, generator=g))
        self.mix = torch.nn.Parameter(torch.randn(width, width, generator=g) / width)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, width, generator=g))


def tiny_forward(model, ids):
    x = model.emb[ids]
    return x + torch.tanh(torch.cumsum(x, 0) @ model.mix)


def chain(ids, start, weights=None):
    ids = np.asarray(ids, np.int32)
    row = {'sample': ids[start:]}
    if weights is not None:
        row['weight'] = np.asarray(weights, np.float32)
    return {'ids': ids, 'segs': [(start, len(ids) - start, row)], 'round': 'teacher:x'}


def test_tm_ce_is_the_tempered_kl_plus_the_teacher_entropy_and_chunking_is_exact():
    g = torch.Generator().manual_seed(1)
    weight, teacher_weight = torch.randn(13, 5, generator=g), torch.randn(13, 5, generator=g)
    h, th = torch.randn(9, 5, generator=g), torch.randn(9, 5, generator=g)
    T = 2.5
    ce, entropy = exp0_sft.tm_ce(weight, h, th, teacher_weight, T, chunk=4)
    q_log = torch.log_softmax(th @ teacher_weight.T / T, -1)
    p_log = torch.log_softmax(h @ weight.T / T, -1)
    kl = (q_log.exp() * (q_log - p_log)).sum(-1)
    assert torch.allclose(ce - entropy, kl, atol=1e-5)
    assert torch.allclose(entropy, -(q_log.exp() * q_log).sum(-1), atol=1e-5)
    whole, _ = exp0_sft.tm_ce(weight, h, th, teacher_weight, T, chunk=100)
    assert torch.allclose(ce, whole, atol=1e-6)


def test_tm_gradient_vanishes_when_the_student_is_the_teacher_and_not_otherwise():
    g = torch.Generator().manual_seed(2)
    weight = torch.randn(13, 5, generator=g)
    th = torch.randn(7, 5, generator=g)
    h = th.clone().requires_grad_(True)
    ce, _ = exp0_sft.tm_ce(weight, h, th, weight, 2.5, chunk=3)
    ce.sum().backward()
    assert h.grad.abs().max() < 1e-5
    h2 = (th + 0.5).requires_grad_(True)
    exp0_sft.tm_ce(weight, h2, th, weight, 2.5)[0].sum().backward()
    assert h2.grad.abs().max() > 1e-3


def test_the_coin_is_deterministic_and_p_clean_one_is_plain_sft():
    draws = [exp0_sft.ambient_clean(7, s, 0.75) for s in range(4000)]
    assert draws == [exp0_sft.ambient_clean(7, s, 0.75) for s in range(4000)]
    assert 0.72 < sum(draws) / len(draws) < 0.78
    assert draws != [exp0_sft.ambient_clean(8, s, 0.75) for s in range(4000)]
    assert all(exp0_sft.ambient_clean(7, s, 1.0) for s in range(500))


def test_step_backward_trains_the_weighted_tm_loss_against_the_frozen_teacher():
    model = Tiny()
    live = dict(model.named_parameters())
    teacher = {name: p.detach().clone() for name, p in live.items()}
    with torch.no_grad():                       # the student has moved away from the start checkpoint
        for p in live.values():
            p.add_(0.3 * torch.randn_like(p))
    masters = {name: p.detach().clone() for name, p in live.items()}
    chains = [chain([1, 2, 3, 4, 5], 2, [1., 0., 2.]), chain([6, 7, 8, 9], 1)]

    def restore():
        with torch.no_grad():
            torch._foreach_copy_(list(live.values()), [masters[name] for name in live])
    rows = exp0_sft.ambient_teacher_rows(model, tiny_forward, chains, live, teacher, restore, 0, None, 0, fast=False)
    for name, p in live.items():                # the policy is back exactly
        assert torch.equal(p, masters[name])
    student = Tiny()
    with torch.no_grad():
        for name, p in student.named_parameters():
            p.copy_(teacher[name])
    for index, c in enumerate(chains):          # teacher rows are the start checkpoint's hidden rows at the targets
        ids, positions, _ = exp0_sft.chain_tensors(c, 'cpu')
        assert torch.allclose(rows[index], tiny_forward(student, ids)[positions].detach())

    ambient = {'temperature': 2.5, 'head': teacher['lm_head.weight'], 'rows': rows}
    total, _ = exp0_sft.step_backward(model, tiny_forward, chains, 4.0, 0, None, 0, fast=False, ambient=ambient)
    expected, kl = 0., 0.
    for index, c in enumerate(chains):
        ids, positions, _ = exp0_sft.chain_tensors(c, 'cpu')
        w = exp0_sft.chain_weights(c, 'cpu')
        s = torch.log_softmax(model.lm_head(tiny_forward(model, ids)[positions]).detach() / 2.5, -1)
        q = torch.log_softmax(rows[index] @ teacher['lm_head.weight'].T / 2.5, -1)
        expected += float(((-(q.exp() * s).sum(-1)) * w).sum())
        kl += float(((q.exp() * (q - s)).sum(-1) * w).sum())
    assert abs(total - expected) < 1e-4
    assert abs(ambient['kl'] - kl) < 1e-4
    assert model.lm_head.weight.grad is not None and model.lm_head.weight.grad.abs().sum() > 0


def test_step_backward_without_ambient_is_the_weighted_nll():
    model = Tiny()
    chains = [chain([1, 2, 3, 4, 5], 2, [1., 0., 2.])]
    total, _ = exp0_sft.step_backward(model, tiny_forward, chains, 3.0, 0, None, 0, fast=False)
    ids, positions, targets = exp0_sft.chain_tensors(chains[0], 'cpu')
    nll = torch.nn.functional.cross_entropy(model.lm_head(tiny_forward(model, ids)[positions]).detach(), targets,
                                            reduction='none')
    assert abs(total - float((nll * exp0_sft.chain_weights(chains[0], 'cpu')).sum())) < 1e-4
