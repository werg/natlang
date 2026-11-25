"""Compiler/OPSD loss semantics: full and top-k + tail-bucket distillation, and the FO-MAML gradient path."""
from __future__ import annotations

import torch

from natlang_neuralese.common import stable_softmax as S
from natlang_neuralese.compiler.header_encoder import HeaderEncoder, HeaderEncoderConfig
from natlang_neuralese.compiler.train_compiler import (
    context_distillation_loss,
    distillation_kl_loss,
    fo_maml_inner_adapt_z,
)


def test_distillation_kl_full_and_tail_bucket():
    torch.manual_seed(0)
    student = torch.randn(2, 5, 50, requires_grad=True)
    teacher = torch.randn(2, 5, 50) * 2
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, :2] = False
    full = distillation_kl_loss(student, teacher, mask)
    exact = torch.nn.functional.kl_div(torch.log_softmax(student[mask], -1), torch.log_softmax(teacher[mask], -1),
                                       log_target=True, reduction='none').sum(-1).mean()
    torch.testing.assert_close(full, exact, rtol=1e-5, atol=1e-6)
    # top-k with a tail bucket: equals the full KL once k covers the vocabulary, and lower-bounds it otherwise
    torch.testing.assert_close(distillation_kl_loss(student, teacher, mask, topk=50), exact, rtol=1e-4, atol=1e-5)
    assert float(distillation_kl_loss(student, teacher, mask, topk=8)) <= float(exact) + 1e-5
    full.backward()
    assert student.grad[:, :2].abs().max() == 0 and student.grad[:, 2:].abs().max() > 0


def test_context_distillation_ce_only_on_supervised_labels():
    torch.manual_seed(1)
    student = torch.randn(1, 6, 30, requires_grad=True)
    teacher = student.detach().clone()
    labels = torch.full((1, 6), -100)
    labels[0, 3:] = torch.tensor([1, 2, 3])
    loss = context_distillation_loss(student, teacher, labels, ce_weight=1.0)
    ce = torch.nn.functional.cross_entropy(student[0, 3:], labels[0, 3:])
    torch.testing.assert_close(loss, ce, rtol=1e-5, atol=1e-5)  # the KL term is zero at agreement


def test_fo_maml_query_gradient_reaches_header_encoder():
    torch.manual_seed(2)
    enc = HeaderEncoder(HeaderEncoderConfig(hidden_dim=16, num_tap_layers=1, num_slots=4, value_dim=8, key_dim=4,
                                            num_heads=2))
    head = torch.nn.Linear(8, 3)
    taps = (torch.randn(1, 7, 16),)
    final = torch.randn(1, 7, 16)
    z0, k, w = enc(taps, final)
    support_target = torch.tensor([0])
    query_target = torch.tensor([2])

    def loss_at(z, target):
        return torch.nn.functional.cross_entropy(head(z.mean(1)), target)

    z_k = fo_maml_inner_adapt_z(z0, k, w, lambda z, _k, _w: loss_at(z, support_target), steps=3, inner_lr=0.5)
    assert not torch.allclose(z_k, z0)                      # the inner loop moved z
    loss_at(z_k, query_target).backward()
    enc_grad = sum(float(p.grad.abs().sum()) for p in enc.parameters() if p.grad is not None)
    assert enc_grad > 0                                     # first-order path into HeaderEncoder
    assert float(head.weight.grad.abs().sum()) > 0          # and into the head
    # identity Jacobian: d(query)/d(z0) equals d(query)/d(z_K)
    z0d = z0.detach().clone().requires_grad_(True)
    zk = fo_maml_inner_adapt_z(z0d, k, w, lambda z, _k, _w: loss_at(z, support_target), steps=3, inner_lr=0.5)
    zk_leaf = zk.detach().clone().requires_grad_(True)
    (g0,) = torch.autograd.grad(loss_at(zk, query_target), z0d)
    (gk,) = torch.autograd.grad(loss_at(zk_leaf, query_target), zk_leaf)
    torch.testing.assert_close(g0, gk)


def test_t0_sequences_share_prefix_chains_and_mark_boundaries():
    from natlang_neuralese.train.step_verifier import rollout_sequences

    def turn(inv, prompt, completion):
        return {'invocation_id': inv, 'sampling': {'prompt': [{'kind': 'tokens', 'token_ids': prompt}],
                                                  'samples': [{'token_id': t} for t in completion]}}
    rollout = {'turns': [turn('a', [1, 2], [3, 4]), turn('a', [1, 2, 3, 4, 9], [5]),
                         turn('b', [7], [8]), turn('a', [1, 6], [6]),
                         {'invocation_id': 'c', 'sampling': {'prompt': [{'kind': 'native_state'}], 'samples': [{'token_id': 1}]}}]}
    seqs = rollout_sequences(rollout)
    assert {'ids': [1, 2, 3, 4, 9, 5], 'boundaries': [3, 5], 'turns': [0, 1]} in seqs   # one forward, two boundaries
    assert {'ids': [1, 6, 6], 'boundaries': [2], 'turns': [3]} in seqs                  # non-extending prompt: new chain
    assert {'ids': [7, 8], 'boundaries': [1], 'turns': [2]} in seqs
    assert len(seqs) == 3                                                                # native-state prompt skipped
    for seq in seqs:
        assert all(seq['ids'][b] == rollout['turns'][t]['sampling']['samples'][-1]['token_id']
                   for b, t in zip(seq['boundaries'], seq['turns']))
