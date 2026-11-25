"""OPSD per-call twins on the shared privileged-context mechanism (rl/opsd_privileged.py)."""
from __future__ import annotations

import pytest
import torch

from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.rl.opsd_privileged import OPSD_FACT_KINDS, opsd_facts, opsd_loss, twin_sequences
from natlang_neuralese.rl.privileged_context import PrivilegedRendering

BOS, IM_START, IM_END = 1, 2, 3


def encode(text):
    return [100 + (ord(ch) % 50) for ch in text]


def rendering():
    return PrivilegedRendering('lfm3', BOS, IM_START, IM_END, 60, 61, 'f' * 64, encode)


def case(world='w1', split='train'):
    return {'id': world, 'split': split, 'semantics': {'expected': 'secret_answer_42'},
            'curriculum': {'reference': {'failures': 2}, 'decisive': ['the ledger lists 3 rows']}}


def custody(protected=()):
    return ProtectedWorlds.from_cases({'held': ('0' * 64, [case(world) for world in protected])})


def test_opsd_facts_are_diagnostic_and_withhold_answers():
    feedback = [{'case_id': 'w1', 'attempt': 0, 'feedback': 'returned "kitchen"; expected secret_answer_42'},
                {'case_id': 'w1', 'attempt': 1, 'feedback': 'the move left files in claims/'}]
    _, facts = opsd_facts(case(), custody=custody(), attempt_feedback=feedback)
    kinds = {fact.kind for fact in facts}
    assert kinds <= set(OPSD_FACT_KINDS) and 'expected_result' not in kinds
    texts = '\n'.join(fact.text for fact in facts)
    assert 'secret_answer_42' not in texts           # a feedback line carrying the answer is dropped whole
    assert 'claims/' in texts and 'ledger lists 3 rows' in texts


def test_opsd_facts_refuse_protected_worlds():
    with pytest.raises(Exception):
        opsd_facts(case('held-1'), custody=custody(protected=('held-1',)))


def test_twin_inserts_the_turn_after_the_leading_system_turn_and_scores_the_exact_completion():
    r = rendering()
    system = [IM_START, *encode('system\n'), *encode('rules'), IM_END, *encode('\n')]
    user = [IM_START, *encode('user\n'), *encode('do it'), IM_END, *encode('\n'), IM_START, *encode('assistant\n')]
    prompt = [BOS, *system, *user]
    completion = [7, 8, 9, IM_END]
    twin = twin_sequences(r, prompt, completion, ['Earlier attempt 0 received: wrong file'])
    assert twin.placement == 'after_leading_system_turn' and twin.insert_at == 1 + len(system)
    assert list(twin.twin_ids) == prompt[:twin.insert_at] + r.turn(['Earlier attempt 0 received: wrong file']) \
        + prompt[twin.insert_at:] + completion
    assert [twin.student_ids[p + 1] for p in twin.student_positions] == completion
    assert [twin.twin_ids[p + 1] for p in twin.twin_positions] == completion


def test_opsd_loss_masks_shortcuts_and_trains_ce_only_on_verified_positions():
    student = torch.zeros(3, 10)
    student[1, 5] = -8.0
    student = student.requires_grad_(True)
    twin = torch.zeros(3, 10)
    twin[0, 3] = 2.0
    twin[1, 5] = 10.0                 # the twin prefers the sampled token by ~18 nats: a hindsight-only jump, masked
    targets = torch.tensor([3, 5, 4])
    loss, metrics = opsd_loss(student, twin, targets)
    assert metrics['n_kept'] == 2 and metrics['n_verified'] == 0 and metrics['ce'] == 0
    loss.backward()
    assert float(student.grad[1].abs().sum()) == 0 and float(student.grad[0].abs().sum()) > 0
    # CE only where a verified continuation is marked
    student.grad = None
    verified = torch.tensor([False, False, True])
    loss, metrics = opsd_loss(student, twin, targets, verified=verified, ce_weight=1.0)
    assert metrics['n_verified'] == 1 and metrics['ce'] > 0
    # the tail-bucket form agrees with the full divergence once k covers the vocabulary
    full, _ = opsd_loss(student, twin, targets, kind='jsd')
    bucket, _ = opsd_loss(student, twin, targets, kind='jsd', topk=10)
    torch.testing.assert_close(full, bucket, rtol=1e-4, atol=1e-6)


def test_chunked_twin_loss_equals_one_pass():
    from types import SimpleNamespace
    from natlang_neuralese.rl import opsd_privileged as O
    torch.manual_seed(0)
    emb = torch.nn.Embedding(200, 8)
    head = torch.nn.Linear(8, 20, bias=False)
    learner = SimpleNamespace()
    O_hidden, O_head = O._positions_hidden, O._head
    try:
        O._positions_hidden = lambda _l, ids, pos, grad: torch.tanh(emb(torch.tensor(ids)).cumsum(0))[list(pos)]
        O._head = lambda _l, rows: head(rows).float()
        twin = twin_sequences(rendering(), [BOS, 4, 5, 6], [7, 8, 9, 10, 11], ['fact'])
        one = O.twin_loss(learner, twin, grad=True, chunk=100)
        g_one = emb.weight.grad.clone()
        emb.weight.grad = None
        many = O.twin_loss(learner, twin, grad=True, chunk=2)
        assert abs(one['divergence'] - many['divergence']) < 1e-5
        torch.testing.assert_close(emb.weight.grad, g_one, rtol=1e-4, atol=1e-6)
    finally:
        O._positions_hidden, O._head = O_hidden, O_head
