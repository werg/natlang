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
        O._positions_hidden = lambda _l, ids, pos, grad, **_kw: torch.tanh(emb(torch.tensor(ids)).cumsum(0))[list(pos)]
        O._head = lambda _l, rows, **_kw: head(rows).float()
        twin = twin_sequences(rendering(), [BOS, 4, 5, 6], [7, 8, 9, 10, 11], ['fact'])
        one = O.twin_loss(learner, twin, grad=True, chunk=100)
        g_one = emb.weight.grad.clone()
        emb.weight.grad = None
        many = O.twin_loss(learner, twin, grad=True, chunk=2)
        assert abs(one['divergence'] - many['divergence']) < 1e-5
        torch.testing.assert_close(emb.weight.grad, g_one, rtol=1e-4, atol=1e-6)
    finally:
        O._positions_hidden, O._head = O_hidden, O_head


def test_synthesize_contrast_feedback_is_structural_and_withholds_answers():
    from natlang_neuralese.rl.opsd_privileged import synthesize_contrast_feedback
    failed = {
        'task_id': 'w1',
        'outcome': 'semantic_mismatch',
        'turns': [
            {'target': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'eval', 'arguments': '{}'}}]}},
            {'target': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'return_result', 'arguments': '{"value":"wrong"}'}}]}},
        ],
    }
    passing = {
        'task_id': 'w1',
        'outcome': 'task_success',
        'turns': [
            {'target': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'read_file', 'arguments': '{"path":"ledger.csv"}'}}]}},
            {'target': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'eval', 'arguments': '{}'}}]}},
            {'target': {'role': 'assistant', 'tool_calls': [{'function': {'name': 'return_result', 'arguments': '{"value":"secret_answer_42"}'}}]}},
        ],
    }
    items = synthesize_contrast_feedback(failed, passing)
    assert len(items) == 1
    note = items[0]['feedback']
    assert 'read_file' in note and 'secret_answer_42' not in note and 'ledger.csv' not in note
    _, facts = opsd_facts(case(), custody=custody(), attempt_feedback=items)
    assert any('read_file' in f.text for f in facts)
    assert all('secret_answer_42' not in f.text for f in facts)


def test_head_is_differentiable_and_scores_trainable_rows():
    from types import SimpleNamespace
    from natlang_neuralese.rl.opsd_privileged import _head
    torch.manual_seed(1)
    weight = torch.randn(12, 4, dtype=torch.bfloat16)
    rows_param = torch.randn(2, 4, requires_grad=True)
    learner = SimpleNamespace(model=SimpleNamespace(lm_head=SimpleNamespace(weight=weight)),
                              trainable_rows=SimpleNamespace(token_ids=torch.tensor([3, 7]), rows=rows_param))
    h = torch.randn(5, 4, requires_grad=True)
    logits = _head(learner, h)
    torch.testing.assert_close(logits[:, 3], h @ rows_param[0])
    torch.testing.assert_close(logits[:, 0], h @ weight[0].float())
    logits.sum().backward()
    assert h.grad.abs().sum() > 0 and rows_param.grad.abs().sum() > 0


def test_twin_loss_scores_the_twin_with_the_frozen_start_policy_by_default():
    """T1 run1 (live teacher, no anchor) collapsed to an input-independent policy with divergence ~1e-7: the twin is
    scored by the start policy (reference_view) unless ``teacher='live'``; the student is always the live policy."""
    import contextlib
    from types import SimpleNamespace
    from natlang_neuralese.rl import opsd_privileged as O
    torch.manual_seed(0)
    emb = torch.nn.Embedding(200, 8)
    head = torch.nn.Linear(8, 20, bias=False)
    delta = torch.nn.Parameter(torch.zeros(8))         # the "adapter": zero in the start policy
    state = {'ref': False}
    calls = []

    @contextlib.contextmanager
    def reference_view():
        state['ref'] = True
        try:
            yield
        finally:
            state['ref'] = False
    learner = SimpleNamespace(reference_view=reference_view)

    def hidden(_l, ids, pos, *, grad, start=False):
        calls.append((start, grad))
        if start:
            with reference_view():
                return torch.tanh(emb(torch.tensor(ids)).cumsum(0))[list(pos)].detach()
        return torch.tanh(emb(torch.tensor(ids)).cumsum(0) + delta)[list(pos)]
    O_hidden, O_head = O._positions_hidden, O._head
    try:
        O._positions_hidden = hidden
        O._head = lambda _l, rows, **_kw: head(rows).float()
        with torch.no_grad():
            delta.add_(0.5)                              # the live policy has moved away from the start
        twin = twin_sequences(rendering(), [BOS, 4, 5, 6], [7, 8, 9, 10, 11], ['fact'])
        O.twin_loss(learner, twin, grad=True)
        assert calls == [(True, False), (False, True)]
        assert delta.grad is not None and delta.grad.abs().sum() > 0
        calls.clear()
        out = O.twin_loss(learner, twin, grad=False, teacher='live')
        assert calls == [(False, False), (False, False)]
        assert out['student_entropy'] > 0 and out['student_nll'] > 0
        with pytest.raises(ValueError):
            O.twin_loss(learner, twin, grad=False, teacher='other')
    finally:
        O._positions_hidden, O._head = O_hidden, O_head


def test_start_head_ignores_live_trainable_rows():
    from types import SimpleNamespace
    from natlang_neuralese.rl.opsd_privileged import _head
    weight = torch.randn(12, 4, dtype=torch.bfloat16)
    learner = SimpleNamespace(model=SimpleNamespace(lm_head=SimpleNamespace(weight=weight)),
                              trainable_rows=SimpleNamespace(token_ids=torch.tensor([3]), rows=torch.randn(1, 4)))
    h = torch.randn(5, 4)
    torch.testing.assert_close(_head(learner, h, start=True), h @ weight.float().t())


def test_opsd_loss_reports_collapse_telemetry():
    student = torch.zeros(3, 10)
    student[:, 2] = 30.0                                 # a collapsed student: one token everywhere
    twin = student.clone()
    _, metrics = opsd_loss(student, twin, torch.tensor([5, 6, 7]))
    assert metrics['divergence'] < 1e-6
    assert metrics['student_entropy'] < 1e-6 and metrics['student_nll'] > 20


def test_twin_loss_anchor_kl_and_focus_weights_match_one_pass_across_chunks():
    from types import SimpleNamespace
    from natlang_neuralese.rl import opsd_privileged as O
    torch.manual_seed(2)
    emb = torch.nn.Embedding(200, 8)
    head = torch.nn.Linear(8, 20, bias=False)
    delta = torch.nn.Parameter(torch.full((8,), 0.25))
    learner = SimpleNamespace()
    O_hidden, O_head = O._positions_hidden, O._head
    try:
        def hidden(_l, ids, pos, *, grad, start=False):
            h = torch.tanh(emb(torch.tensor(ids)).cumsum(0))
            if start:
                return h[list(pos)].detach()
            return (h + delta)[list(pos)]
        O._positions_hidden = hidden
        O._head = lambda _l, rows, **_kw: head(rows).float()
        twin = twin_sequences(rendering(), [BOS, 4, 5, 6], [7, 8, 9, 10, 11], ['fact'])
        one = O.twin_loss(learner, twin, grad=True, chunk=100, anchor_kl=0.05, focus_divergent_tokens=True)
        g_one = delta.grad.clone()
        delta.grad = None
        many = O.twin_loss(learner, twin, grad=True, chunk=2, anchor_kl=0.05, focus_divergent_tokens=True)
        assert one['anchor_kl'] > 0 and one['base_guidance_kl'] > 0 and one['focused_divergence'] > 0
        assert abs(one['focused_divergence'] - many['focused_divergence']) < 1e-5
        assert abs(one['anchor_kl'] - many['anchor_kl']) < 1e-5
        torch.testing.assert_close(delta.grad, g_one, rtol=1e-4, atol=1e-6)
    finally:
        O._positions_hidden, O._head = O_hidden, O_head
