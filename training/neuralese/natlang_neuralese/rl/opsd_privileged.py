"""On-policy privileged self-distillation (OPSD) on per-call sequences of failed rollouts (`plans/NATLANG_COMPILER.md`
§8.2, Recipe T1), on the shared privileged-context mechanism.

Privileged context has one implementation: ``rl/privileged_context.py`` (custody, fact extraction with answers
withheld, the declared guidance turn and its placement) and ``rl/privileged_scoring.py`` (whole-graph twins). This module
binds it to one model call of an archived rollout, where the learner's unit is a (prompt, completion) sequence:

1. Facts: ``case_facts(case, custody=..., attempt_feedback=..., answers='withhold')``, restricted to the diagnostic
   kinds ``OPSD_FACT_KINDS`` (verifier feedback on the failed attempt, recorded reference failures, decisive evidence)
   and to what the call may see (``applicable``). Answer kinds never enter, and procedure facts that still carry an
   answer leaf are dropped by ``case_facts``.
2. Twin: the call's exact prompt token ids with the text guidance turn (``PrivilegedRendering.turn``) inserted at the
   declared place (``insertion_point``: after the leading system turn, else after BOS), followed by the student's exact
   sampled completion ids. This is the ``text-twin`` form the JOLT+ learner inserts; the runtime renders the same turn
   (``guidance_messages``; tested in ``test_jolt_privileged_context``). The twin scores exactly the student's tokens at
   exactly the student's prefix (plus the turn), checked by ``twin_sequences``.
3. Loss (``opsd_loss``): divergence between the student's distribution and the stop-gradient twin at every completion
   position, on full distributions by default (``common.stable_softmax``; ``topk`` uses top-k plus a tail bucket for
   both sides). The direction defaults to forward KL(twin || student), the JOLT+ student default (DECISIONS
   2026-11-20: its gradient does not vanish on tokens the student finds unlikely); ``'jsd'`` is the plan's symmetric
   form. The anti-shortcut mask drops positions whose sampled token the twin prefers by more than ``max_log_ratio``
   nats (outside declared syntax positions). Cross-entropy applies only where ``verified`` marks a verified
   continuation: a failed episode's own tokens are never CE targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .privileged_context import ANSWER_KINDS, applicable, case_facts

OPSD_SCHEMA = 'natlang.opsd-call-twin/1'
# Diagnostic facts only (the plan: verifier diff, violated clause, structural diagnosis; never the expected answer).
OPSD_FACT_KINDS = ('attempt_feedback', 'reference_failure', 'decisive')
DIVERGENCES = ('forward_kl', 'jsd', 'reverse_kl')


def opsd_facts(case, *, custody, attempt_feedback=(), invocation=None, kinds=OPSD_FACT_KINDS):
    """``(custody receipt, facts)`` the twin of one call may carry: diagnostic kinds, answers withheld, visible to it.

    ``invocation``: ``{name, role, text}`` of the call (default: the root). Raises ``CustodyError`` for a protected
    world (custody fails closed)."""
    if set(kinds) & set(ANSWER_KINDS):
        raise ValueError('OPSD twins never carry answer facts')
    receipt, facts = case_facts(case, custody=custody, attempt_feedback=attempt_feedback, answers='withhold')
    invocation = invocation or {'name': 'root', 'role': 'root', 'text': ''}
    return receipt, tuple(fact for fact in facts if fact.kind in kinds and applicable(fact, invocation))


@dataclass(frozen=True)
class CallTwin:
    """Student and twin token sequences of one call, with the completion positions aligned 1:1."""
    student_ids: tuple
    twin_ids: tuple
    student_positions: tuple   # logits rows predicting each completion token in the student sequence
    twin_positions: tuple      # the same rows in the twin sequence
    targets: tuple             # the completion token ids (the student's own samples)
    insert_at: int
    placement: str
    turn_ids: tuple
    receipt: dict


def twin_sequences(rendering, prompt_ids, completion_ids, fact_texts, *, receipt=None):
    """The call's student sequence and its privileged twin (the guidance turn inserted at the declared place)."""
    prompt, completion = list(map(int, prompt_ids)), list(map(int, completion_ids))
    if not prompt or not completion:
        raise ValueError('an OPSD twin needs a prompt and a sampled completion')
    if not fact_texts:
        raise ValueError('an OPSD twin needs at least one privileged fact')
    turn = list(rendering.turn(list(fact_texts)))
    at, placement = rendering.insertion_point(prompt)
    student = prompt + completion
    twin = prompt[:at] + turn + prompt[at:] + completion
    s_pos = tuple(range(len(prompt) - 1, len(student) - 1))
    t_pos = tuple(position + len(turn) for position in s_pos)
    if [student[p + 1] for p in s_pos] != completion or [twin[p + 1] for p in t_pos] != completion:
        raise AssertionError('twin positions do not score the student completion')
    if twin[:at] != student[:at] or twin[at + len(turn):] != student[at:]:
        raise AssertionError('the twin changed the student prefix outside the inserted turn')
    return CallTwin(tuple(student), tuple(twin), s_pos, t_pos, tuple(completion), at, placement, tuple(turn),
                    {'schema': OPSD_SCHEMA, 'rendering': rendering.digest, 'placement': placement,
                     'insert_at': at, 'turn_tokens': len(turn), **(receipt or {})})


def opsd_loss(student_logits, twin_logits, targets, *, kind='forward_kl', beta=0.5, topk=None, syntax_mask=None,
              max_log_ratio=6.0, verified=None, ce_weight=0.3):
    """Masked OPSD divergence at aligned completion positions, plus CE on verified continuations only.

    ``student_logits`` [N, V] (with gradient) and ``twin_logits`` [N, V] (detached) at the N aligned completion
    positions; ``targets`` [N] the student's sampled tokens there. ``verified`` [N] bool marks positions of verified
    continuations (a verified repair or success; default: none, so no CE). Returns ``(loss, metrics)``; the loss is the
    mean divergence over kept positions plus ``ce_weight`` times the mean CE over verified positions."""
    from ..common import stable_softmax as S
    if kind not in DIVERGENCES:
        raise ValueError(f'OPSD divergence is one of {DIVERGENCES}')
    if twin_logits.requires_grad:
        raise ValueError('the OPSD twin is a stop-gradient target')
    n = int(targets.numel())
    zero = student_logits.sum() * 0.0
    if n == 0:
        return zero, {'n': 0., 'n_kept': 0., 'shortcut_masked_frac': 0., 'divergence': 0., 'n_verified': 0., 'ce': 0.}
    targets = targets.long()
    with torch.no_grad():
        s_y = S.token_logprobs(student_logits.detach(), targets)
        t_y = S.token_logprobs(twin_logits, targets)
        log_ratio = (t_y - s_y).float()
        shortcut = log_ratio > float(max_log_ratio)
        if syntax_mask is not None:
            shortcut &= ~syntax_mask.to(torch.bool)
        kept = ~shortcut
    n_kept = int(kept.sum())
    if n_kept:
        if topk is None:
            values = S.divergence(student_logits[kept], twin_logits[kept], kind, beta=beta)
        else:
            t_logp = torch.log_softmax(twin_logits[kept].float(), -1)
            head, ids = torch.topk(t_logp, min(int(topk), t_logp.shape[-1]), -1)
            tail = torch.logsumexp(t_logp.scatter(-1, ids, float('-inf')), -1)
            values = S.topk_tail_divergence(student_logits[kept], ids, head, kind, beta=beta, teacher_log_tail=tail)
        divergence = values.float().mean()
    else:
        divergence = zero
    loss = divergence
    n_verified = 0
    ce = zero
    if verified is not None and ce_weight > 0:
        mask = verified.to(torch.bool)
        n_verified = int(mask.sum())
        if n_verified:
            ce = -S.token_logprobs(student_logits[mask], targets[mask]).float().mean()
            loss = loss + float(ce_weight) * ce
    metrics = {'n': float(n), 'n_kept': float(n_kept), 'shortcut_masked_frac': 1. - n_kept / n,
               'divergence': float(divergence.detach()), 'n_verified': float(n_verified), 'ce': float(ce.detach()),
               'mean_log_ratio': float(log_ratio[kept].mean()) if n_kept else 0.}
    return loss, metrics
