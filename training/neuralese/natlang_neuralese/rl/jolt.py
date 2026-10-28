"""JOLT+ self-teaching (TRAINING_INSIGHTS_20261028.md section 7): one policy as privileged teacher and as student.

Privileged blocks, custody and graph edits are the shared mechanism of ``rl/privileged_context.py`` and
``rl/privileged_scoring.py``; this module binds them into the whole-graph RL objective. Every admitted episode is
scored in ONE packed graph forward holding two views of the whole actual graph:

* the sampled view, exactly as the actor executed it (all producer/consumer turns, imports at their recorded
  positions, gradients through the real dependencies), and
* its twin: ``teacher_form(replay, blocks)`` for a student episode, ``student_form(replay, blocks)`` for a teacher
  attempt (blocks inserted/removed per invocation at their declared place; producer spans, import boundaries and
  spans rebased by the shared graph edit). The twin is a stop-gradient target, so its rows equal ``teacher_scores``
  of the same edited replay while sharing the student's forward batch.

Per sampled position of a role-r episode, with p the sampled view and p~ the twin under the actor's exact
temperature/mask law:

    loss_t = -A_r U_t(score-centered/1)  +  beta_r KL(p || sg[p~])

Student episodes: KL(pi(.|s) || sg pi(.|s,c)) is J_OPSD and A_S is the return minus the student group mean (the
JOLT+ outcome term). Teacher attempts (sampled with the blocks): A_T is the teacher group's centered return and
KL(pi(.|s,c) || sg pi(.|s)) keeps the guidance within the student's reach. The KL uses logit differences
(``reverse_kl``). A custody refusal or an unsupported graph (semantic codec) gives a student episode no privileged
term; it still trains on its outcome term and the refusal is recorded in the objective and metrics.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

JOLT_SCHEMA = 'jolt-plus/1'
PRIVILEGED_SUFFIX = {'student': '#privileged', 'teacher': '#unprivileged'}
# beta_S = 1: the dense reverse KL per token (typically 0.05-1 nat) and the std-scale outcome term enter at
# comparable magnitude. beta_T = 0.5: the reward-consistent teacher target pi_S exp(A/beta_T) tilts the
# student's policy by at most about e^2 for |A| <= 1, keeping teacher guidance followable. Development
# measurements tune both; they are recipe parameters.
DEFAULT_JOLT = {'schema': JOLT_SCHEMA, 'beta_s': 1.0, 'beta_t': 0.5,
                'outcome': {'schema': 'score-centered/1', 'k': 128, 'importance': 'tis', 'eps': 1e-6}}
MAX_FEEDBACK_ATTEMPTS = 3
MAX_FEEDBACK_CHARS = 800


def jolt_declaration(value):
    """Canonical JOLT+ declaration or None when ``value`` is not a JOLT+ declaration."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or value.get('schema') != JOLT_SCHEMA:
        return None
    if set(value) - set(DEFAULT_JOLT):
        raise ValueError('jolt-plus/1 accepts only beta_s, beta_t and outcome')
    from .policy_math import policy_loss_declaration
    result = {**DEFAULT_JOLT, **value}
    outcome = policy_loss_declaration(result['outcome'])
    if outcome is None or outcome['schema'] != 'score-centered/1':
        raise ValueError('jolt-plus/1 outcome terms use the score-centered/1 estimator')
    result['outcome'] = outcome
    for name in ('beta_s', 'beta_t'):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'jolt-plus/1 {name} must be a finite nonnegative number')
        result[name] = float(result[name])
    return result


class _ReverseKL(torch.autograd.Function):
    """KL(p || sg q) from logits, gradient g = p (delta - E_p delta) with delta from logit differences.

    delta_v = (z_v - z~_v) - (lse z - lse z~) is formed from the logits themselves rather than by subtracting two
    rounded log-probabilities, which matters near agreement (the self-teacher is the same model with extra
    context). Masked tokens (-inf in both laws) contribute nothing. Single switch point for the shared
    stable-softmax module.
    """
    @staticmethod
    def forward(ctx, logits, twin_logits):
        z, t = logits.float(), twin_logits.float()
        finite = torch.isfinite(z)
        if bool((finite != torch.isfinite(t)).any()):
            raise ValueError('JOLT+ twin and sampled view must share the exact token mask')
        lse = torch.logsumexp(z, -1, keepdim=True)
        p = torch.where(finite, (z - lse).exp(), torch.zeros_like(z))
        difference = torch.where(finite, z - t, torch.zeros_like(z))
        # lse(t) - lse(z) = log E_p exp(-(z - t)), evaluated as log1p(E_p expm1(-(z - t))) so that near agreement
        # the first-order terms cancel analytically instead of between two rounded log-normalizers.
        shift = torch.log1p((p * torch.expm1(-difference)).sum(-1, keepdim=True))
        delta = torch.where(finite, difference + shift, torch.zeros_like(z))
        kl = (p * delta).sum(-1)
        ctx.save_for_backward(p, delta, kl)
        return kl

    @staticmethod
    def backward(ctx, grad):
        p, delta, kl = ctx.saved_tensors
        return grad[:, None] * p * (delta - kl[:, None]), None


def reverse_kl(logits, twin_logits):
    """KL(softmax(logits) || sg softmax(twin_logits)) per row; both are temperature-scaled masked logits."""
    return _ReverseKL.apply(logits, twin_logits.detach())


def jolt_terms(logits, twin_logits, labels, behavior_logp, top_ids, top_logq, advantage, *, beta, outcome,
               masked):
    """Per-position JOLT+ loss pieces from scaled/masked logits: (outcome -A U, KL(p || sg p~), total, details)."""
    from .policy_math import score_centered_surrogate
    kl = reverse_kl(logits, twin_logits)
    if advantage != 0:
        surrogate, details = score_centered_surrogate(torch.log_softmax(logits.float(), -1), labels, behavior_logp,
            top_ids, top_logq, importance=outcome['importance'], eps=outcome['eps'], masked=masked)
        outcome_loss = -float(advantage) * surrogate
    else:
        outcome_loss, details = torch.zeros_like(kl), {}
    return outcome_loss, kl, outcome_loss + beta * kl, details


def attempt_feedback(rollout):
    """Runtime error/repair feedback text of one scored earlier attempt (its reward details)."""
    reward = rollout.get('reward') or {}
    parts = []
    for key in ('feedback', 'diagnosis', 'errors', 'repair', 'failure'):
        if reward.get(key):
            value = reward[key]
            parts.append(f"{key}: {value if isinstance(value, str) else json.dumps(value, sort_keys=True)}")
    return '; '.join(parts)[:MAX_FEEDBACK_CHARS]


def root_calls(rollout, turns):
    """The episode's root invocation ids, read from its execution graph.

    A root is an ``invocation`` node (``call:<id>``) without a ``caller`` input. Rollouts captured
    without invocation events fall back to the call that owns the first captured turn.
    """
    calls, called = set(), set()
    for event in rollout.get('execution_graph') or ():
        if not isinstance(event, dict) or event.get('kind') != 'invocation':
            continue
        node = event.get('node')
        if not isinstance(node, str) or not node.startswith('call:'):
            continue
        calls.add(node[len('call:'):])
        if any(isinstance(edge, dict) and edge.get('port') == 'caller' for edge in event.get('inputs') or ()):
            called.add(node[len('call:'):])
    roots = calls - called
    if roots:
        return roots
    if calls:
        raise ValueError('execution graph has no root invocation (every invocation has a caller)')
    return {(turns[0].get('graph_turn') or '').rsplit('#turn', 1)[0]}


def privileged_source(rollouts, *, cases, rendering, custody, teacher_trajectories=None, command=None,
                      policy_root=None):
    """``source(rollout, replay, turns) -> {invocation: PrivilegedBlock}`` over the shared mechanism.

    Raises ``CustodyError`` (or ValueError for unsupported graphs) when the episode gets no privileged teacher.
    Attempt feedback comes from the same group's earlier scored student attempts.
    """
    from .privileged_context import CustodyError, privileged_blocks
    from .privileged_scoring import graph_invocations
    students = {}
    for row in rollouts:
        if row.get('jolt_role', 'student') == 'student':
            students.setdefault((row.get('task_id'), json.dumps(row.get('group'), sort_keys=True)), []).append(row)

    def build(rollout, replay, turns):
        case = cases.get(rollout.get('task_id'))
        if case is None:
            raise CustodyError('episode has no bound training world')
        group = students.get((rollout['task_id'], json.dumps(rollout.get('group'), sort_keys=True)), [])
        feedback = []
        for row in sorted(group, key=lambda item: item['sample'] if type(item.get('sample')) is int else -1):
            if (type(row.get('sample')) is int and type(rollout.get('sample')) is int
                    and row['sample'] < rollout['sample'] and row.get('reward') is not None):
                text = attempt_feedback(row)
                if text:
                    feedback.append({'case_id': case.get('id'), 'attempt': row['sample'], 'feedback': text})
        invocations = graph_invocations(replay, turns, root_calls=root_calls(rollout, turns))
        return privileged_blocks(rendering, case, invocations,
            custody=custody, teacher_trajectories=(teacher_trajectories or {}).get(rollout['task_id'], ()),
            attempt_feedback=feedback[-MAX_FEEDBACK_ATTEMPTS:], command=command, policy_root=policy_root)
    return build


def _renamed(invocations, suffix):
    """The same invocations under twin names; imports from graph invocations follow their twins."""
    from dataclasses import replace
    names = {node.name for node in invocations}
    return [replace(node, name=node.name + suffix,
                    imports=tuple(replace(edge, source=edge.source + suffix) if edge.source in names else edge
                                  for edge in node.imports))
            for node in invocations]


def _captured_blocks(turns):
    """Blocks a native-engine actor flagged as 'privileged' in its capture, keyed by graph turn."""
    from .privileged_context import PrivilegedBlock
    blocks = {}
    for turn in turns:
        own = 0
        for segment in (turn.get('sampling') or {}).get('prompt') or []:
            if segment.get('kind') != 'tokens':
                continue
            if segment.get('privileged'):
                blocks[turn['graph_turn']] = PrivilegedBlock(turn['graph_turn'], own, tuple(segment['token_ids']),
                                                             '', {})
                break
            own += len(segment['token_ids'])
    return blocks


def _before_itself(block):
    from dataclasses import replace
    if not block.token_ids or block.receipt.get('placement') != 'after_leading_system_turn':
        return block
    at = block.insert_at - len(block.token_ids)
    if at < 0:
        return block
    return replace(block, insert_at=at, receipt={**block.receipt, 'insert_at': at,
                                                  'placement': 'after_bos' if at else 'start'})


def add_privileged_twin(replay, rollout, graph, source, *, turns):
    """Append the episode's whole-graph twin to ``graph``; bind every sampled action to its twin query.

    Returns (actions, the row's JOLT+ record). A student episode refused by custody (or on an unsupported graph)
    keeps its plain outcome term: actions are unchanged and the record names the refusal. A teacher attempt must
    carry exactly the blocks the learner rebuilds for its student form.
    """
    from ..model.native_graph_plan import compile_graph, invocation_dict
    from .privileged_context import CustodyError
    from .privileged_scoring import student_form, teacher_form
    role = rollout.get('jolt_role', 'student')
    if role not in ('student', 'teacher'):
        raise ValueError('JOLT+ episode role must be student or teacher')
    captured = _captured_blocks(turns)
    if role == 'student' and captured:
        raise ValueError('a student episode was sampled with privileged context; mark it as a teacher attempt')
    try:
        if source is None:
            raise CustodyError('no privileged source is available to this learner')
        if role == 'teacher':
            # The declared insertion point (after the leading system turn) and the fact selection (by the
            # invocation's messages) are the same on the teacher-role tokens, so the learner rebuilds the exact
            # blocks; student_form verifies the attempt carried them token for token.
            blocks = source(rollout, replay, turns)
            if not any(block.token_ids for block in blocks.values()):
                raise CustodyError('no privileged fact fits any invocation of this teacher attempt')
            try:
                twin = student_form(replay, blocks)
            except ValueError:
                # Prompts without a leading system turn: on the teacher tokens the inserted block itself is the
                # leading system turn, so the rebuilt insertion point sits right after it.
                blocks = {name: _before_itself(block) for name, block in blocks.items()}
                twin = student_form(replay, blocks)
        else:
            blocks = source(rollout, replay, turns)
            if not any(block.token_ids for block in blocks.values()):
                raise CustodyError('no privileged fact fits any invocation of this episode')
            twin = teacher_form(replay, blocks)
    except (CustodyError, ValueError) as error:
        if role == 'teacher':
            raise ValueError('JOLT+ teacher attempt carries no verified privileged block: ' + str(error)) from error
        return list(replay.actions), {'role': role, 'privileged': None, 'refusal': str(error)}
    suffix = PRIVILEGED_SUFFIX[role]
    renamed = _renamed(twin.plan.invocations, suffix)
    used = {edge.source for node in replay.plan.invocations for edge in node.imports}
    external_sizes = {name: len(state.token_ids) for name, state in replay.external_blocks.items() if name in used}
    combined = compile_graph((*replay.plan.invocations, *renamed), external_sizes=external_sizes)
    if tuple(combined.offsets[:len(replay.plan.invocations)]) != tuple(replay.plan.offsets):
        raise ValueError('JOLT+ twin changed actual graph offsets')
    offsets = {node.name: offset for node, offset in zip(combined.invocations, combined.offsets)}
    if len(twin.actions) != len(replay.actions):
        raise ValueError('JOLT+ twin actions are not aligned with the sampled actions')
    actions = []
    for action, edited in zip(replay.actions, twin.actions):
        if (edited['turn'], edited['token_id'], edited['sample']) != (action['turn'], action['token_id'],
                                                                     action['sample']):
            raise ValueError('JOLT+ twin actions are not aligned with the sampled actions')
        actions.append({**action, 'twin_query': offsets[edited['turn'] + suffix] + edited['local_query']})
    for node in renamed:
        value = invocation_dict(node)
        value['token_ids'] = list(node.token_ids)
        value['target_weights'] = [0.] * len(node.token_ids)
        graph['nodes'].append(value)
    return actions, {'role': role, 'twin_turns': [node.name for node in renamed],
                     'privileged': {name: block.receipt for name, block in sorted(blocks.items())}}


def jolt_action_loss(model, hidden, actions, row, declaration, *, chunk_tokens=128, top_k=None, readout=None):
    """Per-action JOLT+ losses of one admitted episode from the single packed forward of actual + twin views."""
    from torch.utils.checkpoint import checkpoint
    from .native_replay import behavior_top_k_tensors
    if chunk_tokens < 1:
        raise ValueError('native RL loss chunks must be positive')
    info = row['jolt']
    role = info['role']
    privileged = info.get('privileged') is not None
    beta = (declaration['beta_s'] if role == 'student' else declaration['beta_t']) if privileged else 0.
    advantage = float(row['advantage'])
    if not math.isfinite(advantage):
        raise ValueError('nonfinite advantage')
    outcome = declaration['outcome']
    groups = {}
    for index, action in enumerate(actions):
        groups.setdefault((action['temperature'],
                           tuple(action['allowed_token_ids']) if action['allowed_token_ids'] is not None else None,
                           tuple(action['banned_token_ids']) if action.get('banned_token_ids') is not None else None),
                          []).append(index)
    terms = [None] * len(actions)
    sums = {'outcome': 0., 'kl': 0.}
    def score(states, twin_states, labels, behavior, ids, logq, temperature, allowed, banned):
        from ..common.sampling import masked_logits
        head = readout or model.logits
        logits = masked_logits(head(states), temperature, allowed=allowed, banned=banned)
        with torch.no_grad():  # the twin is a stop-gradient target; recomputed rather than stored
            twin = masked_logits(head(twin_states), temperature, allowed=allowed, banned=banned)
        outcome_loss, kl, total, _ = jolt_terms(logits, twin, labels, behavior, ids, logq, advantage,
            beta=beta, outcome=outcome, masked=bool(allowed is not None or banned))
        return total, outcome_loss.detach(), kl.detach()
    from ..common.native_higher_order import capture_reference_execution, active_reference_arithmetic
    if active_reference_arithmetic() is not None:
        from ..model.native_adapters import capture_native_execution
        score = capture_reference_execution(capture_native_execution(score))
    for (temperature, allowed, banned), indices in groups.items():
        for start in range(0, len(indices), chunk_tokens):
            selected = indices[start:start+chunk_tokens]
            chosen = [actions[i] for i in selected]
            device = hidden.device
            queries = torch.tensor([action['query'] for action in chosen], device=device)
            twin_queries_ = torch.tensor([action.get('twin_query', action['query']) for action in chosen], device=device)
            labels = torch.tensor([action['token_id'] for action in chosen], device=device)
            behavior = torch.tensor([action['behavior_logp'] for action in chosen], device=device, dtype=torch.float32)
            if top_k is None:
                ids, logq = behavior_top_k_tensors(chosen, outcome['k'], device)
            else:
                rows = torch.tensor(selected)
                ids, logq = top_k[0][rows].to(device), top_k[1][rows].to(device)
            total, outcome_loss, kl = checkpoint(score, hidden[queries], hidden[twin_queries_].detach(), labels,
                behavior, ids, logq, temperature, allowed, banned, use_reentrant=False)
            for index, value in zip(selected, total.unbind()):
                terms[index] = value
            sums['outcome'] += float(outcome_loss.sum())
            sums['kl'] += float(kl.sum())
    return torch.stack(terms), {'role': role, 'privileged': privileged, 'positions': len(actions), 'beta': beta,
                                'advantage': advantage,
                                'outcome_sum': sums['outcome'], 'kl_sum': sums['kl']}


def summarize_jolt(parts, count):
    """Per-term contributions to the batch loss (each divided by the batch's sampled-token count)."""
    result = {'positions': {'student': 0, 'teacher': 0}, 'unprivileged_episodes': 0, 'privileged_episodes': 0}
    for name in ('student_outcome', 'student_opsd', 'teacher_outcome', 'teacher_kl'):
        result[name] = 0.
    for part in parts:
        role = part['role']
        result['privileged_episodes' if part['privileged'] else 'unprivileged_episodes'] += 1
        result['positions'][role] += part['positions']
        result[role + '_outcome'] += part['outcome_sum'] / count
        result[role + ('_opsd' if role == 'student' else '_kl')] += part['beta'] * part['kl_sum'] / count
    result['mean_kl'] = {role: (sum(p['kl_sum'] for p in parts if p['role'] == role) / result['positions'][role]
                                if result['positions'][role] else None) for role in ('student', 'teacher')}
    return result
