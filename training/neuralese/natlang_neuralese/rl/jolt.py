"""JOLT+ self-teaching (TRAINING_INSIGHTS_20261028.md section 7): one policy as privileged teacher and as student.

Every admitted episode is scored in ONE packed graph forward that holds two views of the whole actual graph:

* the sampled view, exactly as the actor executed it (all producer/consumer turns, imports at their recorded
  positions, gradients through the real dependencies), and
* its twin: the same invocations with the privileged training-world block added (student episodes) or removed
  (teacher episodes) at the front of every invocation. Import positions, producer export spans and imported
  source spans are shifted by the actual block lengths, so the twin is itself a positionally exact whole graph
  (shared positional splice), never a set of detached turns. The twin is a stop-gradient target.

Per sampled position of a role-r episode (r = student or teacher), with p the sampled view and p~ the twin under
the actor's exact temperature/mask law:

    loss_t = -A_r U_t(score-centered/1)  +  beta_r KL(p || sg[p~])

Student episodes (no privileged context): KL(pi(.|s) || sg pi(.|s,c)) is J_OPSD (reverse KL to the self-teacher)
and A_S is the student group's return minus its mean (JOLT+ outcome term). Teacher episodes (sampled under
privileged context): A_T is the teacher group's centered return and KL(pi(.|s,c) || sg pi(.|s)) keeps the teacher's
guidance within reach of the student (target pi_T* proportional to pi_S exp(A/beta_T)). Both outcome terms use the
score-centered estimator, so quantized/stale actors stay usable; the KL terms are computed by the learner's own
forward and have no sampler mismatch.

Privileged context comes only from training worlds (split 'train', not sharing any identifier with a protected
development/held packet): the case's authored reference, expected results and decisive oracle notes, verified
teacher trajectories of the same world and runtime feedback of the student's earlier attempts in the same group.
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
                'outcome': {'schema': 'score-centered/1', 'k': 128, 'importance': 'none', 'eps': 1e-6}}
PRIVILEGED_SCHEMA = 'natlang.jolt-privileged-context/1'
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


# ---- privileged training-world context and its leakage guard ----

def _identifiers(case):
    curriculum = case.get('curriculum') if isinstance(case.get('curriculum'), dict) else {}
    values = {('id', case.get('id')), ('split_group', curriculum.get('split_group')),
              ('pair_group', curriculum.get('pair_group'))}
    for key in ('source_groups', 'source_ids', 'dataset_records'):
        for item in case.get(key) or ():
            values.add((key.rstrip('s') if key != 'dataset_records' else 'dataset_record', item))
    return {(kind, value) for kind, value in values if isinstance(value, str) and value}


def protected_identifiers(paths):
    """Identifiers of protected development/held packets (cases.jsonl files); none may feed privileged context."""
    found = set()
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    found |= {value for _, value in _identifiers(json.loads(line))}
    return frozenset(found)


def registered_protected_packets(repo):
    """Every registered evaluation-only packet's cases file (development and held)."""
    repo = Path(repo)
    registry = json.loads((repo / 'training/neuralese_corpora.json').read_text())
    paths = []
    for entry in registry.get('corpora') or ():
        if isinstance(entry, dict) and entry.get('admission') == 'evaluation-only-never-training':
            cases = repo / entry['path'] / 'cases.jsonl'
            if cases.is_file():
                paths.append(cases)
    if not paths:
        raise ValueError('JOLT+ leakage guard found no registered protected evaluation packet')
    return paths


def require_training_world(case, protected):
    """Leakage guard: a privileged source is a training world sharing no identity with protected packets."""
    if not isinstance(case, dict) or case.get('split') != 'train':
        raise ValueError('JOLT+ privileged context comes only from train-split worlds')
    if not isinstance(protected, frozenset) or not protected:
        raise ValueError('JOLT+ privileged context needs the protected development/held identifiers')
    overlap = sorted(value for _, value in _identifiers(case) if value in protected)
    if overlap:
        raise ValueError('JOLT+ privileged context shares identity with a protected evaluation packet: '
                         + ', '.join(overlap[:4]))


def _compact(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def attempt_feedback(rollout):
    """Runtime error/repair feedback text of one earlier attempt (scored reward details only)."""
    reward = rollout.get('reward') or {}
    parts = []
    for key in ('feedback', 'diagnosis', 'errors', 'repair', 'failure'):
        if reward.get(key):
            parts.append(f'{key}: {_compact(reward[key])}')
    if not parts and type(reward.get('quality')) in (int, float):
        parts.append(f"quality: {reward['quality']}")
    return '; '.join(parts)[:MAX_FEEDBACK_CHARS]


def privileged_context(case, *, protected, earlier_attempts=(), verified_trajectories=()):
    """The privileged block of one training world (text and its exact provenance)."""
    require_training_world(case, protected)
    curriculum = case.get('curriculum') if isinstance(case.get('curriculum'), dict) else {}
    semantics = case.get('semantics') if isinstance(case.get('semantics'), dict) else {}
    lines, sources = [], []
    for label, value, source in (
            ('Reference solution', curriculum.get('reference'), 'authored_reference'),
            ('Expected results', semantics.get('expected'), 'reference_values'),
            ('Oracle checks', curriculum.get('decisive'), 'oracle_checks')):
        if value:
            lines.append(f'{label}: {_compact(value)}')
            sources.append(source)
    for trajectory in verified_trajectories:
        lines.append(f'Verified solution trajectory: {_compact(trajectory)}')
        sources.append('verified_teacher_trajectory')
    feedback = [text for text in (attempt_feedback(row) for row in list(earlier_attempts)[:MAX_FEEDBACK_ATTEMPTS])
                if text]
    for number, text in enumerate(feedback, 1):
        lines.append(f'Earlier attempt {number} feedback: {text}')
        sources.append('attempt_feedback')
    if not lines:
        raise ValueError('JOLT+ training world has no privileged reference material')
    text = ('<training-reference>\nPrivileged training reference for this exact task; it is not available when the '
            'task runs.\n' + '\n'.join(lines) + '\n</training-reference>\n')
    from ..common.hashing import canonical_json_sha256_hex
    return {'schema': PRIVILEGED_SCHEMA, 'task_id': case['id'], 'text': text, 'sources': sources,
            'earlier_samples': [row.get('sample') for row in list(earlier_attempts)[:MAX_FEEDBACK_ATTEMPTS]],
            'sha256': canonical_json_sha256_hex({'text': text, 'task_id': case['id']})}


def privileged_source(rollouts, *, cases, protected, tokenize, verified_trajectories=None):
    """``privileged(rollout, members)`` for prepare_objective: exact training-world blocks behind the leakage guard.

    Student episode: reference material plus feedback of the same group's earlier student attempts. Teacher
    attempt: its recorded ``privileged_context`` must be exactly what this builder produces for its world and
    the named earlier student attempts.
    """
    if not protected:
        raise ValueError('JOLT+ needs protected development/held identifiers for its leakage guard')
    students = {}
    for row in rollouts:
        if row.get('jolt_role', 'student') == 'student':
            students.setdefault((row.get('task_id'), json.dumps(row.get('group'), sort_keys=True)), []).append(row)
    def build(rollout, members):
        if rollout.get('split') != 'train':
            raise ValueError('JOLT+ privileged context is only for train-split episodes')
        case = cases.get(rollout['task_id'])
        if case is None:
            raise ValueError('JOLT+ episode has no bound training world')
        group = sorted(students.get((rollout['task_id'], json.dumps(rollout.get('group'), sort_keys=True)), []),
                       key=lambda row: (row.get('sample') is None, row.get('sample')))
        trajectories = (verified_trajectories or {}).get(rollout['task_id'], ())
        if rollout.get('jolt_role', 'student') == 'student':
            earlier = [row for row in group if row.get('sample') is not None and rollout.get('sample') is not None
                       and row['sample'] < rollout['sample'] and row.get('reward') is not None]
            context = privileged_context(case, protected=protected, earlier_attempts=earlier,
                                         verified_trajectories=trajectories)
        else:
            recorded = rollout.get('privileged_context')
            if not isinstance(recorded, dict):
                raise ValueError('JOLT+ teacher attempt lacks its recorded privileged context')
            by_sample = {row.get('sample'): row for row in group}
            if any(sample not in by_sample for sample in recorded.get('earlier_samples') or ()):
                raise ValueError('JOLT+ teacher context names student attempts outside its admitted group')
            context = privileged_context(case, protected=protected,
                earlier_attempts=[by_sample[sample] for sample in recorded.get('earlier_samples') or ()],
                verified_trajectories=trajectories)
            if context != recorded:
                raise ValueError('JOLT+ teacher attempt privileged context differs from its training world')
        return {'token_ids': list(tokenize(context['text'])), 'record': context}
    return build


# ---- whole-graph twin ----

def privileged_prefix_lengths(turns):
    """Per graph turn: length of the leading privileged token segment of its capture (0 when absent)."""
    lengths = {}
    for number, turn in enumerate(turns):
        name = turn.get('graph_turn') or f'turn-{number}'
        prompt = (turn.get('sampling') or {}).get('prompt') or []
        flagged = [index for index, segment in enumerate(prompt) if segment.get('privileged')]
        if flagged and (flagged != [0] or prompt[0].get('kind') != 'tokens'):
            raise ValueError('a privileged block must be the single leading token segment of its invocation')
        lengths[name] = len(prompt[0]['token_ids']) if flagged else 0
    return lengths


def twin_invocations(invocations, *, role, prefix_ids=None, prefix_lengths=None):
    """Twin of every invocation of the actual graph, with exact import/export coordinates.

    ``role='student'`` prepends ``prefix_ids`` (the privileged block) to every invocation; ``role='teacher'``
    removes each invocation's leading privileged block of ``prefix_lengths[name]`` tokens. Imports from graph
    invocations are remapped to their twins (whole teacher-view graph); external sources keep their names.
    """
    from ..model.native_graph_plan import ImportSpan, Invocation
    suffix = PRIVILEGED_SUFFIX[role]
    names = {node.name for node in invocations}
    if any(node.cache_prefix for node in invocations):
        raise ValueError('JOLT+ twins of raw prefix-cache invocations are not supported')
    if role == 'student':
        prefix = tuple(int(token) for token in prefix_ids)
        if not prefix:
            raise ValueError('student twin needs a nonempty privileged block')
        shift = {name: len(prefix) for name in names}
    else:
        shift = {name: -int(prefix_lengths.get(name, 0)) for name in names}
        if not any(shift.values()):
            raise ValueError('teacher episode has no captured privileged block')
    result = []
    for node in invocations:
        delta = shift[node.name]
        if delta >= 0:
            tokens = prefix + tuple(node.token_ids)
        else:
            tokens = tuple(node.token_ids)[-delta:]
        imports = []
        for edge in node.imports:
            if edge.at + delta < 0:
                raise ValueError('an import precedes the privileged block it would remove')
            internal = edge.source in names
            source_delta = shift[edge.source] if internal else 0
            if internal and edge.start + source_delta < 0:
                raise ValueError('an exported span overlaps a privileged block')
            imports.append(ImportSpan(edge.at + delta, edge.source + suffix if internal else edge.source,
                                      edge.start + source_delta, edge.stop + source_delta,
                                      logical=edge.logical, stop_gradient=edge.stop_gradient))
        def moved(spans):
            out = tuple((start + delta, stop + delta) for start, stop in spans)
            if any(start < 0 for start, _ in out):
                raise ValueError('a producing span overlaps a privileged block')
            return out
        result.append(Invocation(node.name + suffix, tokens, imports=tuple(imports),
                                 producing=moved(node.producing), logical_producing=moved(node.logical_producing)))
    return result, shift


def twin_queries(actions, plan, twins, shift, external_sizes, *, role):
    """Query row of each action in the combined (actual + twin) plan, which keeps actual offsets unchanged."""
    from ..model.native_graph_plan import compile_graph
    combined = compile_graph((*plan.invocations, *twins), external_sizes=external_sizes)
    if tuple(combined.offsets[:len(plan.invocations)]) != tuple(plan.offsets):
        raise ValueError('JOLT+ twin changed actual graph offsets')
    offsets = {node.name: offset for node, offset in zip(combined.invocations, combined.offsets)}
    suffix = PRIVILEGED_SUFFIX[role]
    queries = []
    for action in actions:
        local = action['local_query'] + shift[action['turn']]
        if local < 0:
            raise ValueError('a sampled position lies inside a privileged block')
        queries.append(offsets[action['turn'] + suffix] + local)
    return combined, queries


# ---- loss ----

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


def add_privileged_twin(replay, rollout, members, graph, privileged, *, turns):
    """Append the episode's whole-graph twin to ``graph`` and bind every sampled action to its twin query.

    Returns (actions with 'twin_query', the row's JOLT+ record). Student episodes get the privileged block that
    ``privileged`` builds for this exact training world; teacher episodes must carry that same block as the
    leading privileged segment of every captured invocation, and their twin removes it.
    """
    from ..model.native_graph_plan import invocation_dict
    role = rollout.get('jolt_role', 'student')
    block = privileged(rollout, members)
    token_ids = [int(token) for token in block['token_ids']]
    if role == 'student':
        if any(privileged_prefix_lengths(turns).values()):
            raise ValueError('a student episode was sampled with privileged context; mark it as a teacher attempt')
        twins, shift = twin_invocations(replay.plan.invocations, role='student', prefix_ids=token_ids)
    else:
        lengths = privileged_prefix_lengths(turns)
        for number, turn in enumerate(turns):
            name = turn.get('graph_turn') or f'turn-{number}'
            prompt = (turn.get('sampling') or {}).get('prompt') or []
            if not lengths.get(name) or prompt[0]['token_ids'] != token_ids:
                raise ValueError('teacher attempt invocation lacks the exact privileged block of its training world')
        twins, shift = twin_invocations(replay.plan.invocations, role='teacher', prefix_lengths=lengths)
    used = {edge.source for node in replay.plan.invocations for edge in node.imports}
    external_sizes = {name: len(state.token_ids) for name, state in replay.external_blocks.items() if name in used}
    _, queries = twin_queries(replay.actions, replay.plan, twins, shift, external_sizes, role=role)
    for twin in twins:
        value = invocation_dict(twin)
        value['token_ids'] = list(twin.token_ids)
        value['target_weights'] = [0.] * len(twin.token_ids)
        graph['nodes'].append(value)
    actions = [{**action, 'twin_query': query} for action, query in zip(replay.actions, queries)]
    return actions, {'role': role, 'twin_turns': [twin.name for twin in twins],
                     'privileged': {key: block['record'][key] for key in ('schema', 'task_id', 'sha256', 'sources')},
                     'privileged_tokens': len(token_ids)}


def jolt_action_loss(model, hidden, actions, row, declaration, *, chunk_tokens=128, top_k=None, readout=None):
    """Per-action JOLT+ losses of one admitted episode from the single packed forward of actual + twin views."""
    from torch.utils.checkpoint import checkpoint
    from .native_replay import behavior_top_k_tensors
    if chunk_tokens < 1:
        raise ValueError('native RL loss chunks must be positive')
    info = row['jolt']
    role = info['role']
    beta = declaration['beta_s'] if role == 'student' else declaration['beta_t']
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
            twin_queries_ = torch.tensor([action['twin_query'] for action in chosen], device=device)
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
    return torch.stack(terms), {'role': role, 'positions': len(actions), 'beta': beta, 'advantage': advantage,
                                'outcome_sum': sums['outcome'], 'kl_sum': sums['kl']}


def summarize_jolt(parts, count):
    """Per-term contributions to the batch loss (each divided by the batch's sampled-token count)."""
    result = {'positions': {'student': 0, 'teacher': 0}}
    for name in ('student_outcome', 'student_opsd', 'teacher_outcome', 'teacher_kl'):
        result[name] = 0.
    for part in parts:
        role = part['role']
        result['positions'][role] += part['positions']
        result[role + '_outcome'] += part['outcome_sum'] / count
        result[role + ('_opsd' if role == 'student' else '_kl')] += part['beta'] * part['kl_sum'] / count
    result['mean_kl'] = {role: (sum(p['kl_sum'] for p in parts if p['role'] == role) / result['positions'][role]
                                if result['positions'][role] else None) for role in ('student', 'teacher')}
    return result
