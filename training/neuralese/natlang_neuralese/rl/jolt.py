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


def group_feedback(case, students, sample):
    """Attempt feedback an episode's privileged teacher sees: its group's earlier scored student attempts.

    The actor (teacher-attempt collection) and the learner (privileged twin rebuild) both use this selection, so a
    teacher attempt carries exactly the block its learner rebuilds.
    """
    feedback = []
    for row in sorted(students, key=lambda item: item['sample'] if type(item.get('sample')) is int else -1):
        if (type(row.get('sample')) is int and type(sample) is int and row['sample'] < sample
                and row.get('reward') is not None):
            text = attempt_feedback(row)
            if text:
                feedback.append({'case_id': case.get('id'), 'attempt': row['sample'], 'feedback': text})
    return feedback[-MAX_FEEDBACK_ATTEMPTS:]


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


def privileged_source(rollouts, *, cases, rendering, custody, fetch_text=None, teacher_trajectories=None):
    """``source(rollout, replay, turns) -> ({invocation: PrivilegedBlock}, telemetry)``: the episode's verified guidance.

    The actor recorded each model turn's guidance (``privileged_context.captured_block``): the policy's own
    ``writeGuidance`` state imported after the leading system turn (teacher attempts), the same writer's state
    written but not imported (student attempts, for the OPSD twin), or the recorded text fallback. The learner
    re-verifies custody, every credited fact (this world's ``case_facts`` with the group's ``group_feedback``),
    visibility to the consuming invocation (root/child from ``root_calls``), the declared rendering and that the
    in-graph writer's own prompt carried those facts. Raises ``CustodyError``/``ValueError`` when an episode has no
    verified guidance. ``fetch_text(state_id)`` reads a written state's text for the verbatim telemetry.
    """
    from ..common.custody import CustodyError
    from .privileged_context import (captured_block, case_facts, reproduces_reference,
                                     verify_captured_block, writer_saw)
    from .privileged_scoring import graph_invocations, writer_span
    students = {}
    for row in rollouts:
        if row.get('jolt_role', 'student') == 'student':
            students.setdefault((row.get('task_id'), json.dumps(row.get('group'), sort_keys=True)), []).append(row)

    def build(rollout, replay, turns):
        case = cases.get(rollout.get('task_id'))
        if case is None:
            raise CustodyError('episode has no bound training world')
        group = students.get((rollout['task_id'], json.dumps(rollout.get('group'), sort_keys=True)), [])
        receipt, facts = case_facts(case, custody=custody,
            teacher_trajectories=(teacher_trajectories or {}).get(rollout['task_id'], ()),
            attempt_feedback=group_feedback(case, group, rollout.get('sample')))
        invocations = {item['name']: {'name': item['name'].rsplit('#turn', 1)[0], 'role': item['role'],
                                      'text': item['text']}
                       for item in graph_invocations(replay, turns, root_calls=root_calls(rollout, turns))}
        by_turn = {turn.get('graph_turn'): turn for turn in turns}
        blocks, telemetry = {}, {'blocks': 0, 'writer_blocks': 0, 'text_fallbacks': 0, 'verbatim_reference': 0,
                                 'fallback_reasons': []}
        for name, turn in by_turn.items():
            block = captured_block(turn)
            if block is None:
                continue
            used = verify_captured_block(block, rendering, custody_receipt=receipt, facts=facts,
                                         invocation=invocations[name])
            telemetry['blocks'] += 1
            if block.receipt['mode'] == 'text':
                telemetry['text_fallbacks'] += 1
                telemetry['fallback_reasons'].append(block.receipt['fallback'])
            elif block.state_id is not None:
                writer, _, _ = writer_span(replay, block)
                writer_saw(used, '\n'.join(message['content'] for message in by_turn[writer].get('messages') or []
                                          if isinstance(message.get('content'), str)))
                telemetry['writer_blocks'] += 1
                if fetch_text is not None and reproduces_reference(fetch_text(block.state_id), used):
                    telemetry['verbatim_reference'] += 1
            blocks[name] = block
        if not blocks:
            raise CustodyError('the actor recorded no guidance for this episode')
        return blocks, telemetry
    build.rendering = rendering
    return build


def twin_sources(row):
    """{twin invocation: the sampled invocation it mirrors} of one admitted objective row ({} without a twin).

    A twin runs the actor of its sampled invocation: per-invocation coefficient actors (tuning stacks) declare the
    same adapters for both, and the twin is a stop-gradient target, never a separately sampled actor.
    """
    info = row.get('jolt') or {}
    suffix = PRIVILEGED_SUFFIX.get(info.get('role'))
    result = {}
    for name in info.get('twin_turns') or ():
        if suffix is None or not name.endswith(suffix) or not name[:-len(suffix)]:
            raise ValueError('JOLT+ twin turn does not name its sampled invocation')
        result[name] = name[:-len(suffix)]
    return result


def _renamed(invocations, suffix):
    """The same invocations under twin names; imports from twinned invocations follow their twins."""
    from dataclasses import replace
    names = {node.name for node in invocations}
    return [replace(node, name=node.name + suffix,
                    imports=tuple(replace(edge, source=edge.source + suffix) if edge.source in names else edge
                                  for edge in node.imports))
            for node in invocations]


def _as_insertion(block, rendering_turn):
    """A student turn's write-only guidance as the import block its OPSD twin inserts."""
    from dataclasses import replace
    tokens, offset = rendering_turn
    return replace(block, token_ids=tuple(tokens), import_offset=offset)


def add_privileged_twin(replay, rollout, graph, source, *, turns):
    """Append the episode's whole-graph twin to ``graph``; bind every sampled action to its twin query.

    Returns (actions, the row's JOLT+ record). The in-run guidance policy's turns (``selectGuidanceFacts`` and the
    ``writeGuidance`` producer) stay in the sampled view, are imported by name from the twin and have no twin.

    * Teacher attempt: the twin is ``student_form`` (guidance turns and their imports removed). The policy turns
      keep the teacher outcome term (whole-graph RL through the producer/consumer import) with no KL twin.
    * Student episode: the twin is ``teacher_form`` with the writer's write-only state imported at the declared
      place (OPSD target, stop-gradient). The policy turns carry no student loss (they do not affect the student's
      outcome): they are removed from the actions and their target weights.

    A student episode without verified guidance keeps its plain outcome term; the record names the refusal.
    """
    from ..model.native_graph_plan import compile_graph, invocation_dict
    from ..common.custody import CustodyError
    from .privileged_scoring import student_form, teacher_form, writer_span
    role = rollout.get('jolt_role', 'student')
    if role not in ('student', 'teacher'):
        raise ValueError('JOLT+ episode role must be student or teacher')
    flagged = policy_turns(turns)
    for turn in turns:  # integrity, not custody: never turned into a refusal
        capture = turn.get('sampling') or {}
        sampled = any(segment.get('privileged') for segment in capture.get('prompt') or [])
        mode = (capture.get('privileged') or {}).get('mode')
        if role == 'student' and (sampled or mode not in (None, 'write-only')):
            raise ValueError('a student episode was sampled with guidance; mark it as a teacher attempt')
        if (sampled or mode is not None) and capture.get('schema') == 'natlang.native-decision-capture/1':
            raise ValueError('a decision readout carries guidance')
        if (sampled or mode is not None) and turn.get('graph_turn') in flagged:
            raise ValueError('a guidance policy turn carries guidance itself')
    try:
        if source is None:
            raise CustodyError('no privileged source is available to this learner')
        blocks, telemetry = source(rollout, replay, turns)
        modes = {block.receipt['mode'] for block in blocks.values()}
        if role == 'teacher':
            if 'write-only' in modes:
                raise ValueError('a teacher attempt imports its guidance; write-only guidance belongs to students')
            blocks = {name: block for name, block in blocks.items() if block.token_ids}
            if not blocks:
                raise CustodyError('no guidance entered any invocation of this teacher attempt')
            twin = student_form(replay, blocks)

        else:
            turn = source.rendering.state_turn()
            blocks = {name: _as_insertion(block, turn) for name, block in blocks.items()}
            twin = teacher_form(replay, blocks)
        writers = {writer_span(replay, block)[0] for block in blocks.values() if block.state_id is not None}
        if not writers <= flagged:
            raise ValueError('guidance was imported from a turn not recorded as the guidance policy')
    except (CustodyError, ValueError) as error:
        if role == 'teacher':
            raise ValueError('JOLT+ teacher attempt carries no verified guidance: ' + str(error)) from error
        return _without(replay, graph, flagged), {'role': role, 'privileged': None, 'refusal': str(error),
                                                  'policy_turns': sorted(flagged)}
    suffix = PRIVILEGED_SUFFIX[role]
    renamed = _renamed([node for node in twin.plan.invocations if node.name not in flagged], suffix)
    used = {edge.source for node in replay.plan.invocations for edge in node.imports}
    external_sizes = {name: len(state.token_ids) for name, state in replay.external_blocks.items() if name in used}
    combined = compile_graph((*replay.plan.invocations, *renamed), external_sizes=external_sizes)
    if tuple(combined.offsets[:len(replay.plan.invocations)]) != tuple(replay.plan.offsets):
        raise ValueError('JOLT+ twin changed actual graph offsets')
    offsets = {node.name: offset for node, offset in zip(combined.invocations, combined.offsets)}
    if len(twin.actions) != len(replay.actions):
        raise ValueError('JOLT+ twin actions are not aligned with the sampled actions')
    kept = _without(replay, graph, flagged if role == 'student' else set())
    actions = []
    for action, edited in zip(replay.actions, twin.actions):
        if (edited['turn'], edited['token_id'], edited['sample']) != (action['turn'], action['token_id'],
                                                                     action['sample']):
            raise ValueError('JOLT+ twin actions are not aligned with the sampled actions')
        if action['turn'] in flagged:
            if role == 'teacher':
                actions.append(dict(action))  # outcome term only: the policy turns are the same in both views
            continue
        actions.append({**action, 'twin_query': offsets[edited['turn'] + suffix] + edited['local_query']})
    if len(actions) != len(kept):
        raise ValueError('JOLT+ twin lost a sampled action')
    # Sampled decision readouts (rl.decision_policy): each option's continuation queries in its twin option node.
    # The privileged edit lies in the shared prompt, so a continuation keeps its tokens and moves by the edit delta.
    originals = {node.name: node for node in replay.plan.invocations}
    edited = {node.name: node for node in twin.plan.invocations}
    decision_twins = []
    for action in replay.decision_actions:
        per_option = []
        for name, option in zip(action['option_nodes'], action['options']):
            before, after = originals[name], edited[name]
            delta = len(after.token_ids) - len(before.token_ids)
            continuation = before.token_ids[action['prompt_tokens']:]
            if (action['prompt_tokens'] + delta < 1
                    or tuple(after.token_ids[action['prompt_tokens'] + delta:]) != tuple(continuation)):
                raise ValueError('JOLT+ twin edit crosses a decision option continuation')
            per_option.append([{**item, 'query': offsets[name + suffix] + item['local_query'] + delta}
                               for item in option])
        decision_twins.append(per_option)

    for node in renamed:
        value = invocation_dict(node)
        value['token_ids'] = list(node.token_ids)
        value['target_weights'] = [0.] * len(node.token_ids)
        graph['nodes'].append(value)
    return actions, {'role': role, 'twin_turns': [node.name for node in renamed], 'policy_turns': sorted(flagged),
                     'guidance': telemetry,
                     'privileged': {name: block.receipt for name, block in sorted(blocks.items())},
                     **({'decision_twin_queries': decision_twins} if decision_twins else {})}



def policy_turns(turns):
    """Graph turns the actor recorded as in-run guidance policy (``selectGuidanceFacts``/``writeGuidance`` calls)."""
    return {turn.get('graph_turn') for turn in turns if (turn.get('sampling') or {}).get('guidance_policy')}


def _without(replay, graph, writers):
    """Sampled actions minus the given policy turns' tokens (also removed from the graph's target weights)."""
    starts = dict(zip((node.name for node in replay.plan.invocations), replay.plan.offsets))
    nodes = {node['name']: node for node in graph['nodes']}
    kept = []
    for action in replay.actions:
        if action['turn'] in writers:
            nodes[action['turn']]['target_weights'][action['query'] + 1 - starts[action['turn']]] = 0.
        else:
            kept.append(action)
    return kept


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
                                'guidance': info.get('guidance'),
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
    guidance = [part['guidance'] for part in parts if part.get('guidance')]
    result['guidance'] = {key: sum(item[key] for item in guidance)
                          for key in ('blocks', 'writer_blocks', 'text_fallbacks', 'verbatim_reference')}
    result['mean_kl'] = {role: (sum(p['kl_sum'] for p in parts if p['role'] == role) / result['positions'][role]
                                if result['positions'][role] else None) for role in ('student', 'teacher')}
    return result


TEACHER_SEED_OFFSET = 900
GUIDANCE_RUN_SCHEMA = 'natlang.guidance-run/1'


def guidance_controls(base, *, role, custody_receipt, facts, selection=None):
    """Run-level chat controls of one JOLT+ episode: the world's custody-checked facts for the in-run guidance writer.

    The natlang runtime (``applications/guidance-writer``) filters them by visibility per invocation, selects with
    ``selectGuidanceFacts`` (``selection``: the pluggable setting, ``nl`` by default) and runs ``writeGuidance``;
    a teacher attempt imports the written state, a student attempt writes it without importing it (OPSD twin).
    """
    if role not in ('teacher', 'student'):
        raise ValueError('guidance role is teacher or student')
    from .privileged_context import fact_request
    return {**base, 'x_natlang_guidance': {'schema': GUIDANCE_RUN_SCHEMA, 'role': role, 'custody': custody_receipt,
                                           'facts': [fact_request(fact) for fact in facts],
                                           **({'selection': selection} if selection is not None else {})}}


def _load_custody(custody):
    if custody is not None:
        return custody, None
    from ..common.custody import CustodyError, ProtectedWorlds
    try:
        return ProtectedWorlds.load(), None
    except CustodyError as error:
        return None, str(error)


def student_guidance(student_entries, *, ids, cases, config, out, custody=None, selection=None):
    """Bind each student entry to its world's guidance controls (write-only writer for the OPSD twin).

    ``ids``: the prepared packet's case ids by queue ``index``. A world refused by custody keeps the plain controls;
    its students then train on their outcome term only.
    Returns the refusal reason when custody itself is unavailable.
    """
    from pathlib import Path
    from ..common.jsonio import iter_jsonl_rows, write_json_durable
    from .privileged_context import CustodyError, case_facts
    custody, refusal = _load_custody(custody)
    if custody is None:
        return refusal
    worlds = {case.get('id'): case for _, case in iter_jsonl_rows(Path(cases))}
    base = json.loads(Path(config).read_text())
    out = Path(out)
    for entry in student_entries:
        case = worlds.get(ids[int(entry['index'])])
        if case is None:
            continue
        try:
            receipt, facts = case_facts(case, custody=custody)
        except CustodyError:
            continue
        if not facts:
            continue
        path = out / f"row-{entry['group']:03d}" / f"student-{entry['sample']:02d}-chat-request-config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_durable(path, guidance_controls(base, role='student', custody_receipt=receipt, facts=facts,
                                                   selection=selection))
        entry['chat_request_config'] = str(path)
    return None


PLAN_POLICY_PATH = 'applications/guidance-writer/planTeacherAttempts.nl'
MAX_ATTEMPTS_PER_GROUP = 2


def plan_command(endpoint, repo=None):
    """``natlang call`` of the NL attempt plan against the actor endpoint (the controller is not a natlang runtime)."""
    from pathlib import Path
    repo = Path(repo or Path(__file__).resolve().parents[4])
    return {'argv': ['node', str(repo / 'ts-host/bin/natlang.mjs'), 'call', str(repo / PLAN_POLICY_PATH)],
            'env': {'NATLANG_SERVER': endpoint.rstrip('/')}}


def plan_teacher_attempts(groups, *, default, budget, command=None):
    """planTeacherAttempts(groups, budget) -> {group: attempts}; NL by default when ``command`` is given.

    ``groups``: ``[{group, taskId, family, students, successes, meanQuality, feedback, facts}]`` of the scored
    student groups whose worlds have facts. The crisp plan gives every group ``default`` attempts. The NL answer is
    validated (known groups, integers in [0, MAX_ATTEMPTS_PER_GROUP], total within ``budget``); a failed or invalid
    NL plan falls back to crisp and the reason is returned. Returns ``(plan, record)``.
    """
    import os
    import subprocess
    crisp = {item['group']: default for item in groups}
    if command is None or not groups:
        return crisp, {'implementation': 'crisp'}
    try:
        response = subprocess.run(command['argv'], input=json.dumps({'groups': groups, 'budget': budget}), text=True,
                                  capture_output=True, check=True, timeout=600,
                                  env={**os.environ, **command.get('env', {})})
        answer = json.loads(response.stdout)
        known = {item['group'] for item in groups}
        plan = {}
        for item in answer:
            group, count = item.get('group'), item.get('attempts')
            if group not in known or group in plan or type(count) is not int or not 0 <= count <= MAX_ATTEMPTS_PER_GROUP:
                raise ValueError('planTeacherAttempts must return known groups once with 0..2 attempts')
            plan[group] = count
        if sum(plan.values()) > budget:
            raise ValueError('planTeacherAttempts exceeded its attempt budget')
        return {group: plan.get(group, 0) for group in known}, {'implementation': 'nl'}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError) as error:
        return crisp, {'implementation': 'crisp', 'nl_failure': str(error)[:400]}


def teacher_entries(student_entries, student_rows, *, cases, config, out, attempts, seed_base, custody=None,
                    selection=None, plan_command=None):
    """JOLT+ teacher-attempt collector entries for the scored student groups of one collection.

    Each attempt samples the same world after its group's students with ``guidance_controls`` (role teacher): the
    custody-checked facts of the world plus the group's ``group_feedback``. The runtime's writer writes each
    invocation's guidance inside the episode and the engine imports it at the declared place; the learner
    re-verifies what the actor recorded. Returns ``{'entries': [...], 'refusal': None | reason}``; a world without
    custody or facts gets no teacher attempt. ``attempts`` is the crisp count per group and its budget scale;
    ``plan_command`` (``plan_command(endpoint)``) runs the NL ``planTeacherAttempts`` that may skip or add attempts.
    """
    import copy
    from ..common.jsonio import iter_jsonl_rows, write_json_durable
    from ..common.custody import CustodyError
    from .privileged_context import case_facts
    if type(attempts) is not int or attempts < 1:
        raise ValueError('JOLT+ teacher attempts must be a positive integer')
    custody, refusal = _load_custody(custody)
    if custody is None:
        return {'entries': [], 'refusal': refusal}
    worlds = {case.get('id'): case for _, case in iter_jsonl_rows(Path(cases))}
    base = json.loads(Path(config).read_text())
    by_group = {}
    for row in student_rows:
        by_group.setdefault(row.get('group'), []).append(row)
    entries, out = [], Path(out)
    firsts, summaries = [], []
    for first in sorted((entry for entry in student_entries if entry.get('sample') == 0), key=lambda e: e['group']):
        members = by_group.get(first['group'], [])
        case = worlds.get(members[0]['task_id']) if members else None
        if case is None:
            continue
        try:
            _, facts = case_facts(case, custody=custody)
        except CustodyError:
            continue
        if not facts:
            continue
        qualities = [row['reward']['quality'] for row in members if row.get('reward') is not None]
        firsts.append((first, members, case))
        summaries.append({'group': first['group'], 'taskId': case['id'], 'family': case.get('family'),
                          'students': len(members), 'successes': sum(quality >= 1 for quality in qualities),
                          'meanQuality': sum(qualities) / len(qualities) if qualities else 0.,
                          'feedback': [text for text in map(attempt_feedback, members) if text][:MAX_FEEDBACK_ATTEMPTS],
                          'facts': len(facts)})
    plan, record = plan_teacher_attempts(summaries, default=attempts, budget=attempts * len(summaries),
                                         command=plan_command)
    for first, members, case in firsts:
        size = max(entry['sample'] for entry in student_entries if entry['group'] == first['group']) + 1
        for attempt in range(plan[first['group']]):
            sample = size + attempt
            try:
                receipt, facts = case_facts(case, custody=custody,
                                            attempt_feedback=group_feedback(case, members, sample))
            except CustodyError:
                break
            if not facts:
                break
            directory = out / f"row-{first['group']:03d}" / f'teacher-{attempt:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            controls = directory / 'chat-request-config.json'
            write_json_durable(controls, guidance_controls(base, role='teacher', custody_receipt=receipt,
                                                           facts=facts, selection=selection))
            entry = copy.deepcopy(first)
            entry.pop('rl_endpoint', None)
            entry.pop('chat_request_config', None)
            entry.update(key=f"s7-task-{first['group']:03d}-t{attempt:02d}", sample=sample,
                         seed=seed_base + 1000 * first['group'] + TEACHER_SEED_OFFSET + attempt,
                         jobs=str(directory / 'jobs'), log=str(directory / 'collector.log'),
                         output=str(directory / 'results.jsonl'), chat_request_config=str(controls),
                         jolt_role='teacher')
            entries.append(entry)
    return {'entries': entries, 'refusal': None, 'plan': record}
