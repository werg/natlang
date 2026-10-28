"""Whole-graph teacher scoring with privileged blocks (JOLT+ OPSD and J_T inputs).

The teacher role is the same policy with a declared privileged block inserted
into every invocation of the ACTUAL episode graph. The edited graph is the
student's compiled ``NativeReplay`` plan with each block spliced into its
invocation's own tokens: producer spans, import insertion boundaries and
cross-invocation import spans are rebased by the exact per-invocation shift,
so imported KV is the teacher-role producer's state at its consumer's shifted
logical positions (``compile_graph`` assigns positions; nothing is detached).

Interface (small and stable; used by the RL learner and program tuning):

- ``teacher_form(replay, blocks)`` / ``student_form(replay, blocks)``: exact
  graph edits adding or removing verified blocks, with remapped actions.
- ``action_distributions(model, replay, ...)``: the shared readout at every
  sampled action: fp32 raw logits (full) or top-k processed logits plus the
  processed log-normalizer, and the sampled tokens' log-probs.
- ``teacher_scores(model, replay, blocks, ...)``: stop-gradient teacher
  distributions aligned 1:1 with ``replay.actions`` (OPSD target).
- ``teacher_role_logps(model, teacher_replay, blocks)``: log-probs WITH
  gradients of the teacher's own attempts sampled under the blocks (J_T), and
  the student-form replay of the same attempts (for beta_T KL to the student).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..model.native_graph_plan import ImportSpan, Invocation, compile_graph


@dataclass(frozen=True)
class _Edit:
    at: int          # own-token boundary of the edit
    removed: int     # own tokens removed at ``at`` (student_form)
    inserted: tuple  # own tokens inserted at ``at`` (teacher_form)

    @property
    def delta(self):
        return len(self.inserted) - self.removed


def _rebase_plan(plan, edits):
    """Apply per-invocation token edits and rebase every dependent coordinate."""
    if plan.external_names and any(name in edits for name in plan.external_names):
        raise ValueError('privileged blocks apply to graph invocations, not external state')
    index = {node.name: number for number, node in enumerate(plan.invocations)}
    unknown = set(edits) - set(index)
    if unknown:
        raise ValueError('privileged block names no invocation of this graph: ' + ', '.join(sorted(unknown)))

    def local(name, start, stop):
        edit = edits.get(name)
        if edit is None or not edit.delta and not edit.removed:
            return start, stop
        if stop <= edit.at:
            return start, stop
        if start >= edit.at + edit.removed:
            return start + edit.delta, stop + edit.delta
        raise ValueError('a producer or import span crosses the privileged insertion boundary')

    def logical(name, start, stop):
        edit = edits.get(name)
        if edit is None or not edit.delta and not edit.removed:
            return start, stop
        boundary = plan.boundaries[index[name]][edit.at]
        if stop <= boundary:
            return start, stop
        if start >= boundary + edit.removed:
            return start + edit.delta, stop + edit.delta
        raise ValueError('a logical span crosses the privileged insertion boundary')

    nodes = []
    for node in plan.invocations:
        edit = edits.get(node.name)
        tokens = node.token_ids
        if edit is not None:
            if node.cache_prefix:
                raise ValueError('privileged blocks cannot enter a raw prefix cache invocation')
            if not 0 <= edit.at <= edit.at + edit.removed <= len(tokens):
                raise ValueError('privileged block lies outside its invocation')
            if any(edit.at <= item.at <= edit.at + edit.removed for item in node.imports):
                raise ValueError('privileged block cannot share or contain a native-state import boundary')
            tokens = tokens[:edit.at] + edit.inserted + tokens[edit.at + edit.removed:]
        imports = []
        for item in node.imports:
            at = item.at + edit.delta if edit is not None and item.at > edit.at else item.at
            if item.source in plan.external_names:
                start, stop = item.start, item.stop
            elif item.logical:
                start, stop = logical(item.source, item.start, item.stop)
            else:
                start, stop = local(item.source, item.start, item.stop)
            imports.append(replace(item, at=at, start=start, stop=stop))
        producing = tuple(local(node.name, a, b) for a, b in node.producing)
        logical_producing = tuple(logical(node.name, a, b) for a, b in node.logical_producing)
        nodes.append(Invocation(node.name, tuple(tokens), tuple(imports), producing, logical_producing,
                                node.cache_prefix))
    rebased = compile_graph(nodes, external_sizes=dict(zip(plan.external_names, plan.external_sizes)))

    def requery(action):
        name = action['turn']
        number = index[name]
        own = action['query'] - plan.offsets[number]
        edit = edits.get(name)
        if edit is not None:
            if own < edit.at + edit.removed:
                raise ValueError('a scored action precedes the privileged insertion boundary')
            own += edit.delta
        return {**action, 'query': rebased.offsets[number] + own, 'local_query': own}
    return rebased, requery


def _edited(replay, edits):
    if replay.semantic_dependencies:
        # Codec sources bind source-local spans of their own graph; their
        # rebase needs the codec executor's support (tracked open item).
        raise ValueError('privileged scoring of semantic-codec graphs is not supported yet')
    plan, requery = _rebase_plan(replay.plan, edits)
    written = tuple({**item, **dict(zip(('start', 'stop'), _shift_written(replay.plan, edits, item)))}
                    for item in replay.written_sources)
    return replace(replay, plan=plan, actions=tuple(map(requery, replay.actions)),
                   input_actions=tuple(map(requery, replay.input_actions)), written_sources=written)


def _shift_written(plan, edits, item):
    edit = edits.get(item['invocation'])
    if edit is None or item['stop'] <= edit.at:
        return item['start'], item['stop']
    if item['start'] < edit.at + edit.removed:
        raise ValueError('a written state span crosses the privileged insertion boundary')
    return item['start'] + edit.delta, item['stop'] + edit.delta


def teacher_form(replay, blocks):
    """Insert each nonempty block at its declared own-token boundary."""
    return _edited(replay, {name: _Edit(block.insert_at, 0, tuple(block.token_ids))
                            for name, block in blocks.items() if block.token_ids})


def student_form(replay, blocks):
    """Remove verified blocks from a teacher-role replay (exact token check)."""
    edits = {}
    for name, block in blocks.items():
        if not block.token_ids:
            continue
        node = next((node for node in replay.plan.invocations if node.name == name), None)
        if node is None:
            raise ValueError('privileged block names no invocation of this graph')
        span = node.token_ids[block.insert_at:block.insert_at+len(block.token_ids)]
        if tuple(span) != tuple(block.token_ids):
            raise ValueError('teacher-role invocation does not carry its declared privileged block')
        edits[name] = _Edit(block.insert_at, len(block.token_ids), ())
    return _edited(replay, edits)


@dataclass
class ActionDistributions:
    """Per-action distributions aligned 1:1 with ``actions``.

    ``logits`` (full mode): raw fp32 logits [N, V] before temperature and mask.
    ``topk_ids``/``topk_logits`` (top-k mode): processed logits ``z/T`` (mask
    applied) of the k most probable tokens. ``log_normalizer``: logsumexp of
    the processed logits over the full vocabulary, so a processed log-prob is
    ``processed_logit - log_normalizer``. ``logps``: processed log-prob of the
    sampled token. ``temperature`` per action. Stable reverse KL takes logit
    differences from ``logits`` (or top-k logits and normalizers).
    """
    actions: tuple
    token_ids: torch.Tensor
    logps: torch.Tensor
    log_normalizer: torch.Tensor
    temperature: torch.Tensor
    logits: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None
    topk_logits: torch.Tensor | None = None
    receipts: dict | None = None


def processed_logits(raw, action):
    """Temperature and exact allowed/banned mask of one sampled action."""
    from ..common.sampling import token_mask
    values = raw.float() / action['temperature']
    allowed, banned = token_mask(values.shape[-1], allowed=action.get('allowed_token_ids'),
                                 banned=action.get('banned_token_ids'))
    if allowed is not None:
        mask = torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device)
        mask[list(allowed)] = True
        values = values.masked_fill(~mask, float('-inf'))
    elif banned:
        values = values.index_fill(-1, torch.tensor(list(banned), device=values.device), float('-inf'))
    return values


def _execute(model, replay):
    from ..model.native_adapters import require_model_execution_actor
    if replay.invocation_actors is None:
        require_model_execution_actor(model, replay.identity, replay.adapter_binding)
    else:
        from ..model.native_actor_graph import invocation_actor_bank
        bank = invocation_actor_bank(model.port)
        if bank is None or bank.receipt() != replay.invocation_actors:
            raise ValueError('native heterogeneous teacher scoring needs its exact per-invocation actor scope')
    return model(replay.plan, external_blocks=replay.external_blocks,
                 external_identity={name: {key: getattr(block, key) for key in replay.identity}
                                    for name, block in replay.external_blocks.items()}, retain_states=False)


def action_distributions(model, replay, *, topk=None, chunk_tokens=128, verify_reference=False, atol=2e-5):
    """Shared readout at every sampled action of ``replay`` (gradients follow the caller's grad mode)."""
    if topk is not None and (type(topk) is not int or topk < 1):
        raise ValueError('top-k must be a positive integer or None for the full vocabulary')
    if not replay.actions:
        raise ValueError('privileged scoring needs sampled actions')
    hidden = _execute(model, replay)['h_final'][0]
    actions = replay.actions
    if verify_reference:
        # Explicit invocation-major cached execution (shared_state.forward_reference).
        with torch.no_grad():
            reference = model(replay.plan, reference=True, external_blocks=replay.external_blocks,
                external_identity={name: {key: getattr(block, key) for key in replay.identity}
                                   for name, block in replay.external_blocks.items()})['h_final'][0]
        rows = torch.tensor([action['query'] for action in actions], device=hidden.device)
        if not torch.allclose(hidden.detach()[rows].float(), reference[rows].float(), atol=atol, rtol=1e-4):
            raise ValueError('one-pass teacher graph differs from explicit cached execution')
    logps, normalizers, full, top_ids, top_values = [], [], [], [], []
    for start in range(0, len(actions), chunk_tokens):
        chunk = actions[start:start+chunk_tokens]
        rows = torch.tensor([action['query'] for action in chunk], device=hidden.device)
        raw = model.logits(hidden[rows]).float()
        for row, action in zip(raw, chunk):
            values = processed_logits(row, action)
            normalizer = torch.logsumexp(values, -1)
            normalizers.append(normalizer)
            logps.append(values[action['token_id']] - normalizer)
            if topk is None:
                full.append(row)
            else:
                best = values.topk(min(topk, values.shape[-1]))
                top_ids.append(best.indices)
                top_values.append(best.values)
    device = hidden.device
    return ActionDistributions(
        actions=actions, token_ids=torch.tensor([action['token_id'] for action in actions], device=device),
        logps=torch.stack(logps), log_normalizer=torch.stack(normalizers),
        temperature=torch.tensor([action['temperature'] for action in actions], device=device),
        logits=torch.stack(full) if topk is None else None,
        topk_ids=torch.stack(top_ids) if topk is not None else None,
        topk_logits=torch.stack(top_values) if topk is not None else None)


def _aligned(student, teacher):
    keys = ('turn', 'sample', 'token_id', 'temperature', 'allowed_token_ids', 'banned_token_ids', 'behavior_logp')
    if len(student) != len(teacher) or any(
            {key: a.get(key) for key in keys} != {key: b.get(key) for key in keys} for a, b in zip(student, teacher)):
        raise ValueError('teacher actions are not aligned with the student sampled tokens')


def teacher_scores(model, replay, blocks, *, topk=None, chunk_tokens=128, verify_reference=False):
    """Stop-gradient teacher distributions at every student-sampled token.

    ``replay``: the student's compiled whole-episode ``NativeReplay``.
    ``blocks``: {invocation name: PrivilegedBlock} (empty blocks insert nothing).
    Returns ``ActionDistributions`` in ``replay.actions`` order.
    """
    teacher = teacher_form(replay, blocks)
    _aligned(replay.actions, teacher.actions)
    with torch.no_grad():
        result = action_distributions(model, teacher, topk=topk, chunk_tokens=chunk_tokens,
                                      verify_reference=verify_reference)
    result.actions = replay.actions
    result.receipts = {name: block.receipt for name, block in sorted(blocks.items())}
    return result


def teacher_role_logps(model, teacher_replay, blocks, *, chunk_tokens=128):
    """J_T inputs: the teacher's own attempts (sampled WITH the blocks).

    Returns ``(logps, behavior_logps, student_replay)``: log-probs with
    gradients through the whole teacher-role graph, the sampler's recorded
    log-probs, and the student-form replay of the same attempt (blocks
    removed after exact token verification) for the beta_T KL to the student.
    """
    from .native_replay import replay_logps
    student = student_form(teacher_replay, blocks)
    logps, behavior = replay_logps(model, teacher_replay, chunk_tokens=chunk_tokens)
    return logps, behavior, student


def graph_invocations(replay, turns, *, root_calls):
    """Student-form invocation descriptors for ``privileged_context.privileged_blocks``.

    ``turns``: the learner's ``producer_turns``/``whole_episode_turns`` items
    (``graph_turn`` names ``<call>#turnN``); ``root_calls``: the episode's root
    invocation ids. Text is the turn's actual message contents.
    """
    by_name = {turn.get('graph_turn'): turn for turn in turns}
    # A sampled decision's option nodes (``<turn>:option:<i>``) belong to its decision turn.
    owners = {option: name for name, options in (replay.turn_nodes or {}).items() for option in options}
    result = []
    for node in replay.plan.invocations:
        owner = owners.get(node.name, node.name)
        turn = by_name.get(owner)
        if turn is None:
            raise ValueError('privileged scoring needs every graph invocation\'s captured turn')
        text = '\n'.join(message['content'] for message in turn.get('messages') or []
                         if isinstance(message.get('content'), str))
        call = owner.rsplit('#turn', 1)[0]
        result.append({'name': node.name, 'role': 'root' if call in root_calls else 'child', 'text': text,
                       'token_ids': node.token_ids})
    return result
