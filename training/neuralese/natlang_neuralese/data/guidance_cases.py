"""Guidance-writer cases for the ``guidance`` learned operator (fallow register C6; MEMETIC_META_LEARNING_REVIEW_20261103.md §6).

A ``natlang.guidance-case/1`` record is one real writeGuidance call and the consumer calls that imported its state:

    {schema, id, split: train|test, split_groups: [...], family, lineage: {...}, license?,
     writer: {messages: [request messages of the writer call's first turn], tools: [...],
              facts: [{id, kind, text}, ...]  (the selected verified facts the writer was given, in order)},
     guidance: {text, provenance: 'checked', origin: {...}} | null,
     consumers: [{invocation, context: [student-form request messages of the consumer's first turn], tools,
                  target: {kind: 'message', value: <the consumer's checked first reply>}}],
     outcome: {label: 'checked', checked: {...evidence that the consumers then succeeded}}}

``guidance`` is the text writer's own guidance (b069f6061: writeGuidance returns text, the host writes it as the native
state) from an episode whose consumer then succeeded; without one the renderer uses the deterministic mechanical seed
of the facts (data/operator_seeds.guidance_seed), recording that provenance. data/operator_graph.py
``render_guidance`` turns a case into the operator graph.

``cases_from_rollout`` converts one captured JOLT/memetic rollout whose writer answered in text. It needs the world's
verified facts (rl/privileged_context.case_facts, the same custody the learner re-verifies) and the episode's checked
outcome; it never infers success. Memetic score.json guidance (``memetic.candidate.guidance``) is program-scope text a
supervisor composed, not a writeGuidance call: it has no writer site and is distilled by the memetic stage
(memetic_program.guidance_training_graph), not converted here.
"""
from __future__ import annotations

import json

SCHEMA = 'natlang.guidance-case/1'
WRITER_FUNCTION = 'writeGuidance'


def check_case(case: dict) -> dict:
    """Structural validation of one guidance case (the renderer's input contract)."""
    if not isinstance(case, dict) or case.get('schema') != SCHEMA:
        raise ValueError(f'a guidance case has schema {SCHEMA}')
    if not isinstance(case.get('id'), str) or not case['id']:
        raise ValueError('a guidance case has an id')
    if case.get('split') not in ('train', 'test', 'validation'):
        raise ValueError('split is train, test or validation')
    if not isinstance(case.get('split_groups'), list) or not case['split_groups'] \
            or any(not isinstance(group, str) or not group for group in case['split_groups']):
        raise ValueError('split_groups is a nonempty list of group names')
    writer = case.get('writer')
    if not isinstance(writer, dict) or not isinstance(writer.get('messages'), list) or not writer['messages']:
        raise ValueError('writer.messages are the recorded request messages of the writeGuidance call')
    facts = writer.get('facts')
    if not isinstance(facts, list) or not facts or any(
            not isinstance(fact, dict) or not isinstance(fact.get('id'), str) or not isinstance(fact.get('kind'), str)
            or not isinstance(fact.get('text'), str) or not fact['text'].strip() for fact in facts):
        raise ValueError('writer.facts are the nonempty selected facts {id, kind, text}')
    guidance = case.get('guidance')
    if guidance is not None and (not isinstance(guidance, dict) or not isinstance(guidance.get('text'), str)
                                 or not guidance['text'].strip() or guidance.get('provenance') not in ('checked', 'gold')):
        raise ValueError('guidance is null or {text, provenance: checked|gold, origin}')
    consumers = case.get('consumers')
    if not isinstance(consumers, list) or not consumers:
        raise ValueError('a guidance case has at least one consumer')
    for consumer in consumers:
        if (not isinstance(consumer, dict) or not isinstance(consumer.get('context'), list) or not consumer['context']
                or not isinstance(consumer.get('target'), dict) or consumer['target'].get('kind') != 'message'):
            raise ValueError('a consumer is {invocation, context, tools, target: {kind: message, value}}')
    outcome = case.get('outcome')
    if not isinstance(outcome, dict) or outcome.get('label') not in ('checked', 'gold') or not outcome.get('checked'):
        raise ValueError('outcome records the checked consumer success evidence')
    return case


def _content_text(messages):
    return '\n'.join(message['content'] for message in messages if isinstance(message.get('content'), str))


def writer_text(turn) -> str | None:
    """The guidance a text writer returned on its final turn: a plain reply, or a successful return_result value."""
    reply = turn.get('target') or {}
    calls = reply.get('tool_calls') or []
    if not calls:
        content = reply.get('content')
        return content.strip() if isinstance(content, str) and content.strip() else None
    for call in calls:
        function = call.get('function') or call
        if function.get('name') != 'return_result':
            continue
        arguments = function.get('arguments')
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict) and arguments.get('status') == 'success' \
                and isinstance(arguments.get('value'), str) and arguments['value'].strip():
            return arguments['value'].strip()
    return None


def cases_from_rollout(rollout: dict, facts, *, outcome: dict, split_groups, family=None, lineage=None):
    """(cases, omissions) of one captured rollout whose writeGuidance calls answered in text.

    ``facts``: the world's verified ``privileged_context.Fact`` objects; receipt facts must match their digests.
    ``outcome``: ``{success: bool, evidence: {...}}`` from the episode's own oracle. Only a successful episode yields
    cases (its guidance is ``checked`` and its consumer reply is a checked target); a failed one yields none. A writer is linked to its consumer when its recorded request carries
    every receipt fact text and the consumer's invocation name, and exactly one writer call does.
    """
    from ..rl.privileged_context import captured_block
    by_id = {fact.id: fact for fact in facts}
    turns = rollout.get('turns') or []
    calls, order = {}, []
    for turn in turns:
        call = turn.get('invocation_id')
        if call not in calls:
            calls[call] = []
            order.append(call)
        calls[call].append(turn)
    writers = {call: entries for call, entries in calls.items()
               if ((entries[0].get('sampling') or {}).get('guidance_policy') or {}).get('function') == WRITER_FUNCTION}
    cases, omissions = [], []
    if not outcome.get('success'):
        return cases, [{'rollout': rollout.get('id'), 'reason': 'episode-not-successful'}]
    for call in order:
        if call in writers:
            continue
        first = calls[call][0]
        block = captured_block(first)
        if block is None:
            continue
        receipt = block.receipt
        if receipt.get('mode') not in ('write-only', 'import'):
            omissions.append({'consumer': call, 'reason': f"no-writer-state: mode {receipt.get('mode')}"})
            continue
        used = []
        for item in receipt.get('facts') or []:
            fact = by_id.get(item.get('id'))
            if fact is None or fact.digest != item.get('sha256'):
                used = None
                break
            used.append(fact)
        if not used:
            omissions.append({'consumer': call, 'reason': 'receipt-facts-unverified'})
            continue
        linked = [writer for writer, entries in writers.items()
                  if all(fact.text in _content_text(entries[0].get('messages') or []) for fact in used)
                  and str(receipt.get('invocation')) in _content_text(entries[0].get('messages') or [])]
        if len(linked) != 1:
            omissions.append({'consumer': call, 'reason': f'writer-link: {len(linked)} candidate writer calls'})
            continue
        entries = writers[linked[0]]
        # A writer that returned no usable text still gives the real site; its seed is then the mechanical one.
        text = writer_text(entries[-1])
        actor = ((entries[0].get('sampling') or {}).get('policy') or {})
        case = {'schema': SCHEMA, 'id': f"guidance:{rollout.get('id') or rollout.get('task_id')}:{call}",
                'split': 'train', 'split_groups': sorted(set(split_groups)), 'family': family,
                'lineage': {**(lineage or {}), 'rollout': rollout.get('id'), 'task_id': rollout.get('task_id'),
                            'writer_call': linked[0], 'consumer_call': call, 'actor': actor},
                'writer': {'messages': entries[0]['messages'], 'tools': entries[0].get('tools') or [],
                           'facts': [{'id': fact.id, 'kind': fact.kind, 'text': fact.text} for fact in used]},
                'guidance': None if text is None else {
                    'text': text, 'provenance': 'checked',
                    'origin': {'writer_call': linked[0], 'policy': actor,
                               'writer_sha256': ((entries[0].get('sampling') or {}).get('guidance_policy') or {}).get('sha256')}},
                'consumers': [{'invocation': call, 'context': first['messages'], 'tools': first.get('tools') or [],
                               'target': {'kind': 'message', 'value': first['target']}}],
                'outcome': {'label': 'checked', 'checked': outcome.get('evidence') or {'success': True}}}
        cases.append(check_case(case))
    return cases, omissions
