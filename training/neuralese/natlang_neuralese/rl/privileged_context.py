"""Privileged training-world guidance for JOLT+ self-teaching and memetic learning (shared mechanism).

One mechanism serves the whole-graph RL learner (JOLT+ teacher attempts and OPSD twins), the memetic learner and
the teacher-free program-tuning supply. Decomposition: ``PRIVILEGED_CONTEXT_DECOMPOSITION.md``.

Policy is natural language and runs in the natlang runtime that executes the episode
(``applications/guidance-writer``): ``selectGuidanceFacts`` chooses and orders the visible facts and
``writeGuidance`` (executed by the policy model itself) writes them as a ``Neuralese<string>`` producer span inside
the same episode graph. This module is the crisp mechanism around it:

1. ``ProtectedWorlds`` (custody): the exact identities of every registered evaluation-only packet. Privileged facts
   are derived only from ``split == 'train'`` worlds that share no id, source id, source group, split group, pair
   group or world fingerprint with them.
2. ``case_facts`` (extraction): authored reference solution, reference child values, oracle checks expressed as
   facts, verified teacher trajectories of the same world and earlier-attempt runtime feedback.
3. ``applicable`` (visibility, exact data rule): a child sees only its own matched reference value; the root sees
   world-level facts. The runtime applies the same rule before selection; the learner re-verifies it here.
4. ``PrivilegedRendering`` (hash-bound placement): the consumer import is one declared ChatML turn after the
   leading system turn (else after BOS): ``<header>`` then the guidance state between consume markers
   (``mode: import``), or the selected facts' texts packed whole into a token budget (``mode: text``, the recorded
   fallback when the writer fails). ``guidance_plan`` renders a runtime request into engine prompt pieces and its
   receipt; ``verify_captured_block`` re-verifies a captured block against custody, facts, visibility, rendering
   and the in-graph writer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..common.hashing import canonical_json_sha256_hex, sha256_hex

BLOCK_SCHEMA = 'natlang.jolt-privileged-block/2'
REQUEST_SCHEMA = 'natlang.guidance-request/1'
RENDERING_SCHEMA = 'natlang.jolt-privileged-rendering/2'
CUSTODY_SCHEMA = 'natlang.jolt-privileged-custody/1'
WRITER_FUNCTION = 'writeGuidance'
SELECTION_FUNCTION = 'selectGuidanceFacts'
MODES = ('import', 'write-only', 'text')
PROTECTED_ADMISSION = 'evaluation-only-never-training'
FACT_KINDS = ('expected_result', 'expected_stop', 'expected_files', 'expected_effects', 'reference_failure',
              'reference_solution', 'child_value', 'minimum_sequence', 'decisive', 'teacher_trajectory',
              'attempt_feedback', 'fit_hindsight')
REFERENCE_KINDS = ('reference_solution', 'teacher_trajectory', 'child_value', 'expected_result')
VERBATIM_MIN_CHARS = 24


class CustodyError(ValueError):
    """Privileged context would come from a development, confirmation or held world."""


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------- custody

def world_fingerprint(case):
    """Content identity of a case's world, independent of its ids/split labels."""
    semantics = case.get('semantics') or {}
    curriculum = case.get('curriculum') or {}
    world = {key: semantics.get(key) for key in ('root', 'files', 'folder_files', 'services', 'inputs')}
    world['evidence'] = curriculum.get('evidence')
    return canonical_json_sha256_hex(world)


def case_identities(case):
    curriculum = case.get('curriculum') or {}
    ids = {('id', case.get('id'))}
    ids.update(('id', value) for value in case.get('source_ids') or [])
    ids.update(('group', value) for value in case.get('source_groups') or [])
    ids.update(('group', value) for value in case.get('split_groups') or [])
    for key in ('split_group', 'pair_group'):
        if curriculum.get(key):
            ids.add(('group', curriculum[key]))
    ids.add(('world', world_fingerprint(case)))
    return {item for item in ids if isinstance(item[1], str) and item[1]}


@dataclass(frozen=True)
class ProtectedWorlds:
    """Exact protected identities of every registered evaluation-only packet."""
    identities: frozenset
    packets: tuple

    @property
    def digest(self):
        return canonical_json_sha256_hex({'packets': list(self.packets),
                                          'identities': sorted(map(list, self.identities))})

    @classmethod
    def from_cases(cls, packets):
        """``packets``: {packet_id: (cases_sha256, rows)}."""
        identities, bound = set(), []
        for packet_id in sorted(packets):
            digest, rows = packets[packet_id]
            for row in rows:
                identities |= case_identities(row)
            bound.append({'id': packet_id, 'cases_sha256': digest, 'rows': len(rows)})
        return cls(frozenset(identities), tuple(json.dumps(item, sort_keys=True) for item in bound))

    @classmethod
    def from_registry(cls, registry_path, root):
        """Fail closed: every evaluation-only packet with cases must be readable here."""
        from ..common.hashing import sha256_file_hex
        registry = json.loads(Path(registry_path).read_text())
        entries = registry.get('corpora', registry) if isinstance(registry, dict) else registry
        packets = {}
        for entry in entries:
            if not isinstance(entry, dict) or entry.get('admission') != PROTECTED_ADMISSION:
                continue
            include = entry.get('include') or []
            if 'cases.jsonl' not in include:
                continue
            path = Path(root) / entry['path'] / 'cases.jsonl'
            if not path.is_file():
                raise CustodyError(f'protected packet {entry["id"]} is not readable; privileged custody fails closed')
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            packets[entry['id']] = (sha256_file_hex(path), rows)
        if not packets:
            raise CustodyError('no protected evaluation packets were registered; privileged custody fails closed')
        return cls.from_cases(packets)

    def check(self, case):
        """Return the custody receipt of a training world, or raise CustodyError."""
        if not isinstance(case, dict) or not isinstance(case.get('id'), str):
            raise CustodyError('privileged context needs an identified case')
        if case.get('split') != 'train':
            raise CustodyError(f'privileged context refuses split {case.get("split")!r}: only train worlds teach')
        overlap = case_identities(case) & self.identities
        if overlap:
            kinds = sorted({kind for kind, _ in overlap})
            raise CustodyError(f'privileged context refuses a world related to a protected packet ({", ".join(kinds)})')
        return {'schema': CUSTODY_SCHEMA, 'case_id': case['id'], 'split': 'train',
                'world_sha256': world_fingerprint(case), 'protected_sha256': self.digest}


# ---------------------------------------------------------------- facts

@dataclass(frozen=True)
class Fact:
    id: str
    kind: str
    text: str
    invocation: str | None = None   # attempt feedback bound to one invocation
    match: tuple = ()               # child references: substrings of the child prompt

    @property
    def digest(self):
        return canonical_json_sha256_hex({'id': self.id, 'kind': self.kind, 'text': self.text,
                                          'invocation': self.invocation, 'match': list(self.match)})


def _action_text(actions):
    lines = []
    for step, action in enumerate(actions, 1):
        if not (isinstance(action, list) and len(action) == 2 and isinstance(action[0], str)):
            raise ValueError('reference actions must be [tool, arguments] pairs')
        tool, arguments = action
        if tool == 'eval' and isinstance(arguments, dict) and set(arguments) == {'code'}:
            lines.append(f'{step}. eval:\n{arguments["code"]}')
        else:
            lines.append(f'{step}. {tool} {_text(arguments)}')
    return '\n'.join(lines)


def case_facts(case, *, custody, teacher_trajectories=(), attempt_feedback=()):
    """All privileged facts of one custody-checked training world.

    ``teacher_trajectories``: admitted verified trajectories of the same world,
    each ``{case_id, split, verified: True, actor, actions: [[tool, args], ...]}``.
    ``attempt_feedback``: runtime error/repair feedback from this episode's
    earlier student attempts, ``{case_id, attempt, feedback, invocation?}``.
    """
    receipt = custody.check(case)
    curriculum = case.get('curriculum') or {}
    reference = curriculum.get('reference') or {}
    semantics = case.get('semantics') or {}
    facts = []
    def add(kind, text, **extra):
        if isinstance(text, str) and text.strip():
            facts.append(Fact(f'{kind}:{sum(f.kind == kind for f in facts)}', kind, text.strip(), **extra))
    if 'expected' in semantics:
        add('expected_result', f'The verified result of this program is {_text(semantics["expected"])}.')
    if semantics.get('operation'):
        add('expected_stop', f'The correct execution stops with status {_text(semantics["operation"])}.')
    for name, content in sorted((semantics.get('expected_files') or {}).items()):
        add('expected_files', f'After a correct run, file {name} contains:\n{content}')
    for service, effect in sorted((semantics.get('effects') or {}).items()):
        add('expected_effects', f'Required effect on {service}: {_text(effect)}')
    for failure in reference.get('failures') or []:
        add('reference_failure', f'A verified wrong attempt to avoid: {_text(failure)}')
    if reference.get('root'):
        add('reference_solution', 'A verified reference solution:\n' + _action_text(reference['root']))
    for child in reference.get('children') or []:
        match = tuple(child.get('match') or ())
        if not match or any(not isinstance(item, str) for item in match):
            raise ValueError('reference child values need their exact prompt match strings')
        add('child_value', f'The verified answer of this call is {_text(child.get("value"))}.', match=match)
    if curriculum.get('minimum_sequence'):
        add('minimum_sequence', 'Steps of a correct solution:\n' + '\n'.join(
            f'- {_text(step)}' for step in curriculum['minimum_sequence']))
    for item in curriculum.get('decisive') or []:
        add('decisive', f'Decisive evidence: {_text(item)}')
    for trajectory in teacher_trajectories:
        if (trajectory.get('case_id') != case['id'] or trajectory.get('split') != 'train'
                or trajectory.get('verified') is not True or not trajectory.get('actor')):
            raise CustodyError('teacher trajectories must be verified training trajectories of this exact world')
        add('teacher_trajectory', 'A verified successful trajectory:\n' + _action_text(trajectory['actions']))
    for item in attempt_feedback:
        if item.get('case_id') != case['id'] or type(item.get('attempt')) is not int:
            raise CustodyError('attempt feedback must come from this world\'s own earlier attempts')
        add('attempt_feedback', f'Earlier attempt {item["attempt"]} received: {item["feedback"]}',
            invocation=item.get('invocation'))
    return receipt, tuple(facts)


# ---------------------------------------------------------------- visibility

def applicable(fact, invocation):
    """Facts a given invocation may see at all (exact data rule, not preference).

    The runtime filters by this rule before ``selectGuidanceFacts`` (``applications/guidance-writer/visibility.ts``);
    the engine and the learner re-verify it here. ``fact`` is a ``Fact``; ``invocation`` is ``{name, role, text}``
    with role ``root``, ``child`` or ``binding`` (a learned-optimizer instruction/skill state, which sees what a
    root sees).
    """
    if fact.kind == 'child_value':
        return invocation['role'] == 'child' and all(item in invocation['text'] for item in fact.match)
    if fact.kind in ('attempt_feedback', 'fit_hindsight'):
        return fact.invocation is None or fact.invocation == invocation['name']
    return invocation['role'] in ('root', 'binding')


def fact_from_request(item):
    if (not isinstance(item, dict) or not isinstance(item.get('id'), str) or item.get('kind') not in FACT_KINDS
            or not isinstance(item.get('text'), str) or not item['text'].strip()):
        raise ValueError('guidance facts are {id, kind, text, invocation?, match?}')
    match = item.get('match') or ()
    if any(not isinstance(value, str) for value in match):
        raise ValueError('guidance fact match strings must be strings')
    invocation = item.get('invocation')
    if invocation is not None and not isinstance(invocation, str):
        raise ValueError('guidance fact invocation must be a string')
    return Fact(item['id'], item['kind'], item['text'], invocation, tuple(match))


def fact_request(fact):
    return {'id': fact.id, 'kind': fact.kind, 'text': fact.text, 'invocation': fact.invocation,
            'match': list(fact.match)}


# ---------------------------------------------------------------- rendering

@dataclass(frozen=True)
class PrivilegedRendering:
    """Declared rendering of the guidance turn for one backbone template.

    ``encode`` is the template's content tokenizer (``Renderer.text``); it is bound by ``tokenizer_fingerprint``,
    not hashed as code. ``consume_open``/``consume_close`` are the backbone's consumer markers.
    """
    template: str
    bos: int | None
    im_start: int
    im_end: int
    consume_open: int
    consume_close: int
    tokenizer_fingerprint: str
    encode: Callable = field(compare=False, repr=False)
    role: str = 'system'
    header: str = 'Guidance for this call (training only):'
    text_header: str = 'Verified reference for this call (training only):'
    placement: str = 'after_leading_system_turn'
    budget_tokens: int = 512

    @classmethod
    def from_renderer(cls, renderer, *, template, tokenizer_fingerprint, consume_open, consume_close, **options):
        special = renderer.special
        return cls(template, special.bos, special.im_start, special.im_end, consume_open, consume_close,
                   tokenizer_fingerprint, renderer.text, **options)

    def declaration(self):
        return {'schema': RENDERING_SCHEMA, 'template': self.template, 'bos': self.bos, 'im_start': self.im_start,
                'im_end': self.im_end, 'consume_open': self.consume_open, 'consume_close': self.consume_close,
                'tokenizer_fingerprint': self.tokenizer_fingerprint, 'role': self.role, 'header': self.header,
                'text_header': self.text_header, 'placement': self.placement, 'budget_tokens': self.budget_tokens}

    @property
    def digest(self):
        return canonical_json_sha256_hex(self.declaration())

    def turn(self, texts):
        """Text form: the header and whole fact texts in one turn."""
        body = '\n\n'.join([self.text_header, *texts])
        return [self.im_start, *self.encode(f'{self.role}\n'), *self.encode(body), self.im_end, *self.encode('\n')]

    def state_parts(self):
        """Import form, as engine pieces: tokens before and after the consumed state (markers excluded)."""
        head = [self.im_start, *self.encode(f'{self.role}\n'), *self.encode(self.header + '\n')]
        return head, [self.im_end, *self.encode('\n')]

    def state_turn(self):
        """Import form as own tokens (consume markers included) and the own-token offset of the import."""
        head, tail = self.state_parts()
        return [*head, self.consume_open, self.consume_close, *tail], len(head) + 1

    def insertion_point(self, token_ids):
        """Own-token index after the leading system turn, else after BOS."""
        tokens = list(token_ids)
        start = 1 if self.bos is not None and tokens[:1] == [self.bos] else 0
        opening = [self.im_start, *self.encode('system\n')]
        if tokens[start:start+len(opening)] != opening:
            return start, 'after_bos' if start else 'start'
        closing = [self.im_end, *self.encode('\n')]
        for index in range(start+len(opening), len(tokens)):
            if tokens[index] == self.im_end:
                if tokens[index:index+len(closing)] != closing:
                    break
                return index + len(closing), 'after_leading_system_turn'
        raise ValueError('leading system turn is not closed by its declared template close')


@dataclass(frozen=True)
class PrivilegedBlock:
    """One invocation's guidance as own tokens at ``insert_at``; ``import_offset`` locates the consumed state."""
    invocation: str
    insert_at: int
    token_ids: tuple
    text: str
    receipt: dict
    import_offset: int | None = None
    state_id: str | None = None
    source_occurrence: str | None = None

    @property
    def digest(self):
        return canonical_json_sha256_hex(self.receipt)


def _request_parts(request):
    if not isinstance(request, dict) or request.get('schema') != REQUEST_SCHEMA:
        raise ValueError(f'guidance requests use schema {REQUEST_SCHEMA}')
    allowed = {'schema', 'invocation', 'mode', 'custody', 'facts', 'selection', 'writer', 'state', 'fallback'}
    if set(request) - allowed:
        raise ValueError('guidance request has unsupported fields: ' + ', '.join(sorted(set(request) - allowed)))
    invocation, mode = request.get('invocation'), request.get('mode')
    if (not isinstance(invocation, dict) or set(invocation) != {'name', 'role'}
            or not isinstance(invocation['name'], str) or invocation['role'] not in ('root', 'child')):
        raise ValueError('guidance requests name the consuming invocation {name, role: root|child}')
    if mode not in MODES:
        raise ValueError('guidance mode is import, write-only or text')
    custody = request.get('custody')
    if not isinstance(custody, dict) or custody.get('schema') != CUSTODY_SCHEMA or custody.get('split') != 'train':
        raise CustodyError('guidance needs the custody receipt of a training world')
    selection = request.get('selection')
    if (not isinstance(selection, dict) or selection.get('implementation') not in ('nl', 'crisp')
            or not isinstance(selection.get('sha256'), str)):
        raise ValueError('guidance requests record their selection policy {implementation: nl|crisp, sha256}')
    facts = tuple(fact_from_request(item) for item in request.get('facts') or ())
    if len({fact.id for fact in facts}) != len(facts):
        raise ValueError('guidance facts repeat an id')
    state = request.get('state')
    if mode == 'text':
        if state is not None or request.get('writer') is not None:
            raise ValueError('text guidance carries no writer state')
        if not isinstance(request.get('fallback'), str) or not request['fallback']:
            raise ValueError('text guidance is the writer fallback and records its reason')
    else:
        writer = request.get('writer')
        if (not isinstance(state, dict) or set(state) - {'id', 'source_occurrence'} or not isinstance(state.get('id'), str)
                or not isinstance(writer, dict) or writer.get('function') != WRITER_FUNCTION
                or not isinstance(writer.get('sha256'), str)):
            raise ValueError('import and write-only guidance carry the writer {function, sha256} and its state {id}')
        if request.get('fallback') is not None:
            raise ValueError('only text guidance records a writer fallback')
        if not facts:
            raise ValueError('the guidance writer needs at least one fact')
    return invocation, mode, custody, selection, facts, state


def guidance_plan(rendering, request, *, token_ids, text):
    """Engine side: the prompt pieces and receipt of one model turn's guidance request.

    ``token_ids``: the turn's own prompt tokens (no guidance); ``text``: its messages' text (visibility). Pieces are
    ``('privileged', tokens)`` and ``('privileged_state', {id, source_occurrence?})``, inserted at ``insert_at``.
    Returns ``(insert_at, pieces, receipt)``; ``pieces`` is empty for write-only or when no fact fits the budget.
    """
    invocation, mode, custody, selection, facts, state = _request_parts(request)
    visible = {**invocation, 'text': text}
    hidden = [fact.id for fact in facts if not applicable(fact, visible)]
    if hidden:
        raise CustodyError('guidance facts are not visible to this invocation: ' + ', '.join(hidden))
    insert_at, placement = rendering.insertion_point(token_ids)
    included, omitted, pieces = list(facts), [], []
    if mode == 'text':
        included = []
        for fact in facts:
            if len(rendering.turn([item.text for item in [*included, fact]])) <= rendering.budget_tokens:
                included.append(fact)
            else:
                omitted.append({'id': fact.id, 'reason': 'token_budget'})
        tokens = rendering.turn([fact.text for fact in included]) if included else []
        if tokens:
            pieces = [('privileged', tokens)]
    else:
        tokens, _ = rendering.state_turn()
        if mode == 'import':
            head, tail = rendering.state_parts()
            pieces = [('privileged', head), ('privileged_state', dict(state)), ('privileged', tail)]
    receipt = {'schema': BLOCK_SCHEMA, 'invocation': invocation['name'], 'role': invocation['role'], 'mode': mode,
               'rendering_sha256': rendering.digest, 'selection': selection, 'custody': custody,
               'facts': [{'id': fact.id, 'kind': fact.kind, 'sha256': fact.digest} for fact in included],
               'omitted': omitted, 'insert_at': insert_at, 'placement': placement,
               'token_count': len(tokens) if pieces or mode == 'write-only' else 0,
               'token_ids_sha256': canonical_json_sha256_hex(list(tokens) if pieces or mode == 'write-only' else [])}
    if mode == 'text':
        receipt['fallback'] = request['fallback']
    else:
        receipt['writer'] = request['writer']
        receipt['state'] = dict(state)
    return insert_at, pieces, receipt


def captured_block(turn):
    """The guidance block a native-engine actor recorded for one captured turn, or None."""
    capture = turn.get('sampling') or {}
    receipt = capture.get('privileged')
    if receipt is None:
        if any(segment.get('privileged') for segment in capture.get('prompt') or []):
            raise ValueError('privileged prompt tokens without their guidance receipt')
        return None
    if not isinstance(receipt, dict) or receipt.get('schema') != BLOCK_SCHEMA:
        raise ValueError(f'captured guidance needs its {BLOCK_SCHEMA} receipt')
    own, tokens, start, import_offset, state = 0, [], None, None, None
    for segment in capture.get('prompt') or []:
        flagged = segment.get('privileged') is True
        if segment.get('kind') == 'native_state':
            if flagged:
                if import_offset is not None:
                    raise ValueError('a guidance block imports exactly one state')
                import_offset, state = len(tokens), segment
            continue
        if flagged:
            if start is None:
                start = own
            elif start + len(tokens) != own:
                raise ValueError('captured guidance tokens are not contiguous')
            tokens.extend(segment['token_ids'])
        own += len(segment['token_ids'])
    mode = receipt.get('mode')
    state_receipt = receipt.get('state') or {}
    if mode == 'write-only':
        if tokens or state is not None:
            raise ValueError('write-only guidance must not enter the sampled prompt')
        return PrivilegedBlock(turn['graph_turn'], receipt['insert_at'], (), '', receipt,
                               state_id=state_receipt.get('id'), source_occurrence=state_receipt.get('source_occurrence'))
    if start is None:
        if receipt.get('token_count'):
            raise ValueError('captured guidance receipt names tokens the prompt does not carry')
        return PrivilegedBlock(turn['graph_turn'], receipt['insert_at'], (), '', receipt)
    if start != receipt.get('insert_at'):
        raise ValueError('captured guidance sits away from its declared insertion point')
    if (mode == 'import') != (state is not None) or (state is not None and state['id'] != state_receipt.get('id')):
        raise ValueError('captured guidance import differs from its receipt')
    return PrivilegedBlock(turn['graph_turn'], start, tuple(tokens), '', receipt, import_offset,
                           state_receipt.get('id'), state_receipt.get('source_occurrence'))


def verify_captured_block(block, rendering, *, custody_receipt, facts, invocation):
    """Learner side: re-verify a captured block (custody, facts, visibility, rendering). Returns the facts it used.

    ``facts``: every fact of the custody-checked world (``case_facts``); ``invocation``: ``{name, role, text}`` of
    the consuming graph invocation in student form.
    """
    receipt = block.receipt
    if receipt.get('custody') != custody_receipt:
        raise CustodyError('guidance custody differs from the world\'s current custody receipt')
    if receipt.get('rendering_sha256') != rendering.digest or receipt.get('role') != invocation['role']:
        raise ValueError('guidance rendering or invocation role differs from the learner\'s')
    by_id = {fact.id: fact for fact in facts}
    used = []
    for item in receipt.get('facts') or []:
        fact = by_id.get(item.get('id'))
        if fact is None or fact.digest != item.get('sha256') or fact.kind != item.get('kind'):
            raise CustodyError('guidance names a fact this training world does not have')
        if not applicable(fact, invocation):
            raise CustodyError('guidance used a fact not visible to its invocation')
        used.append(fact)
    mode = receipt.get('mode')
    if mode == 'text':
        expected = rendering.turn([fact.text for fact in used]) if used else []
        offset = None
    else:
        if not used:
            raise ValueError('writer guidance without facts')
        expected, offset = rendering.state_turn()
        if mode == 'write-only':
            if block.token_ids:
                raise ValueError('write-only guidance carries tokens')
            return used
    if list(block.token_ids) != list(expected) or block.import_offset != offset:
        raise ValueError('captured guidance tokens differ from the declared rendering')
    if receipt.get('token_ids_sha256') != canonical_json_sha256_hex(list(block.token_ids)):
        raise ValueError('captured guidance tokens differ from their receipt')
    return used


def writer_saw(facts, writer_text):
    """Exact check that the in-graph writer's own prompt carried every fact it was credited with."""
    missing = [fact.id for fact in facts if fact.text not in writer_text]
    if missing:
        raise ValueError('the guidance writer\'s prompt lacks facts its receipt names: ' + ', '.join(missing))


def reproduces_reference(hint_text, facts):
    """Telemetry, not a gate: does a hint copy a reference answer or solution verbatim?"""
    for fact in facts:
        if fact.kind not in REFERENCE_KINDS:
            continue
        body = fact.text.split('\n', 1)[1] if '\n' in fact.text else fact.text
        if fact.kind in ('child_value', 'expected_result'):
            body = body.split(' is ', 1)[-1].rstrip('.')
            if body and body in hint_text and len(body) >= 3:
                return True
            continue
        lines = [line.strip() for line in body.splitlines() if len(line.strip()) >= VERBATIM_MIN_CHARS]
        if any(line in hint_text for line in lines):
            return True
    return False
