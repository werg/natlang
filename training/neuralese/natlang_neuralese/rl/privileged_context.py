"""Privileged training-world guidance for JOLT+ self-teaching and memetic learning (shared mechanism).

One mechanism serves the whole-graph RL learner (JOLT+ teacher attempts and OPSD twins), the memetic learner and
the teacher-free program-tuning supply. Decomposition: ``PRIVILEGED_CONTEXT_DECOMPOSITION.md``.

Policy is natural language and runs in the natlang runtime that executes the episode
(``applications/guidance-writer``): ``selectGuidanceFacts`` chooses and orders the visible facts and
``writeGuidance`` (executed by the policy model itself) writes them as a ``Neuralese<string>`` producer span inside
the same episode graph. This module is the crisp mechanism around it:

1. Custody is the shared ``common.custody.ProtectedWorlds`` (every declared protected source, failing closed).
   Privileged facts are derived only from ``split == 'train'`` worlds related to no protected world.
2. ``case_facts`` (extraction): authored reference solution, reference child values, oracle checks expressed as
   facts, verified teacher trajectories of the same world and earlier-attempt runtime feedback.
3. ``applicable`` (visibility, exact data rule): a child sees only its own matched reference value; the root sees
   world-level facts. The runtime applies the same rule before selection; the learner re-verifies it here.
4. ``PrivilegedRendering`` (hash-bound placement): the consumer import is one declared ChatML turn after the
   leading system turn (else after BOS): ``<header>`` then the guidance state between consume markers
   (``mode: import``), or the selected facts' texts packed whole into a token budget (``mode: text``, the recorded
   fallback when the writer fails or is switched off). A student never receives guidance in its sampled prompt: it
   records the state its writer wrote (``mode: write-only``) or, without one, the exact text turn a teacher would
   have received (``mode: text-twin``, its token ids in the receipt); both are inserted only into the learner's
   stop-gradient OPSD twin. ``guidance_plan`` renders a runtime request into engine prompt pieces and its
   receipt; ``verify_captured_block`` re-verifies a captured block against custody, facts, visibility, rendering
   and the in-graph writer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..common.custody import CUSTODY_SCHEMA, CustodyError
from ..common.hashing import canonical_json_sha256_hex, sha256_hex

BLOCK_SCHEMA = 'natlang.jolt-privileged-block/2'
REQUEST_SCHEMA = 'natlang.guidance-request/1'
RENDERING_SCHEMA = 'natlang.jolt-privileged-rendering/2'
WRITER_FUNCTION = 'writeGuidance'
SELECTION_FUNCTION = 'selectGuidanceFacts'
MODES = ('import', 'write-only', 'text', 'text-twin')
TEXT_MODES = ('text', 'text-twin')
RECORDED_MODES = ('write-only', 'text-twin')  # recorded for the OPSD twin, never in the sampled prompt
FACT_KINDS = ('expected_result', 'expected_stop', 'expected_files', 'expected_effects', 'reference_failure',
              'reference_solution', 'child_value', 'minimum_sequence', 'decisive', 'teacher_trajectory',
              'attempt_feedback', 'fit_hindsight')
REFERENCE_KINDS = ('reference_solution', 'teacher_trajectory', 'child_value', 'expected_result')
VERBATIM_MIN_CHARS = 16


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------- child answer types
# A reference child's verified answer may guide a child call only when it fits that call's declared result type, so a
# policy's yes/no child is never pulled toward a catalog key. Mirrors applications/guidance-writer/visibility.ts
# (``answerFits``): a small reader of the TypeScript type text natlang prints in a call's opening ("You are inside this
# call: f(...): T" and "type X = ..." lines). Anything it does not read, an untyped (unknown/any) result included,
# admits no answer.

ANSWER_PREFIX = 'The verified answer of this call is '
_TOKEN = re.compile(r'\s*(?:("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')|(-?\d+(?:\.\d+)?)|([A-Za-z_$][\w$]*)|(\S))')


def _tokens(text):
    out, at = [], 0
    while at < len(text):
        if text[at:].strip() == '':
            break
        found = _TOKEN.match(text, at)
        if not found:
            return None
        string, number, name, punct = found.groups()
        out.append(('str', json.loads(string) if string.startswith('"') else string[1:-1]) if string is not None
                   else ('num', float(number)) if number is not None else ('name', name) if name is not None
                   else ('p', punct))
        at = found.end()
    return out


def parse_type(text):
    """The type expression ``text`` as a small tree, or None when it is not read."""
    tokens = _tokens(text)
    if tokens is None:
        return None
    position = [0]

    def peek(value=None):
        token = tokens[position[0]] if position[0] < len(tokens) else None
        return token if value is None or (token is not None and token[1] == value and token[0] == 'p') else None

    def take(value):
        if not peek(value):
            raise ValueError(value)
        position[0] += 1

    def union():
        if peek('|'):
            position[0] += 1
        options = [postfix()]
        while peek('|'):
            position[0] += 1
            options.append(postfix())
        return options[0] if len(options) == 1 else ('union', options)

    def postfix():
        node = primary()
        while peek('[') and position[0] + 1 < len(tokens) and tokens[position[0] + 1] == ('p', ']'):
            position[0] += 2
            node = ('array', node)
        return node

    def primary():
        token = peek()
        if token is None:
            raise ValueError('end')
        position[0] += 1
        kind, value = token
        if kind in ('str', 'num'):
            return ('literal', value)
        if kind == 'p' and value == '(':
            node = union(); take(')'); return node
        if kind == 'p' and value == '[':
            items = []
            while not peek(']'):
                items.append(union())
                if peek(','):
                    position[0] += 1
            take(']'); return ('tuple', items)
        if kind == 'p' and value == '{':
            fields = {}
            while not peek('}'):
                name = peek()
                if name is None or name[0] not in ('name', 'str'):
                    raise ValueError('field')
                position[0] += 1
                optional = bool(peek('?'))
                if optional:
                    position[0] += 1
                take(':')
                fields[name[1]] = (union(), optional)
                if peek(';') or peek(','):
                    position[0] += 1
            take('}'); return ('object', fields)
        if kind == 'name':
            if value in ('true', 'false'):
                return ('literal', value == 'true')
            if value in ('string', 'number', 'boolean', 'null', 'undefined', 'unknown', 'any'):
                return ('prim', value)
            if peek('<'):
                position[0] += 1
                arguments = [union()]
                while peek(','):
                    position[0] += 1
                    arguments.append(union())
                take('>')
                if value == 'Array' and len(arguments) == 1:
                    return ('array', arguments[0])
                if value == 'Record' and len(arguments) == 2:
                    return ('record', arguments[1])
                if value == 'Neuralese' and len(arguments) == 1:
                    return arguments[0]
                raise ValueError(value)
            return ('alias', value)
        raise ValueError(value)

    try:
        node = union()
    except (ValueError, IndexError):
        return None
    return node if position[0] == len(tokens) else None


def _fits(value, node, aliases, depth=0):
    if depth > 32:
        return False
    kind = node[0]
    if kind == 'prim':
        name = node[1]
        return (name in ('unknown', 'any') or (name == 'string' and isinstance(value, str))
                or (name == 'number' and isinstance(value, (int, float)) and not isinstance(value, bool))
                or (name == 'boolean' and isinstance(value, bool)) or (name in ('null', 'undefined') and value is None))
    if kind == 'literal':
        return type(value) is type(node[1]) and value == node[1] if not isinstance(node[1], float) else (
            isinstance(value, (int, float)) and not isinstance(value, bool) and value == node[1])
    if kind == 'union':
        return any(_fits(value, option, aliases, depth + 1) for option in node[1])
    if kind == 'array':
        return isinstance(value, list) and all(_fits(item, node[1], aliases, depth + 1) for item in value)
    if kind == 'tuple':
        return (isinstance(value, list) and len(value) == len(node[1])
                and all(_fits(item, part, aliases, depth + 1) for item, part in zip(value, node[1])))
    if kind == 'record':
        return isinstance(value, dict) and all(_fits(item, node[1], aliases, depth + 1) for item in value.values())
    if kind == 'object':
        if not isinstance(value, dict) or set(value) - set(node[1]):
            return False
        return all((name in value and _fits(value[name], field, aliases, depth + 1)) or (name not in value and optional)
                   for name, (field, optional) in node[1].items())
    if kind == 'alias':
        target = aliases.get(node[1])
        return target is not None and _fits(value, target, aliases, depth + 1)
    return False


def declared_result(text):
    """(result type tree, aliases) read from a call's opening text, or (None, {}) when it is not stated or not read."""
    aliases = {}
    for line in text.splitlines():
        found = re.match(r'\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*=\s*(.+?);?\s*$', line)
        if found:
            parsed = parse_type(found.group(2))
            if parsed is not None:
                aliases[found.group(1)] = parsed
    for line in text.splitlines():
        if not line.startswith('You are inside this call: '):
            continue
        signature, depth = line[len('You are inside this call: '):], 0
        for index, char in enumerate(signature):
            if char in '([{<':
                depth += 1
            elif char in ')]}>':
                depth -= 1
                if depth == 0 and char == ')' and signature[index + 1:index + 3] == ': ':
                    return parse_type(signature[index + 3:].strip()), aliases
        break
    return None, {}


def answer_fits(fact_text, child_text):
    """Whether a child_value fact may guide this child: a status outcome (no answer value) always may; an answer value
    only when the child declares a read result type that is not unknown/any and the value fits it."""
    first = fact_text.split('\n', 1)[0]
    if not first.startswith(ANSWER_PREFIX):
        return True
    try:
        value = json.loads(first[len(ANSWER_PREFIX):].removesuffix('.'))
    except ValueError:
        return False
    result, aliases = declared_result(child_text)
    if result is None or result == ('prim', 'unknown') or result == ('prim', 'any'):
        return False
    return _fits(value, result, aliases)


_IDENTIFIER = re.compile(r'"[^"\s]+"|[A-Za-z0-9][\w.:-]*')


def identifier_match(item):
    """A match string naming an input value rather than wording: a quoted value or a token containing a digit."""
    return bool(_IDENTIFIER.fullmatch(item)) and (item.startswith('"') or any(c.isdigit() for c in item))


def child_matches(match, text):
    """Whether a child reference's match strings select a child prompt: all of them, or all of its identifier strings
    (a policy's own child call is matched by the inputs it is called on, not by the reference's wording). The same
    rule as applications/guidance-writer/visibility.ts ``childMatches``."""
    if not match:
        return False
    if all(item in text for item in match):
        return True
    ids = [item for item in match if identifier_match(item)]
    return bool(ids) and all(item in text for item in ids)


def _child_match(child):
    """A reference child's prompt match strings: one string or a list of them (a bare string is one match, not a
    sequence of characters)."""
    match = child.get('match')
    match = (match,) if isinstance(match, str) else tuple(match or ())
    if not match or any(not isinstance(item, str) or not item for item in match):
        raise ValueError('reference child values need their exact prompt match strings')
    return match


def _child_text(child):
    """The verified answer of a reference child call: its ``value``, or what its verified actions (``calls``/``call``,
    ending in return_result) return. Steps before the return are shown too."""
    if child.get('value') is not None:
        return ANSWER_PREFIX + json.dumps(child['value'], sort_keys=True, ensure_ascii=False) + '.'
    calls = child.get('calls') or ([child['call']] if child.get('call') is not None else [])
    if not calls or not (isinstance(calls[-1], list) and len(calls[-1]) == 2 and calls[-1][0] == 'return_result'
                         and isinstance(calls[-1][1], dict)):
        raise ValueError('a reference child needs a value or verified calls ending in return_result')
    result = calls[-1][1]
    if result.get('status', 'success') == 'success':
        text = ANSWER_PREFIX + json.dumps(result.get('value'), sort_keys=True, ensure_ascii=False) + '.'
    else:
        text = (f'The verified outcome of this call is return_result with status {_text(result["status"])}'
                + (f': {_text(result["reason"])}' if result.get('reason') else '') + '.')
    if len(calls) > 1:
        text += '\nA verified way to reach it:\n' + _action_text(calls)
    return text


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
        add('child_value', _child_text(child), match=_child_match(child))
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
        return (invocation['role'] == 'child' and child_matches(fact.match, invocation['text'])
                and answer_fits(fact.text, invocation['text']))
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
        raise ValueError('guidance mode is import, write-only, text or text-twin')
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
    if mode in TEXT_MODES:
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
    Returns ``(insert_at, pieces, receipt)``; ``pieces`` is empty for write-only and text-twin (recorded for the
    OPSD twin, never sampled: a text-twin receipt carries its ``token_ids``) or when no fact fits the budget.
    """
    invocation, mode, custody, selection, facts, state = _request_parts(request)
    visible = {**invocation, 'text': text}
    hidden = [fact.id for fact in facts if not applicable(fact, visible)]
    if hidden:
        raise CustodyError('guidance facts are not visible to this invocation: ' + ', '.join(hidden))
    insert_at, placement = rendering.insertion_point(token_ids)
    included, omitted, pieces = list(facts), [], []
    recorded = mode in RECORDED_MODES
    if mode in TEXT_MODES:
        included = []
        for fact in facts:
            if len(rendering.turn([item.text for item in [*included, fact]])) <= rendering.budget_tokens:
                included.append(fact)
            else:
                omitted.append({'id': fact.id, 'reason': 'token_budget'})
        tokens = rendering.turn([fact.text for fact in included]) if included else []
        if tokens and not recorded:
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
               'token_count': len(tokens) if pieces or recorded else 0,
               'token_ids_sha256': canonical_json_sha256_hex(list(tokens) if pieces or recorded else [])}
    if mode in TEXT_MODES:
        receipt['fallback'] = request['fallback']
        if mode == 'text-twin':
            receipt['token_ids'] = list(tokens)
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
    if mode in RECORDED_MODES:
        if tokens or state is not None:
            raise ValueError(f'{mode} guidance must not enter the sampled prompt')
        if mode == 'text-twin':
            recorded = receipt.get('token_ids')
            if not isinstance(recorded, list) or any(type(token) is not int for token in recorded):
                raise ValueError('text-twin guidance records its token ids')
            return PrivilegedBlock(turn['graph_turn'], receipt['insert_at'], tuple(recorded), '', receipt)
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
    if receipt.get('rendering_sha256') != rendering.digest:
        raise ValueError('guidance rendering differs from the learner\'s')
    if receipt.get('role') != invocation['role'] or receipt.get('invocation') != invocation['name']:
        raise CustodyError('guidance was rendered for another invocation or role than the graph\'s')
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
    if mode in TEXT_MODES:
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
