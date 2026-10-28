"""Privileged training-world context for JOLT+ self-teaching (shared mechanism).

One mechanism serves the whole-graph RL learner and the teacher-free program
tuning supply. Decomposition: ``PRIVILEGED_CONTEXT_DECOMPOSITION.md``.

Units:

1. Custody is the shared ``common.custody.ProtectedWorlds`` (every declared
   protected source, failing closed). Privileged facts are derived only from
   ``split == 'train'`` worlds related to no protected world.
2. ``case_facts`` (crisp extraction): authored reference solution, reference
   child values, oracle checks expressed as facts, verified teacher
   trajectories of the same world and earlier-attempt runtime feedback.
3. ``selectPrivilegedFacts`` (policy): which facts a given invocation sees, in
   which order. Crisp default here; the natural-language implementation is
   ``applications/curriculum-advisor/selectPrivilegedFacts.nl``, executed by an
   optional command with one JSON request on stdin (same interface as
   ``production.select_tasks``).
4. ``PrivilegedRendering`` (crisp, hash-bound): one declared ChatML turn per
   backbone template, inserted at a declared place (after the leading system
   turn, else after BOS), whole facts packed in policy order into a token
   budget. Every block carries a canonical receipt binding rendering, policy,
   facts, custody and the exact token IDs.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..common.custody import CustodyError
from ..common.hashing import canonical_json_sha256_hex, sha256_hex

BLOCK_SCHEMA = 'natlang.jolt-privileged-block/1'
RENDERING_SCHEMA = 'natlang.jolt-privileged-rendering/1'
CRISP_POLICY = 'natlang.select-privileged-facts.crisp/1'
NL_POLICY_PATH = 'applications/curriculum-advisor/selectPrivilegedFacts.nl'
FACT_KINDS = ('expected_result', 'expected_stop', 'expected_files', 'expected_effects', 'reference_failure',
              'reference_solution', 'child_value', 'minimum_sequence', 'decisive', 'teacher_trajectory',
              'attempt_feedback')


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


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


# ---------------------------------------------------------------- policy

def applicable(fact, invocation):
    """Facts a given invocation may see at all (exact mechanism, not preference)."""
    if fact.kind == 'child_value':
        return invocation['role'] == 'child' and all(item in invocation['text'] for item in fact.match)
    if fact.kind == 'attempt_feedback':
        return fact.invocation is None or fact.invocation == invocation['name']
    return invocation['role'] == 'root'


CRISP_ORDER = {'root': ('expected_result', 'expected_stop', 'expected_files', 'expected_effects', 'reference_solution',
                        'attempt_feedback', 'reference_failure', 'minimum_sequence', 'decisive', 'teacher_trajectory'),
               'child': ('child_value', 'attempt_feedback')}


def select_crisp(invocation, facts, budget_tokens):
    del budget_tokens  # packing is the crisp mechanism's, after the policy
    order = CRISP_ORDER[invocation['role']]
    chosen = [fact for fact in facts if fact['kind'] in order]
    values = {fact['text'] for fact in chosen if fact['kind'] == 'child_value'}
    if len(values) > 1:  # ambiguous child match: no verified answer for this call
        chosen = [fact for fact in chosen if fact['kind'] != 'child_value']
    return [fact['id'] for fact in sorted(chosen, key=lambda fact: order.index(fact['kind']))]


def select_facts(invocation, facts, budget_tokens, *, command=None):
    """selectPrivilegedFacts(invocation, facts, budget) -> ordered fact ids."""
    request = [{'id': fact.id, 'kind': fact.kind, 'text': fact.text} for fact in facts if applicable(fact, invocation)]
    if command:
        response = subprocess.run(command, input=json.dumps({'invocation': invocation, 'facts': request,
                                  'budgetTokens': budget_tokens}), text=True, capture_output=True, check=True)
        chosen = json.loads(response.stdout)
    else:
        chosen = select_crisp(invocation, request, budget_tokens)
    known = {fact['id'] for fact in request}
    if (not isinstance(chosen, list) or any(not isinstance(item, str) or item not in known for item in chosen)
            or len(set(chosen)) != len(chosen)):
        raise ValueError('selectPrivilegedFacts must return distinct ids of applicable facts')
    return chosen


def policy_identity(command, root=None):
    if not command:
        return {'implementation': 'crisp', 'sha256': sha256_hex(CRISP_POLICY.encode())}
    path = Path(root or Path(__file__).resolve().parents[4]) / NL_POLICY_PATH
    return {'implementation': 'nl', 'sha256': sha256_hex(path.read_bytes()),
            'command_sha256': canonical_json_sha256_hex(list(command))}


# ---------------------------------------------------------------- rendering

@dataclass(frozen=True)
class PrivilegedRendering:
    """Declared rendering of a privileged ChatML turn for one backbone template.

    ``encode`` is the template's content tokenizer (``Renderer.text``); it is
    bound by ``tokenizer_fingerprint``, not hashed as code.
    """
    template: str
    bos: int | None
    im_start: int
    im_end: int
    tokenizer_fingerprint: str
    encode: Callable = field(compare=False, repr=False)
    role: str = 'system'
    header: str = 'Verified reference for this call (training only):'
    placement: str = 'after_leading_system_turn'
    budget_tokens: int = 512

    @classmethod
    def from_renderer(cls, renderer, *, template, tokenizer_fingerprint, **options):
        special = renderer.special
        return cls(template, special.bos, special.im_start, special.im_end, tokenizer_fingerprint,
                   renderer.text, **options)

    def declaration(self):
        return {'schema': RENDERING_SCHEMA, 'template': self.template, 'bos': self.bos, 'im_start': self.im_start,
                'im_end': self.im_end, 'tokenizer_fingerprint': self.tokenizer_fingerprint, 'role': self.role,
                'header': self.header, 'placement': self.placement, 'budget_tokens': self.budget_tokens}

    @property
    def digest(self):
        return canonical_json_sha256_hex(self.declaration())

    def turn(self, texts):
        body = '\n\n'.join([self.header, *texts])
        return [self.im_start, *self.encode(f'{self.role}\n'), *self.encode(body), self.im_end, *self.encode('\n')]

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
    invocation: str
    insert_at: int
    token_ids: tuple
    text: str
    receipt: dict

    @property
    def digest(self):
        return canonical_json_sha256_hex(self.receipt)


def render_block(rendering, *, invocation, token_ids, facts, custody_receipt, command=None, policy_root=None):
    """Select, pack whole facts into the budget, and bind the rendered block.

    ``invocation``: ``{name, role: 'root'|'child', text}``. Facts that do not fit
    are omitted whole (a later smaller fact can still fit). With no fitting
    fact the block is empty and inserts nothing.
    """
    if invocation.get('role') not in ('root', 'child') or not isinstance(invocation.get('text'), str):
        raise ValueError('privileged invocation needs its role and actual prompt text')
    by_id = {fact.id: fact for fact in facts}
    order = select_facts(invocation, facts, rendering.budget_tokens, command=command)
    included, omitted = [], []
    for ident in order:
        trial = included + [by_id[ident]]
        if len(rendering.turn([fact.text for fact in trial])) <= rendering.budget_tokens:
            included = trial
        else:
            omitted.append({'id': ident, 'reason': 'token_budget'})
    insert_at, placement = rendering.insertion_point(token_ids)
    tokens = tuple(rendering.turn([fact.text for fact in included])) if included else ()
    text = '\n\n'.join([rendering.header, *(fact.text for fact in included)]) if included else ''
    receipt = {'schema': BLOCK_SCHEMA, 'invocation': invocation['name'], 'role': invocation['role'],
               'rendering_sha256': rendering.digest, 'policy': policy_identity(command, policy_root),
               'custody': custody_receipt, 'facts': [{'id': fact.id, 'kind': fact.kind, 'sha256': fact.digest}
                                                     for fact in included],
               'omitted': omitted, 'insert_at': insert_at, 'placement': placement, 'token_count': len(tokens),
               'token_ids_sha256': canonical_json_sha256_hex(list(tokens)),
               'text_sha256': sha256_hex(text.encode())}
    return PrivilegedBlock(invocation['name'], insert_at, tokens, text, receipt)


def verify_block(block, rendering, *, custody):
    """Re-verify a block's receipt against its tokens, rendering and custody."""
    receipt = block.receipt
    if (receipt.get('schema') != BLOCK_SCHEMA or receipt['invocation'] != block.invocation
            or receipt['insert_at'] != block.insert_at or receipt['rendering_sha256'] != rendering.digest
            or receipt['token_count'] != len(block.token_ids)
            or receipt['token_ids_sha256'] != canonical_json_sha256_hex(list(block.token_ids))
            or receipt['text_sha256'] != sha256_hex(block.text.encode())
            or receipt['custody'].get('protected_sha256') != custody.digest
            or receipt['custody'].get('split') != 'train'):
        raise CustodyError('privileged block differs from its bound receipt, rendering or custody')
    return block


def privileged_blocks(rendering, case, invocations, *, custody, teacher_trajectories=(), attempt_feedback=(),
                      command=None, policy_root=None):
    """Blocks for every invocation of one episode graph.

    ``invocations``: ``[{name, role, text, token_ids}]`` with the actual own
    token IDs of each graph invocation (student form, no block).
    """
    receipt, facts = case_facts(case, custody=custody, teacher_trajectories=teacher_trajectories,
                                attempt_feedback=attempt_feedback)
    return {item['name']: render_block(rendering, invocation={key: item[key] for key in ('name', 'role', 'text')},
                                       token_ids=item['token_ids'], facts=facts, custody_receipt=receipt,
                                       command=command, policy_root=policy_root)
            for item in invocations}


def insert_into_prompt_segments(segments, block):
    """Actor-side insertion for teacher-role sampling: the same declared place.

    The block is inserted as an ordinary token segment at own-token index
    ``block.insert_at`` of the captured prompt; native-state imports stay at
    their own-token boundaries (and must not sit exactly at the insertion).
    """
    result, own = [], 0
    inserted = not block.token_ids
    for number, segment in enumerate(segments):
        if segment.get('kind') == 'native_state':
            if not inserted and own == block.insert_at:
                raise ValueError('privileged block cannot share its boundary with a native-state import')
            result.append(segment)
            continue
        ids, flags = list(segment['token_ids']), list(segment['producing'])
        following = segments[number+1] if number+1 < len(segments) else None
        if (not inserted and block.insert_at == own + len(ids) and following is not None
                and following.get('kind') == 'native_state'):
            raise ValueError('privileged block cannot share its boundary with a native-state import')
        if not inserted and own <= block.insert_at <= own + len(ids):
            cut = block.insert_at - own
            if 0 < cut < len(ids) and flags[cut-1] and flags[cut]:
                raise ValueError('privileged block cannot split a producing span')
            pieces = [(ids[:cut], flags[:cut]), (list(block.token_ids), [False]*len(block.token_ids)),
                      (ids[cut:], flags[cut:])]
            result.extend({**segment, 'token_ids': a, 'producing': b} for a, b in pieces if a)
            inserted = True
        else:
            result.append(segment)
        own += len(ids)
    if not inserted:
        raise ValueError('privileged insertion point is outside the captured prompt')
    return result
