"""Soft instructions and soft skills (plan §3.3): directly tuned native state per binding.

A binding's instruction text and each skill document disclosed to its invocations become native KV/hybrid state
spans. Each is initialized by writing the exact text at its actual rendered position in a real training call of that
binding (`engine.native_init_body`, the in-context writer; never bare text) under the base. At initialization every
program residual is exactly zero (zero cores / zero free-LoRA B), so the base IS the binding's initial stack; the
state therefore carries the base identity and is consumed by the adapted actor through the declared same-base
adapter-state exchange. After initialization the state is a parameter: it is updated only by the shared state
optimizer (`native_state_derivation.optimize_state`) with its immutable lineage, never recomputed from text.

Rows place a state where its text was: the instruction text inside the request messages, and every tool result
whose whole content is a disclosed skill document. The consumer reads it between the consume markers exactly as the
serving engine's `_native_consume` does.
"""
from __future__ import annotations

import copy
import hashlib
import json

from . import TuneMathError

SOFT_SCHEMA = 'natlang.tune.soft-state/1'


def text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def soft_key(binding, kind, target):
    return json.dumps([binding, kind, target])


def import_name(kind, target):
    return 'soft:instructions' if kind == 'instructions' else 'soft:' + target


def _parts(text, needle, state_id):
    before, after = text.split(needle)
    return [*([{'type': 'text', 'text': before}] if before else []), {'type': 'native_state', 'id': state_id},
            *([{'type': 'text', 'text': after}] if after else [])]


def place_states(turn, states):
    """Messages with soft-state parts in place of their exact texts, and the placed keys in order.

    ``states`` maps soft keys of this turn's binding to state IDs. The instruction text must occur exactly once in the
    string contents; a skill state replaces every tool result whose whole content is that document (each disclosure is
    one consumption). A row whose instruction text is missing or ambiguous is rejected.
    """
    messages = copy.deepcopy(turn['messages'])
    texts = turn.get('texts') or {}
    placed = []
    instruction_key = soft_key(turn['binding'], 'instructions', None)
    if instruction_key in states:
        text = texts.get('instructions')
        if not isinstance(text, str) or not text:
            raise TuneMathError('a soft-instruction binding row carries no instruction text')
        hits = [index for index, message in enumerate(messages)
                if isinstance(message.get('content'), str) for _ in range(message['content'].count(text))]
        if len(hits) != 1:
            raise TuneMathError(f'instruction text occurs {len(hits)} times in the request; its soft state needs exactly one position')
        messages[hits[0]]['content'] = _parts(messages[hits[0]]['content'], text, states[instruction_key])
        placed.append(instruction_key)
    for target, document in sorted((texts.get('skills') or {}).items()):
        key = soft_key(turn['binding'], 'skill', target)
        if key not in states:
            continue
        for message in messages:
            if message.get('role') == 'tool' and message.get('content') == document:
                message['content'] = [{'type': 'native_state', 'id': states[key]}]
                placed.append(key)
    return messages, placed


def initialize_soft_states(engine, turns, *, skills=True, owner=None):
    """One in-context-written state per (binding, instructions) and per disclosed (binding, skill target).

    Deterministic: the initializing call is the first training turn (by occurrence) of that binding where the text has
    its single position. Returns ``{key: record}`` with the state ID, text digest, token count and init receipt.
    """
    records = {}
    ordered = sorted(turns, key=lambda turn: (turn['binding'] or '', turn['root'], turn.get('call') or '', turn['turn']))
    for turn in ordered:
        if turn.get('partition') != 'train' or not turn.get('binding'):
            continue
        texts = turn.get('texts') or {}
        wanted = []
        if isinstance(texts.get('instructions'), str) and texts['instructions']:
            wanted.append(('instructions', None, texts['instructions']))
        if skills:
            wanted.extend(('skill', target, text) for target, text in sorted((texts.get('skills') or {}).items())
                          if isinstance(text, str) and text)
        for kind, target, text in wanted:
            key = soft_key(turn['binding'], kind, target)
            if key in records:
                if records[key]['textSha256'] != text_sha256(text):
                    raise TuneMathError(f'binding {turn["binding"]} carries two different {kind} texts; one state needs one text')
                continue
            placeholder = engine.native_write_text(text)['id']
            try:
                messages, placed = place_states(turn, {key: placeholder})
            except TuneMathError:
                continue
            if key not in placed:
                continue  # a skill not disclosed in this turn initializes from a later one
            written = engine.native_init_body(messages, turn['tools'], placeholder, text,
                                              placeholder_occurrence=0 if placed.count(key) > 1 else None)
            block = engine.store.get_native_state(written['id']).block
            records[key] = {'binding': turn['binding'], 'kind': kind, 'target': target, 'state': written['id'],
                            'textSha256': text_sha256(text), 'tokens': len(block.token_ids),
                            'init': {'schema': SOFT_SCHEMA, 'writer': 'native_init_body', 'actor': dict(engine.native_identity),
                                     'tokenizer_fingerprint': engine.model_config['tokenizer_fingerprint'],
                                     'context_occurrence': f"{turn['root']}:{turn.get('call') or 'root'}:{turn['turn']}",
                                     'placeholder': placeholder}}
    return records


def soft_items_to_node(engine, prompt, target, *, prompt_weight, state_names):
    """Lower rendered items to one consumer node: block items become consume-marked imports."""
    tokens, weights, imports, external = [], [], [], {}
    from ..model.native_graph_plan import ImportSpan
    for index, (kind, value) in enumerate([*prompt, *target]):
        primary = index >= len(prompt)
        if kind == 'tok':
            tokens.append(int(value))
            weights.append(1. if primary else float(prompt_weight))
            continue
        if kind != 'block' or primary:
            raise TuneMathError('a soft-state row may import state only in its prompt')
        state_id = value['id'] if isinstance(value, dict) else value
        name = state_names.get(state_id)
        if name is None:
            raise TuneMathError(f'the request imports state {state_id}, which is not a soft state of this binding')
        block = engine.store.get_native_state(state_id).block
        count = len(block.token_ids)
        if name in external and external[name]['id'] != state_id:
            raise TuneMathError('one import name maps to two states')
        external[name] = {'id': state_id, 'token_count': count}
        tokens.append(engine.role_tokens.consume_open); weights.append(0.)
        imports.append(ImportSpan(len(tokens), name, 0, count))
        # The forced close marker follows the import and has no consumer predecessor query.
        tokens.append(engine.role_tokens.consume_close); weights.append(0.)
    if weights:
        weights[0] = 0.
    return tokens, weights, imports, external
