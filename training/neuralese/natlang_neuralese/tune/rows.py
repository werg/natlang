"""Collected `natlang tune` turns lowered to native graph training rows for one student.

A teacher turn transfers through text: its exact request messages, offered tools and reply are rendered with the
student's own serving chat template (`serve.targets.target_items`, the rendering inference uses), never by token
index or teacher state. The reply is the primary target; the rendered prompt is auxiliary at the recipe's lower
weight (whole-trajectory supervision). A request carrying native state is not a text trajectory and is rejected here.
"""
from __future__ import annotations

import json

from ..common.hashing import canonical_json_sha256_hex
from ..data.native_graph_schema import SCHEMA as GRAPH_SCHEMA
from . import TuneMathError

TURN_SCHEMA = 'natlang.tune-turn/v1'


def reply_message(reply):
    """The assistant message a runtime history holds for this reply (tool calls with JSON-string arguments)."""
    calls = reply.get('calls') or []
    message = {'role': 'assistant', 'content': reply.get('text') or ''}
    if calls:
        message['tool_calls'] = [{'id': f'call_{index}', 'type': 'function',
                                  'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}
                                 for index, (name, arguments) in enumerate(calls)]
    if reply.get('reasoning'):
        message['reasoning_content'] = reply['reasoning']
    return message


def turn_graph(engine, turn, *, prompt_weight):
    """One collected turn as a single-node native graph row bound to its program binding."""
    if turn.get('schema') != TURN_SCHEMA:
        raise TuneMathError('expected a natlang.tune-turn/v1 row')
    if turn.get('partition') != 'train':
        raise TuneMathError('only training-partition turns become training rows')
    if not 0 <= prompt_weight <= 1:
        raise TuneMathError('prompt weight must lie in [0, 1]')
    weight = evidence_weight(turn)
    from ..serve.targets import target_items
    prompt, target = target_items(engine, turn['messages'], turn['tools'], reply_message(turn['reply']))
    if any(item[0] != 'tok' for item in [*prompt, *target]):
        raise TuneMathError('a teacher text turn rendered native-state items; it is not a text trajectory')
    ids = [item[1] for item in prompt] + [item[1] for item in target]
    if not target:
        raise TuneMathError('the reply rendered no target tokens')
    weights = [0.] + [float(prompt_weight)] * (len(prompt) - 1) + [1.] * len(target)
    occurrence = f"{turn['root']}:{turn.get('call') or 'root'}:{turn['turn']}"
    graph = {'schema': GRAPH_SCHEMA, 'id': 'tune:' + occurrence, 'split': 'train',
             'source_groups': ['tune-component:' + turn['component']],
             'tokenizer_fingerprint': engine.model_config['tokenizer_fingerprint'],
             'nodes': [{'name': 'turn', 'token_ids': ids, 'imports': [], 'producing': [], 'target_weights': weights}],
             'source': {'schema': TURN_SCHEMA, 'sha256': canonical_json_sha256_hex(turn), 'teacher': turn['teacher'],
                        'source': turn['source'], 'evidence': turn['evidence']}}
    return {'binding': turn['binding'], 'graph': graph, 'occurrence': occurrence, 'evidence_weight': weight}


def evidence_weight(turn):
    """Separate evidence, conservatively combined until development calibration exists (§4.3): an oracle-confirmed
    root gives full weight; a successful enclosing program without an oracle is weak attribution."""
    evidence = turn['evidence']
    if evidence.get('rootOracle') is True:
        return 1.
    if evidence.get('enclosingSuccess'):
        return .5
    raise TuneMathError('a turn from an unsuccessful root is not a positive target')


def lower_rows(engine, rows, *, prompt_weight=.1):
    return [turn_graph(engine, row, prompt_weight=prompt_weight) if row.get('schema') == TURN_SCHEMA else row for row in rows]
