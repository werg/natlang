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
ACTION_POLICY = 'natlang.tune-action-outcome/2'


def verify_action_evidence(turn):
    evidence = turn.get('evidence') or {}
    if evidence.get('actionPolicy') != ACTION_POLICY:
        raise TuneMathError('turn lacks the current authenticated action-outcome policy; re-lower the source record')
    if not isinstance(evidence.get('failedHistory'), bool):
        raise TuneMathError('turn lacks a boolean failed-history masking disposition')
    calls = (turn.get('reply') or {}).get('calls') or []
    actions = evidence.get('actionEvents')
    if not isinstance(actions, list) or len(actions) != len(calls):
        raise TuneMathError('turn action evidence does not bind every assistant tool call')
    disposition = 'admissible'
    seen = set()
    for call, action in zip(calls, actions):
        if (not isinstance(call, (list, tuple)) or len(call) != 2 or not isinstance(call[0], str)
                or not isinstance(action, dict) or not isinstance(action.get('toolCallId'), str)
                or not action['toolCallId'] or action['toolCallId'] in seen
                or action.get('name') != call[0]
                or action.get('argumentsSha256') != canonical_json_sha256_hex(call[1])
                or not isinstance(action.get('outcome'), str)):
            raise TuneMathError('turn action evidence differs from its recorded teacher tool call')
        seen.add(action['toolCallId'])
        outcome = action['outcome']
        if outcome in ('rejected', 'refused', 'error', 'failed'):
            disposition = 'rejected'
        elif outcome == 'blocked' and evidence.get('rootOracle') is True and call[0] == 'return_result':
            pass  # The importer only emits this for the exact reviewed honest blocked root.
        elif outcome not in ('ok', 'completed', 'done'):
            disposition = 'unverified'
    if evidence.get('actionDisposition') != disposition:
        raise TuneMathError('turn action disposition disagrees with its recorded action outcomes')
    return disposition


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


def turn_graph(engine, turn, *, prompt_weight, soft=None):
    """One collected turn as a single-node native graph row bound to its program binding.

    ``soft`` maps soft keys (`tune.soft.soft_key`) to the initial state IDs: the binding's instruction text and
    disclosed skill documents are then imported as those states at their positions (plan §3.3), and the row records
    which trained soft parameter each import name is.
    """
    if turn.get('schema') != TURN_SCHEMA:
        raise TuneMathError('expected a natlang.tune-turn/v1 row')
    if turn.get('partition') != 'train':
        raise TuneMathError('only training-partition turns become training rows')
    if not 0 <= prompt_weight <= 1:
        raise TuneMathError('prompt weight must lie in [0, 1]')
    correction = turn.get('correction')
    if correction is not None:
        # An on-policy correction (§4.4): its prompt holds the student's mistaken prefix, which is context and never a
        # target, so only the teacher's continuation is weighted; admission required the repaired root's oracle.
        verification = correction.get('verification') or {}
        if verification.get('rootOracle') is not True or verification.get('prefix') != 'observed-identical':
            raise TuneMathError('a correction row needs a verified continuation and an identically observed prefix')
        prompt_weight = 0.
    if verify_action_evidence(turn) != 'admissible':
        raise TuneMathError('a rejected or unverified assistant action is not a positive target')
    if turn['evidence'].get('failedHistory') is True:
        # Keep failed assistant replies in the rendered request as context; no prompt token in that history is a target.
        prompt_weight = 0.
    weight = evidence_weight(turn)
    from ..serve.targets import target_items
    from .soft import import_name, place_states, soft_items_to_node
    mine = {key: state for key, state in (soft or {}).items() if json.loads(key)[0] == turn['binding']}
    messages, placed = place_states(turn, mine) if mine else (turn['messages'], [])
    prompt, target = target_items(engine, messages, turn['tools'], reply_message(turn['reply']))
    if not target:
        raise TuneMathError('the reply rendered no target tokens')
    names = {mine[key]: import_name(*json.loads(key)[1:]) for key in placed}
    if not names and any(item[0] != 'tok' for item in [*prompt, *target]):
        raise TuneMathError('a teacher text turn rendered native-state items; it is not a text trajectory')
    ids, weights, imports, external = soft_items_to_node(engine, prompt, target, prompt_weight=prompt_weight, state_names=names)
    occurrence = f"{turn['root']}:{turn.get('call') or 'root'}:{turn['turn']}"
    import dataclasses
    graph = {'schema': GRAPH_SCHEMA, 'id': 'tune:' + occurrence, 'split': 'train',
             'source_groups': ['tune-component:' + turn['component']],
             'tokenizer_fingerprint': engine.model_config['tokenizer_fingerprint'],
             'nodes': [{'name': 'turn', 'token_ids': ids, 'imports': [dataclasses.asdict(edge) for edge in imports],
                        'producing': [], 'target_weights': weights}],
             **({'external_states': external} if external else {}),
             'source': {'schema': TURN_SCHEMA, 'sha256': canonical_json_sha256_hex(turn), 'teacher': turn['teacher'],
                        'source': turn['source'], 'evidence': turn['evidence']}}
    row = {'binding': turn['binding'], 'graph': graph, 'occurrence': occurrence, 'evidence_weight': weight}
    if external:
        keys = {import_name(*json.loads(key)[1:]): key for key in placed}
        row['soft'] = {name: keys[name] for name in external}
    return row


def evidence_weight(turn):
    """Separate evidence, conservatively combined until development calibration exists (§4.3): an oracle-confirmed
    root gives full weight; a successful enclosing program without an oracle is weak attribution."""
    evidence = turn['evidence']
    if verify_action_evidence(turn) != 'admissible':
        raise TuneMathError('a rejected or unverified assistant action is not a positive target')
    if evidence.get('rootOracle') is True:
        return 1.
    if evidence.get('enclosingSuccess'):
        return .5
    raise TuneMathError('a turn from an unsuccessful root is not a positive target')


def lower_rows(engine, rows, *, prompt_weight=.1, soft=None):
    return [turn_graph(engine, row, prompt_weight=prompt_weight, soft=soft) if row.get('schema') == TURN_SCHEMA else row
            for row in rows]
