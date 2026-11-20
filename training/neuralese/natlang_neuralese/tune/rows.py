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


# Own replies (sft_sources' own episodes, key prefix 'rl'): every sampled token is a target.
OWN_TEACHERS = ('rl',)


def turn_graph(engine, turn, *, prompt_weight, soft=None, prefix=None):
    """One collected turn as a single-node native graph row bound to its program binding.

    ``soft`` maps soft keys (`tune.soft.soft_key`) to the initial state IDs: the binding's instruction text and
    disclosed skill documents are then imported as those states at their positions (plan §3.3), and the row records
    which trained soft parameter each import name is. ``prefix``: context token ids placed before the first imported
    state (prefix-robust training, ``lower_rows``); a row without soft states takes none.
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
    # Each state stands only for the exact text it was primed from (its block's text), as at serving.
    state_texts = {key: engine.store.get_native_state(state).block.text for key, state in mine.items()}
    messages, placed = place_states(turn, mine, state_texts) if mine else (turn['messages'], [])
    decision = turn.get('decision')
    if decision is not None:
        from ..serve.native_learning import decision_items
        options, chosen = decision.get('options'), decision.get('chosen')
        if (not isinstance(options, list) or len(options) < 2 or any(not isinstance(item, str) for item in options)
                or len(set(options)) != len(options) or type(chosen) is not int or not 0 <= chosen < len(options)):
            raise TuneMathError('a decision turn needs distinct options and its selected option')
        prompt, continuations = decision_items(engine, messages, turn['tools'], options)
        names = {mine[key]: import_name(*json.loads(key)[1:]) for key in placed}
        ids, _, imports, external = soft_items_to_node(engine, prompt, [], prompt_weight=0., state_names=names,
                                                       prefix=prefix)
        import dataclasses
        occurrence = f"{turn['root']}:{turn.get('call') or 'root'}:{turn['turn']}"
        nodes = [{'name':f'decision:option:{index}', 'token_ids':ids + continuation,
                  'imports':[dataclasses.asdict(edge) for edge in imports],
                  'target_weights':[0.] * (len(ids) + len(continuation))}
                 for index, continuation in enumerate(continuations)]
        graph = {'schema':GRAPH_SCHEMA,'id':'tune:'+occurrence,'split':'train',
                 'source_groups':['tune-component:'+turn['component']],
                 'tokenizer_fingerprint':engine.model_config['tokenizer_fingerprint'],'nodes':nodes,
                 'decisions':[{'name':'decision','options':[node['name'] for node in nodes],
                               'prompt_tokens':len(ids),'label':chosen,'label_source':'verified-trajectory',
                               'values':options}], **({'external_states':external} if external else {}),
                 'source':{'schema':TURN_SCHEMA,'sha256':canonical_json_sha256_hex(turn),'teacher':turn['teacher'],
                           'source':turn['source'],'evidence':turn['evidence'],**_prefix_record(prefix, external)}}
        row = {'binding':turn['binding'],'graph':graph,'occurrence':occurrence,'evidence_weight':weight}
        if external:
            keys = {import_name(*json.loads(key)[1:]):key for key in placed}
            row['soft'] = {name:keys[name] for name in external}
        return row
    native_xml=turn.get('native_xml_turn')
    if native_xml:
        from ..data.native_xml_custody import authenticate
        calls=authenticate(native_xml)
        if (calls!=turn['reply'].get('calls') or native_xml['context']!=turn['messages'] or
                native_xml['tools_offered']!=turn['tools'] or native_xml['response_tools']!=turn.get('response_tools')):
            raise TuneMathError('native XML tune turn differs from its exact request and action')
    prompt, target = target_items(engine, messages, turn['tools'], reply_message(turn['reply']),
                                  **({'response_tools':turn['response_tools']} if native_xml else {}))
    if not target:
        raise TuneMathError('the reply rendered no target tokens')
    names = {mine[key]: import_name(*json.loads(key)[1:]) for key in placed}
    if not names and any(item[0] != 'tok' for item in [*prompt, *target]):
        raise TuneMathError('a teacher text turn rendered native-state items; it is not a text trajectory')
    ids, weights, imports, external = soft_items_to_node(engine, prompt, target, prompt_weight=prompt_weight, state_names=names,
                                                         prefix=prefix)
    if turn.get('teacher') not in OWN_TEACHERS:
        # a reply the student did not sample: its think block and the text before its first tool call are context
        from ..serve.targets import non_student_context
        lead = non_student_context(engine.tokenizer, reply_message(turn['reply']), [value for _, value in target])
        for position in range(len(ids) - len(target), len(ids) - len(target) + lead):
            weights[position] = 0.
    occurrence = f"{turn['root']}:{turn.get('call') or 'root'}:{turn['turn']}"
    import dataclasses
    graph = {'schema': GRAPH_SCHEMA, 'id': 'tune:' + occurrence, 'split': 'train',
             'source_groups': ['tune-component:' + turn['component']],
             'tokenizer_fingerprint': engine.model_config['tokenizer_fingerprint'],
             'nodes': [{'name': 'turn', 'token_ids': ids, 'imports': [dataclasses.asdict(edge) for edge in imports],
                        'producing': [], 'target_weights': weights}],
             **({'external_states': external} if external else {}),
             'source': {'schema': TURN_SCHEMA, 'sha256': canonical_json_sha256_hex(turn), 'teacher': turn['teacher'],
                        'source': turn['source'], 'evidence': turn['evidence'], **_prefix_record(prefix, external)}}
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


def _prefix_record(prefix, external):
    if not prefix or not external:
        return {}
    return {'prefix_robust': {'tokens': len(prefix), 'sha256': canonical_json_sha256_hex([int(t) for t in prefix])}}


# Prefix-robust soft-state training (CSPD, Zhang et al. 2026, section 2.2 and appendix A): a soft state is trained
# in its rows' contexts but at inference follows whatever the program put before it (other skills' states, other
# callers' text). On a share of the rows that import soft states, chosen by a content hash, a context prefix of
# PREFIX_TOKENS tokens drawn from other rows' prompt tokens of the same component (chat-template symbols included, it
# may end inside a turn) sits right before the first imported state, as context at weight 0. Serving never adds it.
PREFIX_SHARE = .5
PREFIX_TOKENS = (128, 256)


def _hash_unit(text, salt):
    digest = canonical_json_sha256_hex([salt, text])
    return int(digest[:12], 16) / float(16 ** 12), int(digest[12:24], 16), int(digest[24:36], 16)


def _prompt_lead(row, markers):
    """A lowered row's prompt tokens before its first Neuralese marker (pure rendered text, all weight 0)."""
    node = row['graph']['nodes'][0]
    decisions = row['graph'].get('decisions') or []
    end = decisions[0]['prompt_tokens'] if decisions else len(node['token_ids'])   # an option's text is no prompt
    out = []
    for token, weight in zip(node['token_ids'][:end], (node.get('target_weights') or [0.] * end)[:end]):
        if int(token) in markers or weight:
            break
        out.append(int(token))
    return out


def prefix_for(turn, pool, share, lengths):
    """The context prefix of one soft-state row, or None: selected when its content hash falls below ``share``; drawn
    deterministically from the concatenated prompt leads of the other rows of its component (``pool``: [(occurrence,
    ids)]), ``lengths`` (lo, hi) tokens long. None when the other rows hold fewer than ``lo`` tokens."""
    if not share:
        return None
    unit, pick, start = _hash_unit(canonical_json_sha256_hex(turn), 'prefix-robust')
    if unit >= share:
        return None
    own = f"{turn['root']}:{turn.get('call') or 'root'}:{turn['turn']}"
    others = [ids for occurrence, ids in sorted(pool, key=lambda item: item[0]) if occurrence != own and ids]
    if not others:
        return None
    offset = pick % len(others)
    stream = [token for ids in others[offset:] + others[:offset] for token in ids]
    lo, hi = lengths
    if len(stream) < lo:
        return None
    length = min(len(stream), lo + pick % (hi - lo + 1))
    begin = start % (len(stream) - length + 1)
    return stream[begin:begin + length]


def lower_rows(engine, rows, *, prompt_weight=0., soft=None, prefix_share=0., prefix_tokens=PREFIX_TOKENS):
    """Turn rows lowered to graph rows (other rows pass through). ``prefix_share`` > 0: prefix-robust training of the
    soft states (``prefix_for``); the fit's recipe sets it (``Recipe.soft_prefix_share``)."""
    lowered = [turn_graph(engine, row, prompt_weight=prompt_weight, soft=soft) if row.get('schema') == TURN_SCHEMA else row
               for row in rows]
    if not prefix_share or not soft:
        return lowered
    lo, hi = prefix_tokens
    if not 0 < prefix_share <= 1 or type(lo) is not int or type(hi) is not int or not 0 < lo <= hi:
        raise TuneMathError('prefix-robust training needs a share in (0, 1] and token bounds 0 < lo <= hi')
    markers = set(engine.role_tokens.ids)
    pools = {}
    for turn, row in zip(rows, lowered):
        if turn.get('schema') == TURN_SCHEMA:
            pools.setdefault(turn['component'], []).append((row['occurrence'], _prompt_lead(row, markers)))
    out = []
    for turn, row in zip(rows, lowered):
        prefix = (prefix_for(turn, pools[turn['component']], prefix_share, (lo, hi))
                  if turn.get('schema') == TURN_SCHEMA and row.get('soft') else None)
        out.append(turn_graph(engine, turn, prompt_weight=prompt_weight, soft=soft, prefix=prefix) if prefix else row)
    return out
