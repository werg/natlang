"""Exact whole-episode occurrence coverage, independent of Neuralese use."""
from __future__ import annotations


def validate_model_turn_coverage(captured_turn_names, execution_graph):
    """Return indexed events; reject dropping any in-episode model occurrence.

    An empty graph imposes no communication requirement. A supplied graph must
    describe exactly the captured model turns, including ordinary turns and
    forced producers. Older writer references outside these events remain
    explicit episode inputs rather than being confused with missing producers.
    """
    names = list(captured_turn_names)
    if (any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names)) or not isinstance(execution_graph, list)):
        raise ValueError('whole-episode graph requires unique captured model turns and graph events')
    nodes = {}
    for event in execution_graph:
        if not isinstance(event, dict) or not isinstance(event.get('node'), str) or not event['node']:
            raise ValueError('whole-episode graph event lacks its exact occurrence identity')
        key = event['node']
        if key in nodes:
            raise ValueError('whole-episode graph repeats an occurrence identity')
        nodes[key] = event
    if not nodes:
        return nodes
    expected = {name for name, event in nodes.items() if event.get('kind') == 'model_turn'}
    if expected != set(names):
        raise ValueError('whole-episode graph model-turn inventory differs from captured turns')
    for event in nodes.values():
        if event.get('kind') not in {'block_write', 'block_read'}:
            continue
        if event.get('turn') not in expected:
            raise ValueError('whole-episode graph block occurrence lacks its captured producer/consumer turn')
        if event['kind'] == 'block_read':
            inputs = event.get('inputs')
            if not isinstance(inputs, list) or any(not isinstance(item, dict) for item in inputs):
                raise ValueError('whole-episode graph read lacks its exact input edges')
            sources = [item.get('node') for item in inputs if item.get('block') == event.get('block')]
            if len(sources) != 1 or not isinstance(sources[0], str):
                raise ValueError('whole-episode graph read needs one exact writer occurrence or external input')
            source = nodes.get(sources[0])
            if source is not None and (source.get('kind') != 'block_write'
                    or source.get('block') != event.get('block') or source.get('turn') not in expected):
                raise ValueError('whole-episode graph read changed or omitted its in-episode producer')
    return nodes



def rollout_turn_names(turns):
    counters, result = {}, []
    for turn in turns:
        call = turn.get('invocation_id')
        if not isinstance(call, str) or not call:
            raise ValueError('whole-episode policy turn needs its actual nonempty invocation identity')
        counters[call] = counters.get(call, 0) + 1
        result.append(f'{call}#turn{counters[call]}')
    return result


def validate_rollout_graph(rollout):
    turns = rollout.get('turns') or []
    events = rollout.get('execution_graph', [])
    if events is None:
        events = []
    names = rollout_turn_names(turns)
    nodes = validate_model_turn_coverage(names, events)
    if not nodes and any((turn.get('sampling') or {}).get('writes') or any(
            part.get('kind') == 'native_state' for part in (turn.get('sampling') or {}).get('prompt', []))
            for turn in turns):
        raise ValueError('whole-episode Neuralese RL requires its actual complete execution graph')
    for name, turn in zip(names, turns):
        if turn.get('native_xml_turn'):
            from ..data.native_xml_custody import validate_replay_turn
            validate_replay_turn(turn)
        if nodes and nodes[name].get('call_id') != turn['invocation_id']:
            raise ValueError('whole-episode model-turn event changed its actual invocation owner')
        capture=turn.get('sampling') or {}
        from .native_occurrences import graph_binding
        graph_binding(name,capture,nodes)
    return names, nodes


REASONING_SUFFIX = ':reasoning'


def action_reasoning_capture(turn, think_close):
    """A reasoned decision's action reasoning as an ordinary sampled capture, or None. The action entry of a reasoned
    readout (``decision_reasoning.action``, with its sampling receipt) keeps its reasoning generation's exact capture
    (``decision_reasoning.reasoning_sampling``: rl.task_rollouts.rollout_turns, ts-host tune/rollouts.ts). Its reasoning
    is part of the sampled action: the capture is kept through the first ``think_close`` (``</think>``) included; the
    free reply after it was never used. A capture that writes state, records a protocol failure or holds compact
    columns is not cut (None), nor is any capture without ``think_close``."""
    marker = turn.get('decision_reasoning')
    if think_close is None or not isinstance(marker, dict) or marker.get('action') is not True:
        return None
    capture = marker.get('reasoning_sampling')
    if (not isinstance(capture, dict) or not (turn.get('sampling') or {}).get('decision_sample')
            or not isinstance(capture.get('samples'), list) or not capture['samples'] or capture.get('writes')
            or capture.get('protocol_failure') or 'samples_columns' in capture):
        return None
    ids = [sample.get('token_id') for sample in capture['samples']]
    stop = ids.index(think_close) + 1 if think_close in ids else len(ids)
    inputs, kept_inputs, seen_sample = capture.get('generated_inputs') or [], [], False
    for item in inputs:
        index = item.get('sample_index')
        if index is None and seen_sample:
            return None                   # a forced input after a sampled one: not a plain generation
        if index is not None:
            seen_sample = True
            if index >= stop:
                continue
        kept_inputs.append(item)
    return {**capture, 'samples': capture['samples'][:stop], 'generated_inputs': kept_inputs}


def whole_episode_turns(rollout, think_close=None):
    """Authenticate all captures and preserve explicit external-vs-local origins.

    Captures are returned as independent copies. They are JSON trees (no shared or cyclic references), so a
    pickle round trip is an exact deep copy, several times faster than copy.deepcopy on per-token captures.

    ``think_close`` (the ``</think>`` id; None: off) adds each reasoned decision's action reasoning
    (``action_reasoning_capture``) as one more sampled turn, ``graph_turn`` the action's with ``REASONING_SUFFIX``,
    right before its action entry; its state reads bind as the action entry's do (one readout, one prompt).
    ``native_replay.compile_replay`` counts its node among the action turn's nodes."""
    import pickle
    def copy_tree(value):
        return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    names, nodes = validate_rollout_graph(rollout)
    result = []
    for name, turn in zip(names, rollout['turns']):
        from .native_occurrences import graph_binding
        binding=graph_binding(name,turn.get('sampling') or {},nodes)
        captured_writers={event['node'] for event in nodes.values() if event.get('kind')=='block_write'}
        outside={block:node for block,node in binding['read_bindings'].items() if node not in captured_writers}
        binding['read_bindings']={block:node for block,node in binding['read_bindings'].items() if block not in outside}
        reasoning = action_reasoning_capture(turn, think_close)
        if reasoning is not None:
            own = graph_binding(name, reasoning, nodes)
            if own.get('written_nodes') or own.get('text_result_writes') or own.get('written_occurrences'):
                raise ValueError('a reasoned decision\'s reasoning cannot write native state')
            own['graph_turn'] = name + REASONING_SUFFIX
            result.append({'messages': turn['messages'], 'tools': None,
                           'reply': {'role': 'assistant', 'content': '',
                                     'reasoning_content': (turn.get('target') or {}).get('reasoning_content') or ''},
                           **own, 'external_read_bindings': {}, 'sampling': copy_tree(reasoning)})
        result.append({'messages': turn['messages'], 'tools': turn.get('tools'), 'reply': turn['target'],
            **({'response_tools':copy_tree(turn['response_tools']), 'native_xml_turn':copy_tree(turn['native_xml_turn'])} if turn.get('native_xml_turn') else {}),
            **binding,'external_read_bindings':outside,'sampling':copy_tree(turn.get('sampling') or {})})
    return result
