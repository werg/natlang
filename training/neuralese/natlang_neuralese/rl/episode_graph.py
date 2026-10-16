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


def validate_import_gradient_uses(name, capture, nodes):
    """Compatibility entry point for the shared exact occurrence verifier."""
    from .native_occurrences import capture_occurrences
    capture_occurrences(name,capture,nodes)


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
        if nodes and nodes[name].get('call_id') != turn['invocation_id']:
            raise ValueError('whole-episode model-turn event changed its actual invocation owner')
        capture=turn.get('sampling') or {}
        from .native_occurrences import graph_binding
        graph_binding(name,capture,nodes)
    return names, nodes


def whole_episode_turns(rollout):
    """Authenticate all captures and preserve explicit external-vs-local origins."""
    import copy
    names, nodes = validate_rollout_graph(rollout)
    result = []
    for name, turn in zip(names, rollout['turns']):
        from .native_occurrences import graph_binding
        binding=graph_binding(name,turn.get('sampling') or {},nodes)
        captured_writers={event['node'] for event in nodes.values() if event.get('kind')=='block_write'}
        outside={block:node for block,node in binding['read_bindings'].items() if node not in captured_writers}
        binding['read_bindings']={block:node for block,node in binding['read_bindings'].items() if block not in outside}
        result.append({'messages': turn['messages'], 'tools': turn.get('tools'), 'reply': turn['target'],
            **binding,'external_read_bindings':outside,'sampling':copy.deepcopy(turn.get('sampling') or {})})
    return result
