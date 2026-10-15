"""Exact ordinary-prefix reuse through the shared positional state graph.

This changes physical work only. Each example retains its original loss rows,
normalization and exposure. Authenticated communication and policy objectives
keep their existing layouts; a text match alone never aliases their states.
"""
from ..model.native_graph_plan import Invocation, ImportSpan, compile_graph


def pack_causal_prefixes(examples):
    """Return (plan, original-token physical row maps), or no applicable reuse."""
    if len(examples) < 2:
        return None
    nodes = []
    for example in examples:
        if example.plan.external_names or len(example.plan.invocations) != 1:
            return None
        node = example.plan.invocations[0]
        if node.imports or node.producing or node.logical_producing or node.cache_prefix:
            return None
        nodes.append(node)
    root = {'children': {}, 'ends': []}
    for index, node in enumerate(nodes):
        cursor = root
        for token in node.token_ids:
            cursor = cursor['children'].setdefault(token, {'children': {}, 'ends': []})
        cursor['ends'].append(index)
    invocations, physical_rows = [], [None] * len(nodes)
    stack = [(token, subtree, None, ()) for token, subtree in reversed(tuple(root['children'].items()))]
    physical = 0
    while stack:
        token, subtree, parent, prefix_rows = stack.pop()
        segment = [token]
        while not subtree['ends'] and len(subtree['children']) == 1:
            token, subtree = next(iter(subtree['children'].items()))
            segment.append(token)
        name = f'causal-prefix/{len(invocations)}'
        imports = () if parent is None else (ImportSpan(at=0, source=parent,
            start=0, stop=len(prefix_rows), logical=True),)
        invocations.append(Invocation(name, tuple(segment), imports, cache_prefix=True))
        rows = prefix_rows + tuple(range(physical, physical + len(segment)))
        physical += len(segment)
        for index in subtree['ends']:
            physical_rows[index] = rows
        stack.extend((token, child, name, rows)
                     for token, child in reversed(tuple(subtree['children'].items())))
    if physical >= sum(len(node.token_ids) for node in nodes):
        return None
    if any(rows is None or len(rows) != len(node.token_ids) for rows, node in zip(physical_rows, nodes)):
        raise AssertionError('causal prefix packing lost an original token query')
    plan = compile_graph(invocations)
    if any(plan.producing):
        raise AssertionError('ordinary prefix caching must not activate Neuralese embedding adaptation')
    return plan, tuple(physical_rows)
