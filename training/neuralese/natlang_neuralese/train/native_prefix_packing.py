"""Exact initial raw-context reuse through the shared positional state graph.

Each example retains its loss rows, normalization and exposure. Communication
aliases stop before adaptation/imports; producing suffixes and actual edges stay
complete. External-state and policy-objective routing retain their prior path.
"""
from ..model.native_graph_plan import Invocation, ImportSpan, compile_graph


def pack_causal_prefixes(examples, *, allow_communication=True):
    """Return (plan, original-token physical row maps), or no applicable reuse."""
    if not examples:
        return None
    if any(example.plan.external_names for example in examples):
        return None
    nodes = []
    for example in examples:
        if len(example.plan.invocations) != 1:
            return _pack_initial_raw_contexts(examples) if allow_communication else None
        node = example.plan.invocations[0]
        if node.imports or node.producing or node.logical_producing or node.cache_prefix:
            return _pack_initial_raw_contexts(examples) if allow_communication else None
        nodes.append(node)
    if len(examples) < 2:
        return None
    invocations, physical_rows, _ = _raw_prefix_parts([node.token_ids for node in nodes])
    if sum(len(node.token_ids) for node in invocations) >= sum(len(node.token_ids) for node in nodes):
        return None
    plan = compile_graph(invocations)
    if any(plan.producing):
        raise AssertionError('ordinary prefix caching must not activate Neuralese embedding adaptation')
    return plan, tuple(physical_rows)


def _raw_prefix_parts(sequences):
    root = {'children': {}, 'ends': []}
    for index, node in enumerate(sequences):
        cursor = root
        for token in node:
            cursor = cursor['children'].setdefault(token, {'children': {}, 'ends': []})
        cursor['ends'].append(index)
    invocations, physical_rows = [], [() if not sequence else None for sequence in sequences]
    terminals = [None] * len(sequences)
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
            terminals[index] = name
        stack.extend((token, child, name, rows)
                     for token, child in reversed(tuple(subtree['children'].items())))
    if any(rows is None or len(rows) != len(sequence) for rows, sequence in zip(physical_rows, sequences)):
        raise AssertionError('raw prefix packing lost an original token query')
    return invocations, tuple(physical_rows), tuple(terminals)


def _pack_initial_raw_contexts(examples):
    """Alias only pre-event raw context; retain every real episode dependency."""
    if any(node.cache_prefix for example in examples for node in example.plan.invocations):
        return None  # Already compiled cache graphs retain their existing layout.
    prefixes, metadata = [], []
    for graph_index, example in enumerate(examples):
        for node_index, node in enumerate(example.plan.invocations):
            split = min([len(node.token_ids), *[edge.at for edge in node.imports],
                *[start for start, _ in node.producing],
                *[start for start, _ in node.logical_producing]])
            prefixes.append(node.token_ids[:split])
            metadata.append((graph_index, node_index, split))
    invocations, prefix_rows, terminals = _raw_prefix_parts(prefixes)
    physical = sum(len(node.token_ids) for node in invocations)
    original = sum(example.plan.token_count for example in examples)
    if physical + sum(len(examples[g].plan.invocations[n].token_ids)-split
                      for g,n,split in metadata) >= original:
        return None
    names = [{node.name:f'graph/{g}/{n}:{node.name}' for n,node in enumerate(example.plan.invocations)}
             for g,example in enumerate(examples)]
    rows = [[] for _ in examples]
    for index, (g,n,split) in enumerate(metadata):
        example = examples[g]
        node = example.plan.invocations[n]
        suffix = node.token_ids[split:]
        rows[g].extend(prefix_rows[index])
        rows[g].extend(range(physical,physical+len(suffix)))
        if not suffix and not (node.imports or node.producing or node.logical_producing):
            continue
        imports = ([] if not split else [ImportSpan(at=0,source=terminals[index],
            start=0,stop=split,logical=True)])
        source_indices = {source.name:i for i,source in enumerate(example.plan.invocations)}
        for edge in node.imports:
            source = source_indices[edge.source]
            lo,hi = (edge.start,edge.stop) if edge.logical else (
                example.plan.boundaries[source][edge.start],example.plan.boundaries[source][edge.stop])
            imports.append(ImportSpan(at=edge.at-split,source=names[g][edge.source],
                start=lo,stop=hi,logical=True,stop_gradient=edge.stop_gradient))
        boundaries = example.plan.boundaries[n]
        exports = (*node.logical_producing,*[(boundaries[start],boundaries[stop])
                                            for start,stop in node.producing])
        invocations.append(Invocation(names[g][node.name],suffix,tuple(imports),
                                      logical_producing=tuple(exports)))
        physical += len(suffix)
    plan = compile_graph(invocations)
    for example,mapping in zip(examples,rows):
        if (len(mapping) != example.plan.token_count
                or any(plan.own_positions[new] != example.plan.own_positions[old]
                       or plan.producing[new] != example.plan.producing[old]
                       for old,new in enumerate(mapping))):
            raise AssertionError('communication prefix packing changed own positions/producer flags')
    return plan,tuple(tuple(mapping) for mapping in rows)
