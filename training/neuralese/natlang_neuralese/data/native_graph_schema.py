"""Stable native graph schema admission; shared by trainers and semantic pairing."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from ..model.native_graph_plan import Invocation, ImportSpan, compile_graph

SCHEMA = 'natlang.neuralese-native-graph/1'
COMMUNICATION_EDGE = 'communication'
INSTRUCTION_EDGE = 'instruction'


def edge_kind_of(writer_source_kind, declared=None):
    """The kind of a producer->consumer source edge: `instruction` for an inline `nl` template body (a constant
    instruction the compiler turned into a writer/read pair), else `communication` (a child's actual value).

    The converter declares it on the occurrence/edge; rows converted before it did declare it carry
    `source_kind: 'inline-instruction'`. Contradicting declarations fail closed.
    """
    derived = INSTRUCTION_EDGE if writer_source_kind == 'inline-instruction' else COMMUNICATION_EDGE
    if declared is not None and declared not in (COMMUNICATION_EDGE, INSTRUCTION_EDGE):
        raise ValueError('unknown source edge kind: ' + repr(declared))
    if declared is not None and declared != derived:
        raise ValueError(f'source edge kind {declared!r} contradicts its writer source kind {writer_source_kind!r}')
    return derived


@dataclass
class NativeGraphExample:
    id: str
    split: str
    source_groups: tuple[str, ...]
    tokenizer_fingerprint: str
    plan: object
    target_weights: tuple[float, ...]
    external_states: dict = field(default_factory=dict)
    # Finite decision readouts over sibling option nodes (data.decision_graph.GraphDecision).
    decisions: tuple = ()
    # Learned Neuralese operator positions (model/learned_operators.py), as declared by the row.
    operators: tuple = ()
    # (node name, index into that node's imports) of every import the row declares as `instruction`: the real
    # call's constant instruction text carried as soft-instruction context, not communication (S2, 2026-11-03).
    instruction_imports: tuple = ()


def _graph_fields(row, *, tokenizer_fingerprint):
    """Validate actual token supervision and graph references before any update."""
    if row.get('schema') != SCHEMA:
        raise ValueError('native graph schema is required; legacy projected records need explicit conversion')
    if row.get('tokenizer_fingerprint') != tokenizer_fingerprint:
        raise ValueError('native graph tokenizer differs from the expanded model tokenizer')
    if not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError('native graph needs a source identity')
    if row.get('split') not in {'train', 'test'}:
        raise ValueError('native graph split must be explicit train or test')
    groups = row.get('source_groups')
    if not isinstance(groups, list) or not groups or any(not isinstance(group, str) or not group for group in groups):
        raise ValueError('native graph needs explicit source groups')
    nodes, weights = [], []
    for node in row.get('nodes') or []:
        ids, supervision = node.get('token_ids'), node.get('target_weights')
        if (not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids)
                or not isinstance(supervision, list) or len(supervision) != len(ids)
                or any(type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0
                       for weight in supervision)):
            raise ValueError('native node needs nonnegative token IDs and matching finite target weights')
        if ids and supervision[0] != 0:
            raise ValueError('the first own token has no own predecessor query and cannot be supervised')
        for edge in node.get('imports', []):
            if 'instruction' in edge and edge['instruction'] is not True:
                raise ValueError('an import declares `instruction` only as true; communication imports omit it')
        imports = tuple(ImportSpan(**{key: value for key, value in edge.items() if key != 'instruction'})
                        for edge in node.get('imports', []))
        for edge in imports:
            if edge.at < len(ids) and supervision[edge.at] != 0:
                raise ValueError('a forced token after an imported state has no consumer predecessor query')
        nodes.append(Invocation(name=node['name'], token_ids=tuple(ids), imports=imports,
            producing=tuple(tuple(span) for span in node.get('producing', [])),
            logical_producing=tuple(tuple(span) for span in node.get('logical_producing', []))))
        weights.extend(float(weight) for weight in supervision)
    external = row.get('external_states') or {}
    if (not isinstance(external, dict) or any(not isinstance(name, str) or not name
            or not isinstance(binding, dict)
            or not isinstance(binding.get('id'), str) or not re.fullmatch(r'nzs1_[a-z2-7]{52}', binding['id'])
            or type(binding.get('token_count')) is not int or binding['token_count'] < 0
            for name, binding in external.items())):
        raise ValueError('native graph external states need exact content IDs and nonnegative token counts')
    return nodes, tuple(weights), external


def _instruction_imports(row):
    return tuple((node['name'], index) for node in row.get('nodes') or []
                 for index, edge in enumerate(node.get('imports', [])) if edge.get('instruction') is True)


def parse_graph(row, *, tokenizer_fingerprint):
    """Validate and compile a graph for actual execution."""
    nodes, weights, external = _graph_fields(row, tokenizer_fingerprint=tokenizer_fingerprint)
    from .decision_graph import parse_decisions
    decisions = parse_decisions(row, nodes, weights)
    plan = compile_graph(nodes, external_sizes={name: binding['token_count'] for name, binding in external.items()})
    operators = row.get('operators') or []
    if not isinstance(operators, list) or any(not isinstance(entry, dict) for entry in operators):
        raise ValueError('native graph operators must be a list of declared spans')
    return NativeGraphExample(row['id'], row['split'], tuple(row['source_groups']),
                              tokenizer_fingerprint, plan, weights, external, decisions,
                              tuple(dict(entry) for entry in operators), _instruction_imports(row))


def indexed_graph_metadata(row, *, tokenizer_fingerprint):
    """Normalize an already authenticated indexed row without expanding KV positions.

    This is only a cache-migration seam: callers must compare its exact row binding
    with the previously validated execution index before using any classification.
    It deliberately does not grant graph admission or replace compile_graph.
    """
    from types import SimpleNamespace
    nodes, weights, external = _graph_fields(row, tokenizer_fingerprint=tokenizer_fingerprint)
    from .decision_graph import parse_decisions
    decisions = parse_decisions(row, nodes, weights)
    plan = SimpleNamespace(invocations=tuple(nodes), external_names=tuple(external),
                           external_sizes=tuple(value['token_count'] for value in external.values()))
    return NativeGraphExample(row['id'], row['split'], tuple(row['source_groups']),
                              tokenizer_fingerprint, plan, weights, external, decisions,
                              instruction_imports=_instruction_imports(row))


def communication_classification(example):
    """Classify actual supervised state consumption, independent of source format.

    A local producer and consumer qualify separately from imported external state.
    Merely containing markers or a forced token does not train consumption. An import the row declares as an
    `instruction` (constant instruction text imported as soft-instruction context, e.g. an inline `nl` template
    body) is ordinary supervised context, not communication: a graph whose only imports are instruction imports
    is not a communication graph and earns no communication quota or exposure.
    """
    internal = external = False
    names = {node.name for node in example.plan.invocations}
    instruction = set(getattr(example, 'instruction_imports', ()))
    # A decision readout's option continuations are supervised over the option set (data.decision_graph).
    decided = {option for decision in getattr(example, 'decisions', ()) for option in decision.options}
    offset = 0
    for node in example.plan.invocations:
        weights = example.target_weights[offset:offset + len(node.token_ids)]
        for index, edge in enumerate(node.imports):
            if (node.name, index) in instruction:
                continue
            if edge.stop <= edge.start or not (node.name in decided or any(weight > 0 for weight in weights[edge.at + 1:])):
                continue
            if edge.source in names:
                internal = True
            elif edge.source in example.external_states:
                external = True
        offset += len(node.token_ids)
    return {'communication': internal or external, 'producer_consumer': internal,
            'external_state_consumption': external}


def communication_payload_digests(example):
    """Sorted SHA-256 digests of the distinct token payloads that actually supervised communication imports carry.

    Identical payloads are identified by their exact producer token slice. Instruction imports and imports of
    external states or logical coordinates have no such digest here; a graph whose communication payload cannot
    be identified returns None (unknown coverage, never an empty claim).
    """
    import hashlib
    instruction = set(getattr(example, 'instruction_imports', ()))
    nodes = {node.name: node for node in example.plan.invocations}
    digests, unknown, offset = set(), False, 0
    for node in example.plan.invocations:
        weights = example.target_weights[offset:offset + len(node.token_ids)]
        offset += len(node.token_ids)
        for index, edge in enumerate(node.imports):
            if ((node.name, index) in instruction or edge.stop <= edge.start
                    or not any(weight > 0 for weight in weights[edge.at + 1:])):
                continue
            source = nodes.get(edge.source)
            if source is None or edge.logical:
                unknown = True
                continue
            payload = source.token_ids[edge.start:edge.stop]
            digests.add(hashlib.sha256(','.join(map(str, payload)).encode()).hexdigest())
    return None if unknown or not digests else sorted(digests)
