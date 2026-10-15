"""Stable native graph schema admission; shared by trainers and semantic pairing."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from ..model.native_graph_plan import Invocation, ImportSpan, compile_graph

SCHEMA = 'natlang.neuralese-native-graph/1'


@dataclass
class NativeGraphExample:
    id: str
    split: str
    source_groups: tuple[str, ...]
    tokenizer_fingerprint: str
    plan: object
    target_weights: tuple[float, ...]
    external_states: dict = field(default_factory=dict)


def parse_graph(row, *, tokenizer_fingerprint):
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
        imports = tuple(ImportSpan(**edge) for edge in node.get('imports', []))
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
    plan = compile_graph(nodes, external_sizes={name: binding['token_count'] for name, binding in external.items()})
    return NativeGraphExample(row['id'], row['split'], tuple(groups), tokenizer_fingerprint,
                              plan, tuple(weights), external)


