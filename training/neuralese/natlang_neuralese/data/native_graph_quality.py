"""Source-authenticated whole-graph withdrawal and exact target masking.

These transformations operate on already rendered native graphs.  They do not
infer which targets are wrong: callers must provide a reviewed, exact binding
and token span.  The returned receipts bind both the source graph and the
transformed graph so downstream source builders can preserve their lineage.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

from .native_graph_schema import parse_graph
from .native_sources import _row_bindings


def _canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _binding(row: dict, tokenizer_fingerprint: str) -> str:
    parsed = parse_graph(row, tokenizer_fingerprint=tokenizer_fingerprint)
    return _row_bindings([parsed])[row['id']]


def mask_exact_target_spans(row: dict, *, tokenizer_fingerprint: str,
                            expected_id: str, expected_binding: str,
                            spans: Sequence[Mapping]) -> tuple[dict, dict]:
    """Zero only reviewed token ranges from a graph with the expected binding.

    Each span has ``node``, half-open ``token_start``/``token_stop``,
    ``expected_positive_tokens`` and ``expected_token_ids_sha256``.  The latter
    prevents a stale offset from silently masking a different action after a
    source change.  All graph topology and forward-token invariants are checked
    again after the transformation.
    """
    if not isinstance(row, dict) or row.get('id') != expected_id:
        raise ValueError('target mask row identity differs from reviewed identity')
    before_binding = _binding(row, tokenizer_fingerprint)
    if before_binding != expected_binding:
        raise ValueError('target mask source binding differs from reviewed binding')
    before = parse_graph(row, tokenizer_fingerprint=tokenizer_fingerprint)
    result = copy.deepcopy(row)
    nodes = {node.get('name'): node for node in result.get('nodes', [])}
    applied = []
    seen = set()
    for declaration in spans:
        if not isinstance(declaration, Mapping):
            raise ValueError('target span declaration must be an object')
        name = declaration.get('node')
        start, stop = declaration.get('token_start'), declaration.get('token_stop')
        expected_positive = declaration.get('expected_positive_tokens')
        expected_ids_sha = declaration.get('expected_token_ids_sha256')
        if (name not in nodes or type(start) is not int or type(stop) is not int
                or start < 0 or stop <= start or stop > len(nodes[name].get('token_ids', []))
                or type(expected_positive) is not int or expected_positive < 1
                or not isinstance(expected_ids_sha, str) or len(expected_ids_sha) != 64):
            raise ValueError('target span is incomplete or out of bounds')
        key = (name, start, stop)
        if key in seen:
            raise ValueError('duplicate target span declaration')
        seen.add(key)
        target_node = nodes[name]
        token_ids = target_node['token_ids'][start:stop]
        token_hash = _canonical_sha(token_ids)
        if token_hash != expected_ids_sha:
            raise ValueError('target span token IDs differ from reviewed source bytes')
        weights = target_node['target_weights'][start:stop]
        positive = sum(weight > 0 for weight in weights)
        if positive != expected_positive or positive != stop - start:
            raise ValueError('target span positive-token count differs from review')
        if any(weight <= 0 for weight in weights):
            raise ValueError('mask span includes an already-unsupervised token')
        target_node['target_weights'][start:stop] = [0.0] * (stop - start)
        applied.append({'node': name, 'token_start': start, 'token_stop': stop,
                        'expected_positive_tokens': expected_positive,
                        'token_ids_sha256': token_hash})
    if not applied:
        raise ValueError('target mask requires at least one reviewed span')
    after = parse_graph(result, tokenizer_fingerprint=tokenizer_fingerprint)
    after_binding = _row_bindings([after])[expected_id]
    if after_binding == before_binding:
        raise ValueError('target masking did not change the graph binding')
    before_nodes, after_nodes = before.plan.invocations, after.plan.invocations
    if len(before_nodes) != len(after_nodes) or before.plan.token_count != after.plan.token_count:
        raise ValueError('target mask changed graph topology or forward token count')
    before_shape = [(n.name, [(e.at, e.source, e.start, e.stop) for e in n.imports],
                     n.producing, n.logical_producing) for n in before_nodes]
    after_shape = [(n.name, [(e.at, e.source, e.start, e.stop) for e in n.imports],
                    n.producing, n.logical_producing) for n in after_nodes]
    if before_shape != after_shape:
        raise ValueError('target mask changed producer/consumer graph custody')
    return result, {'schema': 'natlang.native-graph-target-mask/1', 'id': expected_id,
                    'binding_before': before_binding, 'binding_after': after_binding,
                    'row_sha256_before': _canonical_sha(row),
                    'row_sha256_after': _canonical_sha(result),
                    'forward_tokens': before.plan.token_count,
                    'nodes': len(before_nodes), 'spans': applied}


def withdraw_exact_graphs(rows: Iterable[dict], *, tokenizer_fingerprint: str,
                          withdrawals: Mapping[str, str]) -> tuple[list[dict], dict]:
    """Remove reviewed train graphs atomically by exact ID and semantic binding.

    Other rows are passed through unchanged and in their original order.  Every
    requested ID must occur exactly once and match its reviewed graph binding;
    a missing, duplicated, held, or changed row rejects the whole operation.
    """
    if not isinstance(withdrawals, Mapping) or not withdrawals:
        raise ValueError('at least one exact graph withdrawal is required')
    requested = dict(withdrawals)
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or len(v) != 64
           for k, v in requested.items()):
        raise ValueError('withdrawals must map exact graph IDs to semantic bindings')
    kept, removed, counts = [], [], {key: 0 for key in requested}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('source row must be a graph object')
        ident = row.get('id')
        if ident not in requested:
            kept.append(row)
            continue
        counts[ident] += 1
        if counts[ident] != 1:
            raise ValueError('withdrawal graph identity occurs more than once')
        parsed = parse_graph(row, tokenizer_fingerprint=tokenizer_fingerprint)
        if parsed.split != 'train':
            raise ValueError('whole-graph withdrawal only applies to a train graph')
        binding = _row_bindings([parsed])[ident]
        if binding != requested[ident]:
            raise ValueError('withdrawal graph binding differs from reviewed binding')
        removed.append({'id': ident, 'binding': binding, 'row_sha256': _canonical_sha(row),
                        'node_count': len(parsed.plan.invocations),
                        'forward_tokens': parsed.plan.token_count})
    if any(count != 1 for count in counts.values()):
        raise ValueError('each reviewed withdrawal graph must occur exactly once')
    return kept, {'schema': 'natlang.native-whole-graph-withdrawal/1',
                  'removed': removed, 'retained_rows': len(kept)}
