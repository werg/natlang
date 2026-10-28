"""Authenticate a whole recorded trajectory before rendering its state graph.

This is a custody check only; a checked chain is not an exported graph.
"""
from __future__ import annotations

import hashlib
import json
import re


def bind_chain(chain, records, edges):
    ids = chain.get('records')
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not x for x in ids)):
        raise ValueError('chain requires distinct ordered source record IDs')
    if set(records) != set(ids) or len(records) != len(ids):
        raise ValueError('chain record inventory differs from exact source custody')
    if any(records[x].get('id') != x for x in ids):
        raise ValueError('source record key differs from its actual identity')
    split = chain.get('split')
    if split not in ('train', 'test') or any(records[x].get('split') != split for x in ids):
        raise ValueError('chain and all source records require one explicit split')
    if not chain.get('source') or not chain.get('trajectory'):
        raise ValueError('chain source and trajectory are required')
    if any((records[x].get('source_ref') or {}).get('trajectory_id') != chain['trajectory'] for x in ids):
        raise ValueError('source records do not bind to the exact chain trajectory')
    order = {name: n for n, name in enumerate(ids)}
    seen = set()
    for edge in edges:
        if (edge.get('source') != chain['source'] or edge.get('trajectory') != chain['trajectory']
                or edge.get('split') != split or edge.get('writer_split') != split):
            raise ValueError('edge differs from the exact chain source or split')
        w, r = edge.get('writer_record_id'), edge.get('reader_record_id')
        if w not in order or r not in order or order[w] >= order[r]:
            raise ValueError('edge endpoints must follow the authenticated source order')
        if not edge.get('write_name') or not re.fullmatch('[0-9a-f]{64}', str(edge.get('body_sha256', ''))):
            raise ValueError('edge needs its authenticated payload identity')
        digest = hashlib.sha256(json.dumps(edge, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        if digest in seen:
            raise ValueError('duplicate communication edge')
        seen.add(digest)
    if chain.get('edges') != len(edges):
        raise ValueError('edge inventory is not the source chain complete set')
    return {'source': chain['source'], 'trajectory': chain['trajectory'], 'split': split,
            'records': tuple(ids), 'edge_sha256': tuple(sorted(seen))}
