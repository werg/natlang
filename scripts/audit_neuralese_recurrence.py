#!/usr/bin/env python3
"""Audit producer closure and whole-graph split isolation without loading contexts.

Equal text is not evidence of invocation identity. Ambiguous producers are held.
This is a structural gate; correctness and causal-use evaluation remain separate.
"""
import argparse
import collections
import json
from pathlib import Path


def names(value, kind):
    out = set()
    if isinstance(value, dict):
        if kind == 'read' and value.get('type') == 'read': out.add(value['name'])
        if kind == 'write' and isinstance(value.get('$write'), dict): out.add(value['$write']['name'])
        for k,v in value.items():
            if k == 'arguments' and isinstance(v, str):
                try: out |= names(json.loads(v), kind)
                except ValueError: pass
            else: out |= names(v, kind)
    elif isinstance(value, list):
        for v in value: out |= names(v, kind)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    rows, producers = {}, collections.defaultdict(list)
    for path in a.inputs:
        with path.open() as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row['id'] in rows: raise ValueError('duplicate record id: ' + row['id'])
                target = names(row.get('target'), 'write')
                rows[row['id']] = {'reads': names(row.get('messages'), 'read') - target,
                    'writes': target, 'split': row.get('split', 'unspecified'),
                    'source_groups': row.get('source_groups', []),
                    'run': row.get('source_ref', {}).get('trajectory_id')}
                for name in target: producers[name].append(row['id'])
    missing, ambiguous, cross_split, cycles = [], [], [], []
    adjacency = collections.defaultdict(set)
    for ident,row in rows.items():
        for name in row['reads']:
            ps = producers.get(name, [])
            if not ps: missing.append({'reader': ident, 'name': name}); continue
            if len(ps) != 1: ambiguous.append({'reader': ident, 'name': name, 'producers': ps}); continue
            adjacency[ident].add(ps[0])
            if row['split'] != rows[ps[0]]['split']:
                cross_split.append({'reader': ident, 'producer': ps[0], 'name': name})
    depths = {}
    def depth(node, visiting):
        if node in visiting:
            cycles.append(node); return 0
        if node not in depths:
            depths[node] = max([1 + depth(child, visiting | {node}) for child in adjacency[node]] or [0])
        return depths[node]
    for ident in rows: depth(ident, set())
    group_splits = collections.defaultdict(set)
    for row in rows.values():
        for group in row['source_groups']: group_splits[group].add(row['split'])
        if row['run']: group_splits['run:' + str(row['run'])].add(row['split'])
    mixed = {k: sorted(v) for k,v in group_splits.items() if len(v) > 1}
    consumers = collections.defaultdict(set)
    for reader, sources in adjacency.items():
        for producer in sources: consumers[producer].add(reader)
    summary = {'schema': 'natlang.recurrence-audit/2', 'records': len(rows), 'writer_records': sum(bool(r['writes']) for r in rows.values()),
        'reader_records': sum(bool(r['reads']) for r in rows.values()), 'linked_edges': sum(map(len, adjacency.values())),
        'depth_histogram': dict(sorted(collections.Counter(depths.values()).items())),
        'max_producers_per_consumer': max(map(len, adjacency.values()), default=0),
        'max_consumers_per_producer': max(map(len, consumers.values()), default=0),
        'degree_scope': 'distinct decision records; context rereads count as consumers, not new runtime invocations',
        'failures': {'missing_producers': missing, 'ambiguous_producers': ambiguous, 'cross_split_edges': cross_split,
                     'cycles': sorted(set(cycles)), 'mixed_split_source_groups': mixed},
        'structurally_closed': not any((missing, ambiguous, cross_split, cycles, mixed)),
        'admission': 'structural-audit-only; still requires source/quality/causal-use checks'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: ({n:len(v) for n,v in x.items()} if k == 'failures' else x) for k,x in summary.items()}, indent=2))

if __name__ == '__main__': main()
