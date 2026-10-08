#!/usr/bin/env python3
"""Audit producer closure and whole-graph split isolation without loading contexts.

Equal text is not evidence of invocation identity. Ambiguous producers are held.
This is a structural gate; correctness and causal-use evaluation remain separate.
"""
import argparse
import collections
import hashlib
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


def authenticated_external_context_names(row):
    """Return same-run typed reads that are evidence-backed context roots.

    These are observed runtime inputs, not selected model writer examples, so
    they must not create synthetic recurrence edges. Require agreement between
    converter metadata and the full provider receipt carried by the source row.
    """
    import re
    refs = row.get('source_ref', {}).get('provider_expanded_read_contexts', [])
    receipts = {}
    for ref in refs if isinstance(refs, list) else []:
        if not isinstance(ref, dict) or ref.get('schema') != 'natlang.provider-expanded-read-context/2':
            continue
        block = ref.get('block')
        producer = ref.get('producer_write')
        read = ref.get('block_read')
        turn = ref.get('model_turn')
        writer_source_class = ref.get('writer_source_class')
        producer_source_valid = (
            (writer_source_class == 'modern-typed-text-result'
             and producer.get('producer') == 'text-marker-emulation'
             and producer.get('source_kind') == 'typed-text-result') if isinstance(producer, dict) else False)
        if isinstance(producer, dict) and writer_source_class == 'legacy-text-marker-standin-eval-code':
            producer_source_valid = (producer.get('emulation_version') == 'text-marker-standin/2'
                                     and producer.get('marker_context') == 'eval-code'
                                     and producer.get('learned_vectors') is False)
        if (ref.get('origin') != 'same-run-producer' or ref.get('writer_target_selected') is not False
                or not isinstance(block, dict) or not isinstance(block.get('id'), str)
                or not isinstance(block.get('body'), str) or not block.get('type')
                or not isinstance(producer, dict) or producer.get('kind') != 'block_write'
                or not producer_source_valid
                or producer.get('block') != block.get('id')
                or producer.get('result_type') != block.get('type')
                or not isinstance(producer.get('call_id'), str) or not producer.get('call_id')
                or not isinstance(producer.get('node'), str) or not producer.get('node')
                or not isinstance(read, dict) or read.get('kind') != 'block_read'
                or read.get('block') != block.get('id') or read.get('call_id') != ref.get('invocation_id')
                or not isinstance(read.get('node'), str) or not read.get('node')
                or not isinstance(turn, dict) or turn.get('kind') != 'model_turn'
                or turn.get('call_id') != ref.get('invocation_id')
                or not isinstance(turn.get('node'), str) or not turn.get('node')
                or read.get('turn') != turn.get('node')
                or not isinstance(ref.get('invocation_id'), str) or not ref.get('invocation_id')
                or ('parent_invocation_id' in ref and ref.get('parent_invocation_id') is not None
                    and not isinstance(ref.get('parent_invocation_id'), str))):
            continue
        body_sha = hashlib.sha256(block['body'].encode()).hexdigest()
        required_hashes = (body_sha, block.get('body_sha256'), ref.get('source_row_sha256'),
                           ref.get('trace_sha256'), ref.get('transport_provenance_sha256'),
                           ref.get('raw_request_sha256'), ref.get('rendered_request_sha256'),
                           producer.get('text_body_sha256'))
        if (any(not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None
                for value in required_hashes)
                or body_sha != block.get('body_sha256')
                or producer.get('text_body_sha256') != body_sha
                or not any(isinstance(edge, dict) and edge.get('node') == producer.get('node')
                           and edge.get('block') == block.get('id') for edge in read.get('inputs', []))
                or not any(isinstance(edge, dict) and edge.get('node') == read.get('node')
                           and edge.get('block') == block.get('id') for edge in turn.get('inputs', []))):
            continue
        receipts[block['id']] = ref

    valid = set()
    conversion = row.get('neuralese_conversion')
    metadata = conversion.get('external_context_inputs', []) if isinstance(conversion, dict) else []
    for item in metadata if isinstance(metadata, list) else []:
        if not isinstance(item, dict) or item.get('schema') != 'natlang.external-context-input/1':
            continue
        if item.get('origin') != 'same-run-producer' or item.get('writer_target_selected') is not False:
            continue
        block_id = item.get('block_id')
        ref = receipts.get(block_id)
        if ref is None:
            continue
        block = ref['block']
        producer = ref.get('producer_write') or {}
        read = ref.get('block_read') or {}
        turn = ref.get('model_turn') or {}
        pairs = [(item.get('type'), block.get('type')),
                 (item.get('body_sha256'), block.get('body_sha256')),
                 (item.get('invocation_id'), ref.get('invocation_id')),
                 (item.get('parent_invocation_id'), ref.get('parent_invocation_id')),
                 (item.get('source_row_sha256'), ref.get('source_row_sha256')),
                 (item.get('trace_sha256'), ref.get('trace_sha256')),
                 (item.get('transport_provenance_sha256'), ref.get('transport_provenance_sha256')),
                 (item.get('raw_request_sha256'), ref.get('raw_request_sha256')),
                 (item.get('rendered_request_sha256'), ref.get('rendered_request_sha256')),
                 (item.get('producer_call_id'), producer.get('call_id')),
                 (item.get('writer_source_class'), writer_source_class),
                 (item.get('producer_write_node'), producer.get('node')),
                 (item.get('read_node'), read.get('node')),
                 (item.get('model_turn_node'), turn.get('node'))]
        if any(left != right for left, right in pairs):
            continue
        if item.get('learner_representation') != 'typed-read-from-authenticated-runtime-writer-event-context-only':
            continue
        if (not isinstance(block_id, str) or not block_id
                or not isinstance(item.get('producer_write_node'), str) or not item.get('producer_write_node')
                or not isinstance(item.get('producer_call_id'), str) or not item.get('producer_call_id')
                or not isinstance(item.get('read_node'), str) or not item.get('read_node')
                or not isinstance(item.get('model_turn_node'), str) or not item.get('model_turn_node')
                or any(not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None
                       for value in (item.get('body_sha256'), item.get('source_row_sha256'),
                                     item.get('trace_sha256'), item.get('transport_provenance_sha256'),
                                     item.get('raw_request_sha256'), item.get('rendered_request_sha256')))):
            continue
        valid.add('soft-state:' + block_id)
    return valid


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
                reads = names(row.get('messages'), 'read') - target
                external_candidates = reads & authenticated_external_context_names(row)
                rows[row['id']] = {'reads': reads, 'external_candidates': external_candidates,
                    'external_context_roots': [],
                    'writes': target, 'split': row.get('split', 'unspecified'),
                    'source_groups': row.get('source_groups', []),
                    'run': row.get('source_ref', {}).get('trajectory_id')}
                for name in target: producers[name].append(row['id'])
    # A selected producer always takes precedence over context-root metadata:
    # the reader then keeps its real cohort edge and its usual split checks.
    for row in rows.values():
        external = {name for name in row['external_candidates'] if not producers.get(name)}
        row['reads'] -= external
        row['external_context_roots'] = sorted(external)
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
    external_roots = [{'reader': ident, 'name': name} for ident,row in rows.items()
                      for name in row['external_context_roots']]
    summary = {'schema': 'natlang.recurrence-audit/3', 'records': len(rows), 'writer_records': sum(bool(r['writes']) for r in rows.values()),
        'reader_records': sum(bool(r['reads']) for r in rows.values()), 'linked_edges': sum(map(len, adjacency.values())),
        'authenticated_external_context_roots': external_roots,
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
