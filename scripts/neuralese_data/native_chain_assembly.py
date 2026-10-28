"""Assemble jointly rendered source decisions into one positional state graph.

Inputs are one already authenticated, jointly rendered token row per source
decision and exact occurrence spans. This does not render or admit sources.
"""
from __future__ import annotations

import hashlib
import json
from natlang_neuralese.data.native_graph_schema import parse_graph
from scripts.neuralese_data.native_chain_sources import bind_chain
from scripts.neuralese_data.build_native_semantic_pairs_e2e import source_occurrences, writer_payload


def assemble(chain, records, edges, rendered, *, tokenizer, tokenizer_fingerprint, role_ids, source_binding):
    custody = bind_chain(chain, records, edges)
    if set(rendered) != set(custody['records']):
        raise ValueError('joint rendering must cover every source decision exactly')
    if len(role_ids) != 4 or len(set(role_ids)) != 4:
        raise ValueError('four distinct role marker IDs are required')
    p_open, p_close, c_open, c_close = role_ids
    names = {rid: f'n{index}' for index, rid in enumerate(custody['records'])}
    edge_by_sha = {hashlib.sha256(json.dumps(e, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest(): e for e in edges}
    expected_occurrences, writer_occurrences = {}, {}
    for edge_id, edge in edge_by_sha.items():
        writer, reader = records[edge['writer_record_id']], records[edge['reader_record_id']]
        body = writer_payload(writer, edge['write_name'], edge['body_sha256'])
        occurrence = source_occurrences(writer, reader, edge['write_name'], body, edge['body_sha256'],
            writer_occurrence_id=edge.get('writer_occurrence_id'), reader_occurrence_id=edge.get('reader_occurrence_id'))
        writer_occurrences[edge_id] = occurrence['writer_occurrence_id']
        ids, pointers = occurrence['reader_occurrence_ids'], occurrence['reader_json_pointers']
        if (not isinstance(writer_occurrences[edge_id], str) or not writer_occurrences[edge_id]
                or not ids or len(ids) != len(pointers) or len(set(ids)) != len(ids)
                or any(not isinstance(value, str) or not value for value in ids + pointers)):
            raise ValueError('reader occurrence custody is incomplete or ambiguous')
        expected_occurrences.update({(edge_id, x): pointer for x, pointer in zip(ids, pointers)})
    seen_occurrences, covered_edges = set(), set()
    nodes, boundaries, pending, suffixes = [], {}, [], []
    for rid in custody['records']:
        row = rendered[rid]
        digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()
        if (row.get('source_record_sha256') != digest(records[rid])
                or row.get('source_target_sha256') != digest(records[rid]['target'])):
            raise ValueError('joint rendering lacks exact source record and complete target bindings')
        ids, weights = row['token_ids'], row['target_weights']
        if len(ids) != len(weights):
            raise ValueError('rendered token and weight lengths differ')
        spans = sorted(row.get('reads', []), key=lambda s: s['start'])
        previous = 0
        for span in spans:
            start, stop = span['start'], span['stop']
            if (type(start) is not int or type(stop) is not int or not 0 < start < stop < len(ids)
                    or start < previous or ids[start - 1] != c_open or ids[stop] != c_close):
                raise ValueError('consumer occurrence is invalid, overlapping, or lacks exact markers')
            previous = stop
        remap = [None] * (len(ids) + 1)
        new_ids, new_weights, at_by_span = [], [], []
        i = 0
        for span in spans:
            start, stop = span['start'], span['stop']
            while i < start:
                remap[i] = len(new_ids); new_ids.append(ids[i]); new_weights.append(weights[i]); i += 1
            remap[start] = len(new_ids)
            at_by_span.append((span, len(new_ids), ids[start:stop]))
            for position in range(start + 1, stop): remap[position] = None
            i = stop
            remap[i] = len(new_ids); new_ids.append(ids[i]); new_weights.append(0.0); i += 1
        while i < len(ids):
            remap[i] = len(new_ids); new_ids.append(ids[i]); new_weights.append(weights[i]); i += 1
        remap[len(ids)] = len(new_ids)
        suffix = row.get('supervised_suffix_start')
        if type(suffix) is not int or not 0 <= suffix <= len(ids) or remap[suffix] is None:
            raise ValueError('joint source target suffix is missing or inside an imported payload')
        suffixes.append(remap[suffix])
        producing = []
        for start, stop in row.get('producing', []):
            if (not 0 < start < stop < len(ids) or ids[start-1] != p_open or ids[stop] != p_close
                    or remap[start] is None or remap[stop] is None
                    or any(remap[j] is None for j in range(start, stop))):
                raise ValueError('producer span changed or overlaps an imported payload')
            producing.append([remap[start], remap[stop]])
        boundaries[rid] = remap
        nodes.append({'name': names[rid], 'token_ids': new_ids, 'target_weights': new_weights,
                      'producing': producing, 'imports': []})
        pending.extend((rid, *item) for item in at_by_span)
    by_id = dict(zip(custody['records'], nodes))
    for rid, read, at, payload in pending:
        source = read['source_record_id']
        edge_id, occurrence = read.get('edge_sha256'), read.get('occurrence_id')
        edge = edge_by_sha.get(edge_id)
        if (not edge or edge['writer_record_id'] != source or edge['reader_record_id'] != rid
                or not isinstance(occurrence, str) or not occurrence
                or (edge_id, occurrence) in seen_occurrences
                or expected_occurrences.get((edge_id, occurrence)) != read.get('source_pointer')):
            raise ValueError('consumer occurrence lacks an exact unique source edge witness')
        seen_occurrences.add((edge_id, occurrence)); covered_edges.add(edge_id)
        if hashlib.sha256(tokenizer.decode(payload, skip_special_tokens=False).encode()).hexdigest() != edge['body_sha256']:
            raise ValueError('consumer bytes differ from the source edge payload')
        start, stop = read['source_start'], read['source_stop']
        if source not in by_id or start >= stop:
            raise ValueError('consumer references an absent producer span')
        declared = rendered[source].get('producer_bindings') or []
        if not any(x.get('edge_sha256') == edge_id and x.get('writer_occurrence_id') == writer_occurrences[edge_id]
                   and x.get('start') == start and x.get('stop') == stop for x in declared):
            raise ValueError('producer span lacks the exact authenticated writer occurrence')
        mapped_start, mapped_stop = boundaries[source][start], boundaries[source][stop]
        producer = by_id[source]
        if ([mapped_start, mapped_stop] not in producer['producing']
                or producer['token_ids'][mapped_start:mapped_stop] != payload):
            raise ValueError('consumer payload differs from its authenticated producer span')
        by_id[rid]['imports'].append({'at': at, 'source': names[source],
                                       'start': mapped_start, 'stop': mapped_stop})
    if covered_edges != set(edge_by_sha) or seen_occurrences != set(expected_occurrences):
        raise ValueError('joint graph does not cover every authenticated source occurrence')
    graph = {'schema': 'natlang.neuralese-native-graph/1',
             'id': f"chain:{custody['source']}:{custody['trajectory']}", 'split': custody['split'],
             'source_groups': sorted({g for record in records.values() for g in record['source_groups']}),
             'tokenizer_fingerprint': tokenizer_fingerprint, 'nodes': nodes,
             'source_chain': {'custody': custody, 'source_binding': source_binding, 'source_suffixes': suffixes,
                              'record_bindings': {rid: {'source_record_sha256': rendered[rid]['source_record_sha256'],
                                  'source_target_sha256': rendered[rid]['source_target_sha256'],
                                  'source_ref': records[rid].get('source_ref'),
                                  'provenance': records[rid].get('provenance')} for rid in custody['records']},
                              'producer_bindings': {rid: rendered[rid].get('producer_bindings', []) for rid in custody['records']},
                              'consumer_bindings': {rid: rendered[rid].get('reads', []) for rid in custody['records']}}}
    parse_graph(graph, tokenizer_fingerprint=tokenizer_fingerprint)
    return graph
