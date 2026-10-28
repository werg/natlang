"""Joint source-bound rendering inputs for full native trajectory graphs."""
from __future__ import annotations

import copy
import hashlib
import json

from natlang_neuralese.common.jsonio import canonical_json_str
from natlang_neuralese.data.text_corpus import gold_text_rows, target_with_teacher_reasoning
from natlang_neuralese.model.shared_state import ROLE_MARKERS
from scripts.neuralese_data.native_chain_sources import bind_chain
from scripts.neuralese_data.build_native_semantic_pairs_e2e import (
    _assistant_text_target, _replace_source_target_span, pair_derivations,
    source_occurrences, writer_payload,
)


def digest(value):
    return hashlib.sha256(canonical_json_str(value).encode()).hexdigest()


def edge_digest(edge):
    return hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def render_chain(chain, records, edges, pieces, *, tokenizer, tokenizer_fingerprint):
    """Render each complete source decision once, with every witnessed write and read.

    Returned rows are not yet native graphs: role marker materialization and
    import-position assembly must retain these exact occurrence bindings.
    """
    custody = bind_chain(chain, records, edges)
    targets = {rid: copy.deepcopy(target_with_teacher_reasoning(records[rid])) for rid in custody['records']}
    producer_entries = {rid: [] for rid in custody['records']}
    reads = {rid: [] for rid in custody['records']}
    seen_producers, seen_reads = {}, {}
    source_rows = [records[rid] for rid in custody['records']]
    for edge in edges:
        eid = edge_digest(edge)
        wid, rid = edge['writer_record_id'], edge['reader_record_id']
        writer, reader = records[wid], records[rid]
        body = writer_payload(writer, edge['write_name'], edge['body_sha256'])
        occurrence = source_occurrences(writer, reader, edge['write_name'], body, edge['body_sha256'],
            writer_occurrence_id=edge.get('writer_occurrence_id'), reader_occurrence_id=edge.get('reader_occurrence_id'))
        derived = pair_derivations(writer, reader, pieces, tokenizer=tokenizer, payload=body,
            occurrence=occurrence, role_markers=ROLE_MARKERS, tokenizer_sha256=tokenizer_fingerprint,
            source_records=source_rows)
        producer = derived[wid]
        key = (wid, occurrence['writer_occurrence_id'])
        descriptor = producer.get('producer_span')
        if occurrence.get('writer_source_kind') == 'assistant-text-return':
            _assistant_text_target(writer, body, ROLE_MARKERS)  # exact source proof; never use its partial target
            proof = writer['native_source_conversion']['source_target']['assistant_text_result']
            original = writer['target']['content']
            descriptor = {'value_type': 'assistant-text', 'body': body, 'payload_sha256': edge['body_sha256'],
                          'source_start': proof['body_start'], 'source_end': proof['body_end'],
                          'source_content_sha256': hashlib.sha256(original.encode()).hexdigest()}
        if not isinstance(descriptor, dict):
            raise ValueError('source writer requires a jointly renderable authenticated producer span')
        if key in seen_producers:
            if seen_producers[key] != descriptor:
                raise ValueError('one writer occurrence has inconsistent producer derivations')
        else:
            seen_producers[key] = copy.deepcopy(descriptor)
            producer_entries[wid].append({'writer_occurrence_id': key[1], 'body': body,
                                          'span': copy.deepcopy(descriptor), 'edge_sha256': []})
            if descriptor['value_type'] not in ('inline-string', 'assistant-text'):
                targets[wid] = _replace_source_target_span(targets[wid], occurrence, body, ROLE_MARKERS)
        entry = next(x for x in producer_entries[wid] if x['writer_occurrence_id'] == key[1])
        entry['edge_sha256'].append(eid)
        for replacement in derived[rid]['context_replacements']:
            pointer = replacement['pointer']
            item = copy.deepcopy(replacement)
            item['render_tag'] = digest({'occurrence_id': item['occurrence_id'], 'pointer': pointer,
                                         'payload_sha256': item['payload_sha256']})
            signature = (rid, pointer)
            if signature in seen_reads and seen_reads[signature] != item:
                raise ValueError('one reader pointer has inconsistent source derivations')
            if signature not in seen_reads:
                seen_reads[signature] = item
                reads[rid].append({'edge_sha256': eid, 'replacement': item})
            elif seen_reads[signature] == item:
                raise ValueError('one reader occurrence was claimed by multiple source edges')
    derivations = {}
    for rid in custody['records']:
        record = records[rid]
        if not producer_entries[rid] and not reads[rid]: continue
        item = {'schema': 'natlang.native-pair-static-derivation/1', 'source_record_sha256': digest(record)}
        if producer_entries[rid]:
            item.update(source_target_sha256=digest(record['target']), target=targets[rid],
                        producer_spans=[x['span'] for x in producer_entries[rid]])
        if reads[rid]: item['context_replacements'] = [x['replacement'] for x in reads[rid]]
        derivations[rid] = item
    rows, receipt, omissions, provenance = gold_text_rows(source_rows, pieces, tokenizer=tokenizer,
        require_independent_splits=False, prompt_rendering='program-header', target_reasoning=True, include_token_offsets=True,
        tokenizer_sha256=tokenizer_fingerprint, static_derivations=derivations, writer_context_records=source_rows)
    by_id = {row['id']: row for row in rows}
    if omissions or set(by_id) != set(custody['records']) or len(rows) != len(custody['records']):
        raise ValueError('joint chain rendering omitted or deduplicated a source decision: ' + json.dumps(omissions[:5]))
    return {'rows': by_id, 'producer_entries': producer_entries, 'reads': reads,
            'receipt': receipt, 'provenance': provenance}
