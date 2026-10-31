"""Joint source-bound rendering inputs for full native trajectory graphs."""
from __future__ import annotations

import copy
import hashlib
import json

from natlang_neuralese.common.jsonio import canonical_json_str
from natlang_neuralese.data.text_corpus import (gold_text_rows, project_embedded_thinking_target,
                                                target_with_teacher_reasoning)
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


def _plain_result_span(writer, occurrence, body):
    """Producer span of a typed string result in the original return_result call.

    The converter's authenticated producer occurrence points at the string value inside the exact tool-call
    arguments (`/target/tool_calls/<i>/function/arguments<json>/<path>`). The whole call stays the supervised target;
    only that value is marked, exactly like a structured `Neuralese<string>` descriptor's value.
    """
    if occurrence.get('writer_source_kind') not in ('Neuralese<string>', 'typed-text-result'):
        return None
    if (occurrence.get('writer_source_kind') == 'typed-text-result'
            and not str(occurrence.get('write_name', '')).startswith('soft-state:nz1_')):
        return None
    parts = [part.replace('~1', '/').replace('~0', '~') for part in occurrence['writer_json_pointer'].split('/')[1:]]
    json_index = next((index for index, part in enumerate(parts) if part.endswith('<json>')), None)
    if (len(parts) != 6 or parts[:2] != ['target', 'tool_calls'] or not parts[2].isdigit() or json_index != 4
            or parts[3] != 'function' or parts[4] != 'arguments<json>' or json_index + 1 >= len(parts)):
        return None
    try:
        call = writer['target']['tool_calls'][int(parts[2])]
        function = call.get('function', call)
        if function.get('name') != 'return_result':
            return None
        value = json.loads(function['arguments'])
        for part in parts[json_index + 1:]:
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if (parts[json_index + 1:] != ['value'] or value != body or not isinstance(value, str)
            or occurrence.get('payload_sha256') != hashlib.sha256(body.encode()).hexdigest()):
        return None
    try:
        arguments = json.loads(function['arguments'])
    except (TypeError, ValueError):
        return None
    if not isinstance(arguments, dict) or arguments.get('status') != 'success':
        return None
    return {'call_index': int(parts[2]), 'argument_path': parts[json_index + 1:], 'body': body,
            'payload_sha256': occurrence['payload_sha256'], 'value_type': 'string'}


def render_chain(chain, records, edges, pieces, *, tokenizer, tokenizer_fingerprint,
                 prompt_rendering='program-header', retain_prompt_tools=False, serving_entry=None):
    """Render each complete source decision once, with every witnessed write and read.

    Returned rows are not yet native graphs: role marker materialization and
    import-position assembly must retain these exact occurrence bindings.
    """
    custody = bind_chain(chain, records, edges)
    targets = {rid: copy.deepcopy(target_with_teacher_reasoning(records[rid])) for rid in custody['records']}
    target_projections = {}
    outgoing = {edge['writer_record_id'] for edge in edges}
    for rid in custody['records']:
        # Structured tool-call values are outside target.content, so the
        # projection cannot disturb their source-bound spans. An assistant
        # text producer can bind content offsets and must keep the source form.
        if rid in outgoing and not (records[rid].get('target', {}).get('tool_calls')):
            continue
        projection = project_embedded_thinking_target(records[rid])
        if projection is not None:
            targets[rid], target_projections[rid] = projection
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
        plain_result = descriptor is None and _plain_result_span(writer, occurrence, body)
        if plain_result:
            descriptor = plain_result
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
            if (descriptor['value_type'] not in ('inline-string', 'assistant-text', 'unknown')
                    and not plain_result and not descriptor.get('source_target_pointer')):
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
        if not producer_entries[rid] and not reads[rid] and rid not in target_projections: continue
        item = {'schema': 'natlang.native-pair-static-derivation/1', 'source_record_sha256': digest(record)}
        if producer_entries[rid]:
            item.update(source_target_sha256=digest(record['target']), target=targets[rid],
                        producer_spans=[x['span'] for x in producer_entries[rid]])
        if rid in target_projections:
            item.update(source_target_sha256=digest(record['target']), target=targets[rid],
                        target_projection=copy.deepcopy(target_projections[rid]))
        if reads[rid]: item['context_replacements'] = [x['replacement'] for x in reads[rid]]
        derivations[rid] = item
    rows, receipt, omissions, provenance = gold_text_rows(source_rows, pieces, tokenizer=tokenizer,
        require_independent_splits=False, prompt_rendering=prompt_rendering, target_reasoning=True, include_token_offsets=True,
        retain_prompt_tools=retain_prompt_tools, serving_entry=serving_entry,
        tokenizer_sha256=tokenizer_fingerprint, static_derivations=derivations, writer_context_records=source_rows)
    by_id = {row['id']: row for row in rows}
    if omissions or set(by_id) != set(custody['records']) or len(rows) != len(custody['records']):
        raise ValueError('joint chain rendering omitted or deduplicated a source decision: ' + json.dumps(omissions[:5]))
    return {'rows': by_id, 'producer_entries': producer_entries, 'reads': reads,
            'receipt': receipt, 'provenance': provenance}
