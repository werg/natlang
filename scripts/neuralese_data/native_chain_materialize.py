"""Materialize authenticated joint-render marker intervals into native role IDs."""
from __future__ import annotations

import hashlib
from natlang_neuralese.model.shared_state import ROLE_MARKERS
from scripts.neuralese_data.native_chain_render import digest, edge_digest


def materialize(rendered, records, edges, *, tokenizer, role_ids, context_weight=.05):
    edge_map = {edge_digest(edge): edge for edge in edges}
    all_nodes, producer_positions, pending = {}, {}, []
    for rid, row in rendered['rows'].items():
        text, ids = row['text'], row['token_ids']
        offsets = [tuple(x) for x in row['token_offsets']]
        if len(ids) != len(offsets) or tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise ValueError('joint rendered offsets do not bind the actual token text')
        specs = []
        producers = rendered['producer_entries'][rid]
        intervals = row.get('producer_rendered_intervals') or []
        if len(producers) != len(intervals):
            raise ValueError('joint producer occurrence intervals do not close over source writes')
        for entry, interval in zip(producers, intervals):
            start, stop = interval['char_start'], interval['char_stop']
            specs.append({'kind': 'producer', 'entry': entry, 'start': start-len(ROLE_MARKERS[0]),
                          'stop': stop+len(ROLE_MARKERS[1]), 'body_start': start, 'body_stop': stop,
                          'body': entry['body'], 'roles': (0, 1)})
        for item in rendered['reads'][rid]:
            replacement = item['replacement']; interval = row['context_rendered_intervals'][replacement['render_tag']]
            start, stop = interval['char_start'], interval['char_stop']
            body = replacement['replacement']['text'][len(ROLE_MARKERS[2]):-len(ROLE_MARKERS[3])]
            specs.append({'kind': 'reader', 'entry': item, 'start': start, 'stop': stop,
                          'body_start': start+len(ROLE_MARKERS[2]), 'body_stop': stop-len(ROLE_MARKERS[3]),
                          'body': body, 'roles': (2, 3)})
        spans = []
        for spec in specs:
            left, right = spec['roles']
            if (spec['start'] < 0 or spec['stop'] > len(text)
                    or text[spec['start']:spec['body_start']] != ROLE_MARKERS[left]
                    or text[spec['body_stop']:spec['stop']] != ROLE_MARKERS[right]):
                raise ValueError('source-bound marker interval differs from joint rendered text')
            observed = text[spec['body_start']:spec['body_stop']]
            if spec['kind'] == 'reader' and observed != spec['body']:
                raise ValueError('reader render differs from exact source payload')
            if hashlib.sha256(spec['body'].encode()).hexdigest() != (
                edge_map[spec['entry']['edge_sha256'][0]]['body_sha256'] if spec['kind'] == 'producer'
                else edge_map[spec['entry']['edge_sha256']]['body_sha256']):
                raise ValueError('materialized body differs from authenticated source edge')
            positions = [i for i, (a, b) in enumerate(offsets)
                         if b > a and a >= spec['start'] and b <= spec['stop']]
            if (not positions or positions != list(range(positions[0], positions[-1]+1))
                    or offsets[positions[0]][0] != spec['start'] or offsets[positions[-1]][1] != spec['stop']):
                raise ValueError('role wrapper does not align to exact token offsets')
            body_ids = tokenizer(spec['body'], add_special_tokens=False)['input_ids']
            if body_ids and isinstance(body_ids[0], list): body_ids = body_ids[0]
            if not body_ids or tokenizer.decode(body_ids, skip_special_tokens=False) != spec['body']:
                raise ValueError('source payload does not round-trip through standalone tokenization')
            spans.append((positions[0], positions[-1]+1, spec, list(body_ids)))
        spans.sort(key=lambda x: x[0])
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError('joint source marker intervals overlap')
        new_ids, positions, cursor = [], [], 0
        for begin, end, spec, body_ids in spans:
            new_ids.extend(ids[cursor:begin]); start = len(new_ids)
            left, right = spec['roles']
            new_ids.extend([role_ids[left], *body_ids, role_ids[right]])
            positions.append((spec, start+1, start+1+len(body_ids)))
            cursor = end
        new_ids.extend(ids[cursor:])
        suffix = row['supervised_suffix_start']
        for begin, end, _, body_ids in spans:
            if begin < row['supervised_suffix_start'] < end: raise ValueError('marker interval crosses the exact target suffix')
            if end <= row['supervised_suffix_start']: suffix += len(body_ids)+2-(end-begin)
        weights = [context_weight]*suffix + [1.0]*(len(new_ids)-suffix)
        if weights: weights[0] = 0.0
        node = {'token_ids': new_ids, 'target_weights': weights, 'supervised_suffix_start': suffix,
                'source_record_sha256': digest(records[rid]), 'source_target_sha256': digest(records[rid]['target']),
                'producing': [], 'producer_bindings': [], 'reads': []}
        for spec, start, stop in positions:
            if spec['kind'] == 'producer':
                node['producing'].append([start, stop])
                for eid in spec['entry']['edge_sha256']:
                    node['producer_bindings'].append({'edge_sha256': eid, 'start': start, 'stop': stop,
                        'writer_occurrence_id': spec['entry']['writer_occurrence_id']})
                    producer_positions[eid] = (rid, start, stop)
            else:
                item = spec['entry']; replacement = item['replacement']
                pending.append((rid, {'start': start, 'stop': stop, 'edge_sha256': item['edge_sha256'],
                    'occurrence_id': replacement['occurrence_id'], 'source_pointer': replacement['pointer']}))
        all_nodes[rid] = node
    for rid, read in pending:
        source, start, stop = producer_positions[read['edge_sha256']]
        read.update(source_record_id=source, source_start=start, source_stop=stop)
        all_nodes[rid]['reads'].append(read)
    return all_nodes
