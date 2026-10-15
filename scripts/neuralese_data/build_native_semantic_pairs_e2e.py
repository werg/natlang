#!/usr/bin/env python3
"""Render admitted E2E chains into native SFT graphs and witnessed twin pairs.

Records are consumed one complete trajectory at a time. Pair edges are used
only after exact writer target, reader source, split, source groups, payload
hash, text span and per-model token offsets all agree.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from transformers import AutoTokenizer

from natlang_neuralese.data.native_graph_conversion import text_row_graph, witnessed_pair_graph
from natlang_neuralese.data.text_corpus import gold_text_rows, tokenizer_fingerprint
from natlang_neuralese.model.shared_state import ROLE_MARKERS

PAIR_SCHEMA = 'natlang.native-semantic-pair/1'


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path, digest=None):
    with open(path, 'r', encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                if digest is not None:
                    digest.update(line.encode('utf-8'))
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSON') from exc


class CompressedRows:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, self.tmp = tempfile.mkstemp(prefix='.' + self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        os.close(fd)
        self.raw = open(self.tmp, 'wb')
        self.proc = subprocess.Popen(['zstd', '-T2', '-5', '-c'], stdin=subprocess.PIPE, stdout=self.raw)
        self.count = 0

    def write(self, row):
        self.proc.stdin.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
        self.count += 1

    def finish(self):
        self.proc.stdin.close()
        status = self.proc.wait()
        self.raw.flush(); os.fsync(self.raw.fileno()); self.raw.close()
        if status:
            Path(self.tmp).unlink(missing_ok=True)
            raise RuntimeError(f'zstd failed while writing {self.path}: {status}')
        os.replace(self.tmp, self.path)
        return {'path': str(self.path), 'rows': self.count, 'bytes': self.path.stat().st_size,
                'sha256': sha256_file(self.path)}

    def abort(self):
        if self.proc.poll() is None:
            try: self.proc.stdin.close()
            except Exception: pass
            self.proc.terminate(); self.proc.wait()
        if not self.raw.closed: self.raw.close()
        Path(self.tmp).unlink(missing_ok=True)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for key, child in value.items():
            if key == 'arguments' and isinstance(child, str):
                try: child = json.loads(child)
                except json.JSONDecodeError: pass
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value: yield from _walk(child)


def _pointer_part(part):
    return str(part).replace('~', '~0').replace('/', '~1')


def _walk_paths(value, path=''):
    """Yield exact structured occurrences, with stable JSON-pointer paths."""
    if isinstance(value, dict):
        yield path or '/', value
        for key, child in value.items():
            if key == 'arguments' and isinstance(child, str):
                try:
                    parsed = json.loads(child)
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None:
                    yield from _walk_paths(parsed, (path or '') + '/arguments<json>')
            yield from _walk_paths(child, (path or '') + '/' + _pointer_part(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_paths(child, (path or '') + '/' + str(index))


def writer_payload(record, write_name, expected_sha):
    found = []
    for value in _walk(record.get('target') or {}):
        write = value.get('$write') if isinstance(value, dict) else None
        if isinstance(write, dict) and write.get('name') == write_name:
            body = write.get('source')
            if isinstance(body, str): found.append(body)
    unique = set(found)
    if len(unique) != 1:
        raise ValueError(f'writer target has {len(unique)} exact values for {write_name}')
    body = next(iter(unique))
    if sha256_bytes(body.encode()) != expected_sha:
        raise ValueError('writer target payload SHA differs from the admitted edge')
    return body


def source_occurrences(writer_record, reader_record, write_name, body, expected_sha):
    writers, readers = [], []
    # The edge authenticates an emitted target and a later consumed input.
    # Similar historical mentions elsewhere in a row are not the occurrence.
    for pointer, value in _walk_paths(writer_record.get('target') or {}, '/target'):
        write = value.get('$write') if isinstance(value, dict) else None
        if isinstance(write, dict) and write.get('name') == write_name and write.get('source') == body:
            writers.append(pointer + '/$write')
    for pointer, value in _walk_paths(reader_record.get('messages') or [], '/messages'):
        if (isinstance(value, dict) and value.get('type') == 'read'
                and value.get('name') == write_name and value.get('source') == body):
            readers.append(pointer)
    if len(writers) != 1 or len(readers) != 1 or sha256_bytes(body.encode()) != expected_sha:
        raise ValueError(f'communication occurrence is ambiguous or disagrees with payload SHA: writes={len(writers)} reads={len(readers)}')
    return {'writer_json_pointer': writers[0], 'reader_json_pointer': readers[0],
            'payload_sha256': expected_sha}


def reader_has_payload(record, write_name, body, expected_sha):
    matches = [value for value in _walk(record.get('messages') or [])
               if value.get('type') == 'read' and value.get('name') == write_name
               and value.get('source') == body]
    return bool(matches) and sha256_bytes(body.encode()) == expected_sha


def exact_span(row, tokenizer, body, *, side):
    text, ids = row['text'], row['token_ids']
    boundary = row['supervised_suffix_start']
    prefix = tokenizer.decode(ids[:boundary], skip_special_tokens=False)
    if not text.startswith(prefix):
        raise ValueError('rendered text does not preserve its exact target boundary')
    scope_start, scope_stop = (len(prefix), len(text)) if side == 'writer' else (0, len(prefix))
    starts, offset = [], scope_start
    while True:
        offset = text.find(body, offset, scope_stop)
        if offset < 0: break
        starts.append(offset); offset += 1
    if len(starts) != 1:
        raise ValueError(f'payload has {len(starts)} visible occurrences in the authenticated {side} region')
    start, stop = starts[0], starts[0] + len(body)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    if token_ids and isinstance(token_ids[0], list): token_ids = token_ids[0]
    if offsets and isinstance(offsets[0][0], (list, tuple)): offsets = offsets[0]
    if list(token_ids) != ids or len(offsets) != len(ids):
        raise ValueError('whole-row fast tokenization differs from the admitted rendered token IDs')
    selected = [i for i, (left, right) in enumerate(offsets)
                if right > left and left >= start and right <= stop]
    if not selected or selected != list(range(selected[0], selected[-1]+1)):
        raise ValueError('payload is not a contiguous exact token span')
    if offsets[selected[0]][0] != start or offsets[selected[-1]][1] != stop:
        raise ValueError('payload character edge falls inside a tokenizer token')
    if tokenizer.decode(ids[selected[0]:selected[-1]+1], skip_special_tokens=False) != body:
        raise ValueError('payload token range does not decode exactly')
    return {'start': selected[0], 'stop': selected[-1]+1,
            'container_start': selected[0], 'container_stop': selected[-1]+1}


def exact_marked_span(row, tokenizer, body, *, open_marker, close_marker, expected_sha256):
    """Bind structural markers and a separately tokenized native body segment."""
    text, ids = row['text'], row['token_ids']
    wrapper = open_marker + body + close_marker
    if text.count(wrapper) != 1 or sha256_bytes(body.encode()) != expected_sha256:
        raise ValueError('normalized typed payload wrapper or authenticated body is not unique')
    open_id = tokenizer.convert_tokens_to_ids(open_marker)
    close_id = tokenizer.convert_tokens_to_ids(close_marker)
    opens = [index for index, token in enumerate(ids) if token == open_id]
    closes = [index for index, token in enumerate(ids) if token == close_id]
    if len(opens) != 1 or len(closes) != 1 or opens[0] >= closes[0]:
        raise ValueError('normalized structural role tokens are not unique in this model input')
    payload = tokenizer(body, add_special_tokens=False)['input_ids']
    if payload and isinstance(payload[0], list): payload = payload[0]
    if not payload or tokenizer.decode(payload, skip_special_tokens=False) != body:
        raise ValueError('standalone payload tokenization does not reproduce the exact authenticated body')
    return {'container_start': opens[0], 'container_stop': closes[0] + 1,
            'body_text': body, 'payload_token_ids': list(payload),
            'tokenize_payload_alone': True}


def materialize_role_markers(row, *, role_markers, role_ids, payload, side):
    """Replace only the inserted typed-value boundary spellings with role IDs."""
    text, ids = row['text'], row['token_ids']
    wrappers = {'writer': (role_markers[0] + payload + role_markers[1], (0, 1)),
                'reader': (role_markers[2] + payload + role_markers[3], (2, 3))}
    if side not in wrappers:
        raise ValueError('role marker materialization requires an explicit producer or consumer side')
    offset_rows = [tuple(pair) for pair in row['token_offsets']]
    replacements = []
    wrapper, which = wrappers[side]
    if text.count(wrapper) != 1:
        # A payload containing the full wrapper spelling is ambiguous; only
        # the exact occurrence inserted by this transformation is rewritten.
        raise ValueError('normalized role-marker wrapper is ambiguous in the source turn')
    wrapper_start = text.index(wrapper)
    for marker_index in which:
        marker = role_markers[marker_index]
        start = wrapper_start if marker_index in (0, 2) else wrapper_start + len(role_markers[marker_index - 1]) + len(payload)
        stop = start + len(marker)
        token_span = [i for i, (left, right) in enumerate(offset_rows)
                      if right > left and left >= start and right <= stop]
        if (not token_span or token_span != list(range(token_span[0], token_span[-1] + 1))
                or offset_rows[token_span[0]][0] != start or offset_rows[token_span[-1]][1] != stop):
            raise ValueError('role-marker spelling is not bounded by exact rendered token offsets')
        replacements.append((token_span[0], token_span[-1] + 1, role_ids[marker_index], (start, stop)))
    new_ids, new_offsets = [], []
    by_start = {start: (stop, token, offset) for start, stop, token, offset in replacements}
    index = 0
    while index < len(ids):
        if index in by_start:
            stop, token, offset = by_start[index]
            new_ids.append(token); new_offsets.append(offset)
            index = stop
        else:
            new_ids.append(ids[index]); new_offsets.append(offset_rows[index]); index += 1
    boundary = row['supervised_suffix_start']
    for start, stop, _, _ in replacements:
        if stop <= row['supervised_suffix_start']:
            boundary -= stop - start - 1
        elif start < row['supervised_suffix_start'] < stop:
            raise ValueError('role-marker replacement crosses the target boundary')
    row['token_ids'], row['token_offsets'], row['supervised_suffix_start'] = new_ids, [list(x) for x in new_offsets], boundary
    return row


def _replace_pointer(root, pointer, replacement):
    if not pointer.startswith('/') or '<json>' in pointer:
        raise ValueError('typed read occurrence is not a structured message path')
    parts = [part.replace('~1', '/').replace('~0', '~') for part in pointer.split('/')[1:]]
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
    parent[key] = replacement


def normalized_pair_rows(writer_record, reader_record, pieces, *, tokenizer, payload,
                         occurrence, role_markers):
    """Render full source turns with the unique typed source value role-bounded."""
    writer, reader = copy.deepcopy(writer_record), copy.deepcopy(reader_record)
    # The unchanged full-turn graph also enters native SFT. This paired view
    # statically exposes the exact typed writer result and replaces precisely
    # its authenticated reader object by a marker-bounded context span.
    writer['target'] = {'role': 'assistant',
                        'content': role_markers[0] + payload + role_markers[1]}
    _replace_pointer(reader, occurrence['reader_json_pointer'],
        {'type': 'text', 'text': role_markers[2] + payload + role_markers[3]})
    writer_rows, _, writer_omitted, _ = gold_text_rows([writer], pieces, tokenizer=tokenizer,
        require_independent_splits=False, target_reasoning=False, include_token_offsets=True)
    reader_rows, _, reader_omitted, _ = gold_text_rows([reader], pieces, tokenizer=tokenizer,
        require_independent_splits=False, target_reasoning=True, include_token_offsets=True)
    if len(writer_rows) != 1 or writer_omitted or len(reader_rows) != 1 or reader_omitted:
        raise ValueError('normalized full producer/consumer turns did not render uniquely')
    return (materialize_role_markers(writer_rows[0], role_markers=role_markers,
                role_ids=tuple(tokenizer.convert_tokens_to_ids(marker) for marker in role_markers), payload=payload, side='writer'),
            materialize_role_markers(reader_rows[0], role_markers=role_markers,
                role_ids=tuple(tokenizer.convert_tokens_to_ids(marker) for marker in role_markers), payload=payload, side='reader'))


def model_view(edge, writer_row, reader_row, writer_record, reader_record,
               *, tokenizer, source_fingerprint, expanded_fingerprint, role_ids,
               communication_sha256, writer_span, reader_span):
    body = writer_payload(writer_record, edge['write_name'], edge['body_sha256'])
    if not reader_has_payload(reader_record, edge['write_name'], body, edge['body_sha256']):
        raise ValueError('reader source is not the same authenticated writer payload')
    chain_id = edge.get('trajectory')
    edge_identity = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    graph = witnessed_pair_graph(writer_row, reader_row,
        source_tokenizer=tokenizer, source_tokenizer_fingerprint=source_fingerprint,
        expanded_tokenizer_fingerprint=expanded_fingerprint, role_ids=role_ids,
        writer_span=writer_span, reader_span=reader_span,
        payload_sha256=edge['body_sha256'], source_span_id=edge['write_name'],
        edge_id=edge_identity, chain_id=chain_id, context_weight=.05)
    producer = graph['nodes'][0]
    start, stop = producer['producing'][0]
    payload = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
    token_ids, offsets = payload['input_ids'], payload['offset_mapping']
    if token_ids and isinstance(token_ids[0], list): token_ids = token_ids[0]
    if offsets and isinstance(offsets[0][0], (list, tuple)): offsets = offsets[0]
    if producer['token_ids'][start:stop] != list(token_ids):
        raise ValueError('export graph tokens differ from the payload tokenizer IDs')
    return {'graph': graph, 'export': {'node': 'producer', 'start': start, 'stop': stop,
            'logical': False, 'text': body, 'offsets': [list(pair) for pair in offsets]},
            'consumer_nodes': ['consumer'],
            'witness': {'source_id': edge_identity, 'payload_sha256': edge['body_sha256'],
                        'communication_sha256': communication_sha256}}


def provision(tokenizer):
    import inspect
    modern = 'replace_extra_special_tokens' in inspect.signature(tokenizer.add_special_tokens).parameters
    key = 'extra_special_tokens' if modern else 'additional_special_tokens'
    option = 'replace_extra_special_tokens' if modern else 'replace_additional_special_tokens'
    tokenizer.add_special_tokens({key: list(ROLE_MARKERS)}, **{option: False})
    ids = tuple(tokenizer.convert_tokens_to_ids(marker) for marker in ROLE_MARKERS)
    if len(set(ids)) != 4 or any(value is None or value == tokenizer.unk_token_id for value in ids):
        raise ValueError('failed to provision native role tokens')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--pieces', type=Path, required=True)
    parser.add_argument('--edges', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lfm-tokenizer', type=Path, required=True)
    parser.add_argument('--mellum-tokenizer', type=Path, required=True)
    parser.add_argument('--records-sha256', required=True)
    parser.add_argument('--edges-sha256', required=True)
    parser.add_argument('--max-trajectories', type=int, default=None,
                        help='bounded converter test only; omit for corpus admission')
    args = parser.parse_args()
    if args.max_trajectories is not None and args.max_trajectories < 1:
        parser.error('--max-trajectories must be positive')
    tokenizers, source_fps, expanded_fps, role_ids = {}, {}, {}, {}
    model_ids, base_inventory_shas = {}, {}
    from natlang_neuralese.model.native_model import base_inventory
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    for name, path in (('lfm25', args.lfm_tokenizer), ('mellum21', args.mellum_tokenizer)):
        tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)
        inventory_sha = canonical_json_sha256_hex(base_inventory(path))
        architecture = json.loads((path / 'config.json').read_text())['model_type']
        model_ids[name] = architecture + ':' + inventory_sha
        base_inventory_shas[name] = inventory_sha
        source_fps[name] = tokenizer_fingerprint(tok)
        role_ids[name] = provision(tok)
        expanded_fps[name] = tokenizer_fingerprint(tok)
        tokenizers[name] = tok
    records_digest, pieces_digest, edges_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    pieces = {row['name']: row['text'] for row in jsonl(args.pieces, pieces_digest)
              if isinstance(row.get('name'), str) and isinstance(row.get('text'), str)}
    edges_by_chain = defaultdict(list)
    for edge in jsonl(args.edges, edges_digest):
        edges_by_chain[edge['trajectory']].append(edge)
    writers = {split: {model: CompressedRows(args.out / 'graphs' / model / split / 'native-graphs.jsonl.zst')
                       for model in tokenizers} for split in ('train', 'test')}
    pair_writers = {split: CompressedRows(args.out / 'pairs' / split / 'semantic-pairs.jsonl.zst')
                    for split in ('train', 'test')}
    omissions_path = args.out / 'pair-omissions.jsonl'
    omissions_path.parent.mkdir(parents=True, exist_ok=True)
    omissions = open(omissions_path, 'w', encoding='utf-8')
    totals = defaultdict(int)
    seen_chains, seen_edges = set(), set()
    chain_rows, current = [], None

    def process(chain, records):
        if chain in seen_chains:
            raise ValueError('trajectory rows are not contiguous in their admitted source')
        seen_chains.add(chain)
        totals['trajectories'] += 1
        rendered = {}
        for model, tok in tokenizers.items():
            rows, receipt, omitted, _ = gold_text_rows(records, pieces, tokenizer=tok,
                require_independent_splits=False, target_reasoning=True)
            rendered[model] = rows
            totals['rendered_' + model] += len(rows)
            totals['render_omitted_' + model] += len(omitted)
        row_maps = {}
        for model, rows in rendered.items():
            mapping = {}
            for row in rows:
                for record_id in row.get('source_record_ids') or [row['id']]:
                    if record_id in mapping and mapping[record_id] is not row:
                        raise ValueError(f'{model} rendered record maps to conflicting text rows: {record_id}')
                    mapping[record_id] = row
                # Every rendered whole turn also enters native SFT as ordinary one-node supervision.
                # gold_text_rows tokenizes with this already-expanded
                # tokenizer, so its row fingerprint is the expanded one.
                graph = text_row_graph(row, source_tokenizer_fingerprint=expanded_fps[model], tokenizer=tokenizers[model],
                    expanded_tokenizer_fingerprint=expanded_fps[model], context_weight=.05)
                writers[row['split']][model].write(graph)
        record_map = {row['id']: row for row in records}
        for model, rows in rendered.items():
            row_maps[model] = {}
            for row in rows:
                for record_id in row.get('source_record_ids') or [row['id']]:
                    row_maps[model][record_id] = row
        for edge in edges_by_chain.pop(chain, []):
            edge_id = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            edge_identity = edge_id
            if edge_id in seen_edges:
                raise ValueError('duplicate admitted E2E edge identity')
            seen_edges.add(edge_id)
            totals['source_edges'] += 1
            try:
                writer_record = record_map[edge['writer_record_id']]
                reader_record = record_map[edge['reader_record_id']]
                if (writer_record.get('split') != edge.get('split')
                        or reader_record.get('split') != edge.get('split')):
                    raise ValueError('edge, writer and reader do not share the admitted split')
                body = writer_payload(writer_record, edge['write_name'], edge['body_sha256'])
                occurrence = source_occurrences(writer_record, reader_record, edge['write_name'],
                    body, edge['body_sha256'])
                occurrence_binding = {'edge_id': edge_identity, 'trajectory': chain,
                    'writer_source_row_sha256': (writer_record.get('source_ref') or {}).get('source_row_sha256'),
                    'reader_source_row_sha256': (reader_record.get('source_ref') or {}).get('source_row_sha256'),
                    **occurrence}
                communication = {'schema': 'natlang.native-semantic-communication/1',
                    'source_id': edge_identity, 'writer_record_id': edge['writer_record_id'],
                    'reader_record_id': edge['reader_record_id'], 'write_name': edge['write_name'],
                    'occurrence': json.dumps(occurrence_binding, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=False),
                    'source_binding': {'corpus_id': 'neuralese-e2e-cloud-rest27-protected-20261015-v2',
                        'records': {'path': str(args.records), 'sha256': args.records_sha256},
                        'edges': {'path': str(args.edges), 'sha256': args.edges_sha256}}}
                communication_sha = hashlib.sha256(json.dumps(communication, sort_keys=True,
                    separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
                views = {}
                for model in tokenizers:
                    writer_row, reader_row = normalized_pair_rows(writer_record, reader_record, pieces,
                        tokenizer=tokenizers[model], payload=body, occurrence=occurrence,
                        role_markers=ROLE_MARKERS)
                    writer_span = exact_marked_span(writer_row, tokenizers[model], body,
                        open_marker=ROLE_MARKERS[0], close_marker=ROLE_MARKERS[1],
                        expected_sha256=edge['body_sha256'])
                    reader_span = exact_marked_span(reader_row, tokenizers[model], body,
                        open_marker=ROLE_MARKERS[2], close_marker=ROLE_MARKERS[3],
                        expected_sha256=edge['body_sha256'])
                    views[model_ids[model]] = model_view(edge, writer_row, reader_row, writer_record, reader_record,
                        tokenizer=tokenizers[model], source_fingerprint=expanded_fps[model],
                        expanded_fingerprint=expanded_fps[model], role_ids=role_ids[model],
                        communication_sha256=communication_sha,
                        writer_span=writer_span, reader_span=reader_span)
                model_views = list(views.values())
                groups = sorted(set(model_views[0]['graph']['source_groups']) |
                                set(model_views[1]['graph']['source_groups']))
                if model_views[0]['export']['text'] != model_views[1]['export']['text']:
                    raise ValueError('cross-model payload text differs after rendering')
                row = {'schema': PAIR_SCHEMA, 'id': 'e2e-edge:' + edge_id,
                    'split': edge['split'], 'source_groups': groups,
                    'payload_sha256': edge['body_sha256'], 'models': views,
                    'communication': communication, 'communication_sha256': communication_sha,
                    'conversion': {'kind': 'static-transformation', 'native_capture': False,
                        'normalized_typed_occurrence': True,
                        'source_corpus': 'neuralese-e2e-cloud-rest27-protected-20261015-v2',
                        'trajectory': chain, 'edge_id': edge_id,
                        'source_writer_id': edge['writer_record_id'],
                        'source_reader_id': edge['reader_record_id'],
                        'source_span_id': edge['write_name']}}
                pair_writers[edge['split']].write(row)
                totals['paired_edges'] += 1
            except (KeyError, TypeError, ValueError) as exc:
                totals['pair_omissions'] += 1
                omissions.write(json.dumps({'edge': edge, 'reason': str(exc)}, separators=(',', ':')) + '\n')

    try:
        for record in jsonl(args.records, records_digest):
            chain = record.get('teacher_trajectory_id') or (record.get('source_ref') or {}).get('trajectory_id')
            if not isinstance(chain, str) or not chain:
                raise ValueError('admitted E2E record lacks its complete trajectory identity')
            if current is None: current = chain
            if chain != current:
                process(current, chain_rows)
                if args.max_trajectories and len(seen_chains) >= args.max_trajectories:
                    break
                current, chain_rows = chain, []
            chain_rows.append(record)
        else:
            if chain_rows: process(current, chain_rows)
        if not args.max_trajectories:
            if edges_by_chain:
                missing = sum(map(len, edges_by_chain.values()))
                raise ValueError(f'{missing} admitted E2E edges have no corresponding source trajectory rows')
        if not args.max_trajectories and (records_digest.hexdigest() != args.records_sha256
                or edges_digest.hexdigest() != args.edges_sha256):
            raise ValueError('source records/edges SHA differs from the admitted immutable source binding')
        omissions.flush(); os.fsync(omissions.fileno()); omissions.close()
        graph_outputs = {model: {split: writers[split][model].finish() for split in ('train', 'test')}
                         for model in tokenizers}
        pair_outputs = {split: pair_writers[split].finish() for split in ('train', 'test')}
    except BaseException:
        if not omissions.closed: omissions.close()
        for mapping in writers.values():
            for writer in mapping.values(): writer.abort()
        for writer in pair_writers.values(): writer.abort()
        raise
    receipt = {'schema': 'natlang.native-semantic-e2e-build/1',
        'source': {'records': {'path': str(args.records), 'sha256': args.records_sha256},
                   'pieces': {'path': str(args.pieces), 'sha256': pieces_digest.hexdigest()},
                   'edges': {'path': str(args.edges), 'sha256': args.edges_sha256}},
        'tokenizers': {model: {'source': source_fps[model], 'expanded': expanded_fps[model],
                               'role_ids': list(role_ids[model])} for model in tokenizers},
        'model_ids': model_ids, 'base_inventory_sha256': base_inventory_shas,
        'counts': dict(totals), 'bounded_test_only': args.max_trajectories is not None,
        'outputs': {'graphs': graph_outputs, 'pairs': pair_outputs},
        'pair_omissions': {'path': str(omissions_path), 'rows': totals['pair_omissions'],
                           'sha256': sha256_file(omissions_path)},
        'conversion': 'all rendered trajectories become ordinary native one-node SFT graphs; cross-model pairs only for exact admitted writer/reader edges'}
    args.out.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out / 'receipt.json'
    temp = receipt_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    with temp.open('rb') as handle: os.fsync(handle.fileno())
    os.replace(temp, receipt_path)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__': main()
