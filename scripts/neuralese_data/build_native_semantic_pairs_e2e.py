#!/usr/bin/env python3
"""Render admitted E2E chains into native SFT graphs and witnessed twin pairs.

Records are consumed one complete trajectory at a time. Pair edges are used
only after exact writer target, reader source, split, source groups, payload
hash, text span and per-model token offsets all agree.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from transformers import AutoTokenizer
from natlang_neuralese.common.hashing import sha256_hex as sha256_bytes, sha256_file_hex as sha256_file

from natlang_neuralese.data.native_graph_conversion import text_row_graph, witnessed_pair_graph
from natlang_neuralese.data.text_corpus import gold_text_rows, tokenizer_fingerprint
from natlang_neuralese.model.shared_state import ROLE_MARKERS

PAIR_SCHEMA = 'natlang.native-semantic-pair/1'
BUILD_SCHEMA = 'natlang.native-semantic-e2e-build/2'
BUILD_POLICY = 'complete-source-trajectories-expanded-tokenizer-native-graphs-v2'
CHECKPOINT_TRAJECTORIES = 16
RESUME_SCHEMA = 'natlang.native-semantic-e2e-resume/2'


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


class JsonlRows:
    """Atomic plain JSONL chunk for the source-specific omission ledger."""
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, self.tmp = tempfile.mkstemp(prefix='.' + self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        self.raw = os.fdopen(fd, 'wb'); self.count = 0

    def write(self, row):
        self.raw.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
        self.count += 1

    def finish(self):
        self.raw.flush(); os.fsync(self.raw.fileno()); self.raw.close()
        os.replace(self.tmp, self.path)
        return {'path': str(self.path), 'rows': self.count, 'bytes': self.path.stat().st_size,
                'sha256': sha256_file(self.path)}

    def abort(self):
        if not self.raw.closed: self.raw.close()
        Path(self.tmp).unlink(missing_ok=True)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def source_stat_identity(path):
    stat = Path(path).stat()
    return {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns}


def jsonl_with_offsets(path, start=0, digest=None):
    with open(path, 'rb') as source:
        source.seek(start)
        while True:
            line_start = source.tell()
            line = source.readline()
            if not line: break
            line_stop = source.tell()
            if digest is not None: digest.update(line)
            if line.strip():
                try: yield json.loads(line), line_start, line_stop
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}@{line_start}: invalid JSON') from exc


def _chunk_parts(chunk):
    for by_split in chunk['graphs'].values():
        yield from by_split.values()
    yield from chunk['pairs'].values()
    yield chunk['omissions']
    if chunk.get('render_omissions'):
        yield chunk['render_omissions']


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
    if len(writers) != 1 or not readers or sha256_bytes(body.encode()) != expected_sha:
        raise ValueError(f'communication occurrence is ambiguous or disagrees with payload SHA: writes={len(writers)} reads={len(readers)}')
    return {'writer_json_pointer': writers[0], 'reader_json_pointers': readers,
            'payload_sha256': expected_sha,
            'transform': 'static-shared-producer-multi-read' if len(readers) > 1 else 'static-exact-read'}


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


def exact_marked_spans(row, tokenizer, body, *, open_marker, close_marker, expected_sha256,
                       expected_count):
    """Bind every statically rewritten typed read to its exact structural marker span."""
    text, ids = row['text'], row['token_ids']
    wrapper = open_marker + body + close_marker
    starts, cursor = [], 0
    while True:
        start = text.find(wrapper, cursor)
        if start < 0: break
        starts.append(start); cursor = start + len(wrapper)
    if len(starts) != expected_count or sha256_bytes(body.encode()) != expected_sha256:
        raise ValueError('normalized typed payload wrappers differ from the authenticated read occurrences')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if token_ids and isinstance(token_ids[0], list): token_ids = token_ids[0]
    if offsets and isinstance(offsets[0][0], (list, tuple)): offsets = offsets[0]
    if list(token_ids) != ids or len(offsets) != len(ids):
        raise ValueError('whole-row tokenization differs from the statically normalized consumer')
    open_id, close_id = tokenizer.convert_tokens_to_ids(open_marker), tokenizer.convert_tokens_to_ids(close_marker)
    payload = tokenizer(body, add_special_tokens=False)['input_ids']
    if payload and isinstance(payload[0], list): payload = payload[0]
    if not payload or tokenizer.decode(payload, skip_special_tokens=False) != body:
        raise ValueError('standalone payload tokenization does not reproduce the exact authenticated text')
    spans = []
    for start in starts:
        stop = start + len(wrapper)
        opening = [i for i, (left, right) in enumerate(offsets)
                   if left == start and right == start + len(open_marker)]
        closing = [i for i, (left, right) in enumerate(offsets)
                   if left == stop - len(close_marker) and right == stop]
        if (len(opening) != 1 or len(closing) != 1 or ids[opening[0]] != open_id
                or ids[closing[0]] != close_id or opening[0] >= closing[0]):
            raise ValueError('consumer role markers do not bind each exact typed read occurrence')
        spans.append({'container_start': opening[0], 'container_stop': closing[0] + 1,
            'body_text': body, 'payload_token_ids': list(payload), 'tokenize_payload_alone': True})
    if any(left['container_stop'] > right['container_start'] for left, right in zip(spans, spans[1:])):
        raise ValueError('typed read marker spans overlap')
    return spans


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
    starts, cursor = [], 0
    while True:
        start = text.find(wrapper, cursor)
        if start < 0: break
        starts.append(start); cursor = start + len(wrapper)
    if len(starts) != 1:
        # Writer values remain unique; reader fan-out is handled by the shared
        # occurrence-aware path below.
        if side != 'reader' or not starts:
            raise ValueError('normalized role-marker wrapper is ambiguous in the source turn')
    for wrapper_start in starts:
        for marker_index in which:
            marker = role_markers[marker_index]
            start = (wrapper_start if marker_index in (0, 2) else
                     wrapper_start + len(role_markers[marker_index - 1]) + len(payload))
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
                         occurrence, role_markers, tokenizer_sha256):
    """Render full source turns with the unique typed source value role-bounded."""
    writer, reader = copy.deepcopy(writer_record), copy.deepcopy(reader_record)
    # The unchanged full-turn graph also enters native SFT. This paired view
    # statically exposes the exact typed writer result and replaces precisely
    # its authenticated reader object by a marker-bounded context span.
    writer['target'] = {'role': 'assistant',
                        'content': role_markers[0] + payload + role_markers[1]}
    for pointer in occurrence['reader_json_pointers']:
        _replace_pointer(reader, pointer, {'type': 'text',
            'text': role_markers[2] + payload + role_markers[3]})
    writer_rows, _, writer_omitted, _ = gold_text_rows([writer], pieces, tokenizer=tokenizer,
        require_independent_splits=False, target_reasoning=False, include_token_offsets=True,
        tokenizer_sha256=tokenizer_sha256)
    reader_rows, _, reader_omitted, _ = gold_text_rows([reader], pieces, tokenizer=tokenizer,
        require_independent_splits=False, target_reasoning=True, include_token_offsets=True,
        tokenizer_sha256=tokenizer_sha256)
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


_RENDER_CONTEXT = {}


def _render_trajectory(task):
    chain, records_start, records_stop = task
    context = _RENDER_CONTEXT
    args = context['args']
    # Send immutable source byte ranges through the worker queue, not hundreds
    # of megabytes of decoded Python objects. Complete trajectories remain
    # intact: only the worker materializes the exact bounded range needed by
    # the existing cross-row renderer and edge witness checks.
    records = []
    with open(args.records, 'rb') as source:
        source.seek(records_start)
        while source.tell() < records_stop:
            line = source.readline()
            if not line:
                raise ValueError('trajectory source range ended before its receipted byte boundary')
            records.append(json.loads(line))
        if source.tell() != records_stop:
            raise ValueError('trajectory source range does not end on an exact JSONL record boundary')
    tokenizers = context['tokenizers']
    pieces = context['pieces']
    expanded_fps = context['expanded_fps']
    role_ids = context['role_ids']
    model_ids = context['model_ids']
    edges_by_chain = context['edges_by_chain']
    pair_only = context.get('pair_only', False)
    local = defaultdict(int)
    graphs = {split: {model: [] for model in tokenizers} for split in ('train', 'test')}
    pairs = {split: [] for split in ('train', 'test')}
    omitted_pairs = []
    render_omissions = []
    local['trajectories'] += 1
    if not pair_only:
        rendered = {}
        for model, tok in tokenizers.items():
            rows, receipt, omitted, _ = gold_text_rows(records, pieces, tokenizer=tok,
                require_independent_splits=False, target_reasoning=True,
                tokenizer_sha256=expanded_fps[model])
            rendered[model] = rows
            local['rendered_' + model] += len(rows)
            local['render_omitted_' + model] += len(omitted)
            render_omissions.extend({'kind': 'ordinary-native-graph-omission', 'model': model,
                'trajectory': chain, 'omission': row} for row in omitted)
        for model, rows in rendered.items():
            for row in rows:
                # Every rendered whole turn also enters native SFT as ordinary one-node supervision.
                # gold_text_rows tokenizes with this already-expanded tokenizer.
                graph = text_row_graph(row, source_tokenizer_fingerprint=expanded_fps[model], tokenizer=tokenizers[model],
                    expanded_tokenizer_fingerprint=expanded_fps[model], context_weight=.05)
                graphs[row['split']][model].append(graph)
    record_map = {row['id']: row for row in records}
    for edge in edges_by_chain.get(chain, []):
        edge_id = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        edge_identity = edge_id
        local['source_edges'] += 1
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
                    role_markers=ROLE_MARKERS, tokenizer_sha256=expanded_fps[model])
                writer_span = exact_marked_span(writer_row, tokenizers[model], body,
                    open_marker=ROLE_MARKERS[0], close_marker=ROLE_MARKERS[1],
                    expected_sha256=edge['body_sha256'])
                reader_span = exact_marked_spans(reader_row, tokenizers[model], body,
                    open_marker=ROLE_MARKERS[2], close_marker=ROLE_MARKERS[3],
                    expected_sha256=edge['body_sha256'],
                    expected_count=len(occurrence['reader_json_pointers']))
                views[model_ids[model]] = model_view(edge, writer_row, reader_row, writer_record, reader_record,
                    tokenizer=tokenizers[model], source_fingerprint=expanded_fps[model],
                    expanded_fingerprint=expanded_fps[model], role_ids=role_ids[model],
                    communication_sha256=communication_sha,
                    writer_span=writer_span, reader_span=reader_span)
                # Keep the exact structured source occurrence alongside the
                # converted graph so audits can reconstruct why plaintext was
                # removed and which authenticated read(s) the imported state
                # replaced, without reopening or guessing from token strings.
                views[model_ids[model]]['graph']['source_pairing'].update({
                    'communication_sha256': communication_sha,
                    'communication_occurrence': occurrence_binding,
                    'transformation': occurrence['transform'],
                })
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
                        'normalized_typed_occurrence': True, 'transformation': occurrence['transform'],
                    'source_corpus': 'neuralese-e2e-cloud-rest27-protected-20261015-v2',
                    'trajectory': chain, 'edge_id': edge_id,
                    'source_writer_id': edge['writer_record_id'],
                    'source_reader_id': edge['reader_record_id'],
                    'source_span_id': edge['write_name']}}
            pairs[edge['split']].append(row)
            local['paired_edges'] += 1
        except (KeyError, TypeError, ValueError) as exc:
            local['pair_omissions'] += 1
            omitted_pairs.append({'edge': edge, 'reason': str(exc)})
    return {'graphs': graphs, 'pairs': pairs, 'omissions': omitted_pairs,
            'render_omissions': render_omissions, 'totals': dict(local)}


def _initialize_render_worker(config):
    """Load tokenizer/source context once in each isolated CPU worker."""
    from types import SimpleNamespace
    tokenizers, expanded_fps, role_ids = {}, {}, {}
    for model, path in config['tokenizer_paths'].items():
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
        ids = provision(tok)
        fingerprint = tokenizer_fingerprint(tok)
        if fingerprint != config['expanded_fps'][model] or list(ids) != config['role_ids'][model]:
            raise ValueError(f'worker tokenizer identity changed for {model}')
        tokenizers[model], expanded_fps[model], role_ids[model] = tok, fingerprint, ids
    if sha256_file(config['pieces_path']) != config['pieces_sha256']:
        raise ValueError('worker pieces source differs from the parent pinned SHA')
    if sha256_file(config['edges_path']) != config['edges_sha256']:
        raise ValueError('worker edges source differs from the parent pinned SHA')
    pieces = {row['name']: row['text'] for row in jsonl(config['pieces_path'])
              if isinstance(row.get('name'), str) and isinstance(row.get('text'), str)}
    edges_by_chain = defaultdict(list)
    edge_ids = set()
    selected_edges = set(config.get('selected_edge_ids') or [])
    for edge in jsonl(config['edges_path']):
        edge_id = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if edge_id in edge_ids:
            raise ValueError('duplicate admitted E2E edge identity in render worker')
        edge_ids.add(edge_id)
        if not config.get('pair_only') or edge_id in selected_edges:
            edges_by_chain[edge['trajectory']].append(edge)
    _RENDER_CONTEXT.clear()
    _RENDER_CONTEXT.update({'args': SimpleNamespace(records=Path(config['records_path']),
        records_sha256=config['records_sha256'], edges=Path(config['edges_path']),
        edges_sha256=config['edges_sha256']), 'tokenizers': tokenizers, 'pieces': pieces,
        'expanded_fps': expanded_fps, 'role_ids': role_ids,
        'model_ids': config['model_ids'], 'edges_by_chain': edges_by_chain,
        'pair_only': config.get('pair_only', False)})


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
    parser.add_argument('--resume', action='store_true',
                        help='resume only from the last atomically receipted complete trajectory')
    parser.add_argument('--resume-if-present', action='store_true',
                        help='resume a matching incomplete manifest automatically after a ledger restart')
    parser.add_argument('--checkpoint-trajectories', type=int, default=CHECKPOINT_TRAJECTORIES,
                        help='trajectory count between durable output frontiers')
    parser.add_argument('--workers', type=int, default=8,
                        help='independent complete-trajectory render workers')
    parser.add_argument('--repair-pair-omissions', type=Path, default=None,
                        help='convert exactly the omitted edges listed by a prior complete build into a pair-only supplement')
    args = parser.parse_args()
    if args.max_trajectories is not None and args.max_trajectories < 1:
        parser.error('--max-trajectories must be positive')
    if args.checkpoint_trajectories < 1:
        parser.error('--checkpoint-trajectories must be positive')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.repair_pair_omissions and args.max_trajectories is not None:
        parser.error('--repair-pair-omissions cannot be combined with a bounded test run')
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
    edge_ids = set()
    for edge in jsonl(args.edges, edges_digest):
        edge_id = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if edge_id in edge_ids:
            raise ValueError('duplicate admitted E2E edge identity')
        edge_ids.add(edge_id)
        edges_by_chain[edge['trajectory']].append(edge)
    if edges_digest.hexdigest() != args.edges_sha256:
        raise ValueError('source edges SHA differs from the admitted immutable source binding')
    pair_only = args.repair_pair_omissions is not None
    repair_edge_ids = None
    repair_sha = None
    if pair_only:
        repair_sha = sha256_file(args.repair_pair_omissions)
        previous_receipt_path = args.repair_pair_omissions.parent / 'receipt.json'
        if not previous_receipt_path.is_file():
            raise ValueError('pair-repair omissions require the prior build receipt beside the omission ledger')
        previous_receipt = json.loads(previous_receipt_path.read_text())
        previous_binding = previous_receipt.get('source') or {}
        if (previous_binding.get('records', {}).get('sha256') != args.records_sha256
                or previous_binding.get('edges', {}).get('sha256') != args.edges_sha256
                or previous_binding.get('pieces', {}).get('sha256') != pieces_digest.hexdigest()
                or (previous_receipt.get('pair_omissions') or {}).get('sha256') != repair_sha):
            raise ValueError('pair-repair omissions do not bind the exact prior receipt and current immutable source')
        repair_edge_ids = set()
        for row in jsonl(args.repair_pair_omissions):
            edge = row.get('edge')
            if not isinstance(edge, dict):
                raise ValueError('pair-repair source contains a row without its exact omitted edge')
            repair_edge_ids.add(hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        if not repair_edge_ids or not repair_edge_ids <= edge_ids:
            raise ValueError('pair-repair source does not exactly bind a nonempty subset of admitted source edges')
        for chain in list(edges_by_chain):
            selected = [edge for edge in edges_by_chain[chain]
                        if hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                        in repair_edge_ids]
            if selected:
                edges_by_chain[chain] = selected
            else:
                del edges_by_chain[chain]
    state_path = args.out / 'resume-manifest.json'
    records_stat = source_stat_identity(args.records)
    resume_mode = args.resume or (args.resume_if_present and state_path.is_file())
    input_identity = {'records': {'path': str(args.records.resolve()), 'sha256': args.records_sha256},
        'pieces': {'path': str(args.pieces.resolve()), 'sha256': pieces_digest.hexdigest()},
        'edges': {'path': str(args.edges.resolve()), 'sha256': args.edges_sha256}}
    # Keep the original normal-build input identity byte-for-byte compatible
    # with resume manifests written before pair-repair support was added. A
    # pair-only supplement still binds its exact omission receipt and source.
    if pair_only:
        input_identity['repair_pair_omissions'] = {
            'path': str(args.repair_pair_omissions.resolve()), 'sha256': repair_sha}
    tokenizer_models = tuple(tokenizers)
    tokenizer_identity = {model: {'source': source_fps[model], 'expanded': expanded_fps[model],
        'role_ids': list(role_ids[model]), 'model_id': model_ids[model],
        'base_inventory_sha256': base_inventory_shas[model]} for model in tokenizer_models}
    # The parent only needs model names and tokenizer fingerprints from here;
    # workers load their own tokenizers. Releasing the parent copies avoids
    # multiplying several GiB of tokenizer memory across the process pool.
    del tokenizers, tok
    chunks, totals, completed_chains, resume_offset = [], defaultdict(int), [], 0
    if resume_mode:
        if not state_path.is_file():
            raise ValueError('--resume requires a durable resume-manifest.json')
        previous = json.loads(state_path.read_text())
        if (previous.get('schema') != RESUME_SCHEMA
                or previous.get('build_policy') != BUILD_POLICY
                or previous.get('inputs') != input_identity
                or previous.get('tokenizers') != tokenizer_identity
                or previous.get('complete') is True):
            raise ValueError('resume manifest does not bind this incomplete exact source/tokenizer/policy build')
        # Trust the pinned file identity after the initial source SHA binding;
        # rehash only if an in-place update or replacement changed its stat key.
        if previous.get('records_stat') != records_stat and sha256_file(args.records) != args.records_sha256:
            raise ValueError('records source changed since the last durable conversion frontier')
        chunks = previous['chunks']
        completed_chains = previous['completed_chain_ids']
        if (len(completed_chains) != previous['completed_trajectories']
                or len(set(completed_chains)) != len(completed_chains)):
            raise ValueError('resume manifest trajectory frontier is inconsistent')
        for chunk in chunks:
            for part in _chunk_parts(chunk):
                path = Path(part['path']).resolve(strict=True)
                if path.parent != (args.out / 'chunks').resolve() and (args.out / 'chunks').resolve() not in path.parents:
                    raise ValueError('resume part escaped its owned output directory')
                if sha256_file(path) != part['sha256']:
                    raise ValueError('resume output chunk changed: ' + str(path))
        totals.update(previous['totals'])
        resume_offset = previous['resume_offset']
        for orphan in (args.out / 'chunks').glob('.*.tmp'):
            orphan.unlink(missing_ok=True)
    elif state_path.exists():
        raise ValueError('output already contains a resumable conversion; use --resume or --resume-if-present')
    elif (args.out / 'chunks').exists():
        chunks_root = args.out / 'chunks'
        leftovers = list(chunks_root.iterdir())
        if any(not path.is_file() or not path.name.startswith('.') or not path.name.endswith('.tmp')
               for path in leftovers):
            raise ValueError('output has unreceipted conversion chunks; inspect or choose a new output')
        for path in leftovers: path.unlink()
        chunks_root.rmdir()
    args.out.mkdir(parents=True, exist_ok=True)
    seen_chains = set(completed_chains)
    chunk_start_count = len(completed_chains)
    writers, pair_writers, omissions, render_omissions_writer = {}, {}, None, None

    def open_chunk(index):
        nonlocal writers, pair_writers, omissions, render_omissions_writer
        root = args.out / 'chunks'
        writers = ({} if pair_only else
            {split: {model: CompressedRows(root / f'graphs-{model}-{split}-{index:06d}.jsonl.zst')
                     for model in tokenizer_models} for split in ('train', 'test')})
        pair_writers = {split: CompressedRows(root / f'pairs-{split}-{index:06d}.jsonl.zst')
                        for split in ('train', 'test')}
        omissions = JsonlRows(root / f'omissions-{index:06d}.jsonl')
        render_omissions_writer = JsonlRows(root / f'render-omissions-{index:06d}.jsonl')

    def close_chunk(next_offset):
        nonlocal chunk_start_count
        index = len(chunks)
        current = {'graphs': ({} if pair_only else {model: {} for model in tokenizer_models}), 'pairs': {},
                   'omissions': None, 'render_omissions': None}
        try:
            if not pair_only:
                for model in tokenizer_models:
                    for split in ('train', 'test'):
                        current['graphs'][model][split] = writers[split][model].finish()
            for split in ('train', 'test'):
                current['pairs'][split] = pair_writers[split].finish()
            current['omissions'] = omissions.finish()
            current['render_omissions'] = render_omissions_writer.finish()
        except BaseException:
            for mapping in writers.values():
                for writer in mapping.values(): writer.abort()
            for writer in pair_writers.values(): writer.abort()
            omissions.abort()
            render_omissions_writer.abort()
            raise
        chunks.append(current)
        manifest = {'schema': RESUME_SCHEMA, 'build_policy': BUILD_POLICY,
            'inputs': input_identity, 'tokenizers': tokenizer_identity, 'records_stat': records_stat,
            'completed_trajectories': len(completed_chains), 'completed_chain_ids': completed_chains,
            'resume_offset': next_offset, 'totals': dict(totals), 'chunks': chunks, 'complete': False}
        atomic_json(state_path, manifest)
        chunk_start_count = len(completed_chains)
        open_chunk(len(chunks))

    def abort_chunk():
        for mapping in writers.values():
            for writer in mapping.values(): writer.abort()
        for writer in pair_writers.values(): writer.abort()
        if omissions is not None: omissions.abort()
        if render_omissions_writer is not None: render_omissions_writer.abort()

    open_chunk(len(chunks))
    current, chain_start = None, None


    def commit_result(chain, result, next_offset):
        if chain in seen_chains:
            raise ValueError('trajectory rows are not contiguous in their admitted source')
        seen_chains.add(chain)
        edges_by_chain.pop(chain, None)
        for name, count in result['totals'].items():
            totals[name] += count
        for split, by_model in result['graphs'].items():
            for model, rows in by_model.items():
                for graph in rows:
                    writers[split][model].write(graph)
        for split, rows in result['pairs'].items():
            for row in rows: pair_writers[split].write(row)
        for row in result['omissions']: omissions.write(row)
        for row in result.get('render_omissions', []): render_omissions_writer.write(row)
        completed_chains.append(chain)
        if len(completed_chains) - chunk_start_count >= args.checkpoint_trajectories:
            close_chunk(next_offset)

    worker_config = {'tokenizer_paths': {'lfm25': str(args.lfm_tokenizer),
                                         'mellum21': str(args.mellum_tokenizer)},
        'expanded_fps': expanded_fps, 'role_ids': {model: list(ids) for model, ids in role_ids.items()},
        'model_ids': model_ids, 'pair_only': pair_only,
        'selected_edge_ids': sorted(repair_edge_ids) if repair_edge_ids is not None else [],
        'records_path': str(args.records),
        'records_sha256': args.records_sha256, 'pieces_path': str(args.pieces),
        'pieces_sha256': pieces_digest.hexdigest(), 'edges_path': str(args.edges),
        'edges_sha256': args.edges_sha256}
    try:
        if resume_mode:
            completed_set = set(completed_chains)
            for chain in completed_set: edges_by_chain.pop(chain, None)
        last_offset = resume_offset
        pending = []
        stopped_at_limit = False

        def flush_pending(executor):
            nonlocal stopped_at_limit
            if not pending:
                return
            batch, pending[:] = list(pending), []
            results = list(executor.map(_render_trajectory, [(item[0], item[1], item[2]) for item in batch]))
            for (chain_id, _, _, safe_offset), result in zip(batch, results):
                commit_result(chain_id, result, safe_offset)
                if args.max_trajectories and len(completed_chains) >= args.max_trajectories:
                    stopped_at_limit = True
                    break

        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_render_worker, initargs=(worker_config,)) as executor:
            for record, line_start, line_stop in jsonl_with_offsets(args.records, resume_offset,
                    None if resume_mode else records_digest):
                chain = record.get('teacher_trajectory_id') or (record.get('source_ref') or {}).get('trajectory_id')
                if not isinstance(chain, str) or not chain:
                    raise ValueError('admitted E2E record lacks its complete trajectory identity')
                if current is None:
                    current, chain_start = chain, line_start
                if chain != current:
                    if current in edges_by_chain:
                        pending.append((current, chain_start, line_start, line_start))
                    current, chain_start = chain, line_start
                    remaining = (args.max_trajectories - len(completed_chains) - len(pending)
                                 if args.max_trajectories else args.workers)
                    until_checkpoint = args.checkpoint_trajectories - (len(completed_chains) - chunk_start_count)
                    batch_size = max(1, min(args.workers, remaining, until_checkpoint))
                    if len(pending) >= batch_size:
                        flush_pending(executor)
                    if stopped_at_limit:
                        last_offset = line_start
                        break
                last_offset = line_stop
            else:
                if current is not None and (not pair_only or current in edges_by_chain):
                    end_offset = Path(args.records).stat().st_size
                    pending.append((current, chain_start, end_offset, end_offset))
                flush_pending(executor)
                if not args.max_trajectories and not resume_mode and records_digest.hexdigest() != args.records_sha256:
                    raise ValueError('source records SHA differs from the admitted immutable source binding')
        if len(completed_chains) > chunk_start_count or not chunks:
            close_chunk(last_offset)
        if not args.max_trajectories:
            if edges_by_chain:
                missing = sum(map(len, edges_by_chain.values()))
                raise ValueError(f'{missing} admitted E2E edges have no corresponding source trajectory rows')
            if resume_mode and sha256_file(args.records) != args.records_sha256:
                raise ValueError('resumed source records SHA differs from the admitted immutable source binding')
            if not resume_mode and records_digest.hexdigest() != args.records_sha256:
                raise ValueError('source records SHA differs from the admitted immutable source binding')
            if pair_only and totals['source_edges'] != len(repair_edge_ids):
                raise ValueError('pair-repair did not revisit every exact edge in the prior omission ledger')
        abort_chunk()
    except BaseException:
        abort_chunk()
        raise

    def merge_parts(parts, destination):
        destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp', dir=destination.parent)
        with os.fdopen(fd, 'wb') as target:
            for part in parts:
                with open(part['path'], 'rb') as source:
                    while block := source.read(8 * 1024 * 1024): target.write(block)
            target.flush(); os.fsync(target.fileno())
        os.replace(tmp, destination)
        return {'path': str(destination), 'rows': sum(part['rows'] for part in parts),
                'bytes': destination.stat().st_size, 'sha256': sha256_file(destination)}

    graph_outputs = {} if pair_only else {model: {split: merge_parts(
        [chunk['graphs'][model][split] for chunk in chunks],
        args.out / 'graphs' / model / split / 'native-graphs.jsonl.zst')
        for split in ('train', 'test')} for model in tokenizer_models}
    pair_outputs = {split: merge_parts([chunk['pairs'][split] for chunk in chunks],
        args.out / 'pairs' / split / 'semantic-pairs.jsonl.zst') for split in ('train', 'test')}
    omissions_path = args.out / 'pair-omissions.jsonl'
    merge_parts([chunk['omissions'] for chunk in chunks], omissions_path)
    render_omissions_path = args.out / 'render-omissions.jsonl'
    render_omission_parts = [chunk['render_omissions'] for chunk in chunks if chunk.get('render_omissions')]
    render_omission_output = merge_parts(render_omission_parts, render_omissions_path)
    final_manifest = json.loads(state_path.read_text())
    final_manifest['complete'] = args.max_trajectories is None
    final_manifest['outputs'] = {'graphs': graph_outputs, 'pairs': pair_outputs}
    atomic_json(state_path, final_manifest)
    receipt = {'schema': BUILD_SCHEMA,
        'source': {'records': {'path': str(args.records), 'sha256': args.records_sha256},
                   'pieces': {'path': str(args.pieces), 'sha256': pieces_digest.hexdigest()},
                   'edges': {'path': str(args.edges), 'sha256': args.edges_sha256}},
        'tokenizers': {model: {'source': source_fps[model], 'expanded': expanded_fps[model],
                               'role_ids': list(role_ids[model])} for model in tokenizer_models},
        'model_ids': model_ids, 'base_inventory_sha256': base_inventory_shas,
        'counts': dict(totals), 'bounded_test_only': args.max_trajectories is not None,
        'outputs': {'graphs': graph_outputs, 'pairs': pair_outputs},
        'pair_omissions': {'path': str(omissions_path), 'rows': totals['pair_omissions'],
                           'sha256': sha256_file(omissions_path)},
        'render_omissions': render_omission_output,
        'pair_repair': ({'source_omissions': str(args.repair_pair_omissions),
                         'source_omissions_sha256': repair_sha,
                         'source_edge_count': len(repair_edge_ids),
                         'mode': 'pairs-only-static-multiread'} if pair_only else None),
        'resume_manifest': {'path': str(state_path), 'sha256': sha256_file(state_path)},
        'conversion': ('static multi-read repair supplement for exact edges omitted by a prior build'
            if pair_only else 'all rendered trajectories become ordinary native one-node SFT graphs; cross-model pairs use exact admitted writer/reader occurrences')}
    receipt_path = args.out / 'receipt.json'
    atomic_json(receipt_path, receipt)
    # The concatenated zstd frames are the single consumer representation;
    # per-chunk staging is removed after the complete output hashes are durable.
    if args.max_trajectories is None:
        import shutil
        shutil.rmtree(args.out / 'chunks')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__': main()
