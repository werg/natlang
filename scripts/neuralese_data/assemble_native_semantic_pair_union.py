#!/usr/bin/env python3
"""Join a native-pair build and its exact repair supplements without mutation.

Compressed zstd frames are concatenated after streaming every row through an
edge/split/group closure check. Pair rows remain byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex, sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable


def _file_binding(value, key):
    if not isinstance(value, dict) or not isinstance(value.get('path'), str):
        raise ValueError(f'missing {key} source binding')
    path = Path(value['path']).resolve(strict=True)
    actual = sha256_file_hex(path)
    if value.get('sha256') != actual:
        raise ValueError(f'{key} SHA-256 differs from its receipt')
    return path, actual


def _receipt(path):
    receipt_path = Path(path).resolve(strict=True)
    value = json.loads(receipt_path.read_text())
    if value.get('schema') != 'natlang.native-semantic-e2e-build/2':
        raise ValueError(f'unsupported pair-build receipt: {receipt_path}')
    return receipt_path, value


def _pair_parts(receipt, split):
    output = receipt.get('outputs', {}).get('pairs', {}).get(split)
    path, digest = _file_binding(output, split + ' pairs')
    return path, digest


def _insert_pair(db, row, split, model_ids):
    if (row.get('schema') != 'natlang.native-semantic-pair/1' or row.get('split') != split
            or not isinstance(row.get('id'), str) or not row['id'].startswith('e2e-edge:')
            or not isinstance(row.get('source_groups'), list) or not row['source_groups']
            or any(not isinstance(group, str) or not group for group in row['source_groups'])
            or set(row.get('models', {})) != set(model_ids)):
        raise ValueError('paired row has invalid schema, split, identity, groups or participants')
    communication = row.get('communication')
    if (not isinstance(communication, dict)
            or canonical_json_sha256_hex(communication) != row.get('communication_sha256')
            or not isinstance(communication.get('source_binding'), dict)):
        raise ValueError('paired row lacks its exact source communication binding')
    for field in ('source_id', 'writer_record_id', 'reader_record_id', 'write_name', 'occurrence'):
        if not isinstance(communication.get(field), str) or not communication[field]:
            raise ValueError('paired row communication lacks ' + field)
    def digests(value):
        found = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith('sha256'):
                    found.append(item)
                elif isinstance(item, dict):
                    found.extend(digests(item))
        return found
    source_digests = digests(communication['source_binding'])
    if not source_digests or any(not isinstance(value, str) or len(value) != 64
                                 or any(ch not in '0123456789abcdef' for ch in value)
                                 for value in source_digests):
        raise ValueError('paired row communication is not bound to an immutable source digest')
    edge_id = row['id'][len('e2e-edge:'):]
    if len(edge_id) != 64 or any(ch not in '0123456789abcdef' for ch in edge_id):
        raise ValueError('paired row id is not a source-edge SHA-256')
    if not isinstance(row.get('payload_sha256'), str) or len(row['payload_sha256']) != 64:
        raise ValueError('paired row has no payload identity')
    for name in model_ids:
        model = row['models'][name]
        if not isinstance(model, dict) or not isinstance(model.get('graph'), dict):
            raise ValueError('paired row has no full native graph for ' + name)
        graph = model['graph']
        if (graph.get('split') != split or graph.get('source_groups') != row['source_groups']
                or graph.get('schema') != 'natlang.neuralese-native-graph/1'):
            raise ValueError('participant graph disagrees with its semantic-pair row')
        witness, export = model.get('witness') or {}, model.get('export') or {}
        if (witness.get('source_id') != communication['source_id']
                or witness.get('communication_sha256') != row['communication_sha256']
                or witness.get('payload_sha256') != row['payload_sha256']
                or not isinstance(export.get('text'), str)
                or sha256_hex(export['text'].encode()) != row['payload_sha256']):
            raise ValueError('participant export/witness differs from the authenticated shared payload')
    try:
        db.execute('INSERT INTO pairs VALUES (?, ?, ?)', (edge_id, split, row['id']))
    except sqlite3.IntegrityError as exc:
        raise ValueError('duplicate source edge in pair builds: ' + edge_id) from exc
    for group in row['source_groups']:
        db.execute('INSERT OR IGNORE INTO groups VALUES (?, ?)', (split, group))


def _insert_omission(db, row, split):
    edge = row.get('edge')
    if not isinstance(edge, dict) or edge.get('split') != split:
        raise ValueError('omission does not bind an exact source edge in its declared split')
    edge_id = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    try:
        db.execute('INSERT INTO omissions VALUES (?, ?, ?)', (edge_id, split, row.get('reason', 'unspecified')))
    except sqlite3.IntegrityError as exc:
        raise ValueError('duplicate source-edge omission: ' + edge_id) from exc


def assemble(base_receipt_path, repair_receipt_paths, edges_path, out_dir):
    base_path, base = _receipt(base_receipt_path)
    repairs = [_receipt(path) for path in repair_receipt_paths]
    edge_path, _ = _file_binding({'path': str(edges_path),
                                  'sha256': base.get('source', {}).get('edges', {}).get('sha256')}, 'source edges')
    expected_source = base.get('source')
    if not isinstance(expected_source, dict):
        raise ValueError('base receipt lacks immutable source bindings')
    for path, repair in repairs:
        for name in ('records', 'pieces', 'edges'):
            if repair.get('source', {}).get(name) != expected_source.get(name):
                raise ValueError('repair receipt source differs from the original build: ' + name)
        base_omissions = base.get('pair_omissions') or {}
        repair_binding = repair.get('pair_repair') or {}
        if (repair_binding.get('mode') != 'pairs-only-static-multiread'
                or repair_binding.get('source_omissions_sha256') != base_omissions.get('sha256')):
            raise ValueError('repair does not bind the exact original omission ledger')
    model_ids = sorted(base.get('model_ids', {}).values())
    if len(model_ids) != 2 or len(set(model_ids)) != 2:
        raise ValueError('base build receipt must bind both actual model identities')

    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise ValueError('refusing to overwrite an existing pair-union output directory')
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = out_dir.with_name('.' + out_dir.name + f'.{os.getpid()}.tmp')
    if stage_dir.exists():
        raise ValueError('pair-union staging path already exists')
    stage_dir.mkdir()
    db_name = stage_dir / '.pair-union.sqlite'
    db = sqlite3.connect(db_name)
    db.execute('CREATE TABLE pairs (edge_id TEXT PRIMARY KEY, split TEXT NOT NULL, row_id TEXT NOT NULL)')
    db.execute('CREATE TABLE omissions (edge_id TEXT PRIMARY KEY, split TEXT NOT NULL, reason TEXT)')
    db.execute('CREATE TABLE groups (split TEXT NOT NULL, group_id TEXT NOT NULL, PRIMARY KEY(split, group_id))')
    pair_sources, omission_sources = [], []
    try:
        sources = [base] + [receipt for _, receipt in repairs]
        for source_index, receipt in enumerate(sources):
            for split in ('train', 'test'):
                pair_path, pair_sha = _pair_parts(receipt, split)
                pair_sources.append((split, pair_path, pair_sha))
                with open_jsonl_text(pair_path) as rows:
                    for line in rows:
                        if line.strip():
                            _insert_pair(db, json.loads(line), split, model_ids)
            omission = receipt.get('pair_omissions') or {}
            omission_path, omission_sha = _file_binding(omission, 'pair omissions')
            omission_sources.append((None, omission_path, omission_sha))
            with open_jsonl_text(omission_path) as rows:
                for line in rows:
                    if line.strip():
                        row = json.loads(line)
                        edge = row.get('edge')
                        if not isinstance(edge, dict):
                            raise ValueError('omission ledger row lacks its exact source edge')
                        _insert_omission(db, row, edge.get('split'))
        db.commit()

        source_edges = 0
        split_by_edge = {}
        with edge_path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                edge = json.loads(line)
                edge_id = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                split = edge.get('split')
                if split not in ('train', 'test') or edge.get('writer_split') != split:
                    raise ValueError('source edge has inconsistent or missing train/test binding')
                if edge_id in split_by_edge:
                    raise ValueError('duplicate admitted source edge identity: ' + edge_id)
                split_by_edge[edge_id] = split
                source_edges += 1
        observed_pairs = dict(db.execute('SELECT edge_id, split FROM pairs'))
        observed_omissions = dict(db.execute('SELECT edge_id, split FROM omissions'))
        if set(observed_pairs) | set(observed_omissions) != set(split_by_edge):
            missing = len(set(split_by_edge) - (set(observed_pairs) | set(observed_omissions)))
            extra = len((set(observed_pairs) | set(observed_omissions)) - set(split_by_edge))
            raise ValueError(f'pair/omission closure differs from source edges (missing={missing}, extra={extra})')
        if set(observed_pairs) & set(observed_omissions):
            raise ValueError('an admitted source edge is both trained and omitted')
        for edge_id, split in split_by_edge.items():
            if observed_pairs.get(edge_id, observed_omissions.get(edge_id)) != split:
                raise ValueError('edge train/test identity changed during conversion: ' + edge_id)
        train_groups = {row[0] for row in db.execute("SELECT group_id FROM groups WHERE split='train'")}
        held_groups = {row[0] for row in db.execute("SELECT group_id FROM groups WHERE split='test'")}
        overlap = train_groups & held_groups
        if overlap:
            raise ValueError('semantic pair source groups cross train/held: ' + ', '.join(sorted(overlap)[:8]))

        outputs = {}
        for split in ('train', 'test'):
            staged_destination = stage_dir / split / 'semantic-pairs.jsonl.zst'
            destination = out_dir / split / 'semantic-pairs.jsonl.zst'
            staged_destination.parent.mkdir(parents=True, exist_ok=True)
            temp = staged_destination.with_name('.' + staged_destination.name + '.tmp')
            rows = db.execute('SELECT COUNT(*) FROM pairs WHERE split=?', (split,)).fetchone()[0]
            with temp.open('wb') as target:
                for source_split, source_path, _ in pair_sources:
                    if source_split != split:
                        continue
                    with source_path.open('rb') as source:
                        while block := source.read(8 * 1024 * 1024):
                            target.write(block)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp, staged_destination)
            outputs[split] = {'path': str(destination), 'rows': rows, 'bytes': staged_destination.stat().st_size,
                              'sha256': sha256_file_hex(staged_destination), 'encoding': 'concatenated-zstd-frames'}
        omission_counts = {split: db.execute('SELECT COUNT(*) FROM omissions WHERE split=?', (split,)).fetchone()[0]
                           for split in ('train', 'test')}
        pair_counts = {split: db.execute('SELECT COUNT(*) FROM pairs WHERE split=?', (split,)).fetchone()[0]
                       for split in ('train', 'test')}
        receipt = {'schema': 'natlang.native-semantic-pair-union/1',
            'source_build_receipt': {'path': str(base_path), 'sha256': sha256_file_hex(base_path)},
            'repair_receipts': [{'path': str(path), 'sha256': sha256_file_hex(path)} for path, _ in repairs],
            'source_edges': {'path': str(edge_path), 'sha256': expected_source['edges']['sha256'], 'rows': source_edges},
            'model_ids': model_ids, 'outputs': outputs,
            'counts': {'pairs': pair_counts, 'unresolved_omissions': omission_counts,
                       'source_edges': source_edges, 'closed': True},
            'omission_sources': [{'path': str(path), 'sha256': digest}
                                 for split, path, digest in omission_sources],
            'conversion': 'exact-source-edge pair rows and explicit unresolved omissions; no source edge dropped'}
        write_json_durable(stage_dir / 'receipt.json', receipt)
        db.close()
        db_name.unlink(missing_ok=True)
        os.replace(stage_dir, out_dir)
        return receipt
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    finally:
        db.close()
        Path(db_name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-receipt', type=Path, required=True)
    parser.add_argument('--repair-receipt', type=Path, action='append', default=[])
    parser.add_argument('--edges', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.base_receipt, args.repair_receipt, args.edges, args.out), sort_keys=True))


if __name__ == '__main__':
    main()
