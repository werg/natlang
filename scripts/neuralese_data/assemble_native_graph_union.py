#!/usr/bin/env python3
"""Create immutable append-only native graph inputs from two admitted builds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable


def _bound(binding, label):
    if not isinstance(binding, dict) or not isinstance(binding.get('path'), str):
        raise ValueError(f'{label} lacks a path binding')
    path = Path(binding['path']).resolve(strict=True)
    digest = sha256_file_hex(path)
    if digest != binding.get('sha256'):
        raise ValueError(f'{label} differs from its admitted SHA-256')
    return path, digest


def _json(path):
    return json.loads(Path(path).resolve(strict=True).read_text())


def _validate_full_manifest(base_path, base):
    """Prove the FULL receipt and every advertised graph are the admitted files."""
    manifest_ref = base.get('corpus_manifest')
    if not isinstance(manifest_ref, str) or not manifest_ref:
        raise ValueError('FULL build has no immutable corpus-manifest binding')
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    manifest = _json(manifest_path)
    if (manifest.get('schema') != 'natlang.corpus-snapshot/1'
            or manifest.get('id') != base.get('id')):
        raise ValueError('FULL corpus manifest does not bind this build identity')
    run_path = (manifest_path.parents[2] / manifest['path']).resolve(strict=True)
    if run_path != base_path.parent:
        raise ValueError('FULL receipt is outside its admitted corpus path')
    advertised = {item.get('path'): item for item in manifest.get('files', [])
                  if isinstance(item, dict) and isinstance(item.get('path'), str)}
    if len(advertised) != len(manifest.get('files', [])):
        raise ValueError('FULL corpus manifest contains malformed or duplicate file bindings')

    def verify(relative, expected_sha):
        item = advertised.get(relative)
        if not isinstance(item, dict) or item.get('sha256') != expected_sha:
            raise ValueError(f'FULL manifest omits or changes file binding: {relative}')
        target = (run_path / relative).resolve(strict=True)
        if not target.is_relative_to(run_path):
            raise ValueError('FULL manifest path escapes its corpus root')
        if sha256_file_hex(target) != expected_sha:
            raise ValueError(f'FULL immutable file digest differs: {relative}')

    verify('receipt.json', sha256_file_hex(base_path))
    for alias, model in base.get('models', {}).items():
        child_rel = model.get('child_receipt')
        child_path = (base_path.parent / child_rel).resolve(strict=True)
        if not child_path.is_relative_to(run_path):
            raise ValueError(f'{alias} child receipt escapes FULL corpus root')
        verify(str(child_path.relative_to(run_path)), sha256_file_hex(child_path))
        child = _json(child_path)
        for row in child.get('outputs', []):
            target = (child_path.parent / row['path']).resolve(strict=True)
            if not target.is_relative_to(run_path):
                raise ValueError(f'{alias} graph output escapes FULL corpus root')
            verify(str(target.relative_to(run_path)), row['sha256'])


def _paths(base_receipt_path, added_receipt_path, alias):
    base_path = Path(base_receipt_path).resolve(strict=True)
    base = _json(base_path)
    if base.get('schema') != 'natlang.native-graph-full-build/1':
        raise ValueError('base input is not an admitted FULL native graph build')
    _validate_full_manifest(base_path, base)
    model = base.get('models', {}).get(alias)
    if not isinstance(model, dict):
        raise ValueError('FULL build lacks participant ' + alias)
    child = _json(base_path.parent / model['child_receipt'])
    added_path = Path(added_receipt_path).resolve(strict=True)
    added = _json(added_path)
    if added.get('schema') != 'natlang.native-semantic-e2e-build/2':
        raise ValueError('addition input is not a complete native E2E graph build')
    if model.get('model_id') != added.get('model_ids', {}).get(alias):
        raise ValueError('FULL and added graph builds have different exact model inventories')
    if child.get('expanded_tokenizer_fingerprint') != added.get('tokenizers', {}).get(alias, {}).get('expanded'):
        raise ValueError('FULL and added graph builds have different role-expanded tokenizers')
    full_rows = {item['split']: item for item in child['outputs']}
    added_rows = added.get('outputs', {}).get('graphs', {}).get(alias, {})
    resolved = {}
    for split in ('train', 'test'):
        old = full_rows.get(split)
        new = added_rows.get(split)
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise ValueError('both builds must have train and protected held graphs')
        old_path = (base_path.parent / model['child_receipt']).parent / old['path']
        old_binding = {'path': str(old_path), 'sha256': old['sha256'], 'rows': old['rows']}
        old_path, _ = _bound(old_binding, f'{alias} FULL {split}')
        new_path, _ = _bound(new, f'{alias} addition {split}')
        resolved[split] = (old_path, old, new_path, new)
    return base, added, child, model, resolved


def _scan_graphs(db, path, split, expected_fingerprint, model, label):
    count = 0
    with open_jsonl_text(path) as rows:
        for line_number, line in enumerate(rows, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get('schema') != 'natlang.neuralese-native-graph/1'
                    or row.get('split') != split or not isinstance(row.get('id'), str)
                    or not row['id'] or row.get('tokenizer_fingerprint') != expected_fingerprint):
                raise ValueError(f'{label}:{line_number}: invalid graph identity/split/tokenizer')
            groups = row.get('source_groups')
            if (not isinstance(groups, list) or not groups
                    or any(not isinstance(group, str) or not group for group in groups)):
                raise ValueError(f'{label}:{line_number}: missing factual source groups')
            row_sha = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()
            try:
                db.execute('INSERT INTO graph_rows VALUES (?, ?, ?, ?, ?)', (model, row['id'], split, label, row_sha))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f'{label}:{line_number}: duplicate graph ID across union: {row["id"]}') from exc
            for group in groups:
                db.execute('INSERT OR IGNORE INTO groups VALUES (?, ?)', (split, group))
            count += 1
    return count


def assemble(base_receipt, added_receipt, out_dir):
    base_receipt = Path(base_receipt).resolve(strict=True)
    added_receipt = Path(added_receipt).resolve(strict=True)
    output = Path(out_dir).resolve()
    if output.exists():
        raise ValueError('refusing to overwrite an existing graph union')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name('.' + output.name + f'.{os.getpid()}.tmp')
    if stage.exists():
        raise ValueError('graph-union staging path already exists')
    stage.mkdir()
    database = stage / '.graph-union.sqlite'
    db = sqlite3.connect(database)
    db.execute('CREATE TABLE graph_rows (model TEXT NOT NULL, id TEXT NOT NULL, split TEXT NOT NULL, '
               'build TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(model, id))')
    db.execute('CREATE TABLE groups (split TEXT NOT NULL, group_id TEXT NOT NULL, PRIMARY KEY(split, group_id))')
    try:
        summaries, inputs = {}, {}
        base_data, added_data = _json(base_receipt), _json(added_receipt)
        if base_data.get('schema') != 'natlang.native-graph-full-build/1':
            raise ValueError('base input is not an admitted FULL graph build')
        _validate_full_manifest(base_receipt, base_data)
        if added_data.get('schema') != 'natlang.native-semantic-e2e-build/2':
            raise ValueError('addition input is not a complete E2E graph build')
        for alias in ('lfm25', 'mellum21'):
            _, added, full_child, model, splits = _paths(base_receipt, added_receipt, alias)
            fingerprint = full_child['expanded_tokenizer_fingerprint']
            if added['tokenizers'][alias]['expanded'] != fingerprint:
                raise ValueError(f'{alias} addition changed the expanded tokenizer')
            summaries[alias], inputs[alias] = {}, {}
            counts = {}
            for split, (old_path, old_record, new_path, new_record) in splits.items():
                old_count = _scan_graphs(db, old_path, split, fingerprint, alias, alias + ':FULL:' + split)
                new_count = _scan_graphs(db, new_path, split, fingerprint, alias, alias + ':REST:' + split)
                if old_count != old_record['rows'] or new_count != new_record['rows']:
                    raise ValueError(f'{alias} {split} graph row count differs from its immutable receipt')
                destination = stage / alias / split / 'native-graphs.jsonl.zst'
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('wb') as target:
                    for source in (old_path, new_path):
                        with source.open('rb') as stream:
                            while block := stream.read(8 * 1024 * 1024):
                                target.write(block)
                    target.flush()
                    os.fsync(target.fileno())
                final_path = output / alias / split / destination.name
                rows = old_count + new_count
                counts[split] = rows
                summaries[alias][split] = {'path': str(final_path), 'rows': rows,
                    'bytes': destination.stat().st_size, 'sha256': sha256_file_hex(destination),
                    'full_prefix_rows': old_count, 'appended_rows': new_count,
                    'full_prefix_sha256': old_record['sha256'], 'appended_sha256': new_record['sha256']}
                inputs[alias][split] = {'full': {'path': str(old_path), 'sha256': old_record['sha256']},
                    'appended': {'path': str(new_path), 'sha256': new_record['sha256']}}
            alias_by_name = {'lfm25': 'lfm2', 'mellum21': 'mellum'}
            summaries[alias]['model_id'] = model['model_id']
            summaries[alias]['tokenizer_fingerprint'] = fingerprint
            summaries[alias]['model_architecture'] = alias_by_name[alias]
        db.commit()
        train_groups = {row[0] for row in db.execute("SELECT group_id FROM groups WHERE split='train'")}
        held_groups = {row[0] for row in db.execute("SELECT group_id FROM groups WHERE split='test'")}
        overlap = train_groups & held_groups
        if overlap:
            raise ValueError('FULL+REST source groups cross train/held: ' + ', '.join(sorted(overlap)[:12]))
        counts_by_split = {split: db.execute('SELECT COUNT(*) FROM graph_rows WHERE split=?', (split,)).fetchone()[0]
                           for split in ('train', 'test')}
        receipt = {'schema': 'natlang.native-graph-append-union/1',
            'base_receipt': {'path': str(base_receipt), 'sha256': sha256_file_hex(base_receipt)},
            'addition_receipt': {'path': str(added_receipt), 'sha256': sha256_file_hex(added_receipt)},
            'models': summaries, 'inputs': inputs,
            'counts': {'per_split': counts_by_split, 'train_source_groups': len(train_groups),
                       'held_source_groups': len(held_groups), 'train_held_group_intersection': 0,
                       'rows_unique_by_id': True},
            'schedule': 'FULL rows remain the exact first compressed zstd frame; all admitted addition rows follow in source order',
            'training_admission': 'derived exact append-only continuation input; held rows remain in a separate test split'}
        write_json_durable(stage / 'receipt.json', receipt)
        db.close()
        database.unlink(missing_ok=True)
        os.replace(stage, output)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        db.close()
        database.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-receipt', type=Path, required=True)
    parser.add_argument('--addition-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.full_receipt, args.addition_receipt, args.out), sort_keys=True))


if __name__ == '__main__':
    main()
