#!/usr/bin/env python3
"""Project the authored semantic priority snapshot; reuse the existing campaign runner.

Exact eligibility and identity checks only. Semantic policy is the supplied NL
plan and its explicit order. The output is generation input, never admission.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked_hash(path, expected):
    actual = sha256_file_hex(path)
    if actual != expected:
        raise ValueError(f'Pinned identity changed: {path}; expected {expected}, actual {actual}')
    return actual


def materialize(args):
    source = args.source.resolve(strict=True)
    review_path = args.source_review.resolve(strict=True)
    source_hash = checked_hash(source, args.source_sha256)
    review_hash = checked_hash(review_path, args.review_sha256)
    plan = read_json(args.priority)
    if plan.get('schema') != 'natlang.semantic-stack-priority-plan/1':
        raise ValueError('Unsupported priority plan')
    review = read_json(review_path)
    if review.get('source_sha256') != source_hash or review.get('training_admission') is not False:
        raise ValueError('Source review is not bound to these source bytes and non-admission state')
    catalog_index, catalog_pins = {}, {}
    for catalog_path in plan['catalogs']:
        path = ROOT / catalog_path
        catalog_pins[catalog_path] = sha256_file_hex(path)
        value = read_json(path)
        for entry in value if isinstance(value, list) else value['programs']:
            if entry['id'] in catalog_index:
                raise ValueError(f"Duplicate authored program ID: {entry['id']}")
            catalog_index[entry['id']] = entry
    proof_index = {}
    for proof in review['cases']:
        key = f"semantic-stack:{proof['provider']}:{proof['id']}"
        if key in proof_index:
            raise ValueError(f'Duplicate source review ID: {key}')
        proof_index[key] = proof
    rows, row_index = [], {}
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in row_index:
                raise ValueError(f"Duplicate source case ID: {row['id']}")
            row_index[row['id']] = row
            rows.append(row)
    selected, pending = [], []
    order = plan['ordered_source_ids']
    if len(order) != len(set(order)):
        raise ValueError('Priority order contains duplicate source IDs')
    for source_id in order:
        short_id = source_id.split(':', 2)[-1]
        entry = catalog_index.get(short_id)
        row = row_index.get(source_id)
        proof = proof_index.get(source_id)
        reasons = []
        if entry is None or entry.get('generation_ready') is not True:
            reasons.append('authored_fixture_not_generation_ready')
        if row is None:
            reasons.append('not_in_reference_qualified_source')
        if proof is None or proof.get('reference_ok') is not True:
            reasons.append('reference_proof_absent_or_failed')
        if reasons:
            pending.append({'source_id': source_id, 'reasons': reasons})
            continue
        fixture = entry['source_fixture']
        record_count = fixture.get('record_count')
        if record_count is None:
            records = fixture.get('records')
            record_count = len(records) if isinstance(records, list) else len(fixture['files']) - 1
        if record_count < plan['eligibility']['sizable_records_at_least']:
            raise ValueError(f'{source_id}: fixture has only {record_count} records')
        if row.get('split') != 'train' or row.get('collection_guidance', {}).get('training_admission') is not False:
            raise ValueError(f'{source_id}: split/non-admission state differs')
        semantics = row['semantics']
        if semantics.get('folder_files') != fixture['files']:
            raise ValueError(f'{source_id}: source-visible fixture differs from current catalog')
        oracle = entry.get('oracle')
        if not isinstance(oracle, dict) or 'expected' not in oracle or semantics.get('expected') != oracle['expected']:
            raise ValueError(f'{source_id}: exact oracle differs from authored catalog')
        selected.append(row)
    if not selected:
        raise ValueError('No eligible prepared source rows; pending reasons: ' + json.dumps(pending))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    selected_source = out / 'priority-source.ir.jsonl'
    selected_source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    manifest = {
        'schema': 'natlang.semantic-stack-priority-selection/1',
        'policy': plan['policy'],
        'priority_plan': str(args.priority.resolve()),
        'priority_plan_sha256': sha256_file_hex(args.priority),
        'catalog_sha256': catalog_pins,
        'parent_source': str(source), 'parent_source_sha256': source_hash,
        'parent_source_review': str(review_path), 'parent_source_review_sha256': review_hash,
        'selected_source': str(selected_source), 'selected_source_sha256': sha256_file_hex(selected_source),
        'selected_ids': [row['id'] for row in selected],
        'pending': pending,
        'unselected_reference_cases': [row['id'] for row in rows if row['id'] not in order],
        'teacher_observations_credited': 0, 'training_admission': False,
        'next': 'Replay and decontaminate selected bytes, then use existing prepare_luna_folder_campaign.py. Qualify actual call topology in a diagnostic before full teacher collection.',
    }
    manifest_path = out / 'priority-selection.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'selected_source': str(selected_source), 'selected': len(selected), 'pending': len(pending), 'sha256': manifest['selected_source_sha256'], 'manifest': str(manifest_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--priority', type=Path, default=ROOT / 'training/semantic_stack/sol_priority_queue.json')
    parser.add_argument('--out', type=Path, required=True, help='Fresh output directory')
    materialize(parser.parse_args())


if __name__ == '__main__':
    main()
