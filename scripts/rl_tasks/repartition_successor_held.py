#!/usr/bin/env python3
"""Repair cross-split related components in an unpublished successor holdout extension.

Copies an already qualified successor by hardlink, preserving its train source and historical held rows byte for
byte. Only newly appended held cases are reassigned, component-wide, to the majority of their original split labels
(ties go to confirmation). The repair fails closed on overlap with training, protected worlds, or opposite held split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
from pathlib import Path

from natlang_neuralese.common.custody import ProtectedWorlds, related_components, world_identities


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def row_sha(row):
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def replace_json(path: Path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(temp, path)


def write_rows(path: Path, rows):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ready', required=True, type=Path)
    ap.add_argument('--v3-held-package', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    ready = json.loads(args.ready.read_text())
    source = Path(ready['output']).resolve()
    if file_sha(source / 'qualification.json') != ready['qualification_sha256']:
        raise SystemExit('source qualification hash differs from ready pointer')
    if file_sha(source / 'cases.jsonl') != ready['cases_sha256']:
        raise SystemExit('source cases hash differs from ready pointer')
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f'output exists: {out}')

    conflicts = read_rows(source / 'split-conflicts.jsonl')
    if not conflicts:
        raise SystemExit('no held split conflicts to repair')
    conflict_ids = {case_id for conflict in conflicts for case_id in conflict['case_ids']}
    verification_rows = {}
    verification_hashes = {}
    for report_path in sorted((args.v3_held_package / 'verification').glob('*.json')):
        report = json.loads(report_path.read_text())
        matched = False
        for result in report.get('results', []):
            if result.get('id') in conflict_ids:
                verification_rows[result['id']] = {'ok': result.get('ok'), 'reference_ok': result.get('reference_ok'),
                    'wrong_rejected': result.get('wrong_rejected'), 'report': report_path.name}
                matched = True
        if matched:
            verification_hashes[report_path.name] = file_sha(report_path)
    missing_or_failed = [case_id for case_id in conflict_ids
                         if case_id not in verification_rows or not verification_rows[case_id].get('ok')]
    if missing_or_failed:
        raise SystemExit(f'held candidates lack passing exact verifier evidence: {missing_or_failed[:8]}')
    candidate = {}
    source_split = {}
    for split in ('dev', 'confirmation'):
        path = args.v3_held_package / 'held' / f'{split}.jsonl'
        for row in read_rows(path):
            if row['id'] in candidate:
                raise SystemExit(f'duplicate candidate held id: {row["id"]}')
            candidate[row['id']] = row
            source_split[row['id']] = split

    existing = {split: read_rows(source / 'held' / f'{split}.jsonl')
                for split in ('dev', 'confirmation')}
    historical = {split: {'sha256': file_sha(source / 'held' / f'{split}.jsonl'),
                          'rows': len(existing[split]), 'ids_sha256': hashlib.sha256(
                              '\n'.join(r['id'] for r in existing[split]).encode()).hexdigest()}
                  for split in existing}
    existing_ids = {row['id'] for rows in existing.values() for row in rows}
    train_identities = set()
    for line in (source / 'cases.jsonl').open():
        if line.strip():
            train_identities.update(world_identities(json.loads(line), evidence_items=False))
    protected = ProtectedWorlds.load()
    reassignments = []
    added = {'dev': [], 'confirmation': []}
    all_candidate = []
    for conflict in conflicts:
        ids = conflict['case_ids']
        try:
            rows = [candidate[case_id] for case_id in ids]
        except KeyError as exc:
            raise SystemExit(f'conflict case absent from v3 held package: {exc}') from exc
        counts = collections.Counter(source_split[row['id']] for row in rows)
        target = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        component_ids = sorted(ids)
        component_groups = sorted({group for row in rows for group in row.get('source_groups', [])})
        component_rows = []
        for row in rows:
            if row['id'] in existing_ids:
                raise SystemExit(f'held row already exists in source: {row["id"]}')
            before = row_sha(row)
            moved = json.loads(json.dumps(row))
            moved['source_split_before_merge'] = source_split[row['id']]
            moved['split'] = target
            after = row_sha(moved)
            entry = {'case_id': row['id'], 'source_split': source_split[row['id']], 'assigned_split': target,
                     'old_row_sha256': before, 'new_row_sha256': after}
            reassignments.append(entry)
            added[target].append(moved)
            component_rows.append(moved)
            all_candidate.append(moved)
        component_identities = set().union(*(world_identities(row, evidence_items=False) for row in component_rows))
        if component_identities & train_identities:
            raise SystemExit(f'candidate component overlaps train: {component_ids[0]}')
        if component_identities & protected.identities:
            raise SystemExit(f'candidate component overlaps protected source: {component_ids[0]}')

    # Reject any component assignment that introduces a related world into the opposite held partition.
    for split, opposite in (('dev', 'confirmation'), ('confirmation', 'dev')):
        opposite_ids = set().union(*(world_identities(row, evidence_items=False) for row in existing[opposite]))
        if any(world_identities(row, evidence_items=False) & opposite_ids for row in added[split]):
            raise SystemExit(f'reassigned {split} component overlaps historical {opposite} held worlds')
    combined = {split: existing[split] + added[split] for split in existing}
    historical_cross = (set().union(*(world_identities(row, evidence_items=False) for row in existing['dev'])) &
                        set().union(*(world_identities(row, evidence_items=False) for row in existing['confirmation'])))
    final_cross = (set().union(*(world_identities(row, evidence_items=False) for row in combined['dev'])) &
                   set().union(*(world_identities(row, evidence_items=False) for row in combined['confirmation'])))
    newly_introduced_cross = final_cross - historical_cross
    if newly_introduced_cross:
        raise SystemExit(f'repartition adds cross-split identities: {sorted(k for k, _ in newly_introduced_cross)}')

    # Candidate held components have no overlap with one another by construction; confirm against every input row.
    if len({row['id'] for rows in combined.values() for row in rows}) != sum(map(len, combined.values())):
        raise SystemExit('duplicate id in final held packets')

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, out, copy_function=os.link)
    for split in ('dev', 'confirmation'):
        write_rows(out / 'held' / f'{split}.jsonl', combined[split])
    write_rows(out / 'split-conflicts.jsonl', [])
    write_rows(out / 'held-split-reassignments.jsonl', reassignments)

    qualification = json.loads((out / 'qualification.json').read_text())
    qualification['source_id'] = out.name
    qualification['source_transition']['new_held_split_conflicts'] = 0
    qualification['source_transition']['new_held_split_reassignments'] = len(reassignments)
    qualification['source_transition']['held_component_assignment_rule'] = \
        'majority original split per complete related component; ties to confirmation'
    qualification['held'] = {split: {'rows': len(combined[split]),
        'sha256': file_sha(out / 'held' / f'{split}.jsonl')} for split in combined}
    qualification['current_protection_bindings']['own_held_rows'] = sum(map(len, combined.values()))
    qualification['held_split_reassignments'] = {
        'path': 'held-split-reassignments.jsonl', 'rows': len(reassignments),
        'sha256': file_sha(out / 'held-split-reassignments.jsonl'),
        'components': len(conflicts), 'historical_prefixes_unchanged': historical,
        'source_held_inputs': {split: {'sha256': file_sha(args.v3_held_package / 'held' / f'{split}.jsonl'),
                                      'rows': sum(source_split[r['id']] == split for r in all_candidate)}
                              for split in ('dev', 'confirmation')},
        'train_overlaps': 0, 'protected_overlaps': 0, 'new_cross_split_overlaps': 0,
        'preexisting_historical_cross_split_identity_count': len(historical_cross),
        'historical_cross_split_identities_modified': 0,
        'source_verification': {'cases': len(verification_rows), 'all_passed': True,
            'reports': verification_hashes,
            'results': verification_rows}}
    qualification['split_conflicts_sha256'] = file_sha(out / 'split-conflicts.jsonl')
    replace_json(out / 'qualification.json', qualification)

    receipt = json.loads((out / 'receipt.json').read_text())
    receipt['source_id'] = out.name
    receipt['held'] = qualification['held']
    receipt['split_conflicts'] = {'path': 'split-conflicts.jsonl',
        'sha256': qualification['split_conflicts_sha256'], 'rows': 0}
    receipt['held_split_reassignments'] = {'path': 'held-split-reassignments.jsonl',
        'sha256': qualification['held_split_reassignments']['sha256'], 'rows': len(reassignments)}
    replace_json(out / 'receipt.json', receipt)

    verification = json.loads((out / 'verification.json').read_text())
    verification['held_component_conflicts'] = 0
    verification['held_component_reassignments'] = len(conflicts)
    replace_json(out / 'verification.json', verification)
    pointer = args.out.with_suffix(args.out.suffix + '.ready.json')
    replace_json(pointer, {'schema': 'natlang.retryable-assembly-ready/1', 'output': str(out),
        'qualification_sha256': file_sha(out / 'qualification.json'),
        'cases_sha256': file_sha(out / 'cases.jsonl'), 'cases_rows': ready['cases_rows'],
        'held_reassigned_rows': len(reassignments)})
    print(json.dumps({'output': str(out), 'ready': str(pointer), 'train_rows': ready['cases_rows'],
        'held': qualification['held'], 'reassigned_rows': len(reassignments),
        'components': len(conflicts), 'historical_held_prefixes_unchanged': historical,
        'train_protected_cross_split_overlaps': 0}, indent=2))


if __name__ == '__main__':
    main()
