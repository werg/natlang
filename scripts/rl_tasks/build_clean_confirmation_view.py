#!/usr/bin/env python3
"""Derive a confirmation slice disjoint from development while preserving all held rows for protection.

Historical dev/confirmation packets remain immutable. The clean view excludes complete confirmation components
related to any development component and keeps every excluded row in `protected-held-union.jsonl`; it never adds a
held row to training. Independence is claimed only under the bound world-identity custody rules.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from natlang_neuralese.common.custody import related_components, world_identities
from natlang_neuralese.common.hashing import sha256_file_hex as sha


def load_rows(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def write_rows(path: Path, rows):
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def row_sha(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def identity_subtype(identity):
    kind, value = identity
    if kind != 'binding':
        return kind
    for prefix in ('source:', 'inputs:', 'service-world:'):
        if value.startswith(prefix):
            return prefix[:-1]
    return 'other-binding'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ready', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    ready = json.loads(args.ready.read_text())
    source = Path(ready['output']).resolve()
    qualification = json.loads((source / 'qualification.json').read_text())
    if sha(source / 'cases.jsonl') != ready['cases_sha256']:
        raise SystemExit('source train cases hash differs from ready pointer')
    if sha(source / 'qualification.json') != ready['qualification_sha256']:
        raise SystemExit('source qualification hash differs from ready pointer')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    dev_path = source / 'held' / 'dev.jsonl'
    conf_path = source / 'held' / 'confirmation.jsonl'
    dev = load_rows(dev_path)
    confirmation = load_rows(conf_path)
    if any(row.get('split') != 'dev' for row in dev) or any(row.get('split') != 'confirmation' for row in confirmation):
        raise SystemExit('source held rows have split labels inconsistent with their packet')
    dev_ids = set().union(*(world_identities(row, evidence_items=False) for row in dev))

    clean = []
    exclusions = []
    exclusion_type_counts = collections.Counter()
    for component in related_components(confirmation,
            identities=lambda row: world_identities(row, evidence_items=False)):
        rows = [confirmation[index] for index in component]
        overlaps = set().union(*(world_identities(row, evidence_items=False) for row in rows)) & dev_ids
        if overlaps:
            kinds = collections.Counter(identity_subtype(identity) for identity in overlaps)
            exclusion_type_counts.update(kinds)
            exclusions.append({'case_ids': sorted(row['id'] for row in rows),
                'row_sha256': {row['id']: row_sha(row) for row in rows},
                'shared_dev_identities': [{'kind': kind, 'value': value} for kind, value in sorted(overlaps)],
                'shared_identity_types': dict(kinds), 'reason': 'whole confirmation component relates to development'})
        else:
            clean.extend(rows)

    clean_ids = {row['id'] for row in clean}
    excluded_ids = {case_id for record in exclusions for case_id in record['case_ids']}
    if clean_ids & excluded_ids or clean_ids | excluded_ids != {row['id'] for row in confirmation}:
        raise SystemExit('clean/excluded confirmation IDs do not exactly partition the historical confirmation packet')
    clean_ids_set = set().union(*(world_identities(row, evidence_items=False) for row in clean))
    overlap_after = clean_ids_set & dev_ids
    if overlap_after:
        raise SystemExit(f'clean confirmation view still overlaps dev: {sorted(overlap_after)[:4]}')

    # Full original held union remains protected, including every row omitted from the clean evaluation view.
    protected_union = dev + confirmation
    protected_ids = {row['id'] for row in protected_union}
    if not excluded_ids <= protected_ids or clean_ids & excluded_ids:
        raise SystemExit('excluded confirmation rows are missing from the protected held union')

    write_rows(out / 'dev.jsonl', dev)
    write_rows(out / 'confirmation.jsonl', clean)
    write_rows(out / 'protected-held-union.jsonl', protected_union)
    write_rows(out / 'confirmation-exclusions.jsonl', exclusions)
    view = {'schema': 'natlang.clean-confirmation-view/1',
        'source_ready_path': str(args.ready.resolve()), 'source_ready_sha256': sha(args.ready),
        'source_id': qualification['source_id'], 'source_cases_sha256': ready['cases_sha256'],
        'source_qualification_sha256': ready['qualification_sha256'],
        'historical_dev': {'rows': len(dev), 'sha256': sha(dev_path), 'output_sha256': sha(out / 'dev.jsonl')},
        'historical_confirmation': {'rows': len(confirmation), 'sha256': sha(conf_path)},
        'clean_confirmation': {'rows': len(clean), 'sha256': sha(out / 'confirmation.jsonl')},
        'excluded_confirmation': {'rows': len(excluded_ids), 'components': len(exclusions),
            'path': 'confirmation-exclusions.jsonl', 'sha256': sha(out / 'confirmation-exclusions.jsonl'),
            'identity_type_counts': dict(exclusion_type_counts)},
        'protected_held_union': {'rows': len(protected_union), 'unique_ids': len(protected_ids),
            'path': 'protected-held-union.jsonl', 'sha256': sha(out / 'protected-held-union.jsonl'),
            'excluded_confirmation_rows_retained': len(excluded_ids)},
        'custody': {'identity_function': 'natlang_neuralese.common.custody.world_identities(evidence_items=False)',
            'dev_confirmation_shared_identities_before': len(dev_ids & set().union(*(
                world_identities(row, evidence_items=False) for row in confirmation))),
            'dev_confirmation_shared_identities_after': len(overlap_after),
            'excluded_rows_added_to_training': 0,
            'claim': 'disjoint under bound custody identities; semantic independence remains unjudged'},
        'runtime_assets': {'NATLANG_RL_TASK_ASSETS': str(source / 'assets'),
            'NATLANG_RL_TASK_ASSETS_V3': str(source / 'assets-v3')}}
    (out / 'view.json').write_text(json.dumps(view, indent=2) + '\n')
    (args.out.with_suffix(args.out.suffix + '.ready.json')).write_text(json.dumps({
        'schema': 'natlang.clean-confirmation-view-ready/1', 'output': str(out),
        'view_sha256': sha(out / 'view.json'), 'confirmation_sha256': view['clean_confirmation']['sha256'],
        'protected_union_sha256': view['protected_held_union']['sha256'],
        'source_cases_sha256': ready['cases_sha256']}, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'clean_confirmation_rows': len(clean),
        'excluded_rows': len(excluded_ids), 'excluded_components': len(exclusions),
        'preserved_protected_rows': len(protected_union), 'dev_confirmation_overlap_after': 0}, indent=2))


if __name__ == '__main__':
    main()
