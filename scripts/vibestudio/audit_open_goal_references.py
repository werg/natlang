#!/usr/bin/env python3
"""Reconcile a pinned open-goal catalog with managed reference-run traces."""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def fingerprint(goal):
    return hashlib.sha256(json.dumps(goal, ensure_ascii=False).encode()).hexdigest()


def audit(catalog_path, roots, families):
    catalog_bytes = catalog_path.read_bytes()
    catalog = {}
    digests = set()
    source_splits = {}
    for line in catalog_bytes.splitlines():
        row = json.loads(line)
        scenario, goal_hash = row['scenario'], row['goalSha256']
        if (row['interaction'] != 'open-user-goal' or not scenario.startswith('open-goal-') or
                row['split'] not in {'train', 'validation', 'test'} or
                scenario in catalog or goal_hash in digests or not row['sourceGroups']):
            raise ValueError(f'Invalid or duplicate open goal: {scenario}')
        for group in row['sourceGroups']:
            if source_splits.setdefault(group, row['split']) != row['split']:
                raise ValueError(f'Open-goal source group crosses splits: {group}')
        catalog[scenario] = row
        digests.add(goal_hash)

    selected = {key: row for key, row in catalog.items() if not families or row['family'] in families}
    if not selected or families - {row['family'] for row in selected.values()}:
        raise ValueError('No matching catalog entries for one or more selected families')

    proofs = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f'No managed result directory at {root}')
        for path in sorted(root.rglob('trajectory-open-goal-oracle-*-full.json.gz')):
            scenario = path.name.removeprefix('trajectory-').removesuffix('-full.json.gz')
            scenario = scenario.replace('open-goal-oracle-', 'open-goal-', 1)
            if scenario not in selected or scenario in proofs:
                continue
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                trace = json.load(stream)
            execution = trace.get('execution') or {}
            result = trace.get('result') or {}
            diagnostics = execution.get('diagnostics') or {}
            proof = diagnostics.get('openGoalVerification') or {}
            checks = proof.get('checks') or {}
            if ((trace.get('test') or {}).get('name') != scenario.replace('open-goal-', 'open-goal-oracle-', 1) or
                    result.get('passed') is not True or result.get('details') != checks or execution.get('error') or
                    diagnostics.get('natlangTeacher') is not None or proof.get('passed') is not True or
                    proof.get('caseId') != scenario.removeprefix('open-goal-') or
                    not proof.get('contextId') or
                    proof['contextId'] != (execution.get('provenance') or {}).get('contextId') or
                    not isinstance(checks, dict) or not checks or
                    any(value is not True for value in checks.values()) or
                    not isinstance(proof.get('goal'), str) or
                    fingerprint(proof['goal']) != selected[scenario]['goalSha256']):
                continue
            build = diagnostics.get('buildProvenance') or {}
            with path.open('rb') as stream:
                archive_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            proofs[scenario] = {
                'path': str(path),
                'archive_sha256': archive_sha256,
                'system_testing_effective_version': build.get('effectiveVersion'),
                'checks': sorted(checks),
            }

    coverage = defaultdict(lambda: {
        'train': 0, 'held': 0, 'verified_train': 0, 'verified_held': 0,
    })
    for scenario, row in selected.items():
        bucket = 'train' if row['split'] == 'train' else 'held'
        family = coverage[row['family']]
        family[bucket] += 1
        family[f'verified_{bucket}'] += scenario in proofs

    return {
        'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
        'selected': len(selected),
        'verified': len(proofs),
        'missing': sorted(selected.keys() - proofs.keys()),
        'families': dict(sorted(coverage.items())),
        'families_without_verified_holdout': sorted(
            family for family, counts in coverage.items() if not counts['verified_held']
        ),
        'proofs': proofs,
    }


def corpus_floor(baseline):
    with baseline.open(encoding='utf-8') as stream:
        snapshot = json.loads(stream.readline())
    size = snapshot.get('agentic_baseline')
    floor = snapshot.get('minimum_distinct_open_goals')
    if (snapshot.get('schema') != 'vibestudio.open-goal-inventory/1' or
            type(size) is not int or size <= 0 or floor != 2 * size):
        raise ValueError(f'Invalid pinned baseline at {baseline}')
    return floor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--family', action='append', default=[])
    parser.add_argument('--baseline', type=Path, help='audit the full corpus against the pinned inventory floor')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.baseline and args.family:
        parser.error('--baseline audits the full corpus')
    result = audit(args.catalog, args.roots, set(args.family))
    shortfall = 0
    missing_holdout = False
    if args.baseline:
        floor = corpus_floor(args.baseline)
        shortfall = max(0, floor - result['verified'])
        missing_holdout = bool(result['families_without_verified_holdout'])
        result.update(minimum_verified_goals=floor, remaining_verified_goals=shortfall)
    print(json.dumps(result, indent=2))
    raise SystemExit(int(args.require_complete and bool(result['missing'] or shortfall or missing_holdout)))
