#!/usr/bin/env python3
"""Reconcile a pinned open-goal catalog with managed reference-run traces."""
import argparse
import gzip
import hashlib
import json
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
            proofs[scenario] = {'path': str(path), 'checks': sorted(checks)}

    return {
        'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
        'selected': len(selected),
        'verified': len(proofs),
        'missing': sorted(selected.keys() - proofs.keys()),
        'proofs': proofs,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog', type=Path)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--family', action='append', default=[])
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = audit(args.catalog, args.roots, set(args.family))
    print(json.dumps(result, indent=2))
    raise SystemExit(int(args.require_complete and bool(result['missing'])))
