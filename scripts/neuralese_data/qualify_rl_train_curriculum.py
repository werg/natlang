#!/usr/bin/env python3
"""Protection intersections and additional-source qualification proof for a procedural RL train curriculum.

Checks the packet's ids, source ids and source groups against the current held/quality source-group projections and
the decision-hold files exactly as the RL task-source resolver's audits do, then writes
`qualification.json` (schema natlang.rl-additional-task-source-qualification/1) so the packet can be appended to an
RL task source as an additional admitted train source. Exit 1 on any intersection or non-train row.

  qualify_rl_train_curriculum.py --packet runs/natlang-rl-train-curriculum-lfm3-20261028 --source-id ID
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha_file  # noqa: E402
from natlang_neuralese.common.paths import resolve  # noqa: E402

BUNDLE = resolve('data_nvme', 'e2e-current-protection-bundle-20261015-v1')
HELD = BUNDLE / 'held-union/full/source-groups/text-data.jsonl'
QUALITY = BUNDLE / 'quality-groups.jsonl'


def group_set(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text().splitlines():
        if line.strip():
            out.update(json.loads(line).get('source_groups', []))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--packet', required=True, type=Path)
    ap.add_argument('--source-id', required=True)
    a = ap.parse_args()
    cases = a.packet / 'cases.jsonl'
    rows = [json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
    held, quality = group_set(HELD), group_set(QUALITY)
    hold_dir = ROOT / 'training/decision-holds'
    hold_ids: set[str] = set()
    hold_files = {}
    for path in sorted(hold_dir.glob('*.jsonl')):
        hold_files[path.name] = sha_file(path)
        for line in path.read_text().splitlines():
            if line.strip() and isinstance(json.loads(line).get('id'), str):
                hold_ids.add(json.loads(line)['id'])
    ids = {r['id'] for r in rows}
    all_ids = ids | {s for r in rows for s in r.get('source_ids', [])}
    groups = {g for r in rows for g in r.get('source_groups', [])}
    inter = {'decision_ids': sorted(all_ids & hold_ids), 'held_groups': sorted(groups & held),
             'quality_groups': sorted(groups & quality)}
    row_sha = {r['id']: hashlib.sha256(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                                       .encode('utf-8')).hexdigest() for r in rows}
    verification = json.loads((a.packet / 'verification.json').read_text())
    proof = {
        'schema': 'natlang.rl-additional-task-source-qualification/1',
        'source_id': a.source_id,
        'source_cases_sha256': sha_file(cases),
        'qualified_task_rows': len(rows),
        'all_train': all(r.get('split') == 'train' for r in rows),
        'all_source_bound': all(r.get('source_ids') and r.get('source_groups') for r in rows),
        'qualification_kind': ('Procedural world construction and actual collector reference execution (verifyCases) '
                               'on every row; no teacher generations and no student capability qualification claimed. '
                               'curriculum.rl.qualification is deliberately absent (live-window value is "qualified").'),
        'training_admission': ('DGX supervisor under standing owner authorization (LFM3 JOLT+ RL stage and tuning lane); '
                               'whole student episodes scored by existing exact outcomes, actual communication graphs '
                               'updated whole.'),
        'current_protection_intersections': inter,
        'current_protection_bindings': {
            'decision_hold_files': hold_files,
            'held_groups': {'path': str(HELD), 'sha256': sha_file(HELD), 'groups': len(held)},
            'quality_groups': {'path': str(QUALITY), 'sha256': sha_file(QUALITY), 'groups': len(quality)}},
        'verification': {'path': 'verification.json', 'sha256': sha_file(a.packet / 'verification.json'),
                         'reference_executions': verification['cases'], 'failures': len(verification['failures'])},
        'task_ids': [r['id'] for r in rows],
        'union_task_row_sha256': row_sha,
    }
    (a.packet / 'qualification.json').write_text(json.dumps(proof, indent=1) + '\n')
    print(json.dumps({'rows': len(rows), 'intersections': {k: len(v) for k, v in inter.items()},
                      'all_train': proof['all_train'], 'verification': proof['verification']['failures']}))
    return 1 if any(inter.values()) or not proof['all_train'] or verification['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
