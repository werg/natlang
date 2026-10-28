#!/usr/bin/env python3
"""Protection intersections and additional-source qualification proof for a procedural RL train curriculum.

Checks every row against the shared protected-world custody (common.custody.ProtectedWorlds over
training/protected-worlds.json: evaluation packets, held/quality source groups, decision holds, test reservations;
ids, groups, world bindings and content fingerprints), then writes
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
from natlang_neuralese.common.custody import REGISTRY, ProtectedWorlds  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--packet', required=True, type=Path)
    ap.add_argument('--source-id', required=True)
    a = ap.parse_args()
    cases = a.packet / 'cases.jsonl'
    rows = [json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
    protected = ProtectedWorlds.load()  # fails closed
    related = [{'id': r['id'], 'kinds': protected.related(r)} for r in rows if protected.related(r)]
    inter = {'protected_worlds': related}
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
        'current_protection_bindings': {'registry': REGISTRY, 'registry_sha256': sha_file(ROOT / REGISTRY),
                                        'protected_sha256': protected.digest,
                                        'sources': [json.loads(item) for item in protected.sources]},
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
