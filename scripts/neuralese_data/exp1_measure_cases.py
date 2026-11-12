#!/usr/bin/env python3
"""Pool tasks that Experiment 1 has not measured, for base pass-rate collection (grpo.py --pass-rates).

The pools of grpo.py minus the held sets (held-v2 and the w5 held manifest, with their split groups and shared source
groups), minus the tasks already collected, minus the broken codebase families; family round-robin so a partial run covers
many families. Writes cases.jsonl for exp1_collect.py.

    exp1_measure_cases.py --tasks 512 --out DIR/cases.jsonl --exclude-cases exp1-rft/cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'training/neuralese'), str(Path(__file__).resolve().parents[2])]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grpo  # noqa: E402
from natlang_neuralese.common.jsonio import read_jsonl  # noqa: E402
from natlang_neuralese.rl.held_eval import exclusion_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tasks', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--exclude-cases', type=Path, action='append', default=[])
    ap.add_argument('--exclude-manifest', action='append', default=[])
    ap.add_argument('--exclude-family-regex', default=r'^ext_codebase-(py|natlang)_codebase_')
    ap.add_argument('--seed', type=int, default=20261112)
    args = ap.parse_args()
    rows, seen = [], set()
    for path in grpo.POOLS:
        for row in read_jsonl(path):
            if row['id'] not in seen:
                seen.add(row['id'])
                rows.append(row)
    excluded = set(exclusion_ids(grpo.HELD, rows))
    for manifest in args.exclude_manifest:
        excluded |= exclusion_ids(manifest, rows)
    groups = {g for r in rows if r['id'] in excluded for g in r.get('source_groups') or ()}
    excluded |= {r['id'] for r in rows if groups.intersection(r.get('source_groups') or ())}
    for path in args.exclude_cases:
        excluded |= {r['id'] for r in read_jsonl(path)}
    broken = re.compile(args.exclude_family_regex) if args.exclude_family_regex else None
    eligible = [r for r in rows if r['id'] not in excluded and r.get('split', 'train') == 'train'
                and not (broken and broken.search(str(r.get('family') or '')))]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    by_family = {}
    for row in eligible:
        by_family.setdefault(row.get('family'), []).append(row)
    families = sorted(by_family, key=lambda f: str(f))
    rng.shuffle(families)
    chosen = []
    while len(chosen) < args.tasks and any(by_family.values()):
        for family in families:
            if by_family[family] and len(chosen) < args.tasks:
                chosen.append(by_family[family].pop())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r) + '\n' for r in chosen))
    print(json.dumps({'pool_rows': len(rows), 'excluded': len(excluded), 'eligible': len(eligible), 'chosen': len(chosen),
                      'families': len({r.get('family') for r in chosen})}))


if __name__ == '__main__':
    main()
