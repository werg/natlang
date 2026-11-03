#!/usr/bin/env python3
"""Repetition gate for an authored RL task batch against the existing task pool.

  authored_novelty.py BATCH_CASES.jsonl --pool POOL.jsonl [--pool ...] -o novelty.json

BATCH_CASES is the converted ``cases.jsonl`` that qualify_authored.py writes. Programs and instructions are compared
after normalizing digits and quoted literals (scripts/rl_tasks/measure_repetition.py), so swapping a name, number or
symbol does not make a new task. Near duplicates: word 5-shingle Jaccard >= 0.8 between normalized whole programs.

The batch PASSES when:
- no batch program equals a pool program or another batch program (normalized);
- at most 10% of batch tasks are near duplicates of a pool or earlier batch program;
- every family in the batch, together with the same family already in the pool, has >= 20 distinct normalized
  programs, or the batch adds >= 5 distinct programs to it (a family still growing toward 20 is allowed).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_repetition import program_features  # noqa: E402


def shingles(text, size=5):
    words = text.split()
    return {' '.join(words[i:i + size]) for i in range(max(1, len(words) - size + 1))}


def rows(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('batch')
    parser.add_argument('--pool', action='append', default=[])
    parser.add_argument('-o', '--out')
    args = parser.parse_args(argv)
    pool_programs, pool_family_programs = set(), collections.defaultdict(set)
    pool_shingles = collections.defaultdict(list)
    for path in args.pool:
        for row in rows(path):
            features = program_features(row)
            pool_programs.add(features['program'])
            pool_family_programs[features['family']].add(features['program'])
            pool_shingles[features['family']].append(shingles(features['program']))
    batch = [(row, program_features(row)) for row in rows(args.batch)]
    exact, near, seen = [], [], set()
    added = collections.defaultdict(set)
    for row, features in batch:
        program = features['program']
        if program in pool_programs or program in seen:
            exact.append(row['id'])
        seen.add(program)
        mine = shingles(program)
        for other in pool_shingles[features['family']]:
            if len(mine & other) / max(1, len(mine | other)) >= 0.8:
                near.append(row['id'])
                break
        pool_shingles[features['family']].append(mine)
        added[features['family']].add(program)
    families = {}
    for family, programs in added.items():
        total = len(pool_family_programs[family] | programs)
        families[family] = {'pool': len(pool_family_programs[family]), 'batch_new': len(programs - pool_family_programs[family]),
                            'total': total, 'ok': total >= 20 or len(programs - pool_family_programs[family]) >= 5}
    near_share = len(near) / len(batch) if batch else 0.
    result = {'schema': 'natlang.rl-authored-novelty/1', 'batch': str(Path(args.batch).resolve()),
              'pool': [str(Path(p).resolve()) for p in args.pool], 'tasks': len(batch),
              'exact_duplicates': exact, 'near_duplicates': near, 'near_duplicate_share': round(near_share, 4),
              'families': families}
    result['result'] = ('PASS' if batch and not exact and near_share <= 0.1 and all(f['ok'] for f in families.values())
                        else 'FAIL')
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)
    return 0 if result['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
