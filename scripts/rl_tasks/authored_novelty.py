#!/usr/bin/env python3
"""Repetition gate for an authored RL task batch against the existing task pool.

  authored_novelty.py BATCH_CASES.jsonl --pool POOL.jsonl [--pool ...] -o novelty.json

BATCH_CASES is the converted ``cases.jsonl`` that qualify_authored.py writes. A task is its program together with its
invocation inputs: several invocations of one program on materially different worlds are wanted (multi-operation
codebases call one program many times). Programs and inputs are compared after normalizing digits and quoted
literals (scripts/rl_tasks/measure_repetition.py), so a task whose program and inputs differ from another only in
names, numbers or quoted values is the same task.

- exact duplicate: same normalized program AND same normalized inputs as a pool task or an earlier batch task;
- near duplicate: within the same family, word 5-shingle Jaccard >= 0.8 between the normalized programs AND >= 0.8
  between the normalized inputs.

The batch PASSES when it has no exact duplicates, at most 10% near duplicates, and every family in the batch,
together with the same family already in the pool, has >= 20 distinct normalized programs, or the batch adds >= 5
distinct programs to it. Distinct programs per family are reported separately from tasks.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_repetition import normalize, program_features  # noqa: E402


def shingles(text, size=5):
    words = text.split()
    return {' '.join(words[i:i + size]) for i in range(max(1, len(words) - size + 1))}


def normalize_value(value):
    """Inputs with every string normalized like program text (digits, quoted literals inside the text)."""
    if isinstance(value, str):
        return normalize(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return '#'
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {normalize(str(key)): normalize_value(item) for key, item in value.items()}
    return value


def task_key(row, features):
    """Normalized program plus normalized inputs: digits and quoted literals do not make a different task."""
    inputs = normalize_value((row.get('semantics') or {}).get('inputs'))
    return features['program'], json.dumps(inputs, sort_keys=True, ensure_ascii=False)


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
    pool_tasks, pool_family_programs = set(), collections.defaultdict(set)
    pool_shingles = collections.defaultdict(list)
    for path in args.pool:
        for row in rows(path):
            features = program_features(row)
            key = task_key(row, features)
            pool_tasks.add(key)
            pool_family_programs[features['family']].add(features['program'])
            pool_shingles[features['family']].append((shingles(key[0]), shingles(key[1])))
    batch = [(row, program_features(row)) for row in rows(args.batch)]
    exact, near, seen = [], [], set()
    added = collections.defaultdict(set)
    for row, features in batch:
        program = features['program']
        key = task_key(row, features)
        if key in pool_tasks or key in seen:
            exact.append(row['id'])
        seen.add(key)
        mine = (shingles(key[0]), shingles(key[1]))
        for other in pool_shingles[features['family']]:
            if all(len(a & b) / max(1, len(a | b)) >= 0.8 for a, b in zip(mine, other)):
                near.append(row['id'])
                break
        pool_shingles[features['family']].append(mine)
        added[features['family']].add(program)
    families = {}
    for family, programs in added.items():
        total = len(pool_family_programs[family] | programs)
        families[family] = {'pool_programs': len(pool_family_programs[family]),
                            'batch_tasks': sum(1 for _, f in batch if f['family'] == family),
                            'batch_new_programs': len(programs - pool_family_programs[family]),
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
