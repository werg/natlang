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

Text alone is not a program: renaming the root or reformatting a sentence changes the text, not the program. Programs
are therefore also counted as functional programs (measure_repetition.FunctionalPrograms): the same structure (call
graph, helper argument/return types, kind/readout, helper calls, spelled-out code steps and reference scripts,
literals abstracted) AND word-trigram Jaccard >= 0.7 of the normalized prose. The threshold is calibrated on the audited
round-1 batches against the auditors' distinct-program counts (10 of 12 batches within 30%); it is a proxy, and a
rule sentence toggled inside one helper can be undercounted. A batch program whose text is new to its family but which
is functionally an existing program is reported as ``functional_repeats``.

The batch PASSES when it has no exact duplicates, at most 10% near duplicates, and every family in the batch,
together with the same family already in the pool, has >= 20 distinct normalized programs, or the batch adds >= 5
distinct programs to it; and, in each family, at least half of the batch's new programs (at least one) are
functionally new, unless the family already has >= 20 functional signatures. Distinct programs and functional
signatures per family are reported separately from tasks.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_repetition import FunctionalPrograms, normalize, program_features, write_text_atomic  # noqa: E402


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
    semantics = row.get('semantics') or {}
    inputs = semantics.get('inputs')
    if semantics.get('folder_files'):  # a directory reducer's world is part of its invocation
        inputs = {'inputs': inputs, 'folder_files': semantics['folder_files']}
    inputs = normalize_value(inputs)
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
    functional = FunctionalPrograms()
    pool_shingles = collections.defaultdict(list)
    for path in args.pool:
        for row in rows(path):
            features = program_features(row)
            key = task_key(row, features)
            pool_tasks.add(key)
            pool_family_programs[features['family']].add(features['program'])
            functional.add(features['family'], features['functional'], features['functional_words'])
            pool_shingles[features['family']].append((shingles(key[0]), shingles(key[1])))
    pool_functional = collections.Counter(functional.counts)
    batch = [(row, program_features(row)) for row in rows(args.batch)]
    exact, near, seen = [], [], set()
    added = collections.defaultdict(set)
    new_functional = collections.Counter()
    functional_repeats, seen_programs = [], collections.defaultdict(set)
    for row, features in batch:
        family = features['family']
        is_new = functional.add(family, features['functional'], features['functional_words'])
        new_functional[family] += is_new
        if not is_new and features['program'] not in pool_family_programs[family] | seen_programs[family]:
            functional_repeats.append(row['id'])
        seen_programs[family].add(features['program'])
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
        new_programs = len(programs - pool_family_programs[family])
        batch_functional = new_functional[family]
        functional_total = pool_functional[family] + batch_functional
        functional_ok = (functional_total >= 20 or new_programs == 0
                         or (batch_functional >= 1 and 2 * batch_functional >= new_programs))
        families[family] = {'pool_programs': len(pool_family_programs[family]),
                            'batch_tasks': sum(1 for _, f in batch if f['family'] == family),
                            'batch_new_programs': new_programs, 'total': total,
                            'pool_functional': pool_functional[family],
                            'batch_new_functional': batch_functional, 'functional_total': functional_total,
                            'text_ok': total >= 20 or new_programs >= 5, 'functional_ok': functional_ok}
        families[family]['ok'] = families[family]['text_ok'] and functional_ok
    near_share = len(near) / len(batch) if batch else 0.
    result = {'schema': 'natlang.rl-authored-novelty/1', 'batch': str(Path(args.batch).resolve()),
              'pool': [str(Path(p).resolve()) for p in args.pool], 'tasks': len(batch),
              'exact_duplicates': exact, 'near_duplicates': near, 'near_duplicate_share': round(near_share, 4),
              'functional_repeats': functional_repeats,
              'families': families}
    result['result'] = ('PASS' if batch and not exact and near_share <= 0.1 and all(f['ok'] for f in families.values())
                        else 'FAIL')
    text = json.dumps(result, indent=1)
    if args.out:
        write_text_atomic(args.out, text + '\n')
        print(f'wrote {Path(args.out).resolve()}', file=sys.stderr)
    print(text)
    return 0 if result['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
