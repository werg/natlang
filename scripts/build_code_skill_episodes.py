#!/usr/bin/env python3
"""Python-coding skill episodes from KodCode, scored by unit-test pass fraction in a sandbox. No model calls.

Each episode is one KodCode subset (a family of problem styles): four support and four query problems,
plus four transfer problems from the next subset. Problems are drawn at moderate difficulty so that
skill edits have room to help, and problems close to public benchmarks are excluded."""
import argparse
import glob
import hashlib
import json
import os
import re

import pyarrow.parquet as pq

# Package and Docs problems need third-party libraries that the sandbox image does not provide.
SUBSETS = ['Leetcode', 'Codeforces', 'Code_Contests', 'Taco', 'Apps', 'Algorithm', 'Data_Structure', 'Filter', 'Prefill', 'Evol']
COLUMNS = ['subset', 'style', 'question_id', 'question', 'test', 'test_info', 'gpt_pass_percentage', 'benchmark_similarity']


def digest(value):
    return hashlib.sha256((value if isinstance(value, str) else json.dumps(value, sort_keys=True)).encode()).hexdigest()


def target():
    return {'kind': 'improvement-case', 'entry': 'solve.nl', 'exportName': 'default',
            'source': {'schema': 'natlang.skill-code-target/1', 'id': 'kodcode-python-v1'},
            'files': {'solve.nl': '---\nargs: { problem: string, signature: string }\nreturns: string\n---\n'
                      'Write a Python module that solves the problem and defines the function with the given signature. '
                      'Return only the code. The host runs hidden unit tests against it in a sandbox without network access, '
                      'using only the Python standard library.\n'}}


def case(row):
    signature = '\n'.join(info['function_declaration'] for info in row['test_info'] if info.get('function_declaration'))
    return {'id': 'case-' + digest('kodcode:' + row['question_id'])[:20], 'group': 'g-' + digest('kodcode-group:' + row['question_id'])[:24],
            'args': [row['question'], signature], 'expected': {'kind': 'python-tests', 'tests': row['test']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kodcode', default='/mnt/external/sdkb-archive/raw/agentic-20260927/kodcode-v1/data')
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes-per-subset', type=int, default=10)
    parser.add_argument('--min-pass', type=float, default=0.2)
    parser.add_argument('--max-pass', type=float, default=0.7)
    args = parser.parse_args()
    pools = {subset: [] for subset in SUBSETS}
    for path in sorted(glob.glob(os.path.join(args.kodcode, '*.parquet'))):
        table = pq.read_table(path, columns=COLUMNS)
        for row in table.to_pylist():
            if row['style'] != 'instruct' or row['subset'] not in pools:
                continue
            if not (args.min_pass <= (row['gpt_pass_percentage'] or 0) <= args.max_pass) or (row['benchmark_similarity'] or 0) >= 0.8:
                continue
            # The sandbox runner calls plain test functions: no pytest, fixtures or parameters.
            if 'from solution import' not in row['test'] or not row['test_info'] or 'pytest' in row['test'] \
                    or re.search(r'def test\w*\(\s*[^)\s]', row['test']):
                continue
            pools[row['subset']].append(row)
    need = 8 * args.episodes_per_subset + 4 * args.episodes_per_subset
    for subset, rows in pools.items():
        rows.sort(key=lambda row: digest(row['question_id']))
        if len(rows) < need:
            raise SystemExit(f'{subset}: only {len(rows)} eligible problems, need {need}')
    episodes = []
    for index, subset in enumerate(SUBSETS):
        other = SUBSETS[(index + 1) % len(SUBSETS)]
        own, theirs = pools[subset], pools[other]
        for n in range(args.episodes_per_subset):
            chosen = own[8 * n: 8 * n + 8]
            # Transfer problems come from the other subset's tail, which its own episodes never use.
            transfer = theirs[len(theirs) - 4 * (n + 1): len(theirs) - 4 * n]
            support, query, moved = [case(r) for r in chosen[:4]], [case(r) for r in chosen[4:]], [case(r) for r in transfer]
            groups = sorted(c['group'] for c in support + query + moved)
            # The last fifth (at least two) of each subset's episodes is held out, alternating validation and test.
            held = max(2, args.episodes_per_subset // 5)
            rank = n - (args.episodes_per_subset - held)
            split = ('validation' if rank % 2 == 0 else 'test') if rank >= 0 else 'train'
            episodes.append({'version': 'natlang.skill-episode/1', 'id': f'skill-code-kodcode-{subset.lower()}-{n}',
                             'family': f'code:{subset}', 'split': split,
                             'source_groups': ['group-commitment:sha256:' + digest(groups)], 'license': 'CC-BY-NC-4.0 (KodCode)',
                             'target': target(), 'library': {'kind': 'empty', 'skills': {}},
                             'support': {'cases': support}, 'query': {'cases': query},
                             'transfer': {'family': f'code:{other}', 'target': target(), 'cases': moved},
                             'operations': ['create', 'revise', 'select', 'test'], 'limits': {'maxSteps': 6},
                             'provenance': {'generator': 'natlang.skill-code-episodes/1', 'source': 'kodcode-v1', 'subset': subset,
                                            'transfer_subset': other,
                                            'metric': {'schema': 'natlang.skill-graded/1', 'kind': 'python-tests'},
                                            'transfer_metric': {'schema': 'natlang.skill-graded/1', 'kind': 'python-tests'}}})
    os.makedirs(args.out, exist_ok=True)
    body = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in episodes)
    with open(os.path.join(args.out, 'code-episodes.jsonl'), 'x') as stream:
        stream.write(body)
    manifest = {'schema': 'natlang.skill-code-episodes/1', 'episodes': len(episodes),
                'splits': {s: sum(1 for e in episodes if e['split'] == s) for s in ['train', 'validation', 'test']},
                'subsets': SUBSETS, 'pass_band': [args.min_pass, args.max_pass], 'license': 'CC-BY-NC-4.0 (KodCode)',
                'model_calls': 0, 'sha256': digest(body)}
    with open(os.path.join(args.out, 'code-episodes.manifest.json'), 'x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
