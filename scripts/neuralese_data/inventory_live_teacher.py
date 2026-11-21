#!/usr/bin/env python3
"""Inventory retained Vibestudio runs without fetching, rewriting, or collecting data."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from natlang_neuralese.common import data_registry
from natlang_neuralese.common.hashing import sha256_file_hex

from live_teacher_sources import load_episode, open_goal_fingerprint
from sft_sources import held_closure, held_hit, live_holdout_registry, require_registered_open_goal


def files(roots, pattern):
    return sorted({path.resolve() for root in roots for path in root.rglob(pattern)})


def capture_pins(path):
    run_id = next((directory.name for directory in path.parents if directory.name.startswith('st_')), None)
    for directory in path.parents:
        pins = directory / 'source.json'
        if pins.is_file():
            doc = json.loads(pins.read_text())
            if (doc.get('repositories') and doc.get('validator_sha256') and doc.get('run_id') and
                    run_id in (None, doc['run_id'])):
                return pins, doc['run_id']
    return None, run_id


def inventory(roots, held, live_training, model):
    manifests, runs = [], []
    verified_runs, goals = set(), set()
    registry = data_registry.load()
    paths = [(path, registry.find(path)) for path in files(roots, 'manifest.json')]
    for path, dataset in sorted(paths, key=lambda item: (item[1] is None, item[0])):
        spec = json.loads(path.read_text())
        if spec.get('schema') != 'natlang.live-teacher-task/1':
            continue
        row = {'path': str(path), 'task_id': spec['id'], 'family': spec['family'], 'split': spec['split'],
               'run_id': (spec.get('world') or {}).get('run_id')}
        manifests.append(row)
        if spec['split'] != 'train':
            row['status'] = 'held-split'
            continue
        try:
            episode = load_episode(path)
            task = episode['task']
            row.update(model=episode['teacher']['model_id'], decisions=len(episode['turns']),
                       invocations=len(episode['call_tree']), groups=task['source_groups'], source_ids=task['source_ids'])
            verified_runs.add(row['run_id'])
            interaction = task.get('interaction')
            if held_hit(task['id'], task['source_ids'], task['source_groups'], held):
                row['status'] = 'held-overlap'
                continue
            if interaction:
                goal = interaction['goal']
                require_registered_open_goal({'task_id': task['id'], 'open_goal': goal,
                                              'source_groups': task['source_groups']}, live_training)
                row['goal_sha256'] = open_goal_fingerprint(goal)[1]
            identity = row.get('goal_sha256', task['id'])
            if row['model'] != model:
                row['status'] = 'other-teacher'
            elif identity in goals:
                row['status'] = 'duplicate-goal'
            elif not dataset or dataset.get('admission') == data_registry.INVENTORY:
                row['status'] = 'awaiting-data-registration'
            else:
                row['status'] = 'ready-for-extraction'
                row['dataset'] = dataset['id']
                goals.add(identity)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='invalid-or-unregistered', error=str(exc))

    raw_digests = set()
    pinned_packets = [(path, *capture_pins(path)) for path in files(roots, '*-full.json.gz')]
    for path, pins, run_id in sorted(pinned_packets, key=lambda item: (item[1] is None, item[0])):
        digest = sha256_file_hex(path)
        if digest in raw_digests:
            continue
        raw_digests.add(digest)
        try:
            with gzip.open(path, 'rt') as packet:
                doc = json.load(packet)
        except (OSError, ValueError) as exc:
            runs.append({'path': str(path), 'status': 'unreadable', 'error': str(exc)})
            continue
        diagnostics = (doc.get('execution') or {}).get('diagnostics') or {}
        teacher = diagnostics.get('natlangTeacher') or {}
        passed = (doc.get('result') or {}).get('passed') is True
        scenario = (doc.get('test') or {}).get('name')
        row = {'path': str(path), 'scenario': scenario, 'run_id': run_id, 'passed': passed}
        runs.append(row)
        if not teacher.get('calls'):
            row['status'] = 'no-natlang-call-tree'
            continue
        row['pins'] = str(pins) if pins else None
        row['model'] = sorted({gateway['model'] for call in teacher['calls']
                               for turn in (call.get('record') or {}).get('model_turns') or []
                               if (gateway := ((turn.get('reply') or {}).get('raw_response') or {}).get('vibestudio'))
                               and gateway.get('model')})
        if not passed or teacher.get('trainingEligible') is not True:
            row['status'] = 'did-not-pass'
        elif run_id and run_id in verified_runs:
            row['status'] = 'already-verified'
        elif pins:
            row['status'] = 'candidate-for-offline-export'
        else:
            row['status'] = 'missing-source-pins'

    return {'model': model, 'roots': list(map(str, roots)),
            'manifest_counts': dict(Counter(row['status'] for row in manifests)),
            'packet_counts': dict(Counter(row['status'] for row in runs)),
            'manifests': manifests, 'packets': runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--held', required=True, type=Path, help='Pinned held manifest')
    parser.add_argument('--live-catalog', required=True, type=Path, help='Pinned Vibestudio split catalog')
    parser.add_argument('--model', required=True, help='Teacher model eligible for the extraction list')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    held = held_closure(args.held)
    live_training = live_holdout_registry(args.live_catalog, held)
    result = inventory(args.roots, held, live_training, args.model)
    with args.out.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key.endswith('_counts')}, indent=2))


if __name__ == '__main__':
    main()
