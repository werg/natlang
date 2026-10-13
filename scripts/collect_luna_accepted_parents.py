#!/usr/bin/env python3
"""Collect exact-success Luna parents from closed, source-pinned campaigns.

This performs mechanical checks only. Its output is a review candidate, not
training admission; native turn approval and an explicit corpus decision follow.
"""
import argparse
import collections
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def world(row):
    semantics = {key: value for key, value in row['semantics'].items()
                 if key not in ('expected', 'expected_files', 'oracle', 'files_oracle')}
    return json.dumps([row['family'], semantics], sort_keys=True, separators=(',', ':'))


def source_bindings(manifest):
    if 'source_paths' in manifest:
        return {Path(path): digest for path, digest in manifest['source_paths'].items()}
    return {Path(manifest['source_path']): manifest['source_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', action='append', type=Path, required=True)
    parser.add_argument('--previous-parents', action='append', type=Path, default=[])
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    held_rows = rows(args.held)
    seen_worlds = {world(row) for row in held_rows}
    held_groups = {group for row in held_rows for group in row.get('source_groups', [])}
    previous_ids = set()
    for path in args.previous_parents:
        for parent in rows(path):
            previous_ids.add(parent['task']['program_ir']['id'])
            seen_worlds.add(world(parent['task']['program_ir']))
    chosen = []
    selected_worlds = set()
    source_files = {}
    outcomes = collections.Counter()
    evidence = []
    for campaign in args.campaign:
        manifest_path = campaign / 'campaign-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        sources = source_bindings(manifest)
        for path, expected in sources.items():
            if sha256(path) != expected:
                raise ValueError(f'source hash mismatch: {path}')
            source_files[path] = rows(path)
        for worker in sorted(campaign.glob('worker-[0-9][0-9]')):
            queues = rows(worker / 'queue.jsonl')
            journals = [worker / 'journal.jsonl', *(p / 'journal.jsonl' for p in worker.glob('retry-*'))]
            finishes = {}
            for journal in journals:
                if journal.exists():
                    for event in rows(journal):
                        if event.get('event') == 'finish':
                            finishes[event['key']] = event
            if set(finishes) != {job['key'] for job in queues}:
                raise ValueError(f'campaign worker not terminal: {worker}')
            for job in queues:
                source = Path(job['source'])
                original = source_files[source][job['index']]
                result_paths = [Path(job['output'])]
                for retry in sorted(worker.glob('retry-*')):
                    review = retry / 'review.json'
                    if review.exists() and json.loads(review.read_text()).get('original_key') == job['key']:
                        result_paths.append(retry / 'results.jsonl')
                results = [row for path in result_paths if path.exists() for row in rows(path)]
                finish = finishes[job['key']]
                if finish.get('status') not in ('complete', 'complete_with_skips') or not finish.get('output_accounting', {}).get('complete'):
                    raise ValueError(f'noncomplete job: {job["key"]}')
                if len(results) != 1:
                    raise ValueError(f'expected one result: {job["key"]}: {len(results)}')
                result = results[0]
                if result['task']['program_ir'] != original:
                    raise ValueError(f'program source mismatch: {job["key"]}')
                outcome = result['outcome']
                accepted = (outcome.get('accepted') is True and
                            outcome.get('oracle', {}).get('accepted') is True and
                            outcome.get('checks') and all(value is True for value in outcome['checks'].values()))
                if not accepted:
                    outcomes['rejected'] += 1
                    continue
                key = world(original)
                if original['id'] in previous_ids or key in seen_worlds or key in selected_worlds:
                    outcomes['duplicate_or_held'] += 1
                    continue
                if original['split'] != 'train':
                    raise ValueError(f'accepted nontrain source: {original["id"]}')
                if held_groups.intersection(original.get('source_groups', [])):
                    raise ValueError(f'accepted held source group: {original["id"]}')
                selected_worlds.add(key)
                chosen.append(result)
                outcomes['accepted'] += 1
            evidence.append({'campaign': str(campaign), 'worker': worker.name,
                             'queue_sha256': sha256(worker / 'queue.jsonl')})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        for parent in chosen:
            stream.write(json.dumps(parent, ensure_ascii=False, separators=(',', ':')) + '\n')
    review = {'schema': 'natlang.luna-parent-candidate-review/1', 'training_admission': False,
              'held_sha256': sha256(args.held),
              'previous_parent_sha256': {str(p): sha256(p) for p in args.previous_parents},
              'sources_sha256': {str(p): sha256(p) for p in source_files},
              'campaign_manifests_sha256': {str(p / 'campaign-manifest.json'): sha256(p / 'campaign-manifest.json')
                                            for p in args.campaign},
              'workers': evidence, 'counts': dict(outcomes),
              'candidate_sha256': sha256(args.out),
              'decision': 'Exact source and outcome checks only; native materialization and semantic review pending.'}
    args.out.with_suffix('.review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'counts': dict(outcomes)}))


if __name__ == '__main__':
    main()
