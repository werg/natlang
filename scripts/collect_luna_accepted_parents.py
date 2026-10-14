#!/usr/bin/env python3
"""Collect exact-success Luna parents from closed, source-pinned campaigns.

This performs mechanical checks only. Its output is a review candidate, not
training admission; native turn approval and an explicit corpus decision follow.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256


def rows(path):
    return list(iter_rows(path))


def iter_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def world(row):
    semantics = {key: value for key, value in row['semantics'].items()
                 if key not in ('expected', 'expected_files', 'oracle', 'files_oracle')}
    return json.dumps([row['family'], semantics], sort_keys=True, separators=(',', ':'))


def source_bindings(manifest):
    if 'source_paths' in manifest:
        return {Path(path): digest for path, digest in manifest['source_paths'].items()}
    if manifest.get('schema') == 'natlang.recovery-teacher-campaign/1':
        return {Path(manifest['source']): manifest['source_sha256']}
    return {Path(manifest['source_path']): manifest['source_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', action='append', type=Path, required=True)
    parser.add_argument('--previous-parents', action='append', type=Path, default=[])
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    held_rows = rows(args.held)
    seen_worlds = {world(row) for row in held_rows}
    held_groups = {group for row in held_rows for group in row.get('source_groups', [])}
    previous_ids = set()
    for path in args.previous_parents:
        # Previously admitted snapshots can be hundreds of MB. Only their
        # identities/worlds are needed; do not retain full trajectories here.
        for parent in iter_rows(path):
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
        exclusions_path = campaign / 'reviewed-exclusions.json'
        exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
        if exclusions and exclusions.get('schema') != 'natlang.generation-reviewed-exclusions/1':
            raise ValueError(f'unrecognized reviewed exclusions: {exclusions_path}')
        excluded = exclusions.get('cases', {})
        sources = source_bindings(manifest)
        for path, expected in sources.items():
            if sha256(path) != expected:
                raise ValueError(f'source hash mismatch: {path}')
            source_files[path] = rows(path)
        for worker in sorted(campaign.glob('worker-[0-9][0-9]')):
            queues = rows(worker / 'queue.jsonl')
            journals = [worker / 'journal.jsonl', *(p / 'journal.jsonl' for p in worker.glob('retry-*'))]
            finishes = {}
            finish_history = collections.defaultdict(list)
            for journal in journals:
                if journal.exists():
                    for event in rows(journal):
                        if event.get('event') == 'finish':
                            finishes[event['key']] = event
                            finish_history[event['key']].append(event)
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
                review = excluded.get(job['key'])
                if review is not None:
                    if not (review.get('reason') and any(
                            event.get('status') == review.get('finish_status') and
                            event.get('output_accounting', {}).get('exact_result_rows') == 0 and
                            event.get('output_accounting', {}).get('output_sha256') == review.get('output_sha256')
                            for event in finish_history[job['key']])):
                        raise ValueError(f'reviewed exclusion no longer matches an exact zero-result finish: {job["key"]}')
                    if any(result.get('outcome', {}).get('accepted') is True for result in results):
                        raise ValueError(f'reviewed exclusion has an accepted retry requiring new review: {job["key"]}')
                    outcomes['reviewed_exclusion'] += 1
                    continue
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
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.out.parent,
                                     prefix=args.out.name + '.', suffix='.building', delete=False) as stream:
        temporary = Path(stream.name)
        for parent in chosen:
            stream.write(json.dumps(parent, ensure_ascii=False, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    candidate_hash = sha256(temporary)
    if args.out.exists():
        if sha256(args.out) != candidate_hash:
            temporary.unlink()
            raise ValueError(f'existing candidate differs from current exact selection: {args.out}')
        temporary.unlink()
    else:
        os.replace(temporary, args.out)
    review = {'schema': 'natlang.luna-parent-candidate-review/1', 'training_admission': False,
              'held_sha256': sha256(args.held),
              'previous_parent_sha256': {str(p): sha256(p) for p in args.previous_parents},
              'sources_sha256': {str(p): sha256(p) for p in source_files},
              'campaign_manifests_sha256': {str(p / 'campaign-manifest.json'): sha256(p / 'campaign-manifest.json')
                                            for p in args.campaign},
              'reviewed_exclusions_sha256': {str(p / 'reviewed-exclusions.json'): sha256(p / 'reviewed-exclusions.json')
                                             for p in args.campaign if (p / 'reviewed-exclusions.json').exists()},
              'workers': evidence, 'counts': dict(outcomes),
              'candidate_sha256': candidate_hash,
              'decision': 'Exact source and outcome checks only; native materialization and semantic review pending.'}
    review_path = args.out.with_suffix('.review.json')
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=review_path.parent,
                                     prefix=review_path.name + '.', suffix='.building', delete=False) as stream:
        temporary_review = Path(stream.name)
        stream.write(json.dumps(review, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_review, review_path)
    print(json.dumps({'out': str(args.out), 'counts': dict(outcomes)}))


if __name__ == '__main__':
    main()
