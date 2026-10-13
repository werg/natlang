#!/usr/bin/env python3
"""Select exact-answer rejections from a closed repair campaign for another teacher.

This is an exact evidence filter, not a model-quality or training-admission
decision. Resource failures, incomplete runs, and held source groups are never
passed through as failed model decisions.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select(source, primary, held, output):
    if output.exists():
        raise FileExistsError(output)
    source = source.resolve()
    primary = primary.resolve()
    held = held.resolve()
    manifest_path = primary / 'campaign-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (Path(manifest['source']).resolve() != source or
            manifest['source_sha256'] != sha256_file_hex(source)):
        raise ValueError('primary campaign source identity mismatch')
    source_rows = rows(source)
    held_groups = {group for row in rows(held) for group in row.get('source_groups', [])}
    decisions = {}
    for worker in sorted(primary.glob('worker-[0-9][0-9]')):
        jobs = rows(worker / 'queue.jsonl')
        finishes = {}
        for journal in [worker / 'journal.jsonl', *sorted(worker.glob('retry-*/journal.jsonl'))]:
            if journal.exists():
                for event in rows(journal):
                    if event.get('event') == 'finish':
                        finishes[event['key']] = event
        if set(finishes) != {job['key'] for job in jobs}:
            raise ValueError(f'primary worker not terminal: {worker}')
        for job in jobs:
            finish = finishes[job['key']]
            if (finish.get('status') not in ('complete', 'complete_with_skips') or
                    not finish.get('output_accounting', {}).get('complete')):
                raise ValueError(f'primary job not complete: {job["key"]}')
            paths = [Path(job['output'])]
            for retry in sorted(worker.glob('retry-*')):
                review = retry / 'review.json'
                if review.exists() and json.loads(review.read_text()).get('original_key') == job['key']:
                    paths.append(retry / 'results.jsonl')
            results = [row for path in paths if path.exists() for row in rows(path)]
            if len(results) != 1:
                raise ValueError(f'expected one primary result: {job["key"]}: {len(results)}')
            original = source_rows[job['index']]
            if results[0]['task']['program_ir'] != original or original['id'] in decisions:
                raise ValueError(f'primary program identity mismatch: {job["key"]}')
            decisions[original['id']] = results[0]['outcome']
    if set(decisions) != {row['id'] for row in source_rows}:
        raise ValueError('primary campaign did not cover every selected handoff')
    selected = []
    for row in source_rows:
        outcome = decisions[row['id']]
        checks = outcome.get('checks') or {}
        answer_only = (outcome.get('status') == 'done' and
                       outcome.get('accepted') is False and
                       outcome.get('rejection_reasons') == ['answer'] and
                       checks.get('answer') is False and
                       all(value is True for key, value in checks.items() if key != 'answer'))
        if answer_only:
            if row.get('split') != 'train' or held_groups.intersection(row.get('source_groups', [])):
                raise ValueError(f'fallback source is held or nontrain: {row["id"]}')
            selected.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    review = {'schema': 'natlang.repair-fallback-selection/1',
              'source': str(source), 'source_sha256': sha256_file_hex(source),
              'primary_manifest': str(manifest_path),
              'primary_manifest_sha256': sha256_file_hex(manifest_path),
              'held': str(held), 'held_sha256': sha256_file_hex(held),
              'selected': len(selected), 'source_cases': len(source_rows),
              'output_sha256': sha256_file_hex(output),
              'selection': 'terminal exact-answer rejections only; source order preserved',
              'training_admission': False}
    output.with_suffix('.manifest.json').write_text(json.dumps(review, indent=2) + '\n')
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--primary-campaign', type=Path, required=True)
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.source, args.primary_campaign, args.held, args.out)))


if __name__ == '__main__':
    main()
