#!/usr/bin/env python3
"""Pin train-only seeded recovery cases for teacher collection.

The source supplies the task and an actual first action to execute. The native
collector runs that action, shows its real feedback to the teacher, and grades
the continuation. Collection does not confer training admission.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output, *, line, workers, case_seconds):
    source = source.resolve()
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    if line not in ('lfm25', 'mellum21') or not 1 <= workers <= 6:
        raise ValueError('line must be lfm25 or mellum21; workers must be 1..6')
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError('empty recovery source')
    ids = set()
    for row in rows:
        if (row.get('split') != 'train' or not row.get('source_groups') or
                'failure_seed' not in row.get('semantics', {})):
            raise ValueError('recovery sources need train split, source groups, and an executed failure seed')
        if row['id'] in ids:
            raise ValueError(f'duplicate recovery source id: {row["id"]}')
        ids.add(row['id'])
    output.mkdir(parents=True)
    manifest = {'schema': 'natlang.recovery-teacher-campaign/1',
                'line': line, 'source': str(source), 'source_sha256': sha(source),
                'cases': len(rows), 'workers': workers,
                'selection': 'all train-only seeded failure cases, one per source id',
                'training_admission': False,
                'required_review': 'exact oracle, continuation action quality, held-group exclusion, native render'}
    (output / 'campaign-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for number in range(workers):
        worker = output / f'worker-{number + 1:02d}'
        worker.mkdir()
        with (worker / 'queue.jsonl').open('w') as stream:
            for ordinal, index in enumerate(range(number, len(rows), workers)):
                case = worker / f'case-{ordinal:03d}'
                job = {'key': f'recovery-{line}-{number + 1:02d}-{ordinal:03d}',
                       'source': str(source), 'index': index, 'count': 1,
                       'output': str(case / 'results.jsonl'), 'jobs': str(case / 'jobs'),
                       'log': str(case / 'collector.log'), 'seed': 880000 + index,
                       'context_tokens': 32768, 'action_max_tokens': 4096,
                       'max_turns': 24, 'max_model_requests': 96,
                       'case_seconds': case_seconds, 'request_retries': 3,
                       'transport_retries': 0, 'text_neuralese_emulation': True}
                stream.write(json.dumps(job, sort_keys=True) + '\n')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--line', choices=('lfm25', 'mellum21'), required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--case-seconds', type=int, default=1800)
    args = ap.parse_args()
    if args.case_seconds < 1:
        ap.error('--case-seconds must be positive')
    print(json.dumps(prepare(args.source, args.out, line=args.line,
                             workers=args.workers, case_seconds=args.case_seconds)))


if __name__ == '__main__':
    main()
