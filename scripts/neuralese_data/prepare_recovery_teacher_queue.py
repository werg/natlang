#!/usr/bin/env python3
"""Pin train-only seeded failures or replay-checked handoffs for teacher collection.

The source supplies the task and an actual first action to execute. The native
collector runs that action, shows its real feedback to the teacher, and grades
the continuation. Collection does not confer training admission.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import read_jsonl


def prepare(source, output, *, line, workers, case_seconds):
    source = source.resolve()
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    if line not in ('lfm25', 'mellum21') or not 1 <= workers <= 6:
        raise ValueError('line must be lfm25 or mellum21; workers must be 1..6')
    rows = read_jsonl(source)
    if not rows:
        raise ValueError('empty recovery source')
    ids = set()
    source_modes = set()
    for row in rows:
        semantics = row.get('semantics', {})
        if not isinstance(semantics, dict):
            raise ValueError(f'invalid semantics: {row.get("id")}')
        seeded = 'failure_seed' in semantics
        handoff = row.get('handoff', {})
        replayed = (isinstance(handoff, dict) and handoff.get('version') == 'natlang.handoff/2'
                    and handoff.get('kind') in ('failed_action', 'wrong_result')
                    and isinstance(handoff.get('source'), dict)
                    and isinstance(handoff['source'].get('program_id'), str)
                    and isinstance(handoff.get('rejected'), dict))
        if (row.get('version') != 'natlang.program/2' or
                row.get('kind') not in ('lambda_graph', 'lambda_source') or
                not isinstance(semantics.get('root'), str) or
                semantics['root'] not in semantics.get('files', {}) or
                row.get('split') != 'train' or not row.get('source_groups') or
                not (seeded or replayed) or (handoff and not replayed)):
            raise ValueError('recovery source needs focused train IR v2, source groups, and a seeded failure or replay-checked handoff')
        if row['id'] in ids:
            raise ValueError(f'duplicate recovery source id: {row["id"]}')
        ids.add(row['id'])
        # A handoff from a seeded run retains the seed in its source record.
        # The replayed prefix, rather than a second seed injection, determines
        # where the teacher resumes.
        source_modes.add('replay_checked_handoff' if replayed else 'seeded_failure')
    output.mkdir(parents=True)
    manifest = {'schema': 'natlang.recovery-teacher-campaign/1',
                'line': line, 'source': str(source), 'source_sha256': sha256_file_hex(source),
                'cases': len(rows), 'workers': workers,
                'source_modes': sorted(source_modes),
                'continuation_turns': 24,
                'selection': 'all train-only exact failure seeds or replay-checked handoffs, one per source id',
                'training_admission': False,
                'required_review': 'exact oracle, continuation action quality, held-group exclusion, native render'}
    (output / 'campaign-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for number in range(workers):
        worker = output / f'worker-{number + 1:02d}'
        worker.mkdir()
        with (worker / 'queue.jsonl').open('w') as stream:
            for ordinal, index in enumerate(range(number, len(rows), workers)):
                case = worker / f'case-{ordinal:03d}'
                handoff = rows[index].get('handoff') or {}
                replay_turns = sum(len(call) for call in handoff.get('prefix', []))
                # The collector counts replayed turns toward max_turns. Give
                # every teacher the same live continuation allowance even for
                # a handoff taken late in a long trajectory.
                max_turns = replay_turns + 24
                job = {'key': f'recovery-{line}-{number + 1:02d}-{ordinal:03d}',
                       'source': str(source), 'index': index, 'count': 1,
                       'output': str(case / 'results.jsonl'), 'jobs': str(case / 'jobs'),
                       'log': str(case / 'collector.log'), 'seed': 880000 + index,
                       'context_tokens': 32768, 'action_max_tokens': 4096,
                       'max_turns': max_turns, 'max_model_requests': max(96, 4 * max_turns),
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
