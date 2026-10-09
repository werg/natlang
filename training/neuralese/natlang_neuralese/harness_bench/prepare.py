"""Prepare trajectories for replay (plans/neuralese/HARNESS_BENCH.md §1.3): normalize, then join each with its task.

Reads SWE-rebench OpenHands trajectories and the SWE-rebench task records, and writes one JSON line per trajectory: the
pi transcript (openhands.normalize, mapping `pi`), the task's repository, base commit and environment setup commit.
`natlang run applications/pi -- replay` rebuilds each step's workspace from the base commit and the teacher's edits.
The task's gold and test patches are never copied here: they are hindsight-only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from .openhands import normalize, resolved, rows


def tasks(directory: str) -> dict[str, dict]:
    """instance_id → the task fields replay needs."""
    found: dict[str, dict] = {}
    for path in sorted(glob.glob(f'{directory}/data/*.parquet')):
        table = pq.read_table(path, columns=['instance_id', 'repo', 'base_commit', 'environment_setup_commit', 'image_name'])
        for row in table.to_pylist():
            found.setdefault(row['instance_id'], row)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--trajectories', required=True)
    parser.add_argument('--tasks', required=True, help='the SWE-rebench task corpus directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--resolved-only', action='store_true')
    parser.add_argument('--workspace', default='/workspace/project',
                        help="the agent's working directory in the records; each task's own is renamed to it")
    args = parser.parse_args(argv)
    known = tasks(args.tasks)
    summary = {'seen': 0, 'written': 0, 'no_task': 0, 'unresolved_skipped': 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as handle:
        columns = ['trajectory_id', 'instance_id', 'repo', 'trajectory', 'model_patch', 'resolved']
        for number, row in enumerate(rows(args.trajectories, columns)):
            if number < args.offset:
                continue
            if args.limit and summary['seen'] >= args.limit:
                break
            summary['seen'] += 1
            if args.resolved_only and not resolved(row.get('resolved')):
                summary['unresolved_skipped'] += 1
                continue
            task = known.get(row['instance_id'])
            if task is None:
                summary['no_task'] += 1
                continue
            record = normalize(row, 'pi', args.workspace).record()
            record.update({'base_commit': task['base_commit'], 'environment_setup_commit': task['environment_setup_commit'],
                           'image_name': task['image_name'],
                           'source_row_sha256': hashlib.sha256(json.dumps(row.get('trajectory'), sort_keys=True, default=str).encode()).hexdigest()})
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            summary['written'] += 1
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
