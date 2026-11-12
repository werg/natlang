#!/usr/bin/env python3
"""Score an exp1 collection in shards (task_rollouts.score keeps every episode of a run in memory: 1,000 episodes is
more than 12 GB). Each shard is a directory with the run's packet/preparation, the merged journal and a slice of the
queue (finished episodes only); its `rollouts.jsonl` feeds `exp1_build_rows.py --rollouts a --rollouts b ...`.

    exp1_score_shards.py --run collect --shards 8 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]
from natlang_neuralese.common.jsonio import read_jsonl  # noqa: E402
from natlang_neuralese.rl import task_rollouts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shards', type=int, default=8)
    args = ap.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    work = run / 'workers'
    journal, finished = [], set()
    for entry in read_jsonl(run / 'queue.jsonl'):
        rows = [row for path in sorted(work.glob(f"{entry['key']}.a*.journal.jsonl")) for row in read_jsonl(path)]
        done = [row for row in rows if row.get('event') == 'finish']
        if done and done[-1].get('status') != 'incomplete':
            finished.add(entry['key'])
            journal += [row for row in rows if row.get('event') in ('start', 'finish')]
    entries = [e for e in read_jsonl(run / 'queue.jsonl') if e['key'] in finished]
    summary = {'finished': len(entries), 'shards': []}
    for index in range(args.shards):
        directory = out / f'shard-{index:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('packet.jsonl', 'preparation.json'):
            link = directory / name
            if not link.exists():
                link.symlink_to(run / name)
        (directory / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries[index::args.shards]))
        keys = {e['key'] for e in entries[index::args.shards]}
        (directory / 'journal.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in journal if r['key'] in keys))
        report = task_rollouts.score(directory, directory / 'rollouts.jsonl')
        summary['shards'].append({k: report[k] for k in ('rollouts', 'scored', 'successes', 'outcome_counts')})
        print(json.dumps(summary['shards'][-1]), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
