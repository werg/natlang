#!/usr/bin/env python3
"""Per-task base pass rates of an exp1 collection (what grpo.py --pass-rates reads), incrementally.

Format: task_id -> {family, finished, scored, successes, timeouts, mean_tokens}. scored = successes plus verified failures
(a wall-clock timeout or an infrastructure error is unscored); mean_tokens = mean generated tokens over finished episodes.
Outcomes of finished episodes are cached (`pass-rate-cache.json` beside the output) so a refresh only scores new ones
(task_rollouts.score without an emit keeps no turns, so memory stays small).

    exp1_pass_rates.py --run collect --out base-pass-rates.json [--loop 1800]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]
from natlang_neuralese.common.jsonio import read_jsonl  # noqa: E402
from natlang_neuralese.rl import task_rollouts  # noqa: E402

BATCH = 60


def refresh(run, out):
    run, out = Path(run).resolve(), Path(out)
    cache_path = out.with_name('pass-rate-cache.json')
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    work = run / 'workers'
    entries, journals = [], {}
    for entry in read_jsonl(run / 'queue.jsonl'):
        if entry['key'] in cache:
            continue
        rows = [row for path in sorted(work.glob(f"{entry['key']}.a*.journal.jsonl")) for row in read_jsonl(path)]
        done = [row for row in rows if row.get('event') == 'finish']
        if done and done[-1].get('status') != 'incomplete' and Path(entry['output']).is_file():
            entries.append(entry)
            activity = [row.get('completion_tokens') for row in rows if row.get('event') == 'activity' and row.get('completion_tokens') is not None]
            journals[entry['key']] = ([row for row in rows if row.get('event') in ('start', 'finish')], activity[-1] if activity else None,
                                      done[-1].get('status'))
    scratch = out.with_name('pass-rate-scratch')
    for start in range(0, len(entries), BATCH):
        batch = entries[start:start + BATCH]
        scratch.mkdir(exist_ok=True)
        for name in ('packet.jsonl', 'preparation.json'):
            link = scratch / name
            if not link.exists():
                link.symlink_to(run / name)
        (scratch / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in batch))
        (scratch / 'journal.jsonl').write_text(''.join(json.dumps(r) + '\n' for e in batch for r in journals[e['key']][0]))
        task_rollouts.score(scratch)
        rewards = {(r['group'], r['sample']): r for r in read_jsonl(scratch / 'rewards.jsonl')}
        for e in batch:
            r = rewards[(e['group'], e['sample'])]
            cache[e['key']] = {'task_id': r['task_id'], 'family': r.get('family'), 'reward': r['reward'], 'outcome': r['outcome'],
                               'tokens': journals[e['key']][1], 'status': journals[e['key']][2]}
        cache_path.write_text(json.dumps(cache))
    rates = {}
    for item in cache.values():
        rate = rates.setdefault(item['task_id'], {'family': item['family'], 'finished': 0, 'scored': 0, 'successes': 0,
                                                  'timeouts': 0, '_tokens': []})
        rate['finished'] += 1
        rate['scored'] += item['reward'] is not None
        rate['successes'] += item['reward'] == 1.0
        rate['timeouts'] += item['status'] == 'timeout' or (item['outcome'] == 'budget_exhaustion' and item['reward'] is None)
        if item['tokens'] is not None:
            rate['_tokens'].append(item['tokens'])
    for rate in rates.values():
        tokens = rate.pop('_tokens')
        rate['mean_tokens'] = round(sum(tokens) / len(tokens)) if tokens else None
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(sorted(rates.items())), indent=0))
    os.replace(tmp, out)
    return {'tasks': len(rates), 'episodes': len(cache), 'new': len(entries),
            'successes': sum(r['successes'] for r in rates.values()), 'scored': sum(r['scored'] for r in rates.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--loop', type=int, default=0, help='seconds between refreshes (0: once)')
    args = ap.parse_args()
    while True:
        print(json.dumps({'at': time.strftime('%H:%M'), **refresh(args.run, args.out)}), flush=True)
        if not args.loop:
            return
        time.sleep(args.loop)


if __name__ == '__main__':
    main()
