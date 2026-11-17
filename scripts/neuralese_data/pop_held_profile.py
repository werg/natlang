#!/usr/bin/env python3
"""Where a held evaluation's actor time goes: an SGLang server log (decode vs prefill, batch occupancy, KV pool use) and
a held_eval run directory (episode wall times, slot occupancy, the tail after the last dispatch).

  pop_held_profile.py sglang ACTOR.log [--since 'YYYY-MM-DD HH:MM:SS'] [--until ...]
  pop_held_profile.py episodes HELD_RUN_DIR

Decode/prefill split: a least-squares fit of each interval between decode log lines (40 decode steps) on the decode
steps by batch size, the prefill batches and their new tokens inside it (plans/neuralese/POP_HELD_THROUGHPUT_20261117.md).
"""
import argparse
import collections
import json
import re
import sys
from datetime import datetime
from pathlib import Path

LINE = re.compile(r'\[(\S+ \S+)\] (Decode|Prefill) batch, (.*)')


def batches(path, since=None, until=None):
    for line in open(path, errors='replace'):
        m = LINE.match(line)
        if not m or (since and m.group(1) < since) or (until and m.group(1) > until):
            continue
        fields = {}
        for part in m.group(3).split(','):
            if ':' in part:
                key, value = part.rsplit(':', 1)
                fields[key.strip()] = value.strip()
        yield datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S').timestamp(), m.group(2), fields


def quantile(values, q):
    values = sorted(values)
    return values[int(q * (len(values) - 1))] if values else None


def sglang(args):
    import numpy as np
    events = list(batches(args.log, args.since, args.until))
    decode = [e for e in events if e[1] == 'Decode']
    prefill = [e for e in events if e[1] == 'Prefill']
    if not decode:
        sys.exit('no decode batches in the window')
    span = events[-1][0] - events[0][0]
    running = [int(e[2]['#running-req']) for e in decode]
    tokens = 40 * sum(running)
    new = sum(int(e[2]['#new-token']) for e in prefill)
    cached = sum(int(e[2]['#cached-token']) for e in prefill)
    sizes = sorted(set(running))
    rows, seconds, pending, last = [], [], [], None
    for t, kind, fields in events:
        if kind == 'Prefill':
            pending.append(fields)
            continue
        if last is not None:
            row = [0.] * (len(sizes) + 2)
            row[sizes.index(int(fields['#running-req']))] = 40
            row[-2] = len(pending)
            row[-1] = sum(int(p['#new-token']) for p in pending) / 1000
            rows.append(row)
            seconds.append(t - last)
        last, pending = t, []
    coef = np.linalg.lstsq(np.array(rows), np.array(seconds), rcond=None)[0]
    decode_s = sum(40 * coef[sizes.index(r)] for r in running)
    prefill_s = coef[-2] * len(prefill) + coef[-1] * new / 1000
    usage = {key: [float(e[2][key]) for e in decode if key in e[2]]
             for key in ('full token usage', 'swa token usage', 'mamba usage')}
    report = {
        'span_seconds': round(span), 'generated_tokens': tokens, 'generated_tokens_per_second': round(tokens / span, 1),
        'running_mean': round(sum(running) / len(running), 2),
        'running_share': {r: round(n / len(running), 3) for r, n in sorted(collections.Counter(running).items())},
        'queue_req_share_positive': round(sum(int(e[2]['#queue-req']) > 0 for e in decode) / len(decode), 4),
        'decode_step_ms_by_running': {r: round(1000 * coef[i], 1) for i, r in enumerate(sizes)
                                      if running.count(r) >= 20},
        'decode_seconds_fit': round(decode_s), 'prefill_seconds_fit': round(prefill_s),
        'prefill_new_tokens': new, 'prefix_hit': round(cached / (new + cached), 4) if new + cached else None,
        'pool_usage': {key: {'p50': quantile(v, .5), 'p90': quantile(v, .9), 'p99': quantile(v, .99), 'max': max(v)}
                       for key, v in usage.items() if v},
        'retractions': sum('retract' in line.lower() for line in open(args.log, errors='replace')
                           if 'server_args' not in line),
    }
    print(json.dumps(report, indent=1))


def episodes(args):
    run = Path(args.dir)
    queue = [json.loads(line) for line in open(run / 'sealed-queue.jsonl')]
    rewards = {(r['group'], r['sample']): r for r in map(json.loads, open(run / 'rewards.jsonl'))} \
        if (run / 'rewards.jsonl').exists() else {}
    rows = []
    for entry in queue:
        journal = run / 'workers' / f"{entry['key']}.journal.jsonl"
        if not journal.exists():
            continue
        lines = [json.loads(line) for line in open(journal)]
        start = [r['time'] for r in lines if r.get('event') == 'start']
        finish = [r for r in lines if r.get('event') == 'finish']
        if start and finish:
            outcome = rewards.get((entry['group'], entry['sample']), {}).get('outcome')
            rows.append({'start': start[0], 'finish': finish[-1]['time'], 'wall': finish[-1]['elapsed_seconds'],
                         'tokens': finish[-1].get('completion_tokens') or 0, 'outcome': outcome})
    t0, t1 = min(r['start'] for r in rows), max(r['finish'] for r in rows)
    events = sorted([(r['start'], 1) for r in rows] + [(r['finish'], -1) for r in rows])
    held, current, last = collections.Counter(), 0, t0
    for t, step in events:
        held[current] += t - last
        last, current = t, current + step
    total = sum(held.values())
    walls = [r['wall'] for r in rows]
    by_outcome = collections.defaultdict(list)
    for r in rows:
        by_outcome[r['outcome']].append(r['wall'])
    print(json.dumps({
        'episodes': len(rows), 'span_seconds': round(t1 - t0), 'completion_tokens': sum(r['tokens'] for r in rows),
        'wall_seconds': {q: round(quantile(walls, q)) for q in (.1, .5, .9, .99, 1.)},
        'slot_time_share_by_outcome': {str(k): round(sum(v) / sum(walls), 3) for k, v in by_outcome.items()},
        'concurrent_episodes_mean': round(sum(k * v for k, v in held.items()) / total, 2),
        'concurrent_share': {k: round(v / total, 3) for k, v in sorted(held.items())},
        'tail_after_last_start_seconds': round(t1 - max(r['start'] for r in rows)),
    }, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sglang')
    s.add_argument('log')
    s.add_argument('--since')
    s.add_argument('--until')
    e = sub.add_parser('episodes')
    e.add_argument('dir')
    args = parser.parse_args()
    {'sglang': sglang, 'episodes': episodes}[args.command](args)


if __name__ == '__main__':
    main()
