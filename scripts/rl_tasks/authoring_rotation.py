#!/usr/bin/env python3
"""Rotation ledger for continuous RL task authoring (plans/neuralese/RL_TASK_AUTHORING_KIT.md).

Each authored batch is one cell: (dataset seed, construct, problem type). Workers take the least-used cell so
successive batches differ; the ledger records what each batch actually admitted.

  authoring_rotation.py next   --ledger L.jsonl --datasets D.txt --constructs C.txt --problems P.txt --worker W \\
                               [--claim --batch DIR] [--claim-hours 6]
  authoring_rotation.py record --ledger L.jsonl --dataset D --construct C --problem P --worker W --batch DIR \\
                               --admitted N
``next --claim`` reserves the chosen cell for that batch (a pending ledger entry), so concurrent and sequential
workers get different cells; ``record`` finalizes the batch's claim. A claim not recorded within ``--claim-hours``
expires. Admitted counts come only from recorded entries.
`next` picks the least-covered construct and the mode (A dataset-inspired / B original invention) with fewer admitted
tasks. Mode A returns the least-used dataset x problem cell; mode B returns the least-used problem type and every
invented domain used so far, so the worker chooses an unexplored domain (record it as ``invented:<domain>``).
Ties rotate by worker so concurrent workers start apart.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path


def lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith('#')]


def ledger(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    nxt = sub.add_parser('next')
    rec = sub.add_parser('record')
    for p in (nxt, rec):
        p.add_argument('--ledger', required=True)
    nxt.add_argument('--datasets', required=True)
    nxt.add_argument('--constructs', required=True)
    nxt.add_argument('--problems', required=True)
    nxt.add_argument('--worker', default='')
    nxt.add_argument('--claim', action='store_true', help='reserve the chosen cell for --batch')
    nxt.add_argument('--batch', help='batch directory the claim belongs to')
    nxt.add_argument('--claim-hours', type=float, default=6.)
    for key in ('dataset', 'construct', 'problem', 'worker', 'batch'):
        rec.add_argument('--' + key, required=True)
    rec.add_argument('--admitted', type=int, required=True)
    args = parser.parse_args(argv)
    import fcntl
    # The ledger and its directory are created on first use; a missing ledger is an empty rotation.
    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ledger).touch(exist_ok=True)
    lock = open(str(args.ledger) + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)  # claims and records are serialized across workers
    all_entries = ledger(args.ledger)
    recorded_batches = {entry['batch'] for entry in all_entries if entry.get('status', 'recorded') == 'recorded'}
    entries = [entry for entry in all_entries if entry.get('status', 'recorded') == 'recorded']
    if args.command == 'record':
        entry = {'unix': time.time(), 'status': 'recorded', 'dataset': args.dataset, 'construct': args.construct,
                 'problem': args.problem, 'worker': args.worker, 'batch': args.batch, 'admitted': args.admitted}
        with open(args.ledger, 'a') as handle:
            handle.write(json.dumps(entry) + '\n')
        print(json.dumps(entry))
        return 0
    admitted, batches = collections.Counter(), collections.Counter()
    by_construct, by_mode = collections.Counter(), collections.Counter()
    invented = collections.Counter()
    for entry in entries:
        cell = (entry['dataset'], entry['construct'], entry['problem'])
        admitted[cell] += entry['admitted']
        batches[cell] += 1
        by_construct[entry['construct']] += entry['admitted']
        mode = 'B-original' if entry['dataset'].startswith('invented:') else 'A-dataset'
        by_mode[mode] += entry['admitted']
        if mode == 'B-original':
            invented[entry['dataset'][len('invented:'):]] += entry['admitted']
    now = time.time()
    claimed = {(entry['dataset'], entry['construct'], entry['problem']) for entry in all_entries
               if entry.get('status') == 'claimed' and entry['batch'] not in recorded_batches
               and now - entry['unix'] < args.claim_hours * 3600}
    claimed_constructs = collections.Counter(cell[1] for cell in claimed)
    constructs, problems = lines(args.constructs), lines(args.problems)
    offset = int(hashlib.sha256(args.worker.encode()).hexdigest()[:8], 16) if args.worker else 0
    # Least-covered construct first (ties rotate by worker), then the mode with fewer admitted tasks.
    construct = min(constructs, key=lambda c: (by_construct[c] + claimed_constructs[c],
                                               (constructs.index(c) - offset) % len(constructs)))
    mode = 'B-original' if by_mode['B-original'] <= by_mode['A-dataset'] else 'A-dataset'
    if mode == 'A-dataset':
        cells = [(d, construct, p) for d in lines(args.datasets) for p in problems]
        order = {cell: (index - offset) % len(cells) for index, cell in enumerate(cells)}
        cell = min(cells, key=lambda c: (c in claimed, admitted[c], batches[c], order[c]))
        choice = {'mode': mode, 'dataset': cell[0], 'construct': construct, 'problem': cell[2]}
    else:
        used = collections.Counter()
        for entry in entries:
            if entry['dataset'].startswith('invented:') and entry['construct'] == construct:
                used[entry['problem']] += entry['admitted']
        busy = {cell[2] for cell in claimed if cell[0].startswith('invented:') and cell[1] == construct}
        problem = min(problems, key=lambda p: (p in busy, used[p], (problems.index(p) - offset) % len(problems)))
        choice = {'mode': mode, 'dataset': 'invented:<choose a NEW domain, not in invented_domains_used>',
                  'construct': construct, 'problem': problem,
                  'invented_domains_used': sorted(invented)}
    if args.claim:
        if not args.batch or not args.worker:
            raise SystemExit('--claim needs --worker and --batch')
        dataset = choice['dataset'] if mode == 'A-dataset' else 'invented:pending'
        claim = {'unix': now, 'status': 'claimed', 'dataset': dataset, 'construct': construct,
                 'problem': choice['problem'], 'worker': args.worker, 'batch': args.batch, 'admitted': 0}
        with open(args.ledger, 'a') as handle:
            handle.write(json.dumps(claim) + '\n')
        choice['claimed_for_batch'] = args.batch
    print(json.dumps({**choice, 'admitted_by_construct': dict(by_construct), 'admitted_by_mode': dict(by_mode),
                      'active_claims': len(claimed) + bool(args.claim)}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
