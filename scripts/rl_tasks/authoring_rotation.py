#!/usr/bin/env python3
"""Rotation ledger for continuous RL task authoring (plans/neuralese/RL_TASK_AUTHORING_KIT.md).

Each authored batch is one cell: (dataset seed, construct, problem type). Workers take the least-used cell so
successive batches differ; the ledger records what each batch actually admitted.

  authoring_rotation.py next   --ledger L.jsonl --datasets D.txt --constructs C.txt --problems P.txt --worker W \\
                               [--claim --batch DIR] [--claim-hours 6]
  authoring_rotation.py record --ledger L.jsonl --dataset D --construct C --problem P --worker W --batch DIR \\
                               --admitted N [--credit CONSTRUCT=N ...] [--credit-from ADMITTED_CASES.jsonl]
Each recorded entry also stores ``cells``: the (seed, construct, problem) cells its admitted rows actually cover (seed from
the row's dataset group or ``invented:<subject domain>``, problem type from the row's shape/family/instruction keywords),
which ``next`` counts instead of the cell the batch was claimed for; ``reconcile`` derives them for every logged batch.
A batch usually covers several constructs. ``record`` credits every genuine construct of the batch, not only the cell's
primary one: ``--credit CONSTRUCT=N`` (repeatable) names the number of admitted tasks genuinely covering each, or
``--credit-from`` reads them from the batch's admitted converted cases (``curriculum.external.constructs``; tags the
audit removed are not there). Without either, only the primary construct is credited, as before.
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


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve  # noqa: E402

DEFAULT_POOL = resolve('data_hdd', 'natlang-rl-authoring', 'admitted')


def admitted_ids_by_batch(admissions):
    """batch directory -> ids admitted for it (a batch admitted in several steps accumulates)."""
    found = {}
    path = Path(admissions)
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                if 'batch_dir' in entry and 'admitted_case_ids' in entry:
                    found.setdefault(entry['batch_dir'], []).extend(entry['admitted_case_ids'])
    return found


# Problem types by the words of a row's shape, family and root instruction. The rotation cell a batch was CLAIMED for says
# nothing about what it wrote (round 6: a long-document batch was recorded as rule-application-with-exceptions), so the
# ledger derives the cells from the admitted rows themselves.
PROBLEM_KEYWORDS = {
    'running-ledger-with-corrections': ('ledger', 'running', 'carried', 'correction', 'balance', 'tally', 'supersed'),
    'schedule-or-allocation-repair': ('schedule', 'rota', 'allocat', 'repair', 'assign', 'seating', 'roster', 'capacity', 'slot'),
    'multi-source-reconciliation': ('reconcil', 'two sources', 'cross-check', 'cross document', 'cross-document', 'discrepan', 'merge', 'compare'),
    'rule-application-with-exceptions': ('rule', 'exception', 'threshold', 'tariff', 'tier', 'policy', 'precedence', 'rate', 'price', 'pricing'),
    'planning-with-resource-limits': ('plan', 'budget', 'limit', 'quota', 'resource', 'pipeline', 'stock', 'yield', 'weight'),
    'diagnosis-from-evidence': ('diagnos', 'evidence', 'culprit', 'fault', 'cause', 'triage', 'symptom', 'clue', 'audit'),
    'negotiation-or-multi-round-game': ('negotiat', 'multi-round', 'game', 'offer', 'bid', 'bracket', 'tournament'),
    'document-obligation-tracking': ('obligation', 'clause', 'document', 'digest', 'minutes', 'contract', 'section', 'bulletin', 'report'),
    'routing-and-escalation-gates': ('rout', 'escalat', 'gate', 'queue', 'ladder', 'dispatch', 'request', 'urgen'),
    'codebase-operation-on-state': ('codebase', 'repo', 'module', 'source file', 'operation', 'rename', 'version', 'changelog', 'index'),
}


def problem_of(row, problems=None):
    """The problem type a row's own text points at (``unclassified`` when no keyword of any type occurs)."""
    external = row.get('curriculum', {}).get('external', {})
    files = row.get('semantics', {}).get('files', {})
    root = row.get('semantics', {}).get('root')
    text = ' '.join([str(external.get('shape', '')), str(row.get('family', '')), str(external.get('subject_domain', '')),
                     (files.get(root) or '')[:700]]).lower()
    keywords = {k: v for k, v in PROBLEM_KEYWORDS.items() if not problems or k in problems}
    scores = {k: sum(text.count(w) for w in words) for k, words in keywords.items()}
    best = max(scores, key=lambda k: (scores[k], -list(keywords).index(k)), default=None)
    return best if best and scores[best] else 'unclassified'


def seed_of(row):
    """The dataset a row drew from (``authored-group:<dataset>:<id>``) or ``invented:<subject domain>``."""
    for group in row.get('source_groups') or []:
        if group.startswith('authored-group:'):
            return group.split(':')[1]
    return 'invented:' + str(row.get('curriculum', {}).get('external', {}).get('subject_domain') or row.get('family'))


def cells_of(rows, problems=None):
    """Cells a set of admitted rows actually covers: one (seed, construct, problem) cell per construct of each row (a row
    with no construct covers ``none``), counted in rows."""
    found = collections.Counter()
    for row in rows:
        seed, problem = seed_of(row), problem_of(row, problems)
        for construct in row['curriculum']['external'].get('constructs') or ['none']:
            found[(seed, construct, problem)] += 1
    return [{'dataset': d, 'construct': c, 'problem': p, 'rows': n} for (d, c, p), n in sorted(found.items())]


def credits_of(rows):
    return dict(collections.Counter(c for row in rows for c in row['curriculum']['external'].get('constructs', [])))


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
    rec.add_argument('--admitted', type=int)
    rec.add_argument('--credit', action='append', default=[], metavar='CONSTRUCT=N')
    rec.add_argument('--credit-from', metavar='CASES.jsonl')
    rec.add_argument('--admissions', default=str(DEFAULT_POOL / 'admissions.jsonl'),
                     help='admission log: a batch with an entry is credited only for its admitted case ids')
    fix = sub.add_parser('reconcile', help='rewrite recorded entries to the admitted rows of the admission log')
    fix.add_argument('--ledger', required=True)
    fix.add_argument('--pool', default=str(DEFAULT_POOL))
    fix.add_argument('--problems', help='problems.txt (restricts the derived problem types to its names)')
    args = parser.parse_args(argv)
    import fcntl
    # The ledger and its directory are created on first use; a missing ledger is an empty rotation.
    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ledger).touch(exist_ok=True)
    lock = open(str(args.ledger) + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)  # claims and records are serialized across workers
    all_entries = ledger(args.ledger)
    if args.command == 'reconcile':
        by_batch = admitted_ids_by_batch(Path(args.pool) / 'admissions.jsonl')
        pool = {}
        for line in (Path(args.pool) / 'cases.jsonl').read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                pool[row['id']] = row
        changed = 0
        for entry in all_entries:
            ids = by_batch.get(entry['batch'])
            if entry.get('status', 'recorded') != 'recorded' or ids is None:
                continue
            rows = [pool[i] for i in dict.fromkeys(ids) if i in pool]
            credits = credits_of(rows)
            cells = cells_of(rows, lines(args.problems) if args.problems else None)
            if (entry['admitted'] != len(rows) or (entry.get('credits') or {}) != credits
                    or entry.get('cells') != cells):
                entry.update(admitted=len(rows), reconciled_from_admissions=True)
                entry.pop('credits', None)
                if credits:
                    entry['credits'] = credits
                entry['cells'] = cells
                changed += 1
        backup = Path(str(args.ledger) + f'.pre-reconcile-{int(time.time())}')
        backup.write_text(Path(args.ledger).read_text())
        Path(args.ledger).write_text(''.join(json.dumps(entry) + '\n' for entry in all_entries))
        print(json.dumps({'reconciled_entries': changed, 'backup': str(backup)}))
        return 0
    recorded_batches = {entry['batch'] for entry in all_entries if entry.get('status', 'recorded') == 'recorded'}
    entries = [entry for entry in all_entries if entry.get('status', 'recorded') == 'recorded']
    if args.command == 'record':
        credits, cells = {}, []
        if args.credit_from:
            rows = [json.loads(line) for line in Path(args.credit_from).read_text().splitlines() if line.strip()]
            # Credit only admitted rows: the qualified cases file also holds the tasks the audit dropped (round 3
            # over-credited 100+ construct tasks and 7 admitted counts by reading it whole).
            known = admitted_ids_by_batch(args.admissions)
            admitted_ids = known.get(args.batch) or known.get(str(Path(args.batch).resolve()))
            if admitted_ids is not None:
                rows = [row for row in rows if row['id'] in set(admitted_ids)]
                args.admitted = len(rows)
            credits = credits_of(rows)
            cells = cells_of(rows)
            if args.admitted is None:
                args.admitted = len(rows)
        for item in args.credit:
            name, _, count = item.partition('=')
            credits[name] = int(count)
        if args.admitted is None:
            raise SystemExit('record needs --admitted (or --credit-from)')
        entry = {'unix': time.time(), 'status': 'recorded', 'dataset': args.dataset, 'construct': args.construct,
                 'problem': args.problem, 'worker': args.worker, 'batch': args.batch, 'admitted': args.admitted,
                 **({'credits': credits} if credits else {}), **({'cells': cells} if cells else {})}
        with open(args.ledger, 'a') as handle:
            handle.write(json.dumps(entry) + '\n')
        print(json.dumps(entry))
        return 0
    admitted, batches = collections.Counter(), collections.Counter()
    by_construct, by_mode = collections.Counter(), collections.Counter()
    invented = collections.Counter()
    for entry in entries:
        for construct, count in (entry.get('credits') or {entry['construct']: entry['admitted']}).items():
            by_construct[construct] += count
        # The cells a batch covered (derived from its admitted rows) replace the cell it was claimed for.
        covered = entry.get('cells') or [{'dataset': entry['dataset'], 'construct': entry['construct'],
                                          'problem': entry['problem'], 'rows': entry['admitted']}]
        seen_cells = set()
        for item in covered:
            cell = (item['dataset'], item['construct'], item['problem'])
            admitted[cell] += item['rows']
            seen_cells.add(cell)
        for cell in seen_cells:
            batches[cell] += 1
        # Mode counts and invented domains count each row once, however many constructs it covers.
        per_seed = collections.Counter()
        for item in covered:
            per_seed[item['dataset']] = max(per_seed[item['dataset']], item['rows'])
        for seed, count in per_seed.items():
            mode = 'B-original' if seed.startswith('invented:') else 'A-dataset'
            by_mode[mode] += count
            if mode == 'B-original':
                invented[seed[len('invented:'):]] += count
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
            for item in entry.get('cells') or [{'dataset': entry['dataset'], 'construct': entry['construct'],
                                                'problem': entry['problem'], 'rows': entry['admitted']}]:
                if item['dataset'].startswith('invented:') and item['construct'] == construct:
                    used[item['problem']] += item['rows']
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
