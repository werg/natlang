#!/usr/bin/env python3
"""One batch of verifier-judged authored tasks (semantic-FP, plans/semantic-fp/05 O1; kit section 18), end to end.

  sfp_batch.py qualify BATCH_DIR --runtime TS_HOST [--n N] [--concurrency C]
  sfp_batch.py blind BATCH_DIR ANSWERS.json [--n N]
  sfp_batch.py status BATCH_DIR [BATCH_DIR ...]

``qualify`` runs, on BATCH_DIR/tasks.jsonl, each step writing under BATCH_DIR:
 1. qualify_authored.py   convert, collector replay (reference, wrong answers, aliases, keyword baseline that must run
                          and be rejected), full smoke replay, custody, lint (cue rules, verifier controls, construct
                          checks, groups that are not one real source item)              qualified-N/
 2. authored_novelty.py   against the admitted authored pool and every registry grpo-pool source   novelty-N.json
 3. held separation       ``held_eval.training_exclusion_ids`` over the converted rows: every ``held-eval*`` role
                          (held-v3, held-v4 and its reservations); a train row it names fails the batch  held-N.json
 4. measure_repetition.py the batch alone and against the grpo pools (instruction, program, functional program and
                          structure repetition)                                          repetition-N.json
 5. blind packet          blind_solve.py packet: programs and inputs only, for an independent solver   blind-packet.jsonl
and prints the summary (also sfp-N.json). ``blind`` judges the solver's answers with each task's own verifier
(blind_solve.py compare) into blind-N.json. A batch is admitted (admit_authored.py) only when sfp-N.json says PASS and
the blind solve agrees (or each disagreement is settled and the task fixed or dropped). Run ``qualify`` under the memory
ledger (CPU only, about 3 GB at concurrency 4).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO)]
PY = sys.executable
from natlang_neuralese.common.paths import resolve_str  # noqa: E402
ADMITTED = resolve_str('data_hdd', 'natlang-rl-authoring/admitted/cases.jsonl')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def run(args, log):
    env = dict(os.environ, PYTHONPATH=':'.join([str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO)]))
    with open(log, 'w') as out:
        return subprocess.run([str(a) for a in args], stdout=out, stderr=subprocess.STDOUT, cwd=REPO, env=env).returncode


def grpo_pools():
    import grpo
    return [str(p) for p in grpo.default_pools()]


def qualify(batch, runtime, n, concurrency):
    tasks = batch / 'tasks.jsonl'
    qdir = batch / f'qualified-{n}'
    summary = {'batch': str(batch), 'n': n, 'runtime': runtime, 'tasks': len(rows(tasks))}
    code = run([PY, HERE / 'qualify_authored.py', tasks, qdir, '--runtime', runtime, '--concurrency', concurrency],
               batch / f'qualify-{n}.log')
    report = json.loads((qdir / 'report.json').read_text()) if (qdir / 'report.json').exists() else {}
    verify = report.get('steps', {}).get('verify', {})
    summary['qualify'] = {'exit': code, 'passed': report.get('passed'), 'failed': report.get('failed'),
                          'reference_ok': verify.get('reference_ok'), 'wrong_controls': verify.get('wrong_controls'),
                          'wrong_accepted': verify.get('wrong_accepted'), 'wrong_errors': verify.get('wrong_errors'),
                          'keyword_baselines': verify.get('keyword_baselines'),
                          'keyword_baselines_not_rejected': verify.get('keyword_baselines_not_rejected'),
                          'alias_probes': verify.get('alias_probes'), 'alias_rejected': verify.get('alias_rejected'),
                          'smoke': report.get('steps', {}).get('smoke'), 'custody': report.get('steps', {}).get('custody'),
                          'failures': report.get('failures', {})}
    cases = qdir / 'verified' / 'cases.jsonl'
    pools = grpo_pools()
    novelty_args = [PY, HERE / 'authored_novelty.py', cases, '--pool', ADMITTED]
    for pool in pools:
        novelty_args += ['--pool', pool]
    code = run(novelty_args + ['-o', batch / f'novelty-{n}.json'], batch / f'novelty-{n}.log')
    novelty = json.loads((batch / f'novelty-{n}.json').read_text()) if (batch / f'novelty-{n}.json').exists() else {}
    summary['novelty'] = {'exit': code, 'result': novelty.get('result'), 'families': novelty.get('families')}
    from natlang_neuralese.rl import held_eval
    converted = rows(cases)
    excluded = sorted(held_eval.training_exclusion_ids(converted))
    held = {'rows': len(converted), 'excluded': excluded,
            'excluded_train': [row['id'] for row in converted if row['id'] in set(excluded) and row.get('split') == 'train'],
            'by_split': {s: sum(row.get('split') == s for row in converted) for s in ('train', 'dev', 'confirmation')}}
    (batch / f'held-{n}.json').write_text(json.dumps(held, indent=1) + '\n')
    summary['held'] = {k: (len(v) if isinstance(v, list) else v) for k, v in held.items()}
    rep_args = [PY, HERE / 'measure_repetition.py', '--cases', cases, '--label', batch.parent.name]
    for pool in pools:
        rep_args += ['--against', pool]
    code = run(rep_args + ['-o', batch / f'repetition-{n}.json'], batch / f'repetition-{n}.log')
    rep = json.loads((batch / f'repetition-{n}.json').read_text()) if (batch / f'repetition-{n}.json').exists() else {}
    summary['repetition'] = {'exit': code, **{k: rep.get(k) for k in ('rows', 'distinct_programs', 'effective_programs',
                                                                    'distinct_structures', 'effective_structures',
                                                                    'distinct_functional_programs')},
                             'against_pool': rep.get('against_pool')}
    run([PY, HERE / 'blind_solve.py', 'packet', tasks, '-o', batch / 'blind-packet.jsonl'], batch / 'blind-packet.log')
    ok = (summary['qualify']['exit'] == 0 and summary['novelty']['exit'] == 0 and not held['excluded_train'])
    summary['result'] = 'PASS' if ok else 'FAIL'
    (batch / f'sfp-{n}.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1


def blind(batch, answers, n):
    code = run([PY, HERE / 'blind_solve.py', 'compare', batch / 'tasks.jsonl', answers], batch / f'blind-{n}.log')
    text = (batch / f'blind-{n}.log').read_text()
    (batch / f'blind-{n}.json').write_text(json.dumps({'exit': code, 'answers': str(answers), 'log': text[-4000:]}, indent=1) + '\n')
    print(text[-4000:])
    return code


def status(batches):
    for batch in batches:
        batch = Path(batch)
        reports = sorted(batch.glob('sfp-*.json'))
        last = json.loads(reports[-1].read_text()) if reports else {}
        blinds = sorted(batch.glob('blind-*.json'))
        print(json.dumps({'batch': str(batch), 'tasks': len(rows(batch / 'tasks.jsonl')), 'result': last.get('result'),
                          'passed': last.get('qualify', {}).get('passed'),
                          'blind_exit': json.loads(blinds[-1].read_text())['exit'] if blinds else None}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('qualify')
    q.add_argument('batch')
    q.add_argument('--runtime', required=True)
    q.add_argument('--n', type=int, default=1)
    q.add_argument('--concurrency', type=int, default=4)
    b = sub.add_parser('blind')
    b.add_argument('batch')
    b.add_argument('answers')
    b.add_argument('--n', type=int, default=1)
    s = sub.add_parser('status')
    s.add_argument('batches', nargs='+')
    args = ap.parse_args(argv)
    if args.cmd == 'qualify':
        return qualify(Path(args.batch), args.runtime, args.n, args.concurrency)
    if args.cmd == 'blind':
        return blind(Path(args.batch), Path(args.answers), args.n)
    return status(args.batches)


if __name__ == '__main__':
    sys.exit(main())
