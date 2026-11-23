#!/usr/bin/env python3
"""One batch of verifier-judged authored tasks (semantic-FP, plans/semantic-fp/05 O1; kit section 18), end to end.

  sfp_batch.py qualify BATCH_DIR --runtime TS_HOST [--n N] [--concurrency C]
  sfp_batch.py blind BATCH_DIR ANSWERS.json [--n N]
  sfp_batch.py status BATCH_DIR [BATCH_DIR ...]
  sfp_batch.py supply OUT_DIR --min-worker N --max-worker M --stamp TAG

``qualify`` runs, on BATCH_DIR/tasks.jsonl, each step writing under BATCH_DIR:
 1. qualify_authored.py   convert, collector replay (reference, wrong answers, aliases, keyword baseline that must run
                          and be rejected), full smoke replay, custody, lint (cue rules, verifier controls, construct
                          checks, groups that are not one real source item)              qualified-N/
 2. authored_novelty.py   against the admitted authored pool and every registry grpo-pool source   novelty-N.json
 3. held separation       ``held_eval.training_exclusion_ids`` over the converted rows: every ``held-eval*`` role
                          (held-v3, held-v4 and its reservations); a train row it names fails the batch; and no
                          instruction, program or functional program shared with held-v4's authored tasks  held-N.json
 4. measure_repetition.py the batch alone and against the grpo pools (instruction, program, functional program and
                          structure repetition)                                          repetition-N.json
 5. blind packet          blind_solve.py packet: programs and inputs only, for an independent solver   blind-packet.jsonl
and prints the summary (also sfp-N.json). ``blind`` judges the solver's answers with each task's own verifier
(blind_solve.py compare) into blind-N.json. A batch is admitted (admit_authored.py) only when sfp-N.json says PASS and
the blind solve agrees (or each disagreement is settled and the task fixed or dropped). Run ``qualify`` under the memory
ledger (CPU only, about 3 GB at concurrency 4).

``supply`` assembles the admitted batches of workers N..M into a grpo-pool source and checks it before promotion:
build_supply.py (train rows not in any pool; records the dataset), register_authored_held.py --stamp TAG (the batches'
dev/confirmation rows become evaluation-only packets, so custody withholds their worlds), custody (ProtectedWorlds
identities of every supply row: 0 related is required), held-v3 ``held_eval verify`` with the supply as the avoided
ids and groups and again with the held-v3 exposure groups (0 overlap required), training_exclusion_ids (0), repetition
against the grpo pools, and simulate_selection --grpo with the supply added. It writes OUT_DIR/checks.json and prints
the promote command; promotion (`scripts/data_registry.py promote ID --role grpo-pool`) and the registry commit are
the agent's.
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
HELD_V4_AUTHORING = Path(resolve_str('data_nvme', 'lfm3-rl-dgx-20261029/ops/held-v4/authoring'))


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
    # held-v4 is still being authored (ops/held-v4/authoring/<family>/qualified-*/cases.jsonl, B1): no instruction,
    # program or functional program of the batch may be one of its tasks, whatever the family.
    import measure_repetition
    held_v4 = sorted(str(p) for p in HELD_V4_AUTHORING.glob('*/qualified-*/cases.jsonl'))
    # overlap_with_pool reads train rows only; every row counts here, on both sides.
    def every_row(paths, target):
        target.write_text(''.join(json.dumps(dict(row, split='train')) + '\n' for path in paths for row in rows(path)))
        return str(target)
    overlap = (measure_repetition.overlap_with_pool([every_row([cases], batch / f'.held-v4-batch-{n}.jsonl')],
                                                    [every_row(held_v4, batch / f'.held-v4-pool-{n}.jsonl')])
               if held_v4 and converted else {})
    held['held_v4_overlap'] = {k: overlap.get(k) for k in ('pool_rows', 'rows_with_instruction_in_pool',
                                                            'rows_with_program_in_pool', 'rows_with_functional_program_in_pool')}
    (batch / f'held-{n}.json').write_text(json.dumps(held, indent=1) + '\n')
    summary['held']['held_v4_overlap'] = held['held_v4_overlap']
    v4_shared = sum(held['held_v4_overlap'].get(k) or 0 for k in ('rows_with_instruction_in_pool', 'rows_with_program_in_pool',
                                                                  'rows_with_functional_program_in_pool'))
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
    ok = (summary['qualify']['exit'] == 0 and summary['novelty']['exit'] == 0 and not held['excluded_train']
          and not v4_shared)
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


HELD_V3 = Path(resolve_str('data_nvme', 'lfm3-rl-dgx-20261029/ops/held-v3'))


def supply(out, min_worker, max_worker, stamp):
    out.mkdir(parents=True, exist_ok=True)
    checks = {'out': str(out), 'workers': [min_worker, max_worker], 'stamp': stamp}
    checks['build_supply'] = run([PY, HERE / 'build_supply.py', out, '--min-worker', min_worker, '--max-worker', max_worker],
                                 out / 'build_supply.log')
    checks['register_held'] = run([PY, HERE / 'register_authored_held.py', '--stamp', stamp], out / 'register_held.log')
    cases = rows(out / 'cases.jsonl')
    checks['rows'] = len(cases)
    from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
    protected = ProtectedWorlds.load()
    related = {row['id']: sorted({kind for kind, _ in world_identities(row, evidence_items=False) & protected.identities})
               for row in cases}
    checks['custody_related'] = {k: v for k, v in related.items() if v}
    from natlang_neuralese.rl import held_eval
    ids = [row['id'] for row in cases]
    groups = sorted({g for row in cases for g in [row.get('curriculum', {}).get('split_group'), *row.get('source_groups', [])] if g})
    (out / 'avoid-ids.json').write_text(json.dumps(ids) + '\n')
    (out / 'avoid-groups.json').write_text(json.dumps(groups) + '\n')
    manifest = HELD_V3 / 'manifest.json'
    exposure = json.loads((HELD_V3 / 'exposure' / 'exposure-groups.json').read_text())
    for name, extra in (('held_v3_verify', ()), ('held_v3_verify_with_exposure', exposure)):
        report = held_eval.verify(manifest, set(ids), set(groups) | set(extra))
        checks[name] = {'overlap_ids': len(report['overlap_ids']), 'overlap_groups': len(report['overlap_groups'])}
    checks['training_exclusion_ids'] = len(held_eval.training_exclusion_ids(cases))
    pools = [p for p in grpo_pools() if Path(p).resolve() != (out / 'cases.jsonl').resolve()]
    rep_args = [PY, HERE / 'measure_repetition.py', '--cases', out / 'cases.jsonl']
    for pool in pools:
        rep_args += ['--against', pool]
    run(rep_args + ['-o', out / 'repetition.json'], out / 'repetition.log')
    rep = json.loads((out / 'repetition.json').read_text()) if (out / 'repetition.json').exists() else {}
    checks['repetition'] = {k: rep.get(k) for k in ('rows', 'distinct_programs', 'effective_programs', 'distinct_structures',
                                                     'effective_structures', 'constructs')} | {'against_pool': rep.get('against_pool')}
    sim_args = [PY, HERE / 'simulate_selection.py', '--grpo', '--groups', 2000]
    for pool in pools + [str(out / 'cases.jsonl')]:
        sim_args += ['--cases', pool]
    checks['simulate_selection'] = run(sim_args + ['-o', out / 'selection-sim.json'], out / 'selection-sim.log')
    ok = (checks['build_supply'] == 0 and checks['register_held'] == 0 and not checks['custody_related']
          and not checks['training_exclusion_ids']
          and all(checks[k]['overlap_ids'] == 0 and checks[k]['overlap_groups'] == 0
                  for k in ('held_v3_verify', 'held_v3_verify_with_exposure')))
    checks['result'] = 'PASS' if ok else 'FAIL'
    (out / 'checks.json').write_text(json.dumps(checks, indent=1) + '\n')
    print(json.dumps(checks, indent=1))
    built = [json.loads(line) for line in (out / 'build_supply.log').read_text().splitlines() if line.startswith('{')]
    if built:
        print('promote with:', built[-1].get('admit'))
    return 0 if ok else 1


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
    u = sub.add_parser('supply')
    u.add_argument('out')
    u.add_argument('--min-worker', type=int, required=True)
    u.add_argument('--max-worker', type=int, required=True)
    u.add_argument('--stamp', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'qualify':
        return qualify(Path(args.batch), args.runtime, args.n, args.concurrency)
    if args.cmd == 'blind':
        return blind(Path(args.batch), Path(args.answers), args.n)
    if args.cmd == 'supply':
        return supply(Path(args.out), args.min_worker, args.max_worker, args.stamp)
    return status(args.batches)


if __name__ == '__main__':
    sys.exit(main())
