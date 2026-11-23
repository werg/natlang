#!/usr/bin/env python3
"""Corpus driver of the program migrator (plans/semantic-fp/07_CORPUS_MIGRATION.md, M2/M3).

The decisions are the natlang program's (``applications/migration`` target ``program``: keep or delete, step kinds,
rewrites, retry or give up). This script is plumbing around it: it prepares one unit per program from a baseline
report, runs the program's ``propose`` per unit (in parallel, on the served local model or a named teacher), builds the
equivalence cases of the candidates, applies the exact acceptance rule (``equivalence.compare``, 07 §4.2, plus a clean
idiom floor), asks the program's ``settle`` about rejected ones, writes accepted files into a checkout, and packages the
verified before/after pairs.

    migrate_corpus.py units --baseline REPORT.jsonl --out UNITS [--program P ...] [--problems DECISIONS.jsonl]
    migrate_corpus.py propose --units UNITS --out PROPOSALS [--jobs 8] [--server URL --model ID | --provider ID [--model ID]]
    migrate_corpus.py cases --proposals PROPOSALS --out CASES [--kind gate ...]
    migrate_corpus.py settle --original A/report.jsonl --candidate B/report.jsonl --proposals PROPOSALS
                             --out DECISIONS.jsonl [--attempts-left N] [--history DECISIONS.jsonl ...]
    migrate_corpus.py apply --decisions DECISIONS.jsonl --repo CHECKOUT
    migrate_corpus.py package --decisions DECISIONS.jsonl ... --out DIR [--exclude-reserved RESERVATIONS.json]
    migrate_corpus.py tasks --cases CASES ... --decisions DECISIONS.jsonl ... --out DIR [--exclude-reserved R] [--oracle exact]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import equivalence  # noqa: E402
from natlang_neuralese.common.jsonio import read_jsonl  # noqa: E402

DECISION_SCHEMA = 'natlang.semantic-fp-migration-decision/1'
PAIR_SCHEMA = 'natlang.semantic-fp-migration-pair/1'


# ---- programs ---------------------------------------------------------------------------------------------------------
def program_root(program, repo=REPO):
    """The source directory of ``design-brief:NNN``, ``astra:APP`` or ``sol:APP``."""
    corpus, _, name = program.partition(':')
    if corpus == 'design-brief':
        [found] = sorted((repo / equivalence.DESIGN_SOURCE).glob(f'{name}-*'))
        return found
    if corpus in ('astra', 'sol'):
        return repo / 'applications' / name
    raise ValueError(f'unknown program {program}')


def brief_of(program, root):
    """What the program is for, from its brief (design briefs) or its package description (applications)."""
    if (root / 'application.json').exists():
        app = json.loads((root / 'application.json').read_text())
        purposes = '; '.join(i.get('purpose', '') for i in app.get('interfaces') or [])
        return f"{app.get('title', program)}: {purposes}".strip()
    if (root / 'natlang.json').exists():
        return json.loads((root / 'natlang.json').read_text()).get('description') or program
    return program


def baselines(rows):
    """{program: Baseline} from equivalence report rows (any case kinds the report holds)."""
    out = collections.defaultdict(lambda: {'cases': 0, 'passingCases': 0, 'passes': 0, 'scored': 0})
    for row in rows:
        b = out[row['program']]
        b['cases'] += 1
        b['passingCases'] += row['passes'] > 0
        b['passes'] += row['passes']
        b['scored'] += row['scored']
    return dict(out)


def write_units(baseline_rows, out, *, programs=None, problems=None, repo=REPO):
    """One ``Unit`` JSON per program (``host.ts``): root, brief, baseline, and the problems of an earlier attempt."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for program, baseline in sorted(baselines(baseline_rows).items()):
        if programs and program not in programs:
            continue
        root = program_root(program, repo)
        unit = {'program': program, 'root': str(root), 'brief': brief_of(program, root), 'baseline': baseline}
        if problems and problems.get(program):
            unit['problems'] = problems[program]['problems']
            unit['history'] = problems[program]['history']
        path = out / f"{program.replace(':', '-')}.json"
        path.write_text(json.dumps(unit, indent=1) + '\n')
        written.append(path)
    return written


# ---- the natlang program ----------------------------------------------------------------------------------------------
def natlang_command(args, *, natlang=None):
    return [natlang or os.environ.get('NATLANG_BIN') or 'natlang', 'run', str(REPO / 'applications' / 'migration'),
            '--target', 'program', '--progress', 'off', *args]


def model_env(server=None, model=None):
    env = dict(os.environ)
    if server:
        env['NATLANG_SERVER'] = server
        env['NATLANG_MODEL'] = model or 'native-lfm3'
    return env


def propose_one(unit, out, *, server=None, model=None, provider=None, timeout=4 * 3600):
    """Run ``propose`` for one unit into ``out/<program>``; returns the proposal record (with the model that ran it)."""
    unit = Path(unit)
    name = unit.stem
    target = Path(out) / name
    target.mkdir(parents=True, exist_ok=True)
    extra = (['--provider', provider] + (['--model', model] if model else [])) if provider else []
    command = natlang_command([*extra, '--', 'propose', str(unit), str(target)])
    started = __import__('time').time()
    done = subprocess.run(command, env=model_env(None if provider else server, model), capture_output=True, text=True,
                          timeout=timeout, cwd=REPO)
    (target / 'propose.log').write_text(done.stdout[-20000:] + '\n--- stderr ---\n' + done.stderr[-20000:])
    record = {'unit': str(unit), 'exit': done.returncode, 'seconds': round(__import__('time').time() - started, 1),
              'model': {'provider': provider, 'model': model} if provider else {'server': server, 'model': model or 'native-lfm3'}}
    proposal = target / 'proposal.json'
    if proposal.exists():
        record.update({k: v for k, v in json.loads(proposal.read_text()).items() if k in ('program', 'status', 'keep', 'keepBy', 'problems')})
    (target / 'run.json').write_text(json.dumps(record, indent=1) + '\n')
    return record


def propose_all(units, out, *, jobs=8, **model):
    units = sorted(Path(units).glob('*.json')) if Path(units).is_dir() else [Path(units)]
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(propose_one, unit, out, **model): unit for unit in units}
        for future in concurrent.futures.as_completed(futures):
            try:
                record = future.result()
            except Exception as error:   # one program's failure is its own record, never the batch's
                record = {'unit': str(futures[future]), 'error': repr(error)[:500]}
            print(json.dumps(record), flush=True)


def proposals(directory):
    """{program: (proposal, run record, candidate dir)} of a proposals directory."""
    found = {}
    for path in sorted(Path(directory).glob('*/proposal.json')):
        proposal = json.loads(path.read_text())
        run = json.loads((path.parent / 'run.json').read_text()) if (path.parent / 'run.json').exists() else {}
        found[proposal['program']] = (proposal, run, path.parent / 'program')
    return found


def candidate_sources(directory):
    """{program: candidate dir} of the proposals whose candidate passed its exact checks."""
    return {program: str(dir_) for program, (proposal, _, dir_) in proposals(directory).items()
            if proposal['status'] == 'candidate' and dir_.is_dir()}


# ---- acceptance -------------------------------------------------------------------------------------------------------
def settle(original_rows, candidate_rows, proposal_dir, *, attempts_left=0, history=None, judge=None):
    """One decision per proposed program: ``accept`` (07 §4.2 holds and the floor is clean), ``delete`` (the keep
    judgment or its floor), ``clean`` (no finding), or for a rejected candidate the program's ``settle`` judgment
    (``retry`` / ``give-up``). ``judge(comparison, attempts_left, history)`` runs that judgment."""
    found = proposals(proposal_dir)
    by_program = collections.defaultdict(list)
    for row in candidate_rows:
        by_program[row['program']].append(row)
    originals = collections.defaultdict(list)
    for row in original_rows:
        originals[row['program']].append(row)
    decisions = []
    for program, (proposal, run, directory) in sorted(found.items()):
        base = {'schema': DECISION_SCHEMA, 'program': program, 'root': proposal['root'], 'proposal': str(directory.parent),
                'model': run.get('model'), 'findings': proposal['findings'], 'keepBy': proposal.get('keepBy')}
        if proposal['status'] in ('clean', 'delete'):
            decisions.append({**base, 'decision': proposal['status'], 'reason': proposal['status']})
            continue
        if proposal['status'] != 'candidate':
            decisions.append({**base, 'decision': 'rejected', 'reason': 'the candidate failed its exact checks',
                              'problems': proposal['problems']})
            continue
        if originals[program] and all(r['scored'] for r in originals[program]) and not any(r['passes'] for r in originals[program]):
            # 07 §3's crisp floor on the measured baseline: no passing original case at k samples.
            decisions.append({**base, 'decision': 'delete', 'reason': 'no original case passes', 'keepBy': 'floor'})
            continue
        mine = {row['case_id'] for row in originals[program]}
        rows = [r for r in by_program.get(program, []) if r['case_id'] in mine]
        if not rows or any(r['scored'] < r['samples'] for r in rows):
            decisions.append({**base, 'decision': 'pending', 'reason': 'the candidate is not fully measured'})
            continue
        [comparison] = equivalence.compare(originals[program], rows)
        problems = list(comparison['problems'])
        # A 'candidate' passed problemsOf: no check error, no floor finding in any changed file, and every flagged file
        # is changed, so its floor is clean; the comparison is the remaining condition.
        if comparison['accepted']:
            decisions.append({**base, 'decision': 'accept', 'comparison': comparison})
            continue
        prior = (history or {}).get(program, [])
        judged = judge({'accepted': False, 'problems': problems,
                        'expectedPassingFirst': comparison['expected_passing_first'],
                        'expectedPassingSecond': comparison['expected_passing_second']}, attempts_left, prior) if judge else 'give-up'
        if judged not in ('retry', 'give-up'):
            judged = 'give-up'
        decisions.append({**base, 'decision': judged, 'comparison': comparison, 'problems': problems,
                          'history': prior + ['; '.join(problems)]})
    return decisions


def settle_judge(scratch):
    """The program's ``settle`` target as a function (on the served model the environment names)."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    def judge(comparison, attempts_left, history):
        path = scratch / f'settle-{abs(hash(json.dumps(comparison, sort_keys=True)))}.json'
        path.write_text(json.dumps({'comparison': comparison, 'attemptsLeft': attempts_left, 'history': history}))
        done = subprocess.run(natlang_command(['--', 'settle', str(path)]), capture_output=True, text=True, cwd=REPO,
                              env=model_env(os.environ.get('NATLANG_SERVER'), os.environ.get('NATLANG_MODEL')))
        try:
            return json.loads(done.stdout.strip().splitlines()[-1])['next']
        except Exception:
            return 'give-up'
    return judge


# ---- commit and package -----------------------------------------------------------------------------------------------
def apply_decisions(decisions, repo):
    """Write accepted files into ``repo`` (a checkout) and delete the programs the keep judgment deleted. Returns the
    paths written and removed, repository-relative."""
    repo = Path(repo)
    written, removed = [], []
    for decision in decisions:
        root = Path(decision['root'])
        rel_root = root.relative_to(REPO) if root.is_relative_to(REPO) else root.relative_to(repo)
        if decision['decision'] == 'accept':
            proposal = json.loads((Path(decision['proposal']) / 'proposal.json').read_text())
            for change in proposal['changes']:
                target = repo / rel_root / change['path']
                if change['after'] is None:
                    target.unlink(missing_ok=True)
                    removed.append(str(rel_root / change['path']))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(change['after'])
                    written.append(str(rel_root / change['path']))
        elif decision['decision'] == 'delete':
            shutil.rmtree(repo / rel_root, ignore_errors=True)
            removed.append(str(rel_root))
    return written, removed


def reserved_programs(reservations):
    """Programs held for held-v4 (07 §5): their pairs never enter training."""
    spec = json.loads(Path(reservations).read_text())
    out = {'design-brief:' + c['brief'].split('-', 1)[0] for c in (spec.get('design_briefs') or {}).get('chosen', [])}
    out |= {c['application'].replace('/', ':') for c in (spec.get('applications') or {}).get('chosen', [])}
    return out


def package(decision_rows, out, *, reserved=frozenset()):
    """One ``natlang.semantic-fp-migration-pair/1`` row per accepted migration: the original and migrated files of the
    program, its findings, the step kinds and the equivalence comparison. Reserved programs are left out."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for decision in decision_rows:
        if decision['decision'] != 'accept':
            continue
        if decision['program'] in reserved:
            skipped.append(decision['program'])
            continue
        proposal = json.loads((Path(decision['proposal']) / 'proposal.json').read_text())
        root = Path(decision['root'])
        before = equivalence.program_files(root)
        after = dict(before)
        for change in proposal['changes']:
            if change['after'] is None:
                after.pop(change['path'], None)
            else:
                after[change['path']] = change['after']
        rows.append({'schema': PAIR_SCHEMA, 'id': f"migration:{decision['program']}", 'program': decision['program'],
                     'split_group': split_group(decision['program']), 'family': 'refactor-to-semantic-fp',
                     'findings': decision['findings'], 'model': decision.get('model'),
                     'changes': [{k: change.get(k) for k in ('path', 'role', 'steps', 'repairAttempts')} for change in proposal['changes']],
                     'before': {'files': before, 'sha256': equivalence.files_sha256(before)},
                     'after': {'files': after, 'sha256': equivalence.files_sha256(after)},
                     'comparison': decision['comparison']})
    (out / 'pairs.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    return rows, skipped


def program_tasks(case_rows, decision_rows, out, *, reserved=frozenset(), oracles=('exact', 'verifier')):
    """The accepted programs' own cases as ordinary ``natlang.program/2`` tasks whose root is the migrated program
    (07 §5 "an idiomatic program for the RL pool"): the measured candidate rows, with their original split restored,
    an id of their own (the split group, and so held exclusion by group, is unchanged) and the migration recorded."""
    accepted = {}
    for decision in decision_rows:
        if decision['decision'] == 'accept':
            accepted[decision['program']] = decision
    rows = []
    for row in case_rows:
        marker = row.get('equivalence') or {}
        program = marker.get('program')
        if program not in accepted or program in reserved or marker.get('oracle') not in oracles:
            continue
        rows.append({**{k: v for k, v in row.items() if k != 'equivalence'}, 'id': f"{row['id']}:sfp-migrated",
                     'split': marker.get('split') or row.get('split'),
                     'semantic_fp_migration': {'program': program, 'source_sha256': marker.get('source_sha256'),
                                               'original_id': row['id'], 'model': accepted[program].get('model')}})
    rows.sort(key=lambda r: r['id'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    return rows


def split_group(program):
    corpus, _, name = program.partition(':')
    return f'design-brief-app-{name}' if corpus == 'design-brief' else f'{corpus}-app-{name}'


# ---- CLI --------------------------------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    u = sub.add_parser('units')
    u.add_argument('--baseline', type=Path, required=True)
    u.add_argument('--out', type=Path, required=True)
    u.add_argument('--program', action='append')
    u.add_argument('--problems', type=Path, help='decisions of an earlier attempt: retried programs carry their problems')
    p = sub.add_parser('propose')
    p.add_argument('--units', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--server')
    p.add_argument('--model')
    p.add_argument('--provider')
    c = sub.add_parser('cases')
    c.add_argument('--proposals', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--kind', action='append')
    s = sub.add_parser('settle')
    s.add_argument('--original', type=Path, required=True)
    s.add_argument('--candidate', type=Path)
    s.add_argument('--proposals', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--attempts-left', type=int, default=0)
    s.add_argument('--history', type=Path, action='append', default=[])
    a = sub.add_parser('apply')
    a.add_argument('--decisions', type=Path, required=True)
    a.add_argument('--repo', type=Path, required=True)
    k = sub.add_parser('package')
    k.add_argument('--decisions', type=Path, action='append', required=True)
    k.add_argument('--out', type=Path, required=True)
    k.add_argument('--exclude-reserved', type=Path)
    t = sub.add_parser('tasks')
    t.add_argument('--cases', type=Path, action='append', required=True, help='a candidate cases directory (cases.jsonl)')
    t.add_argument('--decisions', type=Path, action='append', required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--exclude-reserved', type=Path)
    t.add_argument('--oracle', action='append', help='oracle levels to keep (default: exact and verifier)')
    args = parser.parse_args(argv)

    if args.command == 'units':
        problems = None
        if args.problems:
            problems = {d['program']: {'problems': d.get('problems') or [], 'history': d.get('history') or []}
                        for d in read_jsonl(args.problems) if d['decision'] == 'retry'}
        programs = set(args.program or []) or (set(problems) if problems is not None else None)
        for path in write_units(list(read_jsonl(args.baseline)), args.out, programs=programs, problems=problems):
            print(path)
    elif args.command == 'propose':
        propose_all(args.units, args.out, jobs=args.jobs, server=args.server, model=args.model, provider=args.provider)
    elif args.command == 'cases':
        sources = candidate_sources(args.proposals)
        if not sources:
            raise SystemExit('no checked candidate to measure')
        sha = equivalence.build_cases(args.out, programs=sorted(sources), sources={k: Path(v) for k, v in sources.items()},
                                      version='migrated', kinds=args.kind)
        print(json.dumps({'manifest': str(args.out / 'manifest.json'), 'sha256': sha, 'programs': sorted(sources)}))
    elif args.command == 'settle':
        history = {}
        for path in args.history:
            for d in read_jsonl(path):
                history.setdefault(d['program'], []).extend(d.get('history') or [])
        decisions = settle(list(read_jsonl(args.original)), list(read_jsonl(args.candidate)) if args.candidate else [],
                           args.proposals, attempts_left=args.attempts_left, history=history,
                           judge=settle_judge(args.out.parent / 'settle-inputs'))
        args.out.write_text(''.join(json.dumps(d, sort_keys=True) + '\n' for d in decisions))
        print(json.dumps(collections.Counter(d['decision'] for d in decisions)))
    elif args.command == 'apply':
        written, removed = apply_decisions(list(read_jsonl(args.decisions)), args.repo)
        print(json.dumps({'written': written, 'removed': removed}, indent=1))
    elif args.command == 'package':
        rows = [d for path in args.decisions for d in read_jsonl(path)]
        reserved = reserved_programs(args.exclude_reserved) if args.exclude_reserved else frozenset()
        packed, skipped = package(rows, args.out, reserved=reserved)
        print(json.dumps({'pairs': len(packed), 'reserved_skipped': skipped, 'out': str(args.out / 'pairs.jsonl')}))
    elif args.command == 'tasks':
        cases = [row for directory in args.cases for row in read_jsonl(directory / 'cases.jsonl')]
        decisions = [d for path in args.decisions for d in read_jsonl(path)]
        reserved = reserved_programs(args.exclude_reserved) if args.exclude_reserved else frozenset()
        rows = program_tasks(cases, decisions, args.out, reserved=reserved, oracles=tuple(args.oracle or ['exact', 'verifier']))
        # Every held set's exclusion (held-v3, held-v4 and its reservations, by id and split group), as trainers apply it.
        from natlang_neuralese.rl import held_eval
        excluded = held_eval.training_exclusion_ids(rows)
        if excluded:
            rows = [row for row in rows if row['id'] not in excluded]
            (args.out / 'tasks.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        print(json.dumps({'tasks': len(rows), 'held_excluded': len(excluded),
                          'programs': len({r['semantic_fp_migration']['program'] for r in rows}), 'out': str(args.out / 'tasks.jsonl')}))


if __name__ == '__main__':
    main()
