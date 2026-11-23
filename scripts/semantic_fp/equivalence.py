#!/usr/bin/env python3
"""Equivalence service for the legacy prose corpora (plans/semantic-fp/07_CORPUS_MIGRATION.md, M1).

Runs a program on its own corpus cases against a served actor at k samples, reports per case pass, model calls,
prompt and completion tokens and wall time, and compares two versions of a program on the same cases and seeds.

Corpora (one program per design brief, one per Astra/Sol application):

* design briefs: ``codebases/design_briefs/NNN-*`` with ``training/semantic_applications/design_briefs/NNN-*/cases.json``
  (3 exact-oracle gate cases and 5 rubric-judged workflow cases each), frozen to ``natlang.program/2`` rows by
  ``ts-host/scripts/design-briefs/build.mjs``;
* Astra/Sol applications: every ``project.json`` under ``training/semantic_applications/{astra,sol}/<app>/`` (one world
  each; 14 worlds for 11 apps), packaged by ``ts-host/scripts/inline-curriculum/package-semantic-application.mjs`` and
  scored by the world's ``verify-result.mjs`` (``trusted_exact_verifier``).

Execution is held eval's (``rl/held_eval.py``: work-stealing ``run_bonsai_queue`` episodes, the collector's oracle,
``task_rollouts.score``), with the student at the root (``task_roots.student_root_row``) and the RL sampling law. Each
sample's seed is a function of its case id and sample index only, so two versions (or two runs over different program
subsets) pair seed for seed. Rubric-judged cases are scored only with a separately configured judge (``--judge-*``,
forwarded to the collector as ``judge_model``); without one they are reported ``judge_pending`` and never as failures.

    equivalence.py cases --out DIR [--program design-brief:001 ...] [--source PROGRAM=DIR] [--root PROGRAM:OLD=NEW]
    equivalence.py run --cases DIR --out RUN --endpoint URL=SLOTS [--samples 4] [--judge-model-id ID --judge-provider P]
    equivalence.py report RUN [--append baseline.jsonl]
    equivalence.py compare A/report.jsonl B/report.jsonl [--out FILE]
    equivalence.py summary baseline.jsonl
    equivalence.py requeue RUN     (attempts that wrote no result, e.g. during an actor outage, run again on resume)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))

from natlang_neuralese.common.hashing import sha256_hex as sha256_bytes  # noqa: E402
from natlang_neuralese.common.jsonio import read_jsonl  # noqa: E402

SCHEMA = 'natlang.semantic-fp-equivalence/1'
DESIGN_SOURCE = 'codebases/design_briefs'
DESIGN_CASES = 'training/semantic_applications/design_briefs'
APP_ROOTS = ('training/semantic_applications/astra', 'training/semantic_applications/sol')
# Seeds pair by (case id, sample); the base is held eval's so the law is the same.
SEED_SPAN = 1 << 24
# A legacy program interprets its plumbing in model calls; the held budget of 40 calls would score most of them 0 for a
# budget stop. The bound here only stops a runaway episode.
MAX_MODEL_REQUESTS = 400


def node_binary():
    found = os.environ.get('NATLANG_NODE') or shutil.which('node')
    if not found:
        raise SystemExit('node is required (set NATLANG_NODE)')
    return found


def case_seed(seed_base, case_id, sample):
    """The seed of one sample: a function of the case id and sample index only (not of the case's queue position)."""
    return seed_base + (int(sha256_bytes(case_id.encode())[:6], 16) * 8 + sample) % SEED_SPAN


# ---- programs and their rows ------------------------------------------------------------------------------------------
def program_of_design_row(row):
    family = row.get('family') or ''
    if not family.startswith('design_brief_'):
        raise ValueError(f'not a design-brief row: {row.get("id")}')
    return 'design-brief:' + family.removeprefix('design_brief_')


def app_projects(repo=REPO):
    """``[(program, world, project.json path)]`` of every Astra/Sol world: the program is the application directory."""
    found = []
    for base in APP_ROOTS:
        corpus = Path(base).name
        for app in sorted((repo / base).iterdir()):
            if not app.is_dir():
                continue
            for project in sorted(app.rglob('project.json')):
                world = project.parent.relative_to(app).as_posix()
                found.append((f'{corpus}:{app.name}', '.' if world == '.' else world, project))
    return found


def design_rows(repo=REPO, *, node=None):
    """Every design-brief case as the freezer's ``natlang.program/2`` row (build.mjs validates all 100 programs)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'packet'
        subprocess.run([node or node_binary(), str(repo / 'ts-host/scripts/design-briefs/build.mjs'), '--out', str(out)],
                       cwd=repo, check=True, stdout=subprocess.DEVNULL)
        rows = []
        for path in sorted(out.glob('*.ir.jsonl')):
            rows.extend(read_jsonl(path))
    return rows


def app_rows(repo=REPO, *, node=None, programs=None):
    """``[(program, world, row)]``: each Astra/Sol world packaged without source-review custody (an execution
    measurement needs the program, its world and its verifier, not review receipts)."""
    found = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, (program, world, project) in enumerate(app_projects(repo)):
            if programs is not None and program not in programs:
                continue
            out = Path(tmp) / f'p{index}'
            subprocess.run([node or node_binary(), str(repo / 'ts-host/scripts/inline-curriculum/package-semantic-application.mjs'),
                            '--manifest', str(project), '--out', str(out), '--no-review-custody'],
                           cwd=repo, check=True, stdout=subprocess.DEVNULL)
            [row] = read_jsonl(out / 'source-package.ir.jsonl')
            found.append((program, world, row))
    return found


def oracle_level(row):
    semantics = row['semantics']
    if semantics.get('trusted_exact_verifier'):
        return 'verifier'
    oracle = semantics.get('oracle')
    return oracle if isinstance(oracle, str) else (oracle or {}).get('level')


def case_kind(row):
    level = oracle_level(row)
    return {'verifier': 'world', 'exact': 'gate', 'normalized': 'gate', 'judged': 'workflow'}.get(level, level)


def program_files(directory):
    """The model-visible files of a program version: its ``.nl`` and hand-written ``.ts`` sources and ``natlang.json``,
    by path under ``directory`` (generated declarations and dependency folders excluded)."""
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob('*')):
        rel = path.relative_to(directory).as_posix()
        if not path.is_file() or any(part in ('node_modules', '.natlang') or part.startswith('.') for part in rel.split('/')):
            continue
        if rel.endswith('.nl') or (rel.endswith('.ts') and not rel.endswith('.d.ts')) or rel == 'natlang.json':
            files[rel] = path.read_text(encoding='utf-8')
    if not any(name.endswith('.nl') or name.endswith('.ts') for name in files):
        raise ValueError(f'{directory} holds no program sources')
    return files


def files_sha256(files):
    digest = hashlib.sha256()
    for name in sorted(files):
        for part in (name.encode(), b'\0', files[name].encode(), b'\0'):
            digest.update(part)
    return digest.hexdigest()


def with_version(row, program, version, files=None, roots=None):
    """``row`` measured as ``version`` of ``program``: the same id, inputs, oracle and seeds, with ``files`` replacing
    the program's sources and ``roots`` ({old: new}) remapping its entry. Forced to the train split, which the student
    root transform requires (``task_roots.student_root_row``); the original split is kept in ``equivalence``."""
    semantics = dict(row['semantics'])
    if files is not None:
        semantics['files'] = files
    root = (roots or {}).get(semantics['root'], semantics['root'])
    if root not in semantics['files']:
        raise ValueError(f'{row["id"]}: entry {root} is not a file of {program} ({version})')
    semantics['root'] = root
    marker = {'schema': SCHEMA, 'program': program, 'version': version, 'split': row.get('split'),
              'kind': case_kind(row), 'oracle': oracle_level(row), 'source_sha256': files_sha256(semantics['files'])}
    return {**row, 'split': 'train', 'semantics': semantics, 'equivalence': marker}


def parse_map(values, sep='='):
    out = {}
    for value in values:
        key, found, rest = value.partition(sep)
        if not found:
            raise SystemExit(f'expected KEY{sep}VALUE: {value}')
        out[key] = rest
    return out


def build_cases(out, *, programs=None, sources=None, roots=None, version='original', kinds=None, repo=REPO, node=None):
    """Write ``out/cases.jsonl`` and a held-eval manifest over it (``held_eval.write_manifest``) for ``programs`` (None:
    all 111), each from its source in the repository or from ``sources[program]``; returns the manifest's sha256."""
    from natlang_neuralese.rl import held_eval
    out = Path(out)
    sources, roots = sources or {}, roots or {}
    wanted = None if programs is None else set(programs)
    rows = []
    if wanted is None or any(p.startswith('design-brief:') for p in wanted):
        for row in design_rows(repo, node=node):
            program = program_of_design_row(row)
            if wanted is None or program in wanted:
                rows.append((program, row))
    if wanted is None or any(not p.startswith('design-brief:') for p in wanted):
        rows.extend((program, row) for program, _, row in app_rows(repo, node=node, programs=wanted))
    missing = (wanted or set()) - {program for program, _ in rows}
    if missing:
        raise SystemExit(f'unknown programs: {sorted(missing)}')
    files = {program: program_files(directory) for program, directory in sources.items()}
    root_maps = collections.defaultdict(dict)
    for key, new in roots.items():
        program, _, old = key.partition(':')
        root_maps[program][old] = new
    built = sorted((with_version(row, program, version if program in files else 'original', files.get(program),
                                 root_maps.get(program)) for program, row in rows), key=lambda r: r['id'])
    if kinds:
        built = [row for row in built if row['equivalence']['kind'] in set(kinds)]
    if len({row['id'] for row in built}) != len(built):
        raise ValueError('duplicate case id')
    tasks = [{'id': row['id'], 'family': row.get('family'), 'program': row['equivalence']['program'],
              'kind': row['equivalence']['kind'], 'oracle': row['equivalence']['oracle'],
              'level': ((row.get('curriculum') or {}).get('rl') or {}).get('level'),
              'row_sha256': sha256_bytes(json.dumps(row).encode())} for row in built]
    programs_index = collections.defaultdict(lambda: {'cases': [], 'version': None, 'source_sha256': None})
    for row in built:
        entry = programs_index[row['equivalence']['program']]
        entry['cases'].append(row['id'])
        entry['version'] = row['equivalence']['version']
        entry['source_sha256'] = row['equivalence']['source_sha256']
        entry['source'] = str(sources.get(row['equivalence']['program']) or '')
    manifest = {'schema': held_eval.MANIFEST_SCHEMA, 'purpose': SCHEMA, 'tasks': tasks,
                'programs': dict(sorted(programs_index.items()))}
    return held_eval.write_manifest(out, manifest, built)


# ---- measurement ------------------------------------------------------------------------------------------------------
def judge_binding(args):
    if not args.judge_model_id:
        if args.judge_provider or args.judge_server:
            raise SystemExit('--judge-provider/--judge-server need --judge-model-id')
        return None
    if bool(args.judge_provider) == bool(args.judge_server):
        raise SystemExit('--judge-model-id needs exactly one of --judge-provider or --judge-server')
    return {'model_id': args.judge_model_id,
            **({'provider': args.judge_provider} if args.judge_provider else {'server': args.judge_server})}


RESUME_KEYS = ('policy_id', 'manifest_sha256', 'samples', 'seed_base', 'case_seconds', 'max_model_requests',
               'max_turns', 'context_tokens', 'sampling', 'judge_model')


def record_equivalence(directory, spec):
    """The finished measurement as a registry dataset (kind ``probe``, id ``semantic-fp-equivalence-...``), replacing held eval's recorder: these
    are not held-set episodes."""
    try:
        from natlang_neuralese.common import data_registry
        data_registry.record_dataset(
            data_registry.dataset_id(directory, 'semantic-fp-equivalence'), directory, kind='probe',
            include=['result.json', 'run.json', 'rewards.jsonl', 'packet.jsonl', 'journal.jsonl', 'report.jsonl'],
            format=f'{SCHEMA} report.jsonl + held-eval run files of the episodes', inputs=[spec['manifest']],
            license='project-generated (legacy-corpus behaviour measurement)',
            note=f"policy {spec.get('policy_id')}, {spec.get('samples')} samples per case")
    except Exception as error:   # bookkeeping never fails a measurement
        print(json.dumps({'event': 'data registry error', 'error': repr(error)[:300]}), flush=True)


def measure(cases, out, endpoints, *, samples=4, seed_base=None, case_seconds=3600, max_model_requests=MAX_MODEL_REQUESTS,
            max_turns=128, judge=None, label=None, top_p=None, policy_id=None):
    """Run every case of ``cases/manifest.json`` x ``samples`` on ``endpoints`` ([(url, slots)]) into ``out`` (resumable)
    and write ``out/report.jsonl``. The served policy is what the actors attest (all must agree, or ``policy_id``)."""
    from natlang_neuralese.rl import held_eval
    from natlang_neuralese.rl.actor_client import NativeActorClient
    from natlang_neuralese.common.code_provenance import code_provenance, repository_root
    from natlang_neuralese.common.sampling import RL_TOP_P, sampling_law_of
    cases, out = Path(cases).resolve(), Path(out).resolve()
    manifest = cases / 'manifest.json'
    seed_base = held_eval.SEED_BASE if seed_base is None else seed_base
    infos = [held_eval._read_info(NativeActorClient(url, timeout=10)) for url, _ in endpoints]
    attested = {held_eval.policy_id_of(info.get('rl_policy_binding') or {}) for info in infos}
    if len(attested) != 1 or (policy_id and attested != {policy_id}):
        raise SystemExit(f'the endpoints attest {sorted(map(str, attested))}, expected one policy {policy_id or ""}')
    pinned = json.loads(manifest.read_text())
    repo = repository_root()
    spec = {'purpose': SCHEMA, 'policy_id': attested.pop(), 'label': label, 'endpoint': endpoints[0][0],
            'endpoints': [list(e) for e in endpoints], 'dir': str(out), 'manifest': str(manifest),
            'cases': str(cases / pinned['cases']['path']), 'manifest_sha256': held_eval._sha256(manifest),
            'tasks': len(pinned['tasks']), 'samples': samples, 'seed_base': seed_base,
            'workers': sum(n for _, n in endpoints), 'case_seconds': case_seconds, 'max_model_requests': max_model_requests,
            'max_turns': max_turns, 'context_tokens': held_eval.CONTEXT_TOKENS, 'runtime': str(repo / 'ts-host'),
            'repo': str(repo), 'series': held_eval.series(), 'code': code_provenance(),
            'actor': {'info': infos[0].get('rl_policy_binding')}, 'engine': held_eval.engine_of(infos[0].get('rl_policy_binding')),
            'sampling': sampling_law_of(RL_TOP_P if top_p is None else top_p), 'judge_model': judge,
            'dispatch': {'order': 'longest-first', 'prior': []}}
    served = infos[0].get('context')
    if isinstance(served, int) and served < spec['context_tokens']:
        raise SystemExit(f'the actor serves {served} tokens per request, less than the call budget {spec["context_tokens"]}')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run.json').exists():
        previous = json.loads((out / 'run.json').read_text())
        for key in RESUME_KEYS:
            if previous.get(key) != spec.get(key):
                raise SystemExit(f'{out} holds a run with another {key}: {previous.get(key)!r}')
        spec = {**previous, 'endpoint': spec['endpoint'], 'endpoints': spec['endpoints'], 'workers': spec['workers'],
                'repo': spec['repo'], 'runtime': spec['runtime'], 'resumed_code': spec['code']}
    (out / 'run.json').write_text(json.dumps(spec, indent=1) + '\n')
    queue = out / 'sealed-queue.jsonl'
    if not queue.exists():
        entries = held_eval._entries(spec)
        packet = list(read_jsonl(out / 'packet.jsonl'))
        for entry in entries:
            entry['seed'] = case_seed(seed_base, packet[entry['index']]['id'], entry['sample'])
            if judge:
                entry['judge_model'] = judge
        queue.write_text(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in entries))
    (out / 'result.json').unlink(missing_ok=True)
    held_eval.record_result = lambda *_: None   # not a held-set measurement: recorded below with its report
    result = held_eval.run(out)
    rows = report(out)
    (out / 'report.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    record_equivalence(out, spec)
    return result, rows


def requeue(run, *, timeouts=False):
    """Forget the finished attempts of a stopped run that wrote no result row (the collector could not reach the actor),
    so that resuming the run measures them again; an attempt with a result, pass or fail, is kept. With ``timeouts`` an
    attempt the case wall clock stopped is forgotten too (on an oversubscribed shared actor the clock measures queueing,
    not the program). Returns their keys."""
    run = Path(run)
    dropped = []
    for entry in read_jsonl(run / 'sealed-queue.jsonl'):
        journal = run / 'workers' / f"{entry['key']}.journal.jsonl"
        finish = next((r for r in read_jsonl(journal) if r.get('event') == 'finish'), None) if journal.exists() else None
        if finish is None:
            continue
        output = Path(entry['output'])
        if output.is_file() and output.stat().st_size > 0 and not (timeouts and finish.get('status') == 'timeout'):
            continue
        journal.unlink()
        shutil.rmtree(output.parent, ignore_errors=True)
        dropped.append(entry['key'])
    return dropped


# ---- reports ----------------------------------------------------------------------------------------------------------
def _sample_cost(output):
    """(model calls, prompt tokens, completion tokens, final value, oracle verdict, status) of one episode's result."""
    rows = read_jsonl(output) if Path(output).is_file() else []
    if not rows:
        return None
    result = rows[0]
    calls = prompt = completion = 0
    for step in result.get('trajectory') or []:
        response = (step or {}).get('model_response') if isinstance(step, dict) else None
        if isinstance(response, dict):
            calls += 1
            prompt += int(response.get('prompt_tokens') or 0)
            completion += int(response.get('completion_tokens') or 0)
    outcome = result.get('outcome') if isinstance(result.get('outcome'), dict) else {}
    return {'calls': calls, 'prompt_tokens': prompt, 'completion_tokens': completion,
            'status': outcome.get('status'), 'oracle': outcome.get('oracle')}


def judge_pending(oracle_name, cost, judged):
    """A rubric-judged case whose program returned a value but which ran without a judge: neither a pass nor a failure.
    An episode that returned no value (an error, a budget stop) fails whatever a judge would say."""
    if oracle_name != 'judged' or judged or not cost:
        return False
    return cost.get('status') == 'done' and not (cost.get('oracle') or {}).get('accepted')


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def report(run):
    """One row per case of a finished (or partial) run: passes over scored samples and mean cost per sample."""
    run = Path(run)
    spec = json.loads((run / 'run.json').read_text())
    packet = list(read_jsonl(run / 'packet.jsonl'))
    queue = list(read_jsonl(run / 'sealed-queue.jsonl'))
    rewards = {(r['group'], r['sample']): r for r in read_jsonl(run / 'rewards.jsonl')} if (run / 'rewards.jsonl').exists() else {}
    judged = bool(spec.get('judge_model'))
    by_case = collections.defaultdict(list)
    for entry in queue:
        journal = run / 'workers' / f"{entry['key']}.journal.jsonl"
        finish = next((r for r in (read_jsonl(journal) if journal.exists() else []) if r.get('event') == 'finish'), None)
        # A reward counts only for an attempt this run's journal finished (a requeued attempt's old reward is stale).
        reward = (rewards.get((entry['group'], entry['sample'])) or {}) if finish else {}
        cost = _sample_cost(entry['output']) if finish else None
        row = packet[entry['index']]
        marker = row.get('equivalence') or {}
        pending = judge_pending(marker.get('oracle'), cost, judged)
        by_case[row['id']].append({
            'sample': entry['sample'], 'seed': entry['seed'], 'ran': finish is not None,
            'outcome': reward.get('outcome'), 'reward': None if pending else reward.get('reward'),
            'judge_pending': pending, 'wall_seconds': (finish or {}).get('elapsed_seconds'),
            **({k: cost[k] for k in ('calls', 'prompt_tokens', 'completion_tokens', 'status')} if cost else {})})
    rows = []
    for row in sorted(packet, key=lambda r: r['id']):
        samples = sorted(by_case[row['id']], key=lambda s: s['sample'])
        marker = row.get('equivalence') or {}
        scored = [s for s in samples if s['reward'] is not None]
        rows.append({
            'schema': SCHEMA, 'case_id': row['id'], 'program': marker.get('program'), 'version': marker.get('version'),
            'kind': marker.get('kind'), 'oracle': marker.get('oracle'), 'split': marker.get('split'),
            'source_sha256': marker.get('source_sha256'), 'policy_id': spec['policy_id'],
            'samples': len(samples), 'ran': sum(s['ran'] for s in samples), 'scored': len(scored),
            'passes': sum(s['reward'] == 1.0 for s in scored), 'judge_pending': sum(s['judge_pending'] for s in samples),
            'seeds': [s['seed'] for s in samples],
            'outcomes': dict(collections.Counter(str(s['outcome']) for s in samples if s['ran'])),
            'statuses': dict(collections.Counter(str(s.get('status')) for s in samples if s['ran'])),
            'calls_mean': _mean([s.get('calls') for s in samples]),
            'prompt_tokens_mean': _mean([s.get('prompt_tokens') for s in samples]),
            'completion_tokens_mean': _mean([s.get('completion_tokens') for s in samples]),
            'wall_seconds_mean': _mean([s.get('wall_seconds') for s in samples]),
            'per_sample': samples, 'run': str(run), 'code': (spec.get('code') or {}).get('commit')})
    return rows


def compare(first, second, *, slack_cases=1):
    """Per program, version ``second`` against ``first`` on the same cases and seeds (07 §4.2): accepted when its
    expected passing cases (sum of per-case pass rates over scored samples) is at least the original's minus
    ``slack_cases``, and every case the original passes in all k samples it also passes in all k. Rubric cases without a
    judge and unscored samples are reported, never counted."""
    a = {row['case_id']: row for row in first}
    b = {row['case_id']: row for row in second}
    programs = collections.defaultdict(list)
    for case_id, row in a.items():
        programs[row['program']].append(case_id)
    out = []
    for program, ids in sorted(programs.items()):
        problems, rate_a, rate_b, cost = [], 0.0, 0.0, collections.defaultdict(list)
        for case_id in sorted(ids):
            x, y = a[case_id], b.get(case_id)
            if y is None:
                problems.append(f'{case_id}: missing from the second report')
                continue
            if x['seeds'] != y['seeds']:
                problems.append(f'{case_id}: seeds differ')
            if x.get('policy_id') != y.get('policy_id'):
                problems.append(f'{case_id}: measured on another policy ({x.get("policy_id")} vs {y.get("policy_id")})')
            if not (x['scored'] and y['scored']):
                continue
            rate_a += x['passes'] / x['scored']
            rate_b += y['passes'] / y['scored']
            if x['passes'] == x['samples'] and y['passes'] < y['samples']:
                problems.append(f'{case_id}: the original passes {x["passes"]}/{x["samples"]}, the new version {y["passes"]}/{y["samples"]}')
            for key in ('calls_mean', 'prompt_tokens_mean', 'completion_tokens_mean', 'wall_seconds_mean'):
                if x.get(key) is not None and y.get(key) is not None:
                    cost[key].append((x[key], y[key]))
        if rate_b < rate_a - slack_cases:
            problems.append(f'expected passing cases {rate_b:.2f} < {rate_a:.2f} - {slack_cases}')
        out.append({'program': program, 'cases': len(ids), 'first_version': a[ids[0]]['version'],
                    'second_version': (b.get(ids[0]) or {}).get('version'),
                    'expected_passing_first': round(rate_a, 2), 'expected_passing_second': round(rate_b, 2),
                    'scored_cases': sum(bool(a[i]['scored'] and (b.get(i) or {}).get('scored')) for i in ids),
                    'cost': {key: {'first': _mean([p[0] for p in pairs]), 'second': _mean([p[1] for p in pairs])}
                             for key, pairs in cost.items()},
                    'accepted': not problems, 'problems': problems})
    return out


def corpus_of(program):
    return program.split(':', 1)[0]


def summary(rows):
    """Per corpus and per case kind: programs, cases, scored samples, passes; programs with no passing case."""
    groups = collections.defaultdict(lambda: collections.Counter())
    programs = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        key = (corpus_of(row['program']), row['kind'])
        g = groups[key]
        g['cases'] += 1
        g['samples'] += row['samples']
        g['ran'] += row['ran']
        g['scored'] += row['scored']
        g['passes'] += row['passes']
        g['judge_pending'] += row['judge_pending']
        g['cases_any_pass'] += row['passes'] > 0
        g['cases_all_pass'] += bool(row['scored']) and row['passes'] == row['samples']
        p = programs[row['program']]
        p['scored'] += row['scored']
        p['passes'] += row['passes']
        p['cases_any_pass'] += row['passes'] > 0
    table = [{'corpus': corpus, 'kind': kind, **counts,
              'pass_rate': round(counts['passes'] / counts['scored'], 4) if counts['scored'] else None}
             for (corpus, kind), counts in sorted(groups.items())]
    never = sorted(p for p, c in programs.items() if c['scored'] and not c['passes'])
    unmeasured = sorted(p for p, c in programs.items() if not c['scored'])
    return {'by_corpus_kind': table, 'programs': len(programs),
            'programs_with_a_passing_case': sum(c['passes'] > 0 for c in programs.values()),
            'programs_scored_without_a_pass': never, 'programs_without_a_scored_sample': unmeasured,
            'per_program': {p: dict(c) for p, c in sorted(programs.items())}}


# ---- CLI --------------------------------------------------------------------------------------------------------------
def parse_endpoint(value):
    url, _, slots = value.partition('=')
    return url.rstrip('/'), int(slots or 16)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cases', help='write cases.jsonl + manifest.json for programs (default: all 111)')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--program', action='append', help='design-brief:NNN, astra:APP or sol:APP (repeatable)')
    c.add_argument('--source', action='append', default=[], help='PROGRAM=DIR: measure the version in DIR')
    c.add_argument('--root', action='append', default=[], help='PROGRAM:OLD=NEW: the version\'s entry for a case root')
    c.add_argument('--version', default='candidate', help='label of the --source versions')
    c.add_argument('--kind', action='append', choices=('gate', 'workflow', 'world'),
                   help='only cases of this kind (repeatable; seeds pair across subsets)')
    r = sub.add_parser('run', help='measure the cases on served actors (resumable); writes RUN/report.jsonl')
    r.add_argument('--cases', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--endpoint', action='append', required=True, type=parse_endpoint, help='URL[=SLOTS] (repeatable)')
    r.add_argument('--samples', type=int, default=4)
    r.add_argument('--seed-base', type=int)
    r.add_argument('--case-seconds', type=int, default=3600)
    r.add_argument('--max-model-requests', type=int, default=MAX_MODEL_REQUESTS)
    r.add_argument('--max-turns', type=int, default=128)
    r.add_argument('--policy-id', help='the checkpoint sha256 the actors must attest')
    r.add_argument('--top-p', type=float)
    r.add_argument('--label')
    r.add_argument('--judge-model-id')
    r.add_argument('--judge-provider', help='Pi provider id of the rubric judge')
    r.add_argument('--judge-server', help='OpenAI-compatible URL of the rubric judge')
    r.add_argument('--append', type=Path, help='also append the report rows to this JSONL (e.g. baseline.jsonl)')
    p = sub.add_parser('report', help='rewrite RUN/report.jsonl from a (partial) run')
    p.add_argument('run', type=Path)
    p.add_argument('--append', type=Path)
    q = sub.add_parser('compare', help='second report against first, per program (07 §4.2)')
    q.add_argument('first', type=Path)
    q.add_argument('second', type=Path)
    q.add_argument('--slack-cases', type=float, default=1.0)
    q.add_argument('--out', type=Path)
    w = sub.add_parser('requeue', help='a stopped run: attempts that wrote no result run again on resume')
    w.add_argument('run', type=Path)
    w.add_argument('--timeouts', action='store_true', help='also rerun attempts the case wall clock stopped')
    s = sub.add_parser('summary', help='pass counts per corpus and case kind of a report or baseline JSONL')
    s.add_argument('report', type=Path)
    args = parser.parse_args(argv)

    def append(rows, path):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as handle:
                handle.write(''.join(json.dumps({k: v for k, v in row.items() if k != 'per_sample'}, sort_keys=True) + '\n'
                                     for row in rows))

    if args.command == 'cases':
        sha = build_cases(args.out, programs=args.program, sources={k: Path(v) for k, v in parse_map(args.source).items()},
                          roots=parse_map(args.root), version=args.version, kinds=args.kind)
        manifest = json.loads((args.out / 'manifest.json').read_text())
        print(json.dumps({'manifest': str(args.out / 'manifest.json'), 'sha256': sha, 'cases': len(manifest['tasks']),
                          'programs': len(manifest['programs'])}))
    elif args.command == 'run':
        result, rows = measure(args.cases, args.out, args.endpoint, samples=args.samples, seed_base=args.seed_base,
                               case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                               max_turns=args.max_turns, judge=judge_binding(args), label=args.label, top_p=args.top_p,
                               policy_id=args.policy_id)
        append(rows, args.append)
        print(json.dumps(summary(rows)['by_corpus_kind']))
    elif args.command == 'report':
        rows = report(args.run)
        (args.run / 'report.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        append(rows, args.append)
        print(json.dumps(summary(rows)['by_corpus_kind']))
    elif args.command == 'compare':
        rows = compare(list(read_jsonl(args.first)), list(read_jsonl(args.second)), slack_cases=args.slack_cases)
        text = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)
        if args.out:
            args.out.write_text(text)
        sys.stdout.write(text)
    elif args.command == 'requeue':
        dropped = requeue(args.run, timeouts=args.timeouts)
        print(json.dumps({'requeued': len(dropped), 'keys': dropped}))
    elif args.command == 'summary':
        print(json.dumps(summary(list(read_jsonl(args.report))), indent=1))


if __name__ == '__main__':
    main()
