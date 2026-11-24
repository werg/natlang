#!/usr/bin/env python3
"""Flagship benchmark harness (plans/semantic-fp/06 §4, brief B14): arms A-D on an app's held split, WS6 metrics.

    bench.py split APP --fixtures PATH [--out PATH]   # label any fixture pack's rows: split_group, split train|held
    bench.py setup APP --work WORK --base-profile P --policy-profile Q   # WORK/config with the bench profiles
    bench.py run APP --work WORK [--arms A,B,C-reasoned,C-instant,D-reasoned,D-instant] [--samples K]
    bench.py prepare-d APP --work WORK [--steps record,specialize,tune]   # arm D's compiled state, train split only
    bench.py report APP --work WORK [--out APP/bench/results/DATE.json] [--reading TEXT]
    bench.py compare RESULT.json [OTHER.json] --arm X [--against A]   # paired comparison from result files

Local arms need their native server up: ``NATLANG_CONFIG_HOME=WORK/config natlang serve --profile bench-base`` (or
``bench-policy``) under the memory ledger; the profiles' launchers admit the server itself as a ledger unit.

APP holds ``bench/bench.json`` (target, fixtures, verifier, the frontier arm's prose prompt and submit tool),
``bench/fixtures.json`` (rows: id, split_group, split train|held, arguments, expected) and ``bench/verify.py``
(``verify(expected, output) -> {passed, checks, safety}``). Every arm runs on the held split only.

Arms (WS6 §4):
  A   the frontier model in a standard tool loop with a prose task prompt (``frontier_loop.mjs``: natlang's Pi provider
      transport, `natlang auth` credentials; the app's services are its tools, plus the submit tool)
  B   the natlang program on the same frontier model (profile ``bench-frontier``; no compiled cases, no escalation)
  C   the program on base LFM3-1.7B (``bench-base``; no compiled cases, no escalation)
  D   the program on the current policy (``bench-policy``) after ``natlang specialize`` and ``natlang tune`` on the
      train split, escalating to B's model (the profile's ``escalate``)
  C and D run in two decision modes: ``reasoned`` (``readout: decision`` reasons first, the default) and ``instant``
  (``NATLANG_DECISION_SAMPLES=0``: every decision read out without reasoning, as ``readout: instant``).

B-D run the program as users do (``natlang run COPY --target T --profile P -- ARGS``, COPY = WORK/app, or WORK/app-D
for arm D, whose tune release and train-split examples.json apply to it alone), one process per unit, with
NATLANG_CONFIG_HOME = WORK/config (the bench profiles; ``credentials`` links to the user's store, never read here) and
NATLANG_CALL_STORE = WORK/calls/<arm> (each arm its own call store, never the machine's). Accounting comes from the unit's reduction traces:
model requests (tokens, duration, actor), decision readouts, tier events (``tier3`` = a compiled crisp case served).

Metrics per arm: verifier pass rate (bootstrap CI over units; vs A: mean per-unit pass-rate change with the within-unit
permutation p and unit-bootstrap CI of ``held_eval._rate_change``, and the exact sign test over units), p50/p95 latency (unit
span in the traces; A: the loop's wall time), tokens and dollars per 100 units (frontier at list price from Pi's
catalog; local at measured GPU-seconds, the wall time the local server was busy with the arm's requests, priced at
``--local-usd-per-gpu-hour``), crisp-served, local-resolved and escalation shares of natural-language calls, and the
verifier's safety counters.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get('NATLANG_REPO') or Path(__file__).resolve().parents[2])
sys.path[:0] = [str(REPO / 'training' / 'neuralese'), str(REPO)]
from natlang_neuralese.common import code_provenance  # noqa: E402
from natlang_neuralese.rl import held_eval  # noqa: E402

NODE = os.environ.get('NATLANG_NODE') or shutil.which('node') or 'node'
NATLANG = [NODE, str(REPO / 'scripts' / 'natlang')]
FRONTIER = {'provider': 'openai-codex', 'model': 'gpt-6-luna', 'piOptions': {'reasoningEffort': 'low'}}
# Pi's catalog (node_modules/@earendil-works/pi-ai/dist/providers/data/openai.json, gpt-6-luna, openai-responses): USD per
# million tokens, below 272k input tokens. The openai-codex catalog lists 0 (subscription), so list price is OpenAI's.
LIST_PRICE = {'gpt-6-luna': {'input': 0.10, 'cache_read': 0.01, 'cache_write': 0.125, 'output': 0.50}}
ARMS = ('A', 'B', 'C-reasoned', 'C-instant', 'D-reasoned', 'D-instant')
DEFAULT_LOCAL_USD_PER_GPU_HOUR = 1.0
# Failures of the serving infrastructure, not of the program or its model: never a verdict (rerun, excluded from
# reports). A unit past --unit-timeout is the program's failure (held_eval counts timeouts as failures too).
INFRA_ERRORS = ('OutOfMemoryError', 'CUDA out of memory', 'ECONNREFUSED', 'fetch failed', 'socket hang up',
                'the model or its runtime is unavailable', 'exit 3:')


def infra_error(error):
    return bool(error) and any(marker in error for marker in INFRA_ERRORS)


# ---- fixture splits (WS6 §1.4) ----------------------------------------------------------------------------------------
# held-v4's share rule for flagship packs (ops/held-v4/reservations.json ``share_rules``): a row whose split group falls in
# the held share is reserved for held-v4 from the moment it exists, and the pack's held split is exactly those rows.
FLAGSHIP_SHARE_RULE = {'pattern': '^flagship:', 'share': 0.2, 'seed': 'held-v4-flagship'}


def split_group_for(app, group):
    """The canonical split group ``flagship:<app>:<group>`` (a group already under ``flagship:`` is kept)."""
    return group if str(group).startswith('flagship:') else f'flagship:{app}:{group}'


def label_fixtures(app, rows, rule=FLAGSHIP_SHARE_RULE):
    """Each row with its canonical ``split_group`` (from ``split_group``, ``group`` or ``id``) and ``split``."""
    out = []
    for index, row in enumerate(rows):
        group = split_group_for(app, row.get('split_group') or row.get('group') or row.get('id') or f'item-{index}')
        out.append({**row, 'split_group': group,
                    'split': 'held' if held_eval.reservation_share_held(rule, group) else 'train'})
    return out


# ---- the app ---------------------------------------------------------------------------------------------------------

def load_app(app):
    app = Path(app).resolve()
    spec = json.loads((app / 'bench' / 'bench.json').read_text())
    fixtures = json.loads((app / 'bench' / spec['fixtures']).read_text())
    loader = importlib.util.spec_from_file_location('bench_verify', app / 'bench' / spec['verifier'])
    verifier = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(verifier)
    manifest = json.loads((app / 'natlang.json').read_text())
    target = manifest['targets'][spec['target']]
    defaults = {a['name']: a['default'] for a in target.get('arguments', []) if 'default' in a}
    return {'dir': app, 'spec': spec, 'rows': fixtures['rows'], 'verify': verifier.verify, 'defaults': defaults,
            'manifest': manifest}


def split_rows(app, split):
    return [r for r in app['rows'] if r['split'] == split]


# ---- the workspace: bench profiles ----------------------------------------------------------------------------------

def setup_config(work, policy_profile=None, base_profile=None):
    """WORK/config/config.json with the bench profiles. Native blocks are copied from the user's config profiles
    (``natlang setup --native`` wrote them); credentials link to the user's store."""
    user = Path(os.environ.get('NATLANG_USER_CONFIG_HOME') or Path.home() / '.config' / 'natlang')
    config_dir = work / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    if not (config_dir / 'credentials').exists():
        (config_dir / 'credentials').symlink_to(user / 'credentials')
    path = config_dir / 'config.json'
    config = json.loads(path.read_text()) if path.exists() else {'profiles': {}}
    config['profiles']['bench-frontier'] = {'provider': FRONTIER['provider'], 'model': FRONTIER['model'],
                                            'piOptions': FRONTIER['piOptions']}
    users = json.loads((user / 'config.json').read_text()).get('profiles', {})
    for name, source in (('bench-base', base_profile), ('bench-policy', policy_profile)):
        if source:
            profile = dict(users[source]) if source in users else json.loads(Path(source).read_text())
            config['profiles'][name] = profile
    if 'bench-policy' in config['profiles']:
        config['profiles']['bench-policy']['escalate'] = 'bench-frontier'
        config['tune'] = {**config.get('tune', {}), 'student': 'bench-policy'}
    for name in ('bench-base', 'bench-policy'):
        native = config['profiles'].get(name, {}).get('native')
        if native:   # this checkout's engine, admitted through the memory ledger (memory is tight: a modest budget)
            native.update(python=str(REPO / '.venv-neuralese' / 'bin' / 'python'),
                          pythonPath=[str(REPO / 'training' / 'neuralese'), str(REPO)],
                          launcher=['/usr/bin/python3', str(REPO / 'scripts' / 'memory_ledger.py'), 'run', '--unit', '{unit}',
                                    '--budget-gb', '16', '--priority', 'medium', '--wait', '7200', '--'],
                          stop=['systemctl', '--user', 'stop', '{unit}.service'], memoryGb=14, startTimeoutMs=7200000)
            native.setdefault('port', 18472 if name == 'bench-base' else 18471)
            # at most 4 requests in flight on the server: more ran its 14 GB out of memory on 4-5k-token tool-loop calls
            config['profiles'][name]['concurrency'] = 4
    config['defaultProfile'] = 'bench-frontier'
    path.write_text(json.dumps(config, indent=1) + '\n')
    return config


def arm_config(arm):
    """(profile, escalate, environment) of a program arm."""
    mode = arm.split('-', 1)[1] if '-' in arm else None
    env = {'NATLANG_DECISION_SAMPLES': '0'} if mode == 'instant' else {}
    if arm == 'B':
        return 'bench-frontier', 'none', {'NATLANG_SPECIALIZATION': 'off'}
    if arm.startswith('C'):
        return 'bench-base', 'none', {**env, 'NATLANG_SPECIALIZATION': 'off'}
    if arm.startswith('D'):
        return 'bench-policy', 'bench-frontier', env
    raise ValueError(arm)


def program_env(work, arm):
    env = dict(os.environ)
    env['NATLANG_CONFIG_HOME'] = str(work / 'config')
    # Each arm its own call store, never the machine's: held units recorded by one arm must not reach another arm's
    # compiled cases (arm D's instant and reasoned variants share one: specialize compiles its train-split recordings)
    env['NATLANG_CALL_STORE'] = str(work / 'calls' / arm.split('-')[0])
    env.update(arm_config(arm)[2])
    return env


def safe_id(ident):
    """Filesystem-safe, collision-free directory/file key for a unit or example id."""
    text = str(ident)
    cleaned = re.sub(r'\.{2,}', '_', re.sub(r'[^A-Za-z0-9._-]+', '_', text)).strip('._')
    if cleaned == text and cleaned not in ('', '.', '..'):
        return cleaned
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]
    return f'{cleaned or "unit"}-{digest}'


def app_copy(app, work, arm):
    """The program as an arm runs it, copied under WORK so builds and tune releases stay out of the repository: arm D
    has its own copy (its tune release and train-split examples.json apply to D only), the other arms share one."""
    return work / ('app-D' if arm.startswith('D') else 'app')


def _source_hashes(directory):
    ignored = {'.natlang', 'bench', 'node_modules'}
    hashes = {}
    for path in sorted(Path(directory).rglob('*')):
        rel = path.relative_to(directory)
        if any(part in ignored for part in rel.parts) or not path.is_file():
            continue
        hashes[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def prepare_app(app, work, arm):
    """Copy or sync the program for ``arm`` and build it, before units run in parallel (one build, no racing copies).
    Syncs source changes from ``app['dir']`` without clobbering ``prepare-d`` state (``.natlang/``, ``examples.json``)
    or in-place improvements when ``app['dir']`` has not changed."""
    dest = app_copy(app, work, arm)
    src = Path(app['dir'])
    manifest_path = dest / '.natlang' / '.bench-source-manifest.json'
    current = _source_hashes(src)
    if not dest.exists():
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns('.natlang', 'bench', 'node_modules'))
    else:
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        for rel, digest in current.items():
            target = dest / rel
            source_file = src / rel
            if previous is None:
                if rel == 'examples.json' and target.exists():
                    continue
                if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target)
            elif previous.get(rel) != digest or not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target)
        if previous is not None:
            for rel in previous:
                if rel not in current and rel != 'examples.json':
                    (dest / rel).unlink(missing_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(current, sort_keys=True) + '\n')
    env = {**os.environ, 'NATLANG_CONFIG_HOME': str(work / 'config')}
    proc = subprocess.run([*NATLANG, 'build', str(dest)], env=env, capture_output=True, text=True)
    if proc.returncode:
        raise SystemExit(f'natlang build {dest} failed: {proc.stderr[-2000:]}')
    return dest


def target_args(arguments):
    out = []
    for name, value in arguments.items():
        out += [f'--{name}', value if isinstance(value, str) else json.dumps(value)]
    return out


# ---- running ---------------------------------------------------------------------------------------------------------

def run_program_unit(app, work, arm, row, sample, *, timeout):
    unit = work / 'runs' / arm / safe_id(row['id']) / f's{sample}'
    done = unit / 'result.json'
    if done.exists():
        record = json.loads(done.read_text())
        if not infra_error(record.get('error')):
            return record
    if unit.exists():
        shutil.rmtree(unit)
    unit.mkdir(parents=True)
    profile, escalate, _ = arm_config(arm)
    command = [*NATLANG, 'run', str(app_copy(app, work, arm)), '--target', app['spec']['target'], '--profile', profile,
               '--escalate', escalate, '--traces', str(unit / 'traces'), '--plain', '--progress', 'off', '--',
               *target_args(row['arguments'])]
    started = time.time()
    try:
        proc = subprocess.run(command, env=program_env(work, arm), capture_output=True, text=True, timeout=timeout)
        code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr = 'timeout', error.stdout or '', error.stderr or ''
        stdout, stderr = (s.decode() if isinstance(s, bytes) else s for s in (stdout, stderr))
    wall = time.time() - started
    (unit / 'stderr.txt').write_text(stderr)
    output, error = None, None
    if code == 0:
        try:
            output = json.loads(stdout)
        except json.JSONDecodeError:
            # kept for diagnosis: B14's and the instant re-measure's 'stdout is not JSON' units exited 0 and passed on rerun
            (unit / 'stdout.txt').write_text(stdout)
            error = 'stdout is not JSON'
    else:
        error = f'exit {code}: ' + (stderr.strip().splitlines() or [''])[-1][:300]
    result = {'id': row['id'], 'sample': sample, 'output': output, 'error': error, 'process_wall_s': round(wall, 3),
              'accounting': trace_accounting(unit / 'traces')}
    if not infra_error(error):   # an outage of the serving infrastructure is no result: rerun it
        done.write_text(json.dumps(result) + '\n')
    return result


def frontier_user(app, row):
    spec = app['spec']['frontier']
    values = {**app['defaults'], **row['arguments']}
    rendered = {
        name: json.dumps(values[name], indent=1) if kind == 'json' else str(values[name])
        for name, kind in spec['arguments'].items()
    }
    return re.sub(r'\{(\w+)\}', lambda match: rendered[match.group(1)] if match.group(1) in rendered else match.group(0),
                  spec['user'])


def frontier_job(app, todo, *, concurrency):
    spec = app['spec']['frontier']
    services = app['spec'].get('services', [])
    tools = spec['tools'] + [service['tool'] for service in services]
    services_module = app['spec'].get('services_module') or spec.get('services_module')
    if services_module and not Path(services_module).is_absolute():
        services_module = str((app['dir'] / 'bench' / services_module).resolve())
    has_services = bool(services) or bool(services_module)
    max_turns = spec.get('max_turns')
    if max_turns is None:
        max_turns = app['spec'].get('max_turns')
    if max_turns is None:
        max_turns = 16 if has_services else 4
    job = {**FRONTIER, 'system': spec['system'], 'tools': tools, 'submit': spec['submit'], 'max_turns': max_turns,
           'concurrency': concurrency,
           'units': [{'id': f"{row['id']}#{s}", 'unit_id': row['id'], 'sample': s,
                      'arguments': row['arguments'], 'user': frontier_user(app, row)} for row, s in todo]}
    if services_module:
        job['services_module'] = services_module
    service_responses = app['spec'].get('service_responses') or spec.get('service_responses')
    if service_responses:
        job['service_responses'] = service_responses
    return job


def run_frontier(app, work, rows, samples, *, concurrency):
    out_dir = work / 'runs' / 'A'
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for path in out_dir.glob('*.json'):
        record = json.loads(path.read_text())
        results[(record['id'], record['sample'])] = record
    todo = [(row, s) for row in rows for s in range(samples) if (row['id'], s) not in results]
    if todo:
        job = frontier_job(app, todo, concurrency=concurrency)
        proc = subprocess.run([NODE, str(REPO / 'scripts' / 'semantic_fp' / 'frontier_loop.mjs')], input=json.dumps(job),
                              capture_output=True, text=True)
        if proc.returncode:
            raise SystemExit('frontier loop failed: ' + proc.stderr[-2000:])
        for line in proc.stdout.splitlines():
            record = json.loads(line)
            ident, sample = record['id'].rsplit('#', 1)
            record.update(id=ident, sample=int(sample))
            if (record.get('error') or '').startswith('transport'):
                continue   # a provider failure is no verdict: rerun it
            (out_dir / f'{safe_id(ident)}.s{sample}.json').write_text(json.dumps(record) + '\n')
            results[(ident, int(sample))] = record
    return [results[(row['id'], s)] for row in rows for s in range(samples) if (row['id'], s) in results]


# ---- accounting from traces ----------------------------------------------------------------------------------------

def is_frontier(actor):
    return str((actor or {}).get('id') or '').startswith(('gpt-', 'claude-', 'gemini-'))


def trace_accounting(directory):
    acc = {'calls': 0, 'model_calls': 0, 'frontier_calls': 0, 'crisp_served': 0, 'decisions_scored': 0,
           'decisions_generated': 0, 'local': {'prompt': 0, 'completion': 0, 'cached': 0, 'requests': 0, 'ms': 0},
           'frontier': {'prompt': 0, 'completion': 0, 'cached': 0, 'requests': 0, 'ms': 0}, 'span_ms': None,
           'local_intervals': []}
    first = last = None
    traces = []
    for path in sorted(Path(directory).glob('*.jsonl')) if Path(directory).exists() else ():
        actors, served_crisp, internal, call = set(), False, False, False
        call_id = parent_call_id = None
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            stamp = event.get('observed_at')
            if stamp:
                t = datetime.datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()
                first, last = (t if first is None else min(first, t)), (t if last is None else max(last, t))
            kind = event.get('kind')
            busy = (t - float(event.get('duration_ms') or 0) / 1000, t) if stamp and event.get('duration_ms') else None
            if kind == 'manifest':
                call_id = event.get('call_id') or event.get('run_id')
                parent_call_id = event.get('parent_call_id')
                internal, call = bool(parent_call_id), bool(event.get('definition_id'))
            elif kind == 'model_request' and event.get('phase') == 'end':
                side = 'frontier' if is_frontier(event.get('actor')) else 'local'
                actors.add(side)
                bucket = acc[side]
                bucket['prompt'] += int(event.get('prompt_tokens') or 0)
                bucket['completion'] += int(event.get('completion_tokens') or 0)
                bucket['cached'] += int(event.get('cached_tokens') or 0)
                bucket['requests'] += 1
                bucket['ms'] += float(event.get('duration_ms') or 0)
                if side == 'local' and busy:
                    acc['local_intervals'].append(busy)
            elif kind == 'decision_readout' and event.get('phase') not in ('unavailable', 'start', None):
                scored = event.get('scored') not in (False, 'False')
                acc['decisions_scored' if scored else 'decisions_generated'] += 1
                if scored:   # a scored readout makes no model_request: its time and tokens are in the readout event
                    side = 'frontier' if is_frontier(event.get('actor')) else 'local'
                    actors.add(side)
                    acc[side]['requests'] += 1
                    acc[side]['ms'] += float(event.get('duration_ms') or 0)
                    if side == 'local' and busy:
                        acc['local_intervals'].append(busy)
                    acc[side]['prompt'] += int(event.get('prompt_tokens') or 0)
                    acc[side]['completion'] += int(event.get('completion_tokens') or event.get('reasoning_tokens') or 0)
            elif kind == 'tier_served' and event.get('tier') == 'tier3':
                served_crisp = True
        traces.append({'call_id': call_id, 'parent_call_id': parent_call_id, 'internal': internal,
                       'call': call, 'actors': actors, 'served_crisp': served_crisp})
    by_id = {t['call_id']: t for t in traces if t['call_id']}
    for trace in traces:
        if not trace['parent_call_id'] or not trace['actors']:
            continue
        seen = set()
        pid = trace['parent_call_id']
        while pid and pid not in seen:
            seen.add(pid)
            parent = by_id.get(pid)
            if parent is None:
                break
            parent['actors'].update(trace['actors'])
            pid = parent['parent_call_id']
    for trace in traces:
        if trace['internal'] or not trace['call']:
            continue   # a call made inside another call is part of its parent's work; iteration traces are no calls
        acc['calls'] += 1
        acc['crisp_served'] += trace['served_crisp'] and not trace['actors']
        acc['model_calls'] += bool(trace['actors'])
        acc['frontier_calls'] += 'frontier' in trace['actors']
    if first is not None:
        acc['span_ms'] = round((last - first) * 1000)
    return acc


# ---- metrics ---------------------------------------------------------------------------------------------------------

def busy_seconds(intervals):
    """Wall time covered by at least one of ``intervals`` ((start, end) seconds): how long the local server was busy."""
    total, end = 0., None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def bootstrap_ci(per_unit, resamples=10000, seed=0):
    """95% unit-bootstrap interval of the mean pass rate ({unit: [successes, n]})."""
    units = list(per_unit)
    if not units:
        return None
    rng = random.Random(seed)
    rates = {u: s / n for u, (s, n) in per_unit.items()}
    boot = sorted(sum(rates[rng.choice(units)] for _ in units) / len(units) for _ in range(resamples))
    return [round(boot[int(.025 * resamples)], 4), round(boot[int(.975 * resamples) - 1], 4)]


def usd_frontier(tokens, model=FRONTIER['model']):
    price = LIST_PRICE[model]
    cache_write = tokens.get('cache_write', 0)
    uncached = max(0, tokens['prompt'] - tokens['cached'])
    return (uncached * price['input'] + tokens['cached'] * price['cache_read']
            + cache_write * price['cache_write'] + tokens['completion'] * price['output']) / 1e6


def summarize(app, arm, records, local_rate):
    verdicts, per_unit, latencies, safety = [], {}, [], {}
    totals = {'frontier': {'prompt': 0, 'completion': 0, 'cached': 0, 'cache_write': 0},
              'local': {'prompt': 0, 'completion': 0, 'cached': 0}}
    gpu_ms, calls, crisp, model_calls, frontier_calls, errors = 0., 0, 0, 0, 0, 0
    intervals = []
    rows = {r['id']: r for r in app['rows']}
    for record in records:
        verdict = app['verify'](rows[record['id']]['expected'], record.get('output'))
        if record.get('error'):
            errors += 1
        verdicts.append({'id': record['id'], 'sample': record['sample'], 'passed': verdict['passed'],
                         'checks': verdict['checks'], 'error': record.get('error'), 'output': record.get('output')})
        entry = per_unit.setdefault(record['id'], [0, 0])
        entry[0] += bool(verdict['passed'])
        entry[1] += 1
        for key, value in verdict.get('safety', {}).items():
            safety[key] = safety.get(key, 0) + value
        if arm == 'A':
            usage = record['usage']
            totals['frontier']['prompt'] += usage['input'] + usage['cache_read']
            totals['frontier']['cached'] += usage['cache_read']
            totals['frontier']['cache_write'] += usage.get('cache_write', 0)
            totals['frontier']['completion'] += usage['output']
            latencies.append(record['latency_ms'])
            calls += 1
            model_calls += 1
            frontier_calls += 1
        else:
            acc = record['accounting']
            for side in ('frontier', 'local'):
                for key in ('prompt', 'completion', 'cached'):
                    totals[side][key] += acc[side][key]
            totals['frontier']['cache_write'] += acc['frontier'].get('cache_write', 0)
            gpu_ms += acc['local']['ms']
            intervals += acc.get('local_intervals', [])
            latencies.append(acc['span_ms'] if acc['span_ms'] is not None else record['process_wall_s'] * 1000)
            calls += acc['calls']
            crisp += acc['crisp_served']
            model_calls += acc['model_calls']
            frontier_calls += acc['frontier_calls']
    n = len(records)
    if not n:
        return {'units': 0}, {}
    frontier_usd = usd_frontier(totals['frontier'])
    # GPU time: how long the local server was busy with this arm's requests (the union of their intervals), so parallel
    # reasoning samples and concurrent units are not counted twice; the summed request time is reported beside it.
    gpu_s = busy_seconds(intervals)
    local_usd = gpu_s / 3600 * local_rate
    per100 = 100 / n
    checks = {}
    for v in verdicts:
        for key, value in v['checks'].items():
            if isinstance(value, bool):
                checks.setdefault(key, []).append(value)
    has_services = bool(app['spec'].get('services'))
    has_untrusted = bool(app['spec']['untrusted']) if 'untrusted' in app['spec'] else has_services
    safety_summary = {
        'schema_violation': 0,
        'unverified_commits': 'n/a (no commits in this app)' if not has_services else None,
        'injection_escapes': 'n/a (no untrusted boundary in this app)' if not has_untrusted else None,
        **safety,
    }
    return {
        'units': len(per_unit), 'runs': n, 'errors': errors,
        'pass_rate': round(sum(s / c for s, c in per_unit.values()) / len(per_unit), 4),
        'pass_rate_ci95_unit_bootstrap': bootstrap_ci(per_unit),
        'check_rates': {k: round(sum(v) / len(v), 4) for k, v in checks.items()},
        'latency_ms': {'p50': percentile(latencies, .5), 'p95': percentile(latencies, .95)},
        'tokens_per_100_units': {side: {k: round(v * per100) for k, v in t.items()} for side, t in totals.items()},
        'usd_per_100_units': {'frontier_list_price': round(frontier_usd * per100, 5), 'local_gpu': round(local_usd * per100, 5),
                              'total': round((frontier_usd + local_usd) * per100, 5)},
        'local_gpu_seconds_per_unit': round(gpu_s / n, 3),
        'local_request_seconds_per_unit': round(gpu_ms / 1000 / n, 3),
        'nl_calls': calls,
        'crisp_served_share': round(crisp / calls, 4) if calls else None,
        'local_resolved_share': round((calls - frontier_calls) / calls, 4) if calls else None,
        'escalation_share': (round(frontier_calls / calls, 4) if calls else None) if arm.startswith('D') else None,
        'safety': safety_summary,
        'per_unit': verdicts,
    }, per_unit


def compare_to(reference, other, resamples=10000):
    """``other`` against ``reference`` ({unit: [successes, n]}): held_eval's mean per-unit rate change, within-unit
    permutation p and unit-bootstrap CI, the sign test over units, and ``beats`` (WS6 §3: higher, p < 0.05)."""
    tasks = sorted(set(reference) & set(other))
    change = held_eval._rate_change(reference, other, tasks, resamples=resamples)
    only_ref = sum(reference[t][0] / reference[t][1] > other[t][0] / other[t][1] for t in tasks)
    only_other = sum(other[t][0] / other[t][1] > reference[t][0] / reference[t][1] for t in tasks)
    return {**change, 'units_only_reference_better': only_ref, 'units_only_this_better': only_other,
            'unit_sign_p': round(held_eval.binomial_two_sided(only_ref, only_other), 4),
            'beats': bool(change['delta'] is not None and change['delta'] > 0 and change['permutation_p'] < 0.05)}


def per_unit_of(metrics):
    """{unit: [successes, n]} from an arm's ``per_unit`` verdicts in a result file."""
    out = {}
    for verdict in metrics['per_unit']:
        entry = out.setdefault(verdict['id'], [0, 0])
        entry[0] += bool(verdict['passed'])
        entry[1] += 1
    return out


# ---- commands --------------------------------------------------------------------------------------------------------

def cmd_run(args):
    app, work = load_app(args.app), Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    rows = split_rows(app, 'held')[:args.limit] if args.limit else split_rows(app, 'held')
    for arm in args.arms.split(','):
        if arm not in ARMS:
            raise SystemExit(f'unknown arm {arm}; arms: {", ".join(ARMS)}')
        started = time.time()
        if arm == 'A':
            records = run_frontier(app, work, rows, args.samples, concurrency=args.concurrency)
        else:
            prepare_app(app, work, arm)
            with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
                records = list(pool.map(lambda job: run_program_unit(app, work, arm, *job, timeout=args.unit_timeout),
                                        [(row, s) for row in rows for s in range(args.samples)]))
        passed = sum(bool(app['verify'](next(r for r in app['rows'] if r['id'] == rec['id'])['expected'], rec.get('output'))['passed'])
                     for rec in records)
        print(f'{arm}: {passed}/{len(records)} passed, {sum(bool(r.get("error")) for r in records)} errors, '
              f'{time.time() - started:.0f}s', flush=True)


def cmd_prepare_d(args):
    """Arm D's compiled state, from the train split only: its examples become the program copy's examples.json;
    each runs once on the policy (``record_examples``: the calls land in D's call store), ``natlang specialize`` compiles the
    recorded calls into guarded crisp cases, and ``natlang tune`` trains the program's weights on the same examples
    (teacher: the policy profile's ``escalate``, i.e. arm B's model)."""
    app, work = load_app(args.app), Path(args.work).resolve()
    rows = split_rows(app, 'train')
    if args.limit:   # a seeded sample of the train split (fixtures are ordered by label: a prefix would be one class)
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    env = program_env(work, 'D-reasoned')
    copy = prepare_app(app, work, 'D')
    (copy / 'examples.json').write_text(json.dumps(app_module_examples(app, rows), indent=1) + '\n')
    log = work / 'prepare-d.log'
    steps = {
        'specialize': [*NATLANG, 'specialize', '--profile', 'bench-frontier', '--program', str(copy)],
        'tune': [*NATLANG, 'tune', str(copy)],
    }
    failed = None
    with log.open('a') as out:
        for step in args.steps.split(','):
            started = time.time()
            if step == 'record':
                out.write(f'== record: {record_examples(copy, env, args.concurrency)} ({time.time() - started:.0f}s)\n')
                out.flush()
                continue
            if step not in steps:
                raise SystemExit(f'unknown prepare-d step {step!r}')
            proc = subprocess.run(steps[step], env=env, capture_output=True, text=True)
            out.write(f'== {step} exit {proc.returncode} ({time.time() - started:.0f}s): {" ".join(steps[step][1:])}\n'
                      f'{proc.stdout[-6000:]}\n{proc.stderr[-6000:]}\n')
            out.flush()
            if proc.returncode:
                failed = (step, proc.returncode)
                break
    print(log.read_text()[-6000:])
    if failed is not None:
        raise SystemExit(f'prepare-d step {failed[0]} failed with exit {failed[1]}')


def _example_key(example):
    ident = example.get('id') or example.get('name')
    if ident:
        return safe_id(ident)
    digest = hashlib.sha256(json.dumps(example, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    return f'ex-{digest}'


def record_examples(copy, env, concurrency):
    """Run each of the program copy's examples once on the policy (``natlang call FN --inputs``, ``concurrency`` at a
    time: ``natlang test`` starts them all at once, which the reference server cannot hold), so arm D's call store holds
    the train split's calls for ``natlang specialize``. Returns {examples, failed calls, wrong answers}."""
    examples = json.loads((copy / 'examples.json').read_text())
    inputs_dir = copy.parent / 'record-inputs'
    inputs_dir.mkdir(exist_ok=True)

    def one(example):
        key = _example_key(example)
        path, done = inputs_dir / f'{key}.json', inputs_dir / f'{key}.outcome'
        if done.exists():   # resumable: an example recorded by an earlier launch is not called again
            return done.read_text()
        path.write_text(json.dumps(example['inputs']))
        proc = subprocess.run([*NATLANG, 'call', str(copy / example['function']), '--inputs', str(path), '--json',
                               '--profile', 'bench-policy'], env=env, capture_output=True, text=True, cwd=copy)
        try:
            outcome = 'failed' if proc.returncode else 'right' if json.loads(proc.stdout) == example['expected'] else 'wrong'
        except json.JSONDecodeError:
            outcome = 'failed'
        if outcome != 'failed':
            done.write_text(outcome)
        return outcome
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        outcomes = list(pool.map(one, examples))
    return {'examples': len(examples), **{k: outcomes.count(k) for k in ('right', 'wrong', 'failed')}}


def app_module_examples(app, rows):
    """Root examples for ``natlang tune`` from train rows: the app's ``bench/tune_examples.py`` maps a row to
    ``natlang test`` entries ({name, function, inputs, expected, group})."""
    path = app['dir'] / 'bench' / 'tune_examples.py'
    loader = importlib.util.spec_from_file_location('bench_tune_examples', path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return [entry for row in rows for entry in module.examples(row, app['defaults'])]


def cmd_report(args):
    app, work = load_app(args.app), Path(args.work).resolve()
    rows = split_rows(app, 'held')
    ids = {r['id'] for r in rows}
    arms, per_units = {}, {}
    for arm in ARMS:
        if arm == 'A':
            records = [json.loads(p.read_text()) for p in sorted((work / 'runs' / 'A').glob('*.json'))] if (work / 'runs' / 'A').exists() else []
        else:
            records = []
            for path in sorted((work / 'runs' / arm).glob('*/s*/result.json')):
                record = json.loads(path.read_text())
                record['accounting'] = trace_accounting(path.parent / 'traces')   # the current accounting rules
                records.append(record)
        infra = sum(infra_error(r.get('error')) for r in records if r['id'] in ids)
        records = [r for r in records if r['id'] in ids and not infra_error(r.get('error'))]
        if not records:
            continue
        metrics, per_unit = summarize(app, arm, records, args.local_usd_per_gpu_hour)
        metrics['infra_errors_excluded'] = infra
        arms[arm] = metrics
        per_units[arm] = per_unit
    for arm in arms:
        if arm != 'A' and 'A' in per_units:
            arms[arm]['vs_A'] = compare_to(per_units['A'], per_units[arm])
    for arm in arms:
        if arm.startswith('D') and 'C' + arm[1:] in per_units:
            arms[arm]['vs_C'] = compare_to(per_units['C' + arm[1:]], per_units[arm])
    for arm in arms:   # cost of this arm per unit of arm A's
        if arm != 'A' and 'A' in arms and arms['A']['usd_per_100_units']['total']:
            arms[arm]['cost_ratio_vs_A'] = round(arms[arm]['usd_per_100_units']['total'] / arms['A']['usd_per_100_units']['total'], 3)
    config_path = work / 'config' / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {'profiles': {}}
    profiles = config.get('profiles') or {}
    result = {
        'schema': 'natlang.bench-result/1', 'app': app['spec']['app'], 'date': datetime.date.today().isoformat(),
        'code': code_provenance.code_provenance() if hasattr(code_provenance, 'code_provenance') else None,
        'split': 'held', 'held_units': len(rows),
        'arms_config': {
            'A': {**FRONTIER, 'loop': 'frontier_loop.mjs: system + prose task, tools = app services + submit'},
            'B': {'profile': profiles.get('bench-frontier'), 'specialization': 'off', 'escalate': None,
                  'decisions': 'generated (a Pi provider cannot score replies)'},
            'C': {'profile': redact(profiles.get('bench-base')), 'specialization': 'off', 'escalate': None},
            'D': {'profile': redact(profiles.get('bench-policy')), 'specialization': 'on (state from prepare-d)',
                  'escalate': 'bench-frontier', 'prepare': (work / 'prepare-d.log').read_text()[-3000:] if (work / 'prepare-d.log').exists() else None},
            'modes': {'reasoned': 'readout: decision reasons first (default decisionSamples)', 'instant': 'NATLANG_DECISION_SAMPLES=0'},
        },
        'pricing': {'frontier_usd_per_million_tokens': LIST_PRICE, 'frontier_price_source': 'pi-ai catalog openai.json (list)',
                    'local_usd_per_gpu_hour': args.local_usd_per_gpu_hour,
                    'local_gpu_seconds': 'wall time the local server was busy with the arm (union of its request intervals), '
                                         'on the shared DGX GB10 and the Python native engine'},
        'arms': arms,
        'reading': args.reading,
    }
    out = Path(args.out) if args.out else app['dir'] / 'bench' / 'results' / f'{result["date"]}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    for arm, m in arms.items():
        vs = m.get('vs_A') or {}
        print(f"{arm:11s} pass {m['pass_rate']:.3f} {m['pass_rate_ci95_unit_bootstrap']}  vs A {vs.get('delta')} "
              f"{vs.get('ci95_task_bootstrap')} p={vs.get('permutation_p')}  p50 {m['latency_ms']['p50']}ms "
              f"p95 {m['latency_ms']['p95']}ms  $/100 {m['usd_per_100_units']['total']}  crisp {m['crisp_served_share']} "
              f"esc {m['escalation_share']} errors {m['errors']}")
    print(out)


def cmd_split(args):
    app = Path(args.app).resolve().name if Path(args.app).exists() else args.app
    data = json.loads(Path(args.fixtures).read_text())
    rows = data if isinstance(data, list) else data.get('rows') or data.get('items') or []
    labeled = label_fixtures(app, rows)
    held = sum(row['split'] == 'held' for row in labeled)
    text = json.dumps({'app': app, 'total': len(labeled), 'held': held, 'train': len(labeled) - held,
                       'split_rule': FLAGSHIP_SHARE_RULE, 'rows': labeled}, indent=1) + '\n'
    Path(args.out).write_text(text) if args.out else print(text, end='')


def cmd_compare(args):
    """Arm ``--arm`` of the second result file (default: the first) against ``--against`` of the first."""
    first = json.loads(Path(args.result).read_text())
    second = json.loads(Path(args.other).read_text()) if args.other else first
    if first.get('app') != second.get('app'):
        raise SystemExit(f"cannot compare different apps: {first.get('app')!r} vs {second.get('app')!r}")
    ref = per_unit_of(first['arms'][args.against])
    other = per_unit_of(second['arms'][args.arm])
    if not (set(ref) & set(other)):
        raise SystemExit(f'no overlapping units between {args.against} and {args.arm}')
    print(json.dumps(compare_to(ref, other, resamples=args.resamples), indent=1))


def _split_delta(baseline, proposed, split_name):
    b = (baseline or {}).get(split_name) or {}
    p = (proposed or {}).get(split_name) or {}
    b_rate, p_rate = b.get('passRate'), p.get('passRate')
    return {
        'before': {'passed': b.get('passed'), 'total': b.get('total'), 'pass_rate': b_rate},
        'after': {'passed': p.get('passed'), 'total': p.get('total'), 'pass_rate': p_rate},
        'delta': round(p_rate - b_rate, 4) if isinstance(b_rate, (int, float)) and isinstance(p_rate, (int, float)) else None,
    }


def cmd_improve(args):
    """Invoke ``natlang improve`` on the app copy under ``--work`` and report confirmation-split before/after deltas."""
    app, work = load_app(args.app), Path(args.work).resolve()
    arm = args.arm
    copy = prepare_app(app, work, arm)
    cmd = [*NATLANG, 'improve', str(copy), '--model', args.model, '--json']
    if args.proposal:
        cmd += ['--proposal', str(Path(args.proposal).resolve())]
    if args.provider:
        cmd += ['--provider', args.provider]
    if args.server:
        cmd += ['--server', args.server]
    if args.dry_run:
        cmd += ['--dry-run']
    store_dir = work / 'calls' / arm.split('-')[0]
    if store_dir.exists():
        cmd += ['--store', str(store_dir)]
    env_arm = 'D-reasoned' if arm == 'D' else arm
    proc = subprocess.run(cmd, env=program_env(work, env_arm), capture_output=True, text=True)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f'natlang improve failed (exit {proc.returncode}): {proc.stderr[-2000:] or proc.stdout[-2000:]}') from error
    report = {
        'schema': 'natlang.bench-improve/1',
        'app': app['spec']['app'],
        'directory': str(copy),
        'status': payload.get('status'),
        'changed': payload.get('changed', []),
        'reasons': payload.get('reasons', []),
        'confirmation': _split_delta(payload.get('baseline'), payload.get('proposed'), 'confirmation'),
        'dev': _split_delta(payload.get('baseline'), payload.get('proposed'), 'dev'),
        'record': payload.get('record'),
        'raw': payload,
    }
    text = json.dumps(report, indent=1) + '\n'
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end='')
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return report


def redact(profile):
    if not profile:
        return profile
    native = profile.get('native') or {}
    return {**{k: v for k, v in profile.items() if k != 'native'},
            **({'native': {k: native.get(k) for k in ('sha256', 'source', 'step', 'checkpoint')}} if native else {})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('app')
    run.add_argument('--work', required=True)
    run.add_argument('--arms', default=','.join(ARMS))
    run.add_argument('--samples', type=int, default=1)
    run.add_argument('--concurrency', type=int, default=4)
    run.add_argument('--unit-timeout', type=float, default=1800)
    run.add_argument('--limit', type=int)
    setup = sub.add_parser('setup', help='write WORK/config with the bench profiles')
    setup.add_argument('app')
    setup.add_argument('--work', required=True)
    setup.add_argument('--base-profile', help="a user config profile (or a profile JSON file) serving base LFM3")
    setup.add_argument('--policy-profile', help='a user config profile (or a profile JSON file) serving the current policy; '
                       'or run `natlang setup --native SOURCE --profile bench-policy` with NATLANG_CONFIG_HOME=WORK/config, '
                       'then setup again')
    prep = sub.add_parser('prepare-d')
    prep.add_argument('app')
    prep.add_argument('--work', required=True)
    prep.add_argument('--steps', default='record,specialize,tune')
    prep.add_argument('--concurrency', type=int, default=4)
    prep.add_argument('--limit', type=int, help='train rows to use, a seeded sample (the reference engine on a shared '
                      'GPU decodes ~15 tokens/s: the whole train split takes hours)')
    report = sub.add_parser('report')
    report.add_argument('app')
    report.add_argument('--work', required=True)
    report.add_argument('--out')
    report.add_argument('--local-usd-per-gpu-hour', type=float, default=DEFAULT_LOCAL_USD_PER_GPU_HOUR)
    report.add_argument('--reading', help='the honest reading of the result, recorded in the file')
    split = sub.add_parser('split', help='label a fixture pack: canonical split_group and held-v4 split per row')
    split.add_argument('app')
    split.add_argument('--fixtures', required=True)
    split.add_argument('--out')
    compare = sub.add_parser('compare', help='paired comparison of two arms from result files')
    compare.add_argument('result')
    compare.add_argument('other', nargs='?')
    compare.add_argument('--arm', required=True)
    compare.add_argument('--against', default='A')
    compare.add_argument('--resamples', type=int, default=10000)
    imp = sub.add_parser('improve', help='run natlang improve on the app copy and report confirmation-split deltas')
    imp.add_argument('app')
    imp.add_argument('--work', required=True)
    imp.add_argument('--model', required=True)
    imp.add_argument('--proposal', help='optional proposal JSON file to pass to natlang improve')
    imp.add_argument('--provider', help='Pi provider id')
    imp.add_argument('--server', help='OpenAI-compatible server URL')
    imp.add_argument('--arm', default='D', help='which app copy/call-store to improve (default: D)')
    imp.add_argument('--dry-run', action='store_true')
    imp.add_argument('--out', help='write JSON report to this path')
    args = parser.parse_args(argv)
    if args.command == 'split':
        cmd_split(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'setup':
        setup_config(Path(args.work).resolve(), args.policy_profile, args.base_profile)
        print(Path(args.work).resolve() / 'config' / 'config.json')
    else:
        {'run': cmd_run, 'prepare-d': cmd_prepare_d, 'report': cmd_report, 'improve': cmd_improve}[args.command](args)


if __name__ == '__main__':
    main()
