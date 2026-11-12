"""A fixed held-out task set measured on a served policy (owner request 2026-11-10: dev16 cannot resolve our updates).

dev16 is 16 single-sample cases (about +-1.7 cases of noise). A held set pins pool tasks stratified over the pool's
families and levels (``build_manifest``; ``pin`` writes a standalone one such as ops/held-v2 with its ``exclusion-ids.json``),
and training code never selects them or their ``split_group`` siblings (``exclusion_ids``, ``curriculum.select_tasks``
``exclude_ids``). ``measure`` runs N samples per held task against a served actor (``serve`` launches one for a checkpoint)
and writes ``result.json``; ``compare``/``paired`` pair two results on matched (task, sample) episodes (McNemar exact).

- ``manifest.json`` + ``cases.jsonl``: task ids and row digests, the pool's digest, and the excluded split groups.
- ``begin``/``Handle.finish`` and ``run``: a detached driver process measures one policy; failures and a cut-short run are
  recorded, and a partial run is still a balanced one (sample-major order) and pairs on what it finished.
- A comparison series is the collector runtime (ts-host tree) plus the resolved ``NATLANG_MODEL_FACING`` variants, recorded
  in every result: only results of one series are compared, a change of either starts a new series with its own baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..common.jsonio import read_jsonl, write_json_durable

MANIFEST_SCHEMA = 'natlang.rl-held-eval-manifest/1'
RESULT_SCHEMA = 'natlang.rl-held-eval-result/1'
DEFAULT_TASKS, DEFAULT_SAMPLES, DEFAULT_WORKERS, DEFAULT_GRACE = 64, 2, 16, 1200
EFFECT_STATE_SHARE = 1 / 3   # the selector's class_quota is .34: the held mixture mirrors what training practises
MAX_GROUP_ROWS = 24          # a held task whose split group is larger would shrink the training pool by that much
SEED = 'held-eval-20261110'
SEED_BASE = 4200000
LOOP = ('budget_exhaustion', 'repetition')


def settings(config):
    """None when opted out, else the option object with defaults."""
    value = config.get('held_eval')
    if value is False:
        return None
    options = dict(value) if isinstance(value, dict) else {}
    return {'tasks': DEFAULT_TASKS, 'samples': DEFAULT_SAMPLES, 'workers': DEFAULT_WORKERS, 'grace_seconds': DEFAULT_GRACE,
            **options}


def directory_of(out):
    return Path(out) / 'held-eval'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash(*parts, seed=SEED):
    return hashlib.sha256('\x00'.join(str(p) for p in (seed, *parts)).encode()).hexdigest()


def _split_group(row):
    return ((row.get('curriculum') or {}).get('split_group')) or row.get('id')


# ---- the manifest ----------------------------------------------------------------------------------------------------
def avoid_ids(state, directories=()):
    """Task ids the lineage already attempted or trained on (its state and the sealed selections of ``directories``)."""
    ids = set()
    for key in ('attempted_task_ids', 'collected_task_ids', 'trained_task_ids'):
        ids |= set(state.get(key) or ())
    ids |= set((state.get('task_curriculum_history') or {}))
    ids |= {g['task_id'] for g in state.get('group_stats') or () if g.get('task_id')}
    for directory in directories:
        for path in Path(directory).glob('round-*-selection.json'):
            ids |= {case['id'] for case in json.loads(path.read_text()).get('cases') or ()}
    return ids


def build_manifest(cases_path, *, avoid, tasks=DEFAULT_TASKS, include=(), seed=SEED):
    """(manifest, held rows): ``tasks`` pool rows, family- and level-stratified, of which a third carry the effect-state class.
    Deterministic in the pool and ``avoid``. Rows whose split group holds an avoided task are not candidates. ``include``:
    task ids held already (an earlier, smaller manifest) that the new one keeps; the rest is chosen around them."""
    from .curriculum import metadata
    cases = list(read_jsonl(cases_path))
    include = set(include)
    if include & set(avoid):
        raise ValueError('an included held task is also to be avoided: ' + ', '.join(sorted(include & set(avoid))[:5]))
    avoid_groups = {_split_group(row) for row in cases if row.get('id') in avoid}
    groups = {}
    for row in cases:
        groups[_split_group(row)] = groups.get(_split_group(row), 0) + 1
    candidates = []
    for index, row in enumerate(cases):
        meta = metadata(row)
        if (row.get('split') == 'train' and row['id'] not in avoid and _split_group(row) not in avoid_groups
                and groups[_split_group(row)] <= MAX_GROUP_ROWS):
            candidates.append({'row': index, 'id': row['id'], 'group': _split_group(row), **meta})
    levels = sorted({c['level'] for c in candidates})
    chosen = [c for c in candidates if c['id'] in include]
    if len(chosen) != len(include):
        raise ValueError(f'{len(include) - len(chosen)} included held tasks are not eligible pool rows')
    used_groups = {c['group'] for c in chosen}
    family_count = {}
    for c in chosen:
        family_count[c['family']] = family_count.get(c['family'], 0) + 1
    effect = max(0, round(tasks * EFFECT_STATE_SHARE) - sum(1 for c in chosen if c.get('mix_class') == 'effect_state'))

    def take(pool, quota):
        by_family = {}
        for c in pool:
            by_family.setdefault(c['family'], []).append(c)
        order = sorted(by_family, key=lambda f: _hash('family', f, seed=seed))
        picked = []
        for cap in range(1, len(pool) + 2):   # a family's next task only when every family has as many
            for family in order:
                if len(picked) >= quota:
                    break
                if family_count.get(family, 0) >= cap:
                    continue
                # Balanced levels over the picks of this class: the least-picked level this family has candidates for.
                have = {lv: sum(1 for p in chosen + picked if p['level'] == lv) for lv in levels}
                options = [c for c in by_family[family] if c['group'] not in used_groups]
                if not options:
                    continue
                target = min({c['level'] for c in options}, key=lambda lv: (have[lv], _hash('level', family, lv, seed=seed)))
                pick = min((c for c in options if c['level'] == target), key=lambda c: _hash('row', c['id'], seed=seed))
                picked.append(pick)
                used_groups.add(pick['group'])
                family_count[family] = family_count.get(family, 0) + 1
            if len(picked) >= quota:
                break
        return picked

    chosen += take([c for c in candidates if c.get('mix_class') == 'effect_state' and c['id'] not in include], effect)
    chosen += take([c for c in candidates if c.get('mix_class') != 'effect_state' and c['id'] not in include],
                   tasks - len(chosen))
    if len(chosen) < tasks:
        raise ValueError(f'the pool has only {len(chosen)} eligible held tasks (wanted {tasks})')
    chosen.sort(key=lambda c: c['row'])
    rows = [cases[c['row']] for c in chosen]
    manifest = {
        'schema': MANIFEST_SCHEMA, 'seed': seed, 'included': len(include), 'tasks': [
            {'id': c['id'], 'pool_row': c['row'], 'family': c['family'], 'level': c['level'],
             **({'mix_class': c['mix_class']} if c.get('mix_class') else {}), 'split_group': c['group'],
             'row_sha256': hashlib.sha256(json.dumps(cases[c['row']], sort_keys=True).encode()).hexdigest()}
            for c in chosen],
        'excluded_split_groups': sorted({c['group'] for c in chosen}),
        'pool': {'path': str(cases_path), 'sha256': _sha256(cases_path), 'rows': len(cases)},
        'avoided_task_ids': len(avoid), 'avoided_split_groups': len(avoid_groups),
        'by_level': {str(lv): sum(1 for c in chosen if c['level'] == lv) for lv in levels},
        'families': len({c['family'] for c in chosen}),
        'effect_state': sum(1 for c in chosen if c.get('mix_class') == 'effect_state')}
    return manifest, rows


def write_manifest(directory, manifest, rows):
    """Write ``cases.jsonl`` then ``manifest.json`` (never overwrites a pinned manifest); returns the manifest's sha256."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists():
        raise FileExistsError('a held-eval manifest is pinned already: ' + str(directory / 'manifest.json'))
    cases = ''.join(json.dumps(row) + '\n' for row in rows).encode('utf-8')
    (directory / 'cases.jsonl').write_bytes(cases)
    manifest = {**manifest, 'cases': {'path': 'cases.jsonl', 'sha256': hashlib.sha256(cases).hexdigest(), 'rows': len(rows)}}
    data = (json.dumps(manifest, indent=1) + '\n').encode('utf-8')
    (directory / 'manifest.json').write_bytes(data)
    return hashlib.sha256(data).hexdigest()


_MANIFESTS = {}


def load_manifest(path, sha256):
    path = Path(path)
    key = (str(path), sha256)
    if key not in _MANIFESTS:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError('the held-eval manifest changed after it was pinned')
        manifest = json.loads(data)
        if manifest.get('schema') != MANIFEST_SCHEMA:
            raise ValueError('unsupported held-eval manifest')
        _MANIFESTS[key] = manifest
    return _MANIFESTS[key]


# ---- selector exclusion and its transition ------------------------------------------------------------------------------
def exclusion_ids(manifest, cases=None):
    """Ids of pool rows training must never take: the held tasks of ``manifest`` (a path or the loaded manifest) and every
    row sharing a held ``split_group``. ``cases``: the pool rows (or a path); by default the manifest's pool, whose sha256
    must match the pinned one. Experiments call this to exclude a held set (e.g. ``ops/held-v2/manifest.json``)."""
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    if cases is None:
        pool = manifest['pool']
        if _sha256(pool['path']) != pool['sha256']:
            raise ValueError('the pool changed since the held manifest was pinned: ' + pool['path'])
        cases = pool['path']
    if isinstance(cases, (str, Path)):
        cases = read_jsonl(cases)
    held, groups = {t['id'] for t in manifest['tasks']}, set(manifest['excluded_split_groups'])
    return frozenset(row['id'] for row in cases if row.get('id') in held or _split_group(row) in groups)


# ---- series identity --------------------------------------------------------------------------------------------------
def model_facing(config=None):
    """The variants the collector resolves: a pinned option, the environment's, else the runtime's adopted default."""
    pinned = (settings(config or {}) or {}).get('model_facing')
    raw = pinned if pinned is not None else os.environ.get('NATLANG_MODEL_FACING', '')
    raw = raw.strip()
    if raw:
        return raw
    source = Path(__file__).resolve().parents[4] / 'ts-host/src/runtime/model-facing.ts'
    match = re.search(r'const ADOPTED[^=]*=\s*\[([^\]]*)\]', source.read_text()) if source.exists() else None
    return ','.join(re.findall(r"'([^']+)'", match.group(1))) if match else 'unknown'


def series(config=None):
    from ..common.code_provenance import _git, repository_root
    root = repository_root()
    tree = _git(root, 'rev-parse', 'HEAD:ts-host')
    dirty = bool(_git(root, 'status', '--porcelain', '--untracked-files=no', '--', 'ts-host'))
    variants = ','.join(sorted(model_facing(config).split(',')))
    key = hashlib.sha256(f'{tree}|{variants}'.encode()).hexdigest()[:12]
    return {'key': key, 'ts_host_tree': tree, 'ts_host_dirty': dirty, 'model_facing': variants}


# ---- the measurement ----------------------------------------------------------------------------------------------------
def policy_directory(out, policy_id):
    return directory_of(out) / f'policy-{policy_id[:16]}'


def _entries(spec):
    """The sealed, sample-major queue (a partly finished run stays balanced) bound to the held cases and the policy."""
    from types import SimpleNamespace
    from . import task_rollouts
    directory = Path(spec['dir'])
    task_rollouts.prepare(SimpleNamespace(
        cases=Path(spec['cases']), out=directory, rows=spec['tasks'], offset=0, group=spec['samples'], temperature=1.,
        case_seconds=spec['case_seconds'], max_model_requests=spec['max_model_requests'], max_turns=spec['max_turns'],
        seed_base=spec.get('seed_base', SEED_BASE), root_mode='student', match_seeds_from=None,
        policy_checkpoint_sha256=spec['policy_id']))
    entries = list(read_jsonl(directory / 'queue.jsonl'))
    for entry in entries:
        entry.update(prompt_rendering='full', context_tokens=spec['context_tokens'])
    entries.sort(key=lambda entry: (entry['sample'], entry['index']))
    return entries


def _finished(journal):
    return journal.exists() and any(row.get('event') == 'finish' for row in read_jsonl(journal))


def collect(directory, entries, endpoints, *, repo, runtime, env, stop, log=None):
    """Work-stealing slots (``endpoints``: [(url, slots)]) over the sealed ``entries``, one ``run_bonsai_queue`` process per
    episode under ``directory/workers`` (a slow episode never holds a shard of others back). Entries with a finished journal
    are skipped (resume). Returns once every entry ran or ``stop`` is set (its running episodes are terminated)."""
    directory, repo = Path(directory), Path(repo)
    work = directory / 'workers'
    work.mkdir(exist_ok=True)
    todo = [entry for entry in entries if not _finished(work / f"{entry['key']}.journal.jsonl")]
    lock, cursor, children, done = threading.Lock(), [0], set(), [0]
    started = time.time()

    def serving(url):
        from .actor_client import NativeActorClient
        for _ in range(6):
            try:
                NativeActorClient(url, timeout=10).get_bytes('/v1/neuralese/info')
                return True
            except Exception:
                time.sleep(10)
        return False

    def slot(url):
        while not stop.is_set():
            if not serving(url):   # a lost actor would turn the rest of the queue into infrastructure failures
                if log:
                    log(json.dumps({'event': 'held eval actor lost', 'endpoint': url}))
                stop.set()
                return
            with lock:
                if cursor[0] >= len(todo):
                    return
                entry = todo[cursor[0]]
                cursor[0] += 1
            key = entry['key']
            queue, journal = work / f'{key}.queue.jsonl', work / f'{key}.journal.jsonl'
            journal.unlink(missing_ok=True)   # an unfinished attempt restarts whole
            queue.write_text(json.dumps(entry, sort_keys=True) + '\n')
            command = [sys.executable, str(repo / 'scripts/run_bonsai_queue.py'), str(queue), str(journal),
                       '--runtime', str(runtime), '--server', url, '--chat-completions-url', url + '/v1/chat/completions',
                       '--model-id', 'natlang-rl', '--model-concurrency', '1', '--chat-request-config',
                       str(directory / 'sampling-config.json'), '--neuralese-server']
            with lock:
                if stop.is_set():
                    return
                with (work / f'{key}.log').open('a') as handle:
                    child = subprocess.Popen(command, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT)
                children.add(child)
            child.wait()
            with lock:
                children.discard(child)
                done[0] += 1
                if log and done[0] % 16 == 0:
                    log(json.dumps({'held_eval_progress': done[0], 'of': len(todo),
                                    'minutes': round((time.time() - started) / 60, 1)}))

    def watch():
        stop.wait()
        with lock:
            for child in list(children):
                if child.poll() is None:
                    child.terminate()

    threading.Thread(target=watch, daemon=True).start()
    threads = [threading.Thread(target=slot, args=(url.rstrip('/'),), daemon=True)
               for url, slots in endpoints for _ in range(max(1, min(slots, len(todo))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        while thread.is_alive():
            thread.join(5)


def run(directory):
    """The detached driver: work-stealing collection of the sealed queue (``workers`` slots), score what finished and write
    ``result.json``. SIGTERM stops the collectors and still writes the (partial) result."""
    directory = Path(directory)
    spec = json.loads((directory / 'run.json').read_text())
    started = time.time()
    queue = directory / 'sealed-queue.jsonl'
    if not queue.exists():
        entries = _entries(spec)
        queue.write_text(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in entries))
    entries = list(read_jsonl(queue))
    (directory / 'sampling-config.json').write_text(json.dumps({'guidance': False}) + '\n')
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    endpoints = spec.get('endpoints') or [[spec['endpoint'], spec['workers']]]
    env = {**os.environ, 'NATLANG_MODEL_FACING': spec['series']['model_facing']}
    collect(directory, entries, endpoints, repo=spec['repo'], runtime=spec['runtime'], env=env, stop=stop,
            log=lambda line: print(line, flush=True))
    try:
        result = finalize(directory, spec, started, complete=not stop.is_set())
    except BaseException as error:   # never leave a run without a verdict
        write_json_durable(directory / 'result.json', {'schema': RESULT_SCHEMA, 'policy_id': spec['policy_id'],
                           'error': repr(error)[:500], 'series': spec['series']})
        raise
    print(json.dumps({'held_eval': spec['policy_id'][:16], 'successes': result['successes'], 'scored': result['scored'],
                      'timeouts': result['timeouts'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    return result


def _timed_out(finish):
    return bool(finish) and (finish.get('status') == 'timeout' or finish.get('resource_limit_reason') == 'case_wall_budget')


def finalize(directory, spec, started, *, complete):
    """Score the finished episodes into ``result.json``. A wall-clock timeout (``case_wall_budget``: load-dependent, not
    the policy's) is unscored and counted apart (``timeouts``), never as a failure; ``success_rate`` is over the scored."""
    from . import task_rollouts
    directory = Path(directory)
    journal = directory / 'journal.jsonl'
    with journal.open('w') as merged:
        for path in sorted([*directory.glob('shard-*/journal.jsonl'), *directory.glob('workers/*.journal.jsonl')]):
            merged.write(path.read_text())
    report = task_rollouts.score(directory)
    rewards = list(read_jsonl(directory / 'rewards.jsonl'))
    queue = {entry['key']: entry for entry in read_jsonl(directory / 'sealed-queue.jsonl')}
    finishes = {row['key']: row for row in read_jsonl(journal) if row.get('event') == 'finish'}
    by_position = {(entry['group'], entry['sample']): key for key, entry in queue.items()}
    packet = list(read_jsonl(directory / 'packet.jsonl'))
    samples = []
    for row in rewards:
        key = by_position[(row['group'], row['sample'])]
        finish = finishes.get(key)
        efficiency = row.get('execution_efficiency') or {}
        samples.append({'task_id': row['task_id'], 'family': row.get('family'), 'level': _level(packet[row['group']]),
                        'sample': row['sample'], 'outcome': row['outcome'], 'reward': row['reward'],
                        'ran': key in finishes, 'finish_status': row.get('finish_status'), 'timed_out': _timed_out(finish),
                        'elapsed_seconds': (finish or {}).get('elapsed_seconds'),
                        'completion_tokens': (finish or {}).get('completion_tokens'),
                        'model_calls': efficiency.get('model_calls'), 'repeated_actions': efficiency.get('repeated_actions'),
                        'neuralese_exchanges': row.get('neuralese_exchanges', 0)})
    ran = [s for s in samples if s['ran']]
    scored = [s for s in ran if s['reward'] is not None]
    counts = {}
    for s in ran:
        counts[s['outcome']] = counts.get(s['outcome'], 0) + 1
    calls = sum(s['model_calls'] or 0 for s in scored)
    successes = sum(s['reward'] == 1.0 for s in scored)
    timeouts = sum(s['timed_out'] for s in ran)
    tokens = sum(s['completion_tokens'] or 0 for s in ran)
    elapsed = round(time.time() - started, 1)
    result = {'schema': RESULT_SCHEMA, 'policy_id': spec['policy_id'], 'round': spec.get('round'),
              'manifest_sha256': spec['manifest_sha256'], 'series': spec['series'], 'code': spec['code'],
              'endpoint': spec['endpoint'], 'actor': spec.get('actor'), 'tasks': spec['tasks'], 'samples_per_task': spec['samples'],
              'seed_base': spec.get('seed_base', SEED_BASE), 'case_seconds': spec.get('case_seconds'),
              'started_at': started, 'finished_at': time.time(), 'elapsed_seconds': elapsed,
              'complete': bool(complete and len(ran) == len(samples)), 'ran': len(ran), 'scored': len(scored),
              'successes': successes, 'success_rate': round(successes / len(scored), 4) if scored else None,
              'timeouts': timeouts, 'timeout_rate': round(timeouts / len(ran), 4) if ran else None,
              'unscored_other': len(ran) - len(scored) - sum(s['timed_out'] and s['reward'] is None for s in ran),
              'outcome_counts': counts,
              'loop_rate': (sum(counts.get(k, 0) for k in LOOP) / len(ran)) if ran else None,
              'repeated_action_rate': (sum(s['repeated_actions'] or 0 for s in scored) / calls) if calls else None,
              'neuralese_use_rate': (sum(s['neuralese_exchanges'] > 0 for s in scored) / len(scored)) if scored else None,
              'completion_tokens': tokens, 'tokens_per_second': round(tokens / elapsed, 1) if elapsed else None,
              'score': {'groups_with_spread': report.get('groups_with_spread')}, 'samples': samples}
    write_json_durable(directory / 'result.json', result)
    return result


def _level(task):
    return (((task.get('curriculum') or {}).get('rl')) or {}).get('level')


def _alive(record):
    from .detached_processes import alive
    return alive(record)


class Handle:
    """The detached measurement of one policy (started here or adopted after a controller restart)."""

    def __init__(self, directory, record, grace):
        self.directory, self.record, self.grace = Path(directory), record, grace

    def finish(self, detaching=False, immediate=False):
        """Give an unfinished measurement ``grace`` seconds (none when the collection failed or a stop was requested), then
        stop it (it still writes its partial result); with ``detaching`` (restart in place) leave it running for the
        relaunched controller. Never raises."""
        try:
            if detaching or not _alive(self.record):
                return
            deadline = time.monotonic() + (0 if immediate else self.grace)
            while _alive(self.record) and time.monotonic() < deadline:
                time.sleep(2)
            if _alive(self.record):
                os.kill(self.record['pid'], signal.SIGTERM)
                end = time.monotonic() + 300
                while _alive(self.record) and time.monotonic() < end:
                    time.sleep(2)
                print(json.dumps({'event': 'held eval cut short', 'directory': str(self.directory),
                                  'grace_seconds': self.grace}), flush=True)
        except BaseException as error:
            print(json.dumps({'event': 'held eval finish failed', 'error': repr(error)[:300]}), flush=True)


def begin(config, *, out, policy_id, endpoint, round_number=None, log=print):
    """Start the measurement of ``policy_id`` served at ``endpoint`` (None when opted out, unadopted, already measured, or
    failed to start: the failure is logged and the round goes on). A measurement left running by a restarted controller is
    adopted."""
    try:
        options = settings(config)
        binding = config.get('held_eval_binding')
        if options is None or binding is None:
            return None
        directory = policy_directory(out, policy_id)
        if (directory / 'result.json').exists() and json.loads((directory / 'result.json').read_text()).get('complete'):
            return None   # a cut-short or failed run resumes from its journals (this policy's round is still running)
        from .detached_processes import load_record, record_process
        record = load_record(directory / 'driver.json')
        if record is not None and _alive(record):
            return Handle(directory, record, options['grace_seconds'])
        from .actor_client import NativeActorClient
        info = json.loads(NativeActorClient(endpoint, timeout=5).get_bytes('/v1/neuralese/info'))
        actor = info.get('rl_policy_binding') or {}
        if actor.get('checkpoint_sha256') != policy_id:
            raise ValueError('the actor does not serve the policy to measure')
        manifest = load_manifest(binding['manifest'], binding['sha256'])
        from ..common.code_provenance import code_provenance, repository_root
        repo = repository_root()
        directory.mkdir(parents=True, exist_ok=True)
        spec = {'policy_id': policy_id, 'round': round_number, 'endpoint': endpoint, 'dir': str(directory),
                'cases': str(Path(binding['manifest']).parent / manifest['cases']['path']),
                'manifest_sha256': binding['sha256'], 'tasks': len(manifest['tasks']), 'samples': options['samples'],
                'workers': options['workers'], 'case_seconds': config.get('case_seconds', 600),
                'max_model_requests': config.get('max_model_requests', 128), 'max_turns': config.get('max_turns', 160),
                'context_tokens': config.get('context_tokens') or 32768,
                'runtime': str(config.get('runtime') or repo / 'ts-host'), 'repo': str(repo),
                'series': series(config), 'code': code_provenance(),
                'actor': {'export': actor.get('actor_export'), 'code': actor.get('code')}}
        (directory / 'run.json').write_text(json.dumps(spec, indent=1) + '\n')
        (directory / 'result.json').unlink(missing_ok=True)
        with (directory / 'driver.log').open('a') as handle:
            process = subprocess.Popen([sys.executable, '-m', 'natlang_neuralese.rl.held_eval', 'run', '--dir', str(directory)],
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                       cwd=str(repo), env=os.environ.copy())
        record = record_process(directory / 'driver.json', process.pid, policy_id=policy_id, endpoint=endpoint)
        log(json.dumps({'event': 'held eval started', 'policy': policy_id[:16], 'tasks': spec['tasks'],
                        'samples': spec['samples'], 'workers': spec['workers'], 'series': spec['series']['key']}))
        return Handle(directory, record, options['grace_seconds'])
    except BaseException as error:   # a measurement never blocks the round
        log(json.dumps({'event': 'held eval did not start', 'error': repr(error)[:400]}))
        return None


# ---- comparison -------------------------------------------------------------------------------------------------------
def results(out):
    """The valid held-eval results of a directory, by start time."""
    found = []
    for path in directory_of(out).glob('policy-*/result.json'):
        result = json.loads(path.read_text())
        if result.get('schema') == RESULT_SCHEMA and 'error' not in result:
            found.append(result)
    return sorted(found, key=lambda r: r['started_at'])


def binomial_two_sided(a, b):
    """Exact two-sided sign-test p of ``a`` against ``b`` discordant outcomes (p = 0.5 each)."""
    n = a + b
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def paired(first, second):
    """``second`` against ``first`` over the matched (task, sample) episodes both scored: McNemar's exact test on the
    discordant samples, and the exact sign test over tasks whose mean success differs (samples of a task are not independent,
    so the task-level p is the conservative one)."""
    def table(result):
        return {(s['task_id'], s['sample']): s['reward'] == 1.0 for s in result['samples']
                if s['ran'] and s['reward'] is not None}
    a, b = table(first), table(second)
    common = sorted(set(a) & set(b))
    only_first = sum(a[k] and not b[k] for k in common)
    only_second = sum(b[k] and not a[k] for k in common)
    tasks = {}
    for task, sample in common:
        tasks.setdefault(task, []).append((a[(task, sample)], b[(task, sample)]))
    up = sum(sum(y for _, y in v) > sum(x for x, _ in v) for v in tasks.values())
    down = sum(sum(y for _, y in v) < sum(x for x, _ in v) for v in tasks.values())
    return {'pairs': len(common), 'tasks': len(tasks), 'first_successes': sum(a[k] for k in common),
            'second_successes': sum(b[k] for k in common), 'only_first': only_first, 'only_second': only_second,
            'mcnemar_exact_p': round(binomial_two_sided(only_first, only_second), 4),
            'tasks_up': up, 'tasks_down': down, 'task_sign_p': round(binomial_two_sided(up, down), 4)}


def compare(first, second, *, resamples=2000):
    """The standalone comparison of two result.json files (``second`` against ``first``): the paired test (McNemar on
    matched scored samples, task-level sign test), the task-clustered bootstrap 95% interval of the success-rate change
    over the matched samples, each side's timeouts and loop rate, and whether the two ran under one code commit and series.
    Pairs lost to a timeout on either side are counted, never scored."""
    import random

    def timed_out(sample):
        return bool(sample.get('timed_out') or sample.get('finish_status') == 'timeout')

    def table(result):
        return {(s['task_id'], s['sample']): s for s in result['samples'] if s['ran']}
    a, b = table(first), table(second)
    common = set(a) & set(b)
    scored = [k for k in common if a[k]['reward'] is not None and b[k]['reward'] is not None]
    by_task = {}
    for task, sample in scored:
        by_task.setdefault(task, []).append((a[(task, sample)]['reward'] == 1.0, b[(task, sample)]['reward'] == 1.0))
    tasks = sorted(by_task)
    rng = random.Random(0)
    deltas = []
    for _ in range(resamples if tasks else 0):
        pick = [by_task[rng.choice(tasks)] for _ in tasks]
        n = sum(len(v) for v in pick)
        deltas.append(sum(y - x for v in pick for x, y in v) / n)
    deltas.sort()

    def side(result):
        return {'policy': result['policy_id'], 'code': (result.get('code') or {}).get('git_commit'),
                'series': result['series']['key'], 'seed_base': result.get('seed_base', SEED_BASE), 'ran': result['ran'],
                'scored': result['scored'], 'successes': result['successes'],
                'success_rate': round(result['successes'] / result['scored'], 4) if result['scored'] else None,
                'timeouts': sum(timed_out(s) for s in result['samples'] if s['ran']),
                'loop_rate': result['loop_rate'], 'repeated_action_rate': result.get('repeated_action_rate'),
                'elapsed_seconds': result.get('elapsed_seconds'), 'complete': result.get('complete')}
    n = sum(len(v) for v in by_task.values())
    return {'first': side(first), 'second': side(second), 'same_code': side(first)['code'] == side(second)['code'],
            'same_series': side(first)['series'] == side(second)['series'],
            'same_manifest': first.get('manifest_sha256') == second.get('manifest_sha256'),
            'matched_pairs': len(common), 'pairs_lost_to_timeout': sum(
                1 for k in common if timed_out(a[k]) or timed_out(b[k])),
            'paired': paired(first, second),
            'success_rate_delta': round(sum(y - x for v in by_task.values() for x, y in v) / n, 4) if n else None,
            'delta_ci95_task_bootstrap': [round(deltas[int(.025 * len(deltas))], 4),
                                          round(deltas[int(.975 * len(deltas)) - 1], 4)] if deltas else None,
            'loop_rate_delta': (round(second['loop_rate'] - first['loop_rate'], 4)
                                if first['loop_rate'] is not None and second['loop_rate'] is not None else None)}


def trend(out):
    """Per measured policy, by start time: its numbers, the paired comparison with the previous result of its series and with
    the first of its series (None for the first one or when the series changed)."""
    rows, firsts, previous = [], {}, {}
    for result in results(out):
        key = result['series']['key']
        row = {'policy': result['policy_id'], 'round': result.get('round'), 'series': key,
               'complete': result['complete'], 'ran': result['ran'], 'scored': result['scored'],
               'successes': result['successes'], 'loop_rate': result['loop_rate'],
               'repeated_action_rate': result['repeated_action_rate'], 'outcome_counts': result['outcome_counts'],
               'vs_previous': paired(previous[key], result) if key in previous else None,
               'vs_first': paired(firsts[key], result) if key in firsts and firsts[key] is not previous.get(key) else None,
               'series_detail': result['series']}
        firsts.setdefault(key, result)
        previous[key] = result
        rows.append(row)
    return rows


# ---- standalone measurement (experiments) ----------------------------------------------------------------------------
BASE_SNAPSHOT = Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
# The actor's throughput settings are serve_native_checkpoint.py's defaults (concurrency, KV pool, FP8 KV).
ACTOR_ARGS = ('--engine', 'sglang', '--sglang-projection', 'vendor')


def _free_port():
    import socket
    for port in range(8400, 8900, 2):   # the SGLang server takes port + 1000
        with socket.socket() as a, socket.socket() as b:
            try:
                a.bind(('127.0.0.1', port))
                b.bind(('127.0.0.1', port + 1000))
                return port
            except OSError:
                continue
    raise RuntimeError('no free actor port')


def serve(checkpoint, sha256, store, *, unit, budget_gb, extra=(), base=BASE_SNAPSHOT, wait_seconds=3600,
          admission_seconds=14400, log=print):
    """Serve ``checkpoint`` as a ledger unit (SGLang, vendor projection) and wait until it attests ``sha256``; returns
    the endpoint URL (``store/endpoint``). An active ``unit`` that already serves it is reused. The unit is stopped with
    ``systemctl --user stop UNIT``."""
    from .actor_client import NativeActorClient
    from ..common.code_provenance import repository_root
    repo = repository_root()
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    active = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip()
    if active in ('active', 'activating') and (store / 'endpoint').exists():
        url = (store / 'endpoint').read_text().strip()
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            try:
                info = json.loads(NativeActorClient(url, timeout=10).get_bytes('/v1/neuralese/info'))
                if (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') == sha256:
                    return url
            except Exception:
                pass
            time.sleep(10)
        raise TimeoutError(f'the running actor {unit} does not attest {sha256[:12]}')
    port = _free_port()
    python = repo / '.venv-neuralese/bin/python'
    command = ['/usr/bin/python3', str(repo / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb', str(budget_gb),
               '--class', 'experiment', '--priority', 'medium', '--no-resumable', '--wait', str(admission_seconds),
               '--workdir', str(repo), '--',
               'env', f'NATLANG_REPO={repo}', f'PYTHONPATH={repo}/training/neuralese:{repo}', 'HF_HUB_OFFLINE=1',
               'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', str(python), str(repo / 'scripts/neuralese_data/serve_native_checkpoint.py'),
               '--store-dir', str(store), '--checkpoint', str(checkpoint), '--checkpoint-sha256', sha256, '--base', str(base),
               '--port', str(port), '--prompt-rendering', 'full', *ACTOR_ARGS, '--hf-export-dir', str(store / 'sglang-hf'), *extra]
    url = f'http://127.0.0.1:{port}'
    (store / 'endpoint').write_text(url + '\n')
    subprocess.run(command, check=True)
    log(json.dumps({'event': 'held eval actor launched', 'unit': unit, 'endpoint': url}))
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            info = json.loads(NativeActorClient(url, timeout=10).get_bytes('/v1/neuralese/info'))
            if (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') == sha256:
                return url
        except Exception:
            pass
        state = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip()
        if state in ('failed', 'inactive'):
            raise RuntimeError(f'actor unit {unit} is {state}')
        time.sleep(10)
    raise TimeoutError(f'actor {unit} did not attest {sha256[:12]}')


def measure(out, manifest, endpoints, policy_id, *, samples=4, seed_base=SEED_BASE, case_seconds=3600,
            max_model_requests=64, max_turns=128, context_tokens=32768, label=None, log=print):
    """Measure the policy served at ``endpoints`` ([(url, slots)]) on every task of ``manifest`` x ``samples`` into
    ``out/result.json`` (resumable: an existing ``run.json`` must name the same manifest, policy and sampling)."""
    from .actor_client import NativeActorClient
    from ..common.code_provenance import code_provenance, repository_root
    out, manifest = Path(out), Path(manifest)
    for url, _ in endpoints:
        info = json.loads(NativeActorClient(url, timeout=10).get_bytes('/v1/neuralese/info'))
        binding = info.get('rl_policy_binding') or {}
        if binding.get('checkpoint_sha256') != policy_id:
            raise ValueError(f'{url} serves {str(binding.get("checkpoint_sha256"))[:12]}, not {policy_id[:12]}')
    pinned = json.loads(manifest.read_text())
    repo = repository_root()
    spec = {'policy_id': policy_id, 'label': label, 'endpoint': endpoints[0][0], 'endpoints': [list(e) for e in endpoints],
            'dir': str(out.resolve()), 'manifest': str(manifest.resolve()),
            'cases': str(manifest.resolve().parent / pinned['cases']['path']), 'manifest_sha256': _sha256(manifest),
            'tasks': len(pinned['tasks']), 'samples': samples, 'seed_base': seed_base, 'workers': sum(n for _, n in endpoints),
            'case_seconds': case_seconds, 'max_model_requests': max_model_requests, 'max_turns': max_turns,
            'context_tokens': context_tokens, 'runtime': str(repo / 'ts-host'), 'repo': str(repo),
            'series': series(), 'code': code_provenance(), 'actor': {'info': info.get('rl_policy_binding')}}
    if spec['code'].get('dirty') or spec['series']['ts_host_dirty']:
        log(json.dumps({'event': 'held eval warning', 'warning': 'the checkout is dirty; the result records it'}))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run.json').exists():
        previous = json.loads((out / 'run.json').read_text())
        for key in ('policy_id', 'manifest_sha256', 'samples', 'seed_base', 'case_seconds', 'max_model_requests', 'max_turns'):
            if previous[key] != spec[key]:
                raise ValueError(f'{out} holds a run with another {key}: {previous[key]!r}')
        spec = {**previous, 'endpoint': spec['endpoint'], 'endpoints': spec['endpoints'], 'workers': spec['workers'],
                'resumed_code': spec['code']}
    (out / 'run.json').write_text(json.dumps(spec, indent=1) + '\n')
    (out / 'result.json').unlink(missing_ok=True)
    return run(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='the detached driver of one policy directory (written by begin)')
    r.add_argument('--dir', type=Path, required=True)
    b = sub.add_parser('build', help='build and pin the manifest of a controller directory')
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--tasks', type=int, default=DEFAULT_TASKS)
    b.add_argument('--avoid-from', type=Path, action='append', default=[], help='directories with round-*-selection.json of other lineages')
    b.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('pin', help='build and pin a standalone held manifest (e.g. ops/held-v2) with its exclusion-ids.json')
    p.add_argument('--cases', type=Path, required=True, help='the task pool (cases.jsonl)')
    p.add_argument('--dir', type=Path, required=True, help='directory for manifest.json, cases.jsonl, exclusion-ids.json')
    p.add_argument('--tasks', type=int, required=True)
    p.add_argument('--seed', default=SEED)
    p.add_argument('--include-manifest', type=Path, help='an earlier held manifest whose tasks the new one keeps')
    p.add_argument('--avoid-ids', type=Path, action='append', default=[],
                   help='JSON list of task ids (with their split groups) the held set must not contain (e.g. a training selection)')
    p.add_argument('--avoid-from', type=Path, action='append', default=[],
                   help='controller directories whose state and round-*-selection.json tasks to avoid')
    p.add_argument('--dry-run', action='store_true')
    m = sub.add_parser('measure', help='standalone: N samples per held task of a served actor or a checkpoint -> result.json')
    m.add_argument('--manifest', type=Path, required=True, help='e.g. ops/held-v2/manifest.json')
    m.add_argument('--out', type=Path, required=True, help='run directory (resumable)')
    m.add_argument('--endpoint', action='append', default=[], help='served actor URL[=SLOTS] (repeatable)')
    m.add_argument('--checkpoint', type=Path, help='serve this checkpoint as a ledger unit instead (stopped afterwards)')
    m.add_argument('--checkpoint-sha256', help='its sha256 (default: hashed)')
    m.add_argument('--policy-id', help='with --endpoint: the sha256 the actor must serve (default: what it attests)')
    m.add_argument('--samples', type=int, default=4)
    m.add_argument('--workers', type=int, default=64, help='concurrent episodes per endpoint without =SLOTS')
    m.add_argument('--seed-base', type=int, default=SEED_BASE)
    m.add_argument('--case-seconds', type=int, default=3600)
    m.add_argument('--max-model-requests', type=int, default=64)
    m.add_argument('--max-turns', type=int, default=128)
    m.add_argument('--context-tokens', type=int, default=32768)
    m.add_argument('--label')
    m.add_argument('--actor-unit', help='ledger unit name of the actor (default heldv2-actor-<sha8>)')
    m.add_argument('--actor-budget-gb', type=int, default=40)
    m.add_argument('--actor-arg', action='append', default=[], help='extra serve_native_checkpoint.py argument (repeatable)')
    m.add_argument('--keep-actor', action='store_true')
    v = sub.add_parser('serve', help='serve a checkpoint as a ledger actor unit (or reuse it) and print its URL')
    v.add_argument('--checkpoint', type=Path, required=True)
    v.add_argument('--checkpoint-sha256')
    v.add_argument('--store', type=Path, required=True)
    v.add_argument('--unit', required=True)
    v.add_argument('--budget-gb', type=int, default=40)
    v.add_argument('--actor-arg', action='append', default=[])
    c = sub.add_parser('compare', help='paired comparison of two result.json files (second against first)')
    c.add_argument('first', type=Path)
    c.add_argument('second', type=Path)
    c.add_argument('--out', type=Path)
    s = sub.add_parser('start', help='measure the policy an actor serves, now (the controller hook does this at each round)')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--endpoint', required=True)
    s.add_argument('--policy-id', required=True)
    s.add_argument('--wait', action='store_true', help='stay until the driver exits')
    args = parser.parse_args(argv)
    if args.command == 'run':
        run(args.dir)
    elif args.command == 'build':
        config = json.loads((args.out / 'config.json').read_text())
        state = json.loads((args.out / 'state.json').read_text())
        manifest, rows = build_manifest(config['cases'], avoid=avoid_ids(state, [args.out, *args.avoid_from]), tasks=args.tasks)
        if args.dry_run:
            print(json.dumps({k: v for k, v in manifest.items() if k not in ('tasks', 'excluded_split_groups')}, indent=1))
            for task in manifest['tasks']:
                print(task['family'], task['level'], task.get('mix_class', '-'), task['id'])
            return
        print(write_manifest(directory_of(args.out), manifest, rows))
    elif args.command == 'compare':
        report = compare(json.loads(args.first.read_text()), json.loads(args.second.read_text()))
        text = json.dumps(report, indent=1)
        if args.out:
            args.out.write_text(text + '\n')
        print(text)
    elif args.command == 'serve':
        sha = args.checkpoint_sha256 or _sha256(args.checkpoint)
        print(serve(args.checkpoint, sha, args.store, unit=args.unit, budget_gb=args.budget_gb, extra=args.actor_arg,
                    log=lambda line: print(line, file=sys.stderr, flush=True)))
    elif args.command == 'measure':
        endpoints = [(item.rsplit('=', 1)[0].rstrip('/'), int(item.rsplit('=', 1)[1])) if re.search(r'=\d+$', item)
                     else (item.rstrip('/'), args.workers) for item in args.endpoint]
        unit = None
        if args.checkpoint:
            sha = args.checkpoint_sha256 or _sha256(args.checkpoint)
            unit = args.actor_unit or f'heldv2-actor-{sha[:8]}'
            url = serve(args.checkpoint, sha, args.out / 'actor', unit=unit, budget_gb=args.actor_budget_gb,
                        extra=args.actor_arg)
            endpoints.append((url, args.workers))
            policy = sha
        else:
            from .actor_client import NativeActorClient
            policy = args.policy_id or (json.loads(NativeActorClient(endpoints[0][0], timeout=10).get_bytes(
                '/v1/neuralese/info')).get('rl_policy_binding') or {}).get('checkpoint_sha256')
        if not endpoints or not policy:
            raise SystemExit('give --endpoint or --checkpoint')
        try:
            measure(args.out, args.manifest, endpoints, policy, samples=args.samples, seed_base=args.seed_base,
                    case_seconds=args.case_seconds, max_model_requests=args.max_model_requests, max_turns=args.max_turns,
                    context_tokens=args.context_tokens, label=args.label)
        finally:
            if unit and not args.keep_actor:
                subprocess.run(['systemctl', '--user', 'stop', unit])
    elif args.command == 'pin':
        avoid = set()
        for path in args.avoid_ids:
            avoid |= set(json.loads(path.read_text()))
        for directory in args.avoid_from:
            state = json.loads((directory / 'state.json').read_text()) if (directory / 'state.json').exists() else {}
            avoid |= avoid_ids(state, [directory])
        include = [t['id'] for t in json.loads(args.include_manifest.read_text())['tasks']] if args.include_manifest else ()
        manifest, rows = build_manifest(args.cases, avoid=avoid, tasks=args.tasks, include=include, seed=args.seed)
        if args.include_manifest:
            manifest['included_manifest'] = {'path': str(args.include_manifest), 'sha256': _sha256(args.include_manifest)}
        summary = {k: v for k, v in manifest.items() if k not in ('tasks', 'excluded_split_groups')}
        if args.dry_run:
            print(json.dumps(summary, indent=1))
            return
        sha = write_manifest(args.dir, manifest, rows)
        ids = sorted(exclusion_ids(args.dir / 'manifest.json'))
        (args.dir / 'exclusion-ids.json').write_text(json.dumps(ids, indent=0) + '\n')
        print(json.dumps({'manifest_sha256': sha, 'exclusion_ids': len(ids), **summary}, indent=1))
    else:
        config = json.loads((args.out / 'config.json').read_text())
        state = json.loads((args.out / 'state.json').read_text())   # read only: the controller owns its state
        manifest = Path(settings(config).get('manifest') or directory_of(args.out) / 'manifest.json')
        config = {**config, 'held_eval_binding': {'manifest': str(manifest), 'sha256': _sha256(manifest), 'from_round': 0}}
        handle = begin(config, out=args.out, policy_id=args.policy_id, endpoint=args.endpoint, round_number=state.get('round'))
        if handle is None:
            raise SystemExit('nothing started')
        if args.wait:
            while _alive(handle.record):
                time.sleep(5)


if __name__ == '__main__':
    main()
