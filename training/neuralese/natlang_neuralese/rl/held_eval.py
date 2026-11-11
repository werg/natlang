"""A fixed held-out task set measured on every new policy (owner request 2026-11-10: dev16 cannot resolve our updates).

dev16 is 16 single-sample cases (about +-1.7 cases of noise). The RL controller instead pins about 64 tasks of its own task
pool, stratified over the pool's families and levels, excludes them (and their ``split_group`` siblings) from RL selection
for good, and runs them on the DGX actor of every newly updated policy: 2 samples per task, the production sampling, a
few hours of shared slots at most. The held tasks never enter a round, a learner update or a distillation package, so
their success is an unbiased read of the policy; ``scripts/rl_tasks/behavior_trend.py`` pairs each policy with the previous
one and with the first measured one on the matched (task, sample) episodes.

- ``manifest.json`` + ``cases.jsonl`` (``{out}/held-eval/``): task ids and row digests, the pool's digest, and the excluded
  split groups. Built once from the pool and the lineage's already attempted tasks (``build_manifest``); its sha256 is
  recorded in the controller state by the transition ``held-eval-exclusion-20261110`` (``adopt_defaults``). The selector
  applies it from the first round whose selection is not yet sealed (``exclusion``, ``rl.production.select_tasks``).
- ``begin``/``Handle.finish``: ``collect_with_actor`` starts the measurement of the round's policy beside the round's
  collection, against the same actor, and gives an unfinished measurement ``grace_seconds`` once the collection is done.
  The measurement is a detached driver process (``run``): failures and a cut-short run are recorded and never stop or delay the round
  beyond the grace; a partial run is still a balanced one (sample-major order) and pairs on what it finished.
- A comparison series is the collector runtime (ts-host tree) plus the resolved ``NATLANG_MODEL_FACING`` variants, recorded
  in every result: only results of one series are compared, a change of either starts a new series with its own baseline.

``held_eval: false`` opts out (selection exclusion and measurement); ``held_eval`` may also be an object with ``tasks``,
``samples``, ``workers``, ``grace_seconds``, ``model_facing`` (pinned variants) and ``manifest`` (path) options.
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
TRANSITION = 'held-eval-exclusion-20261110'
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


def _hash(*parts):
    return hashlib.sha256('\x00'.join(str(p) for p in (SEED, *parts)).encode()).hexdigest()


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


def build_manifest(cases_path, *, avoid, tasks=DEFAULT_TASKS):
    """(manifest, held rows): ``tasks`` pool rows, family- and level-stratified, of which a third carry the effect-state class.
    Deterministic in the pool and ``avoid``. Rows whose split group holds an avoided task are not candidates."""
    from .curriculum import metadata
    cases = list(read_jsonl(cases_path))
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
    effect = round(tasks * EFFECT_STATE_SHARE)
    chosen, used_groups, family_count = [], set(), {}

    def take(pool, quota):
        by_family = {}
        for c in pool:
            by_family.setdefault(c['family'], []).append(c)
        order = sorted(by_family, key=lambda f: _hash('family', f))
        picked = []
        for cap in (1, 2, 3):   # a second and third task of a family only when the families are exhausted
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
                target = min({c['level'] for c in options}, key=lambda lv: (have[lv], _hash('level', family, lv)))
                pick = min((c for c in options if c['level'] == target), key=lambda c: _hash('row', c['id']))
                picked.append(pick)
                used_groups.add(pick['group'])
                family_count[family] = family_count.get(family, 0) + 1
            if len(picked) >= quota:
                break
        return picked

    chosen += take([c for c in candidates if c.get('mix_class') == 'effect_state'], effect)
    chosen += take([c for c in candidates if c.get('mix_class') != 'effect_state'], tasks - len(chosen))
    if len(chosen) < tasks:
        raise ValueError(f'the pool has only {len(chosen)} eligible held tasks (wanted {tasks})')
    chosen.sort(key=lambda c: c['row'])
    rows = [cases[c['row']] for c in chosen]
    manifest = {
        'schema': MANIFEST_SCHEMA, 'seed': SEED, 'tasks': [
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
def adopt_defaults(config, state, state_path, *, out, sealed_selection=False):
    """Controllers adopt the held set through a recorded transition. The manifest is built from the pool and the lineage's
    attempted tasks on the first adoption and pinned (its sha256 recorded); the exclusion applies from the first round whose
    selection is not sealed yet. ``held_eval: false`` opts out. Returns the config with ``held_eval_binding`` set."""
    options = settings(config)
    if options is None or config.get('coefficient_policy'):
        return config
    transitions = state.setdefault('held_eval_transitions', [])
    record = next((item for item in transitions if item.get('transition') == TRANSITION), None)
    if record is None:
        directory = Path(options.get('manifest') or directory_of(out) / 'manifest.json').parent
        if not (directory / 'manifest.json').exists():
            try:
                manifest, rows = build_manifest(config['cases'], avoid=avoid_ids(state, [out]), tasks=options['tasks'])
            except ValueError as error:   # a pool too small to hold tasks out keeps training on all of it; retried at a restart
                print(json.dumps({'event': 'held eval not adopted', 'reason': str(error)[:200]}), flush=True)
                return config
            write_manifest(directory, manifest, rows)
        number = state.get('round', 0)
        sealed_next = (Path(out) / f'round-{number + 1:06d}-selection.json').exists()
        record = {'schema': 'natlang.rl-held-eval-transition/1', 'transition': TRANSITION, 'recorded_at_round': number,
                  'from_round': number + (1 if sealed_selection or state.get('active_update_round') else 0) + int(sealed_next),
                  'manifest': str(directory / 'manifest.json'), 'manifest_sha256': _sha256(directory / 'manifest.json')}
        transitions.append(record)
        write_json_durable(state_path, state)
    return {**config, 'held_eval_binding': {'manifest': record['manifest'], 'sha256': record['manifest_sha256'],
                                            'from_round': record['from_round']}}


def exclusion(config, cases, round_number):
    """Ids of pool rows RL selection must not take in ``round_number``: the held tasks and their split-group siblings."""
    binding = config.get('held_eval_binding')
    if not binding or settings(config) is None or round_number < binding['from_round']:
        return frozenset()
    manifest = load_manifest(binding['manifest'], binding['sha256'])
    held, groups = {t['id'] for t in manifest['tasks']}, set(manifest['excluded_split_groups'])
    return frozenset(row['id'] for row in cases if row.get('id') in held or _split_group(row) in groups)


def static_relation(out):
    """A predicate over prepared SFT rows (``source_groups``): True for a row related to a held task of the controller
    ``out`` (its pinned manifest's split groups and its cases' source groups, also matched without the source prefix, so
    ``2wiki-q:X`` meets ``2wiki:X``). None when the controller holds no tasks out. Static sources queued into the
    lineage's SFT windows use it: held tasks never enter training."""
    directory = directory_of(out)
    if not (directory / 'manifest.json').exists():
        return None
    groups = set(json.loads((directory / 'manifest.json').read_text())['excluded_split_groups'])
    for row in read_jsonl(directory / 'cases.jsonl'):
        groups.update(row.get('source_groups') or ())
        groups.add(_split_group(row))
    bare = {group.split(':', 1)[1] for group in groups if isinstance(group, str) and ':' in group}

    def related(row):
        mine = [group for group in row.get('source_groups') or () if isinstance(group, str)]
        return any(group in groups or (':' in group and group.split(':', 1)[1] in bare) for group in mine)
    return related


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
        seed_base=SEED_BASE, root_mode='student', match_seeds_from=None, policy_checkpoint_sha256=spec['policy_id']))
    entries = list(read_jsonl(directory / 'queue.jsonl'))
    for entry in entries:
        entry.update(prompt_rendering='full', context_tokens=spec['context_tokens'])
    entries.sort(key=lambda entry: (entry['sample'], entry['index']))
    return entries


def run(directory):
    """The detached driver: shard the sealed queue over ``workers`` collector processes (``run_bonsai_queue``), score what
    finished and write ``result.json``. SIGTERM stops the collectors and still writes the (partial) result."""
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
    children, lock = [], threading.Lock()

    def terminate(*_):
        stop.set()
        with lock:
            for child in children:
                if child.poll() is None:
                    child.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    workers = max(1, min(spec['workers'], len(entries)))
    root = Path(spec['repo'])
    env = {**os.environ, 'NATLANG_MODEL_FACING': spec['series']['model_facing']}

    def shard(index):
        shard_dir = directory / f'shard-{index:02d}'
        shard_dir.mkdir(exist_ok=True)
        mine = entries[index::workers]
        (shard_dir / 'queue.jsonl').write_text(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in mine))
        endpoint = spec['endpoint'].rstrip('/')
        command = [sys.executable, str(root / 'scripts/run_bonsai_queue.py'), str(shard_dir / 'queue.jsonl'),
                   str(shard_dir / 'journal.jsonl'), '--runtime', spec['runtime'], '--server', endpoint,
                   '--chat-completions-url', endpoint + '/v1/chat/completions', '--model-id', 'natlang-rl',
                   '--model-concurrency', '1', '--chat-request-config', str(directory / 'sampling-config.json'),
                   '--neuralese-server']
        with lock:
            if stop.is_set():
                return
            with (shard_dir / 'collector.log').open('a') as log:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
        child.wait()

    threads = [threading.Thread(target=shard, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        result = finalize(directory, spec, started, complete=not stop.is_set())
    except BaseException as error:   # never leave a run without a verdict
        write_json_durable(directory / 'result.json', {'schema': RESULT_SCHEMA, 'policy_id': spec['policy_id'],
                           'error': repr(error)[:500], 'series': spec['series']})
        raise
    print(json.dumps({'held_eval': spec['policy_id'][:16], 'successes': result['successes'], 'scored': result['scored']}),
          flush=True)


def finalize(directory, spec, started, *, complete):
    from . import task_rollouts
    directory = Path(directory)
    journal = directory / 'journal.jsonl'
    with journal.open('w') as merged:
        for path in sorted(directory.glob('shard-*/journal.jsonl')):
            merged.write(path.read_text())
    report = task_rollouts.score(directory)
    rewards = list(read_jsonl(directory / 'rewards.jsonl'))
    queue = {entry['key']: entry for entry in read_jsonl(directory / 'sealed-queue.jsonl')}
    finished = {row['key'] for row in read_jsonl(journal) if row.get('event') == 'finish'}
    by_position = {(entry['group'], entry['sample']): key for key, entry in queue.items()}
    packet = list(read_jsonl(directory / 'packet.jsonl'))
    samples = []
    for row in rewards:
        key = by_position[(row['group'], row['sample'])]
        efficiency = row.get('execution_efficiency') or {}
        samples.append({'task_id': row['task_id'], 'family': row.get('family'), 'level': _level(packet[row['group']]),
                        'sample': row['sample'], 'outcome': row['outcome'], 'reward': row['reward'],
                        'ran': key in finished, 'finish_status': row.get('finish_status'),
                        'model_calls': efficiency.get('model_calls'), 'repeated_actions': efficiency.get('repeated_actions'),
                        'neuralese_exchanges': row.get('neuralese_exchanges', 0)})
    ran = [s for s in samples if s['ran']]
    scored = [s for s in ran if s['reward'] is not None]
    counts = {}
    for s in ran:
        counts[s['outcome']] = counts.get(s['outcome'], 0) + 1
    calls = sum(s['model_calls'] or 0 for s in scored)
    result = {'schema': RESULT_SCHEMA, 'policy_id': spec['policy_id'], 'round': spec.get('round'),
              'manifest_sha256': spec['manifest_sha256'], 'series': spec['series'], 'code': spec['code'],
              'endpoint': spec['endpoint'], 'actor': spec.get('actor'), 'tasks': spec['tasks'], 'samples_per_task': spec['samples'],
              'started_at': started, 'finished_at': time.time(), 'elapsed_seconds': round(time.time() - started, 1),
              'complete': bool(complete and len(ran) == len(samples)), 'ran': len(ran), 'scored': len(scored),
              'successes': sum(s['reward'] == 1.0 for s in scored), 'outcome_counts': counts,
              'loop_rate': (sum(counts.get(k, 0) for k in LOOP) / len(ran)) if ran else None,
              'repeated_action_rate': (sum(s['repeated_actions'] or 0 for s in scored) / calls) if calls else None,
              'neuralese_use_rate': (sum(s['neuralese_exchanges'] > 0 for s in scored) / len(scored)) if scored else None,
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
def lineage(out):
    """({policy sha: position}, {rolled-back policy shas}) of the controller's current lineage, or ({}, set()) without a state.
    A policy's position is where the controller committed it: the policy round 0 collected from first, each RL update's result
    after its round, each SFT window's result after the update that preceded it. Policies a rewind dropped
    (``state['rewind_history'][*]['rolled_back_policies']``) and the lineage no longer holds are the rolled-back ones."""
    path = Path(out) / 'state.json'
    if not path.is_file():
        return {}, set()
    state = json.loads(path.read_text())
    positions = {}
    for item in state.get('task_curriculum_spread_yield') or []:
        if item.get('round') == 0 and item.get('policy_id'):
            positions[item['policy_id']] = (-1, 0)
    for record in state.get('completed_update_history') or []:
        positions[record['checkpoint']['sha256']] = (record['round'], 0)
    for window in (state.get('interleave') or {}).get('windows') or []:
        positions[window['checkpoint']['sha256']] = (window['controller_round'] - 1, 1)
    dropped = {sha for record in state.get('rewind_history') or [] for sha in record.get('rolled_back_policies') or []}
    return positions, dropped - set(positions)


def results(out, rolled_back=False):
    """The valid held-eval results of a controller directory in lineage order (the controller's commit order; a result of a
    policy the state does not hold follows, by start time). ``rolled_back``: only the results of policies a rewind dropped."""
    found = []
    for path in directory_of(out).glob('policy-*/result.json'):
        result = json.loads(path.read_text())
        if result.get('schema') == RESULT_SCHEMA and 'error' not in result:
            found.append(result)
    positions, dropped = lineage(out)
    if rolled_back:
        return sorted((r for r in found if r['policy_id'] in dropped), key=lambda r: r['started_at'])
    found = [r for r in found if r['policy_id'] not in dropped]
    return sorted(found, key=lambda r: (positions.get(r['policy_id'], (float('inf'), 0)), r['started_at']))


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


def trend(out):
    """Per measured policy of the current lineage, in lineage order: its numbers, the paired comparison with the previous result
    of its series and with the first of its series (None for the first one or when the series changed). Rolled-back policies
    are not here (``rolled_back``)."""
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


def rolled_back(out):
    """The measured policies a rewind dropped, with their numbers and never a pairing (they are not part of the lineage)."""
    return [{'policy': r['policy_id'], 'round': r.get('round'), 'series': r['series']['key'], 'complete': r['complete'],
             'ran': r['ran'], 'scored': r['scored'], 'successes': r['successes'], 'loop_rate': r['loop_rate']}
            for r in results(out, rolled_back=True)]


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
