"""A fixed held-out task set measured on a served policy (owner request 2026-11-10: dev16 cannot resolve our updates).

dev16 is 16 single-sample cases (about +-1.7 cases of noise). A held set pins pool tasks stratified over the pool's
families and levels (``build_manifest``; ``pin`` writes a standalone one such as ops/held-v2 with its ``exclusion-ids.json``),
and training code never selects them or their ``split_group`` siblings (``exclusion_ids``, ``curriculum.select_tasks``
``exclude_ids``). ``measure`` runs N samples per held task against a served actor (``serve`` launches one for a checkpoint)
and writes ``result.json``; ``compare`` tests the mean per-task pass-rate change (within-task permutation test, task
bootstrap; a wall-clock timeout counts as a failure, dropping them is reported beside it), with McNemar on matched
(task, sample) episodes (``paired``) as a secondary field.

- ``manifest.json`` + ``cases.jsonl``: task ids and row digests, the pool's digest, and the excluded split groups.
- ``run`` measures one policy directory; failures and a cut-short run are recorded, and a partial run is still a balanced
  one (sample-major order) and pairs on what it finished.
- A comparison series is the collector runtime (ts-host tree), the Python sources that run and score episodes
  (``SCORING_SOURCES``) and the resolved ``NATLANG_MODEL_FACING`` variants, recorded in every result. ``compare`` pairs
  results of different series by default (owner 2026-11-18: a base re-measured for every ts-host commit cost 3.5-7 h of
  Pop each time) and records the differences with a warning; ``--strict-series`` pairs only one series and one engine.
"""
from __future__ import annotations

import argparse
import collections
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

from ..common.jsonio import read_jsonl, iter_jsonl_rows, write_json_durable

MANIFEST_SCHEMA = 'natlang.rl-held-eval-manifest/1'
RESULT_SCHEMA = 'natlang.rl-held-eval-result/1'
DEFAULT_TASKS, DEFAULT_SAMPLES, DEFAULT_WORKERS, DEFAULT_GRACE = 64, 2, 16, 1200
EFFECT_STATE_SHARE = 1 / 3   # the selector's class_quota is .34: the held mixture mirrors what training practises
MAX_GROUP_ROWS = 24          # a held task whose split group is larger would shrink the training pool by that much
SEED = 'held-eval-20261110'
SEED_BASE = 4200000
LOOP = ('budget_exhaustion', 'repetition')
# One episode budget in model calls for training (grpo.py) and held measurement (EPISODE_COST_20261112: a budget stop
# scores 0). Held results measured before 2026-11-13 ran at 64; they are another baseline.
EPISODE_MODEL_CALLS = 40
# The Python sources that run episodes and decide their outcome; a change of any is a new series (with the ts-host tree).
SCORING_SOURCES = ('scripts/run_bonsai_queue.py', 'training/neuralese/natlang_neuralese/eval/task_packet.py',
                   'training/neuralese/natlang_neuralese/rl/task_rollouts.py',
                   'training/neuralese/natlang_neuralese/rl/rollout_math.py',
                   'training/neuralese/natlang_neuralese/common/task_roots.py')


def sample_timed_out(sample):
    """A wall-clock stop (unscored): the sample's flag, or its finish status in results written before the flag."""
    return bool(sample.get('timed_out') or sample.get('finish_status') == 'timeout')


def ran_out(sample):
    """The episode ran out of budget rather than ending: a loop outcome (call budget, repetition) or a wall-clock stop.
    Throughput moves episodes between the two (a slower run cuts at the wall clock what a faster one ends at the call
    budget), so this rate, unlike ``loop_rate``, does not move with the actor's speed."""
    return sample.get('outcome') in LOOP or sample_timed_out(sample)


def ran_out_rate(result):
    ran = [s for s in result['samples'] if s['ran']]
    return round(sum(map(ran_out, ran)) / len(ran), 4) if ran else None


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


def _source_groups(row):
    return set(row.get('source_groups') or ())


def build_manifest(cases_path, *, avoid, tasks=DEFAULT_TASKS, include=(), seed=SEED, avoid_groups=(), replace=(), reject=()):
    """(manifest, held rows): ``tasks`` pool rows, family- and level-stratified, of which a third carry the effect-state class.
    Deterministic in the pool and the arguments. Rows whose split group or a source group is shared with an avoided task
    (``avoid`` ids, resolved in the pool, and ``avoid_groups``: split/source group names of exposed rows outside it) are
    not candidates. ``include``: task ids held already (an earlier manifest) that the new one keeps. ``replace``: ids of
    ``include`` to drop (a wrong or ambiguous oracle); each gets a fresh task of its family and level, else of its family
    at the nearest level, else the stratified fill. ``reject``: rows that are never candidates (pool errata), without
    taking their split groups out."""
    from .curriculum import metadata
    cases = list(read_jsonl(cases_path))
    replace = set(replace)
    if replace - set(include):
        raise ValueError('replaced held tasks are not in the included manifest: ' + ', '.join(sorted(replace - set(include))[:5]))
    include = [i for i in include if i not in replace]
    include_set = set(include)
    if include_set & set(avoid):
        raise ValueError('an included held task is also to be avoided: ' + ', '.join(sorted(include_set & set(avoid))[:5]))
    avoid_groups = set(avoid_groups)
    for row in cases:
        if row.get('id') in avoid:
            avoid_groups.add(_split_group(row))
            avoid_groups |= _source_groups(row)
    reject = set(reject)
    groups = {}
    for row in cases:
        groups[_split_group(row)] = groups.get(_split_group(row), 0) + 1
    candidates, blocked = [], []
    for index, row in enumerate(cases):
        meta = metadata(row)
        exposed = _split_group(row) in avoid_groups or bool(_source_groups(row) & avoid_groups)
        if (row.get('split') == 'train' and row['id'] not in avoid and not exposed and row['id'] not in reject
                and groups[_split_group(row)] <= MAX_GROUP_ROWS):
            candidates.append({'row': index, 'id': row['id'], 'group': _split_group(row), 'source_family': row.get('family'),
                               **meta})
        elif row.get('id') in include_set:
            blocked.append(row['id'])
    if blocked:
        raise ValueError(f'{len(blocked)} included held tasks are no longer eligible (exposed group, rejected or not '
                         'train): ' + ', '.join(sorted(blocked)[:20]))
    levels = sorted({c['level'] for c in candidates})
    chosen = [c for c in candidates if c['id'] in include_set]
    if len(chosen) != len(include_set):
        raise ValueError(f'{len(include_set) - len(chosen)} included held tasks are not eligible pool rows')
    by_id = {row['id']: (index, row) for index, row in enumerate(cases)}
    # A replaced task's split group stays out too: a sibling would share its source (and often its defect).
    used_groups = {c['group'] for c in chosen} | {_split_group(by_id[i][1]) for i in replace if i in by_id}
    family_count = {}
    for c in chosen:
        family_count[c['family']] = family_count.get(c['family'], 0) + 1
    replaced = []
    for old in sorted(replace, key=lambda i: _hash('replace', i, seed=seed)):
        if old not in by_id:
            raise ValueError(f'replaced held task {old} is not a pool row')
        meta = metadata(by_id[old][1])
        options = [c for c in candidates if c['family'] == meta['family'] and c['group'] not in used_groups
                   and c['id'] not in include_set]
        # Prefer the replaced task's class (effect-state or not, keeping the class share) and source family (e.g.
        # multihiertt within table-text QA).
        same_class = [c for c in options if c.get('mix_class') == meta.get('mix_class')] or options
        options = [c for c in same_class if c['source_family'] == by_id[old][1].get('family')] or same_class
        pick = None
        if options:
            level = min({c['level'] for c in options}, key=lambda lv: (abs(lv - meta['level']), lv))
            pick = min((c for c in options if c['level'] == level), key=lambda c: _hash('row', c['id'], seed=seed))
            chosen.append(pick)
            used_groups.add(pick['group'])
            family_count[pick['family']] = family_count.get(pick['family'], 0) + 1
        replaced.append({'id': old, 'family': meta['family'], 'level': meta['level'],
                         'by': pick['id'] if pick else None, 'by_level': pick['level'] if pick else None})
    picked_ids = {c['id'] for c in chosen}
    effect = min(max(0, round(tasks * EFFECT_STATE_SHARE) - sum(1 for c in chosen if c.get('mix_class') == 'effect_state')),
                 max(0, tasks - len(chosen)))

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

    chosen += take([c for c in candidates if c.get('mix_class') == 'effect_state' and c['id'] not in picked_ids], effect)
    chosen += take([c for c in candidates if c.get('mix_class') != 'effect_state' and c['id'] not in picked_ids],
                   tasks - len(chosen))
    if len(chosen) < tasks:
        raise ValueError(f'the pool has only {len(chosen)} eligible held tasks (wanted {tasks})')
    chosen.sort(key=lambda c: c['row'])
    rows = [cases[c['row']] for c in chosen]
    manifest = {
        'schema': MANIFEST_SCHEMA, 'seed': seed, 'included': len(include_set), 'tasks': [
            {'id': c['id'], 'pool_row': c['row'], 'family': c['family'], 'level': c['level'],
             **({'mix_class': c['mix_class']} if c.get('mix_class') else {}), 'split_group': c['group'],
             'row_sha256': hashlib.sha256(json.dumps(cases[c['row']], sort_keys=True).encode()).hexdigest()}
            for c in chosen],
        'excluded_split_groups': sorted({c['group'] for c in chosen}),
        'pool': {'path': str(cases_path), 'sha256': _sha256(cases_path), 'rows': len(cases)},
        'avoided_task_ids': len(avoid), 'avoided_groups': len(avoid_groups), 'rejected_rows': len(reject),
        **({'replaced': replaced} if replace else {}),
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
    row sharing a held ``split_group``, plus the same for each earlier held manifest it lists in ``also_excludes`` (a
    successor set keeps its predecessor's tasks out of training: held-v3 excludes held-v2). ``cases``: the pool rows (or
    a path); by default the manifest's pool, whose sha256 must match the pinned one. Experiments call this to exclude a
    held set (e.g. ``ops/held-v3/manifest.json``)."""
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    if cases is None:
        pool = manifest['pool']
        if _sha256(pool['path']) != pool['sha256']:
            raise ValueError('the pool changed since the held manifest was pinned: ' + pool['path'])
        cases = pool['path']
    if isinstance(cases, (str, Path)):
        cases = list(read_jsonl(cases))
    held, groups = {t['id'] for t in manifest['tasks']}, set(manifest['excluded_split_groups'])
    excluded = {row['id'] for row in cases if row.get('id') in held or _split_group(row) in groups}
    for earlier in manifest.get('also_excludes') or ():
        if _sha256(earlier['path']) != earlier['sha256']:
            raise ValueError('an excluded earlier held manifest changed: ' + earlier['path'])
        excluded |= exclusion_ids(earlier['path'], cases)
    return frozenset(excluded)


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


def series(config=None, *, engine_host=None):
    from ..common.code_provenance import _git, repository_root
    root = repository_root()
    tree = _git(root, 'rev-parse', 'HEAD:ts-host')
    dirty = bool(_git(root, 'status', '--porcelain', '--untracked-files=no', '--', 'ts-host'))
    scoring = {path: _git(root, 'rev-parse', f'HEAD:{path}') for path in SCORING_SOURCES}
    scoring_dirty = bool(_git(root, 'status', '--porcelain', '--untracked-files=no', '--', *SCORING_SOURCES))
    variants = ','.join(sorted(model_facing(config).split(',')))
    scored_by = ','.join(f'{path}={blob}' for path, blob in sorted(scoring.items()))
    # Trained prompt states the collector serves (NATLANG_NEURALESE_PROMPT_STATES: the learned operators' own states,
    # ts-host model/prompt-states.ts) change what those calls see: a build is its own series.
    states = os.environ.get('NATLANG_NEURALESE_PROMPT_STATES')
    states = None if not states or states == 'off' else hashlib.sha256((Path(states) / 'result.json').read_bytes()).hexdigest()
    # The engine host (async-train §1: the DGX held series) is part of the key: a series measured on one machine never
    # pairs with another's (``compare`` refuses it, also for results that predate the field: ``engine_host_of``). Pop,
    # the legacy series, keeps its keys (None); the DGX series sets NATLANG_HELD_SERIES_HOST=dgx.
    host = engine_host if engine_host is not None else (os.environ.get('NATLANG_HELD_SERIES_HOST') or None)
    host = None if host == 'pop' else host
    key_parts = f'{tree}|{scored_by}|{variants}'
    if states:
        key_parts += f'|{states}'
    if host:
        key_parts += f'|host={host}'
    key = hashlib.sha256(key_parts.encode()).hexdigest()[:12]
    return {'key': key, 'ts_host_tree': tree, 'ts_host_dirty': dirty, 'scoring_sources': scoring,
            'scoring_dirty': scoring_dirty, 'model_facing': variants,
            **({'prompt_states': states} if states else {}),
            **({'engine_host': host} if host else {})}


# ---- the measurement ----------------------------------------------------------------------------------------------------
def policy_directory(out, policy_id):
    checkpoint, _, adapters = policy_id.partition('+')
    return directory_of(out) / (f'policy-{checkpoint[:16]}' + (f'-{adapters[:16]}' if adapters else ''))


def _entries(spec):
    """The sealed, sample-major queue (a partly finished run stays balanced) bound to the held cases and the policy."""
    from types import SimpleNamespace
    from . import task_rollouts
    directory = Path(spec['dir'])
    task_rollouts.prepare(SimpleNamespace(
        cases=Path(spec['cases']), out=directory, rows=spec['tasks'], offset=0, group=spec['samples'], temperature=1.,
        case_seconds=spec['case_seconds'], max_model_requests=spec['max_model_requests'], max_turns=spec['max_turns'],
        seed_base=spec.get('seed_base', SEED_BASE), root_mode='student', match_seeds_from=None,
        policy_checkpoint_sha256=spec['policy_id'].split('+')[0]))   # an adapted policy: its checkpoint
    entries = list(read_jsonl(directory / 'queue.jsonl'))
    for entry in entries:
        entry.update(prompt_rendering='full', context_tokens=spec['context_tokens'])
    entries.sort(key=lambda entry: (entry['sample'], entry['index']))
    return entries


def _finished(journal):
    return journal.exists() and any(row.get('event') == 'finish' for row in read_jsonl(journal))


def _finish_seconds(journal):
    """The episode's wall seconds from its journal's finish row (None while unfinished)."""
    if not journal.exists():
        return None
    finishes = [row for row in read_jsonl(journal) if row.get('event') == 'finish']
    return finishes[-1].get('elapsed_seconds') if finishes else None


def duration_prior(paths):
    """task_id -> every ran episode's wall seconds over earlier ``result.json`` files (any policy, same held tasks): the
    dispatch order's evidence before (and beside) this run's own episodes of the task."""
    seconds = collections.defaultdict(list)
    for path in paths:
        for sample in json.loads(Path(path).read_text()).get('samples') or ():
            if sample.get('ran') and isinstance(sample.get('elapsed_seconds'), (int, float)):
                seconds[sample['task_id']].append(float(sample['elapsed_seconds']))
    return dict(seconds)


DISPATCH_ORDERS = ('longest-first', 'longest-first-max', 'sealed')


class Dispatch:
    """Which entry a free slot runs next: sample-major (sample s+1 starts only when every sample-s entry has started, so a
    partly finished run stays balanced), and within a sample the task expected to run longest first, so the last
    minutes are not spent waiting for one long episode that started late (longest-processing-time-first).

    ``longest-first`` (default): a task's estimate is the mean of every wall time known for it (``prior``: earlier
    results, ``duration_prior``; and its finished episodes in this run), at least the time so far of its running
    episodes. Episode times of one task are heavy-tailed (held-v3 s7-task-195 over grpo-v4-run2's evals: 24 to 3,601 s),
    so a single short sample-0 episode is weak evidence: the previous estimate (``longest-first-max``: the longest
    episode in this run, else the prior's mean) dispatched such tasks last in sample 1, and their long second episodes
    set the end (run2 step 36: 14% of the decode time at one running sequence). Replayed on run2's eight complete Pop
    evals at 20 slots with the run's earlier results as prior: mean span 12,148 s (longest-first-max, base prior) ->
    11,597 s, within 3% of the bound (11,248 s; POP_HELD_THROUGHPUT_20261117.md). Ties keep the sealed order.
    ``sealed`` keeps the sealed queue's order. Order changes only which episodes share a batch, never an episode's
    task, seed or budget."""

    def __init__(self, entries, *, prior=None, order='longest-first', observed=None):
        if order not in DISPATCH_ORDERS:
            raise ValueError(f'unknown dispatch order {order!r}')
        self.pending = list(entries)
        self.order = order
        # task -> wall seconds of earlier results (a float: one mean, as older callers pass)
        self.prior = {task: [float(v)] if isinstance(v, (int, float)) else [float(x) for x in v]
                      for task, v in (prior or {}).items()}
        self.finished = collections.defaultdict(list)   # task_id -> finished episodes' seconds in this run
        for task, seconds in (observed or {}).items():
            self.finished[task] += [float(seconds)] if isinstance(seconds, (int, float)) else [float(x) for x in seconds]
        self.running = {}   # key -> (task_id, start time)

    def estimate(self, task, now):
        running = [now - start for running_task, start in self.running.values() if running_task == task]
        if self.order == 'longest-first-max':
            seen = ([max(self.finished[task])] if self.finished.get(task) else []) + running
            prior = self.prior.get(task)
            return max(seen) if seen else (sum(prior) / len(prior) if prior else 0.)
        known = self.prior.get(task, []) + self.finished.get(task, [])
        return max([sum(known) / len(known) if known else 0.] + running)

    def take(self, now=None):
        if not self.pending:
            return None
        if self.order == 'sealed':
            entry = self.pending.pop(0)
        else:
            now = time.time() if now is None else now
            sample = min(entry['sample'] for entry in self.pending)
            best = max(((self.estimate(entry['task_id'], now), -position), position)
                       for position, entry in enumerate(self.pending) if entry['sample'] == sample)[1]
            entry = self.pending.pop(best)
        self.running[entry['key']] = (entry['task_id'], time.time() if now is None else now)
        return entry

    def done(self, entry, seconds):
        self.running.pop(entry['key'], None)
        if isinstance(seconds, (int, float)):
            self.finished[entry['task_id']].append(float(seconds))


STOP_DRAIN_SECONDS = 120   # after a stop: terminated episodes get this long, then they are killed and collection returns


def collect(directory, entries, endpoints, *, repo, runtime, env, stop, log=None, prior=None, order='longest-first',
            drain_seconds=STOP_DRAIN_SECONDS):
    """Work-stealing slots (``endpoints``: [(url, slots)]) over the sealed ``entries``, one ``run_bonsai_queue`` process per
    episode under ``directory/workers`` (a slow episode never holds a shard of others back). Entries with a finished journal
    are skipped (resume). The next entry is ``Dispatch``'s choice (``prior``, ``order``). Returns once every entry ran or,
    after ``stop``, within ``drain_seconds``: running episodes get SIGTERM, those still alive at half the drain SIGKILL,
    and a slot thread still blocked at the end is left behind (daemon) with every thread's stack in the log, so the
    partial result is always written (the 2026-11-17 Pop smoke's measure sat 55 min after its SIGTERM)."""
    directory, repo = Path(directory), Path(repo)
    work = directory / 'workers'
    work.mkdir(exist_ok=True)
    todo, observed = [], collections.defaultdict(list)
    for entry in entries:
        seconds = _finish_seconds(work / f"{entry['key']}.journal.jsonl")
        if seconds is None:
            todo.append(entry)
        elif isinstance(seconds, (int, float)):
            observed[entry['task_id']].append(float(seconds))
    dispatch = Dispatch(todo, prior=prior, order=order, observed=observed)
    lock, children, done = threading.Lock(), set(), [0]
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
                entry = dispatch.take()
                if entry is None:
                    return
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
                dispatch.done(entry, _finish_seconds(journal))
                children.discard(child)
                done[0] += 1
                if log and done[0] % 16 == 0:
                    log(json.dumps({'held_eval_progress': done[0], 'of': len(todo),
                                    'minutes': round((time.time() - started) / 60, 1)}))

    def signal_children(kill):
        with lock:
            alive = [child for child in children if child.poll() is None]
        for child in alive:   # outside the lock: a slot thread that just reaped its child needs it
            try:
                child.kill() if kill else child.terminate()
            except OSError:
                pass
        return len(alive)

    def watch():
        stop.wait()
        signal_children(kill=False)
        if not stop_drained.wait(drain_seconds / 2):
            left = signal_children(kill=True)
            if log and left:
                log(json.dumps({'event': 'held eval stop', 'killed_episodes': left}))

    stop_drained = threading.Event()
    threading.Thread(target=watch, daemon=True).start()
    threads = [threading.Thread(target=slot, args=(url.rstrip('/'),), daemon=True)
               for url, slots in endpoints for _ in range(max(1, min(slots, len(todo))))]
    for thread in threads:
        thread.start()
    stopped_at = None
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(1)
        if stop.is_set():
            stopped_at = stopped_at or time.monotonic()
            if time.monotonic() - stopped_at > drain_seconds:
                stuck = [thread for thread in threads if thread.is_alive()]
                if log:
                    import traceback
                    frames = sys._current_frames()
                    log(json.dumps({'event': 'held eval stop', 'stuck_slots': len(stuck), 'stacks': [
                        ''.join(traceback.format_stack(frames[thread.ident])) for thread in stuck
                        if thread.ident in frames]}))
                break
    stop_drained.set()


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
    from ..common.sampling import rl_sampling_controls
    (directory / 'sampling-config.json').write_text(json.dumps(
        rl_sampling_controls(spec.get('sampling', LEGACY_SAMPLING)['top_p'])) + '\n')
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # kill -USR1 PID: every thread's stack to stderr
    endpoints = spec.get('endpoints') or [[spec['endpoint'], spec['workers']]]
    env = {**os.environ, 'NATLANG_MODEL_FACING': spec['series']['model_facing']}
    dispatch = spec.get('dispatch') or {}
    collect(directory, entries, endpoints, repo=spec['repo'], runtime=spec['runtime'], env=env, stop=stop,
            log=lambda line: print(line, flush=True), order=dispatch.get('order', 'longest-first'),
            prior=duration_prior(dispatch.get('prior') or ()))
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
                        'graded_reward': row.get('graded_reward'),
                        'ran': key in finishes, 'finish_status': row.get('finish_status'), 'timed_out': _timed_out(finish),
                        'elapsed_seconds': (finish or {}).get('elapsed_seconds'),
                        'completion_tokens': (finish or {}).get('completion_tokens'),
                        'model_calls': efficiency.get('model_calls'), 'repeated_actions': efficiency.get('repeated_actions'),
                        'neuralese_exchanges': row.get('neuralese_exchanges', 0)})
    elapsed = round(time.time() - started, 1)
    result = {'schema': RESULT_SCHEMA, 'policy_id': spec['policy_id'], 'round': spec.get('round'),
              'manifest_sha256': spec['manifest_sha256'], 'series': spec['series'], 'code': spec['code'],
              'endpoint': spec['endpoint'], 'actor': spec.get('actor'), 'engine': spec.get('engine'), 'tasks': spec['tasks'], 'samples_per_task': spec['samples'],
              'seed_base': spec.get('seed_base', SEED_BASE), 'case_seconds': spec.get('case_seconds'),
              'context_tokens': spec.get('context_tokens', LEGACY_CONTEXT_TOKENS),
              'sampling': spec.get('sampling', LEGACY_SAMPLING),
              'started_at': started, 'finished_at': time.time(), 'elapsed_seconds': elapsed,
              **aggregate(samples, complete=complete, elapsed=elapsed),
              'score': {'groups_with_spread': report.get('groups_with_spread')}}
    write_json_durable(directory / 'result.json', result)
    record_result(directory, spec)
    return result


def record_result(directory, spec):
    """The finished evaluation as a protected registry dataset (``held-eval-...``): its result and episodes, the held
    manifest it measured as input. Bookkeeping never fails an evaluation: an error is printed as an event."""
    try:
        from ..common import data_registry
        data_registry.record_dataset(
            data_registry.dataset_id(directory, 'held-eval'), directory, kind='held-eval', protected=True,
            include=['result.json', 'run.json', 'rewards.jsonl', 'packet.jsonl', 'journal.jsonl'],
            format=f'{RESULT_SCHEMA} result.json + run.json, rewards.jsonl and journal.jsonl of the episodes',
            inputs=[x for x in (spec.get('manifest'),) if x], license='project-generated (held-out evaluation; view only)',
            note=f"policy {spec.get('policy_id')}, {spec.get('samples')} samples per task")
    except Exception as error:
        print(json.dumps({'event': 'data registry error', 'error': repr(error)[:300]}), flush=True)


def aggregate(samples, *, complete, elapsed):
    """The result fields computed from its samples (``finalize``, ``merge``)."""
    ran = [s for s in samples if s['ran']]
    scored = [s for s in ran if s['reward'] is not None]
    counts = {}
    for s in ran:
        counts[s['outcome']] = counts.get(s['outcome'], 0) + 1
    calls = sum(s['model_calls'] or 0 for s in scored)
    successes = sum(s['reward'] == 1.0 for s in scored)
    timeouts = sum(s['timed_out'] for s in ran)
    tokens = sum(s['completion_tokens'] or 0 for s in ran)
    return {'complete': bool(complete and len(ran) == len(samples)), 'ran': len(ran), 'scored': len(scored),
            'successes': successes, 'success_rate': round(successes / len(scored), 4) if scored else None,
            # Secondary: the training reward with partial file-effect credit (held is judged on acceptance).
            'mean_graded_reward': (round(sum(s['graded_reward'] for s in scored) / len(scored), 4)
                                   if scored and all(s.get('graded_reward') is not None for s in scored) else None),
            'timeouts': timeouts, 'timeout_rate': round(timeouts / len(ran), 4) if ran else None,
            'unscored_other': len(ran) - len(scored) - sum(s['timed_out'] and s['reward'] is None for s in ran),
            'outcome_counts': counts,
            'loop_rate': (sum(counts.get(k, 0) for k in LOOP) / len(ran)) if ran else None,
            'ran_out_rate': ran_out_rate({'samples': samples}),
            'repeated_action_rate': (sum(s['repeated_actions'] or 0 for s in scored) / calls) if calls else None,
            'neuralese_use_rate': (sum(s['neuralese_exchanges'] > 0 for s in scored) / len(scored)) if scored else None,
            'completion_tokens': tokens, 'tokens_per_second': round(tokens / elapsed, 1) if elapsed else None,
            'samples': samples}


def merge(parts):
    """One result from complete runs of one policy whose seeds continue each other (part k's ``seed_base`` is the
    first's plus the samples before it: ``measure --samples 2 --seed-base SEED_BASE+2`` measures samples 2-3 of a
    4-sample run). The samples are renumbered into one series, so ``compare`` reads it as a single run. Parts measured on
    different engines are allowed: the result's ``engine`` lists them per sample range (``engine_parts``), and ``compare``
    pairs it only with a result of the same engine mix, where each task sees the same engines in both arms."""
    first = parts[0]
    same = ('policy_id', 'manifest_sha256', 'tasks', 'case_seconds', 'context_tokens', 'sampling')
    offset, samples, engines = 0, [], []
    for part in parts:
        for key in same:
            if part.get(key) != first.get(key):
                raise ValueError(f'parts differ in {key}: {first.get(key)!r} vs {part.get(key)!r}')
        if part['series']['key'] != first['series']['key'] or (part.get('code') or {}).get('git_commit') != \
                (first.get('code') or {}).get('git_commit'):
            raise ValueError('parts ran different code or scoring series')
        if not part.get('complete'):
            raise ValueError('merge takes complete runs only')
        if part.get('seed_base', SEED_BASE) != first.get('seed_base', SEED_BASE) + offset:
            raise ValueError(f'part seeds start at {part.get("seed_base")}, not {first.get("seed_base", SEED_BASE) + offset}')
        samples += [{**s, 'sample': s['sample'] + offset} for s in part['samples']]
        engines.append({'samples': [offset, offset + part['samples_per_task']], 'engine': result_engine(part)})
        offset += part['samples_per_task']
    distinct = {json.dumps(e['engine'], sort_keys=True) for e in engines}
    elapsed = sum(part.get('elapsed_seconds') or 0 for part in parts)
    return {**{k: v for k, v in first.items() if k not in ('samples', 'actor', 'endpoint')},
            'samples_per_task': offset, 'started_at': min(p.get('started_at') or 0 for p in parts),
            'finished_at': max(p.get('finished_at') or 0 for p in parts), 'elapsed_seconds': elapsed,
            'engine': first.get('engine') if len(distinct) == 1 else {'engine_parts': engines},
            'merged_from': [{'seed_base': p.get('seed_base', SEED_BASE), 'samples': p['samples_per_task'],
                             'finished_at': p.get('finished_at')} for p in parts],
            **aggregate(samples, complete=True, elapsed=elapsed)}


def _level(task):
    return (((task.get('curriculum') or {}).get('rl')) or {}).get('level')


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


# The call budget of a held measurement (the harness compacts near it); results measured before it was recorded ran at 32768.
CONTEXT_TOKENS = 131072
LEGACY_CONTEXT_TOKENS = 32768
# Results measured before the sampling law was recorded (2026-11-13) sampled the untruncated distribution at T=1.
LEGACY_SAMPLING = {'temperature': 1., 'top_p': 1.}


def task_counts(result, *, timeouts='failure'):
    """{task: [successes, counted]} over the episodes that ran. ``timeouts='failure'`` counts a wall-clock stop as a
    failure (it falls on hard tasks, so dropping it biases the rate up for the side that times out more); ``'drop'``
    leaves it out. Other unscored episodes (infrastructure) are left out either way."""
    counts = {}
    for s in result['samples']:
        if not s['ran']:
            continue
        if s['reward'] is None and not (timeouts == 'failure' and sample_timed_out(s)):
            continue
        entry = counts.setdefault(s['task_id'], [0, 0])
        entry[0] += s['reward'] == 1.0
        entry[1] += 1
    return counts


def _rate_change(a, b, tasks, *, resamples, seed=0):
    """The mean over ``tasks`` of the per-task pass-rate change b - a ({task: [successes, n]}; sample counts may differ),
    its task-bootstrap 95% interval and the two-sided p of a within-task permutation test: under no change, a task's
    pooled outcomes are exchangeable between the two sides, so each side's success count is hypergeometric given the
    pooled count (Monte Carlo over all tasks at once; tasks that all pass or all fail contribute a constant)."""
    import bisect
    import random
    tasks = [t for t in tasks if t in a and t in b and a[t][1] and b[t][1]]
    if not tasks:
        return {'tasks': 0, 'first_rate': None, 'second_rate': None, 'delta': None, 'ci95_task_bootstrap': None,
                'permutation_p': None}
    diffs = {t: b[t][0] / b[t][1] - a[t][0] / a[t][1] for t in tasks}
    observed = sum(diffs.values()) / len(tasks)
    rng = random.Random(seed)
    boot = sorted(sum(diffs[rng.choice(tasks)] for _ in tasks) / len(tasks) for _ in range(resamples))
    constant, varying = 0.0, []
    for t in tasks:
        (sa, na), (sb, nb) = a[t], b[t]
        k, n = sa + sb, na + nb
        lo, hi = max(0, k - na), min(k, nb)
        if lo == hi:
            constant += lo / nb - (k - lo) / na
            continue
        support = list(range(lo, hi + 1))
        weights = [math.comb(nb, x) * math.comb(na, k - x) for x in support]
        total, cumulative, acc = sum(weights), [], 0
        for w in weights:
            acc += w
            cumulative.append(acc / total)
        varying.append(([x / nb - (k - x) / na for x in support], cumulative))
    extreme = 0
    for _ in range(resamples):
        stat = constant
        for values, cumulative in varying:
            stat += values[min(bisect.bisect_left(cumulative, rng.random()), len(values) - 1)]
        extreme += abs(stat / len(tasks)) >= abs(observed) - 1e-12
    return {'tasks': len(tasks), 'first_rate': round(sum(a[t][0] / a[t][1] for t in tasks) / len(tasks), 4),
            'second_rate': round(sum(b[t][0] / b[t][1] for t in tasks) / len(tasks), 4), 'delta': round(observed, 4),
            'ci95_task_bootstrap': [round(boot[int(.025 * resamples)], 4), round(boot[int(.975 * resamples) - 1], 4)],
            'permutation_p': round((1 + extreme) / (1 + resamples), 4)}


# Engine settings whose difference refuses a pairing even across series: the engine kind, the weight and KV precision,
# and speculative decoding (its sampling law is checked apart). Other attested differences (fork commit, running cap,
# overlap, export format) are recorded with a warning; ``strict_series`` refuses them as well.
PAIRING_ENGINE_FIELDS = ('engine', 'dtype', 'kv_cache_dtype', 'speculative')


# The machine behind an attested actor export format (serve_native_checkpoint --actor-export-format; the DGX serves the
# default ``sglang-hf-...`` export). A result without ``series.engine_host`` (every result before the DGX series) is
# placed by its engine.
EXPORT_FORMAT_HOSTS = (('sglang-pop-', 'pop'), ('native-pop-', 'pop'), ('sglang-colab-g4-', 'colab'),
                       ('sglang-colab-', 'colab'), ('sglang-hf-', 'dgx'))


def engine_host_of(result):
    """The machine a result was measured on: its series' ``engine_host``, else the host of its attested export format
    (Pop for ``sglang-pop-*``, the DGX for the default ``sglang-hf-*``), else None (not attested)."""
    host = (result.get('series') or {}).get('engine_host')
    if host:
        return host
    engine = result_engine(result) or {}
    parts = engine.get('engine_parts') if isinstance(engine, dict) else None
    formats = [((p.get('engine') or {}).get('export_format')) for p in parts] if parts else [engine.get('export_format')]
    hosts = {next((h for prefix, h in EXPORT_FORMAT_HOSTS if str(f).startswith(prefix)), None) for f in formats if f}
    hosts.discard(None)
    return hosts.pop() if len(hosts) == 1 else (','.join(sorted(hosts)) if hosts else None)


def pairing_differences(first, second):
    """(series differences, engine differences) of two results: {field: [first, second]} each."""
    series_a, series_b = first.get('series') or {}, second.get('series') or {}
    series = {}
    for key in ('ts_host_tree', 'model_facing', 'ts_host_dirty', 'scoring_dirty', 'engine_host'):
        if series_a.get(key) != series_b.get(key):
            series[key] = [series_a.get(key), series_b.get(key)]
    scoring_a, scoring_b = series_a.get('scoring_sources') or {}, series_b.get('scoring_sources') or {}
    for path in sorted(set(scoring_a) | set(scoring_b)):
        if scoring_a.get(path) != scoring_b.get(path):
            series[path] = [scoring_a.get(path), scoring_b.get(path)]
    codes = [(r.get('code') or {}).get('git_commit') for r in (first, second)]
    if codes[0] != codes[1]:
        series['git_commit'] = codes
    engines = result_engine(first), result_engine(second)
    engine = {}
    if None not in engines:
        engine = {k: [engines[0].get(k), engines[1].get(k)] for k in sorted(set(engines[0]) | set(engines[1]))
                  if engines[0].get(k) != engines[1].get(k)}
    return series, engine


def compare(first, second, *, resamples=2000, trained_families=None, ignore_ids=(), strict_series=False):
    """The standalone comparison of two result.json files (``second`` against first).

    Primary: the mean per-task pass-rate change over the tasks both measured, with a within-task permutation test (valid
    for unequal sample counts, no arbitrary pairing of independent samples) and a task-bootstrap 95% interval; a
    wall-clock timeout counts as a failure. ``sensitivity_drop_timeouts`` is the same with timeouts left out. Subsets:
    ``without_ignored`` drops ``ignore_ids`` (structurally-zero or wrong-oracle tasks), and with ``trained_families``
    (families the second policy trained on) ``trained_families``/``untrained_families`` split the tasks. Secondary:
    McNemar and the task sign test on matched scored (task, sample) episodes (``paired``), timeouts, loop and ran-out
    rates, and whether the two ran under one code commit and series.

    Results of different series (ts-host tree, scoring sources, model-facing variants, commit) pair by default; the
    differences are in ``series_differences``/``engine_differences`` and ``warnings``. A different sampling law, context
    budget or ``PAIRING_ENGINE_FIELDS`` setting is refused; ``strict_series`` also refuses any series or engine
    difference."""
    budgets = [result.get('context_tokens', LEGACY_CONTEXT_TOKENS) for result in (first, second)]
    if budgets[0] != budgets[1]:
        raise ValueError(f'results measured with different context budgets are not paired: {budgets[0]} vs {budgets[1]}')
    laws = [result.get('sampling', LEGACY_SAMPLING) for result in (first, second)]
    if laws[0] != laws[1]:
        raise ValueError(f'results sampled under different laws are not paired: {laws[0]} vs {laws[1]}')
    engines = result_engine(first), result_engine(second)
    series_differences, engine_differences = pairing_differences(first, second)
    def pairing_engine(engine):   # a merged result's engine mix: per sample range
        if 'engine_parts' in engine:
            return [[p['samples'], pairing_engine(p['engine'] or {})] for p in engine['engine_parts']]
        return {k: engine.get(k) for k in PAIRING_ENGINE_FIELDS}
    refused = {k: v for k, v in engine_differences.items() if strict_series or k in PAIRING_ENGINE_FIELDS}
    if None not in engines and pairing_engine(engines[0]) != pairing_engine(engines[1]):
        refused['engine_parts'] = [pairing_engine(engines[0]), pairing_engine(engines[1])]
    if refused:
        raise ValueError('results served by different engine settings are not paired: ' + json.dumps(refused))
    hosts = engine_host_of(first), engine_host_of(second)
    if None not in hosts and hosts[0] != hosts[1]:
        raise ValueError(f'results measured on different engine hosts ({hosts[0]} and {hosts[1]}) are different held '
                         'series and are not paired; measure both policies on one host')
    if strict_series and (series_differences or first['series']['key'] != second['series']['key']):
        raise ValueError('results of different scoring series are not paired (--strict-series): '
                         + json.dumps(series_differences))
    warnings = []
    if first['series']['key'] != second['series']['key'] or series_differences:
        warnings.append('different code or series (' + ', '.join(sorted(series_differences)) + ')')
    if engine_differences:
        warnings.append('engine differs in ' + ', '.join(sorted(engine_differences)))
    incomplete = [name for name, result in (('first', first), ('second', second)) if not result.get('complete')]
    if incomplete:
        # a stopped measurement's partial result pairs only the tasks it reached (a fast-first subset): never a verdict
        warnings.append('incomplete measurement: ' + ', '.join(incomplete))

    timed_out = sample_timed_out

    def table(result):
        return {(s['task_id'], s['sample']): s for s in result['samples'] if s['ran']}
    a, b = table(first), table(second)
    common = set(a) & set(b)

    def side(result):
        return {'policy': result['policy_id'], 'code': (result.get('code') or {}).get('git_commit'),
                'series': result['series']['key'], 'seed_base': result.get('seed_base', SEED_BASE), 'ran': result['ran'],
                'scored': result['scored'], 'successes': result['successes'],
                'success_rate': round(result['successes'] / result['scored'], 4) if result['scored'] else None,
                'mean_graded_reward': result.get('mean_graded_reward'),
                'timeouts': sum(timed_out(s) for s in result['samples'] if s['ran']),
                'loop_rate': result['loop_rate'], 'ran_out_rate': ran_out_rate(result),
                'repeated_action_rate': result.get('repeated_action_rate'), 'elapsed_seconds': result.get('elapsed_seconds'), 'complete': result.get('complete')}
    counts = {mode: (task_counts(first, timeouts=mode), task_counts(second, timeouts=mode)) for mode in ('failure', 'drop')}
    every = sorted(set(counts['failure'][0]) | set(counts['failure'][1]))
    ignored = set(ignore_ids)
    family = {s['task_id']: s.get('family') for result in (first, second) for s in result['samples']}

    def change(tasks, mode='failure'):
        return _rate_change(*counts[mode], tasks, resamples=resamples)
    subsets = {}
    if ignored:
        subsets['without_ignored'] = {**change([t for t in every if t not in ignored]),
                                      'ignored_tasks': len(ignored & set(every))}
    if trained_families is not None:
        trained = set(trained_families)
        subsets['trained_families'] = change([t for t in every if family.get(t) in trained and t not in ignored])
        subsets['untrained_families'] = change([t for t in every if family.get(t) not in trained and t not in ignored])
    out_first = sum(ran_out(a[k]) and not ran_out(b[k]) for k in common)
    out_second = sum(ran_out(b[k]) and not ran_out(a[k]) for k in common)
    rates = ran_out_rate(first), ran_out_rate(second)
    return {'first': side(first), 'second': side(second), 'engine': engines[0] or engines[1],
            'engine_attested': None not in engines, 'complete': not incomplete, 'same_code': side(first)['code'] == side(second)['code'],
            'same_series': side(first)['series'] == side(second)['series'],
            'same_manifest': first.get('manifest_sha256') == second.get('manifest_sha256'),
            'series_differences': series_differences, 'engine_differences': engine_differences, 'warnings': warnings,
            'primary': {**change(every), 'statistic': 'mean per-task pass-rate change', 'timeouts': 'failure'},
            'sensitivity_drop_timeouts': {**change(every, 'drop'), 'timeouts': 'dropped'},
            'subsets': subsets,
            'matched_pairs': len(common), 'pairs_lost_to_timeout': sum(
                1 for k in common if timed_out(a[k]) or timed_out(b[k])),
            'paired': paired(first, second),
            'loop_rate_delta': (round(second['loop_rate'] - first['loop_rate'], 4)
                                if first['loop_rate'] is not None and second['loop_rate'] is not None else None),
            'ran_out_rate_delta': round(rates[1] - rates[0], 4) if None not in rates else None,
            # Over every matched pair that ran, timeouts included (they are part of the measure).
            'ran_out_paired': {'pairs': len(common), 'first': sum(ran_out(a[k]) for k in common),
                               'second': sum(ran_out(b[k]) for k in common), 'only_first': out_first,
                               'only_second': out_second,
                               'mcnemar_exact_p': round(binomial_two_sided(out_first, out_second), 4)}}


def read_ids(paths):
    """Task ids from JSONL files ({"id": ...} per line) or JSON lists."""
    ids = set()
    for path in paths:
        text = Path(path).read_text()
        if text.lstrip().startswith('['):
            ids |= set(json.loads(text))
        else:
            ids |= {json.loads(line)['id'] for line in text.splitlines() if line.strip()}
    return ids


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
# The held-v2 eval engine (coordinator decision 2026-11-12, pinned in ops/held-v2/engine.json): one SGLang engine per
# checkpoint, FP8 KV. Every measured policy is served with exactly these settings; ``compare`` refuses other pairings.
ACTOR_ARGS = ('--engine', 'sglang', '--sglang-projection', 'vendor', '--sglang-kv-cache-dtype', 'fp8_e4m3',
              '--sglang-kv-gb', '16', '--sglang-max-running', '128', '--sglang-context', '131072')
ACTOR_BUDGET_GB = 40   # --sglang-kv-gb + 24 (measured 2026-11-12: SGLang 23 GB at 16 GB KV, front 14 GB once a write_text reloads the native model)
# max_running: SGLang's running-sequence cap (decode batch shapes and CUDA graphs; Pop 14 -> 20 slots at the
# grpo-v4 pin, plans/neuralese/POP_HELD_THROUGHPUT_20261117.md). Results of actors that did not attest it read None.
ENGINE_FIELDS = ('engine', 'dtype', 'projection', 'kv_cache_dtype', 'overlap_schedule', 'bf16_reduction', 'max_running',
                 'speculative')


def engine_of(binding):
    """The arithmetic-relevant serving settings an actor attests (``rl_policy_binding.serving_engine``): engine, fork
    commit, dtype, projection, KV dtype, overlap, reduction, running-sequence cap, and the actor export format; None when not
    attested."""
    engine = (binding or {}).get('serving_engine')
    if not isinstance(engine, dict):
        return None
    return {**{key: engine.get(key) for key in ENGINE_FIELDS}, 'fork_commit': (engine.get('fork') or {}).get('commit'),
            'export_format': ((binding.get('actor_export') or {}).get('format'))}


def result_engine(result):
    return result.get('engine') or engine_of(((result.get('actor') or {}).get('info')))


def pinned_engine(manifest):
    """The engine pin beside a held manifest (``engine.json``: ``serve_args`` and the ``expect``ed attested fields)."""
    path = Path(manifest).parent / 'engine.json'
    return json.loads(path.read_text()) if path.exists() else None


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
          admission_seconds=14400, priority='medium', actor_args=ACTOR_ARGS, log=print):
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
               '--class', 'experiment', '--priority', priority, '--no-resumable', '--wait', str(admission_seconds),
               '--workdir', str(repo), '--',
               'env', f'NATLANG_REPO={repo}', f'PYTHONPATH={repo}/training/neuralese:{repo}', 'HF_HUB_OFFLINE=1',
               'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', str(python), str(repo / 'scripts/neuralese_data/serve_native_checkpoint.py'),
               '--store-dir', str(store), '--checkpoint', str(checkpoint), '--checkpoint-sha256', sha256, '--base', str(base),
               '--port', str(port), '--prompt-rendering', 'full', *actor_args, '--hf-export-dir', str(store / 'sglang-hf'), *extra]
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


def policy_id_of(binding):
    """The policy an actor serves: its checkpoint's sha256, plus ``+`` and the digest of the default adapter stack it
    applies to every request (``serve_native_checkpoint.py --adapter-release``), so an adapted policy never pairs as or
    resumes into its base."""
    sha = binding.get('checkpoint_sha256')
    adapters = binding.get('adapters')
    if not sha or not adapters:
        return sha
    from ..common.hashing import canonical_json_sha256_hex
    return f'{sha}+{canonical_json_sha256_hex(adapters["stack"])}'


def _read_info(client, attempts=None, wait=20.0):
    """The actor's identity, retried: one connect timeout to a freshly started remote actor (seen over the tailnet
    on 2026-11-19) must not throw away its checkpoint copy and startup. NATLANG_HELD_INFO_ATTEMPTS=1 opts out."""
    import time, urllib.error
    attempts = attempts or int(os.environ.get('NATLANG_HELD_INFO_ATTEMPTS', '6'))
    for attempt in range(attempts):
        try:
            return json.loads(client.get_bytes('/v1/neuralese/info'))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
            time.sleep(wait)


def measure(out, manifest, endpoints, policy_id, *, samples=4, seed_base=SEED_BASE, case_seconds=3600,
            max_model_requests=EPISODE_MODEL_CALLS, max_turns=128, context_tokens=CONTEXT_TOKENS, label=None, allow_other_engine=False, log=print,
            top_p=None, order='longest-first', duration_priors=()):
    """Measure the policy served at ``endpoints`` ([(url, slots)]) on every task of ``manifest`` x ``samples`` into
    ``out/result.json`` (resumable: an existing ``run.json`` must name the same manifest, policy and sampling).
    ``top_p``: the nucleus of the sampling law (default ``common.sampling.RL_TOP_P``; 1.0: untruncated), recorded as
    the result's ``sampling``; ``compare`` refuses results of different laws. ``order``/``duration_priors``: the dispatch
    order (``Dispatch``; earlier result.json files of the same tasks seed its estimates), recorded as ``dispatch``."""
    from ..common.sampling import RL_TOP_P, sampling_law_of
    from .actor_client import NativeActorClient
    from ..common.code_provenance import code_provenance, repository_root
    out, manifest = Path(out), Path(manifest)
    for url, _ in endpoints:
        info = _read_info(NativeActorClient(url, timeout=10))
        binding = info.get('rl_policy_binding') or {}
        if policy_id_of(binding) != policy_id:
            raise ValueError(f'{url} serves {str(policy_id_of(binding))[:80]}, not {policy_id[:80]}')
    pinned = json.loads(manifest.read_text())
    repo = repository_root()
    spec = {'policy_id': policy_id, 'label': label, 'endpoint': endpoints[0][0], 'endpoints': [list(e) for e in endpoints],
            'dir': str(out.resolve()), 'manifest': str(manifest.resolve()),
            'cases': str(manifest.resolve().parent / pinned['cases']['path']), 'manifest_sha256': _sha256(manifest),
            'tasks': len(pinned['tasks']), 'samples': samples, 'seed_base': seed_base, 'workers': sum(n for _, n in endpoints),
            'case_seconds': case_seconds, 'max_model_requests': max_model_requests, 'max_turns': max_turns,
            'context_tokens': context_tokens, 'runtime': str(repo / 'ts-host'), 'repo': str(repo),
            'series': series(), 'code': code_provenance(), 'actor': {'info': info.get('rl_policy_binding')},
            'engine': engine_of(info.get('rl_policy_binding')),
            'sampling': sampling_law_of(RL_TOP_P if top_p is None else top_p),
            'dispatch': {'order': order, 'prior': [str(Path(p).resolve()) for p in duration_priors]}}
    pin = pinned_engine(manifest)
    if pin and not allow_other_engine:
        attested = spec['engine'] or {}
        wrong = {k: attested.get(k) for k, v in pin['expect'].items() if attested.get(k) != v}
        if wrong:
            raise ValueError(f'the actor does not serve the pinned eval engine ({Path(manifest).parent / "engine.json"}): {wrong}')
        harness = pin.get('harness') or {}
        if harness.get('context_tokens') not in (None, context_tokens):
            raise ValueError(f'the pinned harness budget is {harness["context_tokens"]} tokens, not {context_tokens}')
    served = info.get('context')
    if isinstance(served, int) and served < context_tokens and not allow_other_engine:
        raise ValueError(f'the actor serves {served} tokens per request, less than the call budget {context_tokens}; '
                         'relaunch it with --sglang-context >= the budget')
    if spec['code'].get('dirty') or spec['series']['ts_host_dirty'] or spec['series'].get('scoring_dirty'):
        log(json.dumps({'event': 'held eval warning', 'warning': 'the checkout is dirty; the result records it'}))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run.json').exists():
        previous = json.loads((out / 'run.json').read_text())
        previous.setdefault('context_tokens', LEGACY_CONTEXT_TOKENS)
        previous.setdefault('sampling', LEGACY_SAMPLING)
        for key in ('policy_id', 'manifest_sha256', 'samples', 'seed_base', 'case_seconds', 'max_model_requests', 'max_turns',
                    'context_tokens', 'sampling'):
            if previous[key] != spec[key]:
                raise ValueError(f'{out} holds a run with another {key}: {previous[key]!r}')
        # A resumed run must keep its engine: the result attests the first one, so episodes of a relaunched actor with
        # another KV dtype, projection or fork would be mixed in silently (compare pairs results by engine).
        if previous.get('engine') and spec['engine'] and previous['engine'] != spec['engine']:
            raise ValueError(f'{out} holds a run measured on another engine: {previous["engine"]!r}, '
                             f'the endpoint attests {spec["engine"]!r}')
        # The remaining episodes run on this checkout (its pin may be a newer one; the old one may be removed), and
        # resumed_code records it.
        spec = {**previous, 'endpoint': spec['endpoint'], 'endpoints': spec['endpoints'], 'workers': spec['workers'],
                'dispatch': spec['dispatch'], 'repo': spec['repo'], 'runtime': spec['runtime'],
                'resumed_code': spec['code']}
    (out / 'run.json').write_text(json.dumps(spec, indent=1) + '\n')
    (out / 'result.json').unlink(missing_ok=True)
    return run(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='measure the run.json of one policy directory (resumable)')
    r.add_argument('--dir', type=Path, required=True)
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
    p.add_argument('--avoid-groups', type=Path, action='append', default=[],
                   help='JSON list of split/source group names of exposed rows (rows outside --cases, e.g. other pools)')
    p.add_argument('--replace-ids', type=Path, action='append', default=[],
                   help='JSONL ({"id": ...}) or JSON list of --include-manifest tasks to replace (wrong/ambiguous oracle)')
    p.add_argument('--reject-ids', type=Path, action='append', default=[],
                   help='JSONL or JSON list of pool rows that may never be held tasks (pool errata); groups stay eligible')
    p.add_argument('--also-exclude', type=Path, action='append', default=[],
                   help='an earlier held manifest whose exclusion this one carries (exclusion_ids covers both)')
    p.add_argument('--dry-run', action='store_true')
    y = sub.add_parser('verify', help='check a pinned manifest against training exposure (ids and split/source groups)')
    y.add_argument('--manifest', type=Path, required=True)
    y.add_argument('--pool', type=Path,
                   help='relocated exact source pool; its SHA256 must match the original immutable manifest')
    y.add_argument('--avoid-ids', type=Path, action='append', default=[])
    y.add_argument('--avoid-groups', type=Path, action='append', default=[])
    y.add_argument('--avoid-from', type=Path, action='append', default=[])
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
    m.add_argument('--max-model-requests', type=int, default=EPISODE_MODEL_CALLS,
                   help='episode budget in model calls (default: the training budget, one for both)')
    m.add_argument('--max-turns', type=int, default=128)
    m.add_argument('--context-tokens', type=int, default=CONTEXT_TOKENS,
                   help='call budget (the pinned engine.json harness may require one; compare refuses other pairings)')
    m.add_argument('--label')
    m.add_argument('--top-p', type=float, help='nucleus of the sampling law at temperature 1 (default: common.sampling.'
                   'RL_TOP_P, 0.95; 1.0: untruncated). Recorded in the result; compare refuses pairs of different laws')
    m.add_argument('--actor-unit', help='ledger unit name of the actor (default heldv2-actor-<sha8>)')
    m.add_argument('--actor-budget-gb', type=int, default=ACTOR_BUDGET_GB, help='ledger claim: --sglang-kv-gb + 24')
    m.add_argument('--allow-other-engine', action='store_true', help='measure an actor that is not the pinned eval engine')
    m.add_argument('--actor-arg', action='append', default=[], help='extra serve_native_checkpoint.py argument (repeatable)')
    m.add_argument('--keep-actor', action='store_true')
    m.add_argument('--dispatch', choices=DISPATCH_ORDERS, default='longest-first',
                   help='episode order within a sample: the task expected to run longest first (the mean of its known '
                        'wall times: --duration-prior results and its episodes in this run), longest-first-max (the '
                        'previous estimate: its longest episode in this run, else the prior), or the sealed queue order; '
                        'never changes what is measured')
    m.add_argument('--duration-prior', type=Path, action='append', default=[],
                   help='an earlier result.json of the same held tasks (any policy): per-task wall seconds that order the '
                        'episodes (repeatable: every earlier result adds evidence)')
    v = sub.add_parser('serve', help='serve a checkpoint as a ledger actor unit (or reuse it) and print its URL')
    v.add_argument('--checkpoint', type=Path, required=True)
    v.add_argument('--checkpoint-sha256')
    v.add_argument('--store', type=Path, required=True)
    v.add_argument('--unit', required=True)
    v.add_argument('--budget-gb', type=int, default=ACTOR_BUDGET_GB, help='ledger claim: --sglang-kv-gb + 24')
    v.add_argument('--actor-arg', action='append', default=[])
    v.add_argument('--priority', default='medium', choices=('low', 'medium', 'high'))
    c = sub.add_parser('compare', help='paired comparison of two result.json files (second against first)')
    c.add_argument('first', type=Path)
    c.add_argument('second', type=Path)
    c.add_argument('--out', type=Path)
    c.add_argument('--ignore-ids', type=Path, action='append', default=[],
                   help='JSONL ({"id": ...}) or JSON list of held tasks to leave out of the without_ignored subset '
                        '(structurally-zero or wrong-oracle tasks, e.g. ops/held-v2/ignore-ids-*.jsonl)')
    c.add_argument('--trained-families', type=Path,
                   help='JSON list of families the second policy trained on: splits the change into trained/untrained')
    c.add_argument('--resamples', type=int, default=2000)
    c.add_argument('--strict-series', action='store_true',
                   help='pair only results of one scoring series and one attested engine (fork commit, running cap '
                        'included); by default other series pair with a warning and the differences recorded')
    g = sub.add_parser('merge', help='one result.json from complete runs of one policy with continuing seeds '
                                     '(e.g. samples 0-1 and, with --seed-base +2, samples 2-3)')
    g.add_argument('out', type=Path, help='result.json to write')
    g.add_argument('parts', type=Path, nargs='+', help="result.json files in seed order")
    args = parser.parse_args(argv)
    if args.command == 'run':
        run(args.dir)
    elif args.command == 'compare':
        report = compare(json.loads(args.first.read_text()), json.loads(args.second.read_text()), resamples=args.resamples,
                         trained_families=json.loads(args.trained_families.read_text()) if args.trained_families else None,
                         ignore_ids=read_ids(args.ignore_ids), strict_series=args.strict_series)
        if report['warnings']:
            print('held_eval compare WARNING: ' + '; '.join(report['warnings']), file=sys.stderr, flush=True)
        text = json.dumps(report, indent=1)
        if args.out:
            args.out.write_text(text + '\n')
        print(text)
    elif args.command == 'merge':
        result = merge([json.loads(path.read_text()) for path in args.parts])
        write_json_durable(args.out, result)
        print(json.dumps({k: result[k] for k in ('policy_id', 'samples_per_task', 'ran', 'scored', 'successes',
                                                 'success_rate', 'ran_out_rate')}))
    elif args.command == 'serve':
        sha = args.checkpoint_sha256 or _sha256(args.checkpoint)
        print(serve(args.checkpoint, sha, args.store, unit=args.unit, budget_gb=args.budget_gb, extra=args.actor_arg,
                    priority=args.priority,
                    log=lambda line: print(line, file=sys.stderr, flush=True)))
    elif args.command == 'measure':
        endpoints = [(item.rsplit('=', 1)[0].rstrip('/'), int(item.rsplit('=', 1)[1])) if re.search(r'=\d+$', item)
                     else (item.rstrip('/'), args.workers) for item in args.endpoint]
        unit = None
        if args.checkpoint:
            sha = args.checkpoint_sha256 or _sha256(args.checkpoint)
            unit = args.actor_unit or f'heldv2-actor-{sha[:8]}'
            url = serve(args.checkpoint, sha, args.out / 'actor', unit=unit, budget_gb=args.actor_budget_gb,
                        extra=args.actor_arg, actor_args=(pinned_engine(args.manifest) or {}).get('serve_args', ACTOR_ARGS))
            endpoints.append((url, args.workers))
        from .actor_client import NativeActorClient
        policy = args.policy_id or policy_id_of(json.loads(NativeActorClient(endpoints[0][0], timeout=10).get_bytes(
            '/v1/neuralese/info')).get('rl_policy_binding') or {})
        if not endpoints or not policy:
            raise SystemExit('give --endpoint or --checkpoint')
        try:
            measure(args.out, args.manifest, endpoints, policy, samples=args.samples, seed_base=args.seed_base,
                    case_seconds=args.case_seconds, max_model_requests=args.max_model_requests, max_turns=args.max_turns,
                    context_tokens=args.context_tokens, label=args.label, allow_other_engine=args.allow_other_engine,
                    top_p=args.top_p, order=args.dispatch, duration_priors=args.duration_prior)
        finally:
            if unit and not args.keep_actor:
                subprocess.run(['systemctl', '--user', 'stop', unit])
    elif args.command == 'pin':
        avoid = _avoid_set(args)
        include = [t['id'] for t in json.loads(args.include_manifest.read_text())['tasks']] if args.include_manifest else ()
        manifest, rows = build_manifest(args.cases, avoid=avoid, tasks=args.tasks, include=include, seed=args.seed,
                                        avoid_groups=_avoid_group_set(args), replace=read_ids(args.replace_ids),
                                        reject=read_ids(args.reject_ids))
        if args.include_manifest:
            manifest['included_manifest'] = {'path': str(args.include_manifest), 'sha256': _sha256(args.include_manifest)}
        inputs = [*args.avoid_ids, *args.avoid_groups, *args.replace_ids, *args.reject_ids]
        if inputs:
            manifest['pin_inputs'] = [{'path': str(path), 'sha256': _sha256(path)} for path in inputs]
        if args.avoid_from:
            manifest['avoided_from'] = [str(path) for path in args.avoid_from]
        if args.also_exclude:
            manifest['also_excludes'] = [{'path': str(path), 'sha256': _sha256(path)} for path in args.also_exclude]
        summary = {k: v for k, v in manifest.items() if k not in ('tasks', 'excluded_split_groups')}
        if args.dry_run:
            print(json.dumps(summary, indent=1))
            return
        sha = write_manifest(args.dir, manifest, rows)
        ids = sorted(exclusion_ids(args.dir / 'manifest.json'))
        (args.dir / 'exclusion-ids.json').write_text(json.dumps(ids, indent=0) + '\n')
        print(json.dumps({'manifest_sha256': sha, 'exclusion_ids': len(ids), **summary}, indent=1))
    elif args.command == 'verify':
        report = verify(args.manifest, _avoid_set(args), _avoid_group_set(args), pool_path=args.pool)
        print(json.dumps(report, indent=1))
        if report['overlap_ids'] or report['overlap_groups']:
            raise SystemExit(1)


def _avoid_set(args):
    avoid = set(read_ids(args.avoid_ids))
    for directory in args.avoid_from:
        state = json.loads((directory / 'state.json').read_text()) if (directory / 'state.json').exists() else {}
        avoid |= avoid_ids(state, [directory])
    return avoid


def _avoid_group_set(args):
    return set(read_ids(args.avoid_groups))


def verify(manifest_path, avoid, avoid_groups=(), *, pool_path=None):
    """Held tasks of a pinned manifest that training saw: ids in ``avoid``, or a split/source group shared with an avoided
    pool row or listed in ``avoid_groups``."""
    manifest = json.loads(Path(manifest_path).read_text())
    cases_path = Path(manifest_path).parent / manifest['cases']['path']
    cases_sha = _sha256(cases_path)
    if cases_sha != manifest['cases']['sha256']:
        raise ValueError('held cases changed since the manifest was pinned')
    actual_pool = Path(pool_path) if pool_path is not None else Path(manifest['pool']['path'])
    pool_sha = _sha256(actual_pool)
    if pool_sha != manifest['pool']['sha256']:
        raise ValueError('source pool differs from the original immutable manifest')
    rows = {row['id']: row for row in read_jsonl(cases_path)}
    groups = set(avoid_groups)
    for _, row in iter_jsonl_rows(actual_pool):
        if row.get('id') in avoid:
            groups.add(_split_group(row))
            groups |= _source_groups(row)
    held = [t['id'] for t in manifest['tasks']]
    overlap_groups = sorted(i for i in held if _split_group(rows[i]) in groups or _source_groups(rows[i]) & groups)
    return {'manifest': str(manifest_path), 'manifest_sha256': _sha256(manifest_path),
            'source_pool': {'original_path': manifest['pool']['path'], 'verified_path': str(actual_pool),
                            'sha256': pool_sha, 'relocated': actual_pool != Path(manifest['pool']['path'])},
            'cases_sha256': cases_sha, 'held': len(held), 'avoid_ids': len(avoid), 'avoid_groups': len(groups),
            'overlap_ids': sorted(set(held) & set(avoid)), 'overlap_groups': overlap_groups}

if __name__ == '__main__':
    main()
