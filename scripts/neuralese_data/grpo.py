#!/usr/bin/env python3
"""Plain GRPO for LFM3-1.7B (plans/neuralese/APPROACH_REVIEW_20261111.md section 4, "Hour 24-48").

One process runs the whole loop; nothing else (no controller, custody or transitions):

* rollouts: ``--slots`` collector slots run episodes (one ``run_bonsai_queue`` process each, the RL sampling capture) against
  the run's SGLang actor (``serve_native_checkpoint.py --engine sglang``, a child of this process). Tasks come shuffled from the pool
  minus the held set (``held_eval.exclusion_ids``); each task is a group of ``--group`` samples. Reward = task outcome
  (no teacher, no privileged twin, no Neuralese bonus, no efficiency shaping).
* overlap: collection never stops. The learner trains on the groups that finished since its last step (at least
  ``--min-groups``, at most ``--max-groups``); unfinished groups keep running and join a later step, so nothing waits on
  stragglers. A group sampled more than ``--max-staleness`` policy versions ago is dropped and counted.
* learner (the Experiment 0 model path, ``exp0_sft`` / ``lfm3_fast_train``): FP32 master weights, AdamW, BF16 compute. Every
  model call is (exact actor prompt ids, sampled ids) (``lower_rollout_turns.episode_turns``; calls of one invocation chain
  into one sequence as in ``exp0_sft.chains_of``). A consumer call's imported Neuralese state is fixed context read from the
  actor's block store and spliced into every layer as the actor's SGLang fork did (``lfm3_fast_train.import_tensors``);
  no gradient reaches the producer, whose own sampled tokens train as its own calls. Loss on sampled tokens only:
  ``-min(rho, C) * A * logp + beta * k3(ref || policy)``, ``rho = exp(logp_learner - logp_actor)`` (truncated importance
  sampling against the actor's own sampling logprobs), ``A`` the group-normalized episode reward on every call of the
  episode, the reference the start policy. Each trained episode weighs the same: its terms are averaged over its units
  (sampled tokens; a sampled decision counts as one average call of its episode), then over the step's episodes
  (``--loss-norm token``: one sum over all tokens). A tail token the actor sampled below ``--tail-logp-floor`` gets no
  policy term in an episode with a positive advantage. Sampling is top-p (``--top-p``, ``common.sampling.RL_TOP_P``): the
  policy term's ``logp`` is the sampled token's log-probability under the learner's own nucleus
  (``lfm3_fast_train.linear_nll_nucleus``; the sampled token always in the set), while ``rho`` and the KL term use the
  untruncated log-probabilities (the actor records the untruncated one; DECISIONS 2026-11-13 for the reasoning).
* weight sync: after each step the BF16 weights are copied once into a pinned host buffer, written as the actor's HF
  safetensors (same layout as ``export/native_hf``; frozen rows and producer table kept) and loaded in place
  (SGLang ``pause_generation`` retract -> ``update_weights_from_disk`` -> ``continue_generation``), in a background thread.
  Episodes keep running across a sync; each token's actor logprob is from the weights that sampled it.
* checkpoints: a pruned native checkpoint (``exp0_sft.write_checkpoint``, servable by ``serve_native_checkpoint``) every
  ``--save-every`` steps and the FP32 training state (masters + AdamW moments) every ``--state-every`` steps, both written
  from pinned host memory in background threads (training never waits on I/O). Keeps the last two native checkpoints plus
  every held-evaluated one, and the last training state.
* held evaluation: the start policy first, then every ``--held-every-hours`` the newest native checkpoint is measured on
  the held set (``python -m natlang_neuralese.rl.held_eval measure`` on the training actor, collection paused) and
  compared with the start policy (``held_eval compare`` -> ``held/step-N/compare-vs-step0.json``). With ``--held-remote pop``
  the measurements run on Pop's GPU instead, in parallel with training (``pop_held_loop``; the start policy as
  ``held/step-00000-pop``), and collection never pauses.
* task selection: ``--explore`` of groups from unmeasured pool tasks, the rest cycling tasks whose pass rate lies in
  ``--pass-band``: the run's own latest scored group of a task where it has one (current policy), else the measured base
  rate (``--pass-rates``). ``--neuralese-variants`` (off by default): a picked task that has a Neuralese-declaring variant
  (``scripts/rl_tasks/neuralese_variants.py``) runs as that variant in ``--neuralese-variant-share`` of its groups.

* pin: the run executes from its own pinned copy of one commit (``scripts/pin_checkout.py create COMMIT W/pin-<c12>``,
  then launch ``W/pin-<c12>/scripts/neuralese_data/grpo.py`` with that copy as workdir); every child (actor, episodes,
  held_eval measure/compare) gets the copy's PYTHONPATH/NATLANG_REPO, so all of them record one commit and series.

    grpo.py train --work W --checkpoint C [...]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import queue
import random
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

from natlang_neuralese.common.paths import resolve_str  # noqa: E402
from natlang_neuralese.common.sampling import RL_TOP_P, check_top_p, sampling_law_of  # noqa: E402
from natlang_neuralese.rl.held_eval import EPISODE_MODEL_CALLS, LEGACY_SAMPLING  # noqa: E402

BASE = str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
POOLS = [resolve_str('data_hdd', 'natlang-rl-composed-20261103', 'memetic-program-source-20261105', 'cases.jsonl'),
         # fallow K1: bgkit KR reader families, 4,734 rows verified by reference/negative/alias replay (verification.json)
         resolve_str('data_hdd', 'natlang-rl-bgkit-kr-20261103', 'natlang-rl-bgkit-kr-20261103', 'cases.jsonl')]
HELD = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'held-v2', 'manifest.json')
# Pool rows known to be wrong (an oracle's expected answer, a garbled task): every *.jsonl here is excluded by default.
ERRATA = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'pool-errata')


def errata_files(args):
    """The errata JSONL files a run excludes: ``--exclude-ids`` when given, else every ``ERRATA/*.jsonl`` (none with
    ``--no-default-errata``)."""
    if args.exclude_ids:
        return list(args.exclude_ids)
    if getattr(args, 'no_default_errata', False) or not Path(ERRATA).is_dir():
        return []
    return [str(path) for path in sorted(Path(ERRATA).glob('*.jsonl'))]


def drop_cache(paths):
    """Drop the page cache of files read once. It is charged to this unit's cgroup, and on the GB10 a CUDA allocation or
    context creation near the cgroup's limit fails (CUDA_ERROR_OUT_OF_MEMORY) instead of reclaiming it: every GRPO smoke
    OOM of 2026-11-12 happened right after reading the 9.5 GB checkpoint."""
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def actor_generated(url):
    """The actor front's generated-token counter (``/metrics`` natlang_sglang_generated_tokens_total), None if unreadable."""
    try:
        with urllib.request.urlopen(url + '/metrics', timeout=10) as response:
            for line in response.read().decode().splitlines():
                if line.startswith('natlang_sglang_generated_tokens_total '):
                    return int(float(line.split()[1]))
    except Exception:
        return None
    return None


def meminfo():
    fields = dict(line.split(':', 1) for line in open('/proc/meminfo'))
    return {k: round(int(fields[k].split()[0]) / 2**20, 1) for k in ('MemFree', 'MemAvailable', 'Cached', 'SwapFree')}


def vmstat():
    """Cumulative swap traffic (pages): a learner phase that swaps is memory-bound, not compute-bound."""
    fields = dict(line.split() for line in open('/proc/vmstat'))
    return {k: int(fields[k]) for k in ('pswpin', 'pswpout')}


def log_line(handle, record):
    text = json.dumps(record)
    print(text, flush=True)
    if handle is not None:
        handle.write(text + '\n')
        handle.flush()


def post(url, body=None, timeout=600):
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def git_pin():
    from natlang_neuralese.common.code_provenance import code_provenance
    return code_provenance()


def pinned_env():
    """The environment every child of the run gets (collector episodes, the actor, held_eval measure/compare): this
    process's own checkout (``REPO``: the run's pinned copy, ``scripts/pin_checkout.py``) on PYTHONPATH and NATLANG_REPO,
    so their code, ts-host and recorded commit/series are the run's, whatever main does meanwhile."""
    return {**os.environ, 'NATLANG_REPO': str(REPO), 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}'}


def check_pin(work, commit, unpinned):
    """A run executes from ``WORK/pin-<commit12>`` (a detached ``pin_checkout.py`` copy); ``--unpinned`` (smokes) runs
    the live checkout."""
    marker = REPO / '.natlang-pin.json'
    if unpinned:
        return None
    if not marker.exists() or json.loads(marker.read_text())['commit'] != commit:
        raise SystemExit(f'{REPO} is not a pinned copy of {commit[:12]}: launch from '
                         f'`scripts/pin_checkout.py create <commit> {Path(work) / "pin-<commit12>"}` (or pass --unpinned)')
    if REPO.resolve().parent != Path(work).resolve():
        print(f'note: the pin {REPO} lives outside the work dir {work}', file=sys.stderr)
    return str(REPO)


# ---- rollouts --------------------------------------------------------------------------------------------------------
class Collector:
    """Work-stealing slots over an endless stream of task groups; finished groups are scored and queued for the learner."""

    def __init__(self, args, tasks, work, policy_sha, runtime, first_group=0, cursor=0):
        self.args, self.tasks, self.work, self.policy_sha, self.runtime = args, tasks, work, policy_sha, runtime
        self.lock = threading.Lock()
        self.pending = collections.deque()          # entries not yet taken by a slot
        self.groups = {}                            # group -> {'dir', 'size', 'done', 'versions', 'task_id'}
        self.ready = queue.Queue()                  # finished, scored groups
        self.outstanding = 0                        # groups created and not yet handed to the learner
        self.next_group, self.cursor = first_group, cursor
        self.version = 0                            # the policy version the actor serves now
        self.by_id = {row['id']: row for row in tasks}
        self.learnable_ids, self.learnable_order, self.measured, self.rates_read = [], [], set(), 0.
        self.base_rates, self.own, self.rates_changed = {}, {}, True
        self.variants = {}                          # original task id -> its Neuralese-declaring variant row
        for directory in sorted((work / 'groups').glob('g*')):   # a resumed run keeps its own measurements
            self._record_outcome(directory)
        self.rng = random.Random(args.seed + first_group)
        self.stop = threading.Event()
        self.paused = threading.Event()             # set during a held measurement on the training actor
        self.active_slots = args.slots
        self.generated = 0                          # sampled tokens of finished episodes (throughput)
        self.episodes = 0
        self.env = pinned_env()
        from natlang_neuralese.rl.held_eval import series
        self.env['NATLANG_MODEL_FACING'] = series()['model_facing']
        from lower_rollout_turns import NativeStates
        self.states = NativeStates(args.actor_store or work / 'actor')   # imported Neuralese state of consumer calls

    def _new_group(self):
        from natlang_neuralese.rl import task_rollouts
        from natlang_neuralese.common.jsonio import read_jsonl
        row = self._pick()
        variant = self.variants.get(row['id'])
        if variant is not None and self.rng.random() < self.args.neuralese_variant_share:
            row = variant   # its outcomes are its own (never the original's band rate)
        group = self.next_group
        self.next_group += 1
        directory = self.work / 'groups' / f'g{group:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'cases.jsonl').write_text(json.dumps(row) + '\n')
        a = self.args
        task_rollouts.prepare(SimpleNamespace(
            cases=directory / 'cases.jsonl', out=directory, rows=1, offset=0, group=a.group, temperature=1.,
            case_seconds=a.case_seconds, max_model_requests=a.max_model_requests, max_turns=a.max_turns,
            seed_base=a.seed_base + 100 * group, root_mode='student', match_seeds_from=None,
            policy_checkpoint_sha256=self.policy_sha))
        entries = list(read_jsonl(directory / 'queue.jsonl'))
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=a.context_tokens, rl_endpoint=a.actor)
        (directory / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
        self.groups[group] = {'dir': directory, 'size': len(entries), 'done': 0, 'versions': [], 'task_id': row['id'],
                              'created': time.time()}
        self.outstanding += 1
        self.pending.extend((group, entry) for entry in entries)

    def _record_outcome(self, directory):
        """The task's latest scored group (``--group`` samples of the policy then served) replaces any earlier measurement
        of it: the selection follows the current policy's own outcomes, so a task it solves 0/8 or 8/8 leaves the band and
        an explored task it solves sometimes enters it."""
        from natlang_neuralese.common.jsonio import read_jsonl
        try:
            rows = list(read_jsonl(directory / 'rewards.jsonl'))
        except OSError:
            return
        scored = [row for row in rows if row.get('reward') is not None]
        if scored:
            self.own[scored[0]['task_id']] = (sum(row['reward'] == 1.0 for row in scored), len(scored))
            self.rates_changed = True

    def _learnable(self):
        """Pool rows whose pass rate lies in ``--pass-band`` with at least ``--pass-min-scored`` scored samples, shuffled:
        the run's own latest group of a task (``_record_outcome``) where it has one, else the measured base rate
        (``--pass-rates``, re-read every 10 min)."""
        a = self.args
        if a.pass_rates and time.time() - self.rates_read >= 600:
            self.rates_read = time.time()
            try:
                self.base_rates = json.loads(Path(a.pass_rates).read_text())
                self.rates_changed = True
            except (OSError, ValueError):
                pass
        if not self.rates_changed:
            return
        self.rates_changed = False
        rates = {k: (v['successes'], v['scored']) for k, v in self.base_rates.items() if v.get('scored')}
        rates.update(self.own)
        low, high = a.pass_band
        ids = {k for k, (successes, scored) in rates.items()
               if k in self.by_id and scored >= a.pass_min_scored and low <= successes / scored <= high}
        self.measured = set(self.base_rates) | set(self.own)
        old = set(self.learnable_ids)
        if ids != old:
            self.learnable_ids = sorted(ids)
            self.learnable_order = [k for k in self.learnable_order if k in ids]
            for k in sorted(ids - old):
                self.learnable_order.insert(self.rng.randint(0, len(self.learnable_order)), k)
            print(json.dumps({'event': 'learnable tasks', 'count': len(ids), 'measured': len(self.measured),
                              'own_measured': len(self.own), 'own_in_band': len(ids & set(self.own))}), flush=True)

    def _pick(self):
        """A learnable task (``1 - explore`` of groups, cycling a reshuffled list), else the next unmeasured pool task."""
        self._learnable()
        if self.learnable_ids and self.rng.random() >= self.args.explore:
            if not self.learnable_order:
                self.learnable_order = list(self.learnable_ids)
                self.rng.shuffle(self.learnable_order)
            return self.by_id[self.learnable_order.pop()]
        while True:
            row = self.tasks[self.cursor % len(self.tasks)]
            self.cursor += 1
            if row['id'] not in self.measured or self.cursor > 2 * len(self.tasks):
                return row

    def _take(self):
        if self.paused.is_set():
            return None
        with self.lock:
            if not self.pending and self.outstanding < self.args.max_outstanding:
                self._new_group()
            return self.pending.popleft() if self.pending else None

    def slot(self, index=0):
        sampling = self.work / 'sampling-config.json'
        while not self.stop.is_set():
            if index >= self.active_slots:   # fewer slots while the learner computes (they share the GPU)
                time.sleep(2)
                continue
            item = self._take()
            if item is None:
                time.sleep(2)
                continue
            group, entry = item
            info = self.groups[group]
            workers = info['dir'] / 'workers'
            workers.mkdir(exist_ok=True)
            key = entry['key']
            queue_file, journal = workers / f'{key}.queue.jsonl', workers / f'{key}.journal.jsonl'
            queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
            started = self.version
            command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue_file), str(journal),
                       '--runtime', str(self.runtime), '--server', self.args.actor,
                       '--chat-completions-url', self.args.actor + '/v1/chat/completions', '--model-id', 'natlang-rl',
                       '--model-concurrency', '1', '--chat-request-config', str(sampling), '--neuralese-server']
            with (workers / f'{key}.log').open('a') as handle:
                process = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                while process.poll() is None:
                    if self.stop.wait(2):
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
                        return
            if process.returncode < 0 or self.stop.is_set():
                self._abandon(group, key, process.returncode)
                if self.stop.is_set():
                    return
                continue
            with self.lock:
                if info.get('abandoned'):   # a sibling was interrupted: this group is never scored
                    continue
                info['done'] += 1
                info['versions'].append(started)
                complete = info['done'] == info['size']
            if complete:
                try:
                    self._score(group)
                except Exception as error:   # a scoring failure loses one group, never the loop
                    print(json.dumps({'event': 'group scoring failed', 'group': group, 'error': repr(error)[:300]}),
                          flush=True)
                    with self.lock:
                        self.outstanding -= 1
                        # no longer outstanding: its creation time must not pin the block GC's cutoff for the whole run
                        self.groups.pop(group, None)

    def _abandon(self, group, key, returncode):
        """An episode killed by a signal (a unit stop signals every process of the cgroup, so the episodes die before this
        process's SIGTERM handler runs while the learner holds the main thread; or the kernel's OOM killer) has no verdict.
        Counting it as done scored its group with the killed episodes as ``unresolved`` and the finished ones as the group:
        a subset biased to short episodes that the stopping learner could still train on and that fed the pass band
        (127 of run2's 131 unresolved episodes were such restart casualties; none had reached the wall clock). Its group
        is dropped whole: never scored, its unstarted entries withdrawn, running siblings discarded when they end."""
        with self.lock:
            info = self.groups.get(group)
            if info is None or info.get('abandoned'):
                return
            info['abandoned'] = True
            self.pending = collections.deque(item for item in self.pending if item[0] != group)
            self.groups.pop(group, None)
            self.outstanding -= 1
        print(json.dumps({'event': 'group abandoned', 'group': group, 'episode': key, 'returncode': returncode,
                          'stopping': self.stop.is_set()}), flush=True)

    def _score(self, group):
        from natlang_neuralese.rl import task_rollouts
        info = self.groups[group]
        directory = info['dir']
        with (directory / 'journal.jsonl').open('w') as merged:
            for path in sorted((directory / 'workers').glob('*.journal.jsonl')):
                merged.write(path.read_text())
        task_rollouts.score(directory, directory / 'rollouts.jsonl', neuralese_bonus=0., efficiency_weight=0.)
        (directory / 'versions.json').write_text(json.dumps(info['versions']) + '\n')
        with self.lock:
            self._record_outcome(directory)
        self.ready.put(group)

    def take_groups(self, minimum, maximum, stop=None):
        """At least ``minimum`` finished groups (waits, unless ``stop`` is set), at most ``maximum``."""
        taken = []
        while len(taken) < minimum:
            try:
                taken.append(self.ready.get(timeout=5))
            except queue.Empty:
                if stop is not None and stop.is_set():
                    break
        while len(taken) < maximum:
            try:
                taken.append(self.ready.get_nowait())
            except queue.Empty:
                break
        with self.lock:
            self.outstanding -= len(taken)
        return taken


def trainable_rows(rows, stats, top_p=1.):
    """The calls whose actor log-probabilities are the law the learner scores: every sampled token has a finite logp
    (a call captured without sampling records None, which would turn rho, the loss and then every FP32 master into NaN)
    and the call sampled at temperature 1 without guidance from the learner's own nucleus (``top_p``, the run's
    ``--top-p``; the capture's sampling law names the draw's; 1.0 on both sides: untruncated). A call of another law is
    refused and counted (``skip_other_sampling_law``)."""
    kept = []
    for row in rows:
        if not np.isfinite(row['logp']).all():
            stats['skip_nonfinite_actor_logp'] += 1
        elif (row.get('temperature') not in (None, 1, 1.0) or row.get('guidance')
              or abs(float(row.get('top_p', 1.)) - top_p) > 1e-9):
            stats['skip_other_sampling_law'] += 1
        else:
            kept.append(row)
    return kept


def load_group(collector, group, stats):
    """(task_id, [episode dicts with 'reward', 'rows'], versions) of one scored group."""
    from natlang_neuralese.common.jsonio import read_jsonl
    from lower_rollout_turns import decision_turns, episode_turns
    info = collector.groups[group]
    episodes = []
    for row in read_jsonl(info['dir'] / 'rollouts.jsonl'):
        reward = row.get('reward')
        rows = (trainable_rows(episode_turns(row, stats, collector.states), stats, collector.args.top_p)
                if row.get('turns') else [])
        decisions = decision_turns(row, stats, collector.states) if row.get('turns') else []
        sampled = sum(len(r['sample']) for r in rows)
        collector.generated += sampled
        collector.episodes += 1
        if reward is None or row.get('unscored'):
            stats['episodes_unscored'] += 1
            continue
        quality = float(reward['quality'])
        graded = reward.get('graded')
        binary = getattr(getattr(collector, 'args', None), 'reward', 'graded') == 'binary'
        episodes.append({'reward': quality if binary or graded is None else float(graded),
                         'quality': quality, 'rows': rows, 'decisions': decisions,
                         'task_id': row['task_id'], 'round': group,
                         'outcome': row.get('outcome')})
    return info['task_id'], episodes, info['versions']


def collect_blocks(store_dir, cutoff):
    """Delete the actor store's native-state block files last used before ``cutoff`` (epoch s); (files, bytes) removed.

    The learner reads a block only while lowering a group whose episodes imported it, and the actor touches a block file
    whenever it stores or reads that block (``serve/store.py``), so a block untouched since before the oldest group still
    outstanding (created, in flight or scored and not yet taken) was used only by episodes already trained, dropped or
    measured. The actor's store forgets a row whose file is gone and rewrites the file if the block recurs."""
    removed = freed = 0
    for path in Path(store_dir, 'blocks').glob('*/nzs1_*.safetensors'):   # native states only, never tensor artifacts
        try:
            stat = path.stat()
            if stat.st_mtime < cutoff:
                path.unlink()
                removed += 1
                freed += stat.st_size
        except FileNotFoundError:
            pass
    return removed, freed


def archive_rollouts(directory, archive):
    """``directory/rollouts.jsonl`` (every episode the learner lowered: turns with their exact sampling captures, decision
    receipts with option ids and actor scores) zstd-compressed to ``archive/<group>.rollouts.jsonl.zst`` (temp file renamed).
    Returns the compressed bytes."""
    import zstandard
    source = directory / 'rollouts.jsonl'
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f'{directory.name}.rollouts.jsonl.zst'
    partial = target.with_suffix('.zst.partial')
    with source.open('rb') as reader, partial.open('wb') as writer:
        zstandard.ZstdCompressor(level=6).copy_stream(reader, writer)
    os.replace(partial, target)
    return target.stat().st_size


def prune_group(collector, group, keep, archive=None, stats=None):
    """Forget a lowered group and delete its bulky files, keeping its score/rewards/versions/cases; its rollouts are
    first kept compressed in ``archive`` (a group whose archiving fails keeps everything in place)."""
    directory = collector.groups.pop(group)['dir']
    if keep:
        return
    if archive is not None and (directory / 'rollouts.jsonl').exists():
        try:
            size = archive_rollouts(directory, archive)
        except Exception as error:
            print(json.dumps({'event': 'rollout archive failed', 'group': group, 'error': repr(error)[:300]}), flush=True)
            if stats is not None:
                stats['rollout_archive_failed'] += 1
            return
        if stats is not None:
            stats['rollouts_archived'] += 1
            stats['rollouts_archived_bytes'] += size
    for path in directory.iterdir():
        if path.name not in ('score.json', 'rewards.jsonl', 'versions.json', 'cases.jsonl'):
            shutil.rmtree(path) if path.is_dir() else path.unlink()


# ---- host I/O --------------------------------------------------------------------------------------------------------
def safetensors_header(layout, metadata=None):
    """layout: [(name, dtype_str, shape, nbytes)] -> (header bytes, offsets)."""
    header, offset, offsets = {}, 0, []
    for name, dtype, shape, nbytes in layout:
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset + nbytes]}
        offsets.append(offset)
        offset += nbytes
    if metadata:
        header['__metadata__'] = {k: str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    return struct.pack('<Q', len(raw)) + raw, offsets, offset


class WeightSync:
    """The actor's HF export in one pinned host buffer: snapshot (GPU -> pinned, the only synchronous part), then in a
    background thread write it, load it into SGLang in place, and optionally write a native checkpoint."""

    def __init__(self, model, export_dir, sglang_url, weights_dir, front_url=None):
        import torch
        from safetensors import safe_open
        self.export_dir, self.sglang_url = Path(export_dir), sglang_url
        self.weights_dir = Path(weights_dir)
        self.front_url = front_url
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        for path in self.export_dir.iterdir():
            if path.is_file() and path.name != 'model.safetensors':
                shutil.copyfile(path, self.weights_dir / path.name)
        params = dict(model.named_parameters())
        layout, sources = [], []
        with safe_open(str(self.export_dir / 'model.safetensors'), 'pt') as handle:
            fixed = {}
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f'export tensor {name} is {tensor.dtype}, expected bf16')
                if name in params:
                    if tuple(params[name].shape) != tuple(tensor.shape):
                        raise ValueError(f'export/learner shape mismatch {name}')
                    sources.append(name)
                else:
                    fixed[name] = tensor            # the producer table and anything else the learner does not train
                    sources.append(None)
                layout.append((name, 'BF16', tuple(tensor.shape), tensor.numel() * 2))
        missing = set(params) - {name for name in sources if name} - {'lm_head.weight'}
        if missing:
            raise ValueError(f'learner parameters absent from the export: {sorted(missing)[:5]}')
        self.header, offsets, total = safetensors_header(layout, {'format': 'pt'})
        self.buffer = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        self.views = {}
        for (name, _, shape, nbytes), offset, source in zip(layout, offsets, sources):
            view = self.buffer[offset:offset + nbytes].view(torch.bfloat16).view(shape)
            if source is None:
                view.copy_(fixed[name])
            else:
                self.views[name] = view
        self.params = params
        self.thread = None
        self.last = {}

    def snapshot(self):
        import torch
        self.wait()
        for name, view in self.views.items():
            view.copy_(self.params[name].detach(), non_blocking=True)
        torch.cuda.synchronize()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def load_native(self, path, special):
        """Put another native checkpoint's backbone in the pinned buffer (its interface rows merged into the embedding
        as ``exp0_sft.load_model`` does); the actor's other tables (producer embedding) stay. Refuses another interface."""
        import torch
        from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
        self.wait()
        state = CheckpointPolicy.load(Path(path))
        backbone = state['backbone_state']
        embedding = backbone['model.embed_tokens.weight'].clone()
        interface, config = state['interface_state'], state['model_config']
        rows = [(config['marker_ids'], interface['input_rows'])]
        if config.get('token_override_ids'):
            rows.append((config['token_override_ids'], interface['extra_input_rows']))
        for ids, values in rows:
            embedding[torch.tensor(list(ids))] = values.to(embedding.dtype)
        merged = self.views['model.embed_tokens.weight'][torch.tensor(special)]
        if not torch.equal(embedding[torch.tensor(special)].to(merged.dtype), merged):
            raise ValueError(f'{path} has other interface rows than the served policy')
        for name, view in self.views.items():
            source = embedding if name == 'model.embed_tokens.weight' else backbone[name]
            view.copy_(source.to(view.dtype))
        del state

    def backbone(self):
        """{name: host bf16 tensor} of the snapshot in native backbone naming (copies, safe to keep)."""
        return {name: view.clone() for name, view in self.views.items()}

    def publish(self, version, on_done, also=None, policy=None):
        def run():
            # An exception here used to end the thread silently: ``last`` kept the previous step's ok, so the loop went
            # on as if the actor had been updated and attested. It is now this version's failure (the loop stops on it).
            try:
                body()
            except Exception as error:
                self.last = {'version': version, 'ok': False, 'answer': f'publish failed: {error!r}'[:400]}
                on_done(version, False)

        def body():
            timing = {}
            started = time.time()
            partial = self.weights_dir / 'model.safetensors.partial'
            with partial.open('wb') as handle:
                handle.write(self.header)
                handle.write(memoryview(self.buffer.numpy()))
            os.replace(partial, self.weights_dir / 'model.safetensors')
            timing['export_write_s'] = round(time.time() - started, 2)
            t = time.time()
            if self.sglang_url is None:   # --no-actor-sync (a smoke against a shared actor): nothing to update
                answer, ok = 'no actor sync', True
            else:
              try:
                post(self.sglang_url + '/pause_generation', {'mode': 'retract'})
                answer = post(self.sglang_url + '/update_weights_from_disk',
                              {'model_path': str(self.weights_dir), 'flush_cache': True, 'weight_version': str(version)})
                post(self.sglang_url + '/continue_generation', {})
                ok = json.loads(answer).get('success', False)
              except Exception as error:
                answer, ok = repr(error)[:400], False
                try:
                    post(self.sglang_url + '/continue_generation', {})
                except Exception:
                    pass
            timing['actor_update_s'] = round(time.time() - t, 2)
            extra = {}
            if also is not None:
                t = time.time()
                try:
                    extra = also() or {}
                except Exception as error:   # the actor already serves the new weights: attest them, report the write
                    extra = {'checkpoint_error': repr(error)[:400]}
                timing.update(extra)
                timing['checkpoint_s'] = round(time.time() - t, 2)
            if ok and self.front_url is not None:
                # the actor front attests what it now serves: the step's native checkpoint when one was written, else the
                # served weights' digest (held_eval measure and the rollout captures read this binding)
                t = time.time()
                weights = hashlib.sha256(memoryview(self.buffer.numpy())).hexdigest()
                policy_id = policy or extra.get('checkpoint_sha256') or weights
                try:
                    post(self.front_url + '/v1/neuralese/rebind-policy',
                         {'policy_id': policy_id, 'weights_sha256': weights, 'version': version})
                    timing['policy_id'] = policy_id
                except Exception as error:
                    ok, answer = False, f'rebind failed: {error!r}'[:400]
                timing['rebind_s'] = round(time.time() - t, 2)
            self.last = {'version': version, 'ok': ok, 'answer': None if ok else answer, **timing}
            on_done(version, ok)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


class StateSaver:
    """FP32 masters and AdamW moments streamed GPU -> pinned staging -> file in a background thread. The caller waits
    (``wait``) before the next optimizer step changes them."""

    def __init__(self, staging_mb=512):
        import torch
        self.staging = torch.empty(staging_mb << 20, dtype=torch.uint8, pin_memory=True)
        self.stream = torch.cuda.Stream()
        self.thread = None
        self.last = None

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def save(self, path, tensors, metadata):
        import torch
        self.wait()
        torch.cuda.synchronize()
        names = list(tensors)
        layout = [(name, 'F32', tuple(tensors[name].shape), tensors[name].numel() * 4) for name in names]
        header, _, _ = safetensors_header(layout, metadata)

        def run():
            started = time.time()
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix('.partial')
            chunk = self.staging.numel()
            with partial.open('wb') as handle, torch.cuda.stream(self.stream):
                handle.write(header)
                for name in names:
                    flat = tensors[name].detach().reshape(-1).view(torch.uint8)
                    for start in range(0, flat.numel(), chunk):
                        piece = flat[start:start + chunk]
                        self.staging[:piece.numel()].copy_(piece, non_blocking=True)
                        self.stream.synchronize()
                        handle.write(memoryview(self.staging[:piece.numel()].numpy()))
            os.replace(partial, path)
            self.last = {'path': str(path), 'seconds': round(time.time() - started, 1)}
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


# ---- learner ---------------------------------------------------------------------------------------------------------
def write_pruned_checkpoint(state, native_rows, special, path, source_sha256, trained):
    """``exp0_sft.write_checkpoint``'s pruned native checkpoint from a host snapshot, without holding the source backbone:
    the trained backbone (BF16), the source's native embedding rows at the interface ids, the source's interface."""
    import torch
    backbone = {name: trained[name].contiguous() for name in native_rows['names']}
    embedding = backbone['model.embed_tokens.weight']
    embedding[torch.tensor(special)] = native_rows['embedding_rows'].to(embedding.dtype)
    if native_rows['tied_head']:
        backbone['lm_head.weight'] = embedding
    out = {k: state[k] for k in ('schema', 'model_config', 'interface_state')}
    out.update(backbone_state=backbone, step=int(state['step']), inference_only=True, optimizer_pruned=True,
               optimizer_pruned_keys=['training'],
               inference_source={'schema': 'natlang.native-inference-source/1', 'training_checkpoint_sha256': source_sha256,
                                 'training_checkpoint_bytes': 0, 'step': int(state['step'])})
    torch.save(out, path)
    return None


def decision_tokens(row):
    """Packed tokens of one decision: its prompt once per option plus the option's continuation."""
    return sum(len(row['prompt']) + len(o) for o in row['options'])


def decision_record(item, learner, actor, reference, log_rho):
    """One trained decision's diagnostics (``WORK/decisions/step-N.jsonl``): the actor's captured option scores and ids
    beside the learner's and the start policy's scores of the same options. ``dscore_centered`` (learner - actor per option,
    minus its mean over the options) is what moves log pi(chosen); ``ref_log_rho`` (start policy vs behaviour) isolates
    learner/actor numerics from training drift when the decision was sampled by the start policy (``stale == step``)."""
    row = item['row']
    temperature, chosen = row['temperature'], row['chosen']
    learner, actor = np.asarray(learner, np.float64), np.asarray(actor, np.float64)

    def log_pi(scores):
        z = scores / temperature
        z = z - z.max()
        return float(z[chosen] - np.log(np.exp(z).sum()))
    dscore = learner - actor
    record = {'group': item.get('group'), 'task_id': item.get('task_id'), 'stale': item.get('stale'),
              'actor_version': item.get('actor_version'), 'policy': row.get('policy'), 'invocation': row.get('invocation'),
              'turn': row.get('index'), 'advantage': round(float(item['advantage']), 4), 'temperature': temperature,
              'chosen': chosen, 'options': len(row['options']), 'prompt_tokens': len(row['prompt']),
              'prompt_sha256': hashlib.sha256(np.asarray(row['prompt'], np.int32).tobytes()).hexdigest()[:16],
              'imports': len(row.get('imports') or ()), 'option_tokens': [len(o) for o in row['options']],
              'option_ids': [np.asarray(o).tolist() for o in row['options']],
              'actor_scores': [round(float(v), 4) for v in actor], 'learner_scores': [round(float(v), 4) for v in learner],
              'behavior_logq': round(float(row['behavior_logq']), 5), 'actor_logq_recomputed': round(log_pi(actor), 5),
              'learner_logpi': round(log_pi(learner), 5), 'log_rho': round(float(log_rho), 5),
              'dscore_abs_mean': round(float(np.abs(dscore).mean()), 4),
              'dscore_centered_abs_max': round(float(np.abs(dscore - dscore.mean()).max()), 4)}
    if reference is not None:
        reference = np.asarray(reference, np.float64)
        record.update(ref_scores=[round(float(v), 4) for v in reference], ref_logpi=round(log_pi(reference), 5),
                      ref_log_rho=round(log_pi(reference) - float(row['behavior_logq']), 5))
    return record


def decision_summary(records):
    """Per-step decision diagnostics for train.jsonl: |log rho| distribution (all, by staleness), learner/actor and start
    policy/actor gaps, option counts and lengths."""
    if not records:
        return {}
    abs_rho = np.abs([r['log_rho'] for r in records])
    q = lambda values, p: round(float(np.quantile(values, p)), 4)
    by_stale = collections.defaultdict(list)
    for r, value in zip(records, abs_rho):
        by_stale[r['stale']].append(value)
    worst = max(records, key=lambda r: abs(r['log_rho']))
    summary = {'n': len(records), 'abs_log_rho_max': round(float(abs_rho.max()), 4), 'abs_log_rho_p50': q(abs_rho, .5),
               'abs_log_rho_p90': q(abs_rho, .9), 'abs_log_rho_p99': q(abs_rho, .99),
               'abs_log_rho_over_0.5': int((abs_rho > .5).sum()),
               'abs_log_rho_mean_by_stale': {str(k): round(float(np.mean(v)), 4) for k, v in sorted(by_stale.items())},
               'dscore_centered_abs_max': round(max(r['dscore_centered_abs_max'] for r in records), 4),
               'options_mean': round(float(np.mean([r['options'] for r in records])), 2),
               'options_max': max(r['options'] for r in records),
               'prompt_tokens_mean': round(float(np.mean([r['prompt_tokens'] for r in records]))),
               'prompt_tokens_max': max(r['prompt_tokens'] for r in records),
               'option_tokens_mean': round(float(np.mean([t for r in records for t in r['option_tokens']])), 1),
               'option_tokens_max': max(t for r in records for t in r['option_tokens']),
               'with_imports': sum(1 for r in records if r['imports']),
               'worst': {k: worst.get(k) for k in ('group', 'task_id', 'stale', 'options', 'prompt_tokens', 'option_tokens',
                                                   'log_rho', 'ref_log_rho', 'dscore_centered_abs_max')}}
    if all('ref_log_rho' in r for r in records):
        ref = np.abs([r['ref_log_rho'] for r in records])
        summary.update(ref_abs_log_rho_mean=round(float(ref.mean()), 4), ref_abs_log_rho_max=round(float(ref.max()), 4))
    return summary


def decision_batches(items, max_tokens):
    """Pack sampled decisions (``lower_rollout_turns.decision_turns`` rows with their advantage) into forward batches: each
    option is one document (exact prompt + option continuation), so a decision of K options is K packed documents."""
    batches, batch, size = [], [], 0
    for item in items:
        n = decision_tokens(item['row'])
        if batch and size + n > max_tokens:
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += n
    return batches + ([batch] if batch else [])


def decision_tensors(batch, device):
    """(ids, lengths, positions, targets, option index per target, options per decision, imported state spans
    [(packed row start, block_id)]) of a decision batch."""
    import torch
    ids, lengths, positions, targets, owner, counts, offset, option, imports = [], [], [], [], [], [], 0, 0, []
    for item in batch:
        row = item['row']
        counts.append(len(row['options']))
        for continuation in row['options']:
            prompt = len(row['prompt'])
            imports += [(offset + start, block_id) for start, block_id in row.get('imports') or ()]
            ids += [row['prompt'], continuation]
            lengths.append(prompt + len(continuation))
            positions.append(np.arange(offset + prompt - 1, offset + prompt + len(continuation) - 1))
            targets.append(continuation)
            owner.append(np.full(len(continuation), option))
            offset += prompt + len(continuation)
            option += 1
    as_long = lambda parts: torch.from_numpy(np.concatenate(parts).astype(np.int64)).to(device)
    return as_long(ids), lengths, as_long(positions), as_long(targets), as_long(owner), counts, imports


def group_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std < 1e-6:
        return None
    return [(r - mean) / (std + 1e-6) for r in rewards]


def decision_unit(tokens, calls, mode):
    """Units one sampled decision counts in its episode's normalization. ``call``: the episode's mean sampled tokens per
    trained call (a decision is one model call whose whole reply is one categorical action; 1 when the episode trains
    no token call); ``token``: 1, one sampled token's weight."""
    if mode == 'token' or not calls:
        return 1.
    return max(1., tokens / calls)


def loss_scales(episodes, norm, decision_weight):
    """Per-episode loss scale factors: ``episodes`` {key: (trained tokens, trained calls, trained decisions)} ->
    {key: (factor of each token term, factor of each decision term)}. An episode's units are its tokens plus its
    decisions times ``decision_unit``. ``episode``: every episode with units weighs 1 / episodes and its terms share
    that weight by units (A_i times the mean over its units, averaged over episodes; a chain's factor is
    1 / (episode units * episodes), so accumulating chain by chain is exact). ``token``: one factor 1 / all units."""
    units = {key: (tokens + decisions * decision_unit(tokens, calls, decision_weight),
                   decision_unit(tokens, calls, decision_weight))
             for key, (tokens, calls, decisions) in episodes.items()}
    trained = sum(1 for total, _ in units.values() if total > 0)
    total_units = sum(total for total, _ in units.values())
    scales = {}
    for key, (total, unit) in units.items():
        if total <= 0:
            scales[key] = (0., 0.)
        elif norm == 'episode':
            scales[key] = (1. / (total * trained), unit / (total * trained))
        else:
            scales[key] = (1. / total_units, unit / total_units)
    return scales


def policy_weight(rho, actor_logp, advantage, rho_cap, tail_floor):
    """(weight of the policy term, tail mask): the truncated importance ratio, zero where the actor sampled a tail token
    (its sampling logp below ``tail_floor``) in an episode with a positive advantage. Untruncated T=1 sampling draws
    such tokens about once in 200 (p < e^-7: 0.5% of sampled tokens); raising them is the largest-norm part of a
    positive episode's gradient and never the reason it succeeded. The token stays context and keeps its KL term; a
    negative advantage still pushes it down. ``tail_floor`` 0 or None: no mask."""
    import torch
    weight = rho.clamp(max=rho_cap)
    if not tail_floor or advantage <= 0:
        return weight, torch.zeros_like(weight, dtype=torch.bool)
    masked = actor_logp < tail_floor
    return weight.masked_fill(masked, 0.), masked


def think_mask(sample, close_id):
    """True for a sampled reply's reasoning tokens: up to and including the first ``</think>`` (all, when it has none)."""
    mask = np.ones(len(sample), dtype=bool)
    if close_id is not None:
        found = np.flatnonzero(np.asarray(sample) == close_id)
        if len(found):
            mask[found[0] + 1:] = False
    return mask


def moment_cosine(parameters, optimizer):
    """Cosine between the step's accumulated gradient and AdamW's first moment before the update (None before the first
    update): near 0 when consecutive steps are unrelated noise, positive when they share a direction."""
    import torch
    dot = gg = mm = 0.
    for p in parameters:
        state = optimizer.state.get(p)
        if not state or 'exp_avg' not in state:
            return None
        g, m = p.grad.reshape(-1), state['exp_avg'].reshape(-1)
        dot += torch.dot(g, m).item()
        gg += torch.dot(g, g).item()
        mm += torch.dot(m, m).item()
    return dot / math.sqrt(gg * mm) if gg > 0 and mm > 0 else None


def synthetic_reward(spec, episode):
    """Learning sanity check 4 (plans/neuralese/LEARNING_SANITY_20261112.md): a trivially learnable reward replacing the task
    outcome. ``short-calls:N``: 1 if the episode's model calls sample fewer than N tokens on average, else 0;
    ``short-calls``: minus the mean sampled tokens per call / 1000 (every group has spread)."""
    kind, _, value = spec.partition(':')
    if kind == 'short-calls':
        lengths = [len(row['sample']) for row in episode['rows']]
        if not value:
            return -sum(lengths) / len(lengths) / 1000 if lengths else -10.
        return float(bool(lengths) and sum(lengths) / len(lengths) < float(value))
    raise ValueError(f'unknown synthetic reward {spec!r}')


def probe_chains(args):
    """Fixed probe sequences (``--probe-turns``): their sampled tokens are scored on the learner after every step and on
    the actor after every weight sync, so an update that does not reach the served weights shows."""
    from sanity_logprob_paths import sample_chains
    return sample_chains(SimpleNamespace(turns=args.probe_turns, max_tokens=args.probe_max_tokens, seed=0,
                                         chains=args.probe_chains))


def probe_actor_nll(sglang_url, chains):
    from sanity_logprob_paths import targets_of
    total, count = 0., 0
    for chain in chains:
        positions, _ = targets_of(chain)
        answer = json.loads(post(sglang_url + '/generate', {'input_ids': [int(i) for i in chain['ids']],
            'return_logprob': True, 'logprob_start_len': 0, 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}}))
        items = answer['meta_info']['input_token_logprobs']
        total -= sum(items[p + 1][0] for p in positions)
        count += len(positions)
    return total / max(1, count)


def load_tasks(args, work=None):
    """The shuffled train tasks of the pools (first occurrence of an id wins) minus the held set: its tasks, every row of
    a held ``split_group`` (``held_eval.exclusion_ids``) and every row sharing a ``source_groups`` entry with those.
    Writes ``WORK/pool.json`` (pool sha256s, counts, sha256 of the eligible ids)."""
    import hashlib
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.common.jsonio import read_jsonl
    from natlang_neuralese.rl.held_eval import exclusion_ids
    rows, seen = [], set()
    for path in args.pool:
        for row in read_jsonl(path):
            if row['id'] not in seen:
                seen.add(row['id'])
                rows.append(row)
    excluded = set(exclusion_ids(args.held_manifest, rows))
    for extra in args.exclude_manifest:
        excluded |= exclusion_ids(extra, rows)
    held_groups = {g for r in rows if r['id'] in excluded for g in r.get('source_groups') or ()}
    excluded |= {r['id'] for r in rows if held_groups.intersection(r.get('source_groups') or ())}
    if args.task_ids:
        wanted = set(json.loads(Path(args.task_ids).read_text()))
        rows = [r for r in rows if r['id'] in wanted]
    import re
    broken = re.compile(args.exclude_family_regex) if args.exclude_family_regex else None
    errata = set()   # pool rows known to be wrong (e.g. an oracle's expected answer): never selected
    errata_paths = errata_files(args)
    for path in errata_paths:
        errata |= {row['id'] for row in read_jsonl(path)}
    eligible = [r for r in rows if r['id'] not in excluded and r['id'] not in errata and r.get('split', 'train') == 'train'
                and not (broken and broken.search(str(r.get('family') or '')))]
    random.Random(args.seed).shuffle(eligible)
    if work is not None:
        record = {'pools': [{'path': str(p), 'sha256': sha256_file_hex_cached(Path(p))} for p in args.pool],
                  'held_manifest': args.held_manifest, 'held_manifest_sha256': sha256_file_hex_cached(Path(args.held_manifest)),
                  'union_rows': len(seen), 'excluded_ids': len(excluded), 'eligible': len(eligible),
                  'eligible_ids_sha256': hashlib.sha256('\n'.join(sorted(r['id'] for r in eligible)).encode()).hexdigest(),
                  'families': len({r.get('family') for r in eligible}),
                  'excluded_family_regex': args.exclude_family_regex,
                  'excluded_errata': [{'path': str(p), 'sha256': sha256_file_hex_cached(Path(p))} for p in errata_paths],
                  'excluded_errata_ids': len(errata & {r['id'] for r in rows}),
                  'selection': {'pass_rates': args.pass_rates, 'band': args.pass_band, 'min_scored': args.pass_min_scored,
                                'explore': args.explore, 'own_outcomes': 'latest scored group per task replaces its rate'},
                  'episode_budget': {'max_model_requests': args.max_model_requests, 'case_seconds': args.case_seconds,
                                     'generated_tokens': None}}
        (Path(work) / 'pool.json').write_text(json.dumps(record, indent=1) + '\n')
    return eligible, len(excluded)


def load_variants(args, tasks, work):
    """``--neuralese-variants`` rows whose original (``curriculum.external.variant_of``) is an eligible task, keyed by
    that original; a variant shares its original's split group and source groups, so it is re-checked against the same
    held closure. Records ``WORK/neuralese-variants.json``."""
    if not args.neuralese_variants:
        return {}
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.common.jsonio import read_jsonl
    from natlang_neuralese.rl.held_eval import exclusion_ids
    eligible = {r['id'] for r in tasks}
    rows = list(read_jsonl(args.neuralese_variants))
    excluded = set(exclusion_ids(args.held_manifest, rows))
    variants = {}
    for row in rows:
        original = ((row.get('curriculum') or {}).get('external') or {}).get('variant_of')
        if original in eligible and row['id'] not in excluded and row.get('split', 'train') == 'train':
            variants[original] = row
    record = {'path': args.neuralese_variants, 'sha256': sha256_file_hex_cached(Path(args.neuralese_variants)),
              'rows': len(rows), 'usable': len(variants), 'share': args.neuralese_variant_share}
    (Path(work) / 'neuralese-variants.json').write_text(json.dumps(record, indent=1) + '\n')
    return variants


def train(args):
    import torch
    import exp0_sft
    import lfm3_fast_train
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / 'train.jsonl', 'a')
    tasks, excluded = load_tasks(args, work)
    source_sha = args.checkpoint_sha256 or sha256_file_hex_cached(Path(args.checkpoint))
    code = git_pin()
    pin = check_pin(work, code['git_commit'], args.unpinned)
    runtime = REPO / 'ts-host'
    from natlang_neuralese.rl.held_eval import series
    run = {'started': time.time(), 'code': code, 'pin': pin, 'runtime': str(runtime), 'series': series(), 'argv': sys.argv, 'checkpoint': args.checkpoint,
           'checkpoint_sha256': source_sha, 'pool': args.pool, 'held_manifest': args.held_manifest,
           'excluded_ids': excluded, 'eligible_tasks': len(tasks), 'args': vars(args)}
    with (work / 'runs.jsonl').open('a') as handle:
        handle.write(json.dumps(run) + '\n')
    log_line(log, {'event': 'start', 'code': run['code'], 'eligible_tasks': len(tasks), 'excluded_ids': excluded})

    t0 = time.time()
    actor_ready = {}
    actor_process = []

    def launch_actor():
        # The training actor is a child of this unit (one ledger claim covers learner and actor, so neither waits on the
        # other's admission), served with the pinned held-v2 eval engine (held-v2/engine.json serve_args): training and
        # held measurements sample with the same arithmetic.
        try:
            engine = json.loads((Path(args.held_manifest).parent / 'engine.json').read_text())['serve_args']
            store = work / 'actor'
            store.mkdir(parents=True, exist_ok=True)
            url = f'http://127.0.0.1:{args.actor_port}'
            command = [sys.executable, str(HERE / 'serve_native_checkpoint.py'), '--store-dir', str(store),
                       '--checkpoint', args.checkpoint, '--checkpoint-sha256', source_sha, '--base', args.base,
                       '--port', str(args.actor_port), '--prompt-rendering', 'full', *engine,
                       '--hf-export-dir', str(store / 'sglang-hf'), *args.actor_arg]
            with (work / 'actor.log').open('a') as handle:
                process = subprocess.Popen(command, cwd=REPO, env=pinned_env(), stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            actor_process.append(process)
            deadline = time.time() + 3600
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'actor exited with {process.returncode} (see {work}/actor.log)')
                try:
                    with urllib.request.urlopen(url + '/v1/neuralese/info', timeout=10) as response:
                        info = json.loads(response.read())
                    if (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') == source_sha:
                        actor_ready['url'] = url
                        return
                except OSError:
                    pass
                time.sleep(10)
            raise TimeoutError('actor did not come up in an hour')
        except BaseException as error:
            actor_ready['error'] = error

    def stop_actor(*_):
        for process in actor_process:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
    import atexit
    atexit.register(stop_actor)
    actor_thread = None
    if not args.actor:
        actor_thread = threading.Thread(target=launch_actor, daemon=True)
        actor_thread.start()
    # Loaded on the host, then the checkpoint's page cache is dropped before the GPU copy: on the GB10 a CUDA allocation
    # only gets MemFree, and the 9.5 GB checkpoint's cache made the move to the GPU fail under contention (smoke 2026-11-12).
    # BF16 on the host (the native backbone is BF16, so the FP32 masters made from it are the same values); the source
    # checkpoint is reduced to what a pruned checkpoint needs (the native rows the merged embedding overwrote).
    model, state, special = exp0_sft.load_model(args.checkpoint, args.base, device='cpu', dtype=torch.bfloat16)
    original = state['backbone_state']
    native_rows = {'embedding_rows': original['model.embed_tokens.weight'][torch.tensor(special)].clone(),
                   'tied_head': 'lm_head.weight' in original, 'names': [n for n in original if n != 'lm_head.weight']}
    state = {k: state[k] for k in ('schema', 'model_config', 'interface_state', 'step')}
    del original
    import gc
    gc.collect()
    drop_cache([args.checkpoint, *Path(args.base).glob('*.safetensors')])
    log_line(log, {'event': 'host model loaded', 'meminfo_gb': meminfo(), 'load_s': round(time.time() - t0, 1)})
    model.to('cuda')
    masters = {name: p.detach().clone().float() for name, p in model.named_parameters()}
    model.to(torch.bfloat16)
    reference = None
    if args.kl_beta > 0:          # the start policy, BF16, frozen (a copy: no second checkpoint load)
        import copy
        # The reference runs exactly the policy's training graph (train mode, parameters requiring grad, the same partial
        # checkpointing; its outputs are detached and never backpropagated): a no_grad pass compiles other kernels, whose
        # BF16 rounding alone put KL at 5e-4 (|dlogp| ~0.03) with identical weights (smoke 2026-11-12).
        reference = copy.deepcopy(model)
        reference.gradient_checkpointing_enable()
        reference.config.use_cache = False
        reference.train()
        lfm3_fast_train.prepare(reference)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    lfm3_fast_train.prepare(model)
    forward = lfm3_fast_train.hidden
    live = dict(model.named_parameters())
    parameters = [masters[name] for name in live]
    for master in parameters:
        master.grad = torch.zeros_like(master)
    # Each BF16 gradient is added into its FP32 master gradient as soon as autograd finishes it, then freed: a full set of
    # BF16 gradients never exists at once.
    def into_master(master):
        def hook(param):
            master.grad.add_(param.grad)
            param.grad = None
        return hook
    for name, p in model.named_parameters():
        p.register_post_accumulate_grad_hook(into_master(masters[name]))
    optimizer = torch.optim.AdamW([{'params': [p for p in parameters if p.ndim >= 2], 'weight_decay': args.weight_decay},
                                   {'params': [p for p in parameters if p.ndim < 2], 'weight_decay': 0.}],
                                  lr=args.lr, betas=(0.9, 0.99), eps=1e-8, fused=True)
    special_index = torch.tensor(special, device='cuda')
    embedding_name = next(n for n, p in live.items() if p is model.get_input_embeddings().weight)
    embedding_master = masters[embedding_name]
    saved_rows = embedding_master[special_index].clone()

    step, cursor, first_group = 0, 0, 0
    state_path = Path(args.state_dir) / 'training-state.safetensors'
    if args.resume and state_path.exists():
        from safetensors import safe_open
        with safe_open(str(state_path), 'pt', device='cuda') as handle:
            meta = handle.metadata()
            # materialise AdamW state (zero grads) before the masters are restored: with weight decay this step moves
            # the parameters, and it must not move the restored ones
            optimizer.step()
            for name in live:
                masters[name].copy_(handle.get_tensor('master/' + name))
            for name in live:
                st = optimizer.state[masters[name]]
                st['exp_avg'].copy_(handle.get_tensor('exp_avg/' + name))
                st['exp_avg_sq'].copy_(handle.get_tensor('exp_avg_sq/' + name))
                st['step'].fill_(float(meta['adam_step']))
        step, cursor, first_group = int(meta['step']), int(meta['cursor']), int(meta['next_group'])
        with torch.no_grad():
            for name, p in live.items():
                p.copy_(masters[name])
        log_line(log, {'event': 'resumed', 'step': step, 'cursor': cursor, 'next_group': first_group})
    # Group directories past the saved cursor (collected after the last state save, before a stop) are never reused:
    # numbering continues after the highest existing one.
    existing = [int(d.name[1:]) for d in (work / 'groups').glob('g*') if d.name[1:].isdigit()]
    first_group = max([first_group, *(i + 1 for i in existing)])

    if actor_thread is not None:
        actor_thread.join()
        if 'error' in actor_ready:
            raise actor_ready['error']
        args.actor = actor_ready['url']
        args.actor_export = args.actor_export or str(work / 'actor' / 'sglang-hf')
    log_line(log, {'event': 'actor ready', 'actor': args.actor, 'init_s': round(time.time() - t0, 1)})
    sglang_url = args.sglang or args.actor.rsplit(':', 1)[0] + ':' + str(int(args.actor.rsplit(':', 1)[1]) + 1000)
    # The actor's weights live on tmpfs: the shared NVMe wrote at 60-230 MB/s under other jobs' load (2026-11-12 bench),
    # which put 20-60 s of file write in front of every actor update.
    weights_dir = Path('/dev/shm') / f'natlang-{args.unit_prefix}-actor-weights'
    sync = WeightSync(model, args.actor_export, None if args.no_actor_sync else sglang_url, weights_dir,
                      front_url=None if args.no_actor_sync else args.actor)
    atexit.register(shutil.rmtree, weights_dir, True)
    drop_cache([Path(args.actor_export) / 'model.safetensors'])
    saver = StateSaver()
    collector = Collector(args, tasks, work, source_sha, runtime, first_group=first_group, cursor=cursor)
    collector.variants = load_variants(args, tasks, work)
    collector.version = step
    from natlang_neuralese.common.sampling import rl_sampling_controls
    (work / 'sampling-config.json').write_text(json.dumps(rl_sampling_controls(args.top_p, capture=True)) + '\n')
    versions_live = {'actor': step}
    probes = probe_chains(args) if args.probe_turns else []
    think_close = None   # the reasoning-close token id: splits the logged gradient mass into reasoning and reply tokens
    try:
        from tokenizers import Tokenizer
        think_close = Tokenizer.from_file(str(Path(args.base) / 'tokenizer.json')).token_to_id('</think>')
    except Exception as error:
        log_line(log, {'event': 'no </think> id: grad_mass_share is not logged', 'error': repr(error)[:200]})

    def probe_learner_nll():
        from sanity_logprob_paths import targets_of
        total, count = 0., 0
        for chain in probes:
            positions, targets = targets_of(chain)
            ids = torch.from_numpy(chain['ids'].astype(np.int64)).cuda()
            logp = lfm3_fast_train.token_logprobs(model, ids, torch.from_numpy(positions).cuda(),
                                                  torch.from_numpy(targets.astype(np.int64)).cuda())
            total -= logp.sum().item()
            count += len(positions)
        return total / max(1, count)

    def log_probe(version, where):
        try:
            log_line(log, {'event': 'probe', 'version': version, 'where': where, 'probe_tokens': sum(n for c in probes for _, n, _ in c['segs']),
                'nll': round(probe_actor_nll(sglang_url, probes) if where == 'actor' else probe_learner_nll(), 5)})
        except Exception as error:
            log_line(log, {'event': 'probe failed', 'version': version, 'where': where, 'error': repr(error)[:300]})

    def on_published(version, ok):
        if ok:
            collector.version = version
            versions_live['actor'] = version
            if probes and not args.no_actor_sync:
                log_probe(version, 'actor')
    if probes:
        log_probe(step, 'learner')
        log_probe(step, 'actor')
    step0 = work / 'held' / 'step-00000'
    if (step0 / 'result.json').exists() and not args.held_baseline:
        # A resume on a pin whose harness differs (another held series) cannot compare with the old step 0: that result
        # moves aside (it stays the record of its series) and the start policy is measured again on this pin below.
        measured = json.loads((step0 / 'result.json').read_text())
        old = (measured.get('series') or {}).get('key')
        law = measured.get('sampling') or LEGACY_SAMPLING     # the same for another sampling law (--top-p)
        if old != run['series']['key'] or law != sampling_law_of(args.top_p):
            aside = work / 'held' / (f'step-00000-series-{old}' if old != run['series']['key'] else
                                     f'step-00000-top-p-{law["top_p"]}')
            step0.rename(aside)
            log_line(log, {'event': 'held series changed', 'old_series': old, 'series': run['series']['key'],
                           'old_sampling': law, 'sampling': sampling_law_of(args.top_p), 'moved_to': str(aside)})
    if step:   # a resumed run first serves its own weights
        sync.snapshot()
        sync.publish(step, on_published)
        sync.wait()
        if not sync.last.get('ok'):
            raise RuntimeError(f'actor weight update failed: {sync.last}')
    log_line(log, {'event': 'learner ready', 'init_s': round(time.time() - t0, 1), 'meminfo_gb': meminfo(),
                   'cuda_allocated_gb': round(torch.cuda.memory_allocated() / 2**30, 1),
                   'cuda_reserved_gb': round(torch.cuda.memory_reserved() / 2**30, 1)})
    if args.held_every_hours > 0 and not (work / 'held' / 'step-00000' / 'result.json').exists() \
            and not args.skip_base_held and not args.held_baseline and not args.held_remote:
        collector.paused.set()            # the start policy's held measurement goes first (held() resumes collection)
    slots = [threading.Thread(target=collector.slot, args=(index,), daemon=True) for index in range(args.slots)]
    for thread in slots:
        thread.start()

    checkpoint_root = Path(args.checkpoint_dir or Path(args.state_dir) / 'checkpoints')   # off the busy NVMe by default
    # every trained episode is kept (compressed) beside the checkpoints: the w5 lesson (AGENTS.md: pruned bundles lost 85
    # successful episodes); '-' keeps none
    rollouts_archive = None if args.rollouts_dir == '-' else Path(args.rollouts_dir or checkpoint_root.parent / 'rollouts')
    checkpoints = sorted(checkpoint_root.glob('step-*'))
    held_steps = set(json.loads((work / 'held-steps.json').read_text())) if (work / 'held-steps.json').exists() else set()
    stop = threading.Event()
    # A unit stop signals every process of the cgroup, so the child actor goes at once: slots stop starting episodes
    # (each would fail in seconds on 'fetch failed' and burn its whole group as infrastructure errors, 97 episodes in
    # the two run2 restarts of 2026-11-13); the learner finishes the step it is in.
    signal.signal(signal.SIGTERM, lambda *_: (stop.set(), collector.stop.set()))

    def write_native(step_number):
        directory = checkpoint_root / f'step-{step_number:05d}'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'checkpoint.pt'
        changed = write_pruned_checkpoint(state, native_rows, special, path, source_sha, sync.backbone())
        from natlang_neuralese.common.hashing import sha256_file_hex
        sha = sha256_file_hex(path)
        (directory / 'sha256.txt').write_text(sha + '\n')
        (directory / 'meta.json').write_text(json.dumps({'step': step_number, 'sha256': sha, 'changed_elements': changed,
                                                         'source_sha256': source_sha, 'code': run['code']}) + '\n')
        checkpoints.append(directory)
        for old in checkpoints[:-2]:
            if int(old.name.split('-')[1]) not in held_steps and old.exists():
                shutil.rmtree(old)
        checkpoints[:] = [c for c in checkpoints if c.exists()]
        return {'checkpoint': str(path), 'checkpoint_sha256': sha}

    def held(step_number, samples, name=None, policy=None, resume_collection=True):
        """``held_eval measure`` on the training actor itself (it serves the current policy with the held-v2 engine pin):
        new collection pauses, the actor's weights stay fixed until the measurement ends, then ``compare`` with step 0.
        ``resume_collection=False``: the actor serves another policy than the learner's (start policy, --held-extra), so
        collection stays paused until the caller has restored the learner's weights."""
        if name is None:
            held_steps.add(step_number)
            (work / 'held-steps.json').write_text(json.dumps(sorted(held_steps)) + '\n')
        out = work / 'held' / (name or f'step-{step_number:05d}')
        policy = policy or (sync.last or {}).get('policy_id') or source_sha
        collector.paused.set()
        started = time.time()
        python = [sys.executable, '-m', 'natlang_neuralese.rl.held_eval']
        try:
            with (work / 'held.log').open('a') as handle:
                code = subprocess.run([*python, 'measure', '--manifest', args.held_manifest, '--endpoint',
                                       f'{args.actor}={args.held_workers}', '--policy-id', policy, '--top-p', str(args.top_p), '--samples', str(samples),
                                       '--out', str(out), '--label', f'grpo-{name or f"step-{step_number}"}',
                                       '--max-model-requests', str(args.max_model_requests)],
                                      cwd=REPO, env=pinned_env(), stdout=handle, stderr=subprocess.STDOUT).returncode
                base = Path(args.held_baseline) if args.held_baseline else work / 'held' / 'step-00000' / 'result.json'
                if (step_number or name) and code == 0 and base.exists():
                    subprocess.run([*python, 'compare', str(base), str(out / 'result.json'), '--out',
                                    str(out / 'compare-vs-step0.json')], cwd=REPO, env=pinned_env(), stdout=handle, stderr=subprocess.STDOUT)
        finally:
            if resume_collection:
                collector.paused.clear()
        summary = {}
        if (out / 'result.json').exists():
            result = json.loads((out / 'result.json').read_text())
            summary = {k: result.get(k) for k in ('ran', 'scored', 'successes', 'loop_rate', 'ran_out_rate', 'complete')}
        if (out / 'compare-vs-step0.json').exists():
            summary['compare'] = json.loads((out / 'compare-vs-step0.json').read_text())
        log_line(log, {'event': 'held eval', 'step': step_number, 'arm': name, 'policy': policy, 'returncode': code,
                       'seconds': round(time.time() - started), **summary})

    if args.held_every_hours > 0 and not (work / 'held' / 'step-00000' / 'result.json').exists() \
            and not args.skip_base_held and not args.held_baseline and not args.held_remote:
        if not step:
            held(0, args.held_samples_start)
        else:
            # A resumed run measures the start policy on this pin in place (as --held-extra does), restores its own
            # weights and measures the current step at once: that result anchors the step's series beside the new step 0.
            collector.paused.set()
            try:
                sync.load_native(args.checkpoint, special)
                sync.publish(0, lambda *_: None, policy=source_sha)
                sync.wait()
                if not sync.last.get('ok'):
                    raise RuntimeError(f'loading the start policy into the actor failed: {sync.last}')
                held(0, args.held_samples_start, policy=source_sha, resume_collection=False)
            finally:
                sync.snapshot()
                sync.publish(step, on_published, also=lambda s=step: write_native(s))
                sync.wait()
            if not sync.last.get('ok'):
                raise RuntimeError(f'restoring the learner weights on the actor failed: {sync.last}')
            held(step, args.held_samples)
    if args.held_extra and step == 0:
        # Other policies measured on this actor in place (one engine, one code pin), each against step 0; then the start
        # policy is loaded back before training.
        collector.paused.set()
        try:
            for path in args.held_extra.split(','):
                from natlang_neuralese.common.hashing import sha256_file_hex_cached
                sha = sha256_file_hex_cached(Path(path))
                name = f'extra-{sha[:12]}'
                if (work / 'held' / name / 'result.json').exists():
                    continue
                sync.load_native(path, special)
                sync.publish(0, on_published, policy=sha)
                sync.wait()
                if not sync.last.get('ok'):
                    raise RuntimeError(f'loading {path} into the actor failed: {sync.last}')
                (work / 'held').mkdir(exist_ok=True)
                (work / 'held' / f'{name}.source').write_text(path + '\n')
                held(0, args.held_samples_extra, name=name, policy=sha, resume_collection=False)
        finally:
            sync.snapshot()                   # the learner's own (start) weights back on the actor
            sync.publish(0, on_published, policy=source_sha)
            sync.wait()
            collector.paused.clear()
        if not sync.last.get('ok'):
            raise RuntimeError(f'restoring the start policy on the actor failed: {sync.last}')
    last_held = time.time()

    def pop_held_loop():
        """``--held-remote pop``: held evaluations run on Pop's GPU while the DGX trains. The start policy is measured
        once on Pop (``held/step-00000-pop``: compare pairs only results of one engine); then, every ``--held-every-hours``
        or as soon as the previous evaluation ends if it took longer, the newest native checkpoint is served on Pop
        (``pop_rl_actor.py serve-checkpoint``: rsync, the run's commit) and measured with this run's pinned harness (the
        collector processes run here, against Pop's endpoint) into ``held/step-N`` with ``compare-vs-step0.json``."""
        from pop_rl_actor import pop_sglang
        held_root, tool = work / 'held', [sys.executable, str(HERE / 'pop_rl_actor.py')]
        base_out = held_root / 'step-00000-pop'
        python = [sys.executable, '-m', 'natlang_neuralese.rl.held_eval']
        # Pop's engine settings: the KV dtype is part of the attested engine (export format), so the start policy's
        # result is kept only if it was measured with the same one; max running sequences are capacity only.
        pop_engine = ['--kv-cache-dtype', args.pop_kv_cache_dtype, '--max-running', str(args.pop_max_running)]
        pop_format = pop_sglang(args.pop_kv_cache_dtype, args.pop_max_running)[1]

        def base_engine_matches():
            """False when held/step-00000-pop (finished or partial) was measured on another Pop engine: it moves to
            step-00000-pop-<time>-other-engine and the start policy is measured again (its steps would not pair with it)."""
            for name in ('result.json', 'run.json'):
                if (base_out / name).exists():
                    record = json.loads((base_out / name).read_text())
                    measured = (record.get('engine') or {}).get('export_format')
                    # ... or under another sampling law (--top-p): compare refuses such pairs
                    return measured in (None, pop_format) and \
                        (record.get('sampling') or LEGACY_SAMPLING) == sampling_law_of(args.top_p)
            return True

        def pop_endpoint(policy):
            state = json.loads(subprocess.run([*tool, 'status'], capture_output=True, text=True, cwd=REPO,
                                              env=pinned_env()).stdout or '{}').get('state') or {}
            try:
                with urllib.request.urlopen(state['endpoint'] + '/v1/neuralese/info', timeout=10) as response:
                    binding = json.loads(response.read()).get('rl_policy_binding') or {}
            except (KeyError, OSError, ValueError):
                return None
            served, served_format = binding.get('checkpoint_sha256'), (binding.get('actor_export') or {}).get('format')
            return state['endpoint'] if served == policy and served_format == pop_format else None

        def evaluate(step_number, out, policy, serve):
            started = time.time()
            with (work / 'held-pop.log').open('a') as handle:
                endpoint = pop_endpoint(policy)
                if endpoint is None:
                    subprocess.run([*tool, *serve], cwd=REPO, env=pinned_env(), stdout=handle, stderr=subprocess.STDOUT,
                                   timeout=4 * 3600)
                    endpoint = pop_endpoint(policy)
                if endpoint is None:
                    raise RuntimeError(f'Pop does not serve {policy[:12]} (see {work}/held-pop.log)')
                serving = time.time() - started
                code = subprocess.run([*python, 'measure', '--manifest', args.held_manifest, '--endpoint',
                                       f'{endpoint}={args.held_remote_slots}', '--policy-id', policy, '--top-p', str(args.top_p), '--samples',
                                       str(args.held_samples), '--out', str(out), '--label', f'grpo-pop-step-{step_number}',
                                       '--allow-other-engine', '--max-model-requests', str(args.max_model_requests)],
                                      cwd=REPO, env=pinned_env(), stdout=handle,
                                      stderr=subprocess.STDOUT).returncode
                if step_number and code == 0 and (base_out / 'result.json').exists():
                    subprocess.run([*python, 'compare', str(base_out / 'result.json'), str(out / 'result.json'), '--out',
                                    str(out / 'compare-vs-step0.json')], cwd=REPO, env=pinned_env(), stdout=handle,
                                   stderr=subprocess.STDOUT)
            summary = {}
            if (out / 'result.json').exists():
                result = json.loads((out / 'result.json').read_text())
                summary = {k: result.get(k) for k in ('ran', 'scored', 'successes', 'loop_rate', 'ran_out_rate', 'complete')}
            if (out / 'compare-vs-step0.json').exists():
                summary['compare'] = json.loads((out / 'compare-vs-step0.json').read_text())
            log_line(log, {'event': 'held eval', 'remote': 'pop', 'step': step_number, 'policy': policy, 'returncode': code,
                           'seconds': round(time.time() - started), 'serve_seconds': round(serving), **summary})

        def measured_steps():
            return {int(d.name.split('-')[1]) for d in held_root.glob('step-*') if d.name.split('-')[1].isdigit()
                    and (d / 'result.json').exists()}

        last_start, last_step = 0., max(measured_steps() - {0}, default=0)
        while not stop.is_set():
            try:
                if not base_engine_matches():
                    aside = held_root / f'step-00000-pop-{time.strftime("%Y%m%d%H%M")}-other-engine'
                    base_out.rename(aside)
                    log_line(log, {'event': 'held eval', 'remote': 'pop', 'moved_start_policy_result': str(aside),
                                   'reason': f'measured on another Pop engine than {pop_format}'})
                if not (base_out / 'result.json').exists():
                    evaluate(0, base_out, source_sha, ['start', '--checkpoint', args.checkpoint, '--policy-id', source_sha,
                                                       *pop_engine])
                    if not (base_out / 'result.json').exists():
                        raise RuntimeError(f'the start policy measurement ended without a result (see {work}/held-pop.log)')
                ready = sorted(d for d in checkpoint_root.glob('step-*') if (d / 'sha256.txt').exists()
                               and int(d.name.split('-')[1]) > last_step)
                if ready and time.time() - last_start >= args.held_every_hours * 3600:
                    directory = ready[-1]
                    step_number = int(directory.name.split('-')[1])
                    held_steps.add(step_number)          # kept by write_native's pruning from now on
                    (work / 'held-steps.json').write_text(json.dumps(sorted(held_steps)) + '\n')
                    if (directory / 'checkpoint.pt').exists():
                        sha = (directory / 'sha256.txt').read_text().strip()
                        last_start, last_step = time.time(), step_number
                        evaluate(step_number, held_root / f'step-{step_number:05d}', sha,
                                 ['serve-checkpoint', '--checkpoint', str(directory / 'checkpoint.pt'), '--policy-id', sha,
                                  '--commit', run['code']['git_commit'], '--step', str(step_number), *pop_engine])
            except Exception as error:   # Pop or the network failed: the run trains on, the evaluation retries later
                log_line(log, {'event': 'held eval failed', 'remote': 'pop', 'error': repr(error)[:500]})
                stop.wait(540)
            stop.wait(60)
    if args.held_remote:
        threading.Thread(target=pop_held_loop, daemon=True).start()

    # actor tokens/s while the learner computes (learner-phase slots) and while it waits (all slots), cumulative
    phase_totals = collections.Counter()
    phase_mark = {'time': time.time(), 'tokens': actor_generated(args.actor)}
    while not stop.is_set() and (not args.max_steps or step < args.max_steps):
        phase = collections.OrderedDict()
        t = time.time()
        groups = collector.take_groups(args.min_groups, args.max_groups, stop)
        phase['wait_groups'] = time.time() - t
        t = time.time()
        stats = collections.Counter()
        chains, decisions, rewards, spread, dropped_stale, staleness, task_rewards = [], [], [], 0, 0, [], []
        episode_units = {}   # trained episode -> [sampled tokens, calls, decisions] (loss_scales)
        for group in groups:
            try:
                task_id, episodes, versions = load_group(collector, group, stats)
            except Exception as error:   # one malformed capture loses its group (kept for diagnosis), never the run
                stats['groups_lowering_failed'] += 1
                log_line(log, {'event': 'group lowering failed', 'group': group, 'error': repr(error)[:400]})
                prune_group(collector, group, keep=True)
                continue
            prune_group(collector, group, keep=args.keep_rollouts, archive=rollouts_archive, stats=stats)
            if not episodes:
                continue
            if args.synthetic_reward:
                task_rewards += [e['reward'] for e in episodes]
                for episode in episodes:
                    episode['reward'] = synthetic_reward(args.synthetic_reward, episode)
            stale = step - min(versions) if versions else 0
            rewards += [e['reward'] for e in episodes]
            if stale > args.max_staleness:
                dropped_stale += 1
                continue
            staleness.append(stale)
            advantages = group_advantages([e['reward'] for e in episodes]) if len(episodes) >= 2 else None
            if advantages is None:
                continue
            spread += 1
            for episode, advantage in zip(episodes, advantages):
                kept = exp0_sft.chains_of([episode], max_len=args.max_sequence_tokens)
                stats['calls_too_long_dropped'] += len(episode['rows']) - sum(len(c['segs']) for c in kept)
                key = len(episode_units)
                episode_units[key] = [sum(n for c in kept for _, n, _ in c['segs']), sum(len(c['segs']) for c in kept), 0]
                for chain in kept:
                    chain['advantage'] = advantage
                    chain['episode'] = key
                    chains.append(chain)
                for d in episode['decisions']:
                    # every option is scored as its own copy of the prompt, so a decision is one forward of K prompts:
                    # beyond the sequence cap it would not fit beside the actor (the chains' cap, per decision)
                    if decision_tokens(d) > args.max_sequence_tokens:
                        stats['decisions_too_long_dropped'] += 1
                    else:
                        episode_units[key][2] += 1
                        decisions.append({'row': d, 'advantage': advantage, 'group': group, 'task_id': task_id,
                                          'stale': stale, 'actor_version': step - stale, 'episode': key})
        # Loss normalization (loss_scales): by default every trained episode weighs the same and a sampled decision (one
        # categorical action) counts as one call of its episode. `units` only normalizes the logged token means.
        loss_tokens = sum(length for c in chains for _, length, _ in c['segs'])
        units = loss_tokens + len(decisions)
        scales = loss_scales(episode_units, args.loss_norm, args.decision_weight)
        episodes_trained = sum(1 for token_scale, decision_scale in scales.values() if token_scale or decision_scale)
        phase['lower'] = time.time() - t
        record = {'step': step + 1, 'groups': len(groups), 'groups_with_spread': spread,
                  'spread_fraction': round(spread / max(1, len(groups)), 3), 'dropped_stale_groups': dropped_stale,
                  'staleness': dict(collections.Counter(staleness)), 'episodes': len(rewards),
                  'reward_mean': round(sum(rewards) / max(1, len(rewards)), 4), 'sequences': len(chains),
                  'loss_tokens': loss_tokens, 'decisions': len(decisions), 'lowering': dict(stats),
                  'loss_norm': args.loss_norm, 'decision_weight': args.decision_weight, 'episodes_trained': episodes_trained,
                  'rollouts_archive_free_gb': round(shutil.disk_usage(rollouts_archive).free / 2**30, 1)
                  if rollouts_archive is not None and rollouts_archive.exists() else None,
                  'actor_version': versions_live['actor']}
        if args.synthetic_reward:
            record.update(synthetic_reward=args.synthetic_reward,
                          task_reward_mean=round(sum(task_rewards) / max(1, len(task_rewards)), 4),
                          mean_sampled_per_call=round(sum(len(r['sample']) for c in chains for _, _, r in c['segs'])
                                                      / max(1, sum(len(c['segs']) for c in chains)), 1))
        if not chains and not decisions:
            log_line(log, {**record, 'event': 'no learning signal in these groups',
                           'phase_s': {k: round(v, 1) for k, v in phase.items()}})
            continue
        # learner
        torch.cuda.reset_peak_memory_stats()
        collector.active_slots = args.learner_phase_slots
        learn_started, learn_tokens0 = time.time(), actor_generated(args.actor)
        swap0, mem0 = vmstat(), meminfo()
        progress_mark = time.time()
        if phase_mark['tokens'] is not None and learn_tokens0 is not None:
            phase_totals['collect_s'] += learn_started - phase_mark['time']
            phase_totals['collect_tokens'] += learn_tokens0 - phase_mark['tokens']
        sums = collections.Counter()
        rho_pre_max, rho_post_max, forward_tokens = 0., 0., 0
        rng = random.Random(args.seed + step)
        rng.shuffle(chains)
        for chain in chains:
            ids, positions, targets = exp0_sft.chain_tensors(chain, 'cuda')
            # a consumer call's imported native state: fixed context exactly as the actor fed it (no producer gradient)
            imports = lfm3_fast_train.import_tensors(model, [(start, collector.states(block_id))
                                                             for start, block_id in chain.get('imports') or ()])
            actor = torch.from_numpy(np.concatenate([row['logp'] for _, _, row in chain['segs']])).cuda().float()
            token_scale = scales[chain['episode']][0]
            t = time.time()
            ref = None
            if reference is not None:
                exp0_sft.set_checkpointing(reference, len(ids), args.activation_gb)
                with torch.enable_grad():
                    ref = -exp0_sft.token_nll(reference, forward(reference, ids, None, imports), positions, targets).detach()
                for p in reference.parameters():
                    p.grad = None
                torch.cuda.synchronize()
                phase['reference_forward'] = phase.get('reference_forward', 0.) + time.time() - t
                t = time.time()
            exp0_sft.set_checkpointing(model, len(ids), args.activation_gb)
            h = forward(model, ids, None, imports)
            if args.top_p < 1:
                # Top-p: the policy is the learner's distribution on its own nucleus (a constant set that always holds
                # the sampled token), so the policy term's log-probability is logp - log(nucleus mass); the KL term
                # and rho keep the untruncated logp (the actor's record is untruncated too: rho is the ratio of the two
                # truncated laws with the actor's nucleus mass taken as the learner's, |d log rho| <= -log(top_p)).
                nll, nucleus_nll, outside = exp0_sft.token_nll_nucleus(model, h, positions, targets, args.top_p)
                logp, policy_logp = -nll, -nucleus_nll
                sums['outside_nucleus'] += int(outside.sum())
                sums['nucleus_log_mass'] += (logp.detach() - policy_logp.detach()).sum().item()
            else:
                logp = policy_logp = -exp0_sft.token_nll(model, h, positions, targets)
            with torch.no_grad():
                log_rho = logp.detach() - actor
                rho = log_rho.clamp(max=20).exp()
                weight, tail = policy_weight(rho, actor, chain['advantage'], args.rho_cap, args.tail_logp_floor)
                # gradient mass of the policy term per token (its factor times the logit-gradient norm scale 1 - p),
                # split into reasoning and post-</think> tokens
                mass = token_scale * abs(chain['advantage']) * weight * (1 - policy_logp.detach().exp())
                thinking = torch.from_numpy(np.concatenate([think_mask(row['sample'], think_close)
                                                            for _, _, row in chain['segs']])).cuda()
                sums['mass_think'] += mass[thinking].sum().item()
                sums['mass_post_think'] += mass[~thinking].sum().item()
                sums['tail_masked'] += int(tail.sum())
            loss_terms = -(weight * chain['advantage'] * policy_logp)
            if ref is not None:
                d = ref - logp
                kl = d.exp() - d - 1
                loss_terms = loss_terms + args.kl_beta * kl
                sums['kl'] += kl.detach().sum().item()
                sums['ref_abs_dlogp'] += d.detach().abs().sum().item()
            (loss_terms.sum() * token_scale).backward()
            sums['rho'] += rho.sum().item()
            sums['rho_capped'] += (rho > args.rho_cap).sum().item()
            sums['abs_log_rho'] += log_rho.abs().sum().item()
            if chain.get('imports'):    # learner-actor agreement on calls that read imported Neuralese state
                consumer = torch.from_numpy(np.concatenate([np.full(n, bool(row.get('imports')))
                                                            for _, n, row in chain['segs']])).cuda()
                sums['consumer_tokens'] += int(consumer.sum())
                sums['consumer_abs_log_rho'] += log_rho.abs()[consumer].sum().item()
            sums['logp'] += logp.detach().sum().item()
            rho_pre_max = max(rho_pre_max, rho.max().item())
            rho_post_max = max(rho_post_max, weight.max().item())
            forward_tokens += len(ids)
            del h, logp, policy_logp, loss_terms
            torch.cuda.synchronize()
            phase['policy_forward_backward'] = phase.get('policy_forward_backward', 0.) + time.time() - t
            if time.time() - progress_mark > 300:
                progress_mark = time.time()
                busy = phase.get('reference_forward', 0.) + phase['policy_forward_backward']
                log_line(log, {'event': 'learner progress', 'step': step + 1, 'forward_tokens': forward_tokens,
                               'tokens_per_s': round(forward_tokens / busy), 'meminfo_gb': meminfo(),
                               'swap_pages': {k: v - swap0[k] for k, v in vmstat().items()}})
        # Sampled decisions: log pi(chosen) = log_softmax(s / T)[chosen], s_o the summed log-probability of option o's
        # continuation after the exact prompt (as the actor scored it); TIS and k3 on the chosen option as for a token.
        decision_records = []
        for batch in decision_batches(decisions, args.decision_batch_tokens):
            ids, lengths, positions, targets, owner, counts, spans = decision_tensors(batch, 'cuda')
            imports = lfm3_fast_train.import_tensors(model, [(start, collector.states(block_id)) for start, block_id in spans])
            options = sum(counts)
            def scores_of(net):
                token = -exp0_sft.token_nll(net, forward(net, ids, lengths, imports), positions, targets)
                return torch.zeros(options, device='cuda', dtype=token.dtype).index_add(0, owner, token)
            t = time.time()
            ref_scores = None
            if reference is not None:
                exp0_sft.set_checkpointing(reference, len(ids), args.activation_gb)
                with torch.enable_grad():
                    ref_scores = scores_of(reference).detach()
                for p in reference.parameters():
                    p.grad = None
                torch.cuda.synchronize()
                phase['reference_forward'] = phase.get('reference_forward', 0.) + time.time() - t
                t = time.time()
            exp0_sft.set_checkpointing(model, len(ids), args.activation_gb)
            scores = scores_of(model)
            terms, start = [], 0
            for item, k in zip(batch, counts):
                row = item['row']
                logpi = torch.log_softmax(scores[start:start + k] / row['temperature'], 0)[row['chosen']]
                with torch.no_grad():
                    log_rho = logpi.detach() - row['behavior_logq']
                    rho = log_rho.clamp(max=20).exp()
                    weight, tail = policy_weight(rho, torch.as_tensor(row['behavior_logq'], device='cuda'),
                                                 item['advantage'], args.rho_cap, args.tail_logp_floor)
                    decision_scale = scales[item['episode']][1]
                    sums['mass_decision'] += (decision_scale * abs(item['advantage']) * weight
                                              * (1 - logpi.detach().exp())).item()
                    sums['tail_masked_decisions'] += int(tail)
                    actor_scores = torch.from_numpy(row['scores']).cuda()
                    sums['decision_abs_dscore'] += (scores[start:start + k].detach() - actor_scores).abs().mean().item()
                    decision_records.append(decision_record(
                        item, scores[start:start + k].detach().float().cpu().numpy(), row['scores'],
                        None if ref_scores is None else ref_scores[start:start + k].float().cpu().numpy(), log_rho.item()))
                term = -(weight * item['advantage'] * logpi)
                if ref_scores is not None:
                    d = torch.log_softmax(ref_scores[start:start + k] / row['temperature'], 0)[row['chosen']] - logpi
                    kl = d.exp() - d - 1
                    term = term + args.kl_beta * kl
                    sums['kl'] += kl.item()
                    sums['ref_abs_dlogp'] += d.detach().abs().item()
                terms.append(term * decision_scale)
                sums['decision_rho'] += rho.item()
                sums['decision_abs_log_rho'] += log_rho.abs().item()
                sums['decision_logpi'] += logpi.item()
                rho_pre_max = max(rho_pre_max, rho.item())
                rho_post_max = max(rho_post_max, weight.item())
                start += k
            torch.stack(terms).sum().backward()
            forward_tokens += len(ids)
            del scores, terms
            torch.cuda.synchronize()
            phase['decision_forward_backward'] = phase.get('decision_forward_backward', 0.) + time.time() - t
        collector.states.cache.clear()    # this step's imported states (lowering loads them, the learner reuses them)
        if decision_records:
            (work / 'decisions').mkdir(exist_ok=True)
            with (work / 'decisions' / f'step-{step + 1:05d}.jsonl').open('w') as handle:
                handle.writelines(json.dumps(r) + '\n' for r in decision_records)
        record['decision_diagnostics'] = decision_summary(decision_records)
        t = time.time()
        saver.wait()                      # a state save in flight must finish before the masters change
        phase['wait_state_save'] = time.time() - t
        t = time.time()
        embedding_master.grad[special_index] = 0
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip).item()
        if not math.isfinite(grad_norm):
            # a NaN/inf gradient would make every FP32 master and AdamW moment NaN, and the next state save would
            # overwrite the only training state with it: drop this step's update, keep the weights
            for p in parameters:
                p.grad.zero_()
            collector.active_slots = args.slots
            log_line(log, {**record, 'event': 'non-finite gradient: update skipped', 'grad_norm': repr(grad_norm),
                           'loss_sums': {k: v for k, v in sums.items() if isinstance(v, float)}})
            continue
        cosine = moment_cosine(parameters, optimizer)   # this step's gradient against the previous steps' momentum
        lr = args.lr * min(1., (step + 1) / args.warmup)
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        for p in parameters:
            p.grad.zero_()
        changed_step = changed_start = 0
        with torch.no_grad():
            embedding_master[special_index] = saved_rows
            reference_params = dict(reference.named_parameters()) if reference is not None else {}
            for name, p in live.items():
                new = masters[name].to(torch.bfloat16)
                changed_step += int((new != p).sum())
                if name in reference_params:
                    changed_start += int((new != reference_params[name]).sum())
                p.copy_(new)
        torch.cuda.synchronize()
        phase['optimizer'] = time.time() - t
        elements = sum(p.numel() for p in live.values())
        record.update(bf16_changed_step_fraction=round(changed_step / elements, 6),
                      bf16_changed_vs_start_fraction=round(changed_start / elements, 6) if reference is not None else None)
        if probes:
            record['probe_learner_nll'] = round(probe_learner_nll(), 5)
        step += 1
        collector.active_slots = args.slots
        learn_ended, learn_tokens1 = time.time(), actor_generated(args.actor)
        if learn_tokens0 is not None and learn_tokens1 is not None:
            phase_totals['learn_s'] += learn_ended - learn_started
            phase_totals['learn_tokens'] += learn_tokens1 - learn_tokens0
        phase_mark.update(time=learn_ended, tokens=learn_tokens1)
        # weight sync and checkpoints (background)
        t = time.time()
        sync.snapshot()                   # waits for the previous publish
        phase['snapshot'] = time.time() - t
        if sync.last and not sync.last.get('ok'):
            raise RuntimeError(f'actor weight update failed: {sync.last}')
        held_due = args.held_every_hours > 0 and not args.held_remote and time.time() - last_held > args.held_every_hours * 3600
        save_native = step % args.save_every == 0 or (args.max_steps and step == args.max_steps) or held_due
        sync.publish(step, on_published, also=(lambda s=step: write_native(s)) if save_native else None)
        if args.state_every and step % args.state_every == 0:
            tensors = {}
            for name in live:
                st = optimizer.state[masters[name]]
                tensors['master/' + name] = masters[name]
                tensors['exp_avg/' + name] = st['exp_avg']
                tensors['exp_avg_sq/' + name] = st['exp_avg_sq']
            adam_step = optimizer.state[parameters[0]]['step'].item()
            saver.save(state_path, tensors, {'step': step, 'adam_step': adam_step, 'cursor': collector.cursor,
                                             'next_group': collector.next_group, 'source_sha256': source_sha})
        learner_seconds = sum(v for k, v in phase.items()
                              if k in ('reference_forward', 'policy_forward_backward', 'decision_forward_backward', 'optimizer'))
        n_dec = max(1, len(decisions))
        mass_total = sums['mass_think'] + sums['mass_post_think'] + sums['mass_decision']
        elapsed = time.time() - run['started']
        record.update(
            loss_tokens=loss_tokens, forward_tokens=forward_tokens, lr=lr, grad_norm=round(grad_norm, 4),
            kl_mean=round(sums['kl'] / units, 7), ref_abs_dlogp_mean=round(sums['ref_abs_dlogp'] / units, 6),
            mean_logp=round(sums['logp'] / max(1, loss_tokens), 4),
            grad_moment_cosine=None if cosine is None else round(cosine, 5),
            top_p=args.top_p, outside_nucleus_fraction=round(sums['outside_nucleus'] / max(1, loss_tokens), 6),
            nucleus_mass_mean=round(math.exp(sums['nucleus_log_mass'] / max(1, loss_tokens)), 5),
            tail_logp_floor=args.tail_logp_floor, tail_masked_fraction=round(sums['tail_masked'] / max(1, loss_tokens), 6),
            tail_masked_decisions=sums['tail_masked_decisions'],
            grad_mass_share={k: round(sums['mass_' + k] / mass_total, 4) for k in ('think', 'post_think', 'decision')}
            if think_close is not None and mass_total > 0 else None,
            decision_rho_mean=round(sums['decision_rho'] / n_dec, 4), decision_abs_log_rho_mean=round(sums['decision_abs_log_rho'] / n_dec, 5),
            decision_mean_logpi=round(sums['decision_logpi'] / n_dec, 4),
            decision_abs_dscore_mean=round(sums['decision_abs_dscore'] / n_dec, 4),
            rho_mean=round(sums['rho'] / max(1, loss_tokens), 4), rho_pre_cap_max=round(rho_pre_max, 3),
            rho_post_cap_max=round(rho_post_max, 3), rho_capped_fraction=round(sums['rho_capped'] / max(1, loss_tokens), 5),
            abs_log_rho_mean=round(sums['abs_log_rho'] / max(1, loss_tokens), 5),
            consumer_tokens=sums['consumer_tokens'],
            consumer_abs_log_rho_mean=round(sums['consumer_abs_log_rho'] / max(1, sums['consumer_tokens']), 5),
            learner_tokens_per_s=round(forward_tokens / max(1e-6, learner_seconds)),
            actor_generated_tokens=collector.generated, actor_episodes=collector.episodes,
            actor_generated_tokens_per_s=round(collector.generated / max(1., elapsed)),
            phase_s={k: round(v, 1) for k, v in phase.items()}, previous_sync=sync.last,
            actor_tokens_per_s_learn_phase=round(phase_totals['learn_tokens'] / phase_totals['learn_s']) if phase_totals['learn_s'] else None,
            actor_tokens_per_s_collect_phase=round(phase_totals['collect_tokens'] / phase_totals['collect_s']) if phase_totals['collect_s'] else None,
            learner_phase_slots=args.learner_phase_slots,
            learner_swap_pages={k: v - swap0[k] for k, v in vmstat().items()}, meminfo_learn_start_gb=mem0,
            previous_state_save=saver.last, cuda_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
            outstanding_groups=collector.outstanding, elapsed_s=round(elapsed))
        if not args.keep_rollouts and args.block_gc_grace >= 0:
            # the actor's block store: keep what an outstanding group's episodes may still import (see collect_blocks)
            t = time.time()
            with collector.lock:
                oldest = min([info['created'] for info in collector.groups.values()], default=time.time())
            files, freed = collect_blocks(collector.states.directory, oldest - args.block_gc_grace)
            record.update(blocks_removed=files, blocks_freed_mb=round(freed / 2**20, 1),
                          block_gc_s=round(time.time() - t, 2))
        log_line(log, record)
        if save_native and held_due:
            sync.wait()                   # the actor now serves (and attests) this step's checkpoint
            held(step, args.held_samples)
            last_held = time.time()
            phase_mark.update(time=time.time(), tokens=actor_generated(args.actor))   # the measurement is neither phase
    sync.wait()
    saver.wait()
    collector.stop.set()
    log_line(log, {'event': 'stopped', 'step': step})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--work', required=True)
    t.add_argument('--checkpoint', required=True, help='start policy (native checkpoint); also the KL reference')
    t.add_argument('--checkpoint-sha256')
    t.add_argument('--base', default=BASE)
    t.add_argument('--actor', help='natlang front of an SGLang actor serving --checkpoint (default: a child actor of this '
                   'process with the held-v2 engine arguments)')
    t.add_argument('--actor-port', type=int, default=8460, help='port of the child actor (its SGLang server: +1000)')
    t.add_argument('--actor-arg', action='append', default=[], help='extra serve_native_checkpoint.py argument')
    t.add_argument('--sglang', help='its SGLang server (default: actor port + 1000)')
    t.add_argument('--actor-store', help="the actor's native block store (--store-dir; default WORK/actor): consumer calls "
                   'read their imported Neuralese state from it')
    t.add_argument('--actor-export', help="the actor's HF export dir (layout and frozen tables; default WORK/actor/sglang-hf)")
    t.add_argument('--pool', action='append', help='task pool cases.jsonl (repeatable; default: POOLS)')
    t.add_argument('--held-manifest', default=HELD)
    t.add_argument('--exclude-manifest', action='append', default=[], help='more held manifests to exclude')
    t.add_argument('--task-ids', help='JSON list: train only on these pool ids (smoke)')
    t.add_argument('--group', type=int, default=8)
    t.add_argument('--slots', type=int, default=96, help='concurrent episodes on the actor (its default runs 128 sequences)')
    t.add_argument('--min-groups', type=int, default=24,
                   help='groups per step at least (run2 at 12: about 75 episodes with a nonzero advantage per step, '
                        'consecutive updates did not add up; DECISIONS 2026-11-13)')
    t.add_argument('--learner-phase-slots', type=int, default=8,
                   help='collector slots that take new episodes while the learner computes (actor decode and the learner '
                        'share the GB10; in-flight episodes always finish)')
    t.add_argument('--max-groups', type=int, default=48,
                   help='groups per step at most: about 27k forward tokens per episode (smoke), so 48 groups is ~60-90 min of learner')
    t.add_argument('--max-outstanding', type=int, default=0, help='groups in flight or waiting (default max-groups + slots/group)')
    t.add_argument('--max-staleness', type=int, default=2)
    t.add_argument('--case-seconds', type=int, default=3600, help='wall-clock safety net only (a wall stop is unscored)')
    t.add_argument('--max-model-requests', type=int, default=EPISODE_MODEL_CALLS,
                   help='episode budget in model calls for training and the held evals alike; a budget stop scores as a '
                        'failure (EPISODE_COST_20261112)')
    t.add_argument('--pass-rates', default=resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'exp1-rft', 'base-pass-rates.json'),
                   help="measured base pass rates per task (re-read every 10 min; a task's own latest group replaces its "
                   'rate); empty: own outcomes only')
    t.add_argument('--pass-band', type=float, nargs=2, default=(0.1, 0.9))
    t.add_argument('--pass-min-scored', type=int, default=2)
    t.add_argument('--explore', type=float, default=0.4,
                   help='share of groups from unmeasured pool tasks (TASK_SUPPLY_20261113: a learnable-task group leaves '
                        'the band at 0/8 or 8/8 about 22%% of the time and an explored group enters it about 40%%, so '
                        'below ~0.35 the learnable set shrinks)')
    t.add_argument('--neuralese-variants', help='Neuralese-declaring variants of pool tasks (cases.jsonl from '
                   'scripts/rl_tasks/neuralese_variants.py; off by default: a controlled addition)')
    t.add_argument('--neuralese-variant-share', type=float, default=0.5,
                   help='share of the groups of a task with a variant that run the variant')
    t.add_argument('--reward', choices=('graded', 'binary'), default='graded',
                   help='advantage reward: graded (task_rollouts.graded_reward: partial credit for the file effects an '
                        'episode with only answer/file checks failed achieved; owner 2026-11-13) or binary acceptance. '
                        'Task pass rates for selection stay binary')
    t.add_argument('--exclude-ids', action='append', default=[],
                   help='JSONL of pool rows never selected ({"id": ...} per line; wrong oracles, garbled tasks); '
                        'default: every ops/pool-errata/*.jsonl')
    t.add_argument('--no-default-errata', action='store_true', help='do not exclude the ops/pool-errata rows by default')
    t.add_argument('--exclude-family-regex', default=r'^ext_codebase-(py|natlang)_codebase_',
                   help='families never selected (default: codebase families whose repos are missing on the DGX)')
    t.add_argument('--max-turns', type=int, default=128)
    t.add_argument('--decision-batch-tokens', type=int, default=16384,
                   help='sampled decisions are scored in packed batches of about this many tokens (all options of each)')
    t.add_argument('--max-sequence-tokens', type=int, default=65536,
                   help='longest learner sequence; calls beyond it are dropped and counted (calls_too_long_dropped): at 131k a '
                        'fully checkpointed sequence saves ~16 GB of layer inputs beside the actor')
    t.add_argument('--context-tokens', type=int, default=131072, help="call budget; follows the actor's served window when smaller")
    t.add_argument('--seed', type=int, default=20261112)
    t.add_argument('--seed-base', type=int, default=30_000_000)
    t.add_argument('--lr', type=float, default=1e-5,
                   help='check 4 (LEARNING_SANITY) learned at 1e-5; run2 at 3e-6: half of the BF16 flips reverted (8.0%% changed '
                        'vs start after 5 steps, 16%% summed) and mean |dlogp| vs start stayed at 0.014-0.023, under the '
                        "actor's FP8 noise (0.033)")
    t.add_argument('--warmup', type=int, default=2)
    t.add_argument('--weight-decay', type=float, default=0.)
    t.add_argument('--clip', type=float, default=1.0)
    t.add_argument('--kl-beta', type=float, default=0.02)
    t.add_argument('--rho-cap', type=float, default=2.0)
    t.add_argument('--loss-norm', choices=('episode', 'token'), default='episode',
                   help='episode: each trained episode contributes A_i * (mean over its trained units of w * logp), averaged '
                        'over the step\'s episodes, the KL term the same way; token: one sum over all units / all units (until '
                        '2026-11-13: an episode weighed by its length, failures are 1.7x longer, 97%% of the mass on reasoning)')
    t.add_argument('--top-p', type=float, default=RL_TOP_P,
                   help='nucleus of the sampling law at temperature 1, for collection, the held evals and the learner alike '
                        '(common.sampling.RL_TOP_P; 1.0: untruncated). The policy term scores the sampled token under the '
                        "learner's own nucleus (lfm3_fast_train.linear_nll_nucleus); calls sampled under another top_p are "
                        'refused (skip_other_sampling_law)')
    t.add_argument('--tail-logp-floor', type=float, default=-7.,
                   help='where the actor sampled a token (or decision option) with log-probability below this and the '
                        "episode's advantage is positive, the policy term has zero weight (the KL term stays); 0: off")
    t.add_argument('--decision-weight', choices=('call', 'token'), default='call',
                   help="units of one sampled decision in its episode's normalization. call: the episode's mean sampled "
                        'tokens per trained call (a decision is one model call whose reply is one categorical action, so it '
                        'weighs like an average call of its episode; 1 if the episode trains no token call); token: 1 (until '
                        '2026-11-13: one decision in ~10k tokens)')
    t.add_argument('--activation-gb', type=float, default=12., help='saved activations per sequence; layers beyond are recomputed')
    t.add_argument('--max-steps', type=int, default=0)
    t.add_argument('--save-every', type=int, default=2)
    t.add_argument('--state-every', type=int, default=4)
    t.add_argument('--state-dir', default=resolve_str('data_hdd', 'natlang-grpo-v1'))
    t.add_argument('--checkpoint-dir', help='pruned native checkpoints (default STATE_DIR/checkpoints)')
    t.add_argument('--resume', action='store_true')
    t.add_argument('--keep-rollouts', action='store_true', help='keep group directories (and every actor store block)')
    t.add_argument('--rollouts-dir', help="each lowered group's rollouts.jsonl, zstd-compressed, before its directory is "
                   "pruned (default: CHECKPOINT_DIR/../rollouts; '-': none)")
    t.add_argument('--block-gc-grace', type=float, default=600.,
                   help='after each step delete actor store blocks unused since this many seconds before the oldest '
                   'outstanding group was created (negative: never)')
    t.add_argument('--no-actor-sync', action='store_true',
                   help='never update the actor (smoke against a shared actor of the start policy; versions advance nominally)')
    t.add_argument('--unpinned', action='store_true',
                   help='run from a live checkout instead of a pinned copy (smokes only; see check_pin)')
    t.add_argument('--held-every-hours', type=float, default=4.)
    t.add_argument('--skip-base-held', action='store_true')
    t.add_argument('--held-baseline', help="a start-policy held result.json to compare with (e.g. held-v2's base run) instead "
                   'of measuring step 0 (compare refuses another engine or series)')
    t.add_argument('--held-samples', type=int, default=2, help='samples per held task at the periodic measurements')
    t.add_argument('--held-samples-start', type=int, default=4, help='samples per held task for the start policy (step 0)')
    t.add_argument('--held-extra', help='comma-separated native checkpoints measured on the actor in place after step 0, '
                   'each compared with step 0, before training starts')
    t.add_argument('--held-samples-extra', type=int, default=2)
    t.add_argument('--held-workers', type=int, default=96, help='collector slots of a held measurement on the training actor')
    t.add_argument('--held-remote', choices=('pop',), help="measure held on Pop's GPU in parallel with training instead of "
                   'pausing collection on the training actor (pop_held_loop)')
    t.add_argument('--held-remote-slots', type=int, default=14,
                   help="episodes in flight against Pop's actor: about what its KV pool holds (FP8: ~180k tokens, 11-13k per "
                        'call), so waiting requests do not evict the prefixes of episodes between calls (BF16 at 20 slots: '
                        '41%% prefix hit, 13.7 requests waiting)')
    t.add_argument('--pop-kv-cache-dtype', choices=('fp8_e4m3', 'auto'), default='fp8_e4m3',
                   help="Pop actor's KV dtype (auto = BF16, its engine until 2026-11-13); part of the attested engine: "
                        'a start-policy result of another setting is moved aside and re-measured')
    t.add_argument('--pop-max-running', type=int, default=16, help="Pop actor's concurrent sequences (capacity)")
    t.add_argument('--unit-prefix', default='grpo')
    t.add_argument('--synthetic-reward', help='sanity check: replace the task outcome by a trivially learnable reward '
                   '(short-calls:N = mean sampled tokens per model call under N; short-calls = minus that mean / 1000); the task '
                   'outcome is logged beside it')
    t.add_argument('--probe-turns', help='turns pickle (lower_rollout_turns) of fixed probe chains: their NLL is logged on the '
                   'learner every step and on the actor after every weight sync')
    t.add_argument('--probe-chains', type=int, default=4)
    t.add_argument('--probe-max-tokens', type=int, default=8000)
    args = parser.parse_args()
    if args.command == 'train':
        args.top_p = check_top_p(args.top_p)
        from natlang_neuralese.common.cuda_context import ensure_context
        ensure_context()   # GB10: a context can fail to create under host-memory fragmentation; retried
        args.pool = args.pool or POOLS
        args.max_outstanding = args.max_outstanding or args.max_groups + -(-args.slots // args.group)
        train(args)


if __name__ == '__main__':
    main()
