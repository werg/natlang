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
  into one sequence as in ``exp0_sft.chains_of``). A consumer call's imported Neuralese state is a differentiable
  pass-through: computed in the consumer's own forward from the producer document the episode recorded (the text write's
  ``[produce_open] + text ids``, ``lfm3_fast_train.passthrough_plan``), so the consumer's loss reaches the weights that
  computed the state (``--no-neuralese-passthrough``: the same state detached; a block without a recorded document is the
  actor's stored state, ``lfm3_fast_train.import_tensors``). The producer's own sampled tokens train as its own calls. Loss on sampled tokens only:
  ``-min(rho, C) * A * logp + beta * k2(ref || policy)``, ``rho = exp(logp_learner - logp_actor)`` (truncated importance
  sampling against the actor's own sampling logprobs), ``A`` the group-normalized episode reward on every call of the
  episode, the reference the start policy. Each trained episode weighs the same: its terms are averaged over its units
  (sampled tokens; a sampled decision counts as one average call of its episode), then over the step's episodes
  (``--loss-norm token``: one sum over all tokens). A tail token the actor sampled below ``--tail-logp-floor`` gets no
  policy term in an episode with a positive advantage. Sampling is top-p (``--top-p``, ``common.sampling.RL_TOP_P``): the
  policy term's ``logp`` is the sampled token's log-probability under the learner's own nucleus
  (``lfm3_fast_train.linear_nll_nucleus``; the sampled token always in the set), while ``rho`` and the KL term use the
  untruncated log-probabilities (the actor records the untruncated one; DECISIONS 2026-11-13 for the reasoning).
* weight sync: after each step the BF16 weights are copied once into the actor's HF safetensors on tmpfs, memory-mapped
  and page-locked (``WeightSync``; same layout as ``export/native_hf``; frozen rows and producer table kept), loaded in place
  (SGLang ``pause_generation`` retract -> ``update_weights_from_disk`` -> ``continue_generation``), in a background thread.
  Episodes keep running across a sync; each token's actor logprob is from the weights that sampled it.
* checkpoints: a pruned native checkpoint (``exp0_sft.write_checkpoint``, servable by ``serve_native_checkpoint``) every
  ``--save-every`` steps and the FP32 training state (masters + AdamW moments) every ``--state-every`` steps, both written
  from host memory in background threads (training never waits on I/O). Keeps the last two native checkpoints plus
  every held-evaluated one, and the last training state.
* memory (the unit's budget covers GPU and host alike on the GB10): the KL reference is a file loaded into the policy's
  own parameters for the reference pass (``ReferenceWeights``), the FP32 gradients exist only from the policy pass to the
  optimizer step, imported states are read per sequence, and ``train.jsonl`` logs each learner phase's CUDA peak and
  this process's host memory (``cuda_peak_by_phase_gb``).
* held evaluation: the start policy first, then every ``--held-every-hours`` the newest native checkpoint is measured on
  the held set (``python -m natlang_neuralese.rl.held_eval measure`` on the training actor, collection paused) and
  compared with the start policy (``held_eval compare`` -> ``held/step-N/compare-vs-step0.json``). With ``--held-remote pop``
  the measurements run on Pop's GPU instead, in parallel with training (``pop_held_loop``; the start policy as
  ``held/step-00000-pop``), and collection never pauses.
* task selection: ``--explore`` of groups from unmeasured pool tasks, the rest cycling tasks whose pass rate lies in
  ``--pass-band``: the run's own latest scored group of a task where it has one (current policy), else the measured base
  rate (``--pass-rates``). ``--neuralese-variants`` (off by default): a picked task that has a Neuralese-declaring variant
  (``scripts/rl_tasks/neuralese_variants.py``) runs as that variant in ``--neuralese-variant-share`` of its groups. The
  unmeasured-task walk prefers tasks declaring a construct the held set tests and caps dataset wrappers at
  ``--explore-wrapper-share`` of explored groups (``ExploreWalk``).
* dynamic sampling (``--dynamic-sampling-max-spread``): a group of a task likely to pass or fail every sample runs a first
  wave of k samples; when they agree and the posterior chance that the rest would still give reward spread is small, the
  rest never start (``wave_plan``, ``PassRatePrior``, ``Collector._after_wave``) and the slots go to other groups.
* self-distillation: ``--sd-weight`` replays the run's own rare verified successes; ``--sd-teacher-weight`` adds verified
  teacher-pool successes of tasks in the pass band or all-fail as a ratio-free term (``TeacherDemos``).

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
import statistics
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
MEMETIC_TASKS = resolve_str('data_nvme', 'memetic-tasks-20261118', 'cases.jsonl')
POOLS = [resolve_str('data_hdd', 'natlang-rl-composed-20261103', 'memetic-program-source-20261105', 'cases.jsonl'),
         # fallow K1: bgkit KR reader families, 4,734 rows verified by reference/negative/alias replay (verification.json)
         resolve_str('data_hdd', 'natlang-rl-bgkit-kr-20261103', 'natlang-rl-bgkit-kr-20261103', 'cases.jsonl'),
         # 2026-11-17 task supply (TASK_SUPPLY_20261117.md): 447 authored tasks (21 families, 16 new) and
         # 908 tau-bench/APIGen-MT store-service tasks; --pool with only the two pools above opts out
         resolve_str('data_hdd', 'natlang-rl-task-supply-20261117', 'cases.jsonl')]
# held-v3 (2026-11-13; plans/neuralese/HELD_V3_20261113.md): held-v2 with its wrong/ambiguous-oracle and exposed tasks
# replaced; its exclusion also covers held-v2 (manifest `also_excludes`), so no v2 held task enters training either.
HELD = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'held-v3', 'manifest.json')
# Start-policy calls scored after every step (behaviour_probe.py; built from grpo-v2-run1's version-0 groups)
BEHAVIOUR_PROBE = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'grpo-v2', 'behaviour-probe-v1.json')
# sft_sources WORKs of verified teacher successes (TeacherDemos): teacher-gen pool-v1 and pool-v2-rich replayed under
# the 2026-11-18 runtime first, then the 2026-11-17 SFT corpus WORK (pool-v1, replayed at aa1bd8467)
SD_TEACHER_SOURCES = [resolve_str('data_nvme', 'sft-teacher-demos-20261118'), resolve_str('data_nvme', 'sft-sources-20261117')]
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


def held_complete(out):
    """A finished held measurement in ``out``: a cut ``held_eval measure`` also writes result.json (complete false), and a
    loop that took that for done would compare against a 20-episode start policy (grpo-v2-run1, 2026-11-13)."""
    try:
        return bool(json.loads((Path(out) / 'result.json').read_text()).get('complete'))
    except (OSError, ValueError):
        return False


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
        self.walk = ExploreWalk(tasks, getattr(args, 'explore_wrapper_share', 1.), getattr(args, 'explore_held_weight', 1.))
        if work is not None:
            (Path(work) / 'explore-constructs.json').write_text(json.dumps(self.walk.pool_report(), indent=1) + '\n')
        self.learnable_ids, self.learnable_order, self.measured, self.rates_read = [], [], set(), 0.
        self.base_rates, self.own, self.rates_changed = {}, {}, True
        self.variants = {}                          # original task id -> its Neuralese-declaring variant row
        # dynamic sampling (``wave_plan``): the pass-rate prior, rebuilt at most every 2 min after the rates change
        self.prior, self.prior_built, self.prior_stale = None, 0., True
        self.dynamic = collections.Counter()        # groups split / stopped / released, samples saved (train.jsonl)
        for directory in sorted((work / 'groups').glob('g*')):   # a resumed run keeps its own measurements
            self._record_outcome(directory)
        self.rng = random.Random(args.seed + first_group)
        self.stop = threading.Event()
        self.paused = threading.Event()             # set during a held measurement on the training actor
        self.active_slots = args.slots
        self.running = 0                            # episodes in flight (their node work and KV stay live)
        self.generated = 0                          # sampled tokens of finished episodes (throughput)
        self.episodes = 0
        self.env = pinned_env()
        from natlang_neuralese.rl.held_eval import series
        self.env['NATLANG_MODEL_FACING'] = series()['model_facing']
        from lower_rollout_turns import NativeStates
        self.states = NativeStates(args.actor_store or work / 'actor')   # imported Neuralese state of consumer calls
        self.custody, self.trajectories = None, None
        self.memetic_rows, self.memetic_cursor = [], 0   # "improve program P" rows (--memetic-tasks)
        if getattr(args, 'memetic_tasks', None) and getattr(args, 'memetic_share', 0) > 0:
            from natlang_neuralese.common.jsonio import read_jsonl
            self.memetic_rows = list(read_jsonl(args.memetic_tasks))
            random.Random(args.seed).shuffle(self.memetic_rows)
        # the memetic world's Python (ts-host world-bridge.ts kind memetic): this interpreter, not the pin's missing venv
        self.env['NATLANG_MEMETIC_PYTHON'] = sys.executable
        if getattr(args, 'teacher_attempts', 0) > 0:   # JOLT+ teacher attempts (``_teachers``)
            from natlang_neuralese.common.custody import ProtectedWorlds
            from natlang_neuralese.rl.jolt import teacher_fact_kinds
            teacher_fact_kinds(args.teacher_fact_kinds)   # refuse unknown kinds before the first group
            self.custody = ProtectedWorlds.load()      # every protected source, failing closed (~14 s)
            if getattr(args, 'hindsight_trajectories', None):
                from natlang_neuralese.rl.hindsight import load_index
                self.trajectories = load_index(args.hindsight_trajectories)

    def _new_group(self):
        from natlang_neuralese.rl import task_rollouts
        from natlang_neuralese.common.jsonio import read_jsonl
        a = self.args
        memetic = bool(self.memetic_rows) and self.rng.random() < a.memetic_share
        if memetic:
            row = self.memetic_rows[self.memetic_cursor % len(self.memetic_rows)]
            self.memetic_cursor += 1
        else:
            row = self._pick()
            variant = self.variants.get(row['id'])
            if variant is not None and self.rng.random() < self.args.neuralese_variant_share:
                row = variant   # its outcomes are its own (never the original's band rate)
        group = self.next_group
        self.next_group += 1
        directory = self.work / 'groups' / f'g{group:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        if memetic:
            row = self._bind_memetic(row, directory)
        (directory / 'cases.jsonl').write_text(json.dumps(row) + '\n')
        task_rollouts.prepare(SimpleNamespace(
            cases=directory / 'cases.jsonl', out=directory, rows=1, offset=0,
            group=a.memetic_group if memetic else a.group, temperature=1.,
            case_seconds=a.case_seconds, max_model_requests=a.memetic_model_requests if memetic else a.max_model_requests,
            max_turns=a.max_turns,
            seed_base=a.seed_base + 100 * group, root_mode='student', match_seeds_from=None,
            policy_checkpoint_sha256=self.policy_sha))
        entries = list(read_jsonl(directory / 'queue.jsonl'))
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=a.context_tokens, rl_endpoint=a.actor)
        (directory / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
        wave = None if memetic else self._wave(row['id'], len(entries))
        self.groups[group] = {'dir': directory, 'size': len(entries), 'done': 0, 'versions': [], 'task_id': row['id'],
                              'created': time.time(), 'wave': wave, 'held_back': entries[wave:] if wave else []}
        self.outstanding += 1
        self.pending.extend((group, entry) for entry in entries[:wave])

    def _wave(self, task_id, size):
        """Dynamic sampling: the first-wave size of a new group of ``task_id`` (``wave_plan`` on its ``PassRatePrior``),
        or None to start all its samples at once. The rest wait until the whole first wave has finished, so the decision
        reads a fixed set of samples (never the first to finish, which are biased to short episodes) and a cancelled
        sample was never started (nothing in flight to kill)."""
        a = self.args
        limit = getattr(a, 'dynamic_sampling_max_spread', 0.)
        if not limit or limit <= 0 or size < 4:
            return None
        if self.prior is None or (self.prior_stale and time.time() - self.prior_built >= 120):
            rates = {k: (v['successes'], v['scored']) for k, v in self.base_rates.items() if v.get('scored')}
            rates.update(self.own)
            family = {k: v.get('family') for k, v in self.base_rates.items()}

            def family_of(task_id):
                row = self.by_id.get(task_id)
                return row.get('family') if row is not None else family.get(task_id)
            self.prior, self.prior_built, self.prior_stale = PassRatePrior(rates, family_of), time.time(), False
        wave, _, _ = wave_plan(*self.prior.beta(task_id), size, limit)
        if wave:
            self.dynamic['groups_split'] += 1
        return wave

    def _after_wave(self, group):
        """The first wave of a split group has finished: cancel its held-back samples when the wave agrees (every sample
        scored, one binary outcome and one advantage reward) and the posterior chance that the rest would still give
        the group reward spread is at most ``--dynamic-sampling-max-spread``; otherwise start them, ahead of every other
        group. A cancelled group is scored with the samples it ran (its task's pass rate updates from them) and its
        queue keeps only those; the cancelled entries go to ``cancelled.jsonl``. Returns True when the group is complete."""
        from natlang_neuralese.common.jsonio import read_jsonl
        from natlang_neuralese.rl.task_rollouts import entry_outcome
        info = self.groups[group]
        directory = info['dir']
        held = {entry['key'] for entry in info['held_back']}
        stop = False
        try:
            entries = [e for e in read_jsonl(directory / 'queue.jsonl') if e['key'] not in held]
            packet = list(read_jsonl(directory / 'packet.jsonl'))
            outcomes = []
            for entry in entries:
                journal = directory / 'workers' / f"{entry['key']}.journal.jsonl"
                finish = next((r for r in (read_jsonl(journal) if journal.exists() else ())
                               if r.get('event') == 'finish' and r.get('key') == entry['key']), None)
                outcomes.append(entry_outcome(entry, packet[entry['index']], finish))
            binary = {reward for _, reward, _ in outcomes}
            graded = {graded for _, _, graded in outcomes}
            advantage_rewards = binary if getattr(self.args, 'reward', 'graded') == 'binary' else graded
            if len(binary) == 1 and None not in binary and len(advantage_rewards) == 1 and self.prior is not None:
                success = binary.pop() == 1.0
                a, b = self.prior.beta(info['task_id'])
                chance = spread_after(a, b, info['size'], len(entries), success)
                stop = chance <= self.args.dynamic_sampling_max_spread
        except Exception as error:   # an unreadable wave runs the rest: never a group lost to the shortcut
            print(json.dumps({'event': 'dynamic sampling check failed', 'group': group, 'error': repr(error)[:300]}),
                  flush=True)
        with self.lock:
            if info.get('abandoned'):
                return False
            if stop:
                cancelled = info['held_back']
                with (directory / 'cancelled.jsonl').open('w') as handle:
                    handle.write(''.join(json.dumps(e, sort_keys=True) + '\n' for e in cancelled))
                (directory / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
                info['size'], info['held_back'] = info['done'], []
                self.dynamic['groups_stopped'] += 1
                self.dynamic['samples_saved'] += len(cancelled)
                self.dynamic['stopped_all_' + ('pass' if success else 'fail')] += 1
                return True
            self.pending.extendleft((group, entry) for entry in reversed(info['held_back']))
            info['held_back'] = []
            self.dynamic['groups_released'] += 1
            return False

    def _bind_memetic(self, row, directory):
        """A memetic row bound to this group (RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md section 3): its episode spec,
        copied into the group directory with the run's actor, runtime and sampling controls, so the group's samples share
        one preparation (evidence and anchor baselines) measured under the policy version that samples them."""
        import copy
        import hashlib
        world = row['semantics']['world']
        spec = json.loads(Path(world['task']).read_text())
        spec.update(out=str(directory / 'memetic'), endpoint=self.args.actor, runtime=str(self.runtime),
                    sampling_config=str(self.work / 'sampling-config.json'))
        path = directory / 'memetic-spec.json'
        path.write_text(json.dumps(spec, ensure_ascii=False, sort_keys=True) + '\n')
        row = copy.deepcopy(row)
        row['semantics']['world'] = {**world, 'task': str(path), 'spec_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return row

    def _record_outcome(self, directory):
        """The task's latest scored group (``--group`` samples of the policy then served) replaces any earlier measurement
        of it: the selection follows the current policy's own outcomes, so a task it solves 0/8 or 8/8 leaves the band and
        an explored task it solves sometimes enters it."""
        from natlang_neuralese.common.jsonio import read_jsonl
        try:
            rows = list(read_jsonl(directory / 'rewards.jsonl'))
        except OSError:
            return
        scored = [row for row in rows if row.get('reward') is not None and row.get('jolt_role', 'student') == 'student']
        if scored:
            self.own[scored[0]['task_id']] = (sum(row['reward'] == 1.0 for row in scored), len(scored))
            self.rates_changed = self.prior_stale = True

    def _learnable(self):
        """Pool rows whose pass rate lies in ``--pass-band`` with at least ``--pass-min-scored`` scored samples, shuffled:
        the run's own latest group of a task (``_record_outcome``) where it has one, else the measured base rate
        (``--pass-rates``, re-read every 10 min)."""
        a = self.args
        if a.pass_rates and time.time() - self.rates_read >= 600:
            self.rates_read = time.time()
            try:
                self.base_rates = json.loads(Path(a.pass_rates).read_text())
                self.rates_changed = self.prior_stale = True
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
        row, self.cursor = self.walk.next(self.cursor, self.measured, self.rng)
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
                with self.lock:
                    self.running += 1
                try:
                    while process.poll() is None:
                        if self.stop.wait(2):
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait()
                            return
                finally:
                    with self.lock:
                        self.running -= 1
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
                wave_done = bool(info.get('held_back')) and info['done'] == info['wave']
                complete = info['done'] == info['size'] and not info.get('held_back')
            if wave_done:
                complete = self._after_wave(group)
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
        if self._teachers(group):
            return   # scored again with its teacher attempts once they finish
        self.ready.put(group)

    def _teachers(self, group):
        """JOLT+ (RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md section 1): a group whose students succeeded at most
        ``--teacher-max-successes`` times gets ``--teacher-attempts`` privileged teacher attempts of the same task, queued
        first. Each runs with the task's custody-checked facts of ``--teacher-fact-kinds`` (or, with
        ``--hindsight-trajectories``, a verified run of it) for the in-run guidance writer; the group reaches the learner
        when they finish. Returns the number of attempts queued (0: the group is complete)."""
        from natlang_neuralese.common.jsonio import read_jsonl
        from natlang_neuralese.rl import jolt
        a, info = self.args, self.groups[group]
        if getattr(a, 'teacher_attempts', 0) <= 0 or 'teachers' in info:
            return 0
        info['teachers'] = 0
        directory = info['dir']
        students = [row for row in read_jsonl(directory / 'rollouts.jsonl') if row.get('jolt_role', 'student') == 'student']
        scored = [row for row in students if row.get('reward') is not None and not row.get('unscored')]
        if not scored or sum(row['reward']['quality'] == 1.0 for row in scored) > a.teacher_max_successes:
            return 0
        entries = [entry for entry in read_jsonl(directory / 'queue.jsonl') if entry.get('jolt_role', 'student') == 'student']
        result = jolt.teacher_entries(
            entries, scored, cases=directory / 'packet.jsonl', config=self.work / 'sampling-config.json', out=directory,
            attempts=a.teacher_attempts, seed_base=a.seed_base + 100 * group, custody=self.custody,
            selection=a.teacher_selection, writer='native', fact_kinds=jolt.teacher_fact_kinds(a.teacher_fact_kinds),
            plan_command=jolt.plan_command(a.actor) if a.teacher_plan == 'nl' else None, trajectories=self.trajectories)
        teachers = result['entries']
        if not teachers:
            print(json.dumps({'event': 'no teacher attempts', 'group': group, 'refusal': result.get('refusal')}), flush=True)
            return 0
        for entry in teachers:
            entry['rl_endpoint'] = a.actor
        with (directory / 'queue.jsonl').open('a') as handle:
            handle.write(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in teachers))
        with self.lock:
            info['teachers'] = len(teachers)
            info['size'] += len(teachers)
            self.pending.extendleft((group, entry) for entry in reversed(teachers))
        print(json.dumps({'event': 'teacher attempts', 'group': group, 'task_id': info['task_id'],
                          'attempts': len(teachers), 'hindsight': sum(bool(e.get('hindsight')) for e in teachers),
                          'plan': result.get('plan')}), flush=True)
        return len(teachers)

    def reuse_scored(self, step, max_staleness, memetic_max_staleness=None):
        """Offer again the scored groups a stop left untrained (rollouts and versions on disk, not ``retired``) whose
        episodes were all sampled by policies this resumed run has served (versions <= ``step``) within the staleness
        the learner accepts. A group sampled by a version past the saved state (a policy that is gone) or too stale is
        retired, so a later resume whose own steps reach that version number never takes it for its own.
        Returns (reused, left out)."""
        reused, skipped = [], []
        for directory in sorted((self.work / 'groups').glob('g*')):
            if not directory.name[1:].isdigit() or (directory / RETIRED).exists():
                continue
            if not all((directory / name).exists() for name in ('rollouts.jsonl', 'versions.json', 'score.json')):
                continue
            try:
                versions = json.loads((directory / 'versions.json').read_text())
                task_id = json.loads((directory / 'rewards.jsonl').read_text().splitlines()[0])['task_id']
            except (OSError, ValueError, KeyError, IndexError):
                skipped.append(directory.name)
                continue
            limit = (memetic_max_staleness if memetic_max_staleness is not None and (directory / 'memetic-spec.json').exists()
                     else max_staleness)
            if not versions or max(versions) > step or step - min(versions) > limit:
                (directory / RETIRED).touch()
                skipped.append(directory.name)
                continue
            group = int(directory.name[1:])
            with self.lock:
                self.groups[group] = {'dir': directory, 'size': len(versions), 'done': len(versions), 'versions': versions,
                                      'task_id': task_id, 'created': directory.stat().st_mtime, 'reused': True}
                self.outstanding += 1
            self.ready.put(group)
            reused.append(directory.name)
        return reused, skipped

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


class ReplayCollector:
    """``--replay``: a finished run's archived groups (``ROLLOUTS/<group>.rollouts.jsonl.zst``) offered in group order,
    ``--min-groups`` a step, each as fresh (no staleness), for learner controls (``--advantage-control``): no actor, no
    collection, no held eval. Consumer calls read the old run's actor store (``--actor-store``), which is never pruned here.
    ``--replay-max-group-tokens`` keeps only short groups (the learner bench). ``--replay-schedule`` (a JSON list of
    group-number lists) offers exactly those batches, one a step: the old run's own
    steps, for re-executing its learner on its data. A resumed replay (a branch) continues the stream after the groups its
    state's run had taken (``first_group``)."""

    def __init__(self, args, work, first_group=0):
        from lower_rollout_turns import NativeStates
        self.args, self.work = args, work
        self.archives = sorted(Path(args.replay).glob('g*.rollouts.jsonl.zst'))
        self.by_group = {int(a.name[1:].split('.')[0]): a for a in self.archives}
        self.archives = [a for g, a in sorted(self.by_group.items()) if g >= first_group]
        schedule = getattr(args, 'replay_schedule', None)
        self.schedule = json.loads(Path(schedule).read_text()) if schedule else None
        self.groups, self.lock, self.stop, self.paused = {}, threading.Lock(), threading.Event(), threading.Event()
        self.version = self.generated = self.episodes = self.outstanding = self.cursor = self.next_group = 0
        self.active_slots, self.variants, self.running = 0, {}, 0
        self.states = NativeStates(args.actor_store)

    def take_groups(self, minimum, maximum, stop=None):
        import zstandard
        taken = []
        if self.schedule is not None:
            batch = self.schedule.pop(0) if self.schedule else []
            self.archives = [self.by_group[g] for g in batch]
            minimum = len(batch) or 1
        while self.archives and len(taken) < minimum:
            source = self.archives.pop(0)
            group = int(source.name[1:].split('.')[0])
            directory = self.work / 'groups' / source.name.split('.')[0]
            directory.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as reader, (directory / 'rollouts.jsonl').open('wb') as writer:
                zstandard.ZstdDecompressor().copy_stream(reader, writer)
            with (directory / 'rollouts.jsonl').open() as handle:
                rows = [json.loads(line) for line in handle]
            limit = getattr(self.args, 'replay_max_group_tokens', 0)
            if limit and self.schedule is None and sum((turn.get('usage') or {}).get('completion_tokens', 0)
                                                       for row in rows for turn in row.get('turns') or ()) > limit:
                shutil.rmtree(directory)          # the learner bench: short groups only
                continue
            task_id = rows[0]['task_id']
            self.groups[group] = {'dir': directory, 'task_id': task_id, 'versions': [self.version], 'created': time.time()}
            self.next_group = group + 1
            taken.append(group)
        if len(taken) < minimum and stop is not None:
            stop.set()                     # the archive is used up: no short step
        return taken


def control_advantages(items, episode_advantage, mode, rng):
    """``--advantage-control``: 'zero' gives every trained episode advantage 0 (only the KL term remains: the policy must
    stay put), 'shuffle' permutes the step's episode advantages across episodes (the update keeps its size but no longer
    follows the reward: it must not move the policy in a consistent direction). ``items`` are the step's chains and
    decisions, each with its 'episode' key. Returns the episode -> advantage map used."""
    keys = sorted(episode_advantage)
    values = [0. if mode == 'zero' else episode_advantage[k] for k in keys]
    if mode == 'shuffle':
        rng.shuffle(values)
    new = dict(zip(keys, values))
    for item in items:
        item['advantage'] = new[item['episode']]
    return new


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
    from natlang_neuralese.rl.jolt import student_form
    info = collector.groups[group]
    episodes = []
    for row in read_jsonl(info['dir'] / 'rollouts.jsonl'):
        role = row.get('jolt_role', 'student')
        if role == 'teacher':   # JOLT+: the guided run's actions under the unguided prompt (rl/jolt.py student_form)
            row = student_form(row, stats)
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
        memetic = reward.get('memetic') or {}
        episodes.append({'reward': quality if binary or graded is None else float(graded),
                         **({'anchor': float(memetic['anchor'])} if 'anchor' in memetic else {}),
                         **({'memetic': memetic} if memetic else {}),
                         'quality': quality, 'rows': rows, 'decisions': decisions,
                         'task_id': row['task_id'], 'round': group,
                         'outcome': row.get('outcome'), 'role': role})
    return info['task_id'], episodes, info['versions']


def memetic_group_record(group, task_id, episodes):
    """Per memetic group (train.jsonl ``memetic_groups``): confirmation success on the group's query worlds with each
    sample's learned guidance (``with_guidance``, mean over samples whose workspace scored) against the same worlds
    and seeds with no guidance (``without``, the group's shared preparation), and ``gap`` = with - without. Telemetry
    to watch over a run, never a gate (owner 2026-11-18). None for an ordinary group."""
    memetic = [e['memetic'] for e in episodes if e.get('memetic')]
    if not memetic:
        return None
    scored = [m for m in memetic if not m.get('unscored_workspace')]
    without = next((m['without'] for m in scored if m.get('without') is not None), None)
    with_guidance = sum(m['query_success'] for m in scored) / len(scored) if scored else None
    return {'group': group, 'task_id': task_id, 'samples': len(memetic), 'scored': len(scored),
            'committed': sum(1 for m in scored if m.get('committed')),
            'with_guidance': None if with_guidance is None else round(with_guidance, 4),
            'without': without, 'default': next((m['default'] for m in scored if m.get('default') is not None), None),
            'gap': None if with_guidance is None or without is None else round(with_guidance - without, 4)}


def memetic_gap_history(path):
    """(sum, count) of the per-group gaps a run already logged (a resume continues the running mean)."""
    total = count = 0
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            for item in record.get('memetic_groups') or ():
                if item.get('gap') is not None:
                    total, count = total + item['gap'], count + 1
    return total, count


def memetic_step_record(groups, history):
    """Step-level summary of this step's memetic groups and the run's running mean gap (``history`` is the mutable
    [sum, count] over every group so far, updated here)."""
    gaps = [g['gap'] for g in groups if g['gap'] is not None]
    history[0] += sum(gaps)
    history[1] += len(gaps)
    return {'groups': len(groups), 'groups_with_gap': len(gaps),
            'gap_mean': round(sum(gaps) / len(gaps), 4) if gaps else None,
            'gap_running_mean': round(history[0] / history[1], 4) if history[1] else None,
            'gap_running_groups': history[1]}


def episode_advantages(episodes, stats=None):
    """Advantage of each episode of one group, None for an episode that does not train; None when nothing trains.
    Without teacher attempts: plain GRPO over the group (``group_advantages``). With them (JOLT+): students keep the plain
    advantage over the students alone and teachers get ``rl/jolt.py guided_advantages`` (their reward against the
    students' mean, never negative); a teacher with advantage 0 does not train."""
    anchors = [e['anchor'] for e in episodes if e.get('anchor') is not None]
    if anchors and len(episodes) >= 2:
        # memetic episodes: centred on max(group mean, the best trivial improvement), so a change set that does not beat
        # doing the trivial thing never earns a positive advantage (rl/memetic_workspace.py memetic_reward)
        rewards = [e['reward'] for e in episodes]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
        if std < 1e-6:
            return None
        base = max(mean, max(anchors))
        return [(r - base) / (std + 1e-6) for r in rewards]
    if not any(e.get('role') == 'teacher' for e in episodes):
        return group_advantages([e['reward'] for e in episodes]) if len(episodes) >= 2 else None
    from natlang_neuralese.rl.jolt import guided_advantages
    students = [e for e in episodes if e.get('role') != 'teacher']
    teachers = [e for e in episodes if e.get('role') == 'teacher']
    student, teacher = guided_advantages([e['reward'] for e in students], [e['reward'] for e in teachers])
    if stats is not None:
        stats['teacher_groups'] += 1
        stats['teacher_episodes'] += len(teachers)
        stats['teacher_successes'] += sum(e['quality'] == 1.0 for e in teachers)
        stats['teacher_episodes_trained'] += sum(1 for value in teacher or () if value > 0)
    if student is None and teacher is None:
        return None
    by_id = {id(e): value for e, value in zip(students, student or [None] * len(students))}
    by_id.update({id(e): (value or None) for e, value in zip(teachers, teacher or [None] * len(teachers))})
    return [by_id[id(e)] for e in episodes]


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


RETIRED = 'retired'   # a group that never trains again: lowered for the learner, or left out by a resume


def prune_group(collector, group, keep, archive=None, stats=None):
    """Forget a lowered group and delete its bulky files, keeping its score/rewards/versions/cases; its rollouts are
    first kept compressed in ``archive`` (a group whose archiving fails keeps everything in place)."""
    directory = collector.groups.pop(group)['dir']
    (directory / RETIRED).touch()     # handed to the learner: a resume never offers it again (reuse_scored)
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
        if path.name not in ('score.json', 'rewards.jsonl', 'versions.json', 'cases.jsonl', RETIRED):
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


EMBEDDING = 'model.embed_tokens.weight'
PRODUCER_TABLE = 'model.natlang_producer_embed'     # export.native_hf: the producing input embedding of every token


def producer_adapter_keys(interface_state):
    """A checkpoint's producer-adapter tensors (``SharedStateModel.adapter``): with one the producer table is not the
    embedding, and neither WeightSync nor the learner models it."""
    return sorted(k for k in interface_state if k.startswith('adapter.'))


class WeightSync:
    """The actor's HF export as one memory-mapped file on tmpfs: snapshot (GPU -> the mapped file, the only synchronous
    part; the mapping is page-locked with cudaHostRegister), then in a background thread load it into SGLang in place and
    optionally write a native checkpoint. The file is the only host copy of the weights: a separate pinned buffer that
    each publish wrote out to the file held another 4 GiB of the unit's memory (grpo-v2-run2, 2026-11-15). The file is
    rewritten only by ``snapshot``/``load_native``, which first wait for the publish thread (SGLang's load and the
    checkpoint write read it there).

    The producer table (``PRODUCER_TABLE``, the export's producing input embedding of every token,
    ``export.native_hf.producer_embeddings``: ``adapter(embed(t), producing)``) is published from the trained input
    embedding (``producer_from_embedding``; the checkpoints served here carry no producer adapter, so the table IS the
    embedding): kept at the export's start values it served stale rows to every Neuralese text write from step 1 on,
    while ordinary tokens read the trained ones (found 2026-11-17, NEURALESE_PASSTHROUGH_20261117.md)."""

    def __init__(self, model, export_dir, sglang_url, weights_dir, front_url=None, producer_from_embedding=True):
        import mmap
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
        path = self.weights_dir / 'model.safetensors'
        partial = path.with_name(path.name + '.partial')
        with safe_open(str(self.export_dir / 'model.safetensors'), 'pt') as handle:
            names = list(handle.keys())
            for name in names:
                info = handle.get_slice(name)
                shape = tuple(info.get_shape())
                if info.get_dtype() != 'BF16':
                    raise ValueError(f'export tensor {name} is {info.get_dtype()}, expected bf16')
                if name in params:
                    if tuple(params[name].shape) != shape:
                        raise ValueError(f'export/learner shape mismatch {name}')
                    sources.append(name)
                elif name == PRODUCER_TABLE and producer_from_embedding:
                    if tuple(params[EMBEDDING].shape) != shape:
                        raise ValueError(f'producer table {shape} differs from the embedding {tuple(params[EMBEDDING].shape)}')
                    sources.append(EMBEDDING)
                else:
                    sources.append(None)    # the producer table and anything else the learner does not train
                layout.append((name, 'BF16', shape, math.prod(shape) * 2))
            missing = set(params) - {name for name in sources if name} - {'lm_head.weight'}
            if missing:
                raise ValueError(f'learner parameters absent from the export: {sorted(missing)[:5]}')
            self.header, offsets, total = safetensors_header(layout, {'format': 'pt'})
            # every page is written once here (fixed tensors as exported, trained ones zero until the first snapshot):
            # a GPU copy into freshly faulted pageable memory runs at 0.05-0.1 GB/s on the GB10
            zeros = bytes(64 << 20)
            with partial.open('wb') as out:
                out.write(self.header)
                for (name, _, _, nbytes), source in zip(layout, sources):
                    if source is None:
                        out.write(memoryview(handle.get_tensor(name).contiguous().reshape(-1).view(torch.uint8).numpy()))
                    else:
                        for start in range(0, nbytes, len(zeros)):
                            out.write(zeros[:min(len(zeros), nbytes - start)])
        os.replace(partial, path)
        self.file = path.open('r+b')
        self.map = mmap.mmap(self.file.fileno(), 0)
        base = len(self.header)
        whole = torch.frombuffer(self.map, dtype=torch.uint8)
        self.pinned = False      # page-locked: GPU copies run async at ~35 GB/s (pageable, after the first touch: ~33)
        if torch.cuda.is_available():
            try:
                self.pinned = int(torch.cuda.cudart().cudaHostRegister(whole.data_ptr(), whole.numel(), 0).value) == 0
            except Exception:
                pass
        self.buffer = torch.frombuffer(self.map, dtype=torch.uint8, count=total, offset=base)
        # one tensor (its own storage) per trained weight, so that a native checkpoint saves them as separate clones did
        self.views = {name: torch.frombuffer(self.map, dtype=torch.bfloat16, count=nbytes // 2, offset=base + offset).view(shape)
                      for (name, _, shape, nbytes), offset, source in zip(layout, offsets, sources) if source is not None}
        self.sources = {name: source for (name, *_), source in zip(layout, sources) if source is not None}
        self.params = params
        self.thread = None
        self.last = {}

    def snapshot(self):
        import torch
        self.wait()
        for name, view in self.views.items():
            view.copy_(self.params[self.sources[name]].detach(), non_blocking=self.pinned)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def close(self):
        """Unregister the page-locked mapping (a registration outliving its mapping makes a later cudaHostRegister
        of the reused address range fail) and drop the views; the file stays."""
        import torch
        self.wait()
        if self.pinned:
            torch.cuda.cudart().cudaHostUnregister(torch.frombuffer(self.map, dtype=torch.uint8).data_ptr())
            self.pinned = False
        self.views = self.buffer = None

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
        merged = self.views[EMBEDDING][torch.tensor(special)]
        if not torch.equal(embedding[torch.tensor(special)].to(merged.dtype), merged):
            raise ValueError(f'{path} has other interface rows than the served policy')
        for name, view in self.views.items():
            source = embedding if self.sources[name] == EMBEDDING else backbone[name]
            view.copy_(source.to(view.dtype))
        del state

    def backbone(self):
        """{name: host bf16 tensor} of the snapshot in native backbone naming: views of the mapped file, valid until the
        next ``snapshot``/``load_native`` (both wait for the publish thread, where ``write_native`` reads them). Not copies:
        cloning them put another 4 GB on the unit during every checkpoint write."""
        return {name: view for name, view in self.views.items() if self.sources[name] == name}

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
            timing = {}         # the export file already holds the snapshot (the mapping)
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


CACHE_DROP_BYTES = 512 << 20


def stream_tensors_to_file(handle, tensors, staging, stream):
    """Write CUDA ``tensors``' bytes in order to the open binary ``handle``: GPU -> pinned ``staging`` -> file on ``stream``.
    Every ``CACHE_DROP_BYTES`` the written range is flushed and its page cache dropped: the cache is charged to the
    unit's cgroup, and on the GB10 a CUDA allocation does not reclaim it, so a 13-21 GB write left as many GB of
    cache beside the run (``StateSaver``, ``MomentOffload``)."""
    import torch
    chunk, written, dropped = staging.numel(), 0, 0
    fd = handle.fileno()
    with torch.cuda.stream(stream):
        for t in tensors:
            flat = t.detach().reshape(-1).view(torch.uint8)
            for start in range(0, flat.numel(), chunk):
                piece = flat[start:start + chunk]
                staging[:piece.numel()].copy_(piece, non_blocking=True)
                stream.synchronize()
                handle.write(memoryview(staging[:piece.numel()].numpy()))
                written += piece.numel()
                if written - dropped >= CACHE_DROP_BYTES:
                    handle.flush()
                    os.fdatasync(fd)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    dropped = written
    handle.flush()
    os.fdatasync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


class StateSaver:
    """FP32 masters and AdamW moments streamed GPU -> pinned staging -> file in a background thread. The caller waits
    (``wait``) before the next optimizer step changes them."""

    def __init__(self, staging_mb=64):     # the file write, not the copy, bounds the save: 512 MB staging bought nothing
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
            with partial.open('wb') as handle:
                handle.write(header)
                stream_tensors_to_file(handle, [tensors[name] for name in names], self.staging, self.stream)
            os.replace(partial, path)
            self.last = {'path': str(path), 'seconds': round(time.time() - started, 1)}
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


class MomentOffload:
    """AdamW's moments (exp_avg, exp_avg_sq: 13.6 GB FP32 for LFM3-1.7B) on disk between optimizer steps. They are read
    only by ``optimizer.step`` (and ``moment_cosine`` just before it), yet stayed resident through collection and the
    learner's reference and policy passes, where they were 13.6 of the unit's peak (grpo-v3-run1: learner policy phase
    48 GB reserved). ``put`` streams them GPU -> pinned staging -> file in a background thread (after an in-flight state
    save, which reads them) and then frees their storage; ``get`` reallocates and reads them back, bit for bit, before
    the next step. The tensors keep their identity (only the storage is resized), so the optimizer and the state saver
    see the same objects. Page cache of the file is dropped after each write and read."""

    def __init__(self, path, staging_mb=64):
        import torch
        self.path = Path(path)
        self.staging = torch.empty(staging_mb << 20, dtype=torch.uint8, pin_memory=True)
        self.stream = torch.cuda.Stream()
        self.thread, self.tensors, self.out, self.error, self.last = None, [], False, None, {}

    def wait(self):
        """The write in flight finished. A failed write leaves the moments resident (``out`` stays False; the error is in
        ``last``): the step continues exactly as without offload."""
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            self.last = {'put_error': repr(self.error)[:300]}
            self.error = None

    def put(self, tensors, after=None):
        """Write ``tensors`` (CUDA, contiguous) to the file and free them, in the background; ``after``: a thread-backed
        saver whose ``wait`` must return first (it reads the same tensors)."""
        import torch
        self.wait()
        torch.cuda.synchronize()
        self.tensors = list(tensors)

        def run():
            try:
                started = time.time()
                if after is not None:
                    after.wait()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open('wb') as handle:
                    stream_tensors_to_file(handle, self.tensors, self.staging, self.stream)
                self.out = True                 # the file is complete: from here on ``get`` restores from it
                self.last = {'put_s': round(time.time() - started, 1)}
                for t in self.tensors:
                    t.untyped_storage().resize_(0)
                torch.cuda.empty_cache()
                self.last['cuda_reserved_after_put_gb'] = round(torch.cuda.memory_reserved() / 2**30, 1)
            except BaseException as error:      # freed only after a complete write: the tensors stay resident
                self.error = error
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def get(self):
        """The moments back in place (a no-op when they are resident). Returns the seconds it took."""
        import torch
        started = time.time()
        self.wait()                         # a write still in flight (steps shorter than the write) is waited for here
        if not self.out:
            return time.time() - started
        self.last['get_wait_s'] = round(time.time() - started, 1)
        chunk = self.staging.numel()
        with self.path.open('rb', buffering=0) as handle, torch.cuda.stream(self.stream):
            for t in self.tensors:
                nbytes = t.numel() * t.element_size()
                t.untyped_storage().resize_(nbytes)
                flat = t.detach().reshape(-1).view(torch.uint8)
                for start in range(0, nbytes, chunk):
                    n = min(chunk, nbytes - start)
                    view = memoryview(self.staging[:n].numpy())
                    got = 0
                    while got < n:
                        read = handle.readinto(view[got:])
                        if not read:
                            raise RuntimeError(f'{self.path}: short read')
                        got += read
                    flat[start:start + n].copy_(self.staging[:n], non_blocking=True)
                    self.stream.synchronize()
                    if (start // chunk) % max(1, CACHE_DROP_BYTES // chunk) == 0:
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        torch.cuda.current_stream().wait_stream(self.stream)
        drop_cache([self.path])
        self.out = False
        self.last['get_s'] = round(time.time() - started, 1)
        return time.time() - started


class ReferenceWeights:
    """The KL reference (the start policy's BF16 parameters) as a safetensors file instead of a second model resident on
    the GPU (3.5 GB through every phase, grpo-v2-run2 2026-11-15). The reference pass loads it into the policy's own
    parameters (``load_into``: the same module, kernels and training graph as the policy), and ``restore`` puts the
    policy back from its FP32 masters, which is exact: the policy's BF16 parameters are always their masters rounded
    (init, resume and every optimizer step set them so). Page cache of each read is dropped (``drop_cache``)."""

    def __init__(self, path, params, metadata):
        """Write ``params`` ({name: BF16 tensor}, the start policy) to ``path``, unless the file there already holds them
        for the same ``metadata`` (a resume of the same run and start checkpoint)."""
        import torch
        from safetensors import safe_open
        self.path = Path(path)
        metadata = {k: str(v) for k, v in metadata.items()}
        layout = [(name, 'BF16', tuple(p.shape), p.numel() * 2) for name, p in params.items()]
        if self.path.exists():
            try:
                with safe_open(str(self.path), 'pt') as handle:
                    same = handle.metadata() == {**metadata, 'format': 'pt'} and sorted(handle.keys()) == sorted(params) \
                        and all(tuple(handle.get_slice(n).get_shape()) == tuple(p.shape) for n, p in params.items())
            except Exception:
                same = False
            if same:
                return
        if any(p.dtype != torch.bfloat16 for p in params.values()):
            raise ValueError('the reference parameters must be BF16')
        header, _, _ = safetensors_header(layout, {**metadata, 'format': 'pt'})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + '.partial')
        with partial.open('wb') as handle:
            handle.write(header)
            for p in params.values():
                handle.write(memoryview(p.detach().reshape(-1).view(torch.uint8).cpu().numpy()))
        os.replace(partial, self.path)
        drop_cache([self.path])

    def tensors(self):
        """(name, host BF16 tensor) of the reference, one at a time, each read into its own buffer.

        Not ``safe_open(...).get_tensor``: those tensors are views of one private (copy-on-write) mapping of the whole
        file, which the GPU copy turns into 3.4 GB of anonymous memory, and any one of them kept alive keeps all of it.
        The optimizer phase's loop variable did, so the learner held the whole reference between steps (grpo-v3-run1:
        learner RSS +3.4 GB from the first optimizer phase on, its ``reference-bf16.safetensors`` mapping fully anon in
        smaps, 2026-11-16). Now only the tensor in hand is resident."""
        import torch
        try:
            with open(self.path, 'rb', buffering=0) as handle:
                size = struct.unpack('<Q', handle.read(8))[0]
                header = json.loads(handle.read(size))
                base = 8 + size
                for name, info in header.items():
                    if name == '__metadata__':
                        continue
                    if info['dtype'] != 'BF16':
                        raise ValueError(f'reference tensor {name} is {info["dtype"]}, expected BF16')
                    start, end = info['data_offsets']
                    data = torch.empty(end - start, dtype=torch.uint8)
                    view = memoryview(data.numpy())
                    handle.seek(base + start)
                    read = 0
                    while read < len(view):
                        count = handle.readinto(view[read:])
                        if not count:
                            raise ValueError(f'{self.path} ends inside tensor {name}')
                        read += count
                    yield name, data.view(torch.bfloat16).view(info['shape'])
        finally:
            drop_cache([self.path])

    def load_into(self, params):
        import torch
        with torch.no_grad():
            for name, tensor in self.tensors():
                params[name].copy_(tensor)

    @staticmethod
    def restore(params, masters):
        import torch
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(masters[name])


def release_host_memory():
    """Hand freed host memory back to the system: glibc keeps freed heap (the step's lowered rollouts, decoded states)
    mapped until trimmed, and on unified memory that is memory the GPU allocations then lack."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
    # Unused page-locked blocks of PyTorch's caching host allocator: a resume's safetensors load to CUDA stages through
    # pinned buffers that stayed cached for the whole run (grpo-v3-run1-r1: host_pinned_gb 1.33 after the resume, 0.31
    # fresh). Blocks in use (StateSaver's staging) are kept.
    try:
        import torch
        torch._C._host_emptyCache()
    except Exception:
        pass


def host_pinned_gb():
    """Pinned host memory PyTorch's caching host allocator owns (GB, active and cached blocks)."""
    try:
        import torch
        stats = torch.cuda.host_memory_stats()
        return round(stats.get('allocated_bytes.current', 0) / 2**30, 2)   # active + cached blocks
    except Exception:
        return None


def process_memory_gb():
    """This process's resident host memory by kind (/proc/self/status), GB."""
    out = {}
    try:
        for line in open('/proc/self/status'):
            key, _, value = line.partition(':')
            if key in ('RssAnon', 'RssShmem', 'RssFile'):
                out[key[3:].lower()] = round(int(value.split()[0]) / 2**20, 2)
    except OSError:
        pass
    return out


# ---- learner ---------------------------------------------------------------------------------------------------------
def write_pruned_checkpoint(state, native_rows, special, path, source_sha256, trained):
    """``exp0_sft.write_checkpoint``'s pruned native checkpoint from a host snapshot, without holding the source backbone:
    the trained backbone (BF16), the source's native embedding rows at the interface ids, the source's interface."""
    import torch
    backbone = {name: trained[name].contiguous() for name in native_rows['names']}
    # ``trained`` may be the actor's served weights (``WeightSync.backbone`` views): the interface rows go into a copy
    embedding = backbone['model.embed_tokens.weight'] = backbone['model.embed_tokens.weight'].clone()
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
    [(packed row start, key)], their producer documents {key: document}) of a decision batch."""
    import torch
    ids, lengths, positions, targets, owner, counts, offset, option, imports = [], [], [], [], [], [], 0, 0, []
    sources = {}
    for item in batch:
        row = item['row']
        counts.append(len(row['options']))
        for continuation in row['options']:
            prompt = len(row['prompt'])
            imports += [(offset + start, block_id) for start, block_id in row.get('imports') or ()]
            sources.update(row.get('sources') or {})
            ids += [row['prompt'], continuation]
            lengths.append(prompt + len(continuation))
            positions.append(np.arange(offset + prompt - 1, offset + prompt + len(continuation) - 1))
            targets.append(continuation)
            owner.append(np.full(len(continuation), option))
            offset += prompt + len(continuation)
            option += 1
    as_long = lambda parts: torch.from_numpy(np.concatenate(parts).astype(np.int64)).to(device)
    return as_long(ids), lengths, as_long(positions), as_long(targets), as_long(owner), counts, imports, sources


def sd_add(buffer, episodes, step, args, stats):
    """Self-distillation (RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md section 4): keep the verified successes of a group
    whose students succeeded at most --sd-max-success of the time (rare successes; JOLT+ teacher successes in student
    form count too) for replay in the next --sd-window steps. Episodes whose calls import Neuralese state are left out
    (their blocks may be collected before the replay)."""
    students = [e for e in episodes if e.get('role', 'student') == 'student']
    if not students or sum(e['quality'] == 1.0 for e in students) / len(students) > args.sd_max_success:
        return
    for episode in episodes:
        if episode['quality'] != 1.0 or episode.get('role') == 'sd':
            continue
        if any(row.get('imports') for row in episode['rows']):
            stats['sd_skip_imports'] += 1
            continue
        buffer.append({'episode': {**episode, 'decisions': []}, 'version': step, 'uses': 0})
        stats['sd_added'] += 1


def sd_take(buffer, step, args, stats):
    """The episodes replayed at ``step``: those added before it within --sd-window steps, least replayed and newest
    first, at most --sd-episodes. Older ones leave the buffer."""
    buffer[:] = [item for item in buffer if step - item['version'] <= args.sd_window]
    stats['sd_buffer'] = len(buffer)
    chosen = sorted((item for item in buffer if item['version'] < step), key=lambda item: (item['uses'], -item['version']))
    chosen = chosen[:args.sd_episodes]
    for item in chosen:
        item['uses'] += 1
    return [item['episode'] for item in chosen]


def renderer_tokenizer(export_dir, out):
    """A tokenizer directory for ``sft_sources.TemplateRenderer`` from the actor's HF export (its tokenizer, template and
    model type), so teacher calls render with the exact ids the actor serves."""
    out.mkdir(parents=True, exist_ok=True)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'special_tokens_map.json'):
        if (Path(export_dir) / name).exists():
            shutil.copyfile(Path(export_dir) / name, out / name)
    config = json.loads((Path(export_dir) / 'config.json').read_text())
    (out / 'model_type.json').write_text(json.dumps({'model_type': config['model_type'], 'source': str(export_dir)}) + '\n')
    return out


class TeacherDemos:
    """Verified teacher-pool successes for the self-distillation replay (``--sd-teacher-weight``): every success of the
    sft_sources WORKs (``--sd-teacher-source``; first WORK wins a key) of source ``teacher`` whose replay under that WORK's
    runtime was exact (``sft_sources.verified_episodes``), on an eligible pool task (the run's task list, held closure
    applied) and outside the held closure again (``sft_sources.held_excluded_ids``), at most ``max_per_task`` per task
    (teacher models taking turns, as ``sft_sources.capped``). ``take`` lowers the chosen ones with sft_sources' own
    rendering (``call_row`` on the replayed turns: the student-form prompt the runtime builds, the teacher's reply as
    target, an empty think block at weight 0)."""

    def __init__(self, works, by_id, held_manifest, tokenizer, max_per_task, seed, log=print):
        import random as _random
        from sft_sources import TemplateRenderer, held_closure, held_excluded_ids, verified_episodes
        self.renderer = TemplateRenderer(tokenizer)
        self.funnel = collections.defaultdict(collections.Counter)
        self.rng = _random.Random(seed)
        found, tasks = {}, {}
        for work in works:
            if not (Path(work) / 'replay.jsonl').exists():
                log({'event': 'sd teacher source missing', 'work': str(work)})
                continue
            for episode, record in verified_episodes(work, self.funnel, sources={'teacher'}):
                if episode['task_id'] not in by_id or episode['key'] in found:
                    continue
                tasks[episode['task_id']] = episode['task']
                found[episode['key']] = {'key': episode['key'], 'work': str(work), 'name': record['name'],
                                         'task_id': episode['task_id'],
                                         'teacher': (episode.get('teacher') or {}).get('model_id') or ''}
        excluded = held_excluded_ids(held_manifest, tasks, held_closure(held_manifest))
        by_task = collections.defaultdict(list)
        for item in found.values():
            if item['task_id'] in excluded:
                self.funnel['teacher']['held_excluded'] += 1
            else:
                by_task[item['task_id']].append(item)
        self.by_task = {}
        for task_id, items in sorted(by_task.items()):
            lanes = collections.defaultdict(list)
            for item in sorted(items, key=lambda item: item['key']):
                lanes[item['teacher']].append(item)
            queues = [lanes[k] for k in sorted(lanes)]
            for lane in queues:
                self.rng.shuffle(lane)
            kept = []
            while any(queues) and len(kept) < max_per_task:
                for lane in queues:
                    if lane and len(kept) < max_per_task:
                        kept.append(lane.pop())
            self.funnel['teacher']['over_task_cap'] += len(items) - len(kept)
            for item in kept:
                item['uses'] = 0
            self.by_task[task_id] = kept
        self.episodes = sum(len(v) for v in self.by_task.values())
        log({'event': 'sd teacher demonstrations', 'tasks': len(self.by_task), 'episodes': self.episodes,
             'funnel': {k: dict(v) for k, v in self.funnel.items()}, 'renderer': self.renderer.identity})

    def take(self, eligible, count, stats):
        """Up to ``count`` lowered teacher episodes of the ``eligible`` task ids, one per task, least replayed first
        (ties at random); a render error drops that episode for the run."""
        from sft_sources import call_row, replayed_turns
        candidates = [item for task_id in eligible for item in self.by_task.get(task_id, ())]
        stats['sd_teacher_candidates'] = len(candidates)
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda item: item['uses'])
        out, tasks = [], set()
        for item in candidates:
            if len(out) >= count:
                break
            if item['task_id'] in tasks:
                continue
            try:
                rows = [call_row(self.renderer, 'teacher', item['key'], index, turn)[0]
                        for index, turn in enumerate(replayed_turns(item['work'], item['name']))]
            except Exception as error:
                stats['sd_teacher_render_error'] += 1
                print(json.dumps({'event': 'sd teacher render error', 'key': item['key'], 'error': repr(error)[:300]}),
                      flush=True)
                self.by_task[item['task_id']].remove(item)
                continue
            for row in rows:
                row['teacher'] = True            # no actor log-probabilities: a ratio-free term (train loop)
            item['uses'] += 1
            tasks.add(item['task_id'])
            out.append({'task_id': item['task_id'], 'round': item['key'], 'rows': rows, 'decisions': [],
                        'reward': 1., 'quality': 1., 'role': 'sd_teacher'})
        return out


def sd_teacher_tasks(collector, band):
    """Tasks whose current pass rate (the run's latest group, else the measured base rate) is in the pass band or 0
    (all-fail): where a verified demonstration can still raise the success rate."""
    rates = {k: (v['successes'], v['scored']) for k, v in collector.base_rates.items() if v.get('scored')}
    rates.update(collector.own)
    low, high = band
    return sorted(k for k, (s, n) in rates.items() if n and (s == 0 or low <= s / n <= high))


TEACHER_FIELDS = ('groups', 'episodes', 'successes', 'episodes_trained', 'calls_student_form',
                  'privileged_tokens_removed', 'policy_calls_removed')
SD_FIELDS = ('added', 'episodes', 'tokens', 'buffer', 'skip_imports', 'own_episodes', 'own_tokens', 'teacher_episodes',
             'teacher_tokens', 'teacher_candidates', 'teacher_render_error')


def restore_telemetry(stats, args):
    """Top-level per-step fields of the restored methods for train.jsonl (the heartbeat reads these; ``lowering``
    keeps the raw counters): ``teacher`` (JOLT+ attempts lowered this step: groups, episodes, successes, episodes
    trained with a positive advantage, calls in student form, privileged tokens and policy calls removed) and ``sd``
    (self-distillation replay: added to the buffer, replayed episodes and tokens, buffer size). Every key is present
    (0 when nothing happened) while the method is on; the block is absent when it is off. The teacher tokens' rho
    quantiles are ``teacher_rho``, logged after the learner."""
    out = {}
    if getattr(args, 'teacher_attempts', 0):
        out['teacher'] = {key: int(stats.get('teacher_' + key, 0)) for key in TEACHER_FIELDS}
    if getattr(args, 'sd_weight', 0) or getattr(args, 'sd_teacher_weight', 0):
        out['sd'] = {key: int(stats.get('sd_' + key, 0)) for key in SD_FIELDS}
    return out


def teacher_rho_summary(parts, cap):
    """Distribution of rho over the step's teacher tokens in student form (JOLT+): how much of the guided run the unguided
    policy finds plausible. None without teacher tokens."""
    if not parts:
        return None
    log_rho = np.concatenate(parts)
    q = lambda p: round(float(np.quantile(log_rho, p)), 4)
    return {'tokens': int(log_rho.size), 'log_rho_p01': q(.01), 'log_rho_p10': q(.1), 'log_rho_p50': q(.5),
            'log_rho_p90': q(.9), 'log_rho_p99': q(.99), 'rho_mean': round(float(np.exp(np.minimum(log_rho, 20)).mean()), 4),
            'rho_below_0.1': round(float((log_rho < math.log(.1)).mean()), 4),
            'rho_capped': round(float((log_rho > math.log(cap)).mean()), 4)}


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def all_equal_probability(a, b, m, success):
    """P(the next ``m`` samples all succeed (``success``) or all fail) for a pass rate ~ Beta(a, b): E[p^m] or
    E[(1-p)^m], the beta-binomial's end mass."""
    if m <= 0:
        return 1.
    return math.exp(_log_beta(a + m, b) - _log_beta(a, b) if success else _log_beta(a, b + m) - _log_beta(a, b))


def spread_after(a, b, n, k, success):
    """P(a group of ``n`` has reward spread | its first ``k`` samples all succeeded (or all failed)) for a pass rate
    ~ Beta(a, b) before them: 1 - P(the other n - k agree | the first k agree)."""
    first = all_equal_probability(a, b, k, success)
    return 1. - all_equal_probability(a, b, n, success) / first if first > 0 else 1.


def wave_plan(a, b, n, max_spread, min_saving=1.):
    """Dynamic sampling (DAPO-style, decided before the samples run): the first-wave size k of a group of ``n`` for a
    task whose pass rate is ~ Beta(a, b), or None to run all n at once. After a first wave of k that all agree, the other
    n - k are cancelled when the posterior chance they would still break the tie (``spread_after``) is at most
    ``max_spread``; otherwise they run. k maximizes the expected samples saved, sum over s of P(first k all s) *
    [spread_after <= max_spread] * (n - k); a split costs one wave of latency, so it is planned only when that
    expectation is at least ``min_saving`` samples. Returns (k, expected saving, expected spread groups lost)."""
    best = (None, 0., 0.)
    for k in range(2, n - 1):
        saved = lost = 0.
        for success in (True, False):
            first = all_equal_probability(a, b, k, success)
            spread = spread_after(a, b, n, k, success)
            if spread <= max_spread:
                saved += first * (n - k)
                lost += first * spread
        if saved > best[1]:
            best = (k, saved, lost)
    return best if best[1] >= min_saving else (None, 0., 0.)


class PassRatePrior:
    """Empirical-Bayes pass-rate prior per task (dynamic sampling): the task's own measurement (the run's latest scored
    group, else the measured base rate) on top of a Beta(c m, c (1 - m)) whose mean m is the task's family's mean task
    rate (shrunk to the pool's with 2 pseudo-tasks; the task itself left out) and whose concentration c maximizes the
    beta-binomial likelihood of the measured tasks' outcomes (each against its own leave-one-out family mean; a grid over
    0.1-50). A pool whose tasks mostly pass or fail every sample fits a small c: a few agreeing samples are strong
    evidence; tasks of mixed outcome fit a large one."""

    GRID = tuple(round(0.1 * 1.2 ** i, 4) for i in range(35))   # 0.1 .. ~49

    def __init__(self, rates, family_of, min_scored=2):
        rates = {k: (s, n) for k, (s, n) in rates.items() if n >= min_scored}
        self.rates, self.family_of = rates, family_of
        values = [s / n for s, n in rates.values()]
        self.mean = (sum(values) + 1.) / (len(values) + 2.)
        self.families = {}
        for task_id, (s, n) in rates.items():
            family = self.families.setdefault(family_of(task_id), [0., 0])
            family[0] += s / n
            family[1] += 1
        self.concentration = 2.
        if len(rates) >= 20:
            means = [(self.family_mean(task_id), s, n) for task_id, (s, n) in rates.items()]

            def loglik(c):
                total = 0.
                for m, s, n in means:
                    a, b = max(c * m, 1e-3), max(c * (1 - m), 1e-3)
                    total += _log_beta(a + s, b + n - s) - _log_beta(a, b)   # the binomial coefficient is constant in c
                return total
            self.concentration = max(self.GRID, key=loglik)

    def family_mean(self, task_id):
        total, count = self.families.get(self.family_of(task_id), (0., 0))
        own = self.rates.get(task_id)
        if own is not None:
            total, count = total - own[0] / own[1], count - 1
        return min(max((total + 2. * self.mean) / (count + 2.), 1e-3), 1 - 1e-3)

    def beta(self, task_id):
        """(a, b) of the task's pass rate."""
        m = self.family_mean(task_id)
        a, b = self.concentration * m, self.concentration * (1. - m)
        own = self.rates.get(task_id)
        if own is not None:
            a, b = a + own[0], b + own[1] - own[0]
        return a, b


def group_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std < 1e-6:
        return None
    return [(r - mean) / (std + 1e-6) for r in rewards]


def decision_unit(tokens, calls, mode, cap=8.):
    """Units one sampled decision counts in its episode's normalization. ``call``: the episode's mean sampled tokens per
    trained call (a decision is one model call whose whole reply is one categorical action; 1 when the episode trains
    no token call), at most ``cap``: a choice among a few options carries a few tokens' worth of information, and at a
    whole call's weight (~1,000 units) one step reversed templated inbox decisions by ~18 nats (replay controls,
    DECISIONS 2026-11-14); ``token``: 1, one sampled token's weight. ``cap`` 0: uncapped."""
    if mode == 'token' or not calls:
        return 1.
    unit = max(1., tokens / calls)
    return min(unit, cap) if cap else unit


def loss_scales(episodes, norm, decision_weight, decision_cap=8.):
    """Per-episode loss scale factors: ``episodes`` {key: (trained tokens, trained calls, trained decisions)} ->
    {key: (factor of each token term, factor of each decision term)}. An episode's units are its tokens plus its
    decisions times ``decision_unit``. ``episode``: every episode with units weighs 1 / episodes and its terms share
    that weight by units (A_i times the mean over its units, averaged over episodes; a chain's factor is
    1 / (episode units * episodes), so accumulating chain by chain is exact). ``token``: one factor 1 / all units."""
    units = {key: (tokens + decisions * decision_unit(tokens, calls, decision_weight, decision_cap),
                   decision_unit(tokens, calls, decision_weight, decision_cap))
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


def kl_term(d, estimator):
    """The KL penalty per sampled unit for ``d = logp_ref - logp_policy`` (the actor sampled from the policy). ``k2`` =
    ``d^2 / 2``: its gradient ``-d`` per token is the on-policy reverse-KL gradient and grows linearly. ``k3`` =
    ``e^d - d - 1`` is an unbiased KL estimate, but as a loss its gradient ``1 - e^d`` is unbounded: grpo-v2-run1 step 7
    had a few T=1 tail tokens with ``d`` about 10-12 carry ~90% of the clipped update (grad norm 6.1 vs ~0.6, k3 mean
    0.33 while mean |d| was 0.14; DECISIONS 2026-11-14). k3 stays the logged KL estimate either way."""
    return 0.5 * d * d if estimator == 'k2' else d.exp() - d - 1


def kl_stats(sums, d):
    """Telemetry of one batch of ``d``: the k3 KL estimate, |d|, its max and the units past 5 nats."""
    d = d.detach()
    sums['kl'] += (d.exp() - d - 1).sum().item()
    sums['ref_abs_dlogp'] += d.abs().sum().item()
    sums['ref_dlogp_over_5'] += int((d.abs() > 5).sum())
    sums['ref_dlogp_max'] = max(sums['ref_dlogp_max'], d.abs().max().item() if d.numel() else 0.)


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


def behaviour_logp(chain, logp):
    """(log-probabilities of the law that produced the chain's tokens, per-token row weights or None) for the policy
    term. An actor sample: the actor's recorded logp. A teacher demonstration (role ``sd_teacher``) has none: the
    learner's own (detached) logp stands in, so rho = 1 (a ratio-free cross-entropy at its advantage) and the tail floor
    reads the learner's probability; its rows' weights (an empty think block at 0) multiply the term."""
    import torch
    if chain.get('role') != 'sd_teacher':
        return torch.from_numpy(np.concatenate([row['logp'] for _, _, row in chain['segs']])).to(logp.device).float(), None
    import exp0_sft
    weights = (exp0_sft.chain_weights(chain, logp.device) if any('weight' in row for _, _, row in chain['segs'])
               else None)
    return logp.detach().float(), weights


def reasoning_style(chains, close_id):
    """Reasoning and reply tokens per trained call this step: a behaviour sentinel from masks the learner computes anyway.
    grpo-v2-run1's reasoning halved from step 4 while the reward did not select it, three hours before a held eval
    showed -8 pts (DECISIONS 2026-11-14)."""
    calls = think = 0
    for chain in chains:
        for _, _, row in chain['segs']:
            calls += 1
            think += int(think_mask(row['sample'], close_id).sum())
    tokens = sum(length for c in chains for _, length, _ in c['segs'])
    return {'reasoning_tokens_per_call': round(think / max(1, calls), 1),
            'reply_tokens_per_call': round((tokens - think) / max(1, calls), 1)}


def style_shift(work, style, ratio=0.75):
    """The ratio of this step's reasoning tokens per call to the run's first logged step, and whether it is outside
    [ratio, 1/ratio]. A logged flag, never a stop."""
    first = None
    path = work / 'train.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            if '"reasoning_tokens_per_call"' in line:
                first = json.loads(line).get('reasoning_tokens_per_call')
                break
    if not first:
        return {}
    r = style['reasoning_tokens_per_call'] / first
    return {'reasoning_vs_first_step': round(r, 3), 'reasoning_shift': not ratio <= r <= 1 / ratio}


def subsample_calls(episode, fraction, rng, stats=None):
    """``--call-fraction``: the episode with each call's row kept for the policy and KL terms with probability ``fraction``
    (at least one). A later call's prompt holds the earlier calls' text, so a dropped call stays context of the calls
    after it (``chains_of`` chains across it) and a chain ends at its last kept call: the step forwards fewer tokens.
    Under ``--loss-norm token`` the step's mean over kept units estimates the mean over all units."""
    rows = episode['rows']
    if not rows:
        return episode
    keep = [row for row in rows if rng.random() < fraction] or [rng.choice(rows)]
    if stats is not None:
        stats['calls_subsampled_out'] += len(rows) - len(keep)
    return {**episode, 'rows': keep}


def lr_data_scale(groups_with_spread, reference):
    """The step's LR factor: groups with reward spread over ``reference``, at most 1. AdamW's step size does not shrink with
    the data behind a gradient, so a step on few informative groups moves the policy as far as a full one along a
    gradient that mostly reflects those few episodes: grpo-v2-run1's step 3 (2 groups with spread, a stop) took a full
    step, and the KL to the start policy jumped after it (DECISIONS 2026-11-14). ``reference`` 0: always 1."""
    if not reference:
        return 1.
    return min(1., groups_with_spread / reference)


def grad_spike(work, grad_norm, factor=5., min_steps=1):
    """This step's pre-clip grad_norm over the median of the run's earlier logged steps, and whether it exceeds factor
    (grpo-v2-run1: 6.09 at step 7 against ~0.5 was the k3 KL loss blowing up). From the second step on: the replay
    controls' step-3 spikes (20-60x) went unflagged at min_steps 3. A logged flag, never a stop."""
    path = work / 'train.jsonl'
    norms = [json.loads(line)['grad_norm'] for line in path.read_text().splitlines()
             if '"grad_norm"' in line] if path.exists() else []
    norms = [n for n in norms if isinstance(n, (int, float))]
    if len(norms) < min_steps:
        return {}
    r = grad_norm / statistics.median(norms)
    return {'grad_norm_vs_median': round(r, 2), 'grad_spike': r > factor}


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


def task_features(tasks):
    """id -> (held constructs the task's program declares, whether it is a one-function dataset wrapper):
    ``sft_sources.task_constructs`` (``measure_repetition.program_features``), the one implementation."""
    sys.path.insert(0, str(REPO / 'scripts/rl_tasks'))
    from sft_sources import task_constructs
    return {row['id']: task_constructs(row) for row in tasks}


class ExploreWalk:
    """The walk over unmeasured pool tasks (the ``--explore`` share of groups), in the pool's order (``--explore-order``),
    weighted toward the constructs the held set tests (held-v3: 30% Neuralese results, 30% folder reducers, 18% iterateOn,
    18% call depth >= 3, 11% decide, 9% one-function dataset wrappers). A row declaring a held construct
    (``sft_sources.HELD_CONSTRUCTS`` except ``multi_function``, which the pool has as often as the held set: 69% vs 75%)
    is always taken when the walk reaches it; any other row is taken with probability 1 / ``held_weight``, and a dataset
    wrapper only while wrappers stay at most ``wrapper_share`` of the rows taken. A row passed over stays unmeasured and
    comes round again. ``held_weight`` 1 and ``wrapper_share`` 1: the plain walk."""

    COMMON = frozenset({'multi_function'})

    def __init__(self, tasks, wrapper_share=1., held_weight=1.):
        self.tasks, self.wrapper_share, self.held_weight = tasks, wrapper_share, held_weight
        self.weighted = wrapper_share < 1 or held_weight != 1
        self.features = task_features(tasks) if self.weighted and tasks else {}
        self.counts = collections.Counter()      # rows taken: all, wrappers, declaring a held construct

    def pool_report(self):
        report = {'wrapper_share': self.wrapper_share, 'held_weight': self.held_weight, 'rows': len(self.tasks)}
        if self.features:
            constructs = collections.Counter(k for held, _ in self.features.values() for k in held)
            report.update(wrappers=sum(w for _, w in self.features.values()),
                          declaring_held=sum(bool(set(held) - self.COMMON) for held, _ in self.features.values()),
                          constructs=dict(sorted(constructs.items())))
        return report

    def next(self, cursor, measured, rng):
        """(the next row to explore, the new cursor). After two passes over the pool (everything measured) any row."""
        n, scanned = len(self.tasks), 0
        while True:
            row = self.tasks[cursor % n]
            cursor += 1
            exhausted = cursor > 2 * n
            if row['id'] in measured and not exhausted:
                continue
            scanned += 1
            held, wrapper = self.features.get(row['id'], ((), False))
            held = set(held) - self.COMMON
            take = exhausted or scanned > n or not self.weighted
            if not take:
                if wrapper and self.counts['wrappers'] + 1 > self.wrapper_share * (self.counts['taken'] + 1):
                    continue
                take = bool(held) or rng.random() < 1. / self.held_weight
            if take:
                self.counts['taken'] += 1
                self.counts['wrappers'] += wrapper
                self.counts['declaring_held'] += bool(held)
                return row, cursor


def family_round_robin(rows, rng):
    """``rows`` (already shuffled) reordered so that the exploration walk visits families in turn: each family keeps its
    shuffled order, the families' order is drawn from ``rng``, and the walk takes one row of each remaining family per
    round. Exploration then spends the same share on every family instead of a share proportional to its row count
    (TASK_SUPPLY_20261117: a 25-row family in the 33k-row pool is otherwise explored about once per run)."""
    by_family = {}
    for row in rows:
        by_family.setdefault(str(row.get('family') or ''), []).append(row)
    families = sorted(by_family)
    rng.shuffle(families)
    queues = [by_family[f] for f in families]
    out, index = [], 0
    while len(out) < len(rows):
        for lane in queues:
            if index < len(lane):
                out.append(lane[index])
        index += 1
    return out


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
    if getattr(args, 'explore_order', 'row') == 'family':
        eligible = family_round_robin(eligible, random.Random(args.seed))
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
                                'explore': args.explore, 'explore_order': getattr(args, 'explore_order', 'row'),
                                'own_outcomes': 'latest scored group per task replaces its rate'},
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
    import behaviour_probe
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
    if args.replay:
        # learner controls on a finished run's groups: no actor (its counters read None), no weight sync, no held eval,
        # and the old run's actor store is never pruned
        if not (args.actor_store and args.actor_export):
            raise SystemExit('--replay needs --actor-store and --actor-export (the replayed run\'s WORK/actor, WORK/actor/sglang-hf)')
        args.actor, args.no_actor_sync, args.keep_rollouts = 'http://127.0.0.1:9', True, True
        args.held_every_hours, args.held_remote, args.held_extra = 0, None, None
    elif not args.actor:
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
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    lfm3_fast_train.prepare(model)
    forward = lfm3_fast_train.hidden
    pack_tokens = args.pack_tokens if args.pack_tokens > 0 else exp0_sft.resident_tokens(model, args.activation_gb)
    live = dict(model.named_parameters())
    reference = None
    if args.kl_beta > 0:          # the start policy, BF16, frozen: a file, loaded into the policy for the reference pass
        # The reference runs exactly the policy's training graph (train mode, parameters requiring grad, the same partial
        # checkpointing; its outputs are detached and never backpropagated): a no_grad pass compiles other kernels, whose
        # BF16 rounding alone put KL at 5e-4 (|dlogp| ~0.03) with identical weights (smoke 2026-11-12). It is the policy
        # module itself with the reference weights loaded (ReferenceWeights), written now, before a resume moves them.
        reference = ReferenceWeights(Path(args.state_dir) / 'reference-bf16.safetensors', live,
                                     {'source_sha256': source_sha, 'base': args.base})
    parameters = [masters[name] for name in live]

    # FP32 master gradients exist only while the policy pass and the optimizer need them (6.5 GB): collection and the
    # reference pass run without them. Zeros, then the same accumulation as ever: the step's arithmetic is unchanged.
    def allocate_grads():
        for master in parameters:
            master.grad = torch.zeros_like(master)

    def release_grads():
        for master in parameters:
            master.grad = None
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
    branch_root = Path(args.checkpoint_dir or Path(args.state_dir) / 'checkpoints')
    resume_path = Path(args.resume_state) if args.resume_state else state_path
    if args.resume and not args.resume_state and not resume_path.exists() and args.branch_states:
        resume_path = max(branch_root.glob('state-step-*.safetensors'), default=resume_path)
    if args.resume and resume_path.exists():
        from safetensors import safe_open
        with safe_open(str(resume_path), 'pt', device='cuda') as handle:
            meta = handle.metadata()
            # materialise AdamW state (zero grads) before the masters are restored: with weight decay this step moves
            # the parameters, and it must not move the restored ones
            allocate_grads()
            optimizer.step()
            release_grads()
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
        torch.cuda.synchronize()
        release_host_memory()             # the load's pinned staging buffers
        log_line(log, {'event': 'resumed', 'step': step, 'cursor': cursor, 'next_group': first_group,
                       'state': str(resume_path)})
    # New groups are numbered after the highest existing one; scored groups a stop left untrained are offered again
    # (Collector.reuse_scored).
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
    # atexit never runs for a killed unit: a dead run's tmpfs weights (4 GB each) held RAM until removed by hand
    # (2026-11-16: two hunt copies, 8 GB). Each run records its pid; a new run removes copies whose run is gone.
    for stale in Path('/dev/shm').glob('natlang-*-actor-weights'):
        try:
            os.kill(int((stale / 'owner.pid').read_text()), 0)
        except ProcessLookupError:   # recorded owner is dead (a copy without owner.pid is never touched)
            if stale != weights_dir:
                shutil.rmtree(stale, True)
        except (OSError, ValueError):
            pass
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / 'owner.pid').write_text(str(os.getpid()))
    if not args.frozen_producer_table and producer_adapter_keys(state['interface_state']):
        raise SystemExit(f'the checkpoint has a producer adapter {producer_adapter_keys(state["interface_state"])}: '
                         'the producer table is not the embedding (use --frozen-producer-table)')
    sync = WeightSync(model, args.actor_export, None if args.no_actor_sync else sglang_url, weights_dir,
                      front_url=None if args.no_actor_sync else args.actor,
                      producer_from_embedding=not args.frozen_producer_table)
    atexit.register(shutil.rmtree, weights_dir, True)
    drop_cache([Path(args.actor_export) / 'model.safetensors'])
    saver = StateSaver()
    # AdamW moments on disk outside the optimizer step (MomentOffload; --keep-moments-resident opts out)
    offload = None if args.keep_moments_resident else MomentOffload(work / 'adam-moments.offload')
    if args.replay:
        collector = ReplayCollector(args, work, first_group=first_group)
    else:
        collector = Collector(args, tasks, work, source_sha, runtime, first_group=first_group, cursor=cursor)
        collector.variants = load_variants(args, tasks, work)
    teacher_demos = None
    if args.sd_teacher_weight > 0 and args.sd_teacher_episodes > 0 and not args.replay:
        teacher_demos = TeacherDemos(args.sd_teacher_source or SD_TEACHER_SOURCES, collector.by_id, args.held_manifest,
                                     renderer_tokenizer(args.actor_export, work / 'sd-teacher-tokenizer'),
                                     args.sd_teacher_max_per_task, args.seed, log=lambda r: log_line(log, r))
    collector.version = step
    if args.resume and not args.replay:      # a replay offers archived groups, never this run's own
        reused, left_out = collector.reuse_scored(step, args.max_staleness, args.memetic_max_staleness)
        if reused or left_out:
            log_line(log, {'event': 'reused groups', 'reused': len(reused), 'left_out': left_out[:50],
                           'left_out_count': len(left_out)})
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

    behaviour = None
    if args.behaviour_probe != '-':
        if Path(args.behaviour_probe).exists():
            behaviour = json.loads(Path(args.behaviour_probe).read_text())
        else:
            log_line(log, {'event': 'no behaviour probe', 'path': args.behaviour_probe})
    behaviour_start_path = work / 'behaviour-start.json'

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
    strict_series = ['--strict-series'] if args.held_strict_series else []   # held_eval compare
    step0 = work / 'held' / 'step-00000'
    if (step0 / 'result.json').exists() and not args.held_baseline:
        # A resume under another sampling law (--top-p) cannot compare with the old step 0: that result moves aside (it
        # stays the record of its law) and the start policy is measured again below. Another held series (a pin whose
        # harness differs) pairs with a warning (held_eval compare); --held-strict-series re-measures it as well.
        measured = json.loads((step0 / 'result.json').read_text())
        old = (measured.get('series') or {}).get('key')
        law = measured.get('sampling') or LEGACY_SAMPLING
        if (args.held_strict_series and old != run['series']['key']) or law != sampling_law_of(args.top_p):
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
    if behaviour is not None and not behaviour_start_path.exists():
        if step == 0:
            t = time.time()
            behaviour_start_path.write_text(json.dumps(behaviour_probe.measure(model, behaviour)) + '\n')
            log_line(log, {'event': 'behaviour probe start', 'calls': len(behaviour['calls']),
                           **behaviour_probe.summary(json.loads(behaviour_start_path.read_text())),
                           'seconds': round(time.time() - t, 1)})
        else:
            log_line(log, {'event': 'behaviour probe: no start measurement (resumed); copy the start run\'s '
                                    'behaviour-start.json to compare'})
    log_line(log, {'event': 'learner ready', 'init_s': round(time.time() - t0, 1), 'meminfo_gb': meminfo(),
                   'cuda_allocated_gb': round(torch.cuda.memory_allocated() / 2**30, 1),
                   'cuda_reserved_gb': round(torch.cuda.memory_reserved() / 2**30, 1)})
    if args.held_every_hours > 0 and not (work / 'held' / 'step-00000' / 'result.json').exists() \
            and not args.skip_base_held and not args.held_baseline and not args.held_remote:
        collector.paused.set()            # the start policy's held measurement goes first (held() resumes collection)
    slots = [threading.Thread(target=collector.slot, args=(index,), daemon=True)
             for index in range(0 if args.replay else args.slots)]
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
            number = int(old.name.split('-')[1])
            if number not in held_steps and number > args.keep_checkpoints_until and old.exists():
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
                                    str(out / 'compare-vs-step0.json'), *strict_series], cwd=REPO, env=pinned_env(), stdout=handle, stderr=subprocess.STDOUT)
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
        base_out = Path(args.held_base) if args.held_base else held_root / 'step-00000-pop'
        python = [sys.executable, '-m', 'natlang_neuralese.rl.held_eval']
        # Pop's engine settings: the KV dtype is part of the attested engine (export format), so the start policy's
        # result is kept only if it was measured with the same one and the same attested max running sequences.
        pop_engine = ['--kv-cache-dtype', args.pop_kv_cache_dtype, '--max-running', str(args.pop_max_running)]
        pop_format = pop_sglang(args.pop_kv_cache_dtype, args.pop_max_running)[1]

        def base_engine_matches():
            """False when held/step-00000-pop (finished or partial) was measured on another Pop engine: it moves to
            step-00000-pop-<time>-other-engine and the start policy is measured again (its steps would not pair with it).
            Another engine is another export format (weight and KV precision) or sampling law (--top-p); with
            --held-strict-series also another running-sequence cap (attested max_running). A fork commit or series
            difference pairs with a warning (held_eval compare)."""
            for name in ('result.json', 'run.json'):
                if (base_out / name).exists():
                    record = json.loads((base_out / name).read_text())
                    measured = (record.get('engine') or {}).get('export_format')
                    cap = (record.get('engine') or {}).get('max_running', args.pop_max_running)
                    return measured in (None, pop_format) and \
                        (cap == args.pop_max_running or not args.held_strict_series) and \
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
                # the start policy's per-task wall times order this measurement's first sample (held_eval.Dispatch)
                prior = (['--duration-prior', str(base_out / 'result.json')] if step_number and held_complete(base_out)
                         else [])
                code = subprocess.run([*python, 'measure', '--manifest', args.held_manifest, *prior, '--endpoint',
                                       f'{endpoint}={args.held_remote_slots}', '--policy-id', policy, '--top-p', str(args.top_p), '--samples',
                                       str(args.held_samples), '--out', str(out), '--label', f'grpo-pop-step-{step_number}',
                                       '--allow-other-engine', '--max-model-requests', str(args.max_model_requests)],
                                      cwd=REPO, env=pinned_env(), stdout=handle,
                                      stderr=subprocess.STDOUT).returncode
                if step_number and code == 0 and held_complete(base_out):
                    subprocess.run([*python, 'compare', str(base_out / 'result.json'), str(out / 'result.json'), '--out',
                                    str(out / 'compare-vs-step0.json'), *strict_series], cwd=REPO, env=pinned_env(), stdout=handle,
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
                    and held_complete(d)}

        last_start, last_step = 0., max(measured_steps() - {0}, default=0)
        while not stop.is_set():
            try:
                if args.held_base and not held_complete(base_out):
                    # measured by its own job (e.g. the pin's base on Pop while this run starts): wait, never serve
                    # another policy on Pop meanwhile; checkpoints keep being kept for their evaluations
                    stop.wait(300)
                    continue
                if args.held_base and not base_engine_matches():
                    raise RuntimeError(f'--held-base {base_out} was measured on another engine or sampling law than {pop_format}')
                if not base_engine_matches():
                    aside = held_root / f'step-00000-pop-{time.strftime("%Y%m%d%H%M")}-other-engine'
                    base_out.rename(aside)
                    log_line(log, {'event': 'held eval', 'remote': 'pop', 'moved_start_policy_result': str(aside),
                                   'reason': f'measured on another Pop engine than {pop_format}'})
                if not held_complete(base_out):
                    evaluate(0, base_out, source_sha, ['start', '--checkpoint', args.checkpoint, '--policy-id', source_sha,
                                                       *pop_engine])
                    if not held_complete(base_out):
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
    sd_buffer = []   # self-distillation replay of recent rare successes (``sd_add``/``sd_take``; --sd-weight)
    memetic_history = list(memetic_gap_history(work / 'train.jsonl'))   # [sum, count] of per-group guided gaps
    while not stop.is_set() and (not args.max_steps or step < args.max_steps):
        phase = collections.OrderedDict()
        t = time.time()
        groups = collector.take_groups(args.min_groups, args.max_groups, stop)
        if stop.is_set() and len(groups) < args.min_groups:
            # a stop never trains a short step: these scored groups stay on disk and a resume offers them again
            break
        phase['wait_groups'] = time.time() - t
        t = time.time()
        stats = collections.Counter()
        chains, decisions, rewards, spread, dropped_stale, staleness, task_rewards = [], [], [], 0, 0, [], []
        episode_units = {}   # trained episode -> [sampled tokens, calls, decisions] (loss_scales)
        episode_advantage = {}
        memetic_groups = []
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
            rewards += [e['reward'] for e in episodes if e.get('role') != 'teacher']   # the students' own reward
            memetic_record = memetic_group_record(group, task_id, episodes)
            if memetic_record is not None:
                memetic_groups.append({**memetic_record, 'stale': stale})
            # a memetic episode runs its program on many worlds (tens of minutes): its own staleness bound
            if stale > (args.memetic_max_staleness if memetic_record is not None else args.max_staleness):
                dropped_stale += 1
                continue
            staleness.append(stale)
            if args.sd_weight > 0:
                sd_add(sd_buffer, episodes, step, args, stats)
            advantages = episode_advantages(episodes, stats)
            if advantages is None:
                continue
            spread += 1
            for index, (episode, advantage) in enumerate(zip(episodes, advantages)):
                if advantage is None:
                    continue
                if args.call_fraction < 1:
                    episode = subsample_calls(episode, args.call_fraction,
                                              random.Random(f'{args.seed}:{step}:{group}:{index}'), stats)
                kept = exp0_sft.chains_of([episode], max_len=args.max_sequence_tokens)
                stats['calls_too_long_dropped'] += len(episode['rows']) - sum(len(c['segs']) for c in kept)
                key = len(episode_units)
                episode_advantage[key] = advantage
                episode_units[key] = [sum(n for c in kept for _, n, _ in c['segs']), sum(len(c['segs']) for c in kept), 0]
                for chain in kept:
                    chain['advantage'] = advantage
                    chain['episode'] = key
                    chain['role'] = episode.get('role', 'student')
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
        replays = []
        if args.sd_weight > 0:
            # self-distillation: verified rare successes of the last --sd-window steps again, as a positive-advantage
            # (truncated-IS cross-entropy) term of weight --sd-weight on their own exact actor calls
            replays += [(episode, args.sd_weight, 'sd', 'own') for episode in sd_take(sd_buffer, step, args, stats)]
        if teacher_demos is not None:
            # verified teacher-pool successes of tasks in the pass band or all-fail, in student form; no actor
            # log-probabilities, so a ratio-free cross-entropy of weight --sd-teacher-weight (rho = 1: the learner's
            # own log-probability stands in for the actor's)
            replays += [(episode, args.sd_teacher_weight, 'sd_teacher', 'teacher') for episode in teacher_demos.take(
                sd_teacher_tasks(collector, args.pass_band), args.sd_teacher_episodes, stats)]
        for episode, weight, role, source in replays:
            kept = exp0_sft.chains_of([episode], max_len=args.max_sequence_tokens)
            if not kept:
                continue
            key = len(episode_units)
            episode_advantage[key] = weight
            episode_units[key] = [sum(n for c in kept for _, n, _ in c['segs']), sum(len(c['segs']) for c in kept), 0]
            stats['sd_episodes'] += 1
            stats['sd_tokens'] += episode_units[key][0]
            stats[f'sd_{source}_episodes'] += 1
            stats[f'sd_{source}_tokens'] += episode_units[key][0]
            for chain in kept:
                chain.update(advantage=weight, episode=key, role=role)
                chains.append(chain)
        if args.advantage_control != 'none':
            control_advantages([*chains, *decisions], episode_advantage, args.advantage_control,
                               random.Random(args.seed + step))
        # Loss normalization (loss_scales): by default every trained episode weighs the same and a sampled decision (one
        # categorical action) counts as one call of its episode. `units` only normalizes the logged token means.
        loss_tokens = sum(length for c in chains for _, length, _ in c['segs'])
        units = loss_tokens + len(decisions)
        scales = loss_scales(episode_units, args.loss_norm, args.decision_weight, args.decision_unit_cap)
        episodes_trained = sum(1 for token_scale, decision_scale in scales.values() if token_scale or decision_scale)
        phase['lower'] = time.time() - t
        # The decoded imported states of the step's calls (lowering read them for their token ids) are not held through
        # the learner phase (up to the step's whole set of blocks, GBs of host memory): the learner reads each chain's
        # blocks again when it computes that chain (``chain_states``).
        imported_state_mb = sum(t.numel() * t.element_size() for block in collector.states.cache.values()
                                for layer in block.layers for t in layer.values() if hasattr(t, 'numel')) / 2**20
        collector.states.cache.clear()
        record = {'step': step + 1, 'groups': len(groups), 'groups_with_spread': spread,
                  'advantage_control': args.advantage_control if args.advantage_control != 'none' else None,
                  'replay_groups': [min(groups), max(groups)] if args.replay and groups else None,
                  'spread_fraction': round(spread / max(1, len(groups)), 3), 'dropped_stale_groups': dropped_stale,
                  'staleness': dict(collections.Counter(staleness)), 'episodes': len(rewards),
                  'reward_mean': round(sum(rewards) / max(1, len(rewards)), 4), 'sequences': len(chains),
                  'loss_tokens': loss_tokens, 'decisions': len(decisions), 'lowering': dict(stats),
                  'loss_norm': args.loss_norm, 'decision_weight': args.decision_weight, 'episodes_trained': episodes_trained,
                  'rollouts_archive_free_gb': round(shutil.disk_usage(rollouts_archive).free / 2**30, 1)
                  if rollouts_archive is not None and rollouts_archive.exists() else None,
                  'actor_version': versions_live['actor'], 'imported_state_mb': round(imported_state_mb, 1),
                  'neuralese_passthrough': not args.no_neuralese_passthrough,
                  'producer_document_tokens': sum(exp0_sft.source_tokens(c) for c in chains),
                  **({'memetic_groups': memetic_groups, 'memetic': memetic_step_record(memetic_groups, memetic_history)}
                     if memetic_groups or args.memetic_share > 0 else {}),
                  'dynamic_sampling': dict(getattr(collector, 'dynamic', {})) or None,   # cumulative since this process started
                  **restore_telemetry(stats, args)}
        style = reasoning_style(chains, think_close)
        record.update(style, **({} if args.replay else style_shift(work, style)))   # replayed reasoning is not the policy's
        if record.get('reasoning_shift'):
            print(f"reasoning shift: {style['reasoning_tokens_per_call']} reasoning tokens per call, "
                  f"{record['reasoning_vs_first_step']}x the run's first step", flush=True)
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
        cuda_peaks = {}                   # per learner phase: allocated / reserved peak (GB) and this process's host memory

        def phase_peak(name):
            cuda_peaks[name] = {'allocated': round(torch.cuda.max_memory_allocated() / 2**30, 1),
                                'reserved': round(torch.cuda.max_memory_reserved() / 2**30, 1), **process_memory_gb()}
            torch.cuda.reset_peak_memory_stats()
        collector.active_slots = args.learner_phase_slots
        # episodes still running when the learner starts (they finish while it computes: their collectors and KV are the
        # memory the learner phase shares with them)
        record['episodes_in_flight'] = {'learner_start': collector.running}
        learn_started, learn_tokens0 = time.time(), actor_generated(args.actor)
        swap0, mem0 = vmstat(), meminfo()
        progress_mark = time.time()
        if phase_mark['tokens'] is not None and learn_tokens0 is not None:
            phase_totals['collect_s'] += learn_started - phase_mark['time']
            phase_totals['collect_tokens'] += learn_tokens0 - phase_mark['tokens']
        sums = collections.Counter()
        rho_pre_max, rho_post_max, forward_tokens = 0., 0., 0
        teacher_log_rho = []   # per-chain arrays of the teacher tokens' log rho (``teacher_rho_summary``)
        rng = random.Random(args.seed + step)
        rng.shuffle(chains)
        # Shared prefixes (the system prompt across groups, a group's opening prompt, an episode's earlier calls) are
        # computed once per pass and every batch is packed: a prefix tree over the step's chains (`--share-prefix-tokens`;
        # chains reading imported native state stay whole), exact gradients (lfm3_fast_train.run_prefix_tree).
        share_prefix_tokens, tree_mode = exp0_sft.prefix_tree_settings(args)
        nodes = exp0_sft.step_tree(chains, share_prefix_tokens)

        def tokens_of(index, start):
            return len(chains[index]['ids']) - start + exp0_sft.source_tokens(chains[index])
        chain_tokens = sum(len(c['ids']) for c in chains)
        computed_tokens = lfm3_fast_train.tree_tokens(nodes, [tokens_of(i, 0) for i in range(len(chains))])
        cuda_ids = {}

        def ids_of(index, start, end):
            if index not in cuda_ids:
                cuda_ids[index] = torch.from_numpy(chains[index]['ids'].astype(np.int64)).cuda()
            return cuda_ids[index][start:end]

        def chain_states(spans):
            """[(row, decoded block)] of one sequence's imports, each block read once (not kept past the sequence)."""
            blocks = {}
            for _, block_id in spans:
                if block_id not in blocks:
                    blocks[block_id] = collector.states.load(block_id)
            return [(row, blocks[block_id]) for row, block_id in spans]

        def stored_states(net):
            return lambda spans: lfm3_fast_train.import_tensors(net, chain_states(spans))

        def batch_inputs(net, items):
            """(ids, positions, targets, imports, lengths, per-leaf target counts) of one packed leaf batch [(index,
            start)] (each chain from ``start``; its rows before are a computed prefix). A consumer call's imported
            Neuralese state is computed in the same forward from its producer document (``exp0_sft.row_inputs``: the
            documents follow the batch's chains); a block without one is the actor's stored state as fixed context."""
            ids, positions, targets, _, lengths, imports = exp0_sft.row_inputs(
                [chains[index] for index, _ in items], 'cuda', not args.no_neuralese_passthrough, stored_states(net),
                starts=[start for _, start in items])
            counts = [exp0_sft.chain_loss_tokens(chains[index]) for index, _ in items]
            return ids, positions, targets, imports, lengths, counts

        def run_tree(leaf, backward=True):
            return lfm3_fast_train.run_prefix_tree(model, nodes, ids_of, leaf, tokens_of, pack_tokens, args.activation_gb,
                                                   backward, mode=tree_mode if share_prefix_tokens > 0 else 'levels',
                                                   batch_tokens=args.tree_batch_tokens)

        refs = {}
        # Sampled decisions: log pi(chosen) = log_softmax(s / T)[chosen], s_o the summed log-probability of option o's
        # continuation after the exact prompt (as the actor scored it); TIS and the KL term on the chosen option as for a token.
        batches = list(decision_batches(decisions, args.decision_batch_tokens))

        def decision_inputs(batch):
            ids, lengths, positions, targets, owner, counts, spans, sources = decision_tensors(batch, 'cuda')
            plan = lfm3_fast_train.passthrough_plan([(0, spans, sources)], len(ids),
                                                    frozen_all=args.no_neuralese_passthrough)
            if plan['documents']:
                ids = torch.cat([ids] + [torch.from_numpy(np.asarray(d, np.int64)).cuda() for d in plan['documents']])
                lengths = lengths + [len(d) for d in plan['documents']]
            stored = lfm3_fast_train.import_tensors(model, chain_states(plan['fixed'])) if plan['fixed'] else None
            imports = lfm3_fast_train.passthrough_tensors(plan, 'cuda', stored)
            options = sum(counts)

            def scores_of(net):
                token = -exp0_sft.token_nll(net, forward(net, ids, lengths, imports), positions, targets)
                return torch.zeros(options, device='cuda', dtype=token.dtype).index_add(0, owner, token)
            return ids, counts, scores_of

        ref_scores_of = []                # per decision batch: the start policy's option scores
        if reference is not None:
            # the reference pass: the policy module with the start policy's weights (ReferenceWeights), chains and
            # decisions, before any gradient exists; then the policy's own weights back from its masters
            t = time.time()
            reference.load_into(live)
            try:
                def reference_leaves(items, run):
                    # the same batches as the policy pass below: each row's rounding is the policy's
                    ids, positions, targets, imports, lengths, counts = batch_inputs(model, items)
                    logp = -exp0_sft.token_nll(model, run(ids, lengths, imports), positions, targets).detach()
                    for (index, _), part in zip(items, logp.split(counts)):
                        refs[index] = part
                with torch.enable_grad():
                    run_tree(reference_leaves, backward=False)
                for batch in batches:
                    ids, _, scores_of = decision_inputs(batch)
                    exp0_sft.set_checkpointing(model, len(ids), args.activation_gb)
                    with torch.enable_grad():
                        ref_scores_of.append(scores_of(model).detach())
                for p in live.values():
                    p.grad = None
            finally:
                ReferenceWeights.restore(live, masters)
            torch.cuda.synchronize()
            phase['reference_forward'] = phase.get('reference_forward', 0.) + time.time() - t
            phase_peak('reference')
        allocate_grads()
        record['episodes_in_flight']['policy_start'] = collector.running
        policy_started = time.time()

        def train_leaves(items, run):
            nonlocal rho_pre_max, rho_post_max, forward_tokens, progress_mark
            ids, positions, targets, imports, lengths, counts = batch_inputs(model, items)
            h = run(ids, lengths, imports)
            if args.top_p < 1:
                # Top-p: the policy is the learner's distribution on its own nucleus (a constant set that always holds
                # the sampled token), so the policy term's log-probability is logp - log(nucleus mass); the KL term
                # and rho keep the untruncated logp (the actor's record is untruncated too: rho is the ratio of the two
                # truncated laws with the actor's nucleus mass taken as the learner's, |d log rho| <= -log(top_p)).
                nll, nucleus_nll, outside = exp0_sft.token_nll_nucleus(model, h, positions, targets, args.top_p)
                logp_rows, policy_rows = -nll, -nucleus_nll
                sums['outside_nucleus'] += int(outside.sum())
                sums['nucleus_log_mass'] += (logp_rows.detach() - policy_rows.detach()).sum().item()
            else:
                logp_rows = policy_rows = -exp0_sft.token_nll(model, h, positions, targets)
            loss = 0.
            for (index, _), logp, policy_logp in zip(items, logp_rows.split(counts), policy_rows.split(counts)):
                chain = chains[index]
                actor, row_weight = behaviour_logp(chain, logp)
                token_scale = scales[chain['episode']][0]
                ref = refs.pop(index, None)
                with torch.no_grad():
                    log_rho = logp.detach() - actor
                    rho = log_rho.clamp(max=20).exp()
                    weight, tail = policy_weight(rho, actor, chain['advantage'], args.rho_cap, args.tail_logp_floor)
                    if row_weight is not None:
                        weight = weight * row_weight
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
                    loss_terms = loss_terms + args.kl_beta * kl_term(d, args.kl_estimator)
                    kl_stats(sums, d)
                loss = loss + loss_terms.sum() * token_scale
                sums['rho'] += rho.sum().item()
                if chain.get('role') == 'teacher':   # JOLT+ student form: pi(a|s) / mu(a|s, guidance) of the teacher's tokens
                    teacher_log_rho.append(log_rho.float().cpu().numpy())
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
            loss.backward()
            forward_tokens += len(ids)
            del h, logp_rows, policy_rows, loss
            if time.time() - progress_mark > 300:
                progress_mark = time.time()
                log_line(log, {'event': 'learner progress', 'step': step + 1, 'leaf_tokens': forward_tokens,
                               'policy_s': round(time.time() - policy_started), 'meminfo_gb': meminfo(),
                               'swap_pages': {k: v - swap0[k] for k, v in vmstat().items()}})
        record['prefix_tree'] = run_tree(train_leaves)
        torch.cuda.synchronize()
        phase['policy_forward_backward'] = time.time() - policy_started
        forward_tokens = computed_tokens
        cuda_ids.clear()
        record.update(chain_tokens=chain_tokens, shared_prefix_tokens_saved=chain_tokens - computed_tokens)
        phase_peak('policy')
        decision_records = []
        for batch_index, batch in enumerate(batches):
            ids, counts, scores_of = decision_inputs(batch)
            t = time.time()
            ref_scores = ref_scores_of[batch_index] if reference is not None else None
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
                    term = term + args.kl_beta * kl_term(d, args.kl_estimator)
                    kl_stats(sums, d)
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
        if batches:
            phase_peak('decisions')
        if decision_records:
            (work / 'decisions').mkdir(exist_ok=True)
            with (work / 'decisions' / f'step-{step + 1:05d}.jsonl').open('w') as handle:
                handle.writelines(json.dumps(r) + '\n' for r in decision_records)
        record['decision_diagnostics'] = decision_summary(decision_records)
        t = time.time()
        saver.wait()                      # a state save in flight must finish before the masters change
        phase['wait_state_save'] = time.time() - t
        if offload is not None:
            phase['moments_in'] = offload.get()   # bit-exact read back of the moments the last step wrote out
        t = time.time()
        embedding_master.grad[special_index] = 0
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip).item()
        if not math.isfinite(grad_norm):
            # a NaN/inf gradient would make every FP32 master and AdamW moment NaN, and the next state save would
            # overwrite the only training state with it: drop this step's update, keep the weights
            release_grads()
            collector.active_slots = args.slots
            log_line(log, {**record, 'event': 'non-finite gradient: update skipped', 'grad_norm': repr(grad_norm),
                           'loss_sums': {k: v for k, v in sums.items() if isinstance(v, float)}})
            continue
        cosine = moment_cosine(parameters, optimizer)   # this step's gradient against the previous steps' momentum
        data_scale = lr_data_scale(spread, args.lr_reference_spread)
        lr = args.lr * min(1., (step + 1) / args.warmup) * data_scale
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        release_grads()
        changed_step = changed_start = 0
        with torch.no_grad():
            embedding_master[special_index] = saved_rows
            for name, p in live.items():
                new = masters[name].to(torch.bfloat16)
                changed_step += int((new != p).sum())
                p.copy_(new)
            if reference is not None:      # the start policy read back from its file, one tensor at a time (none kept)
                changed_start = sum(int((live[name] != start_weights.to(live[name].device)).sum())
                                    for name, start_weights in reference.tensors())
        torch.cuda.synchronize()
        phase['optimizer'] = time.time() - t
        phase_peak('optimizer')
        # Hand the learner phase's cached activation blocks back for the collect phase: on unified memory PyTorch's cache is
        # system RAM, and grpo-v2-run1 held ~56 GB of it while 32 slots collected (guard stop 2026-11-14 04:56, swap).
        torch.cuda.empty_cache()
        release_host_memory()
        record['cuda_reserved_collect_gb'] = round(torch.cuda.memory_reserved() / 2**30, 1)
        elements = sum(p.numel() for p in live.values())
        record.update(bf16_changed_step_fraction=round(changed_step / elements, 6),
                      bf16_changed_vs_start_fraction=round(changed_start / elements, 6) if reference is not None else None)
        if probes:
            record['probe_learner_nll'] = round(probe_learner_nll(), 5)
        if behaviour is not None:
            t = time.time()
            start = json.loads(behaviour_start_path.read_text()) if behaviour_start_path.exists() else None
            record['behaviour'] = behaviour_probe.summary(behaviour_probe.measure(model, behaviour), start)
            record['behaviour_shift'] = behaviour_probe.shifted(record['behaviour'])
            phase['behaviour_probe'] = time.time() - t
            phase_peak('behaviour_probe')
            if record['behaviour_shift']:
                print(f"behaviour shift at step {step + 1}: {record['behaviour']}", flush=True)
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
        save_native = (step % args.save_every == 0 or (args.max_steps and step == args.max_steps) or held_due
                       or step <= args.keep_checkpoints_until)
        sync.publish(step, on_published, also=(lambda s=step: write_native(s)) if save_native else None)
        branch = branch_root / f'state-step-{step:05d}.safetensors' if args.branch_states else None
        if branch is not None or (args.state_every and step % args.state_every == 0):
            if branch is not None:     # each step's state, the last N kept: controls branch from any of them
                saver.wait()
                for old in sorted(branch_root.glob('state-step-*.safetensors'))[:-(args.branch_states - 1) or None]:
                    old.unlink()
            tensors = {}
            for name in live:
                st = optimizer.state[masters[name]]
                tensors['master/' + name] = masters[name]
                tensors['exp_avg/' + name] = st['exp_avg']
                tensors['exp_avg_sq/' + name] = st['exp_avg_sq']
            adam_step = optimizer.state[parameters[0]]['step'].item()
            saver.save(branch or state_path, tensors, {'step': step, 'adam_step': adam_step, 'cursor': collector.cursor,
                                             'next_group': collector.next_group, 'source_sha256': source_sha})
        if offload is not None:
            record['moment_offload'] = dict(offload.last) or None   # the previous step's put and this step's get
            offload.put([optimizer.state[masters[name]][key] for name in live for key in ('exp_avg', 'exp_avg_sq')],
                        after=saver)
        learner_seconds = sum(v for k, v in phase.items()
                              if k in ('reference_forward', 'policy_forward_backward', 'decision_forward_backward', 'optimizer'))
        n_dec = max(1, len(decisions))
        mass_total = sums['mass_think'] + sums['mass_post_think'] + sums['mass_decision']
        elapsed = time.time() - run['started']
        record.update(
            loss_tokens=loss_tokens, forward_tokens=forward_tokens, lr=lr, lr_data_scale=round(data_scale, 4),
            grad_norm=round(grad_norm, 4),
            kl_mean=round(sums['kl'] / units, 7), ref_abs_dlogp_mean=round(sums['ref_abs_dlogp'] / units, 6),
            ref_dlogp_max=round(sums['ref_dlogp_max'], 3), ref_dlogp_over_5=sums['ref_dlogp_over_5'], kl_estimator=args.kl_estimator,
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
            teacher_rho=teacher_rho_summary(teacher_log_rho, args.rho_cap),
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
            previous_state_save=saver.last, cuda_peak_gb=max([v['allocated'] for v in cuda_peaks.values()], default=None),
            cuda_peak_by_phase_gb=cuda_peaks, host_pinned_gb=host_pinned_gb(),
            outstanding_groups=collector.outstanding, elapsed_s=round(elapsed))
        record.update(grad_spike(work, grad_norm))
        if record.get('grad_spike'):
            print(f"grad spike: grad_norm {grad_norm:.3g} is {record['grad_norm_vs_median']}x the run's median", flush=True)
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
    if offload is not None:
        offload.wait()
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
    t.add_argument('--keep-moments-resident', action='store_true',
                   help='keep the AdamW moments in memory between steps (default: written to WORK/'
                        'adam-moments.offload, NVMe, after each step and read back bit for bit before the next optimizer step: '
                        '-13.6 GB at the learner peak; MomentOffload)')
    t.add_argument('--learner-phase-slots', type=int, default=0,
                   help='collector slots that take new episodes while the learner computes (in-flight episodes always '
                        'finish). 0: none. The GB10 time-slices the actor and the learner, and a few-sequence decode costs '
                        'nearly a full batch\'s GPU time: run1\'s 4 learner-phase slots decoded at 16 tok/s per sequence '
                        '(34 alone) and slowed the learner (GRPO_STEP_THROUGHPUT_20261117.md)')
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
    t.add_argument('--dynamic-sampling-max-spread', type=float, default=0.1,
                   help='dynamic sampling (Collector._wave/_after_wave, wave_plan): a group whose task is likely to pass or '
                        'fail every sample (empirical-Bayes PassRatePrior from --pass-rates, the run\'s own groups and the '
                        'task\'s family) runs a first wave of k samples; when they all agree and the posterior chance that '
                        'the other n - k would still give reward spread is at most this, those are never started. '
                        'grpo-v3-run1 replayed offline at 0.1: 8.6%% fewer samples (7.8%% of generated tokens), 4.4%% of '
                        'spread groups lost (DECISIONS 2026-11-18). 0: off')
    t.add_argument('--pass-min-scored', type=int, default=2)
    t.add_argument('--explore', type=float, default=0.4,
                   help='share of groups from unmeasured pool tasks (TASK_SUPPLY_20261113: a learnable-task group leaves '
                        'the band at 0/8 or 8/8 about 22%% of the time and an explored group enters it about 40%%, so '
                        'below ~0.35 the learnable set shrinks)')
    t.add_argument('--explore-order', choices=('row', 'family'), default='row',
                   help='order of the unmeasured-task walk: row (the shuffled pool, so a family is explored in proportion '
                        'to its rows) or family (round robin over families, each in shuffled order; equal exploration per '
                        'family, so small new families are reached; TASK_SUPPLY_20261117)')
    t.add_argument('--explore-wrapper-share', type=float, default=0.15,
                   help='explored groups that are one-function dataset wrappers, at most this share (ExploreWalk; held-v3 '
                        'has 9%% wrappers, a family-ordered walk over the pool ~36%%); 1: no cap')
    t.add_argument('--explore-held-weight', type=float, default=3.,
                   help='an unmeasured task declaring a construct the held set tests (sft_sources.HELD_CONSTRUCTS: '
                        'Neuralese results/arguments, decide, folder reducers, iterateOn, readouts, call depth >= 3, '
                        'several functions) is explored this many times as readily as one declaring none (ExploreWalk); 1: off')
    t.add_argument('--neuralese-variants', help='Neuralese-declaring variants of pool tasks (cases.jsonl from '
                   'scripts/rl_tasks/neuralese_variants.py; off by default: a controlled addition)')
    t.add_argument('--neuralese-variant-share', type=float, default=0.5,
                   help='share of the groups of a task with a variant that run the variant')
    t.add_argument('--teacher-attempts', type=int, default=0,
                   help='JOLT+: privileged teacher attempts of the same task for a group whose students succeeded at most '
                        '--teacher-max-successes times, trained in student form (RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md)')
    t.add_argument('--teacher-max-successes', type=int, default=0)
    t.add_argument('--teacher-fact-kinds', default=None, help='comma list of fact kinds that guide a teacher (default: rl/jolt.py '
                   'TEACHER_FACT_KINDS, no answers); "all" keeps every kind')
    t.add_argument('--teacher-selection', choices=('crisp', 'nl'), default='crisp',
                   help='selectGuidanceFacts implementation (crisp: no extra model call)')
    t.add_argument('--teacher-plan', choices=('crisp', 'nl'), default='crisp',
                   help='attempts per group: crisp (--teacher-attempts each) or the NL planTeacherAttempts')
    t.add_argument('--hindsight-trajectories', help='hindsight index (rl/hindsight.py index): a teacher attempt of a task '
                   'with a verified run is guided by writeHint over that run')
    t.add_argument('--sd-weight', type=float, default=0.,
                   help='self-distillation: replay recent rare verified successes as a positive-advantage term of this '
                        'weight (0: off; design 0.1)')
    t.add_argument('--sd-window', type=int, default=4, help='steps a rare success stays replayable')
    t.add_argument('--sd-episodes', type=int, default=16, help='replayed episodes per step at most')
    t.add_argument('--sd-max-success', type=float, default=0.25,
                   help="a group's successes are rare when its students' success share is at most this")
    t.add_argument('--sd-teacher-weight', type=float, default=0.05,
                   help='teacher demonstrations in the SD replay: verified teacher-pool successes (TeacherDemos) of tasks '
                        'in the pass band or all-fail, as a ratio-free cross-entropy term of this weight (they have no actor '
                        'log-probabilities: rho = 1, tail floor on the learner\'s logp, KL as for every chain); 0: off')
    t.add_argument('--sd-teacher-episodes', type=int, default=8, help='teacher demonstrations per step at most (one per task)')
    t.add_argument('--sd-teacher-max-per-task', type=int, default=4,
                   help='distinct verified teacher episodes kept per task (teacher models taking turns)')
    t.add_argument('--sd-teacher-source', action='append', default=None,
                   help='sft_sources WORK whose extracted, replayed teacher successes are the demonstrations (repeatable; '
                        'default SD_TEACHER_SOURCES)')
    t.add_argument('--memetic-tasks', default=MEMETIC_TASKS,
                   help='memetic "improve program P" rows (python -m natlang_neuralese.rl.memetic_tasks build: '
                        'DIR/cases.jsonl; default: every program, support 4); RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md '
                        'section 3')
    t.add_argument('--memetic-share', type=float, default=0.02,
                   help='fraction of new groups that are memetic episodes (0 opts out; each group runs about 140 nested '
                        'rollouts, ~18 ordinary groups, so 0.02 is ~1/4 of actor rollouts); on by default, owner '
                        '2026-11-18: memetic pays off over time and keeps training, the guided-vs-unguided gap in '
                        'train.jsonl is telemetry, never a stop')
    t.add_argument('--memetic-max-staleness', type=int, default=8,
                   help='staleness bound of a memetic group (its episodes run tens of minutes; --max-staleness for others)')
    t.add_argument('--memetic-group', type=int, default=4)
    t.add_argument('--memetic-model-requests', type=int, default=64)
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
    t.add_argument('--share-prefix-tokens', type=int, default=256,
                   help='a prompt prefix this long or longer that several of a step\'s chains share is computed once per '
                        'pass, exact gradients (lfm3_fast_train.prefix_tree); 0: every chain whole')
    import exp0_sft                  # the learners' shared prefix-tree flags
    exp0_sft.add_prefix_tree_args(t)
    t.add_argument('--pack-tokens', type=int, default=-1,
                   help='pack the learner\'s chains (and the prefix tree\'s nodes and continuations) into varlen batches of '
                        'at most this many tokens; default: the longest row --activation-gb holds without recompute')
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
    t.add_argument('--lr-reference-spread', type=int, default=12,
                   help='a step\'s LR is scaled by its groups with reward spread over this, at most 1 (lr_data_scale; '
                        'grpo-v2-run1 step 3 trained 2 such groups at full LR); 0: off')
    t.add_argument('--weight-decay', type=float, default=0.)
    t.add_argument('--clip', type=float, default=1.0)
    t.add_argument('--kl-beta', type=float, default=0.02)
    t.add_argument('--kl-estimator', choices=('k2', 'k3'), default='k2',
                   help='KL penalty as a loss: k2 = d^2/2 (gradient linear in d = logp_ref - logp), k3 = e^d - d - 1 '
                        '(unbounded gradient on rare tail tokens; grpo-v2-run1 step 7). The logged kl_mean is k3 either way')
    t.add_argument('--rho-cap', type=float, default=2.0)
    t.add_argument('--loss-norm', choices=('episode', 'token'), default='token',
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
    t.add_argument('--decision-unit-cap', type=float, default=8.,
                   help='with --decision-weight call, a decision counts at most this many token units (decision_unit); 0: uncapped')
    t.add_argument('--call-fraction', type=float, default=1.,
                   help='train the policy and KL terms on this random fraction of each episode\'s calls (subsample_calls; '
                        'the rest stay context): fewer forward tokens a step for some variance; 1: all')
    t.add_argument('--behaviour-probe', default=BEHAVIOUR_PROBE,
                   help='start-policy calls scored after every step (behaviour_probe.py): the policy on the start policy\'s own '
                        'samples (sampled-token logp change), entropy, </think> hazard; logged as "behaviour", flagged as behaviour_shift; "-": off')
    t.add_argument('--branch-states', type=int, default=0,
                   help='keep the FP32 training state of each of the last N steps in CHECKPOINT_DIR/state-step-N.safetensors '
                        '(about 21 GB each), so a control can --resume-state from any of them; 0: off')
    t.add_argument('--resume-state', help='with --resume: start from this training state file (e.g. a --branch-states '
                   'file of another run) instead of STATE_DIR/training-state.safetensors')
    t.add_argument('--keep-checkpoints-until', type=int, default=0,
                   help='save a native checkpoint after every step up to this step and never prune those (probes, '
                        'grafting, held evals of early steps); 0: off')
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
    t.add_argument('--no-neuralese-passthrough', action='store_true',
                   help='imported Neuralese state as fixed context (the in-graph state from the producer document, '
                   'detached) instead of a pass-through the consumer\'s loss trains (NEURALESE_PASSTHROUGH_20261117)')
    t.add_argument('--frozen-producer-table', action='store_true',
                   help="keep the actor's producer table (the producing input embedding of Neuralese text writes) at "
                   'the export\'s start values instead of publishing it from the trained embedding each step (WeightSync)')
    t.add_argument('--unpinned', action='store_true',
                   help='run from a live checkout instead of a pinned copy (smokes only; see check_pin)')
    t.add_argument('--held-every-hours', type=float, default=4.)
    t.add_argument('--skip-base-held', action='store_true')
    t.add_argument('--held-baseline', help="a start-policy held result.json to compare with (e.g. held-v2's base run) instead "
                   'of measuring step 0 (compare refuses another engine kind, precision or sampling law; another series '
                   'pairs with a warning)')
    t.add_argument('--held-strict-series', action='store_true',
                   help='held comparisons pair only one scoring series and one attested engine (held_eval compare '
                        '--strict-series): a step 0 or Pop base of another series or running cap is re-measured')
    t.add_argument('--held-samples', type=int, default=2, help='samples per held task at the periodic measurements')
    t.add_argument('--held-samples-start', type=int, default=4, help='samples per held task for the start policy (step 0)')
    t.add_argument('--held-extra', help='comma-separated native checkpoints measured on the actor in place after step 0, '
                   'each compared with step 0, before training starts')
    t.add_argument('--held-samples-extra', type=int, default=2)
    t.add_argument('--held-workers', type=int, default=96, help='collector slots of a held measurement on the training actor')
    t.add_argument('--held-remote', choices=('pop',), help="measure held on Pop's GPU in parallel with training instead of "
                   'pausing collection on the training actor (pop_held_loop)')
    t.add_argument('--held-remote-slots', type=int, default=20,
                   help="episodes in flight against Pop's actor, = --pop-max-running: the FP8 hybrid-SWA pool (280k tokens "
                        'at mem-fraction 0.77) holds 20 running sequences (smoke 2026-11-17: full usage p99 0.90, no queue, '
                        'no retraction, +28%% tokens/s over 14; POP_HELD_THROUGHPUT_20261117.md)')
    t.add_argument('--pop-kv-cache-dtype', choices=('fp8_e4m3', 'auto'), default='fp8_e4m3',
                   help="Pop actor's KV dtype (auto = BF16, its engine until 2026-11-13); part of the attested engine: "
                        'a start-policy result of another setting is moved aside and re-measured')
    t.add_argument('--held-base', type=Path, help="with --held-remote: the base's held result directory (a result measured by "
                   'its own job, e.g. the pin base on Pop); the loop waits for it to complete instead of measuring the start '
                   'policy, and compares every checkpoint against it')
    t.add_argument('--pop-max-running', type=int, default=20,
                   help="Pop actor's concurrent sequences; attested (max_running): the base must be measured with the same")
    t.add_argument('--unit-prefix', default='grpo')
    t.add_argument('--replay', help="learner control: train on a finished run's archived groups (its ROLLOUTS dir of "
                   "<group>.rollouts.jsonl.zst, --min-groups a step in group order) instead of collecting; no actor, "
                   "no held eval; needs --actor-store and --actor-export of that run")
    t.add_argument('--replay-max-group-tokens', type=int, default=0,
                   help='with --replay: skip archived groups with more sampled tokens than this (a fast learner bench on '
                        "short groups: 55 of run1's 349 groups are under 16k); 0: all")
    t.add_argument('--replay-schedule', help='with --replay: a JSON list of group-number lists, the batch of each step '
                   '(an old run\'s own steps, to re-execute its learner on its data)')
    t.add_argument('--advantage-control', choices=('none', 'zero', 'shuffle'), default='none',
                   help="learner control: 'zero' trains every episode at advantage 0 (KL only: the policy must stay put), "
                   "'shuffle' permutes each step's episode advantages across episodes (no consistent drift allowed)")
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
        # a split group (dynamic sampling) holds as few as 2 slots: room for slots / 2 groups in flight
        args.max_outstanding = args.max_outstanding or args.max_groups + -(-args.slots // (
            2 if args.dynamic_sampling_max_spread > 0 else args.group))
        train(args)


if __name__ == '__main__':
    main()
