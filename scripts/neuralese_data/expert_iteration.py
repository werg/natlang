#!/usr/bin/env python3
"""Expert iteration: one loop in which every source feeds a per-task pool of verified trajectories in the student's own
distribution (DECISIONS 2026-11-20 "expert iteration"; owner-approved). A cycle:

  select    the GRPO pool (registry role grpo-pool) minus the held closure and errata, ordered for collection: tasks the
            latest measurements put in the pass band, then measured failures, then unmeasured tasks, then (only with
            ``--include-solved``) tasks always solved; families taking turns within each tier -> TASKS.json (ids)
  collect   best-of-k on those tasks with the policy being trained: its SGLang actor (grpo's engine flags and sampling
            law, ``grpo.actor_command``) at full slots, grpo's ``Collector`` (episodes, scoring, dynamic sampling), one
            group of ``--k`` samples per task, until the list ends or ``--hours`` pass. WORK/groups/g*/ hold the same
            records a GRPO run writes (rollouts.jsonl, score.json, rewards.jsonl), so ``sft_sources extract --rl
            WORK:WORK/rollouts`` reads the successes and teacher_gen ``--student-groups`` reads the outcomes.
  frontier  the policy's measured pass rate per task from its collections -> FRONTIER.json (natlang.ei-frontier/1):
            tier solved (rate >= ``--solved``), sometimes, never. ``sft_sources build --frontier`` keeps per task the
            cheapest verified source: own successes where any exist (a solved task keeps ``--anchor-per-task``), teacher
            trajectories (student-filled reasoning first, ``filled_reasoning.py``) only where none does.
  cycle     the driver: pin -> held of the start policy (in the background) -> select -> collect (with
            filled-reasoning generation on the same actor) -> frontier -> teacher frontier campaign -> extract, replay,
            filled build, frontier build -> Ambient SFT (exp0_sft default) -> held of the candidate -> verdict, each
            stage resumable (DIR/stages/<name>.done) and pinned (one commit for actor, harness, scorer, learner and eval).
            The held series (``--held-profile``, async-train §1) is the DGX's by default: the pinned held engine (SGLang,
            vendor projection, FP8 KV, BF16 weights) in a ledger unit, 4 samples per task, every DGX collection yielding
            slots while it runs; ``pop`` keeps the legacy Pop series. Results of two hosts never pair.
            Collection and SFT may run on Colab G4 sessions (``--collect-profile``/``--sft-profile colab``; DGX
            fallback after ``--fallback-wait-minutes``), charged in DIR/cycle-cost.json from each session's start to stop.
            Pipelined (DECISIONS 2026-11-21 "pipelined cycles"; ``--no-pipeline`` opts out): when the SFT checkpoint
            exists the driver launches the next cycle from it (``--speculative-after`` this cycle), which selects and
            collects while Pop judges the candidate; its gate stage, after the collection, reads this cycle's verdict
            and trains from the candidate (adopted) or from this cycle's start (rejected: a 'failure' verdict
            recollects from that start, any other rejection keeps the sibling policy's verified collection and merges
            the start's own frontier over it). Filled reasoning gets a dedicated actor of the trained policy in the
            CPU-only window after the collection (``--no-fill-window``), closed before the SFT takes the GPU, and a
            fixed share beside the collection (``--fill-inflight`` more engine slots than collection slots).

    expert_iteration.py select   --out TASKS.json [--rates-groups GLOB ...] [--frontier OLD.json ...]
    expert_iteration.py collect  --work W --checkpoint C --tasks TASKS.json [--k 4] [--hours 6] [--slots 128]
    expert_iteration.py frontier --out F.json --collect W [--collect W2 ...] --policy SHA
    expert_iteration.py cycle    --dir DIR --start CKPT --commit SHA [...]
    expert_iteration.py dgx-held --out DIR --checkpoint C --policy-id SHA --held-pin PIN --label L --unit U
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import shlex
import signal
import statistics
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE), str(REPO / 'scripts/rl_tasks')]

from natlang_neuralese.common.paths import root  # noqa: E402

FRONTIER_SCHEMA = 'natlang.ei-frontier/1'
DATA = root('data_nvme') / 'lfm3-rl-dgx-20261029'
OPS = DATA / 'ops'
BASE_CKPT = str(root('data_hdd') / 'natlang-development-data/archive/data/lfm3-rl-dgx-20261029/fresh-init/checkpoint.pt')
BASE_SHA = '5ff07638bbedba7c8ec506f763ce5fb7511c7755ce8c045ef764bca00ddc1d6a'
PY = str(REPO / '.venv-neuralese/bin/python')
LEDGER = ['/usr/bin/python3', str(REPO / 'scripts/memory_ledger.py')]
# An async round's size (DECISIONS 2026-11-23 "async round size"): ~12M weighted target tokens is ~360 steps at
# 30k loss tokens per step, ~3.7 h of DGX SFT (36.8 s/step) or ~1 h on an A100: about the snapshot + prepare time
# of the next round, and twice the new data one round interval adds (own successes, fills, teacher). The uncapped
# build trains every filled chain ever made (share=all) each round: r0001 was 37.4M (1247 steps, ~12.7 h on the DGX).
ROUND_TARGET_TOKENS = 12_000_000
# Placement-aware round size (DECISIONS 2026-11-24 "placement-aware round budgets"): a round prepared while a Colab
# session fit for SFT (colab_sft_fit) answers is sized to --colab-round-hours of that GPU's learner time, in forward
# (chain) tokens, since the learner's wall time follows them: sft_sources build --max-forward-tokens, with the target
# cap and the teacher forward budget scaled alike. Throughput: what this loop measured (ROUND_SPEED_FILE, per GPU name,
# written after every round), else these defaults (aei1 r0001, sum of chain tokens over sum of step seconds: the A100
# 337-532 at 9.4 s/step, the GB10 1-354 at 48.8 s/step), else the GB10's times sft_speedup.
COLAB_ROUND_HOURS = 2.0
ROUND_SPEED_FILE = 'round-speed.json'
DEFAULT_CHAIN_TOKENS_PER_S = (('A100', 30_800), ('GB10', 5_900))
DGX_GPU = 'GB10'
FORWARD_PER_TARGET = 10.7
# FALLOW-22 (DECISIONS 2026-11-24 "fallow-22 in the round mix"): 11,259 reference-program fan-in episodes (fetaqa,
# 2wiki, bird) re-rendered on today's runtime (broad_graph_october.py, registry sft-fallow22-broad-graphs-20261124-v1,
# held-v3/v4 verified). ~95 forward tokens per target token and templated root code: a small share (1% of the loss,
# ~10% of a round's forward); --no-fallow22-aux opts out.
FALLOW22_AUX = (f'fallow22={DATA.parent / "sft-fallow22-broad-graphs-20261124-v1" / "chains.pkl"},'
                'share=0.01,task_tokens=2000')     # forward tokens per weighted target token of a round mix (r0001 10.87, r0002 10.57)
FILL_GB = 3.5       # filled_reasoning.py generate client (sft-v8 fill/popfill peaks 2.0-2.7 GB); 6 kept the Pop fill out beside the SFT
# ledger GB per collector slot (an episode's node processes, shared 8 per process; AGENTS.md ~250 MB measured): the DGX
# collection's budget has used 0.3 since sft-v7, and remote-actor (Colab) collections and DGX held clients use the same
COLLECTOR_SLOT_GB = 0.3
# Collection defaults (async-train §3 FP8 collection weights, §8 JOLT+ guided attempts; every collection path, cycle
# and async driver, reads them). FP8: fork ec27f0809a runs online FP8 weights in LFM3's natlang projections and reloads
# them in place; an actor on an older fork checkout or a GPU below sm_89 serves BF16 and logs
# sglang_weight_dtype_fallback (serve_native_checkpoint.serving_weight_dtype). JOLT+: custody resolves the held
# packets' data_nvme: paths (fb75ccb78); guided attempts are leak-filtered and guidance-stripped before any pool.
COLLECT_WEIGHT_DTYPE = 'fp8'
TEACHER_ATTEMPTS = 2
# Memetic "improve program P" groups in every collection (owner 2026-11-18: memetic stays on in training; the guided
# gap is telemetry, not a gate): grpo's --memetic-share default. A memetic sample's own verified success (its committed
# guidance beat the trivial improvements on private query worlds, task_rollouts.memetic_success) enters the SFT mix as
# an own success of the supervisor's calls. Each memetic sample runs ~30 nested rollouts beside its own slot, so at most
# MEMETIC_MAX_GROUPS memetic groups run at once in a collection, and none starts when less than its episode wall clock
# is left before the collection's deadline (the collection would wait for it). 0 opts out (--memetic-share 0).
MEMETIC_SHARE = 0.02
MEMETIC_MAX_GROUPS = 2
# Ledger priorities (scripts/memory_ledger.py: a waiting job preempts only resumable units of strictly lower priority).
# The round pipeline's critical-path stages (frontier, data, build, SFT, GRPO, merge/publish) run high; collections
# (the standing DGX one, remote collection clients, a cycle's collect and recollect) run medium and resumable, so a
# waiting stage preempts them: the ledger requeues a preempted collection with its launch spec (the same policy, pin and
# work dir) and the collector resumes from its scored groups (``collect``: first_group after the last, already_scored),
# its fill waits for the actor on the same port. 2026-11-23: the standing aei1 collection (high, 74 GB) held its own
# round's build (medium, 24 GB) and every medium job behind it for 96 minutes. Opt-out: the driver's environment
# NATLANG_EI_COLLECTION_PRIORITY=high (the earlier rule; requeue-aware joins stay).
STAGE_PRIORITY = 'high'
COLLECTION_PRIORITY = os.environ.get('NATLANG_EI_COLLECTION_PRIORITY', 'medium')
# DGX held series (async-train §1): a held measurement's claim file here makes every DGX collection shrink to
# HELD_PRIORITY_SHARE of its slots while it runs (held episodes take priority over collection)
HELD_PRIORITY_DIR = OPS / 'ei' / 'held-priority'
HELD_PRIORITY_SHARE = 0.75    # the collection yields a quarter of its slots (~ the held actor's running cap)
HELD_PRIORITY_STALE = 300     # seconds without a heartbeat after which a claim is ignored
# The DGX held actor is sized to fit beside a collection (121 GB ledger - 74 GB collection - 15 GB diagnostic headroom
# = 32 GB): the pinned engine (engine.json serve_args: its attested arithmetic, overlap off) with a 3 GB FP8 KV pool
# (~450k tokens with the hybrid SWA layout, 16 x 28k) and 16 running; ledger budget = KV + 24 (actor rule) + 16 client
# slots x 0.3. KV size is not attested and the running cap is the same for every DGX-series result, so the series
# stays one engine. The collection actor cannot measure: it runs the overlap scheduler and 160 running, and the held
# pin refuses overlap_schedule (held_eval measure without --allow-other-engine).
HELD_KV_GB = 3
HELD_MAX_RUNNING = 16
HELD_SLOTS = 16
HELD_ACTOR_GB = HELD_KV_GB + 24


def log(record):
    print(json.dumps({'time': round(time.time(), 1), **record}, default=str), flush=True)


# ---- rates and tiers (pure, tested) ----------------------------------------------------------------------------------
def episode_success(row):
    """One scored rewards.jsonl row's success: a memetic improvement episode's ``memetic_success`` (its committed change
    set beat the trivial improvements on the private query worlds; ``task_rollouts.memetic_success``), else a binary
    reward of 1."""
    if 'memetic_success' in row:
        return bool(row['memetic_success'])
    return row['reward'] == 1.0


def group_outcomes(group_dirs):
    """task id -> [successes, scored] over the student samples of GRPO-format group directories (rewards.jsonl; JOLT+
    teacher attempts and unscored samples left out)."""
    rates = collections.defaultdict(lambda: [0, 0])
    for directory in group_dirs:
        path = Path(directory) / 'rewards.jsonl'
        if not path.exists():
            continue
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('reward') is None or row.get('jolt_role', 'student') != 'student':
                continue
            rates[row['task_id']][0] += episode_success(row)
            rates[row['task_id']][1] += 1
    return dict(rates)


def tier(successes, scored, solved=0.75, min_scored=2):
    """The frontier tier of a measured task: solved (pass rate >= ``solved``), sometimes, never; unmeasured below
    ``min_scored`` samples."""
    if scored < min_scored:
        return 'unmeasured'
    p = successes / scored
    return 'solved' if p >= solved else 'sometimes' if successes else 'never'


def selection_order(rows, rates, rng, include_solved=False):
    """Pool rows ordered for collection: tier 0 measured in (0, 1), 1 measured 0, 2 unmeasured, 3 measured 1 (left out
    unless ``include_solved``); each tier family round-robin (``grpo.family_round_robin``)."""
    from grpo import family_round_robin
    tiers = collections.defaultdict(list)
    for row in rows:
        s, n = rates.get(row['id'], (0, 0))
        t = 2 if not n else 3 if s == n else 1 if s == 0 else 0
        if t == 3 and not include_solved:
            continue
        tiers[t].append(row)
    out = []
    for t in sorted(tiers):
        members = list(tiers[t])
        rng.shuffle(members)
        out += family_round_robin(members, rng)
    return out, {['band', 'failed', 'unmeasured', 'solved'][t]: len(v) for t, v in sorted(tiers.items())}


# ---- select ----------------------------------------------------------------------------------------------------------
def pool_args(args):
    import grpo
    from types import SimpleNamespace
    return SimpleNamespace(pool=args.pool or grpo.default_pools(), held_manifest=args.held_manifest or grpo.default_held(),
                           exclude_manifest=[], task_ids=None, exclude_family_regex=args.exclude_family_regex,
                           exclude_ids=[], no_default_errata=False, seed=args.seed, pass_rates=None, pass_band=(0.1, 0.9),
                           pass_min_scored=1, explore=None, max_model_requests=None, case_seconds=None)


def select(args):
    import glob
    import grpo
    out = Path(args.out)
    rows, excluded = grpo.load_tasks(pool_args(args), None)
    rates = {}
    if args.base_rates:
        rates.update({k: (v['successes'], v['scored']) for k, v in json.loads(Path(args.base_rates).read_text()).items()
                      if v.get('scored')})
    dirs = sorted({d for pattern in args.rates_groups for d in glob.glob(pattern)})
    for task_id, (s, n) in group_outcomes(dirs).items():   # the newest measurements (later runs) replace older ones
        rates[task_id] = (s, n)
    for path in args.frontier:
        for task_id, v in json.loads(Path(path).read_text())['tasks'].items():
            if v.get('scored'):
                rates[task_id] = (v['successes'], v['scored'])
    ordered, counts = selection_order(rows, rates, random.Random(args.seed), args.include_solved)
    if args.limit:
        ordered = ordered[:args.limit]
    out.write_text(json.dumps([r['id'] for r in ordered]) + '\n')
    report = {'eligible': len(rows), 'held_excluded': excluded, 'rated': len(rates), 'group_dirs': len(dirs),
              'tiers': counts, 'selected': len(ordered), 'families': len({r.get('family') for r in ordered})}
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=1) + '\n')
    # the rates the order came from (filled_reasoning generate --prefer-rates: teacher episodes of failed tasks first)
    out.with_suffix('.rates.json').write_text(json.dumps({k: {'successes': s, 'scored': n} for k, (s, n) in rates.items()},
                                                         sort_keys=True) + '\n')
    log({'event': 'selected', **report})


# ---- held priority (pure, tested) ------------------------------------------------------------------------------------
def held_priority_claims(directory, now=None, alive=None):
    """Live held-measurement claims in ``directory`` (``<unit>.json`` with ``pid``, refreshed by a heartbeat): a claim
    whose process is gone or whose heartbeat is older than HELD_PRIORITY_STALE is ignored."""
    now = time.time() if now is None else now
    alive = alive or _pid_alive
    out = []
    for path in sorted(Path(directory).glob('*.json')):
        try:
            claim = json.loads(path.read_text())
            fresh = now - path.stat().st_mtime < HELD_PRIORITY_STALE
        except (OSError, ValueError):
            continue
        if fresh and alive(claim.get('pid')):
            out.append(claim)
    return out


def priority_slots(slots, claims, share=HELD_PRIORITY_SHARE):
    """A collection's active slots: all of them, or ``share`` of them (at least 1) while a held measurement runs."""
    return slots if not claims or share >= 1 else max(1, int(round(slots * share)))


def held_budget_gb(slots=HELD_SLOTS):
    """Ledger GB of one DGX held measurement unit: the held actor and its collector slots."""
    return round(HELD_ACTOR_GB + slots * COLLECTOR_SLOT_GB)


def ledger_queued(unit):
    """Whether the ledger retains ownership, including preemption/requeue transitions."""
    from scripts.memory_ledger import unit_owned
    return unit_owned(unit)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


# ---- collect ---------------------------------------------------------------------------------------------------------
def parse_remote_target(target, default_slots):
    """``--rebind-actor HOST@FRONT_URL[=SLOTS]`` -> (host, front URL, collector slots; ``default_slots`` without one)."""
    host, _, rest = target.rpartition('@')
    front, sep, slots = rest.rpartition('=')
    if sep and slots.isdigit():
        return host, front, int(slots)
    return host, rest, default_slots


def endpoint_alive(url, sha, timeout=10):
    """Whether the actor front at ``url`` answers and still serves policy ``sha``."""
    try:
        with urllib.request.urlopen(url + '/v1/neuralese/info', timeout=timeout) as response:
            info = json.loads(response.read())
        return (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') == sha
    except (OSError, ValueError):
        return False


def parse_actor_endpoints(raw_actors, default_slots):
    """Parse repeatable ``--actor URL[=SLOTS]`` specs into ``[(url, slots), ...]``."""
    if not raw_actors:
        return []
    items = [raw_actors] if isinstance(raw_actors, str) else list(raw_actors)
    endpoints = []
    for raw in items:
        for piece in str(raw).split(','):
            piece = piece.strip()
            if not piece:
                continue
            if '=' in piece:
                url, _, count = piece.rpartition('=')
                endpoints.append((url.rstrip('/'), max(1, int(count))))
            else:
                endpoints.append((piece.rstrip('/'), max(1, int(default_slots))))
    return endpoints


def endpoint_for_slot(endpoints, slot_index):
    """The endpoint of slot ``slot_index``: each endpoint gets as many slots as it serves; None for a slot beyond the
    endpoints' combined capacity (that slot idles, rather than overloading a live endpoint after another died)."""
    expanded = [url for url, count in endpoints for _ in range(max(1, int(count)))]
    return expanded[slot_index] if slot_index < len(expanded) else None


def frontier_collector(base):
    class FrontierCollector(base):
        """grpo's Collector over a fixed task order: one group of ``--group`` samples per task, in order, no new group
        after the deadline. Runs on remote actor endpoints too (Colab sessions, one or several; grpo's ``slot`` with
        this class's ``endpoint_for_slot`` and ``episode_valid``): an endpoint that stops answering or serving the
        policy is dropped, its in-flight episodes are abandoned (their groups never scored: an episode against a dead
        actor, or one a replaced session serves under another policy, is no verdict on the task) and its slots idle until
        it serves the policy again. A live DGX held measurement (``priority_dir``) shrinks the active slots to
        ``priority_share``."""

        def setup(self, order, deadline, endpoints=None, priority_dir=None, priority_share=HELD_PRIORITY_SHARE,
                  verify=None):
            self.order = collections.deque(t for t in order if t in self.by_id and t not in self.own)
            self.deadline = deadline
            self.endpoints = list(endpoints) if endpoints else [(getattr(self.args, 'actor', None),
                                                                 getattr(self.args, 'slots', 1))]
            # verify each finished episode's actor (``episode_valid``): remote actors (several, or one this collector
            # does not serve); default: when more than one endpoint shares the slots
            self.verify = len(self.endpoints) > 1 if verify is None else bool(verify)
            self.dead, self.dead_checked = set(), {}
            self.priority_dir, self.priority_share, self.priority_read = priority_dir, priority_share, 0.
            self.full_slots = sum(count for _, count in self.endpoints)
            self._learnable()

        def _priority(self):
            """At most every 30 s: the active slots under live held claims (``priority_slots``)."""
            if not self.priority_dir or time.time() - self.priority_read < 30:
                return
            self.priority_read = time.time()
            active = priority_slots(self.full_slots, held_priority_claims(self.priority_dir), self.priority_share)
            if active != self.active_slots:
                print(json.dumps({'event': 'held priority', 'active_slots': active, 'slots': self.full_slots}), flush=True)
                self.active_slots = active

        def _pick(self):
            return self.by_id[self.order.popleft()]

        def _memetic_open(self):
            """A new memetic group may start: fewer than ``--memetic-max-groups`` run, and its wall clock (the
            episode's, the larger of the run's and the rows' ``world.case_seconds``, plus a staged group's preparation
            stage, ``memetic_workspace.group_seconds``) still fits before the deadline."""
            from natlang_neuralese.rl.memetic_workspace import group_seconds
            if not super()._memetic_open():
                return False
            running = sum(1 for info in self.groups.values()
                          if info.get('memetic') and info['done'] < info['size'] and not info.get('abandoned'))
            if running >= getattr(self.args, 'memetic_max_groups', MEMETIC_MAX_GROUPS):
                return False
            layout = getattr(self.args, 'memetic_layout', 'staged')
            run_seconds = int(getattr(self.args, 'case_seconds', 0) or 0)
            tail = max([group_seconds(max(run_seconds, int(world.get('case_seconds') or 7200)),
                                      ((r['semantics'].get('inputs') or {}).get('program') or {}).get('budget', {})
                                      .get('evaluations', 2), layout)
                        for r in self.memetic_rows for world in [(r.get('semantics') or {}).get('world') or {}]]
                       or [7200])
            return time.time() + tail < self.deadline

        def _take(self):
            self._priority()
            if self.paused.is_set():
                return None
            with self.lock:
                if (not self.pending and self.order and time.time() < self.deadline
                        and self.outstanding < self.args.max_outstanding):
                    self._new_group()
                return self.pending.popleft() if self.pending else None

        def finished(self):
            with self.lock:
                done = not self.pending and self.outstanding == 0 and (not self.order or time.time() >= self.deadline)
            return done or self.all_dead()

        def all_dead(self):
            return len(self.endpoints) > 1 and len(self.dead) == len(self.endpoints) and self.outstanding == 0

        def live_endpoints(self):
            for url in list(self.dead):      # a dropped endpoint is retried every 2 minutes
                if time.time() - self.dead_checked.get(url, 0) > 120:
                    self.dead_checked[url] = time.time()
                    if endpoint_alive(url, self.policy_sha):
                        self.dead.discard(url)
                        print(json.dumps({'event': 'endpoint back', 'endpoint': url}), flush=True)
            return [(url, count) for url, count in self.endpoints if url not in self.dead]

        def mark_dead(self, url):
            if url not in self.dead:
                self.dead.add(url)
                self.dead_checked[url] = time.time()
                print(json.dumps({'event': 'endpoint dropped', 'endpoint': url}), flush=True)

        def endpoint_for_slot(self, index=0):
            """Slot ``index``'s live endpoint (``endpoint_for_slot`` over ``live_endpoints``); None while its endpoint is
            dropped or beyond the live endpoints' capacity (the slot idles: grpo's ``slot``)."""
            return endpoint_for_slot(self.live_endpoints(), index)

        def episode_valid(self, actor_url, info):
            """With ``verify`` (remote actors, which this collector neither serves nor rebinds): an episode that ran
            against an actor which no longer answers or serves another policy is no verdict. Its endpoint is dropped
            (retried every 2 minutes, ``live_endpoints``) and its task runs again on a live one."""
            if not self.verify:
                return True
            if actor_url not in self.dead and endpoint_alive(actor_url, self.policy_sha):
                return True
            self.mark_dead(actor_url)
            task = info.get('task_id')
            with self.lock:
                if task in self.by_id and task not in self.order and not self.stop.is_set():
                    self.order.appendleft(task)
            return False
    return FrontierCollector


def wait_actor(url, sha, process, log_path, timeout=3600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'actor exited with {process.returncode} (see {log_path})')
        try:
            with urllib.request.urlopen(url + '/v1/neuralese/info', timeout=10) as response:
                info = json.loads(response.read())
            if (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') == sha:
                return info
        except OSError:
            pass
        time.sleep(10)
    raise TimeoutError('actor did not come up in an hour')


def collect(args):
    import grpo
    from natlang_neuralese.common.code_provenance import code_provenance
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.common.sampling import check_top_p, rl_sampling_controls
    work = Path(args.work)
    (work / 'groups').mkdir(parents=True, exist_ok=True)
    sha = args.checkpoint_sha256 or sha256_file_hex_cached(Path(args.checkpoint))
    order = json.loads(Path(args.tasks).read_text())
    actor_args = list(args.actor_arg)
    collect_w_dtype = getattr(args, 'collect_weight_dtype', COLLECT_WEIGHT_DTYPE)
    if collect_w_dtype == 'fp8' and not any('--sglang-weight-dtype' in a for a in actor_args):
        actor_args += ['--sglang-weight-dtype', 'fp8']
    teacher_attempts = int(getattr(args, 'teacher_attempts', TEACHER_ATTEMPTS))
    teacher_max_succ = int(getattr(args, 'teacher_max_successes', 0))
    memetic_share = float(getattr(args, 'memetic_share', MEMETIC_SHARE))
    g = grpo.build_parser().parse_args([
        'train', '--work', str(work), '--checkpoint', args.checkpoint, '--checkpoint-sha256', sha,
        '--group', str(args.k), '--slots', str(args.slots), '--task-ids', args.tasks, '--actor-port', str(args.actor_port),
        '--dynamic-sampling-max-spread', str(args.dynamic_sampling_max_spread), '--held-every-hours', '0',
        '--no-actor-release', '--actor-overlap', 'on', '--memetic-share', str(memetic_share), '--exclude-family-regex',
        args.exclude_family_regex, '--unit-prefix', args.unit_prefix, '--seed', str(args.seed),
        '--teacher-attempts', str(teacher_attempts), '--teacher-max-successes', str(teacher_max_succ),
        *(['--unpinned'] if args.unpinned else []), *[f'--actor-arg={a}' for a in actor_args]])
    g.top_p = check_top_p(g.top_p)
    g.pool = args.pool or grpo.default_pools()
    g.held_manifest = args.held_manifest or grpo.default_held()
    g.memetic_tasks = getattr(args, 'memetic_tasks', None) or (grpo.default_memetic_tasks() if memetic_share > 0 else None)
    g.memetic_max_groups = int(getattr(args, 'memetic_max_groups', MEMETIC_MAX_GROUPS))
    g.memetic_layout = getattr(args, 'memetic_layout', 'staged')
    g.memetic_fit_quantile = float(getattr(args, 'memetic_fit_quantile', 0.9))
    g.memetic_nested_slots = bool(getattr(args, 'memetic_nested_slots', True))
    g.memetic_seed_collection = getattr(args, 'memetic_seed_collection', None)
    g.max_outstanding = args.slots // 2 + 8
    tasks, excluded = grpo.load_tasks(g, work)
    code = grpo.git_pin()
    pin = grpo.check_pin(work, code['git_commit'], args.unpinned)
    from natlang_neuralese.rl.held_eval import series
    run = {'schema': 'natlang.ei-collect/1', 'started': time.time(), 'code': code, 'pin': pin, 'series': series(),
           'argv': sys.argv, 'checkpoint': args.checkpoint, 'checkpoint_sha256': sha, 'k': args.k, 'hours': args.hours,
           'tasks_file': args.tasks, 'tasks_sha256': sha256_file_hex_cached(Path(args.tasks)), 'eligible': len(tasks),
           'memetic': {'share': memetic_share, 'tasks': g.memetic_tasks, 'max_groups': g.memetic_max_groups,
                       'layout': g.memetic_layout, 'fit_quantile': g.memetic_fit_quantile,
                       'nested_slots': g.memetic_nested_slots},
           'excluded_ids': excluded, 'provenance': code_provenance()}
    with (work / 'runs.jsonl').open('a') as handle:
        handle.write(json.dumps(run, default=str) + '\n')
    actor = None
    endpoints = parse_actor_endpoints(args.actor, args.slots)
    if endpoints:     # remote actors (Colab G4): served and stopped by the cycle (pop_rl_actor serve-checkpoint)
        for ep_url, _ in endpoints:
            wait_actor(ep_url, sha, None, None, timeout=60)
        url = endpoints[0][0]
        total_slots = sum(count for _, count in endpoints)
        sglang = None
        # Neuralese consumer state: each episode runs on one endpoint (producer and consumer calls on one actor's
        # store, fetched over its HTTP front); the SFT build reads states through their producer documents in the
        # episode records (pass-through), never from a store, so no remote store is needed after the collection.
    else:
        command, url = grpo.actor_command(g, work, sha)
        with (work / 'actor.log').open('a') as handle:
            actor = subprocess.Popen(command, cwd=REPO, env=grpo.pinned_env(), stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        wait_actor(url, sha, actor, work / 'actor.log')
        endpoints = [(url, args.slots)]
        total_slots = args.slots
        sglang = url.rsplit(':', 1)[0] + ':' + str(args.actor_port + 1000)

    def stop_actor(*_):
        if actor is not None and actor.poll() is None:
            os.killpg(actor.pid, signal.SIGTERM)
    import atexit
    atexit.register(stop_actor)
    g.actor = url
    g.slots = total_slots
    g.max_outstanding = total_slots // 2 + 8
    (work / 'actor.json').write_text(json.dumps({'url': url, 'endpoints': endpoints, 'sglang': sglang,
                                                 'checkpoint_sha256': sha, 'remote': actor is None,
                                                 'export': str(work / 'actor' / 'sglang-hf') if actor else None,
                                                 'pid': actor.pid if actor else None}) + '\n')
    (work / 'sampling-config.json').write_text(json.dumps(rl_sampling_controls(g.top_p, capture=True)) + '\n')
    existing = [int(d.name[1:]) for d in (work / 'groups').glob('g*') if d.name[1:].isdigit()]
    collector = frontier_collector(grpo.Collector)(g, tasks, work, sha, REPO / 'ts-host',
                                                   first_group=max(existing, default=-1) + 1)
    g.neuralese_variants, g.neuralese_variant_share = args.neuralese_variants, args.neuralese_variant_share
    collector.variants = grpo.load_variants(g, tasks, work)
    # a DGX held measurement takes slots of the DGX actor; a remote actor's collection keeps its own
    collector.setup(order, time.time() + args.hours * 3600, endpoints=endpoints,
                    priority_dir=None if args.no_held_priority or actor is None else Path(args.held_priority_dir),
                    priority_share=args.held_priority_share, verify=actor is None)
    log({'event': 'collect start', 'tasks': len(collector.order), 'already_scored': len(collector.own), 'k': args.k,
         'slots': total_slots, 'actor': url, 'endpoints': endpoints})
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: (stop.set(), collector.stop.set()))
    slots = [threading.Thread(target=collector.slot, args=(i,), daemon=True) for i in range(total_slots)]
    for thread in slots:
        thread.start()
    done, last = 0, 0.
    while not stop.is_set() and not collector.finished():
        taken = collector.take_groups(0, 10_000)
        done += len(taken)
        if time.time() - last > 300:
            last = time.time()
            log({'event': 'progress', 'groups_done': done, 'tasks_left': len(collector.order),
                 'running': collector.running, 'outstanding': collector.outstanding,
                 'dynamic': dict(collector.dynamic)})
        time.sleep(5)
    collector.stop.set()
    for thread in slots:
        thread.join(timeout=60)
    stop_actor()
    rates = group_outcomes(sorted((work / 'groups').glob('g*')))
    summary = {'event': 'collect end', 'groups_done_this_run': done, 'tasks_scored': len(rates),
               'tasks_with_success': sum(1 for s, _ in rates.values() if s), 'episodes': sum(n for _, n in rates.values()),
               'successes': sum(s for s, _ in rates.values()), 'tasks_left': len(collector.order),
               'stopped': stop.is_set() or collector.all_dead(), 'endpoints_dropped': sorted(collector.dead)}
    (work / 'collect.json').write_text(json.dumps({**summary, 'run': run}, indent=1, default=str) + '\n')
    log(summary)
    if not summary['stopped']:
        record_collection(work)


def record_collection(work, closed=True):
    """The collection in the data registry (rollouts the SFT build and teacher campaign read); ``closed`` False: a
    snapshot of a collection that keeps growing (the async driver's standing collection)."""
    from natlang_neuralese.common import data_registry as dr
    dr.record_dataset(dr.dataset_id(work, 'grpo-rollouts'), work, kind='rollouts',
                      include=['groups/*/rollouts.jsonl', 'groups/*/rewards.jsonl', 'groups/*/score.json',
                               'groups/*/cases.jsonl', 'groups/*/versions.json', '*.json', 'runs.jsonl'],
                      license='project-generated (own LFM3 policy samples)',
                      format='expert_iteration collect WORK: groups/gNNNNNN/{rollouts,rewards}.jsonl (grpo group format)',
                      producer={'script': 'scripts/neuralese_data/expert_iteration.py collect'}, closed=closed)


# ---- frontier --------------------------------------------------------------------------------------------------------
def frontier(args):
    from natlang_neuralese.rl.held_eval import training_exclusion_ids
    import grpo
    dirs, policies = [], set()
    for work in map(Path, args.collect):
        runs = [json.loads(l) for l in (work / 'runs.jsonl').open() if l.strip()]
        policies |= {r['checkpoint_sha256'] for r in runs}
        dirs += sorted(work.glob('groups/g*'))
    prev_policy = getattr(args, 'previous_policy', None)
    allowed_policies = {args.policy} | ({prev_policy} if prev_policy and getattr(args, 'allow_adjacent_policies', False) else set())
    if not policies or not policies.issubset(allowed_policies):
        raise SystemExit(f'collections served {sorted(policies)}, not allowed frontier policy set {sorted(allowed_policies)}')
    rates = group_outcomes(dirs)
    held = set(training_exclusion_ids([{'id': t} for t in rates], args.held_manifest or grpo.default_held()))
    leaked = sorted(set(rates) & held)
    if leaked:
        raise SystemExit(f'{len(leaked)} held tasks in the collections (e.g. {leaked[0]})')
    tasks = {t: {'successes': s, 'scored': n, 'tier': tier(s, n, args.solved, args.min_scored)}
             for t, (s, n) in sorted(rates.items())}
    data = {'schema': FRONTIER_SCHEMA, 'policy': args.policy,
            **({'previous_policy': prev_policy} if prev_policy else {}),
            'collections': [str(Path(c).resolve()) for c in args.collect],
            'thresholds': {'solved': args.solved, 'min_scored': args.min_scored},
            'tiers': dict(collections.Counter(v['tier'] for v in tasks.values())), 'tasks': tasks}
    Path(args.out).write_text(json.dumps(data, indent=0) + '\n')
    log({'event': 'frontier', 'tasks': len(tasks), 'tiers': data['tiers']})


def merge_frontier(exact, sibling):
    """A frontier for a policy whose own collection (``exact``) measured only some tasks: the rest come from a sibling
    policy's collection (one SFT step away, rejected by held), marked ``measured_by: sibling``."""
    tasks = {t: {**v, 'measured_by': 'sibling'} for t, v in sibling['tasks'].items()}
    tasks.update({t: {**v, 'measured_by': 'own'} for t, v in exact['tasks'].items()})
    return {**exact, 'sibling_policy': sibling['policy'],
            'collections': [*exact.get('collections', ()), *sibling.get('collections', ())],
            'tiers': dict(collections.Counter(v['tier'] for v in tasks.values())), 'tasks': dict(sorted(tasks.items()))}


def rejected_collection_mode(verdict, setting='auto'):
    """What a speculative cycle does with its collection when the candidate it collected from is not adopted:
    ``recollect`` from the start policy after a 'failure' (a significant held regression: maybe a bug, its behaviour
    is not data), else ``keep`` (verified successes of a sibling policy one SFT step from the start, as close to the
    start's distribution as the previous cycle's own data was to its candidate)."""
    if setting != 'auto':
        return setting
    return 'recollect' if verdict == 'failure' else 'keep'


def next_name(name):
    """sft-v7 -> sft-v8 (None without a trailing number)."""
    import re
    match = re.match(r'^(.*?)(\d+)$', name)
    return f'{match.group(1)}{int(match.group(2)) + 1}' if match else None


# flags of `cycle` that name this cycle (replaced in the next one's argv); every one takes a value
OWN_FLAGS = ('--dir', '--name', '--start', '--start-sha', '--commit', '--pin', '--speculative-after', '--prefer-unseen',
             '--fallback-prefer-unseen', '--next-name', '--next-seed-work', '--fill-repo',
             '--data-repo', '--since-held')


def lineage_builds(mix, trained_before):
    """The --prefer-unseen builds of a cycle that trains this cycle's candidate: its own mix and every build its start
    had already trained (``train_policy()[2]``), without repeats. Only ``[mix]`` made sft-v7's own start data (sft-v6's
    build) look unseen to a cycle training sft-v7's candidate."""
    out = []
    for build in [mix, *trained_before]:
        if str(build) not in map(str, out):
            out.append(build)
    return out


def next_cycle_argv(argv, *, directory, name, start, start_sha, commit, previous, prefer_unseen, fallback_prefer_unseen,
                    extra_seed_work=(), since_held=0, held_pin=None, held_series_host=None):
    """The next pipelined cycle's ``cycle`` arguments: this cycle's (``argv`` after ``cycle``) with its own flags
    replaced, speculative after ``previous``, the extra seed WORKs added. ``held_pin``/``held_series_host``: the held
    series this cycle measured on (its harness pin and engine host), carried as such; without them a ``--held-pin`` in
    ``argv`` is carried unchanged."""
    own = OWN_FLAGS + (('--held-pin', '--held-series-host') if held_pin else ())
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        flag = arg.split('=', 1)[0]
        if flag == '--resume-after-hunt':      # one cycle's decision, never inherited
            continue
        if flag in own:
            skip = '=' not in arg
            continue
        out.append(arg)
    seeds = {out[i + 1] for i, a in enumerate(out[:-1]) if a == '--seed-work'}
    out += [x for w in extra_seed_work if w not in seeds for x in ('--seed-work', str(w))]
    out += ['--dir', str(directory), '--name', name, '--start', str(start), '--start-sha', start_sha, '--commit', commit,
            '--speculative-after', str(previous), *[x for b in prefer_unseen for x in ('--prefer-unseen', str(b))],
            *[x for b in fallback_prefer_unseen for x in ('--fallback-prefer-unseen', str(b))]]
    if since_held:
        out += ['--since-held', str(since_held)]
    if held_pin:
        out += ['--held-pin', str(held_pin), '--held-series-host', held_series_host or 'pop']
    return out


def result_host(result):
    """The machine a held result was measured on (``held_eval.engine_host_of``: its series' engine host, else its
    attested export format); a result attesting nothing reads as Pop, where every earlier cycle measured."""
    from natlang_neuralese.rl.held_eval import engine_host_of
    return engine_host_of(result) or 'pop'


def reusable_held(results, policy, samples, series_key, host='pop', *, any_series=False):
    """The first complete held result of ``policy`` at ``samples`` per task in one scoring series measured on ``host``
    (a measurement an earlier cycle made on the same series: pairing rules unchanged, the eval spared 3 h). The host
    is checked apart from the key: a pin that predates host keys gives Pop and DGX results one key. ``any_series``
    (owner 2026-11-22: held comparability across series matters less than eval hours): without such a result, any
    complete result of the policy at >= 2 samples per task, in any series and on any host; ``compare`` still refuses
    pairs of different engine settings and records the series difference as a warning."""
    loaded = []
    for path in results:
        try:
            r = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not (r.get('complete') and r.get('policy_id') == policy):
            continue
        if (r.get('samples_per_task') == samples and (r.get('series') or {}).get('key') == series_key
                and result_host(r) == host):
            return Path(path)
        loaded.append((Path(path), r))
    if any_series:
        for path, r in loaded:
            if int(r.get('samples_per_task') or 0) >= 2:
                return path
    return None


def latest_base(results, samples, host='pop'):
    """The newest complete held result of the base at ``samples`` per task measured on ``host``, in any scoring series
    of that host (a Pop base never pairs with a DGX candidate)."""
    found = []
    for path in results:
        try:
            r = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (r.get('complete') and r.get('policy_id') == BASE_SHA and r.get('samples_per_task') == samples
                and result_host(r) == host):
            found.append((Path(path).stat().st_mtime, Path(path).parent))
    return max(found)[1] if found else None


# ---- cycle -----------------------------------------------------------------------------------------------------------
TEACHER_POOLS = [OPS / 'teacher-gen-20261115' / p for p in ('pool-v1', 'pool-v2-rich', 'pool-progtune-decisions')] + [
    OPS / 'teacher-gen-20261119' / 'gap-v1']
# the provider key files the campaign's launch script sources (~/.config/natlang/NAME.env)
TEACHER_ENV_FILES = ('nvidia', 'gemini', 'anthropic', 'openrouter')
NIM = 'https://integrate.api.nvidia.com/v1'
# Gemini on Vertex AI, billed to the GCP project's credits (the AI Studio key is free tier: no Pro models); the ADC
# user credentials mint an hour-long token per episode (teacher_gen api_key_command)
VERTEX = 'https://aiplatform.googleapis.com/v1/projects/awareness-481308/locations/global/endpoints/openapi'
VERTEX_TOKEN = [str(Path.home() / 'google-cloud-sdk/bin/gcloud'), 'auth', 'application-default', 'print-access-token']
# the teacher campaign's teachers (DECISIONS 2026-11-21 teacher yield, 2026-11-24 teacher providers; ops/ei/teachers.json
# overrides at each launch): Haiku retires itself while the Anthropic account has no credit; Luna at medium effort; GPT-6.1
# Sol, the stronger model on the same Codex subscription, only on tasks another teacher failed (min_failed); the free
# raw-reasoning models: Nemotron 3.5 Super and 3 Ultra on NIM (one account-wide rate limit: few slots), ERNIE 5.5 Flash
# on OpenRouter's free tier (1,000 requests a day, ~26 episodes); Gemini 3.1 Pro and Gemini 4 Argon (Vertex credits,
# reasoning hidden) on tasks another teacher failed; family routing and --attempts in teacher_gen
TEACHERS = [
    {'model_id': 'claude-haiku-5-5', 'provider': 'anthropic', 'reasoning_effort': 'medium', 'slots': 4},
    {'model_id': 'gpt-6-luna', 'provider': 'openai-codex', 'reasoning_effort': 'medium', 'slots': 6},
    {'model_id': 'gpt-6.1-sol', 'provider': 'openai-codex', 'reasoning_effort': 'medium', 'slots': 4, 'min_failed': 1},
    {'model_id': 'nvidia/nemotron-3.5-super-vl-120b-a12b', 'endpoint': NIM, 'api_key_env': 'NVIDIA_API_KEY', 'slots': 2},
    {'model_id': 'nvidia/nemotron-3-ultra-550b-a55b', 'endpoint': NIM, 'api_key_env': 'NVIDIA_API_KEY', 'slots': 1},
    {'model_id': 'baidu/ernie-5.5-flash-preview:free', 'endpoint': 'https://openrouter.ai/api/v1',
     'api_key_env': 'OPENROUTER_API_KEY', 'slots': 1},
    *[{'model_id': model, 'endpoint': VERTEX, 'api_key_env': 'VERTEX_ACCESS_TOKEN', 'api_key_command': VERTEX_TOKEN,
       'slots': slots, 'min_failed': 1} for model, slots in (('google/gemini-4-argon', 4), ('google/gemini-3.1-pro-preview', 2))]]


def frontier_rare(frontier_path, rate=0.25):
    """The frontier's tasks the policy never or rarely solves (``never``, or a success rate <= ``rate``): the teacher
    campaign's targets."""
    return [t for t, v in json.loads(Path(frontier_path).read_text())['tasks'].items()
            if v['tier'] in ('never',) or v['successes'] / max(1, v['scored']) <= rate]


def teacher_command(pin, work, covered, student_groups, limit):
    """``teacher_gen.py`` on a policy's new failures (gap tiers: student_rare first from ``student_groups``; every
    campaign in ``covered`` covers its successes)."""
    return [PY, str(Path(pin) / 'scripts/neuralese_data/teacher_gen.py'),
            '--work', str(work), '--pause-below-available-gb', '25', '--exclude-family-regex', '(?!)',
            *[x for c in covered if Path(c) != Path(work) and (Path(c) / 'manifest.jsonl').exists()
              for x in ('--covered-by', str(c))],
            *[x for g in student_groups for x in ('--student-groups', str(g))], '--rare-rate', '0.25',
            '--limit', str(limit), '--attempts', '3', *[x for t in teachers() for x in ('--teacher', json.dumps(t))]]


TEACHERS_FILE = OPS / 'ei' / 'teachers.json'


def teachers():
    """The teacher campaign's specs: ``ops/ei/teachers.json`` (a JSON list of teacher_gen --teacher specs) when it
    exists, read at each campaign launch, so a teacher change reaches the standing campaign's next launch without a
    driver restart; else TEACHERS."""
    if TEACHERS_FILE.exists():
        specs = json.loads(TEACHERS_FILE.read_text())
        if not isinstance(specs, list) or not specs or not all(isinstance(t, dict) and t.get('model_id') for t in specs):
            raise SystemExit(f'{TEACHERS_FILE}: a non-empty JSON list of teacher specs')
        return specs
    return TEACHERS


def launch_teacher(runner, work, frontier_path, student_groups, unit, covered_extra=()):
    """Launch the teacher campaign for ``frontier_path``'s rare tasks as ledger unit ``unit`` (detached) through
    ``runner`` (a Cycle: its pin and ledger); returns the number of target tasks (0: nothing launched)."""
    rare = frontier_rare(frontier_path)
    if not rare:     # teacher_gen --limit 0 means every task
        return 0
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    covered = [*TEACHER_POOLS, *sorted(OPS.glob('ei/*/teacher')), *covered_extra]
    command = teacher_command(runner.pin, work, covered, student_groups, len(rare))
    script = work / 'launch.sh'
    script.write_text('#!/bin/bash\n# expert-iteration teacher frontier (expert_iteration.py launch_teacher)\nset -u\n'
                      f'set -a; for f in {" ".join(TEACHER_ENV_FILES)}; do . {Path.home()}/.config/natlang/$f.env; done; set +a\n'
                      f'cd {runner.pin}\nexec ' + ' '.join(map(_quote, command)) + '\n')
    script.chmod(0o755)
    # CPU/API-bound, ~4 GB at 20 slots: medium so medium-priority smokes and evals do not preempt it (each preemption
    # kills its running paid episodes); a high-priority training run still can
    runner.ledger(unit, 6, 'medium', ['/bin/bash', str(script)], work / 'launch.log', detach=True)
    return len(rare)


LIVE_REGISTRY = 'live-holdout-registry.jsonl'


def link_live_works(work, live_works):
    """Link every extraction of the ``--live-work`` WORKs (observed-live teacher episodes a standing ingester keeps
    extending, ``live_campaign_ingest.py``) into WORK, as the seed WORKs' are: each round's build then reads the
    extractions present when it prepares, without a restart. Returns the number of new links."""
    n = 0
    for live in map(Path, live_works):
        for f in sorted(live.glob('episodes*.jsonl.zst')):
            link = Path(work) / f'{f.name[:-len(".jsonl.zst")]}-{live.name}.jsonl.zst'
            if not link.is_symlink() and not link.exists():
                os.symlink(f.resolve(), link)
                n += 1
    return n


def live_holdout_registry(live_works):
    """The open-goal holdout registry (catalog copy) the ``--live-work`` WORKs carry, for the build's
    ``--live-holdout-registry`` (an open-goal episode trains only against its catalog's registration), or None."""
    from natlang_neuralese.common.hashing import sha256_file_hex
    found = {}
    for live in map(Path, live_works):
        path = live / LIVE_REGISTRY
        if path.exists():
            found.setdefault(sha256_file_hex(path), path)
    if len(found) > 1:
        raise SystemExit(f'--live-work WORKs carry {len(found)} different holdout registries; the build takes one')
    return next(iter(found.values()), None)


def mix_build_command(repo, *, work, out, frontier, aux, prefer_unseen, teacher_budget, own_share, previous_policy=None,
                      live_registry=None, max_target_tokens=0, registry_repo=None, registry_ref=None,
                      max_forward_tokens=0, unfilled_live_teacher=None):
    """The mix build (``sft_sources build``) of a cycle or an async round: Ambient frontier build from WORK's replayed
    successes (own first, then teacher; observed-live teacher episodes against ``live_registry``), the teacher pools,
    aux parts and the lineage's prefer-unseen builds."""
    return [PY, Path(repo) / 'scripts' / 'neuralese_data' / 'sft_sources.py', 'build', '--work', work,
            '--out', out, '--tokenizer', Path(work) / 'tokenizer', '--frontier', frontier,
            '--allow-stale-teacher', '--october-share', '0', '--exclude-run', 'grpo-v2/',
            '--exclude-run', 'grpo-v4/run1', '--teacher-forward-budget', teacher_budget,
            '--own-loss-share', own_share, '--max-task-target-tokens', '40000', '--val-episodes', '40',
            *[x for p in TEACHER_POOLS for x in ('--teacher-pool', p)],
            *[x for a in aux for x in ('--aux', a)],
            *[x for b in prefer_unseen for x in ('--prefer-unseen', b)],
            *(['--previous-policy', previous_policy] if previous_policy else []),
            *(['--live-holdout-registry', live_registry] if live_registry else []),
            *(['--registry-repo', registry_repo] if registry_repo is not None else []),
            *(['--registry-ref', registry_ref] if registry_ref is not None else []),
            *(['--max-target-tokens', max_target_tokens] if max_target_tokens else []),
            *(['--max-forward-tokens', max_forward_tokens] if max_forward_tokens else []),
            *(['--unfilled-live-teacher', unfilled_live_teacher] if unfilled_live_teacher else [])]


def chain_tokens_per_s(train_jsonl, first_step=0):
    """(chain tokens per second, steps) of a learner log's steps from ``first_step`` on: the sum of their chain tokens
    over the sum of their step seconds (what a round's wall time follows), or (None, 0)."""
    path = Path(train_jsonl)
    if not path.exists():
        return None, 0
    tokens = seconds = steps = 0
    for line in path.read_text(errors='replace').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get('step_seconds') and r.get('chain_tokens') and int(r.get('step', 0)) >= first_step:
            tokens, seconds, steps = tokens + float(r['chain_tokens']), seconds + float(r['step_seconds']), steps + 1
    return (round(tokens / seconds) if seconds else None), steps


def learner_throughput(gpu_name, measured, hardware=None, measured_speedup=None):
    """(chain tokens per second of round SFT on ``gpu_name``, source): this loop's measurement (``measured``: the
    ROUND_SPEED_FILE table), else the defaults by name, else the GB10's times ``sft_speedup`` (None when unknown)."""
    name = str(gpu_name or '')
    if measured.get(name, {}).get('chain_tokens_per_s'):
        return float(measured[name]['chain_tokens_per_s']), 'measured by this loop'
    known = next((v for key, v in DEFAULT_CHAIN_TOKENS_PER_S if key in name), None)
    if known:
        return float(known), 'default (aei1 r0001)'
    from natlang_neuralese.common.execution_profile import sft_speedup
    speedup, source = sft_speedup(hardware or {'gpu_name': name}, measured_speedup or {})
    if not speedup:
        return None, source
    dgx, _ = learner_throughput(DGX_GPU, measured)
    return dgx * speedup, f'GB10 x {speedup} ({source})'


def placement_round_size(*, round_target_tokens, teacher_budget, hours, gpu_name, chain_tokens_per_s,
                         forward_per_target=FORWARD_PER_TARGET):
    """A round's budgets for a learner on ``gpu_name`` at ``chain_tokens_per_s``: ``hours`` of learner time in forward
    tokens (``max_forward_tokens``), the target cap and the teacher forward budget scaled by the same factor against
    the DGX round (``round_target_tokens`` weighted, about ``forward_per_target`` times that forward), never below it."""
    forward = int(hours * 3600 * chain_tokens_per_s)
    scale = max(1.0, forward / (round_target_tokens * forward_per_target)) if round_target_tokens else 1.0
    return {'placement': 'colab', 'gpu': gpu_name, 'hours': hours, 'chain_tokens_per_s': round(chain_tokens_per_s),
            'max_forward_tokens': forward, 'max_target_tokens': int(round_target_tokens * scale),
            'teacher_budget': int(teacher_budget * scale), 'scale': round(scale, 3),
            'forward_per_target': round(forward_per_target, 3)}


def judge(vs_start, success_pts, fail_pts):
    """(verdict, adopt) of a held comparison against the reference (the declared criteria, DECISIONS): failure below
    -``fail_pts`` with p < 0.05, success at +``success_pts`` without more ran-out episodes, else neutral; a neutral
    candidate is adopted when it is ahead and does not run out more."""
    d, p = vs_start['primary']['delta'], vs_start['primary']['permutation_p']
    ran_out = vs_start.get('ran_out_rate_delta') or 0
    if d < -fail_pts / 100 and p < 0.05:
        verdict = 'failure'
    elif d >= success_pts / 100 and ran_out <= 0.02:
        verdict = 'success'
    else:
        verdict = 'neutral'
    return verdict, verdict == 'success' or (verdict == 'neutral' and d > 0 and ran_out <= 0.02)


def promote_policy(runner, checkpoint, sha, *, label, why, unit):
    """Held adopted ``checkpoint`` as the best measured policy: rewrite ``OPS/current-policy.json`` (the
    ``lfm-latest`` native source) and refresh the ``lfm-latest`` and ``lfm-latest-gguf`` profiles (``natlang setup
    --native lfm-latest --gguf q8_0`` as ledger unit ``unit``: the native engine, managed llama-server and the in-page
    wllama then serve it). A failed refresh is an error."""
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.export.native_hf import current_policy_pointer
    write_json_durable(OPS / 'current-policy.json', current_policy_pointer(checkpoint, sha, label=label, why=why))
    natlang = os.environ.get('NATLANG_CLI') or shutil.which('natlang') or str(Path.home() / '.local/bin/natlang')
    runner.ledger(unit, 16, 'medium', [natlang, 'setup', '--native', 'lfm-latest', '--gguf', 'q8_0'],
                  runner.dir / f'{unit}.log', workdir=str(REPO), repo=REPO)


class HeldPairRefused(RuntimeError):
    """``held_eval compare`` refused the pair: the two results come from different engine settings, sampling laws,
    budgets or hosts, so they are no paired measurement."""


class Cycle:
    def __init__(self, args):
        self.a = args
        self.dir = Path(args.dir).resolve()
        (self.dir / 'stages').mkdir(parents=True, exist_ok=True)
        self.pin = Path(args.pin or OPS / 'ei' / f'pin-{args.commit[:12]}')
        self.start_sha = args.start_sha or (BASE_SHA if args.start == BASE_CKPT else
                                            subprocess.check_output(['sha256sum', args.start], text=True).split()[0])
        self.env = {**os.environ, 'NATLANG_REPO': str(self.pin), 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '2',
                    'MKL_NUM_THREADS': '1', 'PYTHONPATH': f'{self.pin}/training/neuralese:{self.pin}'}
        from natlang_neuralese.common.execution_profile import CostTelemetry, parse_colab_hosts, resolve_profile
        self.colab_hosts = parse_colab_hosts(getattr(args, 'colab_hosts', None), OPS / 'pop-behavior-eval' / 'config.json')
        wait_min = float(getattr(args, 'fallback_wait_minutes', 30.0))
        cu_hr = float(getattr(args, 'cu_per_hour', 13.0))
        self.collect_profile = resolve_profile(getattr(args, 'collect_profile', None) or 'dgx', hosts=self.colab_hosts,
                                               cu_per_hour=cu_hr, fallback_wait_minutes=wait_min)
        self.sft_profile = resolve_profile(getattr(args, 'sft_profile', None) or 'dgx', hosts=self.colab_hosts,
                                           cu_per_hour=cu_hr, fallback_wait_minutes=wait_min)
        # the held series (async-train §1): the DGX (default) or Pop (legacy); no other machine has one
        self.held_profile = resolve_profile(getattr(args, 'held_profile', None) or 'dgx')
        if self.held_profile.name not in ('dgx', 'pop'):
            raise SystemExit(f'held evals run on the DGX or Pop, not {self.held_profile.name} (async-train §1)')
        self.held_host = self.held_profile.held_series_host or 'pop'
        # held eval's harness (--held-pin, carried to the next cycle with --held-series-host): one scoring series across
        # cycles, so the start is the previous candidate's measurement and each cycle costs one candidate held
        # (DECISIONS 2026-11-22). A carried pin of another host's series is not this series: a cycle that moves the
        # held series (sft-v10: Pop -> DGX) starts it on its own pin.
        carried = getattr(args, 'held_pin', None)
        if carried and (getattr(args, 'held_series_host', None) or 'pop') != self.held_host:
            carried = None
        self.held_pin = Path(carried or self.pin)
        self.held_env = {**self.env, 'NATLANG_REPO': str(self.held_pin),
                         'PYTHONPATH': f'{self.held_pin}/training/neuralese:{self.held_pin}'}
        if self.held_profile.held_series_host:
            self.held_env['NATLANG_HELD_SERIES_HOST'] = self.held_profile.held_series_host
        if getattr(self.a, 'held_samples', None) is None:
            self.a.held_samples = self.held_profile.held_samples
        self.collect_dir, self.held = self.dir / 'collect', self.dir / 'held'
        self.work = Path(args.sft_work)
        self.mix = self.dir / 'mix'
        self.window_closed = threading.Event()
        self.window_started = False
        self.cost = CostTelemetry.load(self.dir / 'cycle-cost.json', getattr(args, 'name', self.dir.name))

    def record_cost(self, stage, profile, wall_seconds, *, sessions=1, canonical_chain_tokens=0, note='',
                    session_host=None):
        entry = self.cost.record_stage(stage, profile, wall_seconds, sessions=sessions,
                                       canonical_chain_tokens=canonical_chain_tokens, note=note, session_host=session_host)
        summary = self.cost.save(self.dir / 'cycle-cost.json')
        self.note(stage='cost', cost_stage=stage, profile=entry.profile, wall_s=entry.wall_seconds,
                  cu_spent=entry.cu_spent, total_cu=summary['total_cu_spent'],
                  tokens_per_cu=summary['canonical_chain_tokens_per_cu'], note=entry.note)
        return summary

    def session(self, host, profile, opened):
        """Charge a remote session from its start (``opened``) to its stop (``opened`` False): Colab bills the time
        a session runs, busy or idle, so the cycle charges every actor and learner it starts until it stops them."""
        if opened:
            self.cost.open_session(host, profile)
        else:
            self.cost.close_session(host)
        self.cost.save(self.dir / 'cycle-cost.json')

    def choose_profile(self, preferred, stage):
        """(active profile, hosts) for ``stage``: a remote profile waits up to its fallback time for a reachable host,
        then falls back to the DGX (lfm3opti.md §4). ``preferred`` is a profile or its name."""
        from natlang_neuralese.common.execution_profile import resolve_profile, select_stage_profile
        import pop_rl_actor
        if isinstance(preferred, str):
            preferred = resolve_profile(preferred, hosts=self.colab_hosts)
        hosts = (self.colab_hosts if preferred.name == 'colab' else
                 [pop_rl_actor.POP] if preferred.name == 'pop' and pop_rl_actor.POP else ())
        if stage == 'sft' and preferred.name == 'colab':
            # an answering session slower than the DGX for SFT (an L4) is no SFT host: without another the stage
            # falls back to the DGX at once instead of waiting for a session that will not get faster
            unfit = {h: pop_rl_actor.colab_sft_fit(h) for h in hosts if pop_rl_actor.host_reachable(h)}
            unfit = {h: reason for h, (ok, reason) in unfit.items() if not ok}
            if unfit:
                self.note(stage=stage, unfit_sessions=unfit)
                hosts = [h for h in hosts if h not in unfit]
                if not hosts:
                    fallback = resolve_profile('dgx')
                    self.note(stage=stage, profile_fallback='every answering Colab session is slower than the DGX '
                                                            'for SFT', active_profile=fallback.name)
                    return fallback, ['local']
        active, hosts, reason = select_stage_profile(preferred, reachable_fn=pop_rl_actor.host_reachable, hosts=hosts)
        if reason:
            self.note(stage=stage, profile_fallback=reason, active_profile=active.name)
        return active, hosts

    def gate(self):
        path = self.dir / 'stages' / 'gate.done'
        return json.loads(path.read_text()) if path.exists() else None

    def train_policy(self):
        """(checkpoint, sha, prefer-unseen builds) of the policy this cycle trains: the start, or after a speculative
        cycle's gate the policy the previous verdict chose."""
        g = self.gate()
        if g:
            return g['start'], g['start_sha'], g['prefer_unseen']
        return self.a.start, self.start_sha, list(getattr(self.a, 'prefer_unseen', []))

    def collect_policy(self):
        """(checkpoint, sha) the collection serves: the start (speculatively the previous candidate), or the trained
        policy after a gate that recollects."""
        g = self.gate()
        if g and g.get('collection') == 'recollect':
            return g['start'], g['start_sha']
        return self.a.start, self.start_sha

    def note(self, **record):
        line = json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **record}, default=str)
        with (self.dir / 'drive.log').open('a') as handle:
            handle.write(line + '\n')
        print(line, flush=True)

    def done(self, stage):
        return (self.dir / 'stages' / f'{stage}.done').exists()

    def mark(self, stage, **info):
        (self.dir / 'stages' / f'{stage}.done').write_text(json.dumps({'time': time.time(), **info}) + '\n')
        self.note(stage=stage, done=True, **info)

    @staticmethod
    def active(unit):
        return any(subprocess.run(['systemctl', '--user', 'is-active', '-q', f'{unit}{suffix}.service']).returncode == 0
                   for suffix in ('', '-waiter'))

    def present(self, unit):
        """Whether ``unit`` runs or the ledger holds it (waiting, or requeued after a preemption until its relaunch)."""
        return self.active(unit) or ledger_queued(unit)

    def wait(self, unit, poll=60, between=None):
        """Until ``unit`` ends (a preempted, requeued unit has not ended); a failed unit ends the cycle (its stage must
        not be marked done). ``between``: called every poll while it runs (a failure is logged, never the wait's end)."""
        time.sleep(20)
        while self.present(unit):
            if between is not None:
                try:
                    between()
                except Exception as error:
                    self.note(stage='wait', unit=unit, between_failed=repr(error)[:300])
            time.sleep(poll)
        if subprocess.run(['systemctl', '--user', 'is-failed', '-q', f'{unit}.service']).returncode == 0:
            raise SystemExit(f'{unit} failed: see its log; rerun the cycle to resume')

    def ledger(self, unit, budget, priority, command, log_path, detach=False, workdir=None, repo=None):
        """``command`` (argv) as ledger unit ``unit``; waits for it unless ``detach``; a running unit is joined.
        ``repo``: the checkout on NATLANG_REPO/PYTHONPATH (default the pin). A unit the ledger holds requeued (preempted,
        awaiting its relaunch) is joined too, never launched a second time."""
        if not self.present(unit):
            subprocess.run(['systemctl', '--user', 'reset-failed', f'{unit}.service'], stderr=subprocess.DEVNULL)
            shell =' '.join(map(_quote, command)) + f' >> {_quote(str(log_path))} 2>&1'
            values = {**self.env, **({'NATLANG_REPO': str(repo), 'PYTHONPATH': f'{repo}/training/neuralese:{repo}'}
                                     if repo else {})}
            env = ['/usr/bin/env', *[f'{k}={values[k]}' for k in ('NATLANG_REPO', 'PYTHONPATH', 'HF_HUB_OFFLINE',
                                                                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS')]]
            subprocess.run([*LEDGER, 'run', '--unit', unit, '--budget-gb', str(budget), '--priority', priority,
                            '--wait', '36000', '--detach', '--workdir', str(workdir or self.pin), '--',
                            *env, '/bin/sh', '-c', f'exec {shell}'], check=True)
        if not detach:
            self.wait(unit)

    def py(self, script, *args):
        return [PY, str(self.pin / 'scripts' / 'neuralese_data' / script), *map(str, args)]

    # stages
    def pin_stage(self):
        if not (self.pin / '.natlang-pin.json').exists():
            self.ledger(f'{self.a.name}-pin', 4, 'high', ['/usr/bin/python3', str(REPO / 'scripts/pin_checkout.py'),
                                                          'create', self.a.commit, str(self.pin)],
                        self.dir / 'pin.log', workdir=str(REPO))
        if not (self.pin / '.natlang-pin.json').exists():
            raise SystemExit(f'pin failed: {self.dir}/pin.log')
        self.mark('pin', pin=str(self.pin))

    def measure_held(self, name, checkpoint, reuse=True):
        """One held-v3 measurement of ``checkpoint`` into DIR/held/NAME on this cycle's held series (``reuse`` False:
        measured here, never copied from another cycle's result)."""
        kw = {} if reuse else {'reuse': False}
        return self.dgx_held(name, checkpoint, **kw) if self.held_host == 'dgx' else self.pop_held(name, checkpoint, **kw)

    def held_result_ok(self, out):
        """Whether ``out`` holds a complete result of this cycle's held host; a result of another host (a Pop start
        beside a DGX candidate) moves aside to ``<name>-<host>-<time>`` so it is re-measured on this series."""
        path = out / 'result.json'
        if not path.exists():
            if (out / 'run.json').exists() and result_host(json.loads((out / 'run.json').read_text())) != self.held_host:
                self.set_held_aside(out)
            return False
        r = json.loads(path.read_text())
        if result_host(r) != self.held_host:
            self.set_held_aside(out, result_host(r))
            return False
        return bool(r.get('complete'))

    def set_held_aside(self, out, host='other'):
        aside = out.with_name(f'{out.name}-{host}-{int(time.time())}')
        n = 0
        while aside.exists():        # set aside twice within a second
            n += 1
            aside = out.with_name(f'{out.name}-{host}-{int(time.time())}-{n}')
        os.replace(out, aside)
        self.note(stage='held', moved_aside=str(aside), reason=f'measured on {host}, the series is {self.held_host}')

    def dgx_held(self, name, checkpoint, reuse=True):
        """One held-v3 measurement on the DGX held series (async-train §1): ``dgx-held`` as a ledger unit (the pinned
        held engine as its child: SGLang, vendor projection, FP8 KV, BF16 weights, attested against the held manifest's
        engine.json; ``held_eval measure`` with ``--held-samples`` per task), resumable. Preempted (the SFT outranks
        it), it is relaunched and resumes its episodes."""
        out = self.held / name
        if self.held_result_ok(out) or (reuse and self.reuse_held(out, checkpoint)):
            return out
        from natlang_neuralese.common.hashing import sha256_file_hex_cached
        sha = BASE_SHA if str(checkpoint) == BASE_CKPT else sha256_file_hex_cached(Path(checkpoint))
        out.mkdir(parents=True, exist_ok=True)
        unit = f'{self.a.name}-held-{name}'
        budget = held_budget_gb(self.a.held_slots)
        command = [PY, str(Path(__file__).resolve()), 'dgx-held', '--out', out, '--checkpoint', checkpoint,
                   '--policy-id', sha, '--samples', self.a.held_samples, '--slots', self.a.held_slots,
                   '--held-pin', self.held_pin, '--label', f'{self.a.name}-{name}', '--unit', unit]
        started = time.time()
        # medium and resumable: a training stage (high) preempts it, the ledger requeues it and the
        # measurement resumes its episodes (held_eval measure keeps run.json); nothing that trains waits for it
        for attempt in range(50):
            self.ledger(unit, budget, 'medium', list(map(str, command)), out / 'driver.log', repo=self.held_pin)
            while ledger_queued(unit) or self.active(unit):     # preempted: the ledger relaunches it when room returns
                time.sleep(60)
            if self.held_result_ok(out):
                break
            time.sleep(60)
        self.record_cost(f'held:{name}', 'dgx', time.time() - started)
        return out

    def pop_held(self, name, checkpoint, reuse=True):
        """One held-v3 measurement on Pop (ops/grpo-v4/pop-held.sh: the pinned harness, 2 samples)."""
        out = self.held / name
        result = out / 'result.json'
        if result.exists() and json.loads(result.read_text()).get('complete'):
            return out
        if reuse and self.reuse_held(out, checkpoint):
            return out
        import pop_rl_actor
        join_pop_held(out)                  # a measure unit a restarted driver left running: never a second one
        if _result_complete(out):
            return out
        pop_rl_actor.stop_pop_fills()       # the held pin's pop_rl_actor may predate the stop in its serves
        mode = ['base', str(out), str(self.a.held_samples)] if checkpoint == BASE_CKPT else \
            ['ckpt', str(out), str(self.a.held_samples), str(checkpoint)]
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(['/bin/sh', str(OPS / 'grpo-v4' / 'pop-held.sh'), str(self.held_pin), *mode],
                       stdout=(out / 'driver.log').open('a'), stderr=subprocess.STDOUT)
        return out

    def reuse_held(self, out, checkpoint):
        """Copy an earlier cycle's complete measurement of this policy in this pin's scoring series on this cycle's
        held host into ``out`` (``--no-held-reuse`` opts out)."""
        if getattr(self.a, 'no_held_reuse', False):
            return False
        from natlang_neuralese.common.hashing import sha256_file_hex_cached
        policy = BASE_SHA if str(checkpoint) == BASE_CKPT else sha256_file_hex_cached(Path(checkpoint))
        try:
            key = subprocess.check_output([PY, '-c', 'from natlang_neuralese.rl.held_eval import series; '
                                           "print(series()['key'])"], env=self.held_env, text=True,
                                          cwd=self.held_pin).strip()
        except subprocess.CalledProcessError:
            return False
        mine = (out / 'result.json').resolve()
        found = reusable_held([p for p in sorted(OPS.glob('ei/*/held/*/result.json')) if p.resolve() != mine],
                              policy, self.a.held_samples, key, host=self.held_host,
                              any_series=not getattr(self.a, 'strict_held_reuse', False))
        if not found:
            return False
        out.mkdir(parents=True, exist_ok=True)
        (out / 'result.json').write_bytes(found.read_bytes())
        (out / 'reused.json').write_text(json.dumps({'from': str(found), 'series': key, 'policy': policy}) + '\n')
        self.note(stage='held reuse', out=str(out), source=str(found), series=key)
        return True

    def base_unit(self):
        return f'{self.a.name}-popbase' if self.held_host == 'pop' else f'{self.a.name}-heldbase'

    def popbase_stage(self):
        """The start policy (and the base on request) on the held series, in the background: on Pop while the DGX
        collects and trains; on the DGX as ledger units beside the collection (held episodes have priority)."""
        unit = self.base_unit()
        if not self.active(unit):
            subprocess.run(['systemd-run', '--user', '--unit', unit, '--collect', '--setenv', f'PATH={os.environ["PATH"]}',
                            *[f'--setenv={k}={self.held_env[k]}' for k in ('NATLANG_REPO', 'PYTHONPATH')],
                            *(f'--setenv={k}={os.environ[k]}' for k in ('NATLANG_POP_HELD_BUDGET_GB',) if k in os.environ),
                            # the driver's own code (held scheduling); the held pin's harness measures
                            *map(str, (PY, Path(__file__).resolve(), 'pop-held', '--dir', self.dir, '--pin', self.held_pin,
                                       '--start', self.a.start, '--held-samples', self.a.held_samples,
                                       '--held-profile', self.held_profile.name, '--held-slots',
                                       getattr(self.a, 'held_slots', HELD_SLOTS), '--name', self.a.name,
                                       *(('--held-base',) if getattr(self.a, 'held_base', False) else ()),
                                       *(('--after', self.a.speculative_after) if self.speculative else ())))],
                           check=True)
        self.note(stage='popbase', started=unit, host=self.held_host)

    def select_stage(self):
        groups = [str(OPS / 'grpo-v*' / '*' / 'groups' / 'g*'), str(OPS / 'ei' / '*' / 'collect' / 'groups' / 'g*')]
        self.ledger(f'{self.a.name}-select', 6, 'medium',
                    self.py('expert_iteration.py', 'select', '--out', self.dir / 'tasks.json',
                            *[x for g in groups for x in ('--rates-groups', g)], *self.a.select_args.split()),
                    self.dir / 'select.log')
        if not (self.dir / 'tasks.json').exists():
            raise SystemExit(f'select failed: {self.dir}/select.log')
        self.mark('select', **json.loads((self.dir / 'tasks.report.json').read_text()))

    @property
    def speculative(self):
        return bool(getattr(self.a, 'speculative_after', None))

    def max_running(self):
        """The collect actor's running cap: the collection's slots plus the fill share (``--fill-inflight``), so
        filled reasoning draws beside a full collection instead of queueing behind it."""
        return self.a.slots + (self.a.fill_inflight if self.a.fill else 0)

    def set_collect_aside(self, sha):
        """A collection of another policy in DIR/collect (the speculative one before a recollection) moves to
        DIR/collect-spec-<sha12>."""
        runs = self.collect_dir / 'runs.jsonl'
        if not runs.exists():
            return
        served = {json.loads(l)['checkpoint_sha256'] for l in runs.open() if l.strip()}
        if served and served != {sha}:
            aside = self.dir / f'collect-spec-{sorted(served)[0][:12]}'
            os.replace(self.collect_dir, aside)
            self.note(stage='collect', moved_aside=str(aside), served=sorted(served))

    def collect_command(self, unit, start, sha, slots, endpoints=()):
        """``expert_iteration.py collect`` for this cycle (one builder for the DGX actor and remote endpoints): the
        task order, k, hours, Neuralese variants, JOLT+ attempts, FP8 collection weights, slots; the DGX actor's
        engine flags, or ``--actor`` endpoints."""
        collect_w_dtype = getattr(self.a, 'collect_weight_dtype', COLLECT_WEIGHT_DTYPE)
        teacher_attempts = int(getattr(self.a, 'teacher_attempts', TEACHER_ATTEMPTS))
        actor = ([x for ep in endpoints for x in ('--actor', ep)] if endpoints else
                 ['--actor-arg=--sglang-kv-gb', f'--actor-arg={self.a.kv_gb}',
                  '--actor-arg=--sglang-max-running', f'--actor-arg={self.max_running()}'])
        return self.py('expert_iteration.py', 'collect', '--work', self.collect_dir, '--checkpoint', start,
                       '--checkpoint-sha256', sha, '--tasks', self.dir / 'tasks.json',
                       '--k', self.a.k, '--hours', self.a.collect_hours, *self.variant_args(),
                       '--collect-weight-dtype', collect_w_dtype, '--teacher-attempts', teacher_attempts,
                       '--memetic-share', getattr(self.a, 'memetic_share', MEMETIC_SHARE),
                       '--memetic-layout', getattr(self.a, 'memetic_layout', 'staged'),
                       '--memetic-fit-quantile', getattr(self.a, 'memetic_fit_quantile', 0.9),
                       '--memetic-nested-slots' if getattr(self.a, 'memetic_nested_slots', True)
                       else '--no-memetic-nested-slots',
                       '--slots', slots, '--unit-prefix', unit, *actor)

    def serve_colab(self, hosts, start, sha, profile):
        """Serve the collected policy on each reachable Colab G4 session (``pop_rl_actor serve-checkpoint
        --host-profile colab``); [(host, 'URL=SLOTS')] of those that attest it, each charged from its start."""
        served = []
        collect_w_dtype = getattr(self.a, 'collect_weight_dtype', COLLECT_WEIGHT_DTYPE)
        actor_py = [PY, str(self.pin / 'scripts' / 'neuralese_data' / 'pop_rl_actor.py')]
        for host in hosts:
            self.session(host, profile, True)
            try:
                raw = subprocess.check_output(
                    [*actor_py, 'serve-checkpoint', '--host-profile', 'colab', '--host', host,
                     '--checkpoint', str(start), '--policy-id', sha, '--commit', self.a.commit,
                     '--kv-cache-dtype', 'fp8_e4m3', '--weight-dtype', collect_w_dtype],
                    env=self.env, text=True, timeout=3600)
                info = json.loads(raw.strip().splitlines()[-1])
                served.append((host, f'{info["endpoint"]}={profile.collect_slots}'))
            except Exception as err:
                self.note(stage='collect g4 serve failed', host=host, error=repr(err)[:300])
                self.stop_colab(host)
        return served

    def stop_colab(self, host):
        """Stop this cycle's actor on a Colab session and end its charge."""
        actor_py = [PY, str(self.pin / 'scripts' / 'neuralese_data' / 'pop_rl_actor.py')]
        subprocess.run([*actor_py, 'stop', '--host-profile', 'colab', '--host', host], env=self.env,
                       stdout=(self.dir / 'colab-stop.log').open('a'), stderr=subprocess.STDOUT, timeout=600)
        self.session(host, 'colab', False)

    def collect_stage(self):
        start, sha = self.collect_policy()
        recollect = (self.gate() or {}).get('collection') == 'recollect'
        self.set_collect_aside(sha)
        unit = f'{self.a.name}-{"recollect" if recollect else "collect"}'
        started = time.time()
        profile, hosts = self.choose_profile(self.collect_profile, 'collect')
        served = self.serve_colab(hosts, start, sha, profile) if profile.name == 'colab' else []
        if profile.name == 'colab' and not served:
            self.note(stage='collect', profile_fallback='no Colab G4 endpoint came up; falling back to dgx')
        try:
            if served:
                endpoints = [ep for _, ep in served]
                slots = len(endpoints) * profile.collect_slots
                budget = max(8, round(slots * COLLECTOR_SLOT_GB) + 4)
                self.ledger(unit, budget, COLLECTION_PRIORITY,
                            self.collect_command(unit, start, sha, profile.collect_slots, endpoints),
                            self.dir / 'collect.log', detach=True)
            else:
                budget = self.a.kv_gb + 24 + round(self.a.slots * COLLECTOR_SLOT_GB)
                self.ledger(unit, budget, COLLECTION_PRIORITY, self.collect_command(unit, start, sha, self.a.slots),
                            self.dir / 'collect.log', detach=True)
                if self.a.fill:         # filled reasoning draws on the DGX actor (a remote one exposes no SGLang port)
                    self.fill_alongside(unit, sha)
            time.sleep(20)
            while self.present(unit):          # a preempted collection is requeued and resumes: it has not ended
                if self.speculative and not self.gate() and self.recollect_due():
                    self.stop_unit(unit)          # the candidate failed held: its collection is not data (gate recollects)
                    self.stop_unit(f'{self.a.name}-fill')
                    self.note(stage='collect', stopped='previous candidate failed held: recollect at the gate')
                    return
                time.sleep(60)
        finally:
            for host, _ in served:
                self.stop_colab(host)
        if self.a.fill and not served:
            self.stop_unit(f'{self.a.name}-fill')   # its actor is gone: the fill window takes over
        summary = json.loads((self.collect_dir / 'collect.json').read_text())
        name = 'colab' if served else 'dgx'
        self.record_cost('collect', name, time.time() - started, sessions=max(1, len(served)),
                         session_host=','.join(h for h, _ in served) or None)
        if summary.get('stopped'):
            raise SystemExit('collection stopped before its end: rerun the cycle to resume it')
        self.mark('collect', profile=name, sessions=len(served), **{k: v for k, v in summary.items() if k != 'run'})

    def held_schedule(self):
        """(unmeasured cycles before this one, the interval in force), from the previous verdict: an unmeasured one
        counts on; a measured success doubles the interval (up to ``--held-every-max``: reliable progress needs fewer
        evals), a neutral result returns to ``--held-every``, a failure measures the next cycle too (debugging).
        Without a previous verdict that records them, ``--since-held`` and ``--held-every``."""
        base, top = getattr(self.a, 'held_every', 1), max(getattr(self.a, 'held_every_max', 1), 1)
        v = self.previous_verdict() if self.speculative else None
        if v and not v.get('adopt') and (self.gate() or {}).get('verdict_pending'):
            return 0, 1       # trained from a candidate the late verdict rejected: this candidate is judged now
        if not v or 'held_every' not in v:
            return getattr(self.a, 'since_held', 0), base
        if v['verdict'] == 'unmeasured':
            return v['since_held'] + 1, v['held_every']
        if v['verdict'] == 'success':
            return 0, min(2 * v['held_every'], top)
        return 0, 1 if v['verdict'] == 'failure' else base

    def measured(self):
        """Whether this cycle's candidate gets a held eval (``held_schedule``; DECISIONS 2026-11-22 "held every"); the
        others are adopted unmeasured and Pop's hours go to rollouts."""
        since, every = self.held_schedule()
        return since + 1 >= every

    def reference(self):
        """The last held-measured policy this cycle's training descends from: what a measured candidate is judged
        against and what a rejected one rolls back to. A previous verdict without one names its next_start."""
        if self.speculative:
            v = self.previous_verdict()
            if v:
                return v.get('reference') or {'checkpoint': v['next_start']}
        return {'checkpoint': str(self.train_policy()[0])}

    def previous_verdict(self):
        path = Path(self.a.speculative_after) / 'verdict.json'
        return json.loads(path.read_text()) if path.exists() else None

    def recollect_due(self):
        v = self.previous_verdict()
        return bool(v) and not v['adopt'] and \
            rejected_collection_mode(v['verdict'], self.a.rejected_collection) == 'recollect'

    def gate_stage(self):
        """A speculative cycle (started from the previous cycle's candidate while its held eval runs) learns the verdict:
        adopted, it trains the candidate as collected; rejected, it trains the previous start (``rejected_collection_mode``
        decides whether the collection is kept or redone). Training never waits for a held eval: a verdict not in after
        ``--gate-wait-hours`` (default 0) adopts the candidate as collected, as an unmeasured cycle is; the verdict,
        when it comes, is read at this cycle's held stage (its reference and rollback target, and a rejection makes this
        cycle's candidate measured)."""
        deadline = time.time() + 3600 * getattr(self.a, 'gate_wait_hours', 0.)
        while not self.previous_verdict() and time.time() < deadline:
            time.sleep(min(300, max(1, deadline - time.time())))
        v = self.previous_verdict()
        if v is None:
            info = {'start': self.a.start, 'start_sha': self.start_sha, 'prefer_unseen': list(self.a.prefer_unseen),
                    'collection': 'adopted', 'verdict_pending': True, 'previous_verdict': None}
            (self.dir / 'stages' / 'gate.done').write_text(json.dumps({'time': time.time(), **info}) + '\n')
            self.note(stage='gate', done=True, **info)
            return
        if v['verdict'] == 'failure' and not getattr(self.a, 'resume_after_hunt', False):
            self.halt(v, collect_reference=True)
        if v['adopt']:
            info = {'start': self.a.start, 'start_sha': self.start_sha, 'prefer_unseen': list(self.a.prefer_unseen),
                    'collection': 'adopted'}
        else:
            start = v['next_start']
            info = {'start': start, 'start_sha': self.previous_start_sha(start),
                    'prefer_unseen': list(self.a.fallback_prefer_unseen),
                    'collection': rejected_collection_mode(v['verdict'], self.a.rejected_collection)}
        info['previous_verdict'] = {k: v[k] for k in ('verdict', 'adopt', 'vs_start')}
        (self.dir / 'stages' / 'gate.done').write_text(json.dumps({'time': time.time(), **info}) + '\n')
        self.note(stage='gate', done=True, **info)
        if info['collection'] == 'recollect':
            (self.dir / 'stages' / 'collect.done').unlink(missing_ok=True)
            self.collect_stage()
            if not self.done('collect'):
                raise SystemExit('recollection did not finish: rerun the cycle to resume it')

    def previous_start_sha(self, start):
        for line in (Path(self.a.speculative_after) / 'drive.log').open():
            record = json.loads(line)
            if record.get('stage') == 'cycle' and record.get('start') == start and record.get('start_sha'):
                return record['start_sha']
        return BASE_SHA if start == BASE_CKPT else \
            subprocess.check_output(['sha256sum', start], text=True).split()[0]

    # the fill window: a dedicated actor of the trained policy while the cycle's CPU stages run
    def fill_window_open(self):
        if (not self.a.fill or self.a.no_fill_window or self.done('sft') or self.window_closed.is_set()
                or self.window_started):
            return
        self.window_started = True
        threading.Thread(target=self._fill_window, daemon=True).start()

    def fill_actor_command(self, checkpoint, sha):
        """grpo's actor (``grpo.actor_command``: the held engine's serve args, SGLang overlap on) from the pin's
        ``serve_native_checkpoint.py`` on its own port, for filled reasoning alone."""
        import grpo
        g = grpo.build_parser().parse_args(['train', '--work', str(self.dir / 'fill-actor'), '--checkpoint', checkpoint,
                                            '--checkpoint-sha256', sha, '--actor-port', str(self.a.fill_actor_port),
                                            '--no-actor-release', '--actor-overlap', 'on',
                                            '--actor-arg=--sglang-kv-gb', f'--actor-arg={self.a.kv_gb}',
                                            '--actor-arg=--sglang-max-running',
                                            f'--actor-arg={self.a.fill_window_inflight}'])
        g.held_manifest = grpo.default_held()
        command, url = grpo.actor_command(g, self.dir / 'fill-actor', sha)
        command[:2] = [PY, str(self.pin / 'scripts' / 'neuralese_data' / 'serve_native_checkpoint.py')]
        return command, url

    def _fill_window(self):
        start, sha, _ = self.train_policy()
        actor_unit, fill_unit = f'{self.a.name}-fillactor', f'{self.a.name}-fillw'
        try:
            command, url = self.fill_actor_command(start, sha)
            self.ledger(actor_unit, self.a.kv_gb + 24, 'medium', command, self.dir / 'fill-actor.log', detach=True)
            info = wait_actor(url, sha, None, None, timeout=3600)
        except Exception as error:
            self.note(stage='fill window', failed=repr(error)[:300])
            return
        if self.window_closed.is_set():
            return
        actor = {'url': url, 'sglang': url.rsplit(':', 1)[0] + ':' + str(self.a.fill_actor_port + 1000),
                 'export': str(self.dir / 'fill-actor' / 'actor' / 'sglang-hf'), 'checkpoint_sha256': sha}
        self.work_tokenizer()
        self.ledger(fill_unit, 8, 'medium', self.fill_command(actor, window=True), self.dir / 'fill-window.log',
                    detach=True, repo=self.fill_repo())
        self.note(stage='fill window', started=[actor_unit, fill_unit], policy=sha,
                  binding=(info.get('rl_policy_binding') or {}).get('checkpoint_sha256'))

    # Pop rollouts: between its held evals Pop serves the collected policy and draws fills for this cycle's build
    POP_FILL_PORT = 19631          # DGX end of the tunnel to Pop's SGLang (pop_rl_actor RL_PORT + 1000 on Pop)

    def pop_fill_open(self):
        if not self.a.fill or getattr(self.a, 'no_pop_fill', False) or self.done('data'):
            return
        threading.Thread(target=self._pop_fill, daemon=True).start()

    def _pop_fill(self):
        unit, tunnel = f'{self.a.name}-popfill', f'{self.a.name}-popfill-tunnel'
        if self.active(unit):
            return
        while self.active(f'{self.a.name}-popbase') or not (self.collect_dir / 'actor.json').exists():
            if self.done('data'):
                return
            time.sleep(60)
        checkpoint, sha = self.collect_policy()
        import pop_rl_actor
        pop_rl_actor.stop_pop_fills()       # an earlier cycle's fill (the held pin's actor may predate the stop)
        actor_py = [PY, str(self.held_pin / 'scripts' / 'neuralese_data' / 'pop_rl_actor.py')]
        try:
            subprocess.run([*actor_py, 'serve-checkpoint', '--checkpoint', str(checkpoint), '--policy-id', sha,
                            '--commit', subprocess.check_output(['git', '-C', str(self.held_pin), 'rev-parse', 'HEAD'],
                                                                text=True).strip(),
                            '--step', '0', '--kv-cache-dtype', 'fp8_e4m3', '--max-running', str(self.a.pop_fill_running)],
                           env=self.held_env, check=True, stdout=(self.dir / 'pop-fill-serve.log').open('a'),
                           stderr=subprocess.STDOUT, timeout=3600)
            state = json.loads(subprocess.check_output([*actor_py, 'status'], env=self.held_env, text=True))['state']
        except Exception as error:
            self.note(stage='pop fill', failed=repr(error)[:300])
            return
        if self.done('data'):
            return
        if not self.active(tunnel):
            import pop_rl_actor
            subprocess.run(['systemd-run', '--user', '--unit', tunnel, '--collect', '/usr/bin/ssh', '-N', '-o',
                            'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30', '-L',
                            f'{self.POP_FILL_PORT}:127.0.0.1:{pop_rl_actor.RL_PORT + 1000}', pop_rl_actor.POP],
                           check=True)
            time.sleep(5)
        collected = json.loads((self.collect_dir / 'actor.json').read_text())
        actor = {'url': state['endpoint'], 'sglang': f'http://127.0.0.1:{self.POP_FILL_PORT}',
                 'export': collected['export'], 'checkpoint_sha256': sha}
        self.work_tokenizer()
        self.ledger(unit, FILL_GB, 'low', self.fill_command(actor, pop=True), self.dir / 'pop-fill.log', detach=True,
                    repo=self.fill_repo())
        self.note(stage='pop fill', started=[unit, tunnel], policy=sha, endpoint=state['endpoint'])

    def pop_fill_close(self):
        for unit in (f'{self.a.name}-popfill', f'{self.a.name}-popfill-tunnel'):
            if self.active(unit):
                self.stop_unit(unit)
                self.note(stage='pop fill', stopped=unit)
        # the Pop server stays: the next held serve-checkpoint replaces it

    def fill_window_close(self):
        self.window_closed.set()
        for unit in (f'{self.a.name}-fillw', f'{self.a.name}-fillactor'):
            if self.active(unit):
                self.stop_unit(unit)
                self.note(stage='fill window', stopped=unit)

    def variant_args(self):
        """collect's Neuralese-declaring task variants (grpo ``--neuralese-variants``; DECISIONS 2026-11-22 "Neuralese
        variants in expert iteration"): a selected task with a variant runs as it in ``--neuralese-variant-share`` of
        its groups. ``--no-neuralese-variants`` opts out."""
        if getattr(self.a, 'no_neuralese_variants', False) or not self.a.neuralese_variants:
            return []
        return ['--neuralese-variants', self.a.neuralese_variants, '--neuralese-variant-share',
                self.a.neuralese_variant_share]

    def data_repo(self):
        return Path(getattr(self.a, 'data_repo', None) or self.pin)

    def fill_repo(self):
        return Path(getattr(self.a, 'fill_repo', None) or self.data_repo())

    def require_data_readers(self):
        minimum = getattr(self.a, 'minimum_data_reader_commit', None)
        if not minimum:
            return
        for repo in {self.data_repo(), self.fill_repo()}:
            marker = repo / '.natlang-pin.json'
            recorded = json.loads(marker.read_text())
            actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            if recorded.get('commit') != actual:
                raise ValueError(f'data reader pin identity differs: {repo}')
            subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', minimum, actual], check=True)
            subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD', '--'], check=True)

    def data_py(self, script, *args):
        return [PY, str(self.data_repo() / 'scripts/neuralese_data' / script), *map(str, args),
                *(self.live_data_args() if script == 'sft_sources.py' else [])]

    def live_data_owner(self):
        if not getattr(self.a, 'live_work', None):
            return None
        from natlang_neuralese.common import data_registry
        return data_registry.registry_owner_repo()

    def live_data_ref(self, owner=None):
        owner = owner if owner is not None else self.live_data_owner()
        if owner is None:
            return None
        from natlang_neuralese.common import data_registry
        return data_registry.registry_owner_ref(owner)

    def live_data_args(self):
        owner = self.live_data_owner()
        if owner is None:
            return []
        ref = self.live_data_ref(owner)
        return ['--registry-repo', str(owner), *(['--registry-ref', ref] if ref is not None else [])]

    def fill_build_command(self, *args):
        # A data stage may intentionally use different fill and mix readers.
        repo = self.fill_repo()
        return ['/usr/bin/env', 'NATLANG_REPO=' + str(repo),
                f'PYTHONPATH={repo}/scripts/neuralese_data:{repo}/training/neuralese:{repo}',
                PY, str(repo / 'scripts/neuralese_data/filled_reasoning.py'), 'build', *map(str, args),
                *self.live_data_args()]

    def fill_alongside(self, collect_unit, sha=None):
        """Filled reasoning (filled_reasoning.py generate, the rationalization strand's tool) drawn from the same actor
        while it collects (polite: spare engine slots only); stopped when the collection ends."""
        deadline = time.time() + 7200
        while not (self.collect_dir / 'actor.json').exists() and time.time() < deadline and self.active(collect_unit):
            time.sleep(30)
        if not (self.collect_dir / 'actor.json').exists():
            self.note(stage='fill', skipped='no actor')
            return
        self.work_tokenizer()      # --render-tokenizer: the data stage made it only after the collection
        actor = json.loads((self.collect_dir / 'actor.json').read_text())
        if actor['checkpoint_sha256'] != (sha or self.start_sha):
            raise SystemExit('collect actor serves another policy')
        self.ledger(f'{self.a.name}-fill', FILL_GB, 'medium', self.fill_command(actor), self.dir / 'fill.log', detach=True,
                    repo=self.fill_repo())
        self.note(stage='fill', started=f'{self.a.name}-fill', actor=actor['url'])

    def fill_command(self, actor, window=False, pop=False):
        """filled_reasoning.py generate: the seed WORKs then the grpo-sd-teacher role's (first WORK wins a key),
        teacher episodes of the tasks the student fails first (the select stage's rates: failed, then unmeasured, then
        by rate; the frontier mix trains teacher trajectories there). On the collect actor a fixed share: the actor runs
        ``--fill-inflight`` more requests than the collection has slots and fill keeps that many draws in flight (twice
        that when the engine has spare slots), with twice as many episode workers so the cap is reached (each worker
        draws at most 4 candidates at a time and scores between them). In the fill window the dedicated actor's every
        slot (``--fill-window-inflight`` draws, ``--fill-window-workers`` episodes)."""
        self.require_data_readers()
        from natlang_neuralese.common import data_registry
        live = [str(w) for w in getattr(self.a, 'live_work', None) or ()]
        owner = self.live_data_owner()
        ref = self.live_data_ref(owner)
        works = list(dict.fromkeys([*live, *map(str, self.a.seed_work),
                    *data_registry.role_paths('grpo-sd-teacher',
                      **({'repo': owner, **({'ref': ref} if ref is not None else {})} if owner is not None else {}))]))
        rates = self.dir / 'tasks.rates.json'
        if pop:
            n = self.a.pop_fill_running
            share = ['--max-running', n, '--max-inflight', n, '--min-inflight', n, '--workers', 3 * n // 2]
        elif window:
            share = ['--max-running', self.a.fill_window_inflight, '--max-inflight', self.a.fill_window_inflight,
                     '--min-inflight', self.a.fill_window_inflight, '--workers', self.a.fill_window_workers]
        else:
            share = ['--polite', '--max-running', self.max_running(), '--min-inflight', self.a.fill_inflight,
                     '--max-inflight', 2 * self.a.fill_inflight, '--workers', 2 * self.a.fill_inflight]
        return [PY, str(self.fill_repo() / 'scripts' / 'neuralese_data' / 'filled_reasoning.py'), 'generate',
                *map(str, ['--out', self.a.fill, '--sglang', actor['sglang'], '--front', actor['url'],
                           '--render-tokenizer', self.work / 'tokenizer', '--export', actor['export'], *share,
                           *[x for w in works for x in ('--work', w)],
                           # the live WORKs' episodes first, and their new ones as the ingester adds them
                           *[x for w in live for x in ('--prefer-work', w)],
                           *(('--rescan-minutes', 30) if live else ()),
                           *(('--registry-repo', owner) if owner is not None else ()),
                           # Standing fills resolve publication again at each rescan.
                           *(('--prefer-rates', rates) if rates.exists() else ())])]

    def stop_unit(self, unit):
        subprocess.run([*LEDGER, 'cancel', '--unit', unit])
        for suffix in ('-waiter', ''):
            subprocess.run(['systemctl', '--user', 'stop', f'{unit}{suffix}.service'], stderr=subprocess.DEVNULL)

    def frontier_stage(self):
        _, sha = self.collect_policy()
        _, train_sha, _ = self.train_policy()
        measured = self.dir / ('frontier.json' if sha == train_sha else 'frontier-sibling.json')
        self.ledger(f'{self.a.name}-frontier', 4, STAGE_PRIORITY,
                    self.py('expert_iteration.py', 'frontier', '--out', measured,
                            '--collect', self.collect_dir, '--policy', sha), self.dir / 'frontier.log')
        data = json.loads(measured.read_text())
        if sha != train_sha:      # a kept sibling collection: the trained policy's own frontier (its cycle's) wins
            own = Path(self.a.speculative_after) / 'frontier.json'
            exact = json.loads(own.read_text()) if own.exists() else None
            if not exact or exact.get('policy') != train_sha:
                exact = {**data, 'policy': train_sha, 'collections': [], 'tasks': {}}
            data = merge_frontier(exact, data)
            (self.dir / 'frontier.json').write_text(json.dumps(data, indent=0) + '\n')
        self.mark('frontier', tiers=data['tiers'], policy=data['policy'], sibling=data.get('sibling_policy'))

    def teacher_stage(self):
        """The policy's new failures feed the teacher campaign (teacher_gen gap tiers: student_rare first, from this
        collection's outcomes; every earlier campaign covers its successes)."""
        if self.a.no_teacher_campaign:
            return self.mark('teacher', skipped=True)
        work = self.dir / 'teacher'
        rare = launch_teacher(self, work, self.dir / 'frontier.json', [self.collect_dir / 'groups' / 'g*'],
                              f'{self.a.name}-teacher')
        self.mark('teacher', frontier_tasks=rare, **({'work': str(work)} if rare else {}))

    def work_tokenizer(self):
        self.work.mkdir(parents=True, exist_ok=True)
        if not (self.work / 'tokenizer').exists():
            os.symlink(root('data_nvme') / 'sft-sources-20261118/tokenizer', self.work / 'tokenizer')

    def data_stage(self):
        self.require_data_readers()
        w = self.work
        self.work_tokenizer()
        steps = []
        if self.a.seed_work:
            steps.append(['/usr/bin/python3', str(OPS / 'sft-v5' / 'seed_replays.py'), str(w), *self.a.seed_work])
            for old in map(Path, self.a.seed_work):    # their extractions (no re-extract; keys are global)
                for f in old.glob('episodes*.jsonl.zst'):
                    link = w / f'{f.name[:-len(".jsonl.zst")]}-{old.name}.jsonl.zst'
                    if not link.exists():
                        os.symlink(f.resolve(), link)
        link_live_works(w, getattr(self.a, 'live_work', None) or ())
        name = f'episodes-{self.a.name}'
        if not (w / f'{name}-extract.json').exists():
            steps.append(self.data_py('sft_sources.py', 'extract', '--work', w, '--rl',
                                 # groups are read live from the run dir; the collection writes no zst archive, so
                                 # the archive is the (registered) collect dir itself
                                 f'{self.collect_dir}:{self.collect_dir}', '--out-name', name))
        steps.append(self.data_py('sft_sources.py', 'replay', '--work', w, '--workers', self.a.replay_workers,
                             '--runtime-repo', self.pin))
        if self.a.fill and (Path(self.a.fill) / 'fills.jsonl').exists():
            failed = [x for p in self.failed_policies() for x in ('--exclude-policy', p)]
            steps.append(self.fill_build_command('--out', self.a.fill, '--corpus', self.dir / 'filled', *failed))
        lines = [' '.join(map(_quote, s)) for s in steps]
        if not self.a.no_own_neuralese_graph:
            # a failure ends the stage (set -e): sft-v7's silently fell back to a mix without the student's own consumer
            # calls, the ran-out fix; --no-own-neuralese-graph is the deliberate opt-out
            lines.append(' '.join(map(_quote, self.own_neuralese_graph_command())))
        script = self.dir / 'data.sh'
        if not self.present(f'{self.a.name}-data'):
            tmp = script.with_name(script.name + '.tmp')
            tmp.write_text('set -e\n' + ''.join(line + '\n' for line in lines))
            os.replace(tmp, script)
        # 12 GB: neuralese-graph's reader processes (sft_sources --workers, ~0.15 GB each) beside its ~7 GB parent
        self.ledger(f'{self.a.name}-data', 12, STAGE_PRIORITY, ['/bin/sh', str(script)], self.dir / 'data.log', repo=self.data_repo())
        self.mark('data')

    def failed_policies(self):
        """Candidates held judged a 'failure' (every cycle's verdict in OPS/ei): their fills are not data."""
        out = []
        for path in sorted(OPS.glob('ei/*/verdict.json')):
            v = json.loads(path.read_text())
            sha = path.parent / 'sft' / 'sha256.txt'
            if v.get('verdict') == 'failure' and sha.exists():
                out.append(sha.read_text().strip())
        return out

    def own_neuralese_graph_command(self):
        """The verified own successes that carry Neuralese state (every consumer call that read a native import) as
        whole-graph rows (``sft_sources neuralese-graph``). ``build`` skips these episodes (no text replay can carry
        the state), so without this part a cycle's mix holds no consumer call of the student's own: held-v3 after
        sft-v5 (DECISIONS 2026-11-21 "ran-out") showed the student calling Neuralese consumers twice as often while
        they loop ~7.6 calls each. ``--no-own-neuralese-graph`` opts out."""
        return self.data_py('sft_sources.py', 'neuralese-graph', '--work', self.work, '--out', self.dir / 'neuralese-graph-own',
                       '--exclude-run', 'grpo-v2/', '--exclude-run', 'grpo-v4/run1',
                       # this cycle's collection ran on the cycle's pin: its surfaces are current
                       '--current-surface-run', '^rl:' + re.escape('/'.join(self.collect_dir.parts[-2:])) + '$')

    def build_command(self):
        """The mix build (``sft_sources build``) and the checkout it runs from: ``--data-repo`` (data preparation,
        outside the held scoring series), else the pin."""
        self.require_data_readers()
        aux = list(self.a.aux)
        if (self.dir / 'filled' / 'chains.pkl').exists():
            aux.append(f'filled={self.dir / "filled" / "chains.pkl"},share=all,task_tokens=40000,replaces=teacher')
        own_graph = self.dir / 'neuralese-graph-own' / 'chains.pkl'
        if not self.a.no_own_neuralese_graph and own_graph.exists():
            aux.append(f'own-neuralese-graph={own_graph},share={self.a.own_neuralese_graph_share},task_tokens=40000')
        repo = self.data_repo()
        command = mix_build_command(repo, work=self.work, out=self.mix, frontier=self.dir / 'frontier.json', aux=aux,
                                    prefer_unseen=self.train_policy()[2], teacher_budget=self.a.teacher_budget,
                                    own_share=self.a.own_share,
                                    live_registry=live_holdout_registry(getattr(self.a, 'live_work', None) or ()),
                                    registry_repo=self.live_data_owner(), registry_ref=self.live_data_ref())
        return repo, list(map(str, command))

    def build_stage(self):
        repo, command = self.build_command()
        self.ledger(f'{self.a.name}-build', 24, STAGE_PRIORITY, command, self.dir / 'build.log', repo=repo)
        sha = subprocess.check_output(['sha256sum', str(self.mix / 'chains.pkl')], text=True).split()[0]
        (self.mix / 'chains.sha256').write_text(sha + '\n')
        build = json.loads((self.mix / 'build.json').read_text())
        self.mark('build', chains_sha256=sha, episodes=build['episodes'], tasks=build['tasks'],
                  forward_tokens=build['forward_tokens'], parts=build['mix']['parts'])

    def sft_env(self, run, unit, start, start_sha, activation_gb, extra=''):
        """launch-sft.sh's environment for this cycle's SFT (one recipe for the DGX run and the Colab one, whose
        preflight runs launch-sft.sh with PREFLIGHT_ONLY=1 and whose learner runs the recorded exp0_sft arguments)."""
        sha = (self.mix / 'chains.sha256').read_text().split()[0]
        env = {**os.environ, 'CHAINS': str(self.mix / 'chains.pkl'), 'CHAINS_SHA256': sha, 'CKPT': start,
               'SHA': start_sha, 'WORK': str(run), 'UNIT': unit, 'LR': self.a.lr, 'WARMUP': '8',
               'BLT': '32000', 'ACT': str(activation_gb), 'BUDGET': str(self.a.sft_budget)}
        vals = [v for v in ('val-own.pkl', 'val-teacher.pkl') if (self.mix / v).exists()]
        if vals:
            env['VAL'] = str(self.mix / vals[0])
        env['EXTRA'] = ' '.join(p for p in [*(f'--val-chains {self.mix / v}' for v in vals[1:]),
                                            '--val-every 20' if vals else '', extra, self.a.sft_extra] if p)
        return env

    def colab_sft(self, run, unit, start, start_sha, host, profile):
        """The SFT on a Colab G4 session: the DGX preflight (launch-sft.sh PREFLIGHT_ONLY=1: input identity, held
        exclusion, the exp0_sft arguments in WORK/launch.json), then ``pop_rl_actor run-sft`` runs exactly those
        arguments there and pulls the verified checkpoint. Returns True when WORK/checkpoint.pt arrived."""
        env = {**self.sft_env(run, unit, start, start_sha, profile.activation_gib), 'PREFLIGHT_ONLY': '1'}
        subprocess.run(['/bin/sh', str(OPS / 'grpo-v4' / 'launch-sft.sh'), str(self.pin)], env=env, check=True,
                       stdout=(self.dir / 'sft-launch.log').open('a'), stderr=subprocess.STDOUT)
        cmd = [PY, str(self.pin / 'scripts' / 'neuralese_data' / 'pop_rl_actor.py'), 'run-sft', '--host', host,
               '--run-name', self.a.name, '--work', str(run), '--launch-json', str(run / 'launch.json'),
               '--commit', self.a.commit, '--state-sync-minutes', str(profile.state_sync_minutes)]
        self.session(host, profile, True)
        try:
            with (self.dir / 'sft-colab.log').open('a') as handle:
                rc = subprocess.run(cmd, env=self.env, stdout=handle, stderr=subprocess.STDOUT).returncode
        finally:
            self.session(host, profile, False)
        return rc == 0 and (run / 'checkpoint.pt').exists()

    def sft_stage(self):
        run = self.dir / 'sft'
        unit = f'{self.a.name}-sft'
        start, start_sha, _ = self.train_policy()
        started = time.time()
        profile, hosts = self.choose_profile(self.sft_profile, 'sft')
        canonical_tokens = 0
        if (self.mix / 'build.json').exists():
            try:
                canonical_tokens = int(json.loads((self.mix / 'build.json').read_text()).get('forward_tokens') or 0)
            except Exception:
                pass
        if profile.name == 'colab' and not (run / 'checkpoint.pt').exists() and not self.active(unit):
            if self.colab_sft(run, unit, start, start_sha, hosts[0], profile):
                sha = subprocess.check_output(['sha256sum', str(run / 'checkpoint.pt')], text=True).split()[0]
                (run / 'sha256.txt').write_text(sha + '\n')
                self.record_cost('sft', profile, time.time() - started, canonical_chain_tokens=canonical_tokens,
                                 note=f'host={hosts[0]}', session_host=hosts[0])
                self.mark('sft', profile='colab', host=hosts[0], checkpoint=str(run / 'checkpoint.pt'), sha256=sha)
                return
            self.note(stage='sft', profile_fallback='Colab G4 SFT failed; resuming on dgx from the synced learner state')
            started = time.time()
        if not (run / 'checkpoint.pt').exists() and not self.active(unit):
            resume_flag = '--resume' if (run / 'learner-state' / 'learner_manifest.json').exists() else ''
            subprocess.run(['/bin/sh', str(OPS / 'grpo-v4' / 'launch-sft.sh'), str(self.pin)],
                           env=self.sft_env(run, unit, start, start_sha, self.a.sft_activation_gb, resume_flag),
                           check=True, stdout=(self.dir / 'sft-launch.log').open('a'), stderr=subprocess.STDOUT)
        self.wait(unit, poll=120)
        if not (run / 'checkpoint.pt').exists():
            raise SystemExit(f'no SFT checkpoint: {run}/train.log')
        sha = subprocess.check_output(['sha256sum', str(run / 'checkpoint.pt')], text=True).split()[0]
        (run / 'sha256.txt').write_text(sha + '\n')
        self.record_cost('sft', 'dgx', time.time() - started, canonical_chain_tokens=canonical_tokens)
        self.mark('sft', profile='dgx', checkpoint=str(run / 'checkpoint.pt'), sha256=sha)

    def held_stage(self):
        # after the next cycle's launch (nothing trains behind this): a gate that adopted before the previous verdict
        # needs it now, for the schedule, the reference and the rollback target
        while self.speculative and (self.gate() or {}).get('verdict_pending') and not self.previous_verdict():
            time.sleep(300)
        late = self.previous_verdict() if self.speculative and (self.gate() or {}).get('verdict_pending') else None
        if late and late['verdict'] == 'failure' and not getattr(self.a, 'resume_after_hunt', False):
            self.halt(late, stop_next=True)        # this cycle trained from the rejected candidate
        if not self.measured():
            return self.mark('held', skipped=f'--held-every {self.a.held_every}')
        while self.active(self.base_unit()):     # one held measurement at a time (Pop serves one policy; the DGX one
            time.sleep(300)                       # held actor beside the collection)
        if self.held_host == 'pop':
            deadline = time.time() + 3600 * self.a.pop_wait_hours
            while not pop_reachable():
                if time.time() > deadline:   # training goes on; the next cycle is measured (DECISIONS "Pop unreachable")
                    return self.mark('held', deferred='pop unreachable')
                time.sleep(300)
        started = time.time()
        candidate = self.measure_held('candidate', self.dir / 'sft' / 'checkpoint.pt')
        if not json.loads((candidate / 'result.json').read_text()).get('complete'):
            raise SystemExit('candidate held measurement incomplete: rerun the cycle to resume it')
        reference = self.reference()['checkpoint']       # the base unit may have failed (Pop down at the time)
        if reference != BASE_CKPT and not self.held_result_ok(self.held / 'start'):
            self.measure_held('start', reference)        # the reference on this series when no result exists
        if self.held_host == 'pop':
            self.record_cost('held', 'pop', time.time() - started)
        self.mark('held', profile=self.held_host, samples=self.a.held_samples)

    def held_deferred(self):
        path = self.dir / 'stages' / 'held.done'
        return path.exists() and bool(json.loads(path.read_text()).get('deferred'))

    def compare(self, a, b, out):
        done = subprocess.run([PY, '-m', 'natlang_neuralese.rl.held_eval', 'compare', str(a / 'result.json'),
                               str(b / 'result.json'), '--out', str(out)], env=self.held_env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if done.returncode:
            err = (done.stderr or '').strip()
            if 'are not paired' in err or 'are different held' in err:
                raise HeldPairRefused(err.splitlines()[-1][:600])
            raise subprocess.CalledProcessError(done.returncode, done.args, stderr=err)
        return json.loads(out.read_text())

    def verdict_stage(self):
        """The declared criteria (DECISIONS): vs the start policy measured on this pin, and vs the base."""
        train_start = self.train_policy()[0]
        candidate_ckpt = str(self.dir / 'sft' / 'checkpoint.pt')
        if not self.measured() or self.held_deferred():
            result = {'verdict': 'unmeasured', 'adopt': True, 'vs_start': {}, 'next_start': candidate_ckpt,
                      'reference': self.reference(), **({'deferred': 'pop unreachable'} if self.held_deferred() else {})}
            result['since_held'], result['held_every'] = self.held_schedule()
            (self.dir / 'verdict.json').write_text(json.dumps(result, indent=1) + '\n')
            return self.mark('verdict', **result)
        reference = self.reference()
        if reference['checkpoint'] != str(train_start):       # held/start measured the reference (pop-held)
            train_start = reference['checkpoint']
        base, start, candidate = self.held / 'base', self.held / ('base' if train_start == BASE_CKPT else 'start'), \
            self.held / 'candidate'
        vs_start = self.compare(start, candidate, self.dir / 'compare-vs-start.json')
        base_series = 'same'
        if not (base / 'result.json').exists():     # not measured on this pin (--held-base): the latest other series
            base, base_series = latest_base(OPS.glob('ei/*/held/base/result.json'), self.a.held_samples,
                                            self.held_host), 'other'
        vs_base = self.compare(base, candidate, self.dir / 'compare-vs-base.json') if base else None
        d, p = vs_start['primary']['delta'], vs_start['primary']['permutation_p']
        ran_out = vs_start.get('ran_out_rate_delta') or 0
        verdict, adopt = judge(vs_start, self.a.success_pts, self.a.fail_pts)
        result = {'verdict': verdict, 'adopt': adopt, 'vs_start': {'delta': d, 'p': p, 'ran_out_delta': ran_out},
                  'vs_base': vs_base and {'delta': vs_base['primary']['delta'], 'p': vs_base['primary']['permutation_p'],
                                          'series': base_series, 'from': str(base)},
                  'next_start': candidate_ckpt if adopt else str(train_start),
                  'reference': {'checkpoint': candidate_ckpt} if adopt else {'checkpoint': str(train_start)}}
        result['since_held'], result['held_every'] = self.held_schedule()
        if verdict == 'failure':      # a regression is a bug until shown otherwise: the supervisor opens the hunt
            result['debug'] = f'regression vs {reference["checkpoint"]}: hunt it (AGENTS.md "degradation means bug")'
        (self.dir / 'verdict.json').write_text(json.dumps(result, indent=1) + '\n')
        if verdict == 'success' and (vs_base is None or vs_base['primary']['delta'] > 0):
            promote_policy(self, self.dir / 'sft' / 'checkpoint.pt', (self.dir / 'sft' / 'sha256.txt').read_text().strip(),
                           label=self.a.name, why=f'{self.a.name}: {json.dumps(result)}', unit=f'{self.a.name}-promote')
        self.mark('verdict', **result)

    def halt(self, verdict, *, collect_reference=False, stop_next=False):
        """A held failure (below -``--fail-pts`` with p < 0.05) ends the experiment at that eval (AGENTS.md): no
        further SFT and no next cycle; ``bug-hunt.json`` records it. ``collect_reference``: the machines keep working:
        a collection of the reference (the policy the lineage rolls back to) starts detached, data for the restart.
        ``stop_next``: a next cycle launched before the late verdict (it descends from the rejected candidate) stops.
        Rerun with ``--resume-after-hunt`` when the hunt is closed (the rejection then proceeds as before)."""
        reference = (verdict.get('reference') or {}).get('checkpoint') or verdict['next_start']
        record = {'verdict': verdict, 'reference': reference, 'cycle': self.a.name, 'at': time.time(),
                  'debug': verdict.get('debug') or 'regression: hunt it (AGENTS.md "degradation means bug")'}
        (self.dir / 'bug-hunt.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
        if collect_reference and not self.present(f'{self.a.name}-recollect'):
            sha = self.previous_start_sha(reference)
            self.set_collect_aside(sha)
            unit = f'{self.a.name}-recollect'
            budget = self.a.kv_gb + 24 + round(self.a.slots * COLLECTOR_SLOT_GB)
            self.ledger(unit, budget, COLLECTION_PRIORITY, self.collect_command(unit, reference, sha, self.a.slots),
                        self.dir / 'collect.log', detach=True)
            record['collecting'] = unit
        if stop_next and self.done('next'):
            nxt = json.loads((self.dir / 'stages' / 'next.done').read_text())
            for unit in [nxt.get('unit'), *(f'{nxt.get("name")}-{u}' for u in ('collect', 'recollect', 'fill', 'sft'))]:
                if unit and self.present(unit):
                    self.stop_unit(unit)
            record['stopped_next'] = nxt.get('name')
        (self.dir / 'stages' / 'halted.done').write_text(json.dumps(record, default=str) + '\n')
        self.note(stage='halted', **{k: v for k, v in record.items() if k != 'verdict'})
        raise SystemExit(f'{self.a.name}: held failure of the previous candidate ({json.dumps(verdict.get("vs_start"))}): '
                         f'the experiment ends at this eval (AGENTS.md); bug hunt: {self.dir}/bug-hunt.json; rerun with '
                         '--resume-after-hunt when it is closed')

    def next_stage(self):
        """Launch the next cycle from this cycle's candidate (its own driver unit, as ops/ei/launch.sh does), pinned
        to ``--next-commit`` (default origin/main now: improvements enter at the next pin)."""
        if self.a.no_pipeline:
            return self.mark('next', skipped=True)
        v = self.previous_verdict() if self.speculative else None
        if v and v['verdict'] == 'failure' and not getattr(self.a, 'resume_after_hunt', False):
            self.halt(v)
        name = self.a.next_name or next_name(self.a.name)
        if not name:
            return self.mark('next', skipped=f'no next name for {self.a.name}')
        subprocess.run(['git', '-C', str(REPO), 'fetch', '-q', 'origin'])
        commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', self.a.next_commit], text=True).strip()
        start = self.dir / 'sft' / 'checkpoint.pt'
        argv = next_cycle_argv(sys.argv[sys.argv.index('cycle') + 1:], directory=self.dir.parent / name, name=name,
                               start=start, start_sha=(self.dir / 'sft' / 'sha256.txt').read_text().strip(),
                               commit=commit, previous=self.dir, prefer_unseen=lineage_builds(self.mix, self.train_policy()[2]),
                               fallback_prefer_unseen=self.train_policy()[2], extra_seed_work=self.a.next_seed_work,
                               since_held=0 if self.measured() else self.held_schedule()[0] + 1,
                               held_pin=self.held_pin, held_series_host=self.held_host)
        (self.dir.parent / name).mkdir(parents=True, exist_ok=True)
        unit = f'{name}-drive-{int(time.time())}'
        subprocess.run([*LEDGER, 'run', '--unit', unit, '--budget-gb', '2', '--priority', 'medium', '--wait', '36000',
                        '--detach', '--workdir', str(self.dir.parent / name), '--', '/usr/bin/env',
                        f'PYTHONPATH={REPO}/training/neuralese:{REPO}', 'HF_HUB_OFFLINE=1', f'PATH={os.environ["PATH"]}',
                        PY, str(REPO / 'scripts/neuralese_data/expert_iteration.py'), 'cycle', *argv], check=True)
        self.mark('next', name=name, unit=unit, commit=commit, argv=argv)

    def run(self):
        self.note(stage='cycle', start=self.a.start, start_sha=self.start_sha, commit=self.a.commit,
                  speculative_after=getattr(self.a, 'speculative_after', None))
        stages = [('pin', self.pin_stage), ('popbase', self.popbase_stage), ('select', self.select_stage),
                  ('collect', self.collect_stage), ('gate', self.gate_stage), ('frontier', self.frontier_stage),
                  ('teacher', self.teacher_stage), ('data', self.data_stage), ('build', self.build_stage),
                  ('sft', self.sft_stage), ('next', self.next_stage), ('held', self.held_stage),
                  ('verdict', self.verdict_stage)]
        self.pop_fill_open()                   # waits for Pop (popbase) and the collect actor
        for name, stage in stages:
            if name == 'gate' and (not self.speculative or self.gate()):
                continue
            if self.done(name):
                continue
            if name == 'frontier' or (name in ('teacher', 'data', 'build') and not self.done('sft')):
                self.fill_window_open()        # idempotent: a running window is joined
            if name == 'sft':
                self.fill_window_close()       # the SFT has the GPU to itself
            if name == 'data' and self.a.fill:
                self.stop_unit(f'{self.a.name}-fill')
                if getattr(self.a, 'pop_fill_stop_at_data', False):
                    self.pop_fill_close()   # default: Pop draws on for the next cycle until a Pop serve stops it
            stage()
            if name == 'collect' and not self.done('collect') and not self.speculative:
                raise SystemExit('collection did not finish: rerun the cycle to resume it')


def pop_reachable():
    import pop_rl_actor
    try:
        return subprocess.run(['ssh', '-o', 'ConnectTimeout=20', '-o', 'BatchMode=yes', pop_rl_actor.POP, 'true'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _quote(text):
    import shlex
    return shlex.quote(str(text))


def pop_held(args):
    """Background unit of ``cycle``: the start policy on the cycle's held series (``--held-profile``: Pop, the legacy
    default of this subcommand, or the DGX). The base only when it is the start, with ``--held-base``, or when an
    earlier cycle measured it in this series: the verdict needs the start; vs-base is a reference that the latest base
    of the same host gives (DECISIONS 2026-11-22 "held base"), and re-measuring it on every pin cost 3.3 h."""
    a = argparse.Namespace(dir=args.dir, pin=args.pin, start=args.start, start_sha=BASE_SHA, commit='x' * 40,
                           held_samples=args.held_samples, sft_work='/dev/null', no_held_reuse=args.no_held_reuse,
                           strict_held_reuse=getattr(args, "strict_held_reuse", False),
                           held_profile=args.held_profile, held_slots=args.held_slots,
                           name=args.name or Path(args.dir).name)
    c = Cycle(a)
    c.pin = Path(args.pin)
    start = args.start
    if args.after:        # a speculative cycle: the held series is busy with the previous candidate, whose verdict names the start
        verdict = Path(args.after) / 'verdict.json'
        while not verdict.exists():
            time.sleep(300)
        v = json.loads(verdict.read_text())
        start = (v.get('reference') or {}).get('checkpoint') or v['next_start']    # the last measured policy
    if start != BASE_CKPT:
        c.measure_held('start', start)
    if start == BASE_CKPT or args.held_base:
        c.measure_held('base', BASE_CKPT)
    else:
        c.reuse_held(c.held / 'base', BASE_CKPT)


def dgx_held(args):
    """One DGX held measurement, run inside its ledger unit (``Cycle.dgx_held``): claim held priority (every DGX
    collection yields slots to it), serve the policy with the pinned held engine (``engine.json`` serve_args beside the
    held manifest: SGLang, vendor projection, FP8 KV, BF16 weights) from the held pin, run ``held_eval measure``
    (the engine attested against the pin, the training sampling law and episode budget) and stop the actor."""
    import grpo
    from natlang_neuralese.rl.held_eval import BASE_SNAPSHOT, _free_port, pinned_engine
    out, pin = Path(args.out), Path(args.held_pin)
    manifest = Path(args.manifest or grpo.default_held())
    env = {**os.environ, 'NATLANG_REPO': str(pin), 'PYTHONPATH': f'{pin}/training/neuralese:{pin}',
           'NATLANG_HELD_SERIES_HOST': 'dgx', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    HELD_PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
    claim = HELD_PRIORITY_DIR / f'{args.unit}.json'
    claim.write_text(json.dumps({'pid': os.getpid(), 'unit': args.unit, 'policy': args.policy_id, 'out': str(out),
                                 'slots': args.slots, 'started': time.time()}) + '\n')
    beat = threading.Event()

    def heartbeat():
        while not beat.wait(60):
            os.utime(claim)
    threading.Thread(target=heartbeat, daemon=True).start()
    port = _free_port()
    store = out / 'actor'
    store.mkdir(parents=True, exist_ok=True)
    serve = [PY, str(pin / 'scripts/neuralese_data/serve_native_checkpoint.py'), '--store-dir', str(store),
             '--checkpoint', str(args.checkpoint), '--checkpoint-sha256', args.policy_id, '--base', str(BASE_SNAPSHOT),
             '--port', str(port), '--prompt-rendering', 'full', *pinned_engine(manifest)['serve_args'],
             # sized to fit beside a collection (HELD_KV_GB); later flags win over the pin's 16 GB / 128
             '--sglang-kv-gb', str(args.kv_gb), '--sglang-max-running', str(args.max_running),
             '--hf-export-dir', str(store / 'sglang-hf')]
    url = f'http://127.0.0.1:{port}'
    actor = None
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        with (out / 'actor.log').open('a') as handle:
            actor = subprocess.Popen(serve, cwd=pin, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        wait_actor(url, args.policy_id, actor, out / 'actor.log')
        log({'event': 'held actor up', 'endpoint': url, 'policy': args.policy_id})
        rc = subprocess.run([PY, '-m', 'natlang_neuralese.rl.held_eval', 'measure', '--manifest', str(manifest),
                             '--endpoint', f'{url}={args.slots}', '--policy-id', args.policy_id, '--top-p', '0.95',
                             '--samples', str(args.samples), '--out', str(out), '--label', args.label,
                             '--max-model-requests', '40'], cwd=pin, env=env).returncode
        log({'event': 'held measured', 'returncode': rc, 'out': str(out)})
        return rc
    finally:
        beat.set()
        claim.unlink(missing_ok=True)
        if actor is not None and actor.poll() is None:
            os.killpg(actor.pid, signal.SIGTERM)
            try:
                actor.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(actor.pid, signal.SIGKILL)
        import shutil
        shutil.rmtree(store / 'sglang-hf', ignore_errors=True)    # the export (3.4 GB) is rebuilt from the checkpoint


# ---- held-v4 beside the gate (DECISIONS 2026-11-23 "collection health and held cadence") ------------------------------
HELD_V4_CADENCE = 'every=3,significant,health-sfp'
HELD_V4_TRIGGERS = ('significant', 'health-sfp')
HELD_V4_SAMPLES = 2
HELD_V4_QUEUE_SCHEMA = 'natlang.ei-held-v4-queue/1'


def parse_held_v4_cadence(text):
    """``--held-v4-cadence``: ``off``, or comma-separated ``every=N`` (every Nth published round; 0 never),
    ``significant`` (a round whose held-v3 verdict has p < 0.05, either direction) and ``health-sfp`` (a round whose
    collection health report flags a semantic-FP construct)."""
    out = {'every': 0, 'triggers': set()}
    for part in [p.strip() for p in str(text or '').split(',') if p.strip()]:
        if part == 'off':
            return {'every': 0, 'triggers': set()}
        if part.startswith('every='):
            out['every'] = max(0, int(part.split('=', 1)[1]))
        elif part in HELD_V4_TRIGGERS:
            out['triggers'].add(part)
        else:
            raise ValueError(f'--held-v4-cadence: unknown part {part!r} (every=N, {", ".join(HELD_V4_TRIGGERS)}, off)')
    return out


def held_v4_reasons(cadence, *, round_number=None, verdict=None, health=None):
    """Why a policy is due a held-v4 measurement under ``cadence`` (``parse_held_v4_cadence``): its published round
    number, its held-v3 verdict record (``vs_start.p``) or the health report of its collection."""
    out = []
    every = cadence.get('every') or 0
    if round_number and every and round_number % every == 0:
        out.append(f'every-{every}:round-{round_number}')
    if verdict is not None and 'significant' in cadence['triggers']:
        p = (verdict.get('vs_start') or {}).get('p')
        if p is not None and p < 0.05:
            out.append(f'significant:{verdict.get("verdict")}')
    if health is not None and 'health-sfp' in cadence['triggers'] and health.get('sfp_flagged'):
        out.append('health-sfp')
    return out


class HeldV4Queue:
    """Policies due a held-v4 measurement (``held-v4/queue.json``): one entry per policy sha with every reason it was
    queued for; ``next`` is the newest pending one (the newest policy says most); ``finish`` closes an entry."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        try:
            state = json.loads(self.path.read_text())
            self.entries = state['entries'] if state.get('schema') == HELD_V4_QUEUE_SCHEMA else []
        except (OSError, ValueError, KeyError):
            self.entries = []

    def save(self):
        from natlang_neuralese.common.jsonio import write_json_durable
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json_durable(self.path, {'schema': HELD_V4_QUEUE_SCHEMA, 'entries': self.entries})

    def add(self, policy, reasons):
        if not reasons:
            return None
        with self.lock:
            for entry in self.entries:
                if entry['sha256'] == policy['sha256']:
                    entry['reasons'] = sorted(set(entry['reasons']) | set(reasons))
                    self.save()
                    return entry
            entry = {'sha256': policy['sha256'], 'checkpoint': str(policy['checkpoint']), 'reasons': sorted(reasons),
                     'queued_at': round(time.time(), 1), 'state': 'pending', 'attempts': 0}
            self.entries.append(entry)
            self.save()
            return entry

    def next(self):
        with self.lock:
            pending = [e for e in self.entries if e['state'] == 'pending']
            return dict(pending[-1]) if pending else None

    def finish(self, sha, state, **info):
        with self.lock:
            for entry in self.entries:
                if entry['sha256'] == sha:
                    entry.update(state=state, finished_at=round(time.time(), 1), **info)
            self.save()

    def attempt(self, sha):
        with self.lock:
            for entry in self.entries:
                if entry['sha256'] == sha:
                    entry['attempts'] = int(entry.get('attempts', 0)) + 1
                    self.save()
                    return entry['attempts']
        return 0


def _result_complete(out):
    try:
        return bool(json.loads((Path(out) / 'result.json').read_text()).get('complete'))
    except (OSError, ValueError):
        return False


def pop_held_measure(out, checkpoint, *, pin, samples, manifest, preempt=None, poll=30):
    """One held measurement of ``checkpoint`` on Pop with ``manifest`` (``pop-held.sh``, resumable). ``preempt``
    (a callable) is polled: when it answers true the measurement stops (its ``popheld-<OUT name>-*`` ledger unit and the
    script) and resumes from its episodes next time. Returns 'complete', 'preempted' or 'incomplete'."""
    import pop_rl_actor
    out = Path(out)
    if _result_complete(out):
        return 'complete'
    out.mkdir(parents=True, exist_ok=True)
    if join_pop_held(out, preempt=preempt, poll=poll) == 'preempted':
        return 'preempted'
    if _result_complete(out):
        return 'complete'
    pop_rl_actor.stop_pop_fills()
    mode = ['base', str(out), str(samples)] if str(checkpoint) == BASE_CKPT else \
        ['ckpt', str(out), str(samples), str(checkpoint)]
    env = {**os.environ, 'HELD_MANIFEST': str(manifest)}
    with (out / 'driver.log').open('a') as handle:
        process = subprocess.Popen(['/bin/sh', str(OPS / 'grpo-v4' / 'pop-held.sh'), str(pin), *mode], env=env,
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    while process.poll() is None:
        if preempt is not None and preempt():
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            stop_pop_held_units(out)
            return 'preempted'
        time.sleep(poll)
    return 'complete' if _result_complete(out) else 'incomplete'


def stop_pop_held_units(out):
    import pop_rl_actor
    for unit in pop_rl_actor.pop_held_units(f'popheld-{Path(out).name}-*'):
        subprocess.run(['systemctl', '--user', 'stop', f'{unit}.service'], check=False)


def join_pop_held(out, *, preempt=None, poll=30):
    """Wait for a measure unit of ``out`` that is still running (``popheld-<OUT name>-*``: a driver restart killed the
    ``pop-held.sh`` that started it, not the unit) instead of starting a second measurement on the same episodes.
    Returns 'preempted' when ``preempt`` stopped it, else None."""
    import pop_rl_actor
    while pop_rl_actor.pop_held_units(f'popheld-{Path(out).name}-*'):
        if preempt is not None and preempt():
            stop_pop_held_units(out)
            return 'preempted'
        time.sleep(poll)
    return None


# ---- async round driver & phase switching (async-train §2, §4, §5, §7.4) ---------------------------------------------
ROUND_MANIFEST_SCHEMA = 'natlang.async-round/1'
PHASE_STATE_SCHEMA = 'natlang.async-phase-state/2'


def write_round_manifest(round_dir, *, round_id, base_policy_sha, previous_policy_sha, chains_path, episode_ids,
                         prefer_unseen=(), profile='colab', profile_changes=(), extra=None):
    """Commit one async round's mix in strict blob-first order (async-train §4): ``chains.pkl`` + ``chains.sha256``
    are flushed and verified before ``round.json`` is atomically renamed into place (so a consumer never sees a
    partial round)."""
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.common.jsonio import write_json_durable
    round_dir = Path(round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    chains_path = Path(chains_path)
    if not chains_path.is_file():
        raise FileNotFoundError(f'round {round_id}: chains blob missing at {chains_path}')
    chains_sha = sha256_file_hex(chains_path)
    chains_bytes = chains_path.stat().st_size
    sha_tmp = round_dir / 'chains.sha256.partial'
    sha_tmp.write_text(f'{chains_sha}\n')
    os.replace(sha_tmp, round_dir / 'chains.sha256')
    manifest = {
        'schema': ROUND_MANIFEST_SCHEMA,
        'round_id': str(round_id),
        'base_policy_sha': str(base_policy_sha),
        'previous_policy_sha': previous_policy_sha,
        'chains_path': str(chains_path),
        'chains_sha256': chains_sha,
        'chains_bytes': int(chains_bytes),
        'episode_ids': sorted(set(str(e) for e in episode_ids)),
        'prefer_unseen': [str(p) for p in prefer_unseen],
        'profile': str(profile),
        'profile_changes': list(profile_changes),
        'committed_at': round(time.time(), 1),
    }
    if extra:
        manifest.update(extra)
    write_json_durable(round_dir / 'round.json', manifest)
    return manifest


PREPARE_STEPS = ('frontier', 'data', 'build')


def argv_digest(argv):
    """sha256 of a step's argv (its exit record names the command it ran)."""
    import hashlib
    return hashlib.sha256(json.dumps([str(a) for a in argv]).encode()).hexdigest()


def step_command(argv, exit_path):
    """``argv`` run by /bin/sh, which then records its exit status, the argv digest and the time in ``exit_path``
    (atomically) and exits with that status: a driver that died before writing the step's ``.done`` marker still
    learns that the step finished."""
    q = _quote
    record = (f'printf \'{{"rc": %s, "argv_sha256": "{argv_digest(argv)}", "at": %s}}\\n\' "$rc" "$(date +%s)" '
              f'> {q(str(exit_path))}.tmp && mv -f {q(str(exit_path))}.tmp {q(str(exit_path))}')
    return ['/bin/sh', '-c', f'"$@"; rc=$?; {record}; exit $rc', 'step', *map(str, argv)]


def unit_last_run_succeeded(unit):
    """Whether ``unit``'s last run (from its last "Started" in the user journal) ended without a failure: systemd
    logged its resource use ("Consumed") and no failure, kill or stop. False when the journal has no run."""
    try:
        out = subprocess.run(['journalctl', '--user', '-u', f'{unit}.service', '-o', 'cat', '--no-pager', '-n', '400'],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    lines = out.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('Started ')]
    if not starts:
        return False
    tail = lines[starts[-1] + 1:]
    bad = ('Failed with result', 'Main process exited', 'Stopping ', 'Stopped ', 'code=killed', 'code=dumped')
    return any('Consumed' in line for line in tail) and not any(b in line for line in tail for b in bad)


TEACHER_RELAUNCH_MINUTES = 60   # an idle campaign of the same frontier resumes at most this often
SFT_HANDBACK_MAX = 2          # moves of one round from the DGX back to a session (no ping-pong between two hosts)


def dgx_sft_progress(train_jsonl, window=20):
    """{step, of, s_per_step} of a running SFT from its ``train.jsonl``: the last step and the median wall seconds of
    the last ``window`` steps; None before 5 steps are logged."""
    path = Path(train_jsonl)
    if not path.exists():
        return None
    records = []
    for line in path.read_text(errors='replace').splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get('step_seconds') and record.get('of'):
            records.append(record)
    if len(records) < 5:
        return None
    last = records[-window:]
    return {'step': int(records[-1]['step']), 'of': int(records[-1]['of']),
            's_per_step': round(statistics.median(float(r['step_seconds']) for r in last), 2), 'window': len(last)}


def sft_handback_decision(*, steps_total, dgx_step, resume_step, dgx_s_per_step, speedup, move_seconds, margin,
                          min_gain_seconds):
    """Whether a round training on the DGX moves to a session: the DGX's remaining time (its steps left at its measured
    s/step) against the move (``move_seconds``: transfers and the learner's setup) plus the session's steps from the
    pushed state (``resume_step``: the DGX's steps since its last save are redone there) at ``dgx_s_per_step /
    speedup``. It moves only with a clear margin: the session must finish within ``1 - margin`` of the DGX's time and
    at least ``min_gain_seconds`` sooner."""
    dgx_left = max(0, steps_total - dgx_step) * dgx_s_per_step
    colab_left = (move_seconds + max(0, steps_total - resume_step) * dgx_s_per_step / speedup) if speedup else None
    move = bool(colab_left is not None and colab_left <= (1.0 - margin) * dgx_left
                and dgx_left - colab_left >= min_gain_seconds)
    return {'move': move, 'dgx_left_s': round(dgx_left), 'colab_left_s': None if colab_left is None else round(colab_left),
            'gain_s': None if colab_left is None else round(dgx_left - colab_left), 'steps_total': steps_total,
            'dgx_step': dgx_step, 'resume_step': resume_step, 'dgx_s_per_step': dgx_s_per_step, 'speedup': speedup,
            'move_s': round(move_seconds), 'margin': margin, 'min_gain_s': min_gain_seconds}


def record_profile_change(round_dir, *, from_profile, to_profile, reason, at=None):
    """Record a mid-round host profile transition (e.g. Colab G4 -> DGX takeover after 30 min unreachable;
    async-train §4) into ``round.json`` atomically."""
    from natlang_neuralese.common.jsonio import write_json_durable
    path = Path(round_dir) / 'round.json'
    manifest = json.loads(path.read_text())
    changes = list(manifest.get('profile_changes') or [])
    changes.append({
        'from_profile': str(from_profile),
        'to_profile': str(to_profile),
        'reason': str(reason),
        'at': round(time.time() if at is None else float(at), 1),
    })
    manifest['profile_changes'] = changes
    manifest['profile'] = str(to_profile)
    write_json_durable(path, manifest)
    return manifest


def update_round_manifest(round_dir, **fields):
    """Add ``fields`` to a committed ``round.json`` atomically (what the round trained from, set when it starts)."""
    from natlang_neuralese.common.jsonio import write_json_durable
    path = Path(round_dir) / 'round.json'
    write_json_durable(path, {**json.loads(path.read_text()), **fields})


class PhaseController:
    """The async loop's policy lineage, held schedule and SFT <-> GRPO phase (async-train §2.6, §7.4), persisted in
    ``phase-state.json``. ``lineage`` is the chain of published policies the current one descends from (its last entry
    is the current policy, the one before it the previous); ``reference_policy`` is the last held-adopted policy.

    - A published round appends to the lineage and counts toward the held interval (``since_held``, counted in
      rounds); ``held_due`` when ``since_held >= held_every``: with ``held_every`` N every Nth round is measured, the
      cycle path's rule (``Cycle.measured``). Starting a measurement resets ``since_held``.
    - ``success`` adopts the candidate as the reference and doubles ``held_every`` (up to ``held_every_max``);
      ``neutral`` returns it to the base interval (an adopted neutral candidate becomes the reference); ``failure``
      sets it to 1 (the next round is measured) and records a bug-hunt note.
    - A candidate held does not adopt (failure, or neutral without gain) rolls the lineage back to the reference: the
      candidate and everything trained from it leave the lineage, so the previous policy is the reference's own
      predecessor, never the rejected one.
    - Two consecutive neutral measured rounds switch the phase (sft -> grpo, grpo -> sft), restarting from the
      reference (the newest held-adopted policy)."""

    def __init__(self, state_path, *, state=None):
        self.path = Path(state_path)
        if state is not None:
            self.state = dict(state)
        elif self.path.is_file():
            self.state = json.loads(self.path.read_text())
        else:
            raise ValueError(f'PhaseController requires state or existing {self.path}')
        if self.state.get('schema') != PHASE_STATE_SCHEMA:
            raise ValueError(f'{self.path}: schema {self.state.get("schema")!r}, not {PHASE_STATE_SCHEMA}')

    @classmethod
    def initialize(cls, state_path, *, start_checkpoint, start_sha, held_every=2, held_every_max=8,
                   initial_phase='sft', prefer_unseen=(), reference_checkpoint=None, reference_sha=None):
        path = Path(state_path)
        if path.is_file():
            return cls(path)
        start = {'checkpoint': str(start_checkpoint), 'sha256': str(start_sha), 'version': 0, 'round_id': None}
        state = {
            'schema': PHASE_STATE_SCHEMA, 'phase': initial_phase, 'round_counter': 0, 'active_round': None,
            'consecutive_neutral': 0, 'since_held': 0,
            'base_held_every': max(1, int(held_every)), 'held_every': max(1, int(held_every)),
            'held_every_max': max(1, int(held_every_max)),
            # an unmeasured start (speculative cycles) is judged against the last held-adopted policy, not itself
            'reference_policy': ({'checkpoint': str(reference_checkpoint), 'sha256': str(reference_sha)}
                                 if reference_checkpoint else {'checkpoint': str(start_checkpoint), 'sha256': str(start_sha)}),
            'start_policy': {'checkpoint': str(start_checkpoint), 'sha256': str(start_sha)},
            'lineage': [start], 'version_counter': 0, 'trained_rounds': [],
            'prefer_unseen': [str(p) for p in prefer_unseen], 'measuring': None, 'applied_verdicts': [],
            'history': [],
        }
        ctrl = cls(path, state=state)
        ctrl.save()
        return ctrl

    def save(self):
        from natlang_neuralese.common.jsonio import write_json_durable
        write_json_durable(self.path, self.state)

    def reconcile_arguments(self, *, start_checkpoint, start_sha, reference_checkpoint=None, reference_sha=None,
                            held_every=None, held_every_max=None, prefer_unseen=()):
        """A restarted driver's arguments against the state they seeded (2026-11-24: the stored reference sft-v7
        silently outranked --reference sft-v9, and every held verdict was queued against the wrong policy).
        - ``--start`` names the experiment's start: one that differs from the state's start refuses (SystemExit:
          another start is another experiment, a fresh ``--root``).
        - ``--reference``: the state's reference wins only when a held verdict adopted it (the loop moved it on from
          the argument's policy); otherwise the argument is adopted (an open measurement is re-pointed at it).
        - ``--held-every`` / ``--held-every-max`` set the base interval and its cap (the current interval is clamped);
          ``--prefer-unseen`` entries missing from the state are added.
        Every change is appended to ``history`` (``action`` 'arguments'); returns the changes."""
        state, changes = self.state, {}
        recorded = state.get('start_policy')
        if recorded is None and state['lineage'] and int(state['lineage'][0].get('version', -1)) == 0:
            recorded = {k: state['lineage'][0][k] for k in ('checkpoint', 'sha256')}
        if recorded and recorded['sha256'] != str(start_sha):
            raise SystemExit(f'{self.path}: this loop started at {recorded["sha256"][:12]} ({recorded["checkpoint"]}); '
                             f'--start names {str(start_sha)[:12]} ({start_checkpoint}): another start is another '
                             'experiment (use a fresh --root, or pass the recorded --start)')
        if recorded and 'start_policy' not in state:
            state['start_policy'] = dict(recorded)
            changes['start_policy'] = {'recorded': recorded}
        if reference_checkpoint and reference_sha:
            stored = state['reference_policy']
            if stored['sha256'] != str(reference_sha):
                adopted = any(e.get('action') == 'adopt' and e.get('sha256') == stored['sha256']
                              for e in state.get('history') or ())
                if adopted:
                    changes['reference_kept'] = {'stored': stored, 'argument': str(reference_sha),
                                                 'why': 'a held verdict adopted the stored reference'}
                else:
                    new = {'checkpoint': str(reference_checkpoint), 'sha256': str(reference_sha)}
                    state['reference_policy'] = new
                    changes['reference'] = {'from': stored, 'to': new}
                    job = state.get('measuring')
                    if job and job.get('reference', {}).get('sha256') != new['sha256']:
                        job['reference'] = dict(new)
                        changes['measuring_reference'] = new['sha256']
        if held_every is not None and int(held_every) != int(state['base_held_every']):
            changes['base_held_every'] = {'from': state['base_held_every'], 'to': max(1, int(held_every))}
            if int(state['held_every']) == int(state['base_held_every']):
                state['held_every'] = max(1, int(held_every))
            state['base_held_every'] = max(1, int(held_every))
        if held_every_max is not None and int(held_every_max) != int(state['held_every_max']):
            changes['held_every_max'] = {'from': state['held_every_max'], 'to': max(1, int(held_every_max))}
            state['held_every_max'] = max(1, int(held_every_max))
        if int(state['held_every']) > int(state['held_every_max']):
            state['held_every'] = int(state['held_every_max'])
        missing = [str(p) for p in prefer_unseen if str(p) not in state['prefer_unseen']]
        if missing:
            state['prefer_unseen'].extend(missing)
            changes['prefer_unseen_added'] = missing
        if changes:
            state['history'].append({'at': round(time.time(), 1), 'action': 'arguments', 'changes': changes})
            self.save()
        return changes

    @property
    def phase(self):
        return self.state['phase']

    @property
    def current(self):
        return dict(self.state['lineage'][-1])

    @property
    def previous(self):
        return dict(self.state['lineage'][-2]) if len(self.state['lineage']) > 1 else None

    def in_lineage(self, sha):
        return any(p['sha256'] == sha for p in self.state['lineage'])

    def begin_round(self, base_sha):
        """The active round (resumed after a restart while its base is still the lineage head), else a new one."""
        active = self.state.get('active_round')
        if active and active['base_sha256'] == self.current['sha256'] and active['phase'] == self.phase:
            return active['round_id']
        self.state['round_counter'] = int(self.state['round_counter']) + 1
        rid = f'r{self.state["round_counter"]:04d}'
        self.state['active_round'] = {'round_id': rid, 'base_sha256': base_sha, 'phase': self.phase}
        self.save()
        return rid

    def discard_round(self, round_id, reason):
        self.state['active_round'] = None
        self.state['history'].append({'at': round(time.time(), 1), 'round_id': round_id, 'action': 'discard',
                                      'reason': reason})
        self.save()

    def record_round_published(self, round_id, checkpoint, sha256, *, base_sha256, mix_dir=None):
        """Append the round's published policy to the lineage (idempotent per round id). Refused when the round's base
        is no longer the lineage head (a rollback discarded it while it trained)."""
        if str(round_id) in self.state['trained_rounds']:
            return self.current
        if base_sha256 != self.current['sha256']:
            raise ValueError(f'round {round_id} trained from {base_sha256[:12]}, the lineage head is '
                             f'{self.current["sha256"][:12]}')
        self.state['version_counter'] = int(self.state['version_counter']) + 1
        self.state['lineage'].append({'checkpoint': str(checkpoint), 'sha256': str(sha256),
                                      'version': self.state['version_counter'], 'round_id': str(round_id)})
        self.state['since_held'] = int(self.state['since_held']) + 1
        self.state['trained_rounds'].append(str(round_id))
        if mix_dir and str(mix_dir) not in self.state['prefer_unseen']:
            self.state['prefer_unseen'].append(str(mix_dir))
        self.state['active_round'] = None
        self.save()
        return self.current

    def resume_after_hunt(self):
        """The supervisor closed the hunt: rounds start again from the lineage head (the reference)."""
        if self.state.get('halted'):
            self.state['history'].append({'at': round(time.time(), 1), 'action': 'resume after hunt',
                                          'halted': self.state['halted']})
            self.state['halted'] = None
            self.save()

    def held_due(self):
        return int(self.state['since_held']) >= int(self.state['held_every'])

    def start_measurement(self):
        """The current policy goes to held against the reference: (candidate, reference); the interval restarts."""
        job = {'candidate': self.current, 'reference': dict(self.state['reference_policy'])}
        self.state['since_held'] = 0
        self.state['measuring'] = job
        self.save()
        return job

    def _rollback(self, version_counter_bump=True):
        ref = self.state['reference_policy']
        lineage = self.state['lineage']
        for i in range(len(lineage) - 1, -1, -1):
            if lineage[i]['sha256'] == ref['sha256']:
                self.state['lineage'] = lineage[:i + 1]
                return
        # the reference left the lineage earlier (a phase restart): it starts a new one
        self.state['version_counter'] = int(self.state['version_counter']) + 1
        self.state['lineage'] = [{**ref, 'version': self.state['version_counter'], 'round_id': None}]

    def record_verdict(self, verdict_record):
        """Apply one held verdict (``verdict``, ``adopt``, ``sha256`` and ``checkpoint`` of the candidate)."""
        verdict, adopt = verdict_record['verdict'], bool(verdict_record.get('adopt'))
        sha, ckpt = verdict_record['sha256'], verdict_record['checkpoint']
        event = {'at': round(time.time(), 1), 'phase': self.phase, 'verdict': verdict, 'adopt': adopt, 'sha256': sha,
                 'vs_start': verdict_record.get('vs_start')}
        if sha in self.state['applied_verdicts']:
            return {**event, 'action': 'already applied'}
        self.state['applied_verdicts'].append(sha)
        if (self.state.get('measuring') or {}).get('candidate', {}).get('sha256') == sha:
            self.state['measuring'] = None
        if not self.in_lineage(sha):         # an earlier verdict already rolled back below it
            event['action'] = 'moot (not in the lineage)'
            self.state['history'].append(event)
            self.save()
            return event
        if verdict == 'success':
            self.state['held_every'] = min(2 * int(self.state['held_every']), int(self.state['held_every_max']))
            self.state['consecutive_neutral'] = 0
        elif verdict == 'neutral':
            self.state['held_every'] = int(self.state['base_held_every'])
            self.state['consecutive_neutral'] = int(self.state['consecutive_neutral']) + 1
        else:
            self.state['held_every'] = 1
            self.state['consecutive_neutral'] = 0
            event['bug_hunt'] = f'regression vs {self.state["reference_policy"]["sha256"][:12]}: hunt it (AGENTS.md ' \
                                '"degradation means bug")'
            # AGENTS.md: a significant unexplained held regression ends the experiment at that eval
            self.state['halted'] = {'at': round(time.time(), 1), 'sha256': sha, 'checkpoint': str(ckpt),
                                    'reference': dict(self.state['reference_policy']),
                                    'vs_start': verdict_record.get('vs_start'), 'reason': event['bug_hunt']}
        if adopt:
            self.state['reference_policy'] = {'checkpoint': str(ckpt), 'sha256': str(sha)}
            event['action'] = 'adopt'
        else:
            self._rollback()
            self.state['since_held'] = 0
            event.update(action='rollback', rollback_to=dict(self.state['reference_policy']))
        if self.state['consecutive_neutral'] >= 2:
            old = self.state['phase']
            self.state['phase'] = 'grpo' if old == 'sft' else 'sft'
            self.state['consecutive_neutral'] = 0
            self._rollback()
            event.update(switch_phase={'from': old, 'to': self.state['phase']},
                         start_policy=dict(self.state['reference_policy']))
        event['held_every'] = self.state['held_every']
        self.state['history'].append(event)
        self.save()
        return event


ROUND_STAGES = ('prepared', 'trained', 'published')


class AsyncRoundDriver:
    # the DGX and remote collection builders share Cycle.collect_dir (the main loop and the Pop loop both build)
    COLLECT_LOCK = threading.RLock()
    # Pop serves one thing at a time: the Pop loop's collection serve or held-v4's measurement (held_v4_step)
    POP_LOCK = threading.Lock()
    POP_POLL_SECONDS = 120
    """Continuous expert iteration in rounds (async-train §2-§5, §7.4): standing services on the DGX (a collection of
    the newest published policy, filled reasoning on its actor, the teacher campaign on the frontier's rare tasks),
    back-to-back rounds, and the DGX held series beside them.

    A round (``ROOT/<round>``, stage markers ``stages/*.done``, each resumable):
      1. prepared: snapshot the lineage's collections (current and previous policy), the frontier of the current policy
         with the previous one adjacent, extraction (collections, teacher campaigns, GRPO groups), replay, fills,
         the own Neuralese graph and the mix build (``--frontier``, ``--previous-policy``, prefer-unseen over the
         lineage's rounds) -> ``round.json`` (blob first). The next round's preparation runs while this one trains.
      2. trained: LoRA SFT from the current policy, one recipe for the G4 and the DGX (``launch-sft.sh``: epochs 1,
         warmup 10, 30k loss tokens per step; ``--round-id``, ``--export-factors-dir``, ``--factors-only``, ``--resume``),
         on Colab (``pop_rl_actor run-sft``, learner state synced back every 15 minutes) or on the DGX through the
         ledger; a Colab session away longer than ``--fallback-wait-minutes`` hands the round to the DGX from its last
         state (recorded in ``round.json``). In the GRPO phase: ``grpo.py train`` for ``--grpo-steps`` from the current
         policy, its last step's factors.
      3. published: the FP32 factor merge (``merge_factors_to_checkpoint``) and every engine's artifact
         (``publish_merged_policy``: native, SGLang, GGUF) in one ledger unit (``publish-round``) -> ``policy/``; the
         remote actors rebound; the lineage advances (refused if a rollback removed the round's base meanwhile).
    Held: every ``held_every``-th round's policy is measured on the DGX series in the background against the reference
    (``Cycle.dgx_held``); its verdict adopts, rolls back (the lineage returns to the reference) or switches phase. A
    held success rewrites the lfm-latest pointer and refreshes its profiles (``promote_policy``)."""

    def __init__(self, args):
        self.a = args
        self.root = Path(args.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        from natlang_neuralese.common.hashing import sha256_file_hex_cached
        start_sha = args.start_sha or (BASE_SHA if str(args.start) == BASE_CKPT else
                                       sha256_file_hex_cached(Path(args.start)))
        self.controller = PhaseController.initialize(
            self.root / 'phase-state.json', start_checkpoint=args.start, start_sha=start_sha,
            held_every=args.held_every, held_every_max=args.held_every_max, initial_phase=args.initial_phase,
            prefer_unseen=args.prefer_unseen, reference_checkpoint=args.reference,
            reference_sha=args.reference and (args.reference_sha or sha256_file_hex_cached(Path(args.reference))))
        # the arguments a restart passes against the state they once seeded (adopted or refused, never shadowed)
        self.argument_changes = self.controller.reconcile_arguments(
            start_checkpoint=args.start, start_sha=start_sha, reference_checkpoint=args.reference,
            reference_sha=args.reference and (args.reference_sha or sha256_file_hex_cached(Path(args.reference))),
            held_every=args.held_every, held_every_max=args.held_every_max, prefer_unseen=args.prefer_unseen)
        self.c = Cycle(self.cycle_args())
        if self.argument_changes:
            self.note(arguments_reconciled=self.argument_changes)
        self.lock = threading.RLock()
        self.verdicts = []              # verdict records the held thread produced, applied by the main loop
        self.held_wake = threading.Event()
        self.prep_thread = None
        self.colab_hosts = list(self.c.colab_hosts)
        self.held_done = set()          # held-v3 candidates the held thread finished
        self.held_v4_cadence = parse_held_v4_cadence(getattr(args, 'held_v4_cadence', HELD_V4_CADENCE))
        self.held_v4 = HeldV4Queue(self.root / 'held-v4' / 'queue.json')
        self.teacher_lock = threading.Lock()   # one campaign launch at a time (main loop, waits, the standing thread)
        self.handback_next = {}         # round -> when its DGX SFT is next checked for a move to a session
        self.handback_blocked = set()   # rounds whose Colab learner failed: they stay on the DGX
        self.handed_back = None         # the round a hand-back just moved (ends the DGX wait)

    def cycle_args(self):
        """The Cycle helper's arguments (ledger, pin, DGX held series, collection and fill builders) from ours."""
        a = self.a
        return argparse.Namespace(
            dir=str(self.root), name=a.name, start=a.start, start_sha=a.start_sha or BASE_SHA, commit=a.commit,
            pin=a.pin, sft_work=a.sft_work, seed_work=a.seed_work, live_work=a.live_work, k=a.k, collect_hours=a.collect_hours, slots=a.slots,
            kv_gb=a.kv_gb, fill=a.fill, fill_inflight=a.fill_inflight, neuralese_variants=a.neuralese_variants,
            neuralese_variant_share=a.neuralese_variant_share, no_neuralese_variants=a.no_neuralese_variants,
            collect_weight_dtype=a.collect_weight_dtype, teacher_attempts=a.teacher_attempts,
            memetic_share=getattr(a, 'memetic_share', MEMETIC_SHARE),
            memetic_layout=getattr(a, 'memetic_layout', 'staged'),
            memetic_fit_quantile=getattr(a, 'memetic_fit_quantile', 0.9),
            memetic_nested_slots=getattr(a, 'memetic_nested_slots', True), held_profile=getattr(a, 'held_profile', 'pop'),
            held_pin=a.held_pin, held_series_host=getattr(a, 'held_profile', 'pop'), held_samples=a.held_samples,
            held_slots=a.held_slots,
            no_held_reuse=False, strict_held_reuse=a.strict_held_reuse, colab_hosts=a.colab_hosts, fallback_wait_minutes=a.fallback_wait_minutes,
            collect_profile='dgx', sft_profile='dgx', select_args=a.select_args, prefer_unseen=[],
            # The controller's immutable source supplies data tooling; the actor/learner
            # keep --commit/--pin. Explicit old-pin paths opt out at a normal resume.
            fill_repo=getattr(a, 'fill_repo', None) or getattr(a, 'data_repo', None) or str(REPO),
            data_repo=getattr(a, 'data_repo', None) or str(REPO),
            minimum_data_reader_commit=getattr(a, 'minimum_data_reader_commit', None),
            success_pts=a.success_pts, fail_pts=a.fail_pts)

    def note(self, **record):
        self.c.note(stage='async', **record)

    # -- the standing services ----------------------------------------------------------------------------------------
    def collect_dir(self, policy):
        return self.root / 'collect' / f'v{policy["version"]:04d}-{policy["sha256"][:12]}'

    def collection_unit(self, policy):
        return f'{self.a.name}-collect-v{policy["version"]:04d}'

    def select_command(self, out):
        """The collection's task order (``select``: failed first, then unmeasured, then by rate) from every student
        outcome so far, this loop's collections included."""
        groups = [str(OPS / 'grpo-v*' / '*' / 'groups' / 'g*'), str(OPS / 'ei' / '*' / 'collect' / 'groups' / 'g*'),
                  str(self.root / 'collect' / '*' / 'groups' / 'g*'), str(self.root / '*' / 'grpo' / 'groups' / 'g*')]
        frontiers = sorted(self.root.glob('r*/frontier.json'))[-1:]
        return self.c.py('expert_iteration.py', 'select', '--out', out,
                         *[x for g in groups for x in ('--rates-groups', g)],
                         *[x for f in frontiers for x in ('--frontier', f)], *self.a.select_args.split())

    def ensure_collection(self, policy):
        """The standing DGX collection serves exactly ``policy`` (the newest published): an earlier policy's collection
        unit stops (its scored groups stay), this one starts once (``--hours`` is a long horizon: the next publication
        stops it). Filled reasoning draws on its actor beside it."""
        unit, work = self.collection_unit(policy), self.collect_dir(policy)
        for other in sorted((self.root / 'collect').glob('v*')) if (self.root / 'collect').exists() else ():
            other_unit = f'{self.a.name}-collect-{other.name.split("-")[0]}'
            if other_unit == unit:
                continue
            for u in (other_unit, f'{other_unit}-fill', *self.remote_units(other_unit)):
                if self.c.present(u):
                    self.c.stop_unit(u)
                    self.note(collection_stopped=u)
        if self.c.present(unit):           # running, or preempted and requeued (the ledger relaunches it)
            return unit
        work.mkdir(parents=True, exist_ok=True)
        tasks = work / 'tasks.json'
        if not tasks.exists():
            self.c.ledger(f'{unit}-select', 6, 'medium', self.select_command(tasks), work / 'select.log')
        with self.COLLECT_LOCK:
            self.c.collect_dir = work
            argv = self.c.collect_command(unit, policy['checkpoint'], policy['sha256'], self.a.slots)
        argv[argv.index('--tasks') + 1] = str(tasks)
        budget = self.a.kv_gb + 24 + round(self.a.slots * COLLECTOR_SLOT_GB)
        self.c.ledger(unit, budget, COLLECTION_PRIORITY, argv, work / 'collect.log', detach=True)
        self.note(collection_started=unit, policy=policy['sha256'], work=str(work))
        if self.a.fill:
            threading.Thread(target=self._fill_beside, args=(unit, work, policy), daemon=True).start()
        return unit

    def _fill_beside(self, unit, work, policy):
        deadline = time.time() + 7200
        while not (work / 'actor.json').exists() and time.time() < deadline and self.c.active(unit):
            time.sleep(30)
        if not (work / 'actor.json').exists():
            return
        actor = json.loads((work / 'actor.json').read_text())
        if actor['checkpoint_sha256'] != policy['sha256']:
            return
        self.c.work_tokenizer()
        self.c.ledger(f'{unit}-fill', FILL_GB, 'medium', self.c.fill_command(actor), work / 'fill.log', detach=True,
                      repo=self.c.fill_repo())
        self.note(fill_started=f'{unit}-fill', policy=policy['sha256'])

    def remote_targets(self):
        """The ``--rebind-actor HOST@FRONT_URL[=SLOTS]`` targets: {index, host, front, slots, pop, suffix}. A target on
        Pop's host (``pop_rl_actor.POP``) is a standing Pop actor (``pop_step``: served, not rebound in place, and
        yielding to held measurements); its collector unit ends in ``POP_COLLECT_SUFFIX`` (every Pop serve stops it),
        the others in ``-remote{index}``."""
        import pop_rl_actor
        out = []
        for i, target in enumerate(self.a.rebind_actor):
            host, front, slots = parse_remote_target(target, self.a.remote_slots)
            pop = bool(host) and host == pop_rl_actor.POP
            out.append({'index': i, 'host': host, 'front': front, 'slots': slots, 'pop': pop,
                        'suffix': pop_rl_actor.POP_COLLECT_SUFFIX if pop else f'-remote{i}'})
        return out

    def remote_units(self, unit):
        """The remote actors' collector units beside DGX collection unit ``unit`` (one per ``--rebind-actor``)."""
        return [unit + t['suffix'] for t in self.remote_targets()]

    @staticmethod
    def target_endpoint(target):
        import pop_rl_actor
        from urllib.parse import urlparse
        return pop_rl_actor.remote_endpoint(target['host'], urlparse(target['front']).port)

    def remote_actors(self, include_pop=True):
        """[(host, VM-local front URL, DGX-side endpoint)] of the hosted targets (Colab actors beside the learner reach
        the DGX through the host's forwarded ports; Pop over Tailscale)."""
        return [(t['host'], t['front'], self.target_endpoint(t)) for t in self.remote_targets()
                if t['host'] and (include_pop or not t['pop'])]

    def ensure_remote_collection(self, policy):
        """A DGX-side collector per remote actor that serves ``policy`` (``collect --actor``): restarted with the
        policy like the DGX collection (a rebound actor serves the new policy; the old collector drops it). An actor
        not serving ``policy`` is rebound to it first (``ensure_remote_policy``)."""
        self.ensure_remote_policy(policy)
        for target in self.remote_targets():
            if target['host'] and not target['pop']:     # Pop's collector: pop_step (it yields to held measurements)
                self.start_remote_collector(policy, target)

    def remote_collect_argv(self, policy, target, endpoint):
        """(unit, work dir, argv, ledger GB) of ``target``'s DGX-side collector of ``policy``."""
        unit = self.collection_unit(policy) + target['suffix']
        work = self.collect_dir(policy).with_name(self.collect_dir(policy).name + target['suffix'])
        with self.COLLECT_LOCK:
            self.c.collect_dir = work
            argv = self.c.collect_command(unit, policy['checkpoint'], policy['sha256'], target['slots'],
                                          endpoints=[f'{endpoint}={target["slots"]}'])
        argv[argv.index('--tasks') + 1] = str(self.collect_dir(policy) / 'tasks.json')
        return unit, work, argv, max(8, round(target['slots'] * COLLECTOR_SLOT_GB) + 4)

    def start_remote_collector(self, policy, target):
        """``target``'s collector of ``policy`` (``collect --actor``) once its actor serves ``policy``."""
        endpoint = self.target_endpoint(target)
        unit = self.collection_unit(policy) + target['suffix']
        if self.c.present(unit) or not endpoint_alive(endpoint, policy['sha256']):
            return
        unit, work, argv, budget = self.remote_collect_argv(policy, target, endpoint)
        work.mkdir(parents=True, exist_ok=True)
        tasks = self.collect_dir(policy) / 'tasks.json'
        if not tasks.exists():
            self.c.ledger(f'{unit}-select', 6, 'medium', self.select_command(tasks), work / 'select.log')
        self.c.ledger(unit, budget, COLLECTION_PRIORITY, argv, work / 'collect.log', detach=True)
        self.note(remote_collection=unit, host=target['host'], endpoint=endpoint, slots=target['slots'])

    # -- Pop: a standing collection actor whenever no held measurement holds it -------------------------------------
    def pop_target(self):
        return next((t for t in self.remote_targets() if t['pop']), None)

    def pop_loop(self):
        while True:
            try:
                self.pop_step()
            except Exception as error:      # Pop is an extra actor: note and retry
                self.note(pop_collect_failed=repr(error)[:300])
            time.sleep(self.POP_POLL_SECONDS)

    def pop_collector_units(self):
        """Present Pop collector units of this driver (any policy)."""
        import pop_rl_actor
        root = self.root / 'collect'
        units = {f'{self.a.name}-collect-{d.name.split("-")[0]}{pop_rl_actor.POP_COLLECT_SUFFIX}'
                 for d in (root.glob('v*') if root.exists() else ())}
        return sorted(u for u in units if self.c.present(u))

    def pop_step(self):
        """Pop collects the lineage head in the SFT phase whenever nothing else holds it. A held measurement (any
        ``pop-held.sh`` / ``held-eval`` process, another serve in progress, or held-v4 here holding POP_LOCK) stops
        the Pop collector; afterwards Pop is served again with the collection law and its collector restarts. A new
        policy is served (the checkpoint copied, not rebound in place: Pop's driver relaunches a dead engine from the
        checkpoint it was served, so an in-place rebind would revert there)."""
        import pop_rl_actor
        target = self.pop_target()
        if target is None:
            return
        with self.lock:
            cur, phase = dict(self.controller.current), self.controller.phase
        want = self.collection_unit(cur) + target['suffix']
        busy = 'phase ' + phase if phase != 'sft' else pop_rl_actor.pop_busy()
        for unit in self.pop_collector_units():
            if busy or unit != want:
                self.c.stop_unit(unit)
                self.note(pop_collector_stopped=unit, reason=busy or 'another policy')
        if busy or not self.POP_LOCK.acquire(blocking=False):
            return
        try:
            weight_dtype = getattr(self.a, 'collect_weight_dtype', COLLECT_WEIGHT_DTYPE)
            settings, _ = pop_rl_actor.pop_sglang(pop_rl_actor.POP_KV_CACHE_DTYPE, self.a.pop_max_running,
                                                  weight_dtype=weight_dtype, overlap_schedule=True)
            state = pop_rl_actor.read_json(pop_rl_actor.STATE) or {}
            if not (state.get('policy_id') == cur['sha256'] and state.get('sglang') == settings
                    and pop_rl_actor.attested(state['endpoint'], cur['sha256'])):
                if self.c.present(want):
                    self.c.stop_unit(want)
                started = time.time()
                with (self.root / 'pop-serve.log').open('a') as log:
                    subprocess.run([PY, str(Path(pop_rl_actor.__file__).resolve()), 'serve-checkpoint',
                                    '--checkpoint', str(cur['checkpoint']), '--policy-id', cur['sha256'],
                                    '--commit', self.a.commit, '--kv-cache-dtype', pop_rl_actor.POP_KV_CACHE_DTYPE,
                                    '--weight-dtype', weight_dtype, '--max-running', str(self.a.pop_max_running),
                                    '--overlap-schedule'], stdout=log, stderr=subprocess.STDOUT, timeout=3600,
                                   check=True)
                self.note(pop_served=cur['sha256'], seconds=round(time.time() - started))
                if pop_rl_actor.pop_busy():
                    return
        finally:
            self.POP_LOCK.release()
        self.start_remote_collector(cur, target)

    def stop_collections(self, *, remote=True):
        """Stop the DGX collections and their fills (they hold the GPU the DGX learner needs) and, with ``remote``,
        the remote actors' collection clients too."""
        for work in sorted((self.root / 'collect').glob('v*')) if (self.root / 'collect').exists() else ():
            unit = f'{self.a.name}-collect-{work.name.split("-")[0]}'
            for u in (unit, f'{unit}-fill', *(self.remote_units(unit) if remote else ())):
                if self.c.present(u):          # a requeued one too, or the ledger relaunches it beside the learner
                    self.c.stop_unit(u)

    def standing_teacher(self):
        """``ensure_teacher`` from a wait or the loop: a failure is logged, never the caller's end."""
        try:
            self.ensure_teacher()
        except Exception as error:
            self.note(teacher_check_failed=repr(error)[:300])

    def build_code(self):
        """The commit of the build code (``--data-repo``, else the pin): its pin record, else git; what a rebuild is
        recorded under."""
        repo = self.c.data_repo()
        marker = repo / '.natlang-pin.json'
        if marker.exists():
            return json.loads(marker.read_text()).get('commit')
        try:
            return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def rebuild_requested(self, round_dir):
        """Why the prepared round's mix is rebuilt before it trains, or None: ``stages/rebuild.request`` (written by
        hand: picked up at the round's start by a running driver), or ``--rebuild-round RNNNN`` while round.json
        records no rebuild under the current build code (so a restart with the same argv does not rebuild again)."""
        if (round_dir / 'stages' / 'rebuild.request').exists():
            return 'stages/rebuild.request'
        if round_dir.name in (getattr(self.a, 'rebuild_round', None) or ()):
            manifest = json.loads((round_dir / 'round.json').read_text())
            code = self.build_code()
            if not any(r.get('build_code') == code for r in manifest.get('rebuilds') or ()):
                return f'--rebuild-round under build code {code}'
        return None

    def rebuild_round(self, round_dir, policies, reason):
        """Rerun only a prepared round's mix build, under the current build code, from its recorded inputs: the round's
        frontier, data outputs (filled, own Neuralese graph, extractions in --sft-work), size, previous policy and
        prefer-unseen list (round.json), with the lineage it was prepared from (refused when that is no longer the
        lineage, or once its SFT started). The old mix stays as ``mix.pre-rebuild-<time>``; a failed build puts it
        back. ``chains.sha256`` and round.json (blob first, every other recorded field kept, a DGX recap dropped:
        the recap reruns from the new mix when the round starts on the DGX) follow the new mix; round.json
        ``rebuilds`` records each one. Returns True when it rebuilt."""
        from natlang_neuralese.common.jsonio import write_json_durable
        manifest = json.loads((round_dir / 'round.json').read_text())
        marker = round_dir / 'stages' / 'rebuild.request'
        run = round_dir / 'sft'
        started = ((run / 'learner-state' / 'learner_manifest.json').exists() or (run / 'factors').exists()
                   or dgx_sft_progress(run / 'train.jsonl') is not None
                   or self.dgx_sft_running(round_dir, f'{self.a.name}-{round_dir.name}-sft'))
        cur, prev = policies[-1], (policies[-2] if len(policies) > 1 else None)
        lineage_ok = (manifest.get('base_policy_sha') == cur['sha256']
                      and manifest.get('previous_policy_sha') == (prev['sha256'] if prev else None))
        if started or not lineage_ok:
            why = 'its SFT started' if started else 'it was prepared from another lineage'
            self.note(round=round_dir.name, rebuild_refused=why, reason=reason)
            marker.unlink(missing_ok=True)
            return False
        size = manifest.get('size') or json.loads((round_dir / 'size.json').read_text())
        _, _, build = self.prepare_plan(round_dir, policies, size, prefer_unseen=manifest.get('prefer_unseen') or [])
        mix = round_dir / 'mix'
        stamp = time.strftime('%Y%m%dT%H%M%S')
        n = 0
        while (round_dir / f'mix.pre-rebuild-{stamp}{f"-{n}" if n else ""}').exists():
            n += 1
        stamp = f'{stamp}-{n}' if n else stamp
        old = round_dir / f'mix.pre-rebuild-{stamp}'
        code = self.build_code()
        self.note(round=round_dir.name, rebuild=reason, build_code=code, old_mix=str(old))
        if mix.exists():
            os.replace(mix, old)
        t = time.time()
        try:
            self.c.ledger(f'{self.a.name}-{round_dir.name}-rebuild-{stamp}', 32, STAGE_PRIORITY, build,
                          round_dir / 'rebuild.log', repo=self.c.data_repo())
            report = json.loads((mix / 'build.json').read_text())
            if not (mix / 'chains.pkl').is_file():
                raise SystemExit(f'round {round_dir.name}: the rebuild wrote no chains.pkl ({round_dir}/rebuild.log)')
        except BaseException:
            if old.exists():
                shutil.rmtree(mix, ignore_errors=True)
                os.replace(old, mix)
            self.note(round=round_dir.name, rebuild_failed=f'{round_dir}/rebuild.log', restored=str(mix))
            raise
        core = {'schema', 'round_id', 'base_policy_sha', 'previous_policy_sha', 'chains_path', 'chains_sha256',
                'chains_bytes', 'episode_ids', 'prefer_unseen', 'profile', 'profile_changes', 'committed_at'}
        entry = {'at': round(time.time(), 1), 'reason': reason, 'build_code': code, 'seconds': round(time.time() - t, 1),
                 'old_mix': str(old) if old.exists() else None, 'old_chains_sha256': manifest.get('chains_sha256')}
        extra = {**{k: v for k, v in manifest.items() if k not in core and k != 'recap'},
                 'mix_dir': str(mix), 'expected_hours': self.expected_hours(report),
                 'rebuilds': [*(manifest.get('rebuilds') or []), entry]}
        new = write_round_manifest(
            round_dir, round_id=manifest.get('round_id', round_dir.name), base_policy_sha=manifest['base_policy_sha'],
            previous_policy_sha=manifest.get('previous_policy_sha'), chains_path=mix / 'chains.pkl',
            episode_ids=report.get('admitted_episode_keys') or [], prefer_unseen=manifest.get('prefer_unseen') or [],
            profile=manifest.get('profile', self.a.sft_profile), profile_changes=manifest.get('profile_changes') or [],
            extra=extra)
        write_json_durable(round_dir / 'stages' / 'build.done', {'time': time.time(), 'rebuild': entry})
        marker.unlink(missing_ok=True)
        self.note(round=round_dir.name, rebuilt=new['chains_sha256'], old_chains=entry['old_chains_sha256'],
                  episodes=len(new['episode_ids']))
        return True

    def newest_frontier(self):
        """The newest round's measured frontier (``rNNNN/frontier.json`` of the highest round not discarded), or None."""
        found = [f for f in sorted(self.root.glob('r[0-9]*/frontier.json'))
                 if not (f.parent / 'discarded.json').exists()]
        return found[-1] if found else None

    def ensure_teacher(self, round_dir=None):
        """The teacher campaign stands: whenever its unit is idle, the newest frontier's campaign runs
        (``teacher-rNNNN``). Called from the main loop, the end of a round's preparation, the SFT and preparation
        waits and a standing thread every ``--teacher-check-minutes`` (2026-11-24: called only when a round was
        prepared, the campaign stayed idle from 01:00 until a hand launch at 02:10 under a restarted driver joining an
        already-prepared round). A campaign of that frontier which already ran resumes (teacher_gen skips finished
        tasks) at most once per ``TEACHER_RELAUNCH_MINUTES``, and not after a relaunch that scored nothing new (its
        tasks are done: the next frontier starts the next one). ``round_dir`` is ignored (the newest frontier
        decides)."""
        unit = f'{self.a.name}-teacher'
        if self.a.no_teacher_campaign:
            return None
        with self.teacher_lock:
            frontier = self.newest_frontier()
            if frontier is None or self.c.present(unit):
                return None
            work = self.root / f'teacher-{frontier.parent.name}'
            record_path = work / 'driver-launch.json'
            scored = sum(1 for line in (work / 'manifest.jsonl').open() if line.strip()) \
                if (work / 'manifest.jsonl').exists() else 0
            last = json.loads(record_path.read_text()) if record_path.exists() else None
            launched = (work / 'launches.jsonl').exists()
            if launched and last is not None:
                if scored <= int(last.get('scored_at_launch', -1)):
                    return None                 # the last relaunch scored nothing: this frontier's campaign is done
                if time.time() - float(last.get('at', 0)) < 60 * TEACHER_RELAUNCH_MINUTES:
                    return None
            n = launch_teacher(self.c, work, frontier,
                               [self.collect_dir(p) / 'groups' / 'g*' for p in self.lineage_policies()], unit,
                               covered_extra=sorted(self.root.glob('teacher-*')))
            if n:
                from natlang_neuralese.common.jsonio import write_json_durable
                write_json_durable(record_path, {'at': round(time.time(), 1), 'frontier': str(frontier),
                                                 'scored_at_launch': scored, 'resume': launched, 'tasks': n})
            self.note(teacher=unit, frontier=str(frontier), frontier_tasks=n, work=str(work), resume=launched)
            return work

    def teacher_loop(self):
        """The standing campaign's check, beside every wait the main loop makes (run-sft, ledger units, held)."""
        while True:
            time.sleep(60 * self.a.teacher_check_minutes)
            try:
                self.ensure_teacher()
            except Exception as error:
                self.note(teacher_check_failed=repr(error)[:300])

    def lineage_policies(self):
        """The policies whose collections a round reads: the current and the previous published policy."""
        return [p for p in (self.controller.previous, self.controller.current) if p]

    # -- 1. prepare ---------------------------------------------------------------------------------------------------
    def collections_of(self, policies):
        """Every collection of ``policies`` (the DGX one and the remote actors'), in policy order."""
        return [d for p in policies for d in sorted(self.collect_dir(p).parent.glob(self.collect_dir(p).name + '*'))
                if (d / 'runs.jsonl').exists()]

    def dgx_round_size(self, why='DGX'):
        return {'placement': 'dgx', 'why': why, 'max_target_tokens': self.a.round_target_tokens, 'max_forward_tokens': 0,
                'teacher_budget': self.a.teacher_budget, 'scale': 1.0}

    def measured_round_speed(self):
        path = self.root / ROUND_SPEED_FILE
        return json.loads(path.read_text()) if path.exists() else {}

    def forward_per_target(self):
        """Forward per weighted target token of the newest built round mix (FORWARD_PER_TARGET without one)."""
        for build in sorted(self.root.glob('r[0-9]*/mix/build.json'), reverse=True):
            try:
                mix = json.loads(build.read_text())['mix']
                return mix['forward_tokens'] / mix['weighted_target_tokens']
            except (OSError, ValueError, KeyError, ZeroDivisionError):
                continue
        return FORWARD_PER_TARGET

    def round_size(self):
        """A round's budgets by where it is expected to train, decided at preparation: a reachable Colab session fit
        for SFT (``colab_sft_fit``) -> ``--colab-round-hours`` of its learner time (``placement_round_size``); else
        (no session, an unfit one, ``--colab-round-hours 0``, a DGX-profile driver) the DGX round
        (``--round-target-tokens``, ``--teacher-budget``)."""
        hours = getattr(self.a, 'colab_round_hours', COLAB_ROUND_HOURS)
        if hours <= 0 or self.a.sft_profile != 'colab' or not self.colab_hosts:
            return self.dgx_round_size('placement sizing off or no Colab hosts')
        try:
            import pop_rl_actor
            host = next((h for h in pop_rl_actor.reachable_hosts('colab', self.colab_hosts)
                         if pop_rl_actor.colab_sft_fit(h)[0]), None)
            if host is None:
                return self.dgx_round_size('no Colab session fit for SFT answers')
            hardware = pop_rl_actor.colab_hardware(host)
        except Exception as error:      # sizing never stops a round: the DGX size is always safe
            return self.dgx_round_size(f'placement check failed: {repr(error)[:200]}')
        gpu = hardware.get('gpu_name')
        tps, source = learner_throughput(gpu, self.measured_round_speed(), hardware, self.measured_sft_speed())
        if not tps:
            return self.dgx_round_size(f'{gpu}: no throughput estimate ({source})')
        return {**placement_round_size(round_target_tokens=self.a.round_target_tokens,
                                       teacher_budget=self.a.teacher_budget, hours=hours, gpu_name=gpu,
                                       chain_tokens_per_s=tps, forward_per_target=self.forward_per_target()),
                'host': host, 'throughput_source': source}

    def round_mix(self, round_dir):
        """The mix a round trains: round.json ``mix_dir`` (a DGX recap, ``recap_for_dgx``), else ``mix/``."""
        path = Path(round_dir) / 'round.json'
        if path.exists():
            mix = json.loads(path.read_text()).get('mix_dir')
            if mix:
                return Path(mix)
        return Path(round_dir) / 'mix'

    def prefer_unseen_for(self, round_dir):
        """The builds a round's mix prefers unseen over: the lineage's trained rounds (``prefer_unseen``) and the round
        training now, when this one is prepared ahead of it (its mix is not published yet; without it the next round
        repeated most of it: aei1 r0002 held 3,166 of r0001's 3,395 filled units and 1,118 of its 1,645 teacher
        episodes). ``--no-prefer-inflight`` opts out."""
        builds = list(self.controller.state['prefer_unseen'])
        active = (self.controller.state.get('active_round') or {}).get('round_id')
        if active and active != Path(round_dir).name and not getattr(self.a, 'no_prefer_inflight', False):
            mix = self.round_mix(self.round_dir(active))
            if (mix / 'rows.jsonl.zst').exists() and str(mix) not in builds:
                builds.append(str(mix))
        return builds

    def prepare_plan(self, round_dir, policies, size=None, prefer_unseen=None):
        """(frontier argv, data script lines, build argv) of a round's preparation from ``policies``' collections,
        sized by ``size`` (``round_size``; the DGX round without one)."""
        size = size or self.dgx_round_size()
        cur, prev = policies[-1], (policies[-2] if len(policies) > 1 else None)
        collections_ = self.collections_of(policies)
        frontier = [*self.c.py('expert_iteration.py', 'frontier', '--out', round_dir / 'frontier.json',
                               *[x for d in collections_ for x in ('--collect', d)], '--policy', cur['sha256']),
                    *(['--previous-policy', prev['sha256'], '--allow-adjacent-policies'] if prev else [])]
        w = self.c.work
        lines = []
        rl = [*collections_, *[d for d in sorted(self.root.glob('r*/grpo')) if (d / 'groups').exists()]]
        teachers = [d for d in sorted(self.root.glob('teacher-*')) if (d / 'manifest.jsonl').exists()]
        lines.append(self.c.data_py('sft_sources.py', 'extract', '--work', w,
                               *[x for d in rl for x in ('--rl', f'{d}:{d}')], *[x for d in teachers for x in ('--teacher', d)],
                               '--out-name', f'episodes-{self.a.name}-{round_dir.name}'))
        lines.append(self.c.data_py('sft_sources.py', 'replay', '--work', w, '--workers', self.a.replay_workers,
                               '--runtime-repo', self.c.pin))
        if self.a.fill and (Path(self.a.fill) / 'fills.jsonl').exists():
            lines.append(self.c.fill_build_command('--out', self.a.fill, '--corpus', round_dir / 'filled',
                          *[x for s in self.failed_policies() for x in ('--exclude-policy', s)]))
        surfaces = '|'.join(re.escape('/'.join(Path(d).parts[-2:])) for d in collections_) or '(?!)'
        lines.append(self.c.data_py('sft_sources.py', 'neuralese-graph', '--work', w, '--out', round_dir / 'neuralese-graph-own',
                               '--exclude-run', 'grpo-v2/', '--exclude-run', 'grpo-v4/run1',
                               '--current-surface-run', f'^rl:({surfaces})$'))
        aux = list(self.a.aux)
        if not getattr(self.a, 'no_fallow22_aux', False) and not any(a.startswith('fallow22=') for a in aux):
            aux.append(FALLOW22_AUX)
        aux.append(f'filled={round_dir / "filled" / "chains.pkl"},share=all,task_tokens=40000,replaces=teacher')
        aux.append(f'own-neuralese-graph={round_dir / "neuralese-graph-own" / "chains.pkl"},'
                   f'share={self.a.own_neuralese_graph_share},task_tokens=40000')
        build = mix_build_command(self.c.data_repo(), work=w, out=round_dir / 'mix', frontier=round_dir / 'frontier.json',
                                  aux=aux, prefer_unseen=(self.prefer_unseen_for(round_dir) if prefer_unseen is None
                                                          else prefer_unseen),
                                  teacher_budget=size['teacher_budget'], own_share=self.a.own_share,
                                  previous_policy=prev['sha256'] if prev else None,
                                  live_registry=live_holdout_registry(self.a.live_work),
                                  registry_repo=self.c.live_data_owner(),
                                  registry_ref=self.c.live_data_ref(),
                                  max_target_tokens=size['max_target_tokens'],
                                  max_forward_tokens=size['max_forward_tokens'],
                                  unfilled_live_teacher=getattr(self.a, 'unfilled_live_teacher', None))
        return list(map(str, frontier)), [list(map(str, x)) for x in lines], list(map(str, build))

    def failed_policies(self):
        """Policies held judged a failure (this loop's verdicts and every cycle's): their fills are not data."""
        out = self.c.failed_policies()
        for path in sorted((self.root / 'verdicts').glob('*.json')) if (self.root / 'verdicts').exists() else ():
            v = json.loads(path.read_text())
            if v.get('verdict') == 'failure':
                out.append(v['sha256'])
        return out

    def prepare_policies(self, round_dir, policies):
        """The policies a round's preparation is for: the ones it recorded when it started (``stages/prepare.json``,
        else an existing ``frontier.json``'s policy and previous policy), so a driver restarted after the previous
        round published continues a round prepared ahead with the policy of that time instead of re-snapshotting on
        the new one (2026-11-24: r0003's frontier and data were done, and the restart waited ~3 h for the v2
        collection). A record whose policies left the lineage (a rollback) is dropped with its stage markers (its
        running stage units stopped), and the round prepares from ``policies``."""
        stages = round_dir / 'stages'
        record = stages / 'prepare.json'
        shas = None
        if record.exists():
            shas = [p['sha256'] for p in json.loads(record.read_text())['policies']]
        elif (round_dir / 'frontier.json').exists():
            f = json.loads((round_dir / 'frontier.json').read_text())
            shas = [s for s in (f.get('previous_policy'), f.get('policy')) if s]
        if not shas:
            return list(policies)
        by_sha = {p['sha256']: p for p in self.controller.state['lineage']}
        if all(s in by_sha for s in shas):
            return [by_sha[s] for s in shas]
        self.note(round=round_dir.name, prepare_restarted='its recorded policies left the lineage', policies=shas)
        for name in PREPARE_STEPS:
            unit = f'{self.a.name}-{round_dir.name}-{name}'
            if self.c.present(unit):
                self.c.stop_unit(unit)
            for suffix in ('.done', '.exit'):
                (stages / f'{name}{suffix}').unlink(missing_ok=True)
        record.unlink(missing_ok=True)
        (round_dir / 'frontier.json').unlink(missing_ok=True)
        return list(policies)

    def step_finished(self, round_dir, name, unit, argv):
        """How a preparation step with no ``.done`` marker is known to have finished (the driver died between its end
        and the marker): its own exit record (``stages/<step>.exit``, written by the unit: status 0 for this argv), or
        for a unit launched before those records its journal (the last start ended without a failure) together with
        its output. None when it did not (it runs again)."""
        exit_path = round_dir / 'stages' / f'{name}.exit'
        if exit_path.exists():
            try:
                rec = json.loads(exit_path.read_text())
            except ValueError:
                return None
            return 'exit-status' if rec.get('rc') == 0 and rec.get('argv_sha256') == argv_digest(argv) else None
        output = {'frontier': [round_dir / 'frontier.json'],
                  'data': [round_dir / 'neuralese-graph-own' / 'chains.pkl'],
                  'build': [round_dir / 'mix' / 'build.json', round_dir / 'mix' / 'chains.pkl']}.get(name, [])
        if output and all(p.exists() for p in output) and unit_last_run_succeeded(unit):
            return 'journal'
        return None

    def prepare(self, round_dir, policies):
        """Stage 1 (resumable; ``stages/prepared.done``): wait until the newest policy's collection has run
        ``--snapshot-min-hours`` (the frontier needs its outcomes), register the snapshot, frontier, data, build,
        ``round.json``. The policies and the snapshot are recorded (``stages/prepare.json``): a restarted driver
        continues with them (``prepare_policies``) and does not wait or snapshot again; a step that finished before
        the driver died is not rerun (``step_finished``)."""
        if (round_dir / 'stages' / 'prepared.done').exists():
            return json.loads((round_dir / 'round.json').read_text())
        (round_dir / 'stages').mkdir(parents=True, exist_ok=True)
        from natlang_neuralese.common.jsonio import write_json_durable
        given = [p['sha256'] for p in policies]
        policies = self.prepare_policies(round_dir, policies)
        if [p['sha256'] for p in policies] != given:
            self.note(round=round_dir.name, prepare_continues_with=[p['sha256'] for p in policies], lineage=given)
        cur = policies[-1]
        record_path = round_dir / 'stages' / 'prepare.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        if not record:
            record = {'policies': [{k: p.get(k) for k in ('version', 'sha256', 'checkpoint')} for p in policies],
                      'started_at': round(time.time(), 1)}
            write_json_durable(record_path, record)
        snapshot_taken = bool(record.get('snapshot_at')) or (round_dir / 'stages' / 'frontier.done').exists() \
            or (round_dir / 'frontier.json').exists()
        # the newest policy's collections: the DGX one and the remote actors' (which keep collecting while a DGX SFT
        # holds the DGX collection stopped: the snapshot waits for them)
        units = [self.collection_unit(cur), *self.remote_units(self.collection_unit(cur))]
        while not snapshot_taken:
            runs = [d / 'runs.jsonl' for d in self.collections_of([cur])]
            if runs:
                started = min(json.loads(r.read_text().splitlines()[0])['started'] for r in runs)
                if time.time() - started >= 3600 * self.a.snapshot_min_hours:
                    break
            if not any(self.c.present(u) for u in units):
                if runs:
                    break                   # stopped early (a rollback, a phase switch): what it scored is the snapshot
                raise SystemExit(f'round {round_dir.name}: no collection of {cur["sha256"][:12]} ({self.collect_dir(cur)})')
            self.standing_teacher()
            time.sleep(60)
        if not snapshot_taken:
            for d in self.collections_of(policies):
                record_collection(d, closed=False)       # the snapshot the extraction reads (the collection grows on)
            record = {**record, 'snapshot_at': round(time.time(), 1)}
            write_json_durable(record_path, record)
        # the live WORKs' extractions so far (the ingester keeps adding waves; each round takes what is there)
        if self.a.live_work:
            self.note(live_links=link_live_works(self.c.work, self.a.live_work))
        # the round's size by its expected placement, decided once (a restarted driver keeps it: the build may run)
        size_path = round_dir / 'size.json'
        if size_path.exists():
            size = json.loads(size_path.read_text())
        else:
            size = {**self.round_size(), 'decided_at': round(time.time(), 1)}
            write_json_durable(size_path, size)
        self.note(round=round_dir.name, round_size=size)
        frontier, lines, build = self.prepare_plan(round_dir, policies, size)

        # Each step records stages/<step>.done, so a driver restart resumes after the last finished step instead of
        # rerunning frontier and data (hours) beneath a build that may still be reading their outputs.
        def step(name, budget, argv, repo=None):
            done = round_dir / 'stages' / f'{name}.done'
            if done.exists():
                return
            unit = f'{self.a.name}-{round_dir.name}-{name}'
            if not self.c.present(unit):
                how = self.step_finished(round_dir, name, unit, argv)
                if how:
                    self.note(round=round_dir.name, step_finished_before_restart=name, how=how)
                    done.write_text(json.dumps({'time': time.time(), 'recovered': how}) + '\n')
                    return
                (round_dir / 'stages' / f'{name}.exit').unlink(missing_ok=True)
            self.c.ledger(unit, budget, STAGE_PRIORITY, step_command(argv, round_dir / 'stages' / f'{name}.exit'),
                          round_dir / f'{name}.log', repo=repo)
            done.write_text(json.dumps({'time': time.time()}) + '\n')
        step('frontier', 4, frontier)
        self.c.work_tokenizer()
        script = round_dir / 'data.sh'
        # Preserve an admitted/requeued plan and atomically replace inactive plans.
        if not (round_dir / 'stages' / 'data.done').exists() and not self.c.present(f'{self.a.name}-{round_dir.name}-data'):
            tmp = script.with_name(script.name + '.tmp')
            tmp.write_text('set -e\n' + ''.join(' '.join(map(_quote, x)) + '\n' for x in lines))
            os.replace(tmp, script)
        step('data', 12, ['/bin/sh', str(script)], repo=self.c.data_repo())
        # 32 GB: r0001/r0002 peaked at 21.2/17.1 GB under 24; the aux parts now load before the teacher admission and
        # --prefer-unseen reads older mixes' chains.pkl for their trained units (DECISIONS 2026-11-24)
        step('build', 32, build, repo=self.c.data_repo())
        report = json.loads((round_dir / 'mix' / 'build.json').read_text())
        manifest = write_round_manifest(
            round_dir, round_id=round_dir.name, base_policy_sha=cur['sha256'],
            previous_policy_sha=policies[-2]['sha256'] if len(policies) > 1 else None,
            chains_path=round_dir / 'mix' / 'chains.pkl', episode_ids=report.get('admitted_episode_keys') or [],
            prefer_unseen=self.prefer_unseen_for(round_dir), profile=self.a.sft_profile,
            extra={'frontier_policy': cur['sha256'], 'collections': [str(d) for d in self.collections_of(policies)],
                   'size': size, 'mix_dir': str(round_dir / 'mix'),
                   'expected_hours': self.expected_hours(report)})
        (round_dir / 'stages' / 'prepared.done').write_text(json.dumps({'time': time.time()}) + '\n')
        self.ensure_teacher(round_dir)
        return manifest

    def prepare_ahead(self, round_dir, policies):
        """The next round's preparation while this round trains (its frontier is the current policy, adjacent to the
        policy it will train from)."""
        def work():
            try:
                self.prepare(round_dir, policies)
            except BaseException as error:      # the round prepares again when it starts
                self.note(prepare_ahead_failed=round_dir.name, error=repr(error)[:300])
        self.prep_thread = threading.Thread(target=work, daemon=True)
        self.prep_thread.start()

    # -- 2. train -----------------------------------------------------------------------------------------------------
    def sft_env(self, round_dir, policy, *, activation_gb, preflight_only):
        """launch-sft.sh's environment for a round: the one recipe for the G4 and the DGX learner (launch-sft.sh
        writes ``--epochs 1 --warmup 10 --batch-loss-tokens 30000 --min-lr-fraction 0.1 --clip 1.0``; a resumed state
        indexes the same step list wherever it trains) plus the round's flags."""
        self.c.require_data_readers()
        run = round_dir / 'sft'
        extra = ['--round-id', round_dir.name, '--export-factors-dir', str(run / 'factors'), '--factors-only', '--resume']
        vals = [v for v in ('val-own.pkl', 'val-teacher.pkl') if (round_dir / 'mix' / v).exists()]
        if vals:
            extra += [*(x for v in vals[1:] for x in ('--val-chains', str(round_dir / 'mix' / v))), '--val-every', '20']
        env = {**{k: v for k, v in os.environ.items() if k not in ('WARMUP', 'BLT', 'VAL', 'EXTRA', 'PREFLIGHT_ONLY')},
               'CHAINS': str(self.round_mix(round_dir) / 'chains.pkl'),
               'CHAINS_SHA256': (round_dir / 'chains.sha256').read_text().split()[0],
               'CKPT': policy['checkpoint'], 'SHA': policy['sha256'], 'WORK': str(run), 'UNIT': f'{self.a.name}-{round_dir.name}-sft',
               'LR': str(self.a.lr), 'WARMUP': '10', 'BLT': '30000', 'ACT': str(activation_gb),
               'BUDGET': str(self.a.sft_budget), 'EXTRA': ' '.join([*extra, *self.a.sft_extra.split()])}
        if vals:
            env['VAL'] = str(round_dir / 'mix' / vals[0])
        if preflight_only:
            env['PREFLIGHT_ONLY'] = '1'
        return env

    def launch_sft(self, round_dir, policy, *, activation_gb, preflight_only):
        run = round_dir / 'sft'
        run.mkdir(parents=True, exist_ok=True)
        subprocess.run(['/bin/sh', str(OPS / 'grpo-v4' / 'launch-sft.sh'), str(self.c.pin)],
                       env=self.sft_env(round_dir, policy, activation_gb=activation_gb, preflight_only=preflight_only),
                       check=True, stdout=(round_dir / 'sft-launch.log').open('a'), stderr=subprocess.STDOUT)

    def colab_command(self, round_dir, host, profile):
        run = round_dir / 'sft'
        return [PY, str(self.c.pin / 'scripts' / 'neuralese_data' / 'pop_rl_actor.py'), 'run-sft', '--host', host,
                '--run-name', f'{self.a.name}-{round_dir.name}', '--work', str(run), '--launch-json',
                str(run / 'launch.json'), '--commit', self.a.commit, '--state-sync-minutes',
                str(profile.state_sync_minutes), '--state-transfer', getattr(self.a, 'sft_state_transfer', 'auto')]

    def train_sft(self, round_dir, policy):
        """Stage 2 of an SFT round: on Colab while a session answers, else (or after ``--fallback-wait-minutes``
        away) on the DGX from the synced learner state; a round on the DGX moves back to a session fit for SFT when
        the estimated remaining time favours it (``sft_handback``). Done when ``sft/factors/policy_manifest.json``
        exists."""
        factors = round_dir / 'sft' / 'factors'
        unit = f'{self.a.name}-{round_dir.name}-sft'
        while not (factors / 'policy_manifest.json').exists():
            # a DGX learner of this round that runs (a driver restarted mid-SFT) is joined, never raced by a second
            # learner on the session: the hand-back moves it, verified, when that pays
            if self.a.sft_profile == 'colab' and not self.dgx_sft_running(round_dir, unit):
                self.colab_sft(round_dir, policy)
                if (factors / 'policy_manifest.json').exists():
                    break
            # the standing DGX collection (74 GB beside a ~64-80 GB learner on 121 GB) yields for the whole SFT: the
            # ledger would preempt it (COLLECTION_PRIORITY) but then relaunch it beside the learner whenever room
            # appears; stopped (and cancelled if requeued) here, the next round's collection restarts it. Also when a
            # restarted driver joins a running learner: the round's start just queued the DGX collection again
            self.stop_collections(remote=False)
            if not self.dgx_sft_running(round_dir, unit):
                self.recap_for_dgx(round_dir)
                self.launch_sft(round_dir, policy, activation_gb=self.a.sft_activation_gb, preflight_only=False)
            # the remote actors keep collecting the policy the round trains from for the whole SFT; a session the
            # keeper replaced (and served again) gets its collector back, one that was down at the round's start gets
            # one. A session fit for SFT may take the round over (the hand-back stops the unit, ending the wait)
            self.handed_back = None
            between = lambda: self.dgx_sft_between(round_dir, policy, unit)
            self.c.wait(unit, poll=120, between=between)
            while self.handed_back != round_dir.name and self.dgx_learner_pids(round_dir):
                # systemd and the ledger no longer answer for the unit but its learner still trains: keep joining it
                self.note(round=round_dir.name, dgx_learner_without_unit=self.dgx_learner_pids(round_dir), unit=unit)
                time.sleep(120)
                between()
            if self.handed_back != round_dir.name:
                break
            self.ensure_collection(policy)          # the DGX is free again: its standing collection resumes
        if not (factors / 'policy_manifest.json').exists():
            raise SystemExit(f'round {round_dir.name}: no factors after the SFT ({round_dir}/sft/train.log)')
        self.record_sft_speed(round_dir)
        self.record_round_speed(round_dir)
        return factors

    @staticmethod
    def dgx_learner_pids(round_dir):
        """Live DGX ``exp0_sft`` processes training this round (``--out-dir`` ``sft/``), read from /proc."""
        import pop_rl_actor
        return pop_rl_actor.local_learners(Path(round_dir) / 'sft')

    def dgx_sft_running(self, round_dir, unit):
        """Whether this round trains on the DGX: its unit runs or the ledger holds it, or (whatever those answer) an
        exp0_sft process trains its ``sft/``. Every path that would start a learner checks this first."""
        return self.c.present(unit) or bool(self.dgx_learner_pids(round_dir))

    def colab_sft(self, round_dir, policy):
        """The round on a Colab session (``run-sft``) until its factors exist, or until it falls back to the DGX: every
        answering session slower than the DGX, the learner failing 3 times there, or no session for
        ``--fallback-wait-minutes`` (recorded in ``round.json``; the DGX resumes the synced learner state)."""
        from natlang_neuralese.common.execution_profile import resolve_profile
        import pop_rl_actor
        factors = round_dir / 'sft' / 'factors'
        profile = resolve_profile('colab', hosts=self.colab_hosts)
        away_since, failures = None, 0
        while not (factors / 'policy_manifest.json').exists():
            self.standing_teacher()
            up = pop_rl_actor.reachable_hosts('colab', self.colab_hosts) if self.colab_hosts else []
            fit = {h: pop_rl_actor.colab_sft_fit(h) for h in up}
            if up and not any(ok for ok, _ in fit.values()):
                # an answering session slower than the DGX (an L4) keeps its actor; the round trains here now
                record_profile_change(round_dir, from_profile='colab', to_profile='dgx',
                                      reason='Colab session slower than the DGX for SFT: '
                                             + '; '.join(reason for _, reason in fit.values()))
                self.note(round=round_dir.name, takeover='dgx', unfit_sessions=up)
                return
            up = [h for h in up if fit[h][0]]
            if up and failures >= 3:       # the session answers but the learner keeps failing there
                record_profile_change(round_dir, from_profile='colab', to_profile='dgx',
                                      reason='the Colab learner failed 3 times')
                self.handback_blocked.add(round_dir.name)       # no hand-back to where it keeps failing
                self.note(round=round_dir.name, takeover='dgx', colab_failures=failures)
                return
            if up and self.dgx_sft_running(round_dir, f'{self.a.name}-{round_dir.name}-sft'):
                # a DGX learner of this round trains (whatever round.json's profile says): no second learner on the
                # session; train_sft joins it (only the verified hand-back moves a round off the DGX)
                self.note(round=round_dir.name, colab_refused='a DGX learner of this round is running',
                          pids=self.dgx_learner_pids(round_dir))
                return
            if up:
                away_since = None
                failures += 1
                self.launch_sft(round_dir, policy, activation_gb=profile.activation_gib, preflight_only=True)
                self.record_colab_placement(round_dir, up[0])
                self.c.session(up[0], profile, True)
                try:
                    with (round_dir / 'sft-colab.log').open('a') as handle:
                        subprocess.run(self.colab_command(round_dir, up[0], profile), env=self.c.env, stdout=handle,
                                       stderr=subprocess.STDOUT)
                finally:
                    self.c.session(up[0], profile, False)
                continue
            away_since = away_since or time.time()
            if time.time() - away_since >= 60 * self.a.fallback_wait_minutes:
                record_profile_change(round_dir, from_profile='colab', to_profile='dgx',
                                      reason=f'no Colab session for {self.a.fallback_wait_minutes} min')
                self.note(round=round_dir.name, takeover='dgx')
                return
            time.sleep(60)

    def dgx_sft_between(self, round_dir, policy, unit):
        """Every poll of a DGX SFT: the remote collectors, then the hand-back check (each failure logged apart)."""
        try:
            self.ensure_remote_collection(policy)
        except Exception as error:
            self.note(stage='wait', unit=unit, between_failed=repr(error)[:300])
        self.standing_teacher()
        self.sft_handback(round_dir, policy, unit)

    def expected_hours(self, report):
        """A built mix's learner hours on the DGX and on each GPU with a throughput (measured or default)."""
        forward = (report.get('mix') or {}).get('forward_tokens') or report.get('forward_tokens') or 0
        measured = self.measured_round_speed()
        names = {DGX_GPU, *(k for k, _ in DEFAULT_CHAIN_TOKENS_PER_S), *measured}
        return {n: round(forward / tps / 3600, 2) for n in sorted(names)
                for tps in [learner_throughput(n, measured)[0]] if tps}

    def record_colab_placement(self, round_dir, host):
        """``sft/colab-placement.json``: the session and GPU a round's Colab learner runs on (``record_round_speed``)."""
        try:
            import pop_rl_actor
            gpu = pop_rl_actor.colab_hardware(host).get('gpu_name')
            from natlang_neuralese.common.jsonio import write_json_durable
            (round_dir / 'sft').mkdir(parents=True, exist_ok=True)
            write_json_durable(round_dir / 'sft' / 'colab-placement.json',
                               {'host': host, 'gpu': gpu, 'at': round(time.time(), 1)})
        except Exception as error:
            self.note(round=round_dir.name, colab_placement_failed=repr(error)[:300])

    def record_round_speed(self, round_dir):
        """After a round: its learner throughput (chain tokens per second) per GPU -> ROUND_SPEED_FILE, the next
        rounds' placement sizing. The DGX's steps are its logs (``train.dgx-K.jsonl`` before a hand-back, ``train.jsonl``
        when it finished there); a round that finished on a session gives that session's GPU its ``train.jsonl`` steps
        from the hand-back's pushed step on. Needs 20 steps."""
        try:
            run = round_dir / 'sft'
            final = json.loads((round_dir / 'round.json').read_text()).get('profile')
            moves = json.loads((run / 'handback.json').read_text()) if (run / 'handback.json').exists() else []
            segments = [(DGX_GPU, p, 0) for p in sorted(run.glob('train.dgx-*.jsonl'))]
            if final == 'dgx':
                segments.append((DGX_GPU, run / 'train.jsonl', 0))
            elif (run / 'colab-placement.json').exists():
                gpu = json.loads((run / 'colab-placement.json').read_text()).get('gpu')
                if gpu:
                    segments.append((gpu, run / 'train.jsonl', int(moves[-1]['pushed_step']) if moves else 0))
            path = self.root / ROUND_SPEED_FILE
            table = json.loads(path.read_text()) if path.exists() else {}
            for gpu, log, first in segments:
                tps, steps = chain_tokens_per_s(log, first)
                if tps and steps >= 20:
                    table[gpu] = {'chain_tokens_per_s': tps, 'steps': steps, 'round': round_dir.name,
                                  'log': str(log), 'at': round(time.time(), 1)}
                    self.note(round=round_dir.name, round_speed={gpu: table[gpu]})
            from natlang_neuralese.common.jsonio import write_json_durable
            write_json_durable(path, table)
        except (OSError, ValueError, KeyError, IndexError) as error:
            self.note(round=round_dir.name, round_speed_failed=repr(error)[:300])

    def recap_command(self, round_dir, out):
        return self.c.data_py('sft_sources.py', 'recap', '--mix', round_dir / 'mix', '--out', out,
                              '--max-target-tokens', self.a.round_target_tokens,
                              *[x for b in self.prefer_unseen_for(round_dir) for x in ('--prefer-unseen', b)])

    def recap_for_dgx(self, round_dir):
        """A round sized for a Colab session that starts on the DGX (the session was lost or turned unfit before
        its learner began) trains the DGX round size instead: ``sft_sources recap`` bounds its mix to
        ``--round-target-tokens`` (unseen units first, every part scaled alike) into ``mix-dgx/``, and round.json
        (``chains_path``, ``chains_sha256``, ``episode_ids``, ``mix_dir``) and ``chains.sha256`` point at it; the units
        it leaves out stay unseen for the next round. A round whose learner already trained (a learner state or logged
        steps: a session lost mid-round) keeps its step list and resumes on the DGX (the hand-back moves it back when a
        fit session answers). ``--no-dgx-recap`` opts out."""
        path = round_dir / 'round.json'
        manifest = json.loads(path.read_text()) if path.exists() else {}
        size = manifest.get('size') or {}
        if (getattr(self.a, 'no_dgx_recap', False) or size.get('placement') != 'colab'
                or manifest.get('recap') or self.round_mix(round_dir) != round_dir / 'mix'
                or (round_dir / 'sft' / 'learner-state' / 'learner_manifest.json').exists()
                or dgx_sft_progress(round_dir / 'sft' / 'train.jsonl') is not None
                or chain_tokens_per_s(round_dir / 'sft' / 'train.jsonl')[1] > 0):
            return False
        out = round_dir / 'mix-dgx'
        self.c.ledger(f'{self.a.name}-{round_dir.name}-recap', 16, STAGE_PRIORITY, self.recap_command(round_dir, out),
                      round_dir / 'recap.log', repo=self.c.data_repo())
        report = json.loads((out / 'build.json').read_text())
        from natlang_neuralese.common.hashing import sha256_file_hex
        sha = sha256_file_hex(out / 'chains.pkl')
        tmp = round_dir / 'chains.sha256.partial'
        tmp.write_text(f'{sha}\n')
        os.replace(tmp, round_dir / 'chains.sha256')
        cap = report.get('target_cap') or {}
        update_round_manifest(round_dir, mix_dir=str(out), chains_path=str(out / 'chains.pkl'), chains_sha256=sha,
                              chains_bytes=(out / 'chains.pkl').stat().st_size,
                              episode_ids=sorted(report.get('admitted_episode_keys') or []),
                              recap={'at': round(time.time(), 1), 'max_target_tokens': self.a.round_target_tokens,
                                     'kept_weighted_target_tokens': cap.get('kept_weighted_target_tokens'),
                                     'kept_forward_tokens': cap.get('kept_forward_tokens'),
                                     'expected_hours': self.expected_hours(report)})
        self.note(round=round_dir.name, recap_for_dgx=str(out), kept_weighted=cap.get('kept_weighted_target_tokens'))
        return True

    def measured_sft_speed(self):
        """{GPU name: round SFT speed against the DGX} this loop measured on rounds that moved (``sft-speed.json``)."""
        path = self.root / 'sft-speed.json'
        return {k: v['speedup'] for k, v in json.loads(path.read_text()).items()} if path.exists() else {}

    def record_sft_speed(self, round_dir):
        """After a round that moved from the DGX to a session finished there: the session's median s/step on the
        steps it trained against the DGX's at the move -> ``sft-speed.json`` (the next move's estimate)."""
        path = round_dir / 'sft' / 'handback.json'
        if not path.exists():
            return
        try:
            move = json.loads(path.read_text())[-1]
            if round_dir.name in self.handback_blocked or json.loads(
                    (round_dir / 'round.json').read_text()).get('profile') != 'colab':
                return
            steps = [json.loads(line) for line in (round_dir / 'sft' / 'train.jsonl').read_text().splitlines()
                     if line.startswith('{')]
            colab = [float(r['step_seconds']) for r in steps
                     if r.get('step_seconds') and int(r.get('step', -1)) >= int(move['pushed_step'])]
            if len(colab) < 20 or not move.get('gpu'):
                return
            speed = self.root / 'sft-speed.json'
            table = json.loads(speed.read_text()) if speed.exists() else {}
            table[move['gpu']] = {'speedup': round(move['dgx_s_per_step'] / statistics.median(colab), 2),
                                  'round': round_dir.name, 'colab_s_per_step': round(statistics.median(colab), 2),
                                  'dgx_s_per_step': move['dgx_s_per_step'], 'steps': len(colab)}
            from natlang_neuralese.common.jsonio import write_json_durable
            write_json_durable(speed, table)
            self.note(round=round_dir.name, sft_speed=table[move['gpu']], gpu=move['gpu'])
        except (OSError, ValueError, KeyError) as error:
            self.note(round=round_dir.name, sft_speed_failed=repr(error)[:300])

    def sft_handback(self, round_dir, policy, unit):
        """Move a round training on the DGX to a Colab session fit for SFT (``colab_sft_fit``: an A100, not an L4)
        when the estimate favours it with a clear margin (``sft_handback_decision``): the DGX learner's latest save
        (FP32 factors, optimizer, RNG, batch cursor) goes to the VM and is verified there file by file
        (``push_learner_state``: Drive, the tunnel on failure), then the DGX unit stops; ``run-sft`` resumes that state
        with the same recipe and step list (``exp0_sft --resume`` refuses other chains or another schedule), and pushes
        a newer DGX save first if one landed meanwhile. Checked every ``--sft-handback-check-minutes``; at most
        SFT_HANDBACK_MAX moves a round; never for a round whose Colab learner failed. ``--no-sft-handback`` opts out.
        Sets ``handed_back`` (the unit's end then ends the wait)."""
        name = round_dir.name
        if (self.a.sft_profile != 'colab' or not self.colab_hosts or getattr(self.a, 'no_sft_handback', False)
                or name in self.handback_blocked or time.time() < self.handback_next.get(name, 0.0)):
            return
        now = time.time()
        self.handback_next[name] = now + 60 * self.a.sft_handback_check_minutes
        run = round_dir / 'sft'
        moves = json.loads((run / 'handback.json').read_text()) if (run / 'handback.json').exists() else []
        state = run / 'learner-state'
        progress = dgx_sft_progress(run / 'train.jsonl')
        if (len(moves) >= SFT_HANDBACK_MAX or progress is None or not (state / 'learner_manifest.json').exists()
                or not (run / 'launch.json').exists()):
            return
        import pop_rl_actor
        from natlang_neuralese.common.execution_profile import sft_speedup
        host = next((h for h in pop_rl_actor.reachable_hosts('colab', self.colab_hosts)
                     if pop_rl_actor.colab_sft_fit(h)[0]), None)
        if host is None:
            return
        hardware = pop_rl_actor.colab_hardware(host)
        speedup, source = sft_speedup(hardware, self.measured_sft_speed())
        manifest = json.loads((state / 'learner_manifest.json').read_text())
        resume = pop_rl_actor.state_step(manifest)
        cfg = pop_rl_actor.host_config('colab', host=host)
        run_name = f'{self.a.name}-{name}'
        launch = json.loads((run / 'launch.json').read_text())
        state_bytes = sum(p.stat().st_size for p in state.rglob('*') if p.is_file())
        via = getattr(self.a, 'sft_state_transfer', 'auto')
        move_s, parts = pop_rl_actor.sft_move_seconds(host, cfg, run_name=run_name, launch=launch,
                                                      state_bytes=state_bytes, via=via)
        decision = sft_handback_decision(steps_total=progress['of'], dgx_step=progress['step'], resume_step=resume,
                                         dgx_s_per_step=progress['s_per_step'], speedup=speedup, move_seconds=move_s,
                                         margin=self.a.sft_handback_margin,
                                         min_gain_seconds=60 * self.a.sft_handback_min_gain_minutes)
        self.note(round=name, handback_check=decision, host=host, gpu=hardware.get('gpu_name'),
                  speedup_source=source, move_parts=parts)
        if not decision['move']:
            return
        # push right after a DGX save: one landing during the push would be newer than the VM's copy (run-sft pushes
        # it again) and the steps since the pushed save are redone on the session
        recipe = shlex.split(launch.get('exp0_sft_args', ''))
        save_s = 60 * float(recipe[recipe.index('--state-save-minutes') + 1]) if '--state-save-minutes' in recipe \
            else 900.0
        age = now - float(manifest.get('saved_at') or 0)
        if age + parts['state'] + 120 > save_s:
            self.handback_next[name] = float(manifest.get('saved_at') or now) + save_s + 60
            self.note(round=name, handback_waits_for_save=round(self.handback_next[name] - now))
            return
        remote_run = pop_rl_actor.remote_sft_run(cfg, run_name)
        try:
            pop_rl_actor.colab_checkout(host, cfg, self.a.commit)     # run-sft's code, and the VM's drive_tier.py
            pushed = pop_rl_actor.push_learner_state(host, cfg, remote_run, state, run_name=run_name, via=via)
        except Exception as error:      # the DGX learner keeps training; the next check tries again
            self.handback_next[name] = now + 1800
            self.note(round=name, handback_failed=repr(error)[:300], host=host)
            return
        # verified on the VM: only now does the DGX learner stop
        self.c.stop_unit(unit)
        deadline = time.time() + 600
        while self.dgx_sft_running(round_dir, unit) and time.time() < deadline:
            time.sleep(5)
        if self.dgx_sft_running(round_dir, unit):
            self.note(round=name, handback_failed=f'{unit} did not stop', host=host)
            return
        subprocess.run(['systemctl', '--user', 'reset-failed', f'{unit}.service'], stderr=subprocess.DEVNULL)
        after = pop_rl_actor.state_step(json.loads((state / 'learner_manifest.json').read_text()))
        k = len(moves) + 1
        for log_name in ('train.jsonl', 'train.log'):   # run-sft's pulls replace them with the session's logs
            if (run / log_name).exists():
                shutil.copy2(run / log_name, run / f'{Path(log_name).stem}.dgx-{k}{Path(log_name).suffix}')
        entry = {'at': round(now, 1), 'host': host, 'gpu': hardware.get('gpu_name'), 'speedup': speedup,
                 'speedup_source': source, 'dgx_step': progress['step'], 'dgx_s_per_step': progress['s_per_step'],
                 'pushed_step': pushed['step'], 'dgx_state_step_at_stop': after, 'push': pushed,
                 'move_parts': parts, 'decision': decision, 'commit': self.a.commit}
        from natlang_neuralese.common.jsonio import write_json_durable
        write_json_durable(run / 'handback.json', [*moves, entry])
        record_profile_change(round_dir, from_profile='dgx', to_profile='colab',
                              reason=(f'hand-back to {host} ({hardware.get("gpu_name")}, {speedup}x the DGX, '
                                      f'{source}): DGX {decision["dgx_left_s"] / 3600:.1f} h left at '
                                      f'{progress["s_per_step"]} s/step vs {decision["colab_left_s"] / 3600:.1f} h there '
                                      f'(move {decision["move_s"]} s); learner state step {pushed["step"]} verified on '
                                      f'the VM via {pushed["via"]}'))
        self.note(round=name, handback='colab', host=host, pushed_step=pushed['step'], dgx_step=progress['step'],
                  dgx_state_step_at_stop=after, via=pushed['via'], gain_s=decision['gain_s'])
        self.handed_back = name

    def grpo_command(self, round_dir, policy):
        return [PY, str(self.c.pin / 'scripts/neuralese_data/grpo.py'), 'train', '--work', str(round_dir / 'grpo'),
                '--state-dir', str(round_dir / 'grpo-state'), '--checkpoint', policy['checkpoint'],
                '--checkpoint-sha256', policy['sha256'], '--max-steps', str(self.a.grpo_steps),
                '--held-every-hours', '0', '--teacher-attempts', str(self.a.teacher_attempts),
                '--export-factors-dir', str(round_dir / 'grpo-factors'), '--unit-prefix', f'{self.a.name}-{round_dir.name}',
                *self.a.grpo_extra.split()]

    def train_grpo(self, round_dir, policy):
        """Stage 2 of a GRPO round (the polish phase, §7): ``--grpo-steps`` on-policy steps from the current policy
        on the DGX (its own actor: the standing collection yields the GPU); the round's policy is its last step's
        factors over the policy it started from."""
        steps = sorted((round_dir / 'grpo-factors').glob('step-*')) if (round_dir / 'grpo-factors').exists() else []
        last = round_dir / 'grpo-factors' / f'step-{self.a.grpo_steps:05d}'
        if not (last / 'policy_manifest.json').exists():
            self.stop_collections()
            unit = f'{self.a.name}-{round_dir.name}-grpo'
            resume = ['--resume'] if (round_dir / 'grpo-state').exists() and steps else []
            self.c.ledger(unit, self.a.grpo_budget, STAGE_PRIORITY, [*self.grpo_command(round_dir, policy), *resume],
                          round_dir / 'grpo.log')
        if not (last / 'policy_manifest.json').exists():
            raise SystemExit(f'round {round_dir.name}: GRPO left no step-{self.a.grpo_steps} factors')
        return last

    # -- 3. publish ---------------------------------------------------------------------------------------------------
    def publish_command(self, round_dir, policy, factors):
        out = round_dir / 'policy'
        return self.c.py('expert_iteration.py', 'publish-round', '--base-checkpoint', policy['checkpoint'],
                         '--factors', factors, '--out', out, '--label', f'{self.a.name}-{round_dir.name}')

    def publish(self, round_dir, policy, factors):
        """Stage 3: the merged policy and every engine's artifact (``publish-round`` in a ledger unit, idempotent:
        ``policy/published.json`` is written last), then the remote actors rebound to it."""
        published = round_dir / 'policy' / 'published.json'
        if not published.exists():
            self.c.ledger(f'{self.a.name}-{round_dir.name}-publish', 24, STAGE_PRIORITY,
                          self.publish_command(round_dir, policy, factors), round_dir / 'publish.log')
        if not published.exists():
            raise SystemExit(f'round {round_dir.name}: publication failed ({round_dir}/publish.log)')
        return json.loads(published.read_text())

    def rebind_remote(self, published, factors, version):
        """Rebind every ``--rebind-actor`` that answers and does not serve the published policy yet (one already
        rebound is skipped); the record each rebound Colab actor keeps then names the published checkpoint
        (``pop_rl_actor.rebound_record``: a replaced session is served again at it). Every failure is raised after the
        others were tried."""
        import pop_rl_actor
        sha, errors = published['checkpoint_sha256'], []
        endpoints = {(host, front): endpoint for host, front, endpoint in self.remote_actors(include_pop=False)}
        for t in self.remote_targets():
            host, front, target = t['host'], t['front'], self.a.rebind_actor[t['index']]
            if t['pop'] or endpoint_alive(endpoints.get((host, front), front), sha):
                continue                    # Pop: pop_step serves the new policy
            try:
                pop_rl_actor.rebind_actor(front_url=front, policy_id=sha, version=version, factors_dir=factors,
                                          host=host or None, host_profile='colab',
                                          export_dir=None if host else published['sglang_export']['dir'],
                                          checkpoint=published['checkpoint'])
            except Exception as error:
                errors.append(f'{target}: {error!r}'[:300])
        if errors:
            raise RuntimeError('; '.join(errors))

    def ensure_remote_policy(self, policy, retry_seconds=1200):
        """The remote actors serve ``policy`` (the lineage head): a rebind that failed at the publication, a driver
        restarted after it, or a replaced session served again at an older policy is rebound here (at most every
        ``retry_seconds``). Only a round's publication can be rebound (its factors); the start policy is served as
        it is."""
        published = self.round_dir(policy['round_id']) / 'policy' / 'published.json' if policy.get('round_id') else None
        if not any(not t['pop'] for t in self.remote_targets()) or published is None or not published.exists():
            return
        if all(endpoint_alive(endpoint, policy['sha256']) for _h, _f, endpoint in self.remote_actors(include_pop=False)):
            return
        last = getattr(self, '_rebind_tried', {}).get(policy['sha256'], 0)
        if time.time() - last < retry_seconds:
            return
        self.__dict__.setdefault('_rebind_tried', {})[policy['sha256']] = time.time()
        record = json.loads(published.read_text())
        try:
            self.rebind_remote(record, record['factors_dir'], policy['version'])
            self.note(rebound=policy['sha256'], version=policy['version'])
        except Exception as error:
            self.note(rebind_failed=repr(error)[:300], policy=policy['sha256'])

    # -- held ---------------------------------------------------------------------------------------------------------
    def held_name(self, policy):
        return f'policy-{policy["sha256"][:12]}'

    def judge_measurement(self, job):
        """Measure the candidate and (when missing) the reference on the DGX series, compare and judge."""
        cand, ref = job['candidate'], job['reference']
        out = self.c.measure_held(self.held_name(cand), cand['checkpoint'])
        ref_out = self.c.held / self.held_name(ref)
        if not self.c.held_result_ok(ref_out):
            ref_out = self.c.measure_held(self.held_name(ref), ref['checkpoint'])

        def complete(*outs):
            for o in outs:
                if not json.loads((o / 'result.json').read_text()).get('complete'):
                    raise RuntimeError(f'held measurement incomplete: {o}')
        complete(out, ref_out)
        (self.root / 'verdicts').mkdir(exist_ok=True)
        compare_path = self.root / 'verdicts' / f'{cand["sha256"][:12]}-compare.json'
        # a pair compare refuses (different engine settings: 2026-11-24, a reused sft-v7 result of another series) is
        # no reason to fail every retry: the reference is measured again here, on this series (never reused), then if
        # need be the candidate too; a third refusal is a real fault and raises
        for again in ('reference', 'candidate', None):
            try:
                vs = self.c.compare(ref_out, out, compare_path)
                break
            except HeldPairRefused as error:
                self.note(held_pair_refused=str(error)[:300], candidate=cand['sha256'], reference=ref['sha256'],
                          remeasure=again)
                if again is None:
                    raise
                policy, target = (ref, ref_out) if again == 'reference' else (cand, out)
                self.c.set_held_aside(target, 'unpaired')
                fresh = self.c.measure_held(self.held_name(policy), policy['checkpoint'], reuse=False)
                complete(fresh)
                if again == 'reference':
                    ref_out = fresh
                else:
                    out = fresh
        verdict, adopt = judge(vs, self.a.success_pts, self.a.fail_pts)
        record = {'verdict': verdict, 'adopt': adopt, 'sha256': cand['sha256'], 'checkpoint': cand['checkpoint'],
                  'reference': ref, 'vs_start': {'delta': vs['primary']['delta'], 'p': vs['primary']['permutation_p'],
                                                 'ran_out_delta': vs.get('ran_out_rate_delta') or 0}}
        from natlang_neuralese.common.jsonio import write_json_durable
        write_json_durable(self.root / 'verdicts' / f'{cand["sha256"][:12]}.json', record)
        return record

    def held_loop(self):
        """The background held series: one measurement at a time, the newest due policy (training never waits). The
        held-v3 gate goes first; a held-v4 measurement (``held_v4_step``) runs only while no held-v3 job waits, and a
        held-v3 job that becomes due stops it (it resumes later from its episodes)."""
        while True:
            self.held_wake.wait(60)
            self.held_wake.clear()
            job = self.held_v3_pending()
            if not job:
                try:
                    self.held_v4_step()
                except BaseException as error:      # held-v4 never gates: note it and go on
                    self.note(held_v4_failed=repr(error)[:300])
                    time.sleep(300)
                continue
            path = self.root / 'verdicts' / f'{job["candidate"]["sha256"][:12]}.json'
            try:
                record = json.loads(path.read_text()) if path.exists() else self.judge_measurement(job)
            except BaseException as error:     # a failed held unit (SystemExit from Cycle.wait) is retried
                self.note(held_failed=job['candidate']['sha256'], error=repr(error)[:300])
                time.sleep(300)
                self.held_wake.set()
                continue
            self.held_done.add(job['candidate']['sha256'])
            with self.lock:
                self.verdicts.append(record)
            self.held_wake.set()            # a queued held-v4 job may run now

    def held_v3_pending(self):
        with self.lock:
            job = self.controller.state.get('measuring')
        return job if job and job['candidate']['sha256'] not in self.held_done else None

    # -- held-v4 (never gating) ---------------------------------------------------------------------------------------
    def held_v4_manifest(self):
        if getattr(self.a, 'held_v4_manifest', None):
            return Path(self.a.held_v4_manifest)
        from natlang_neuralese.common import data_registry
        return Path(data_registry.role_path('held-eval-semantic-fp'))

    def queue_held_v4(self, policy, reasons):
        if not reasons:
            return
        entry = self.held_v4.add(policy, reasons)
        self.note(held_v4_queued=policy['sha256'], reasons=entry['reasons'])
        self.held_wake.set()

    def held_v4_step(self):
        """Measure the newest queued held-v4 policy on Pop at ``--held-v4-samples`` (``pop_held_measure``) and record
        it beside the held-v3 verdict (``held-v4/hv4-<sha12>/record.json``)."""
        job = self.held_v4.next()
        if not job:
            return
        out = self.root / 'held-v4' / f'hv4-{job["sha256"][:12]}'
        with self.POP_LOCK:                 # a Pop collection serve finishes first; pop_held_measure stops its collector
            state = pop_held_measure(out, job['checkpoint'], pin=self.c.held_pin, samples=self.a.held_v4_samples,
                                     manifest=self.held_v4_manifest(),
                                     preempt=lambda: self.held_v3_pending() is not None)
        if state == 'preempted':
            self.note(held_v4_preempted=job['sha256'])
            return
        if state != 'complete':
            attempts = self.held_v4.attempt(job['sha256'])
            self.note(held_v4_incomplete=job['sha256'], attempts=attempts, out=str(out))
            if attempts >= 3:
                self.held_v4.finish(job['sha256'], 'failed', out=str(out))
            return
        record = self.held_v4_record(job, out)
        self.held_v4.finish(job['sha256'], 'measured', out=str(out))
        self.note(held_v4_measured=job['sha256'], success_rate=record['result'].get('success_rate'),
                  vs_previous=record.get('vs_previous'), reasons=job['reasons'])

    def held_v4_record(self, job, out):
        """The held-v4 result of one policy beside its held-v3 verdict (when it has one) and, when an earlier held-v4
        measurement of this loop exists, compared with the newest one (``held_eval compare``; telemetry only)."""
        from natlang_neuralese.common.jsonio import write_json_durable
        result = json.loads((out / 'result.json').read_text())
        keys = ('complete', 'ran', 'scored', 'successes', 'success_rate', 'ran_out_rate', 'loop_rate', 'timeout_rate',
                'samples_per_task', 'manifest_sha256')
        record = {'sha256': job['sha256'], 'checkpoint': job['checkpoint'], 'reasons': job['reasons'],
                  'manifest': str(self.held_v4_manifest()), 'result': {k: result.get(k) for k in keys}}
        verdict = self.root / 'verdicts' / f'{job["sha256"][:12]}.json'
        if verdict.exists():
            v = json.loads(verdict.read_text())
            record['held_v3'] = {'verdict': v.get('verdict'), 'vs_start': v.get('vs_start'), 'path': str(verdict)}
        earlier = sorted((p for p in (self.root / 'held-v4').glob('hv4-*/record.json') if p.parent != out),
                         key=lambda p: p.stat().st_mtime)
        if earlier:
            base = earlier[-1].parent
            compare_out = out / f'compare-vs-{base.name}.json'
            rc = subprocess.run([PY, '-m', 'natlang_neuralese.rl.held_eval', 'compare', str(base / 'result.json'),
                                 str(out / 'result.json'), '--out', str(compare_out)], env=self.c.held_env,
                                cwd=self.c.held_pin, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            if rc == 0 and compare_out.exists():
                vs = json.loads(compare_out.read_text())
                record['vs_previous'] = {'against': json.loads((base / 'record.json').read_text())['sha256'],
                                         'delta': vs['primary']['delta'], 'p': vs['primary']['permutation_p']}
        write_json_durable(out / 'record.json', record)
        return record

    # -- collection health (never halts) ------------------------------------------------------------------------------
    def health_collections(self, policies):
        """(current, previous, earlier) collection WORKs of a round's health report: the newest policy's, the previous
        policy's, and every other collection of this loop plus ``--health-baseline`` (the lineage's earlier ones)."""
        cur, prev = policies[-1], (policies[-2] if len(policies) > 1 else None)
        current = self.collections_of([cur])
        previous = self.collections_of([prev]) if prev else []
        others = [d for d in sorted((self.root / 'collect').glob('v*')) if (self.root / 'collect').exists()]
        earlier = [d for d in others if d not in current and d not in previous and (d / 'runs.jsonl').exists()]
        earlier += [Path(d) for d in getattr(self.a, 'health_baseline', []) or [] if Path(d) not in earlier]
        return current, previous, earlier

    def health_report(self, round_dir, policies):
        """``rN/health.json`` (``collection_health.report``) once per round, a one-line summary in the drive log, each
        flag in the ops alerts log; a flag on a semantic-FP construct queues held-v4 for the collection's policy."""
        out = round_dir / 'health.json'
        if getattr(self.a, 'no_health_report', False) or out.exists():
            return None
        import collection_health
        try:
            current, previous, earlier = self.health_collections(policies)
            if not current:
                return None
            health = collection_health.report(current, previous, earlier, cache_dir=self.root / 'health-cache',
                                              label=f'{self.a.name} {round_dir.name}')
            health['policy'] = policies[-1]['sha256']
            collection_health.write(health, out)
            self.note(round=round_dir.name, health=collection_health.summary_line(health), health_flags=len(health['flags']),
                      health_sfp=health['sfp_flagged'])
            lines = collection_health.alert_lines(health, f'{self.a.name} {round_dir.name}')
            if lines and self.a.alerts_log:
                with open(self.a.alerts_log, 'a') as handle:
                    handle.write(''.join(line + '\n' for line in lines))
            self.queue_held_v4(policies[-1], held_v4_reasons(self.held_v4_cadence, health=health))
            return health
        except Exception as error:          # the report is telemetry: it never stops a round
            self.note(round=round_dir.name, health_failed=repr(error)[:300])
            return None

    def apply_verdicts(self):
        """Apply finished verdicts in the main loop: adoption, rollback, phase switch; a failure leaves a bug-hunt note
        (``bug-hunt-<sha12>.json``) for the supervisor; a success promotes the policy to lfm-latest."""
        with self.lock:
            pending, self.verdicts = self.verdicts, []
        for path in sorted((self.root / 'verdicts').glob('*.json')) if (self.root / 'verdicts').exists() else ():
            if not path.name.endswith('-compare.json'):
                v = json.loads(path.read_text())
                if v['sha256'] not in self.controller.state['applied_verdicts'] and v not in pending:
                    pending.append(v)
        for record in pending:
            event = self.controller.record_verdict(record)
            if event.get('action') != 'already applied':
                self.queue_held_v4({'sha256': record['sha256'], 'checkpoint': record['checkpoint']},
                                   held_v4_reasons(self.held_v4_cadence, verdict=record))
            self.note(verdict=record['verdict'], policy=record['sha256'], **{k: v for k, v in event.items()
                                                                             if k in ('action', 'switch_phase', 'held_every')})
            if record['verdict'] == 'failure' and 'bug_hunt' in event:
                (self.root / f'bug-hunt-{record["sha256"][:12]}.json').write_text(json.dumps({**record, **event}, indent=1) + '\n')
            if record['verdict'] == 'success' and event.get('action') == 'adopt':
                promote_policy(self.c, record['checkpoint'], record['sha256'], label=f'{self.a.name}',
                               why=f'{self.a.name}: held success {json.dumps(record["vs_start"])}',
                               unit=f'{self.a.name}-promote-{record["sha256"][:8]}')

    # -- the loop -----------------------------------------------------------------------------------------------------
    def round_dir(self, round_id):
        return self.root / round_id

    def run(self):
        """Drive up to ``--max-rounds`` (0: continuous) rounds."""
        self.c.require_data_readers()
        self.record_reader_sources()
        self.c.pin_stage()
        threading.Thread(target=self.held_loop, daemon=True).start()
        self.held_wake.set()
        if self.pop_target() is not None:
            threading.Thread(target=self.pop_loop, daemon=True).start()
        if getattr(self.a, 'resume_after_hunt', False):
            self.controller.resume_after_hunt()
        threading.Thread(target=self.teacher_loop, daemon=True).start()
        completed = 0
        while self.a.max_rounds <= 0 or completed < self.a.max_rounds:
            self.standing_teacher()
            with self.lock:
                self.apply_verdicts()
            if self.controller.state.get('halted'):
                return self.halt()
            cur, phase = self.controller.current, self.controller.phase
            rid = self.controller.begin_round(cur['sha256'])
            round_dir = self.round_dir(rid)
            (round_dir / 'stages').mkdir(parents=True, exist_ok=True)
            self.note(round=rid, phase=phase, policy=cur['sha256'], version=cur['version'])
            if phase == 'sft':
                self.ensure_collection(cur)
                self.ensure_remote_collection(cur)
                if self.prep_thread is not None:
                    self.prep_thread.join()
                prepared = json.loads((round_dir / 'round.json').read_text()) \
                    if (round_dir / 'stages' / 'prepared.done').exists() else None
                if prepared and not all(self.controller.in_lineage(s) for s in
                                        (prepared.get('frontier_policy'), prepared.get('previous_policy_sha')) if s):
                    shutil.rmtree(round_dir)                # prepared ahead from a lineage a rollback discarded
                    (round_dir / 'stages').mkdir(parents=True)
                self.prepare(round_dir, self.lineage_policies())
                reason = self.rebuild_requested(round_dir)
                if reason:
                    self.rebuild_round(round_dir, self.lineage_policies(), reason)
                self.standing_teacher()
                self.health_report(round_dir, self.lineage_policies())
                nxt = f'r{int(self.controller.state["round_counter"]) + 1:04d}'
                self.prepare_ahead(self.round_dir(nxt), self.lineage_policies())
                update_round_manifest(round_dir, trained_from=cur['sha256'], trained_from_version=cur['version'])
                factors = self.train_sft(round_dir, cur)
                mix = self.round_mix(round_dir)
            else:
                self.stop_collections()
                factors = self.train_grpo(round_dir, cur)
                mix = None
            published = self.publish(round_dir, cur, factors)
            with self.lock:
                self.apply_verdicts()
                if self.controller.current['sha256'] != cur['sha256']:
                    self.controller.discard_round(rid, f'the lineage moved to {self.controller.current["sha256"][:12]} '
                                                       'while it trained')
                    (round_dir / 'discarded.json').write_text(json.dumps({'time': time.time()}) + '\n')
                    continue
                new = self.controller.record_round_published(rid, published['checkpoint'], published['checkpoint_sha256'],
                                                             base_sha256=cur['sha256'], mix_dir=mix)
                if self.controller.held_due() and not self.controller.state.get('measuring'):
                    self.controller.start_measurement()
                    self.held_wake.set()
                self.queue_held_v4(new, held_v4_reasons(self.held_v4_cadence,
                                                        round_number=len(self.controller.state['trained_rounds'])))
            # the old policy's remote collectors stop before the rebind (their episodes then would check a changed actor:
            # each would be abandoned); the next round's ensure_remote_collection starts the new policy's
            for unit in self.remote_units(self.collection_unit(cur)):
                if self.c.present(unit):
                    self.c.stop_unit(unit)
            try:
                self.rebind_remote(published, factors, new['version'])
            except Exception as error:      # the remote actor keeps serving the old policy; its collection drops it
                self.note(rebind_failed=repr(error)[:300])
            self.note(round=rid, published=new['sha256'], version=new['version'])
            completed += 1
        return 0

    def halt(self):
        """A failure verdict ended the experiment (AGENTS.md): no new round; the standing collection serves the
        lineage head (the reference the lineage rolled back to) so the machines keep collecting its outcomes while the
        bug hunt runs; ``--resume-after-hunt`` starts rounds again."""
        halted = self.controller.state['halted']
        cur = self.controller.current
        self.ensure_collection(cur)
        message = (f'{self.a.name}: held failure of {halted["sha256"][:12]} vs {halted["reference"]["sha256"][:12]} '
                   f'({json.dumps(halted.get("vs_start"))}): the experiment ends at this eval (AGENTS.md); the lineage '
                   f'is back at {cur["sha256"][:12]}, whose collection keeps running ({self.collection_unit(cur)}). '
                   f'Bug hunt: {self.root}/bug-hunt-{halted["sha256"][:12]}.json; rerun with --resume-after-hunt '
                   'when it is closed.')
        self.note(halted=halted['sha256'], message=message)
        print(message, file=sys.stderr, flush=True)
        return 3

    def reader_sources(self):
        from natlang_neuralese.common.code_provenance import code_provenance
        return {role: {'path': str(repo), **code_provenance(repo)} for role, repo in (
            ('controller', REPO), ('runtime', self.c.pin),
            ('fill', self.c.fill_repo()), ('data', self.c.data_repo()))}

    def record_reader_sources(self):
        from natlang_neuralese.common.jsonio import write_json_durable
        record = {'schema': 'natlang.async-reader-sources/1', 'sources': self.reader_sources(),
                  'runtime_commit': self.a.commit,
                  'minimum_data_reader_commit': getattr(self.a, 'minimum_data_reader_commit', None), 'time': time.time()}
        write_json_durable(self.root / 'reader-sources.json', record)
        self.note(reader_sources=record['sources'])

    def plan(self):
        """``--dry-run``: the commands one round of the current state would run, without running them."""
        cur = self.controller.current
        rd = self.round_dir('r-dry-run')
        frontier, lines, build = self.prepare_plan(rd, self.lineage_policies())
        self.c.collect_dir = self.collect_dir(cur)
        collect = self.c.collect_command(self.collection_unit(cur), cur['checkpoint'], cur['sha256'], self.a.slots)
        collect[collect.index('--tasks') + 1] = str(self.collect_dir(cur) / 'tasks.json')
        remote = []
        for t in self.remote_targets():
            if not t['host']:
                continue
            endpoint = self.target_endpoint(t)
            unit, _work, argv, budget = self.remote_collect_argv(cur, t, endpoint)
            remote.append({'unit': unit, 'host': t['host'], 'front': t['front'], 'endpoint': endpoint,
                           'slots': t['slots'], 'pop': t['pop'], 'serves_policy': endpoint_alive(endpoint, cur['sha256']),
                           'budget_gb': budget, 'collect': list(map(str, argv))})
        plan = {'phase': self.controller.phase, 'policy': cur, 'pin': str(self.c.pin),
                'reader_sources': self.reader_sources(),
                'select': self.select_command(self.collect_dir(cur) / 'tasks.json'), 'collect': collect,
                'remote_collect': remote, 'sft_dgx': {'activation_gb': self.a.sft_activation_gb,
                                                      'budget_gb': self.a.sft_budget},
                'frontier': frontier, 'data': lines, 'build': build,
                'sft_launch': ['/bin/sh', str(OPS / 'grpo-v4' / 'launch-sft.sh'), str(self.c.pin)],
                'colab_sft': self.colab_command(rd, (self.colab_hosts or ['HOST'])[0], argparse.Namespace(state_sync_minutes=15)),
                'grpo': self.grpo_command(rd, cur), 'publish': self.publish_command(rd, cur, rd / 'sft' / 'factors'),
                'held': {'every': self.controller.state['held_every'], 'since': self.controller.state['since_held'],
                         'reference': self.controller.state['reference_policy']}}
        print(json.dumps(plan, indent=1, default=str))
        return plan


def publish_round(args):
    """``publish-round`` (a ledger unit of the async driver): merge shipped factors into their base checkpoint
    (``merge_factors_to_checkpoint``) and write every engine's artifact (``publish_merged_policy``); OUT/published.json
    last."""
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.export.native_hf import publish_merged_policy
    from natlang_neuralese.train.learner_state import merge_factors_to_checkpoint
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'published.json').unlink(missing_ok=True)
    ckpt = out / 'checkpoint.pt'
    merged = merge_factors_to_checkpoint(args.base_checkpoint, args.factors, ckpt)
    published = publish_merged_policy(ckpt, base=args.base, sglang_export_dir=out / 'sglang-hf', gguf_dir=out / 'gguf',
                                      label=args.label, why=f'async round {args.label}')
    (out / 'sha256.txt').write_text(merged['checkpoint_sha256'] + '\n')
    # factors_dir: a Colab learner merges the next round's start from it on the VM instead of receiving 3.4 GB
    # (pop_rl_actor.ensure_remote_checkpoint); code: the commit whose merge wrote these bytes, which the VM merges
    # with (pop_rl_actor.remote_merge_code: a VM checkout at another commit wrote other bytes, r0002 2026-11-24)
    from natlang_neuralese.common.code_provenance import code_provenance
    write_json_durable(out / 'published.json', {**published, 'merge': merged, 'code': code_provenance(),
                                                'factors_dir': str(Path(args.factors).resolve())})
    log({'event': 'round published', 'checkpoint_sha256': merged['checkpoint_sha256'], 'out': str(out)})


def async_drive(args):
    driver = AsyncRoundDriver(args)
    if args.dry_run:
        driver.plan()
        return 0
    return driver.run()


def cycle(args):
    Cycle(args).run()


def main():
    # one pipe and log per collection slot, plus the actor's: not the unit's soft 1024 (common/open_files.py)
    from natlang_neuralese.common.open_files import raise_open_files_limit
    raise_open_files_limit()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def pool_flags(p):
        p.add_argument('--pool', action='append', help='default: the registry role grpo-pool')
        p.add_argument('--held-manifest', help='default: grpo.default_held() (held-v3)')
        p.add_argument('--exclude-family-regex', default='(?!)', help="families left out (default: none, as grpo-v5)")
        p.add_argument('--seed', type=int, default=20261120)

    def memetic_flags(p):
        p.add_argument('--memetic-share', type=float, default=MEMETIC_SHARE,
                       help='share of new collection groups that are memetic "improve program P" episodes (default '
                            'MEMETIC_SHARE; 0 opts out); their verified own successes enter the SFT mix')
        p.add_argument('--memetic-layout', choices=('staged', 'inline'), default='staged',
                       help="a memetic group's layout (grpo --memetic-layout; rl/memetic_workspace.py bind_layout): "
                            'staged (default) prepares it as a collector stage and fits its nested runs in the '
                            "episode's wall clock; inline is the layout before 2026-11-23")
        p.add_argument('--memetic-fit-quantile', type=float, default=0.9,
                       help="grpo --memetic-fit-quantile: a memetic group only for a program whose family's ordinary "
                            'episodes in the collection finish within its nested wall clock at this quantile (0: every '
                            'program)')
        p.add_argument('--memetic-nested-slots', action=argparse.BooleanOptionalAction, default=True,
                       help='grpo --memetic-nested-slots: memetic nested runs in flight take collector slots')
    s = sub.add_parser('select')
    pool_flags(s)
    s.add_argument('--out', required=True)
    s.add_argument('--base-rates', default=str(OPS / 'exp1-rft' / 'base-pass-rates.json'))
    s.add_argument('--rates-groups', action='append', default=[], metavar='GLOB',
                   help='group directories (GRPO runs, earlier collections) whose student outcomes order the tasks')
    s.add_argument('--frontier', action='append', default=[], help='earlier frontier files (newest last)')
    s.add_argument('--include-solved', action='store_true')
    s.add_argument('--limit', type=int, default=0)
    c = sub.add_parser('collect')
    pool_flags(c)
    c.add_argument('--work', required=True)
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--checkpoint-sha256')
    c.add_argument('--tasks', required=True, help='JSON list of task ids in collection order (select)')
    c.add_argument('--k', type=int, default=4, help='samples per task (one group)')
    c.add_argument('--hours', type=float, default=6., help='no new group after this many hours (running ones finish)')
    c.add_argument('--slots', type=int, default=128)
    c.add_argument('--dynamic-sampling-max-spread', type=float, default=0.1,
                   help="grpo's early stop of a group whose first wave agrees (0: all k samples always)")
    c.add_argument('--actor-port', type=int, default=8470)
    c.add_argument('--actor', action='append', default=[],
                   help='running actor front URL[=SLOTS] (repeatable for sharded collection across Colab G4 sessions)')
    c.add_argument('--actor-arg', action='append', default=[])
    c.add_argument('--neuralese-variants', help="grpo's --neuralese-variants cases.jsonl (default: none)")
    c.add_argument('--neuralese-variant-share', type=float, default=0.5)
    c.add_argument('--collect-weight-dtype', choices=('fp8', 'bfloat16'), default=COLLECT_WEIGHT_DTYPE,
                   help='SGLang linear weight dtype for collection actors (default COLLECT_WEIGHT_DTYPE)')
    c.add_argument('--teacher-attempts', type=int, default=TEACHER_ATTEMPTS,
                   help='JOLT+ guided attempts for failing groups (default TEACHER_ATTEMPTS; 0: none)')
    c.add_argument('--teacher-max-successes', type=int, default=0)
    memetic_flags(c)
    c.add_argument('--memetic-tasks', help="memetic task rows (default: the registry role grpo-memetic-tasks)")
    c.add_argument('--memetic-seed-collection', action='append',
                   help="grpo --memetic-seed-collection (default: the sibling collections of --work)")
    c.add_argument('--memetic-max-groups', type=int, default=MEMETIC_MAX_GROUPS,
                   help='memetic groups running at once (each sample runs ~30 nested rollouts beside its slot)')
    c.add_argument('--unit-prefix', default='ei-collect')
    c.add_argument('--unpinned', action='store_true')
    c.add_argument('--held-priority-dir', default=str(HELD_PRIORITY_DIR),
                   help='live DGX held measurements claim files here; the collection yields slots to them')
    c.add_argument('--held-priority-share', type=float, default=HELD_PRIORITY_SHARE,
                   help='share of the slots the collection keeps while a held measurement runs (1: no priority)')
    c.add_argument('--no-held-priority', action='store_true')
    f = sub.add_parser('frontier')
    f.add_argument('--out', required=True)
    f.add_argument('--collect', action='append', required=True)
    f.add_argument('--policy', required=True, help='sha256 of the policy every collection served')
    f.add_argument('--previous-policy', help='adjacent previous policy sha256 recorded for top-priority own tier (async-train §3)')
    f.add_argument('--allow-adjacent-policies', action='store_true',
                   help='allow collections whose runs.jsonl spans --policy and --previous-policy (standing rebind)')
    f.add_argument('--held-manifest')
    f.add_argument('--solved', type=float, default=0.75)
    f.add_argument('--min-scored', type=int, default=2)
    y = sub.add_parser('cycle')
    y.add_argument('--dir', required=True, help='the cycle directory, e.g. ops/ei/sft-v7')
    y.add_argument('--name', required=True, help='unit prefix, e.g. sft-v7')
    y.add_argument('--start', required=True, help='start policy (native checkpoint)')
    y.add_argument('--start-sha')
    y.add_argument('--commit', required=True)
    y.add_argument('--pin')
    y.add_argument('--sft-work', default=str(root('data_nvme') / 'ei-sources'), help='sft_sources WORK shared by the cycles')
    y.add_argument('--seed-work', action='append', default=[], help='earlier sft_sources WORKs: verdicts and extractions')
    y.add_argument('--live-work', action='append', default=[], metavar='WORK',
                   help='a growing WORK of observed-live teacher episodes (live_campaign_ingest.py; async-drive\'s flag)')
    y.add_argument('--select-args', default='', help='more select flags (e.g. "--limit 3000")')
    y.add_argument('--k', type=int, default=4)
    y.add_argument('--neuralese-variants', default=str(REPO / 'runs/natlang-rl-neuralese-variants-20261113/cases.jsonl'),
                   help='Neuralese-declaring variants of pool tasks (registry natlang-rl-neuralese-variants-20261113, '
                   '1,381 rows) that collection runs in --neuralese-variant-share of a task\'s groups')
    y.add_argument('--neuralese-variant-share', type=float, default=0.5)
    y.add_argument('--no-neuralese-variants', action='store_true')
    y.add_argument('--collect-hours', type=float, default=6.)
    y.add_argument('--collect-weight-dtype', choices=('fp8', 'bfloat16'), default=COLLECT_WEIGHT_DTYPE,
                   help='SGLang linear weight dtype for collection actors (default COLLECT_WEIGHT_DTYPE; held stays bfloat16)')
    y.add_argument('--teacher-attempts', type=int, default=TEACHER_ATTEMPTS,
                   help='JOLT+ guided attempts for failing groups in collection (default TEACHER_ATTEMPTS; 0: none)')
    memetic_flags(y)
    y.add_argument('--slots', type=int, default=128)
    y.add_argument('--kv-gb', type=int, default=12)
    y.add_argument('--fill', help='filled_reasoning.py OUT directory: generate on the collect actor, build into the mix')
    y.add_argument('--fill-inflight', type=int, default=32,
                   help='filled-reasoning draws kept in flight on the collect actor, which runs this many requests '
                   'more than the collection has slots (twice that when it has spare slots)')
    y.add_argument('--no-fill-window', action='store_true',
                   help='no dedicated fill actor between the collection and the SFT')
    y.add_argument('--fill-window-inflight', type=int, default=128, help="the fill window actor's running requests")
    y.add_argument('--fill-window-workers', type=int, default=96, help='filled-reasoning episodes in the fill window')
    y.add_argument('--fill-actor-port', type=int, default=8570)
    y.add_argument('--no-pop-fill', action='store_true',
                   help='leave Pop idle between held evals (default: it serves the collected policy for fills)')
    y.add_argument('--pop-fill-running', type=int, default=32, help="Pop fill server's running requests")
    y.add_argument('--pop-wait-hours', type=float, default=1.0,
                   help='a measured cycle waits this long for an unreachable Pop, then is adopted unmeasured (deferred) '
                   'and the next cycle is measured')
    y.add_argument('--pop-fill-stop-at-data', action='store_true',
                   help='stop the Pop fill at the data stage (default: it draws on through build and SFT for the next '
                   "cycle's mix until a Pop serve stops it)")
    y.add_argument('--fill-repo', help='checkout whose filled_reasoning.py runs (default the pin; fills are data '
                   'tooling outside the held scoring series)')
    y.add_argument('--minimum-data-reader-commit', help='required ancestor of both immutable data/fill reader pins; runtime scoring pin unchanged')
    y.add_argument('--data-repo', help="checkout whose sft_sources.py builds the mix (default the pin; data preparation "
                   'outside the held scoring series, e.g. a faster build that keeps chains byte-identical)')
    y.add_argument('--speculative-after', help='previous cycle DIR whose SFT candidate is --start: collect while Pop '
                   'judges it, then train from what its verdict chooses (the gate stage)')
    y.add_argument('--fallback-prefer-unseen', action='append', default=[],
                   help='--prefer-unseen builds if the previous candidate is rejected (what its start trained on)')
    y.add_argument('--rejected-collection', choices=('auto', 'keep', 'recollect'), default='auto',
                   help="a rejected candidate's collection: auto recollects after 'failure', keeps it otherwise")
    y.add_argument('--no-pipeline', action='store_true', help='do not launch the next cycle when the SFT checkpoint exists')
    y.add_argument('--next-name', help='the next cycle (default: the trailing number + 1)')
    y.add_argument('--next-commit', default='origin/main', help='the next cycle\'s pin (resolved when it launches)')
    y.add_argument('--next-seed-work', action='append', default=[], help='seed WORKs the next cycle adds')
    y.add_argument('--no-held-reuse', action='store_true',
                   help="measure base and start on Pop even when an earlier cycle measured them in this series")
    y.add_argument('--strict-held-reuse', action='store_true', help='reuse a held result only from the same scoring series, host and samples per task (default: any series)')
    y.add_argument('--held-every', type=int, default=2,
                   help='held-eval every Nth cycle\'s candidate; the others are adopted unmeasured (1 = every cycle)')
    y.add_argument('--held-every-max', type=int, default=8,
                   help='the interval a run of measured successes doubles up to')
    y.add_argument('--since-held', type=int, default=0, help='unmeasured cycles before this one (set by the pipeline)')
    y.add_argument('--held-pin', help="pin whose harness and scoring run the held evals (default this cycle's pin); "
                   'carried to the next cycle, so the series stays fixed and the start reuses the previous candidate')
    y.add_argument('--held-base', action='store_true',
                   help='measure the base on this pin too (default: only the start; vs-base uses the latest base of '
                        'another series when this one has none)')
    y.add_argument('--no-teacher-campaign', action='store_true')
    y.add_argument('--replay-workers', type=int, default=8)
    y.add_argument('--teacher-budget', type=int, default=180_000_000)
    y.add_argument('--own-share', type=float, default=0.35)
    y.add_argument('--aux', action='append', default=[])
    y.add_argument('--no-own-neuralese-graph', action='store_true',
                   help="leave the WORK's own Neuralese successes (consumer calls included) out of the mix")
    y.add_argument('--own-neuralese-graph-share', type=float, default=0.05,
                   help='loss share cap of the own Neuralese graph part (an aux part; duplicates of other parts train once)')
    y.add_argument('--prefer-unseen', action='append', default=[])
    y.add_argument('--lr', default='5e-5', help='LoRA SFT learning rate (lfm3opti.md §5 default 5e-5 for r=128)')
    y.add_argument('--sft-budget', type=int, default=100,
                   help='SFT ledger GB (GB10 LoRA default 100: the 56 GiB activation default peaks ~85 GB, measured model '
                        '2026-11-22: CUDA 5.2 GiB + 1.2 x --activation-gib, host ~6-15 GB; leaves >= 15 GB DGX headroom)')
    y.add_argument('--sft-activation-gb', type=int, default=56,
                   help='exp0_sft --activation-gib on DGX (56 GiB LoRA: ~19.6k-token packs unrecomputed at the measured 127 '
                        'KB/token/layer; G4 via ExecutionProfile)')
    y.add_argument('--sft-extra', default='')
    y.add_argument('--collect-profile', choices=('dgx', 'colab', 'colab_g4'), default='dgx',
                   help='execution profile for collection (colab, old name colab_g4: shards across the up Colab '
                        'sessions; 30-min DGX fallback)')
    y.add_argument('--sft-profile', choices=('dgx', 'colab', 'colab_g4'), default='dgx',
                   help='execution profile for SFT (colab, old name colab_g4: remote with 15-min state sync; 30-min '
                        'DGX fallback)')
    y.add_argument('--held-profile', choices=('dgx', 'pop'), default='dgx',
                   help='the held series (async-train §1): dgx (default: the pinned held engine on the DGX, BF16 weights, '
                        '4 samples, held episodes before collection) or pop (legacy, 2 samples)')
    y.add_argument('--held-series-host', help="the host of --held-pin's series (set by the pipeline; absent: pop)")
    y.add_argument('--held-slots', type=int, default=HELD_SLOTS, help='collector slots of a DGX held measurement')
    y.add_argument('--gate-wait-hours', type=float, default=0.,
                   help='how long the gate waits for the previous verdict before adopting the candidate as collected '
                        '(0: never; training does not wait for a held eval)')
    y.add_argument('--colab-hosts', default=None,
                   help='comma-separated Colab G4 SSH host aliases (overrides NATLANG_COLAB_HOSTS)')
    y.add_argument('--fallback-wait-minutes', type=float, default=30.0,
                   help='minutes to wait for unreachable remote hosts before falling back to DGX (default 30)')
    y.add_argument('--cu-per-hour', type=float, default=13.0,
                   help='observed Colab G4 compute-unit burn rate per session-hour for cycle-cost.json')
    y.add_argument('--held-samples', type=int, default=None,
                   help='samples per held task (default: the held profile\'s, 4 on the DGX, 2 on Pop)')
    y.add_argument('--resume-after-hunt', action='store_true',
                   help='a held failure halted the pipeline (AGENTS.md: the experiment ends at that eval); the hunt is '
                        'closed: treat the failure as a rejection again (train the previous start)')
    y.add_argument('--success-pts', type=float, default=2.)
    y.add_argument('--fail-pts', type=float, default=3.)
    h = sub.add_parser('pop-held')
    h.add_argument('--dir', required=True)
    h.add_argument('--pin', required=True)
    h.add_argument('--start', required=True)
    h.add_argument('--held-samples', type=int, default=2)
    h.add_argument('--after', help='previous cycle DIR: wait for its verdict; its next_start is the start')
    h.add_argument('--no-held-reuse', action='store_true')
    h.add_argument('--strict-held-reuse', action='store_true')
    h.add_argument('--held-base', action='store_true')
    h.add_argument('--held-profile', choices=('pop', 'dgx'), default='pop')
    h.add_argument('--held-slots', type=int, default=HELD_SLOTS)
    h.add_argument('--name', help='the cycle name (unit prefix; default: DIR name)')
    d = sub.add_parser('dgx-held', help='one DGX held measurement inside its ledger unit (Cycle.dgx_held)')
    d.add_argument('--out', required=True)
    d.add_argument('--checkpoint', required=True)
    d.add_argument('--policy-id', required=True)
    d.add_argument('--samples', type=int, default=4)
    d.add_argument('--slots', type=int, default=HELD_SLOTS)
    d.add_argument('--kv-gb', type=float, default=HELD_KV_GB)
    d.add_argument('--max-running', type=int, default=HELD_MAX_RUNNING)
    d.add_argument('--held-pin', required=True)
    d.add_argument('--manifest', help='default: grpo.default_held() (held-v3)')
    d.add_argument('--label', required=True)
    d.add_argument('--unit', required=True, help='the ledger unit (names the held priority claim)')
    ad = sub.add_parser('async-drive', help='continuous expert iteration in rounds (async-train §2-§5, §7.4; '
                        'AsyncRoundDriver)')
    ad.add_argument('--root', required=True, help='the loop directory (phase-state.json, collect/, rounds r0001...)')
    ad.add_argument('--name', default='async-ei', help='unit prefix')
    ad.add_argument('--start', required=True, help='initial policy (native checkpoint)')
    ad.add_argument('--start-sha')
    ad.add_argument('--reference', help='the last held-adopted policy when --start is unmeasured (default: --start): '
                    'held verdicts compare against it and a rollback returns to it')
    ad.add_argument('--reference-sha')
    ad.add_argument('--strict-held-reuse', action='store_true', help='reuse a held result only from the same scoring series, host and samples per task (default: any series)')
    ad.add_argument('--commit', required=True, help='the pinned commit every round runs (collection, data, SFT, held)')
    ad.add_argument('--pin', help='its pin (default OPS/ei/pin-<commit12>, created when missing)')
    ad.add_argument('--held-pin', help='pin of the held series (default --pin)')
    ad.add_argument('--sft-work', default=str(root('data_nvme') / 'ei-sources'))
    ad.add_argument('--seed-work', action='append', default=[], help='earlier sft_sources WORKs (fill sources)')
    ad.add_argument('--live-work', action='append', default=[], metavar='WORK',
                    help='a growing WORK of observed-live teacher episodes (live_campaign_ingest.py): each round links '
                         'its extractions into --sft-work and builds against its holdout registry; the fill takes '
                         'its episodes first and rescans it')
    ad.add_argument('--sft-profile', choices=('colab', 'dgx'), default='dgx',
                    help='where rounds train: colab (a reachable Colab session, DGX takeover after '
                         '--fallback-wait-minutes) or dgx')
    ad.add_argument('--colab-hosts', default=None, help='Colab SSH host aliases (overrides NATLANG_COLAB_HOSTS)')
    ad.add_argument('--rebind-actor', action='append', default=[], metavar='HOST@FRONT_URL[=SLOTS]',
                    help='a remote collection actor (its VM-local front URL), rebound to each published policy by '
                         'factors and collected from through the host\'s forwarded port (repeatable); =SLOTS sets its '
                         'collector slots. On Pop\'s host (POP_HOST@http://POP_IP:18531=32) a standing Pop actor: served '
                         'with each new policy and yielding to Pop held measurements')
    ad.add_argument('--remote-slots', type=int, default=64, help='collector slots per remote actor (default)')
    ad.add_argument('--pop-max-running', type=int, default=32,
                    help='SGLang running cap of the Pop collection actor (pop_rl_actor.POP_COLLECT_RUNNING)')
    ad.add_argument('--round-target-tokens', type=int, default=ROUND_TARGET_TOKENS,
                    help='a round mix\'s weighted target tokens (sft_sources build --max-target-tokens: every part '
                         'scaled alike, unseen episodes first); 0: the uncapped build (r0001: 37.4M, 1247 steps)')
    ad.add_argument('--colab-round-hours', type=float, default=COLAB_ROUND_HOURS,
                    help='a round prepared while a Colab session fit for SFT answers is sized to this many hours of '
                         'its learner (forward tokens at its measured throughput, round-speed.json; the target cap and '
                         'teacher budget scaled alike, never below the DGX round); 0: every round is the DGX size')
    ad.add_argument('--no-fallow22-aux', action='store_true',
                    help='leave the fallow-22 reference-program fan-in part (FALLOW22_AUX, share 0.01) out of the mix')
    ad.add_argument('--no-dgx-recap', action='store_true',
                    help='a round sized for a session that starts on the DGX trains its whole mix there (default: '
                         'recapped to --round-target-tokens first, sft_sources recap)')
    ad.add_argument('--no-prefer-inflight', action='store_true',
                    help='a round prepared while another trains does not prefer unseen over that round\'s mix')
    ad.add_argument('--unfilled-live-teacher', choices=('auto', 'drop', 'train'), default=None,
                    help='sft_sources build --unfilled-live-teacher (default auto: kept, teacher reasoning as context)')
    ad.add_argument('--fallback-wait-minutes', type=float, default=30.0)
    ad.add_argument('--no-sft-handback', action='store_true',
                    help='a round training on the DGX (a takeover) stays there; by default it moves to a Colab session '
                         'fit for SFT (colab_sft_fit) from its latest verified learner state when the estimate favours it')
    ad.add_argument('--sft-handback-margin', type=float, default=0.25,
                    help='a hand-back needs the session to finish within (1 - margin) of the DGX\'s remaining time')
    ad.add_argument('--sft-handback-min-gain-minutes', type=float, default=60.0,
                    help='... and at least this much sooner')
    ad.add_argument('--sft-handback-check-minutes', type=float, default=10.0)
    ad.add_argument('--sft-state-transfer', choices=('auto', 'drive', 'tunnel'), default='auto',
                    help='how learner state goes to a session (auto: Google Drive, the tunnel on failure)')
    ad.add_argument('--lr', default='5e-5', help='LoRA SFT learning rate (lfm3opti.md §5: 5e-5 at r=128)')
    ad.add_argument('--sft-budget', type=int, default=80,
                    help='DGX SFT ledger GB (a fixed claim): r0001 measured a 63.9 GB cgroup peak at activation 56 '
                         '(CUDA 55.5); 100 starved the remote collector, the Colab fill client, teacher and next prepare')
    ad.add_argument('--sft-activation-gb', type=int, default=56, help='DGX exp0_sft --activation-gb (cycle default)')
    ad.add_argument('--sft-extra', default='')
    ad.add_argument('--teacher-budget', type=int, default=180_000_000)
    ad.add_argument('--own-share', type=float, default=0.35)
    ad.add_argument('--aux', action='append', default=[])
    ad.add_argument('--own-neuralese-graph-share', type=float, default=0.05)
    ad.add_argument('--replay-workers', type=int, default=8)
    ad.add_argument('--k', type=int, default=4)
    ad.add_argument('--slots', type=int, default=128)
    ad.add_argument('--kv-gb', type=int, default=12)
    ad.add_argument('--collect-hours', type=float, default=72.,
                    help="a standing collection's horizon (the next publication stops it first)")
    ad.add_argument('--snapshot-min-hours', type=float, default=1.0,
                    help="a round's snapshot waits until the newest policy's collection ran this long")
    ad.add_argument('--select-args', default='')
    ad.add_argument('--neuralese-variants', default=str(REPO / 'runs/natlang-rl-neuralese-variants-20261113/cases.jsonl'))
    ad.add_argument('--neuralese-variant-share', type=float, default=0.5)
    ad.add_argument('--no-neuralese-variants', action='store_true')
    ad.add_argument('--collect-weight-dtype', choices=('fp8', 'bfloat16'), default=COLLECT_WEIGHT_DTYPE)
    ad.add_argument('--teacher-attempts', type=int, default=TEACHER_ATTEMPTS)
    memetic_flags(ad)
    ad.add_argument('--fill', help='filled_reasoning.py OUT directory: fills on the standing collection actor')
    ad.add_argument('--minimum-data-reader-commit', help='required ancestor of both immutable data/fill reader pins; checked at controller start and each SFT boundary')
    ad.add_argument('--data-repo', help='immutable data-reader checkout (default: controller source); actor/learner --pin is unchanged')
    ad.add_argument('--fill-repo', help='immutable fill-reader checkout (default: --data-repo or controller source); use --pin path to opt out')
    ad.add_argument('--fill-inflight', type=int, default=32)
    ad.add_argument('--no-teacher-campaign', action='store_true')
    ad.add_argument('--teacher-check-minutes', type=float, default=10.0,
                    help='how often the standing teacher campaign is checked (relaunched from the newest frontier when '
                         'its unit is idle) beside the waits')
    ad.add_argument('--rebuild-round', action='append', default=[], metavar='RNNNN',
                    help='rebuild this prepared round\'s mix (only the build step, its recorded inputs) under the '
                         'current build code when it starts, once per build commit; stages/rebuild.request in the '
                         'round does the same for a running driver',)
    ad.add_argument('--grpo-steps', type=int, default=20, help='GRPO steps per round in the polish phase')
    ad.add_argument('--grpo-budget', type=int, default=100, help='GRPO ledger GB (learner + its actor)')
    ad.add_argument('--grpo-extra', default='')
    ad.add_argument('--held-every', type=int, default=2, help='held-eval every Nth round (1: every round)')
    ad.add_argument('--held-profile', choices=('pop', 'dgx'), default='pop',
                    help='the held-v3 gate series: pop (default: Pop is otherwise idle; 2 samples) or dgx (4 samples, '
                         'yields collection slots)')
    ad.add_argument('--held-v4-cadence', default=HELD_V4_CADENCE,
                    help='when held-v4 (registry role held-eval-semantic-fp) measures a policy on Pop, never gating and '
                         'only while Pop has no held-v3 job: every=N published rounds, significant (a held-v3 verdict '
                         'with p < 0.05), health-sfp (the collection health report flags a semantic-FP construct); off')
    ad.add_argument('--held-v4-samples', type=int, default=HELD_V4_SAMPLES)
    ad.add_argument('--held-v4-manifest', help='default: the registry role held-eval-semantic-fp (ops/held-v4)')
    ad.add_argument('--health-baseline', action='append', default=[], metavar='COLLECT_WORK',
                    help="the lineage's earlier collections (outside this loop) each round's health report compares with")
    ad.add_argument('--no-health-report', action='store_true', help='no per-round collection health report')
    ad.add_argument('--alerts-log', default=str(OPS / 'storage-alerts.log'),
                    help='the ops alerts log the heartbeat reads (health flags are appended; empty: none)')
    ad.add_argument('--held-every-max', type=int, default=8)
    ad.add_argument('--held-samples', type=int, default=None, help="default: the held profile's (Pop 2, DGX 4)")
    ad.add_argument('--held-slots', type=int, default=HELD_SLOTS)
    ad.add_argument('--success-pts', type=float, default=2.)
    ad.add_argument('--fail-pts', type=float, default=3.)
    ad.add_argument('--initial-phase', choices=('sft', 'grpo'), default='sft')
    ad.add_argument('--prefer-unseen', action='append', default=[])
    ad.add_argument('--max-rounds', type=int, default=0, help='stop after N rounds (0: continuous)')
    ad.add_argument('--dry-run', action='store_true', help="print one round's commands for the current state and exit")
    ad.add_argument('--resume-after-hunt', action='store_true',
                    help='a held failure halted the loop (AGENTS.md: the experiment ends at that eval); the hunt is '
                         'closed: start rounds again from the lineage head')
    pr = sub.add_parser('publish-round', help='merge shipped factors and write every engine (a ledger unit of async-drive)')
    pr.add_argument('--base-checkpoint', required=True)
    pr.add_argument('--factors', required=True)
    pr.add_argument('--out', required=True)
    pr.add_argument('--base', help='the HF base snapshot (default: the checkpoint\'s own)')
    pr.add_argument('--label', required=True)
    args = parser.parse_args()
    rc = {'select': select, 'collect': collect, 'frontier': frontier, 'cycle': cycle, 'pop-held': pop_held,
          'dgx-held': dgx_held, 'async-drive': async_drive, 'publish-round': publish_round}[args.command](args)
    if rc:
        sys.exit(rc)


if __name__ == '__main__':
    main()
