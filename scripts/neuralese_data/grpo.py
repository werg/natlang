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
* learner (``natlang_neuralese.train.lfm3_learner``, lfm3opti.md §5): ``lora+rows`` at r=32 on a frozen BF16 base, FP32
  factors, rows and AdamW state, BF16 compute (``lfm3_fast``). Every
  model call is (exact actor prompt ids, sampled ids) (``lower_rollout_turns.episode_turns``; calls of one invocation chain
  into one sequence as in ``exp0_sft.chains_of``). A consumer call's imported Neuralese state is a differentiable
  pass-through: computed in the consumer's own forward from the producer document the episode recorded (the text write's
  ``[produce_open] + text ids``, ``lfm3_fast_train.passthrough_plan``), so the consumer's loss reaches the weights that
  computed the state (``--no-neuralese-passthrough``: the same state detached; a block without a recorded document is the
  actor's stored state, ``lfm3_fast_train.import_tensors``). The producer's own sampled tokens train as its own calls. Loss on sampled tokens only:
  ``-min(rho, C) * A * logp + beta * k2(ref || policy)``, ``rho = exp(logp_learner - logp_actor)`` (truncated importance
  sampling against the actor's own sampling logprobs), ``A`` the group-normalized episode reward on every call of the
  episode, the reference the start policy (B = 0: the base with its start rows, ``reference_snapshot``). ``--loss-norm
  token`` (default): one sum over the step's units over their number (units: sampled tokens; a sampled decision counts
  as one average call of its episode, at most 8 tokens); ``episode``: each trained episode weighs the same, its terms
  averaged over its units. The step's LR scales with its groups with reward spread, each counted by the units it trains
  (``lr_data_scale``, ``spread_data``). A tail token the actor sampled below ``--tail-logp-floor`` gets no
  policy term in an episode with a positive advantage. Sampling is top-p (``--top-p``, ``common.sampling.RL_TOP_P``): the
  policy term's ``logp`` is the sampled token's log-probability under the learner's own nucleus
  (``lfm3_fast_train.linear_nll_nucleus``; the sampled token always in the set), while ``rho`` and the KL term use the
  untruncated log-probabilities (the actor records the untruncated one; DECISIONS 2026-11-13 for the reasoning).
* weight sync (``WeightSync``): after each step the policy is snapshotted (GPU clones of factors and rows), merged in
  FP32 and rounded once to BF16 into one of two page-locked tmpfs safetensors files in the actor export's layout, loaded
  in place (SGLang ``pause_generation`` retract -> ``update_weights_from_disk``), bound on every actor route (the front
  reloads its native engine) and attested from measurements (SGLang's own ``/model_info`` and the file's digest, the
  native model's tensor identity, a sentinel sequence's log-probabilities against the learner's), in a background
  thread. A failure serves the last good version again and retries once; a second failure stops the run.
  Episodes keep running across a sync; each token's actor logprob is from the weights that sampled it.
* checkpoints: a pruned native checkpoint (servable by ``serve_native_checkpoint``) every ``--save-every`` steps, written
  from the bytes the actor is about to serve, and the learner state (``train.learner_state``: factors, rows, AdamW, RNG,
  consumed-group ledger, base digest) every ``--state-every`` steps. Keeps the last two native checkpoints plus every
  held-evaluated one, and the last learner state.
* memory (the unit's budget covers GPU and host alike on the GB10): the frozen base, ~0.5 GB of trainable state, two
  ~3.4 GB tmpfs weight files and the actor; imported states are read per sequence, and ``train.jsonl`` logs each learner
  phase's CUDA peak and this process's host memory (``cuda_peak_by_phase_gib``). The two phases trade the GB10's unified
  memory (``--actor-release``, default on): at the start of each learner phase the actor is paused (retract) and SGLang
  releases its KV cache, weights and CUDA graphs (``WeightSync.release``, the actor runs with ``--sglang-memory-saver``);
  the step's publish resumes them and reloads the weights before generation continues.
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
* reasoning RL against fixed teacher actions (``--reasoning-rl-weight``, off by default since 2026-11-22, async-train
  §8: its default objective is the privileged twin's KL term, which the plan keeps off because JOLT+ guided successes
  already carry that signal as SFT targets and GRPO groups; 1.0 opts in): a walk along
  teacher trajectories (JOLT+ teacher successes, ``TeacherDemos``) samples the student's own think blocks before each
  teacher action and carries the best forward as history. Each block trains token by token on the forward KL from the
  policy's privileged twin, which sees the teacher action (``--reasoning-rl-objective twin-fkl``, default since
  2026-11-20); ``sequence`` instead rewards it by the teacher actions' likelihood after it (return-to-go), with search,
  rewrites and carried SD chains. The teacher actions train as a ratio-free cross-entropy (``ReasoningRL``,
  ``rrl_episodes``; the comment above ``RRL_ROLE``).

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

from natlang_neuralese.common import data_registry  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402
from natlang_neuralese.common.sampling import RL_TOP_P, check_top_p, sampling_law_of  # noqa: E402
from natlang_neuralese.rl.held_eval import EPISODE_MODEL_CALLS, LEGACY_SAMPLING  # noqa: E402

BASE = str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')


# The training data a run selects by default is a set of registry roles (training/corpora entries'
# `roles`, pinned with the code; `scripts/data_registry.py role NAME` lists one), not a list in this file:
#   grpo-pool           the task pools, in role order (the first occurrence of a task id wins). A task-supply round is
#                       admitted with `scripts/data_registry.py promote ID --role grpo-pool`.
#   grpo-memetic-tasks  the memetic task set (--memetic-tasks)
#   grpo-sd-teacher     sft_sources WORKs of verified teacher successes (TeacherDemos), in order
#   held-eval           the held-out manifest: held-v3 (plans/neuralese/HELD_V3_20261113.md); its exclusion also covers
#                       held-v2 (manifest `also_excludes`), so no v2 held task enters training either
def default_pools():
    return data_registry.role_paths('grpo-pool')


def default_memetic_tasks():
    return data_registry.role_path('grpo-memetic-tasks')


def default_sd_teacher_sources():
    return data_registry.role_paths('grpo-sd-teacher')


def default_held():
    return data_registry.role_path('held-eval')


# Start-policy calls scored after every step (behaviour_probe.py): v1's 48 reasoning calls, and from 2026-11-19 (v2) 24
# end-of-turn rows from the start policy's decision readouts
BEHAVIOUR_PROBE = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'grpo-v2', 'behaviour-probe-v2.json')
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
    """A finished held measurement in ``out`` (its directory, or the result file itself): a cut ``held_eval measure``
    also writes result.json (complete false), and a loop that took that for done would compare against a 20-episode
    start policy (grpo-v2-run1, 2026-11-13)."""
    path = Path(out) if str(out).endswith('.json') else Path(out) / 'result.json'
    try:
        return bool(json.loads(path.read_text()).get('complete'))
    except (OSError, ValueError):
        return False


def held_priors(base_out, held_root, out):
    """Every complete held result of a run but ``out``'s (the start policy's first): the dispatch order's evidence of
    each task's episode wall time (held_eval.Dispatch)."""
    done = [Path(d) / 'result.json' for d in [base_out, *sorted(Path(held_root).glob('step-*'))]
            if Path(d).resolve() != Path(out).resolve() and held_complete(d)]
    return list(dict.fromkeys(str(path) for path in done))


def unit_active(unit):
    """A user unit, or its ledger admission waiter (``memory_ledger run --detach``), is active."""
    return any(subprocess.run(['systemctl', '--user', 'is-active', '-q', f'{name}.service']).returncode == 0
               for name in (unit, unit + '-waiter'))


def pop_eval_inflight(directories, active=None):
    """(directory, record) of a detached Pop evaluation (``pop-eval.json``, no ``pop-eval-done.json``) whose unit still
    runs, else None."""
    active = active or unit_active
    for directory in directories:
        try:
            record = json.loads((Path(directory) / 'pop-eval.json').read_text())
        except (OSError, ValueError):
            continue
        if not (Path(directory) / 'pop-eval-done.json').exists() and active(record['unit']):
            return Path(directory), record
    return None


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


COMPACT_GROUP_SCHEMA = 'natlang.compact-scored-group/2'   # /1 dropped every sampled token


def parse_extra_actor(piece, default_slots=1):
    """One ``--extra-actor URL[=SLOTS][@HOST]`` spec -> ``{'front', 'slots', 'host', 'sglang'}``: ``HOST`` is the ssh
    host of a remote actor (Colab G4, Pop; its SGLang server listens on its own loopback, port + 1000), none for an
    actor on this machine (SGLang on the front's host, port + 1000)."""
    from urllib.parse import urlparse
    piece = piece.strip()
    host = None
    head, at, tail = piece.rpartition('@')
    if at and '/' not in tail:
        piece, host = head, tail
    count = default_slots
    url, eq, value = piece.rpartition('=')
    if eq:
        piece, count = url, int(value)
    front = piece.rstrip('/')
    parsed = urlparse(front)
    port = (parsed.port or 80) + 1000
    sglang = f'http://127.0.0.1:{port}' if host else f'{parsed.scheme}://{parsed.hostname}:{port}'
    return {'front': front, 'slots': max(1, int(count)), 'host': host, 'sglang': sglang}


def parse_extra_actors(extra_actors=(), default_slots=1):
    return [parse_extra_actor(piece, default_slots) for raw in extra_actors or () for piece in str(raw).split(',')
            if piece.strip()]


def parse_actor_specs(primary_actor, extra_actors=(), default_slots=1):
    """Parse primary ``--actor`` plus repeatable ``--extra-actor URL[=SLOTS][@HOST]`` specs into ``[(url, slots), ...]``
    (async-train §7.1: multi-actor GRPO collection across DGX + Colab G4 BF16 actors)."""
    specs = [(actor['front'], actor['slots']) for actor in parse_extra_actors(extra_actors, default_slots)]
    if primary_actor:
        primary = str(primary_actor).rstrip('/')
        extra_slots = sum(c for _, c in specs)
        primary_slots = max(1, int(default_slots) - extra_slots) if specs else max(1, int(default_slots))
        specs.insert(0, (primary, primary_slots))
    return specs


def extra_actor_start_versions(extra_actors, start_sha, step, info=None):
    """Each ``--extra-actor``'s version before its first publication: 0 for an actor attested on the start policy at a
    fresh start; otherwise unknown (a resume, another policy, unreachable): far behind, so its slots idle until the
    step's publication lands there (``Collector.actor_lagging``)."""
    def read(front):
        with urllib.request.urlopen(front + '/v1/neuralese/info', timeout=60) as response:
            return json.loads(response.read())
    versions = {}
    for actor in extra_actors:
        served = None
        try:
            served = ((info or read)(actor['front']).get('rl_policy_binding') or {}).get('checkpoint_sha256')
        except (OSError, ValueError):
            pass
        versions[actor['front']] = 0 if step == 0 and served == start_sha else -(10 ** 9)
    return versions


def rho_stale(role, stale):
    """A chain's ``rho_by_staleness`` bucket: its episode's staleness for the students' own episodes; None for a JOLT+
    teacher attempt, which trains in student form without its guidance, so its ratio measures no actor drift (replay,
    SD and reasoning-RL chains carry none either)."""
    return None if role == 'teacher' else stale


def rho_by_staleness_summary(buckets):
    """Per-staleness importance-ratio telemetry for ``train.jsonl`` (async-train §7.2): ``{Δv: {tokens, rho_mean,
    abs_log_rho_mean, rho_capped_fraction}}``."""
    out = {}
    for stale in sorted(buckets):
        b = buckets[stale]
        tokens = int(b.get('tokens', 0))
        if tokens <= 0:
            continue
        out[str(stale)] = {
            'tokens': tokens,
            'rho_mean': round(float(b['rho_sum']) / tokens, 4),
            'abs_log_rho_mean': round(float(b['abs_log_rho_sum']) / tokens, 5),
            'rho_capped_fraction': round(float(b['rho_capped']) / tokens, 5),
        }
    return out or None


def packed_block_ids(episodes):
    """Every Neuralese state id the learner may read from the actor's store for these lowered episodes: each call's
    ``imports`` and the imports inside its producer documents (``sources``)."""
    ids = set()
    for episode in episodes:
        for row in [*episode.get('rows', ()), *episode.get('decisions', ())]:
            ids.update(block for _, block in row.get('imports') or ())
            for document in (row.get('sources') or {}).values():
                ids.update(block for _, block in document.get('imports') or ())
    return sorted(ids)


def pack_scored_group(collector, group, out_path):
    """Pack one scored group as exactly what the learner reads (async-train §7.1: scored groups ship from the DGX
    collectors to a remote learner): ``load_group``'s output (task id, lowered episodes with their int32 prompt and
    sampled ids, float32 actor log-probabilities, decisions, rewards with graded/memetic parts, roles; unscored,
    truncated and leaked JOLT+ attempts already dropped and guidance stripped, as on the DGX learner), the group's
    versions, the lowering's counters, and the bytes of every stored Neuralese state those calls import. Lossless:
    ``unpack_scored_group`` returns the same structures, byte for byte."""
    import pickle
    import zstandard as zstd
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    generated, read = collector.generated, collector.episodes
    task_id, episodes, versions = load_group(collector, group, stats)
    states = getattr(collector, 'states', None)
    blocks = {}
    for block in packed_block_ids(episodes):
        path = states.path(block) if states is not None else None
        if path is not None and path.exists():
            blocks[block] = path.read_bytes()
    payload = {'schema': COMPACT_GROUP_SCHEMA, 'group': group, 'task_id': task_id, 'versions': list(versions),
               'episodes': episodes, 'stats': dict(stats), 'generated': collector.generated - generated,
               'episodes_read': collector.episodes - read, 'blocks': blocks}
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zstd.ZstdCompressor(level=3).compress(raw)
    tmp = out_path.with_name(out_path.name + '.partial')
    tmp.write_bytes(compressed)
    os.replace(tmp, out_path)
    return {'path': str(out_path), 'bytes': len(compressed), 'raw_bytes': len(raw), 'episodes': len(episodes),
            'blocks': len(blocks)}


def unpack_scored_group(archive_path, store_dir=None):
    """The payload of a ``pack_scored_group`` archive; its Neuralese states are written into ``store_dir`` (the
    receiving learner's ``--actor-store``: ``blocks/<id[5:7]>/<id>.safetensors``, as ``NativeStates`` reads them)."""
    import pickle
    import zstandard as zstd
    payload = pickle.loads(zstd.ZstdDecompressor().decompress(Path(archive_path).read_bytes()))
    if payload.get('schema') != COMPACT_GROUP_SCHEMA:
        raise ValueError(f'{archive_path}: unexpected compact group schema {payload.get("schema")!r}')
    if payload['blocks'] and store_dir is None:
        raise ValueError(f'{archive_path} carries {len(payload["blocks"])} Neuralese states: give the store to write them to')
    if store_dir is not None:
        from lower_rollout_turns import NativeStates
        states = NativeStates(store_dir)
        for block, data in payload['blocks'].items():
            path = states.path(block)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + '.partial')
                partial.write_bytes(data)
                os.replace(partial, path)
    return payload


# ---- rollouts --------------------------------------------------------------------------------------------------------
class Collector:
    """Work-stealing slots over an endless stream of task groups; finished groups are scored and queued for the learner."""

    def __init__(self, args, tasks, work, policy_sha, runtime, first_group=0, cursor=0):
        self.args, self.tasks, self.work, self.policy_sha, self.runtime = args, tasks, work, policy_sha, runtime
        self.actor_endpoints = parse_actor_specs(getattr(args, 'actor', None),
                                                 getattr(args, 'extra_actor', ()),
                                                 getattr(args, 'slots', 1))
        self.lock = threading.Lock()
        self.pending = collections.deque()          # entries not yet taken by a slot
        self.groups = {}                            # group -> {'dir', 'size', 'done', 'versions', 'task_id'}
def memetic_rows_held_free(rows, args):
    """The memetic rows none of whose support or query worlds (their episode specs) is a held task or shares a held
    split group (``held_eval.exclusion_ids`` of ``--held-manifest`` and every ``--exclude-manifest``): the set was built
    against the held set of its day, and a later held set must keep its tasks out of the worlds the policy trains on."""
    from natlang_neuralese.rl.held_eval import exclusion_ids
    manifests = [m for m in [getattr(args, 'held_manifest', None), *(getattr(args, 'exclude_manifest', None) or [])] if m]
    if not manifests or not rows:
        return rows
    worlds = {}
    for row in rows:
        spec = json.loads(Path(row['semantics']['world']['task']).read_text())
        worlds[row['id']] = spec['support'] + spec['query']
    cases = [case for cases in worlds.values() for case in cases]
    held = set()
    for manifest in manifests:
        held |= exclusion_ids(manifest, cases)
    kept = [row for row in rows if not any(case['id'] in held for case in worlds[row['id']])]
    if len(kept) < len(rows):
        print(json.dumps({'event': 'memetic rows with held worlds left out', 'rows': len(rows) - len(kept)}), flush=True)
    return kept


        self.ready = queue.Queue()                  # finished, scored groups
        self.outstanding = 0                        # groups created and not yet handed to the learner
        self.next_group, self.cursor = first_group, cursor
        self.version = 0                            # the policy version the actor serves now
        # the version each extra actor (``--extra-actor``) serves: set by its own publication (``WeightSync``); an
        # actor whose publication failed keeps its last version and its slots idle once it lags past --max-staleness
        self.actor_versions = {}
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
            self.memetic_rows = memetic_rows_held_free(list(read_jsonl(args.memetic_tasks)), args)
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
        memetic = bool(self.memetic_rows) and self._memetic_open() and self.rng.random() < a.memetic_share
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
                              'created': time.time(), 'wave': wave, 'held_back': entries[wave:] if wave else [],
                              **({'memetic': True} if memetic else {})}
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
    def _memetic_open(self):
        """Whether a new group may be a memetic one (always here; expert-iteration collection bounds them)."""
        return True

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

    def endpoint_for_slot(self, index=0):
        """Return the actor endpoint assigned to slot ``index`` (multi-actor GRPO, async-train §7.1)."""
        endpoints = getattr(self, 'actor_endpoints', None)
        if not endpoints:
            return self.args.actor
        expanded = [url for url, count in endpoints for _ in range(max(1, int(count)))]
        return expanded[index % len(expanded)]

    def served_version(self, url):
        """The policy version ``url`` serves now: the primary actor's is ``version``, an extra actor's its own."""
        return self.actor_versions.get(url, self.version) if url != self.args.actor else self.version

    def actor_lagging(self, url):
        """An extra actor more than --max-staleness versions behind (its publication failed): start nothing there."""
        return self.version - self.served_version(url) > getattr(self.args, 'max_staleness', 2)

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
            actor_url = self.endpoint_for_slot(index)
            if self.actor_lagging(actor_url):
                with self.lock:
                    self.pending.appendleft(item)
                time.sleep(5)
                continue
            entry = {**entry, 'rl_endpoint': actor_url}
            queue_file, journal = workers / f'{key}.queue.jsonl', workers / f'{key}.journal.jsonl'
            queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
            started = self.served_version(actor_url)     # the version this episode's actor serves at its start
            command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue_file), str(journal),
                       '--runtime', str(self.runtime), '--server', actor_url,
                       '--chat-completions-url', actor_url + '/v1/chat/completions', '--model-id', 'natlang-rl',
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
        if getattr(a, 'teacher_attempts', 0) <= 0 or 'teachers' in info or info.get('memetic'):
            return 0   # a memetic episode's reward is graded against its anchor: no privileged attempts of it
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
            plan_command=jolt.plan_command(a.actor) if a.teacher_plan == 'nl' else None, trajectories=self.trajectories,
            answers=getattr(a, 'guidance_answers', 'withhold'))
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


def teacher_guidance_leak(collector, info, row):
    """``rl/jolt.py guidance_leak`` of one JOLT+ teacher attempt (its reasons, [] when it may train), or [] with
    ``--guidance-leaks train``. The world's answer values come from the group's cases.jsonl; text-fallback guidance is
    read from the privileged prompt tokens with the served tokenizer (loaded once)."""
    args = getattr(collector, 'args', None)
    if getattr(args, 'guidance_leaks', 'drop') != 'drop':
        return []
    from natlang_neuralese.rl.jolt import guidance_leak
    from natlang_neuralese.rl.privileged_context import answer_leaves
    answers = []
    cases = info['dir'] / 'cases.jsonl'
    if cases.exists():
        for line in cases.read_text().splitlines():
            case = json.loads(line) if line.strip() else {}
            if case.get('id') == row.get('task_id'):
                answers = answer_leaves(case)
    decode = getattr(collector, 'guidance_decode', None)
    if decode is None and getattr(args, 'base', None):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
            decode = collector.guidance_decode = lambda ids: tokenizer.decode(ids)
        except Exception:   # no tokenizer: writer texts alone are checked
            decode = collector.guidance_decode = None
    return guidance_leak(row, answers=answers, decode=decode)


def load_group(collector, group, stats):
    """(task_id, [episode dicts with 'reward', 'rows'], versions) of one scored group."""
    from natlang_neuralese.common.jsonio import read_jsonl
    from lower_rollout_turns import decision_turns, episode_turns
    from natlang_neuralese.rl.jolt import student_form
    from natlang_neuralese.rl.rollout_math import budget_stopped_rollout, truncated_rollout
    info = collector.groups[group]
    episodes = []
    for row in read_jsonl(info['dir'] / 'rollouts.jsonl'):
        role = row.get('jolt_role', 'student')
        if role == 'teacher':   # JOLT+: the guided run's actions under the unguided prompt (rl/jolt.py student_form)
            leak = teacher_guidance_leak(collector, info, row)
            if leak:
                # its replies quote or name the guidance, or the guidance carried the answer: in student form they would
                # train text the call never saw; it trains nowhere (episode term, SD replay, reasoning RL)
                stats['teacher_guidance_leak'] += 1
                for reason in leak:
                    stats['teacher_guidance_leak_' + reason] += 1
                continue
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
        if truncated_rollout(row) and getattr(getattr(collector, 'args', None), 'truncated_episodes', 'drop') == 'drop':
            # a reply cut by a token cap (finish_reason 'length') ends the episode for the cap, not for what the policy
            # did: its reward would push the cut reasoning down ("stop sooner"). It is neither ranked nor trained
            # (overlong filtering, DECISIONS 2026-11-19). The 40-call budget is a task outcome and still scores 0.
            stats['episodes_truncated'] += 1
            continue
        if budget_stopped_rollout(row) and getattr(getattr(collector, 'args', None), 'budget_stop_episodes', 'train') == 'drop':
            # --budget-stop-episodes drop (opt-in): the 40-call budget treated as a length cap (overlong filtering)
            stats['episodes_budget_stopped'] += 1
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
    registry_files(archive, [target])
    return target.stat().st_size


def registry_files(directory, paths):
    """Record files a run added to one of its growing datasets (rollout archive, step chains). Bookkeeping never stops
    training: a failure is printed as an event."""
    try:
        kind = 'grpo-step-chains' if Path(directory).name == 'step-chains' else 'grpo-rollouts'
        data_registry.record_files(data_registry.dataset_id(directory, kind), paths, directory)
    except Exception as error:
        print(json.dumps({'event': 'data registry error', 'directory': str(directory), 'error': repr(error)[:300]}),
              flush=True)


def registry_run_records(args, work, rollouts_archive, collector, teacher_demos, replay_corpora, eligible):
    """The run's registry records at every start and resume: what it reads (``consume``; data no registry entry holds
    stops the run here, ``NATLANG_ALLOW_UNREGISTERED_DATA=1`` opts out) and the growing datasets it writes (rollout
    archive, step chains), declared with their inputs and producer commit."""
    run = data_registry.run_name('grpo', work)
    reads = [('GRPO task pool', list(args.pool), eligible), ('held exclusion', [args.held_manifest, *args.exclude_manifest], None),
             ('errata (excluded ids)', errata_files(args), None), ('selection pass rates', [args.pass_rates], None)]
    if not args.replay and args.memetic_share > 0:
        reads.append(('memetic tasks', [args.memetic_tasks], None))
    if getattr(collector, 'variants', None) and args.neuralese_variants:
        reads.append(('Neuralese task variants', [args.neuralese_variants], len(collector.variants)))
    if args.hindsight_trajectories and args.teacher_attempts > 0:
        reads.append(('hindsight trajectories', [args.hindsight_trajectories], None))
    if teacher_demos is not None:
        reads.append(('teacher demonstrations (self-distillation)', list(args.sd_teacher_source or default_sd_teacher_sources()), None))
    for corpus in replay_corpora:
        reads.append((f'replay corpus {corpus.name}', [corpus.path], len(corpus.keys)))
    if args.replay:
        reads.append(('learner replay of archived groups', [args.replay], None))
    if args.behaviour_probe:
        reads.append(('behaviour probe', [args.behaviour_probe], None))
    if args.probe_turns:
        reads.append(('probe turns', [args.probe_turns], None))
    inputs = []
    for how, items, amount in reads:
        refs = data_registry.consume([i for i in items if i], run=run, how=how, amount=amount)
        if how in ('GRPO task pool', 'teacher demonstrations (self-distillation)', 'Neuralese task variants', 'memetic tasks') \
                or how.startswith('replay corpus'):
            inputs += [r for r in refs if r.get('id')]
    try:
        for kind, directory, fmt in (('grpo-rollouts', rollouts_archive, 'gNNNNNN.rollouts.jsonl.zst: every episode the learner '
                                      'lowered (turns with sampling captures), one file per group'),
                                     ('grpo-step-chains', work / 'step-chains', 'step-NNNNN.pkl.zst: the exact learner batch of a '
                                      'step (chains, advantages, token scales); only the newest --save-step-chains stay')):
            if directory is None or (kind == 'grpo-step-chains' and not args.save_step_chains):
                continue
            data_registry.declare_dataset(data_registry.dataset_id(directory, kind), directory, kind='rollouts' if kind == 'grpo-rollouts'
                                          else 'step-chains', format=fmt, inputs=inputs, license='project-generated (own LFM3 policy samples)',
                                          note=f'written by {run}')
    except Exception as error:
        print(json.dumps({'event': 'data registry error', 'error': repr(error)[:300]}), flush=True)


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
EMBEDDING = 'model.embed_tokens.weight'
PRODUCER_TABLE = 'model.natlang_producer_embed'     # export.native_hf: the producing input embedding of every token


def producer_adapter_keys(interface_state):
    """A checkpoint's producer-adapter tensors (``SharedStateModel.adapter``): with one the producer table is not the
    embedding, and neither WeightSync nor the learner models it."""
    return sorted(k for k in interface_state if k.startswith('adapter.'))


SENTINEL_TEXT = ('The committee reviewed the proposal on Tuesday and asked for a revised budget. '
                 'def mean(values):\n    return sum(values) / len(values)\n'
                 'Water boils at 100 degrees Celsius at sea level, and lower at altitude.')
# Learner vs actor log-probabilities of the sentinel (mean and largest |difference| over its tokens): the merged BF16
# export served with FP8 KV against the learner's unmerged BF16 forward. Measured on the real actor 2026-11-22: SGLang
# vs learner mean 0.066-0.076, max 0.34-0.42; native vs learner 0.034/0.30. Wrong or torn weights put both far above.
SENTINEL_TOLERANCE = (0.2, 2.0)


def interface_with_rows(interface, token_ids, rows, marker_ids, override_ids):
    """``interface_state`` with the trained Neuralese rows (``rows`` [K, D] at ``token_ids``) in place of the marker and
    override rows (``learner_state.merged_backbone_and_interface``'s rule; a tied checkpoint's ``*output_rows`` stay
    None)."""
    import torch
    out = dict(interface)
    position = {int(t): i for i, t in enumerate(token_ids.tolist())}
    for ids, field, out_field in ((marker_ids, 'input_rows', 'output_rows'),
                                  (override_ids, 'extra_input_rows', 'extra_output_rows')):
        if not ids or out.get(field) is None:
            continue
        updated = rows.index_select(0, torch.tensor([position[int(i)] for i in ids], dtype=torch.long,
                                                    device=rows.device)).detach().cpu()
        out[field] = updated.to(out[field].dtype)
        if out.get(out_field) is not None:
            out[out_field] = updated.to(out[out_field].dtype)
    return out


class WeightSync:
    """Publication of the learner's policy to its actor (lfm3opti.md §6 item 4, v2 §7.1).

    Host memory: two page-locked, memory-mapped safetensors files on tmpfs in the actor export's layout (``slot-a``,
    ``slot-b``, prefaulted once): the current version and the staging one. Fixed tensors (norms, conv kernels, every
    frozen projection) are written once; a publication writes only the merged projections and the embedding (producer
    table) into the staging slot, GPU -> pinned mapping, so no fresh pageable memory is touched per step.

    Each version:
      1. ``snapshot`` (main thread, after the optimizer step): GPU clones of the trainable factors and rows (and the
         learner's log-probabilities of a fixed sentinel sequence), so the next step's optimizer never tears the export.
      2. merge (publish thread): ``bf16(fp32(W0) + s * B @ A)`` from the snapshot into the staging slot; its file digest
         and ``export_tensors_identity``; the native checkpoint (``also``) is written from the same bytes.
      3. pause (retract) -> ``update_weights_from_disk`` of the staging slot -> ``rebind-policy`` (the front measures
         SGLang's ``/model_info`` and the file, and reloads its native route) -> ``attest_all_routes`` on the measured
         routes -> continue -> sentinel: SGLang's (and a resident native engine's) log-probabilities against the
         learner's.
      4. a failure restores the last good version (the current slot, at first the start export) on every route,
         retries the publication once, and otherwise reports the version failed (the trainer stops on it).

    ``stage_native`` puts another native checkpoint into the staging slot (held measurements of other policies);
    ``publish_start`` serves the start export itself."""

    SLOTS = ('slot-a', 'slot-b')

    def __init__(self, learner, export_dir, sglang_url, weights_dir, *, front_url=None, producer_from_embedding=True,
                 base_sha256=None, start_policy=None, interface_ids=((), ()), tied=True, sentinel_ids=None,
                 sentinel_tolerance=SENTINEL_TOLERANCE, extra_actors=(), on_actor=None):
        import mmap
        import torch
        self.learner, self.export_dir, self.sglang_url = learner, Path(export_dir), sglang_url
        # ``--extra-actor`` actors (``parse_extra_actor``): every version the primary actor serves is published to each
        # of them too, attested on all its routes; ``on_actor(front, version)`` reports the version each one serves
        self.extra_actors, self.on_actor = [dict(a) for a in extra_actors], on_actor
        self.weights_dir, self.front_url = Path(weights_dir), front_url
        self.producer_from_embedding, self.base_sha256 = producer_from_embedding, base_sha256
        self.marker_ids, self.override_ids = (tuple(int(i) for i in ids) for ids in interface_ids)
        self.tied, self.sentinel_ids = tied, list(sentinel_ids or ())
        self.sentinel_tolerance = sentinel_tolerance
        self.thread, self.last, self.released, self.pending, self.publishing = None, {}, None, None, None
        params = dict(learner.model.named_parameters())
        export = self.export_dir / 'model.safetensors'
        with export.open('rb') as handle:
            size = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(size))
        self.header_bytes = 8 + size
        layout = {name: info for name, info in header.items() if name != '__metadata__'}
        self.kinds = {}
        for name, info in layout.items():
            if info['dtype'] != 'BF16':
                raise ValueError(f'export tensor {name} is {info["dtype"]}, expected BF16')
            if name in learner.target_map:
                kind = 'lora'
            elif name == EMBEDDING or (name == PRODUCER_TABLE and producer_from_embedding):
                kind = 'embedding'
            elif name in params:
                kind = 'frozen'
            else:
                kind = 'export'
            if kind != 'export':
                source = params[EMBEDDING if kind == 'embedding' else name]
                if tuple(source.shape) != tuple(info['shape']):
                    raise ValueError(f'export/learner shape mismatch {name}: {info["shape"]} vs {tuple(source.shape)}')
            self.kinds[name] = kind
        missing = set(params) - set(layout) - {'lm_head.weight'}
        if missing:
            raise ValueError(f'learner parameters absent from the export: {sorted(missing)[:5]}')
        # tmpfs is memory outside the ledger: a dead owner's slots are swept, ours go at exit (also on SIGTERM)
        from natlang_neuralese.common import owned_scratch
        owned_scratch.sweep_dead(self.weights_dir.parent, 'natlang-*-actor-weights')
        owned_scratch.claim(self.weights_dir)
        owned_scratch.remove_on_exit(self.weights_dir)
        self.slots = []
        for slot in self.SLOTS:
            directory = self.weights_dir / slot
            directory.mkdir(exist_ok=True)
            for path in self.export_dir.iterdir():
                if path.is_file() and path.name != 'model.safetensors':
                    shutil.copyfile(path, directory / path.name)
            target = directory / 'model.safetensors'
            partial = target.with_name(target.name + '.partial')
            shutil.copyfile(export, partial)      # every page written once here (prefaulted tmpfs)
            os.replace(partial, target)
            handle = target.open('r+b')
            mapping = mmap.mmap(handle.fileno(), 0)
            whole = torch.frombuffer(mapping, dtype=torch.uint8)
            pinned = False
            if torch.cuda.is_available():
                try:
                    pinned = int(torch.cuda.cudart().cudaHostRegister(whole.data_ptr(), whole.numel(), 0).value) == 0
                except Exception:
                    pass
            views = {name: torch.frombuffer(mapping, dtype=torch.bfloat16, count=math.prod(info['shape']),
                                            offset=self.header_bytes + info['data_offsets'][0]).view(info['shape'])
                     for name, info in layout.items()}
            self.slots.append({'dir': directory, 'file': handle, 'map': mapping, 'pinned': pinned, 'views': views,
                               'whole': whole})
            if len(self.slots) == 1:   # still a byte copy of the start export: its digests, read from the mapping
                self.start_digests = self._identity(self.slots[0])
            with torch.no_grad():      # the learner's frozen tensors, once (the start export holds the same values)
                for name, kind in self.kinds.items():
                    if kind == 'frozen':
                        views[name].copy_(params[name].detach())
        self.staging = 0
        self.policy_cache = {}
        start = start_policy or {}
        self.last_good = {'version': start.get('version', 0), 'dir': str(self.export_dir),
                          'policy_id': start.get('policy_id'), 'slot': None, **self.start_digests}

    # ---- snapshot and merge ----------------------------------------------------------------------------------------
    def snapshot(self):
        """Freeze the step's policy (main thread, between optimizer steps): GPU clones of every trainable factor and
        row, and the learner's sentinel log-probabilities. The publish thread reads only these."""
        import torch
        self.wait()
        with torch.no_grad():
            leaves = {name: t.detach().clone() for name, t in self.learner.named_trainable_tensors().items()}
        self.pending = {'leaves': leaves, 'sentinel': self.learner_sentinel() if self.sentinel_ids else None,
                        'kind': 'learner'}

    def learner_sentinel(self):
        import torch
        from natlang_neuralese.train import lfm3_fast
        ids = torch.tensor(self.sentinel_ids, dtype=torch.long, device=self.learner.device)
        with torch.no_grad():
            logp = lfm3_fast.token_logprobs(self.learner.model, ids, torch.arange(len(ids) - 1, device=ids.device),
                                            ids[1:])
        return logp.float().cpu().tolist()

    def policy_id(self, leaves):
        """``Lfm3Learner.effective_policy_sha256`` of the snapshot (bound to the base), the id meta.json records."""
        from types import SimpleNamespace
        from natlang_neuralese.train.lfm3_learner import Lfm3Learner
        view = SimpleNamespace(spec=self.learner.spec, base_weights_sha256=self.base_sha256,
                               named_trainable_tensors=lambda: leaves)
        return Lfm3Learner.effective_policy_sha256(view, self.base_sha256)

    def _merge(self, slot, leaves):
        import torch
        params = dict(self.learner.model.named_parameters())
        rows = self.learner.trainable_rows
        with torch.no_grad():
            embedding = params[EMBEDDING].detach().clone()
            if rows is not None:
                embedding[rows.token_ids] = leaves['trainable_rows.rows'].to(embedding.dtype)
            for name, kind in self.kinds.items():
                view = slot['views'][name]
                if kind == 'embedding':
                    view.copy_(embedding, non_blocking=slot['pinned'])
                elif kind == 'lora':
                    layer, _, short = self.learner.target_map[name]
                    scale = float(getattr(self.learner.bundles[layer], short).scale)
                    merged = (params[name].detach().float()
                              + scale * (leaves[f'lora_B.{name}'].float() @ leaves[f'lora_A.{name}'].float()))
                    view.copy_(merged.to(torch.bfloat16), non_blocking=slot['pinned'])
                    del merged
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def stage_native(self, path):
        """Put another native checkpoint (its interface rows merged into the embedding, as the actor's export merges
        them) into the staging slot; the next ``publish`` serves it."""
        import torch
        from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
        self.wait()
        state = CheckpointPolicy.load(Path(path))
        backbone, interface, config = state['backbone_state'], state['interface_state'], state['model_config']
        embedding = backbone[EMBEDDING].clone()
        rows = [(config['marker_ids'], interface['input_rows'])]
        if config.get('token_override_ids'):
            rows.append((config['token_override_ids'], interface['extra_input_rows']))
        for ids, values in rows:
            embedding[torch.tensor(list(ids))] = values.to(embedding.dtype)
        slot = self.slots[self.staging]
        for name, kind in self.kinds.items():
            if kind == 'embedding':
                source = embedding
            elif kind == 'export':
                continue
            else:
                source = backbone[name]
            slot['views'][name].copy_(source.to(torch.bfloat16))
        if any(kind == 'frozen' for kind in self.kinds.values()):
            # the staging slot's frozen tensors now hold this checkpoint's: the learner's are written back on the
            # learner's next publication into this slot
            slot['foreign'] = True
        self.pending = {'kind': 'native', 'path': str(path)}
        del state

    def backbone(self):
        """{name: host BF16 view} of the version being published (valid inside ``publish``'s ``also``)."""
        slot = self.slots[self.staging]
        return {name: view for name, view in slot['views'].items() if name != PRODUCER_TABLE}

    def interface(self, interface):
        """``interface_state`` of the version being published (its trained rows)."""
        leaves = (self.pending or {}).get('leaves')
        rows = self.learner.trainable_rows
        if not leaves or rows is None:
            return interface
        return interface_with_rows(interface, rows.token_ids, leaves['trainable_rows.rows'], self.marker_ids,
                                   self.override_ids)

    def _identity(self, slot):
        from natlang_neuralese.serve.sglang_actor import export_tensors_identity
        return {'weights_sha256': hashlib.sha256(memoryview(slot['map'])).hexdigest(),
                'tensor_identity': export_tensors_identity(slot['views'], self.marker_ids, self.override_ids,
                                                           self.tied)}

    # ---- actor control ---------------------------------------------------------------------------------------------
    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def close(self, remove=True):
        """Unregister the page-locked mappings (a registration outliving its mapping makes a later cudaHostRegister of
        the reused address range fail) and remove the slots (``remove=False`` keeps the files)."""
        import torch
        from natlang_neuralese.common import owned_scratch
        self.wait()
        for slot in self.slots:
            if slot['pinned']:
                torch.cuda.cudart().cudaHostUnregister(slot['whole'].data_ptr())
                slot['pinned'] = False
            slot['views'] = slot['whole'] = None
        if remove:
            for slot in self.slots:
                slot['map'].close()
                slot['file'].close()
            owned_scratch.release(self.weights_dir)

    def release(self, tags):
        """Hand the actor's GPU memory back for the learner phase (``--actor-release``): retract, then SGLang
        ``release_memory_occupation`` of ``tags``; the next ``publish``/``restore`` resumes it. Returns the seconds."""
        started = time.time()
        self.wait()
        if self.sglang_url is None or not tags:
            return 0.
        post(self.sglang_url + '/pause_generation', {'mode': 'retract'})
        try:
            post(self.sglang_url + '/release_memory_occupation', {'tags': list(tags)})
        except Exception:
            post(self.sglang_url + '/continue_generation', {})
            raise
        self.released = list(tags)
        return time.time() - started

    def _resume(self):
        if self.released:
            post(self.sglang_url + '/resume_memory_occupation', {'tags': self.released})
            self.released = None

    def restore(self, version=None):
        """Undo ``release`` without new weights (a skipped update): resume, reload the last good version, continue."""
        self.wait()
        if self.sglang_url is None or not self.released:
            return
        self._resume()
        answer = post(self.sglang_url + '/update_weights_from_disk',
                      {'model_path': self.last_good['dir'], 'flush_cache': True,
                       'weight_version': str(self.last_good['version'])})
        post(self.sglang_url + '/continue_generation', {})
        if not json.loads(answer).get('success', False):
            raise RuntimeError(f'actor weight reload after release failed: {answer[:400]}')

    def _info(self, front_url=None):
        with urllib.request.urlopen((front_url or self.front_url) + '/v1/neuralese/info', timeout=60) as response:
            return json.loads(response.read())

    def _bind(self, target, timing, sglang_url=None, front_url=None):
        """Serve ``target`` ({dir, version, policy_id, weights_sha256, tensor_identity}) on every route of the primary
        actor (or of the actor at ``sglang_url``/``front_url`` on this machine) and check the measured attestation.
        Leaves SGLang generating."""
        from natlang_neuralese.serve.sglang_actor import attest_all_routes
        sglang_url = sglang_url or self.sglang_url
        front_url = front_url or self.front_url
        t = time.time()
        post(sglang_url + '/pause_generation', {'mode': 'retract'})
        try:
            raw = post(sglang_url + '/update_weights_from_disk',
                       {'model_path': str(target['dir']), 'flush_cache': True, 'weight_version': str(target['version'])})
            if not json.loads(raw).get('success', False):
                raise RuntimeError(f'update_weights_from_disk failed: {raw[:300]}')
            timing['actor_update_s'] = round(time.time() - t, 2)
            if front_url is not None:
                t = time.time()
                post(front_url + '/v1/neuralese/rebind-policy',
                     {'policy_id': target['policy_id'], 'weights_sha256': target['weights_sha256'],
                      'tensor_identity': target['tensor_identity'], 'version': target['version'],
                      'export_dir': str(target['dir'])})
                ok, mismatches = attest_all_routes(self._info(front_url), {
                    'policy_id': target['policy_id'], 'version': target['version'], 'model_path': str(target['dir']),
                    'weights_sha256': target['weights_sha256'], 'tensor_identity': target['tensor_identity']})
                if not ok:
                    raise RuntimeError(f'route attestation failed: {json.dumps(mismatches)[:600]}')
                timing['rebind_s'] = round(time.time() - t, 2)
                timing['routes_attested'] = True
        finally:
            post(sglang_url + '/continue_generation', {})

    def _publish_extra(self, target, timing):
        """Publish the version the primary actor now serves to every ``--extra-actor``: one on this machine reloads the
        slot directory itself (``_bind``), a remote one gets the export copied to its host and rebinds there
        (``pop_rl_actor.rebind_actor``: update_weights_from_disk, rebind-policy, attest_all_routes). An actor that fails
        keeps its last version (``on_actor`` is told nothing new) and the run goes on."""
        report = {}
        for actor in self.extra_actors:
            t, entry = time.time(), {}
            try:
                if actor.get('host'):
                    import pop_rl_actor
                    rebind = pop_rl_actor.rebind_actor(
                        front_url=actor['front'], sglang_url=actor['sglang'], policy_id=target['policy_id'],
                        version=target['version'], weights_sha256=target['weights_sha256'],
                        tensor_identity=target['tensor_identity'], export_dir=str(target['dir']), host=actor['host'],
                        host_profile=actor.get('profile', 'colab_g4'))
                    entry['export_dir'] = rebind.get('export_dir')
                else:
                    self._bind(target, entry, sglang_url=actor['sglang'], front_url=actor['front'])
                actor['version'] = target['version']
                entry.update(ok=True)
                if self.on_actor is not None:
                    self.on_actor(actor['front'], target['version'])
            except Exception as error:
                entry.update(ok=False, error=repr(error)[:400], serving=actor.get('version'))
            entry['seconds'] = round(time.time() - t, 2)
            report[actor['front']] = entry
        if report:
            timing['extra_actors'] = report

    def _sentinel(self, learner_logp, timing):
        """SGLang's (and a resident native engine's) sentinel log-probabilities against the learner's and each other."""
        if self.front_url is None or not self.sentinel_ids:
            return
        answer = json.loads(post(self.front_url + '/v1/neuralese/sentinel', {'token_ids': self.sentinel_ids}))
        scored = {'sglang': answer.get('sglang'), 'native': answer.get('native'), 'learner': learner_logp}
        pairs = [(a, b) for a, b in (('sglang', 'learner'), ('native', 'learner'), ('sglang', 'native'))
                 if scored[a] is not None and scored[b] is not None]
        report, failed = {}, []
        for a, b in pairs:
            d = np.abs(np.asarray(scored[a], np.float64) - np.asarray(scored[b], np.float64))
            if len(d) != len(self.sentinel_ids) - 1:
                failed.append(f'{a}/{b}: {len(d)} log-probabilities for {len(self.sentinel_ids) - 1} tokens')
                continue
            report[f'{a}_vs_{b}'] = {'mean': round(float(d.mean()), 5), 'max': round(float(d.max()), 5)}
            if d.mean() > self.sentinel_tolerance[0] or d.max() > self.sentinel_tolerance[1]:
                failed.append(f'{a}/{b} {report[f"{a}_vs_{b}"]}')
        timing['sentinel'] = report
        if failed:
            raise RuntimeError(f'sentinel log-probabilities disagree beyond {self.sentinel_tolerance}: {failed}')

    def _rollback(self, timing):
        """Serve the last good version again on every route (generation continues either way)."""
        try:
            self._bind(self.last_good, {})
            timing['rolled_back_to'] = self.last_good['version']
            return True
        except Exception as error:
            timing['rollback_error'] = repr(error)[:400]
            timing['rolled_back_to'] = None
            return False

    def publish_start(self, version, on_done, policy):
        """Serve the start export itself (``policy`` its checkpoint digest), e.g. after measuring other policies."""
        self.pending = {'kind': 'start', 'policy_id': policy}
        self.publish(version, on_done)

    def publish(self, version, on_done, also=None, policy=None):
        """Publish the pending snapshot (or staged checkpoint) as ``version`` in a background thread; ``on_done(version,
        ok)``. ``also()`` writes the native checkpoint from the same bytes before the actor is updated."""
        pending = self.pending or {'kind': 'none'}

        def run():
            timing = {}
            try:
                ok, answer = body(timing)
            except Exception as error:
                ok, answer = False, f'publish failed: {error!r}'[:600]
            self.last = {'version': version, 'ok': ok, 'answer': None if ok else answer, **timing}
            on_done(version, ok)
            if ok and self.extra_actors and self.publishing is not None:
                # the primary serves the version (episodes there are stamped with it from now on); then every extra
                # actor, each stamped with its own version as it lands (``on_actor``)
                extra = {}
                self._publish_extra(self.publishing, extra)
                self.last.update(extra)

        def body(timing):
            t = time.time()
            if pending['kind'] == 'start':
                target = {'dir': str(self.export_dir), 'slot': None, 'version': version,
                          'policy_id': policy or pending['policy_id'], **self.start_digests}
            else:
                slot = self.slots[self.staging]
                if pending['kind'] == 'learner':
                    if slot.pop('foreign', False):
                        import torch
                        params = dict(self.learner.model.named_parameters())
                        with torch.no_grad():
                            for name, kind in self.kinds.items():
                                if kind == 'frozen':
                                    slot['views'][name].copy_(params[name].detach())
                    self._merge(slot, pending['leaves'])
                    policy_id = policy or self.policy_id(pending['leaves'])
                elif pending['kind'] == 'native':
                    from natlang_neuralese.common.hashing import sha256_file_hex_cached
                    policy_id = policy or sha256_file_hex_cached(Path(pending['path']))
                else:
                    raise RuntimeError('publish without a snapshot or staged checkpoint')
                timing['merge_s'] = round(time.time() - t, 2)
                t = time.time()
                target = {'dir': str(slot['dir']), 'slot': self.staging, 'version': version, 'policy_id': policy_id,
                          **self._identity(slot)}
                timing['digest_s'] = round(time.time() - t, 2)
            self.publishing = target
            if also is not None:
                t = time.time()
                try:
                    timing.update(also() or {})
                except Exception as error:     # the actor is still updated; the write's failure is reported
                    timing['checkpoint_error'] = repr(error)[:400]
                timing['checkpoint_s'] = round(time.time() - t, 2)
            timing.update(policy_id=target['policy_id'], weights_sha256=target['weights_sha256'],
                          tensor_identity=target['tensor_identity'])
            if self.sglang_url is None:
                self._adopt(target)
                return True, 'no actor sync'
            if self.released is not None:
                t = time.time()
                self._resume()
                timing['actor_resume_s'] = round(time.time() - t, 2)
            failures = []
            for attempt in (1, 2):
                try:
                    self._bind(target, timing)
                    self._sentinel(pending.get('sentinel') if pending['kind'] == 'learner' else None, timing)
                    self._adopt(target)
                    timing['attempts'] = attempt
                    return True, None
                except Exception as error:
                    failures.append(repr(error)[:600])
                    self._rollback(timing)
            timing['attempts'] = 2
            return False, f'publication of version {version} failed twice (actor restored to version ' \
                          f'{timing.get("rolled_back_to")}): {failures}'
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def _adopt(self, target):
        """``target`` is served: it becomes the last good version and the other slot becomes staging."""
        self.last_good = dict(target)
        if target.get('slot') is not None:
            self.staging = 1 - target['slot']
        self.pending = None


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


def free_cuda_cache(collect=True):
    """Return the learner's cached CUDA blocks to the system, after the Python garbage collection when ``collect``
    (``--no-learner-gc`` opts out): tensors kept alive only by reference cycles (closures, exception frames) are freed by
    ``gc.collect`` and land in PyTorch's cache, so an ``empty_cache`` that ran before it left them reserved: the
    post-optimizer read was 31-43 GB reserved against 24 GB of live tensors (grpo-v4-run2 steps 20-24) until the moment
    offload's own ``empty_cache`` minutes later, which is when the actor's memory comes back. The cycle found so far was
    ``run_prefix_tree``'s recursive ``walk`` (``lfm3_fast_train.prefix_levels`` is iterative now); a nonzero
    ``gc_freed_gb`` in ``cuda_cache_release`` after that fix names another one. Returns the figures for train.jsonl: what
    the collection freed, allocated, and reserved before and after."""
    import torch
    started = time.time()
    before = torch.cuda.memory_allocated()
    if collect:
        release_host_memory()
    mid, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    gb = lambda n: round(n / 2**30, 2)
    return {'gc_freed_gb': gb(before - mid), 'allocated_gb': gb(mid), 'reserved_before_gb': gb(reserved),
            'reserved_gb': gb(torch.cuda.memory_reserved()), 'seconds': round(time.time() - started, 2)}


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
def write_pruned_checkpoint(state, native_rows, special, path, source_sha256, trained, learner_step=None, extra=None):
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
    source_step = int(state['step'])
    recorded_step = source_step if learner_step is None else int(learner_step)
    out.update(backbone_state=backbone, step=recorded_step, learner_step=recorded_step, source_step=source_step,
               inference_only=True, optimizer_pruned=True, optimizer_pruned_keys=['training'],
               inference_source={'schema': 'natlang.native-inference-source/1', 'training_checkpoint_sha256': source_sha256,
                                 'training_checkpoint_bytes': 0, 'step': recorded_step,
                                 'learner_step': recorded_step, 'source_step': source_step})
    out.update(extra or {})
    from natlang_neuralese.train.checkpoint_offload import staged_write   # NVMe first, then moved (StateSaver)
    with staged_write(path, expected_bytes=sum(v.numel() * v.element_size() for v in backbone.values())) as target:
        torch.save(out, target.path)
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


def decision_forward_key(row):
    """What the learner's forward of a decision depends on: its exact prompt, options and imported state. Sibling
    episodes of a group often sample the same decision (the 2026-11-22 bench: 16 decisions, 2 distinct)."""
    return (np.asarray(row['prompt'], np.int64).tobytes(),
            tuple(np.asarray(o, np.int64).tobytes() for o in row['options']),
            repr(sorted((int(start), str(block)) for start, block in row.get('imports') or ())))


def decision_batches(items, max_tokens):
    """Pack sampled decisions (``lower_rollout_turns.decision_turns`` rows with their advantage) into forward batches: each
    option is one document (exact prompt + option continuation), so a decision of K options is K packed documents. A
    batch is a list of forwards, each the list of the step's decisions with one ``decision_forward_key``: they are scored
    once and each keeps its own loss term (exact; it saves the repeated forwards)."""
    forwards = {}
    for item in items:
        forwards.setdefault(decision_forward_key(item['row']), []).append(item)
    batches, batch, size = [], [], 0
    for same in forwards.values():
        n = decision_tokens(same[0]['row'])
        if batch and size + n > max_tokens:
            batches.append(batch)
            batch, size = [], 0
        batch.append(same)
        size += n
    return batches + ([batch] if batch else [])


def decision_tensors(batch, device):
    """(ids, lengths, positions, targets, option index per target, options per forward, imported state spans
    [(packed row start, key)], their producer documents {key: document}) of a decision batch (``decision_batches``)."""
    import torch
    ids, lengths, positions, targets, owner, counts, offset, option, imports = [], [], [], [], [], [], 0, 0, []
    sources = {}
    for same in batch:
        row = same[0]['row']
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


def sd_buffer_save(path, buffer):
    """The self-distillation buffer to ``path`` (atomic; None: not persisted)."""
    if path is None:
        return
    import pickle
    tmp = Path(path).with_suffix('.tmp')
    with tmp.open('wb') as handle:
        pickle.dump({'schema': 1, 'buffer': buffer}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def sd_buffer_load(path, step):
    """The saved buffer's items added before ``step`` (an item's version is the step that lowered it, which produced
    version + 1: a resume from an earlier state never replays successes of the steps it rolled back); [] when absent or
    unreadable."""
    if path is None or not Path(path).exists():
        return []
    import pickle
    try:
        with Path(path).open('rb') as handle:
            saved = pickle.load(handle)
    except Exception:
        return []
    return [item for item in saved.get('buffer', []) if item['version'] < step]


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
    runtime was exact or only its display drifted (``sft_sources.verified_episodes``, ``drifted``), on an eligible pool task (the run's task list, held closure
    applied) and outside the held closure again (``sft_sources.held_excluded_ids``), at most ``max_per_task`` per task
    (teacher models taking turns, as ``sft_sources.capped``). ``take`` lowers the chosen ones with sft_sources' own
    rendering (``call_row`` on the replayed turns: the student-form prompt the runtime builds, the teacher's reply as
    target, an empty think block at weight 0)."""

    def __init__(self, works, by_id, held_manifest, tokenizer, max_per_task, seed, log=print, uses_path=None,
                 take_seed=None, drifted=True):
        import random as _random
        from sft_sources import TemplateRenderer, held_closure, held_excluded_ids, verified_episodes
        self.renderer = TemplateRenderer(tokenizer)
        self.lock = threading.Lock()
        self.funnel = collections.defaultdict(collections.Counter)
        self.rng = _random.Random(seed)
        found, tasks = {}, {}
        for work in works:
            if not (Path(work) / 'replay.jsonl').exists():
                log({'event': 'sd teacher source missing', 'work': str(work)})
                continue
            for episode, record in verified_episodes(work, self.funnel, sources={'teacher'}, drifted=drifted):
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
        # Least used first spans resumes: the use counters persist in ``uses_path`` (a resume used to restart them at 0
        # with the same tie order, so every resume replayed the same demonstrations first: grpo-v4-run2 resumed 14 times
        # in 40 steps), and ties break by ``take_seed`` (the resumed run's first group), not the construction seed.
        self.uses_path = Path(uses_path) if uses_path else None
        if self.uses_path is not None and self.uses_path.exists():
            saved = json.loads(self.uses_path.read_text())
            for items in self.by_task.values():
                for item in items:
                    item.update(saved.get(item['key']) or {})
        if take_seed is not None:
            self.rng = _random.Random(take_seed)
        log({'event': 'sd teacher demonstrations', 'tasks': len(self.by_task), 'episodes': self.episodes,
             'funnel': {k: dict(v) for k, v in self.funnel.items()}, 'renderer': self.renderer.identity})

    def take(self, eligible, count, stats, use='uses', prefix='sd_teacher'):
        """Up to ``count`` lowered teacher episodes of the ``eligible`` task ids, one per task, least used first (ties at
        random; ``use`` names the counter: the SD replay and the reasoning RL term (``ReasoningRL``) keep their own); a
        render error drops that episode for the run. Thread-safe (the reasoning RL workers take from it too)."""
        from sft_sources import call_row, replayed_turns
        with self.lock:
            candidates = [item for task_id in eligible for item in self.by_task.get(task_id, ())]
            stats[f'{prefix}_candidates'] = len(candidates)
            self.rng.shuffle(candidates)
            candidates.sort(key=lambda item: item.get(use, 0))
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
                stats[f'{prefix}_render_error'] += 1
                print(json.dumps({'event': 'sd teacher render error', 'key': item['key'], 'error': repr(error)[:300]}),
                      flush=True)
                with self.lock:
                    if item in self.by_task[item['task_id']]:
                        self.by_task[item['task_id']].remove(item)
                continue
            for row in rows:
                row['teacher'] = True            # no actor log-probabilities: a ratio-free term (train loop)
            with self.lock:
                item[use] = item.get(use, 0) + 1
            tasks.add(item['task_id'])
            out.append({'task_id': item['task_id'], 'round': item['key'], 'rows': rows, 'decisions': [],
                        'reward': 1., 'quality': 1., 'role': 'sd_teacher', 'teacher_model': item['teacher']})
        return out

    def save(self):
        """The use counters to ``uses_path`` (the train loop calls it after every step)."""
        if self.uses_path is None:
            return
        with self.lock:
            uses = {item['key']: {k: v for k, v in item.items() if k.endswith('uses') and v}
                    for items in self.by_task.values() for item in items}
        tmp = self.uses_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({k: v for k, v in uses.items() if v}, sort_keys=True) + '\n')
        tmp.replace(self.uses_path)


def sd_teacher_tasks(collector, band):
    """Tasks whose current pass rate (the run's latest group, else the measured base rate) is in the pass band or 0
    (all-fail): where a verified demonstration can still raise the success rate."""
    rates = {k: (v['successes'], v['scored']) for k, v in collector.base_rates.items() if v.get('scored')}
    rates.update(collector.own)
    low, high = band
    return sorted(k for k, (s, n) in rates.items() if n and (s == 0 or low <= s / n <= high))


# Reasoning RL against fixed teacher actions (``--reasoning-rl-weight``; owner 2026-11-18). A teacher call's reply is
# (think block, action). Training teacher think blocks taught the student short summary-style reasoning (SFT held
# -16 pts, DECISIONS 2026-11-18), so they are context and teacher trajectories teach only actions. This term trains the
# student's own reasoning from them, searching along each teacher invocation from its first call (``--reasoning-rl-calls``
# N caps it, opt-in); tool outputs come from the trajectory, nothing executes; ``--reasoning-rl-workers`` searches run
# in parallel. Two modes (DECISIONS 2026-11-18 for the allocation): a walk (``run``, 1 - ``--reasoning-rl-rewrite-share``
# 0.3 of invocations) steps through the trajectory with the student's own best reasoning carried at every step, its
# history the deployed situation (preserve_thinking), each step's group ready (with the policy version its draws started
# on) as soon as it is done; a parallel rewrite (``rewrite``) keeps the teacher's recorded history, rewrites every
# position's teacher think block into the student's own by cut-and-continue rounds scored by the teacher actions'
# likelihood, and feeds the self-distillation replay. The walk:
#   1. one teacher-forced pass with empty think blocks scores every step's action at once: ``r0_t``, the action's mean
#      token log-probability without reasoning (the step's difficulty);
#   2. at step t the actor draws think blocks z on the carried history in waves of ``--reasoning-rl-wave`` until one
#      scores high (its action at least ``--reasoning-rl-high-logp`` nats per token) or the step's budget is spent
#      (``--reasoning-rl-samples``; ``--reasoning-rl-max-samples`` where r0_t is below ``--reasoning-rl-hard-logp``);
#   3. each z is scored by one forward of the trajectory's next steps (z, the teacher action, its tool output, then up
#      to ``--reasoning-rl-horizon`` later actions after empty think blocks): its immediate reward r_t (the action's
#      mean token log-probability) and the later actions' rewards, so its return is r_t + sum_s gamma^(s-t) r_s
#      (``--reasoning-rl-gamma``). The reasoning room shrinks as the history grows (``--max-sequence-tokens``); below
#      ``MIN_REASONING_ROOM`` the trajectory ends. Reasoning that helps later actions gets credit too. The best
#      return(s) carry forward as history, as the template keeps reasoning (preserve_thinking): greedy, or a beam of
#      ``--reasoning-rl-beam``;
#   4. every step's samples form a group: z's tokens at its group-normalized return (on-policy, truncated IS against
#      the actor's own sampling log-probabilities), the teacher action after each closed z as a ratio-free
#      cross-entropy at ``--reasoning-rl-action-weight``; the reasoning term with it is the gradient of
#      E_z[log pi(action | prompt, z)] (LaTRO, VeriFree, RLPR, JEPO; Google's SRL decomposes expert trajectories
#      into steps). A group whose returns spread less than ``--reasoning-rl-min-std`` trains only its actions;
#   5. the carried chain (student reasoning that leads to the teacher's actions; a step whose best z did not score
#      high keeps it as context) joins the self-distillation replay (``--sd-weight``) as source ``search``.
# Within a group the action is fixed, so mean and summed log-probability rank alike and a no-reasoning baseline cancels.
#
# The objective (``--reasoning-rl-objective``; DECISIONS 2026-11-20 "reasoning RL without the future agreement bias").
# 'sequence' (before 2026-11-20) trains each think block by its group-normalized return, the likelihood of the actions
# after it: a sequence-level reward on z. Strauss, Fowl & Razin (arXiv 2611.23397, Lemma 1 / Prop. 1) show that such an
# objective has a future agreement bias: it favours tokens after which the rest of the sequence agrees with the target,
# whatever their own probability, and so short, formulaic continuations (over-refusal, early stops). Here that is
# closing ``</think>`` right after naming the action (hazard investigation 3, DECISIONS 2026-11-20 12:00). Selecting
# the best-scoring blocks for self-distillation (the carried ``search`` chains, the ``rewrite`` mode) is the same
# selection. 'twin-fkl' (default) is the paper's myopic token-level objective with forward KL: at every position of
# each sampled think block, KL(twin(. | x', z<t) || policy(. | x, z<t)), the twin being the policy itself with the
# teacher action in a training guidance turn (x'; the OPSD form of Zhao et al., the teacher of the paper's code
# base), computed on the actor (top-k + ``</think>`` input log-probabilities, ``twin_distributions``) and matched in
# the learner (``twin_forward_kl``); the teacher action after a closed block keeps its cross-entropy (a token-level
# term too). Each step draws one wave, every sample trains, the best return is still carried as history (context
# only), and neither carried chains nor rewrites feed the SD replay unless asked for.
RRL_ROLE = 'reasoning_rl'
MIN_REASONING_ROOM = 256   # reasoning tokens a step must have room for, or the trajectory ends there
RRL_SOURCES = ('jolt', 'demos')
RRL_OBJECTIVES = ('twin-fkl', 'sequence')
TWIN_HEADER = 'Verified reference for this call (training only):\n\nThe reply that follows your reasoning:\n'


def rrl_objective(args):
    """``--reasoning-rl-objective`` ('sequence' for an args object from before the flag)."""
    return getattr(args, 'reasoning_rl_objective', 'sequence')


def rrl_rewrite_share(args):
    """``--reasoning-rl-rewrite-share``, defaulting by objective: 0.3 for 'sequence', 0 for 'twin-fkl' (a rewrite keeps
    the blocks that score best, the sequence-level selection the twin objective removes)."""
    share = getattr(args, 'reasoning_rl_rewrite_share', None)
    if share is not None:
        return share
    return 0.3 if rrl_objective(args) == 'sequence' else 0.


def rrl_carry_sd(args):
    """Whether a walk's best chain joins the SD replay (``--reasoning-rl-carry-sd``; default on for 'sequence', off for
    'twin-fkl', where the best-return chain is the sequence-level selection)."""
    carry = getattr(args, 'reasoning_rl_carry_sd', None)
    if carry is not None:
        return carry == 'on'
    return rrl_objective(args) == 'sequence'


def twin_turn_ids(tokenizer, close_id):
    """The twin's guidance turn as served-tokenizer ids (``twin_context``): header (``<|im_start|>system`` and
    ``TWIN_HEADER``), footer (``<|im_end|>`` and a newline), the ``<|im_end|>`` and newline ids. The teacher action's
    own ids go between header and footer unchanged (no detokenize/re-encode round trip)."""
    def encode(text):
        return list(tokenizer.encode(text, add_special_tokens=False).ids)
    im_end, newline = tokenizer.token_to_id('<|im_end|>'), encode('\n')
    if im_end is None or len(newline) != 1:
        raise ValueError('the served tokenizer has no <|im_end|> or no single-token newline')
    return {'header': encode('<|im_start|>system\n' + TWIN_HEADER), 'footer': [im_end] + newline, 'im_end': im_end,
            'newline': newline[0], 'close': close_id}


def split_reply(sample, close_id):
    """Where a reply's action starts: after its first ``</think>`` (the reasoning runs through it); None when the reply
    has no close or nothing after it."""
    found = np.flatnonzero(np.asarray(sample) == close_id) if close_id is not None else []
    if not len(found) or found[0] + 1 >= len(sample):
        return None
    return int(found[0]) + 1


def teacher_call(row, close_id, markers):
    """One teacher call as a reasoning RL step: {'prompt', 'reply', 'action'[, 'action_weight']} (int32 ids) or the
    reason it is left out. The prompt is the call's exact student-form prompt (it ends where the student reasons); the
    action is the teacher reply after its ``</think>``, its row weights kept (template whitespace a demonstration holds
    as context stays at 0). A call importing Neuralese state is left out (the backend request carries no state)."""
    if row.get('imports'):
        return 'imports'
    sample = np.asarray(row['sample'])
    cut = split_reply(sample, close_id)
    if cut is None:
        return 'no_action'
    action = sample[cut:]
    if markers and np.isin(action, list(markers)).any():
        return 'role_markers'
    call = {'prompt': np.asarray(row['prompt'], np.int32), 'reply': sample.astype(np.int32),
            'action': action.astype(np.int32), 'think': sample[:cut].astype(np.int32), 'index': int(row.get('index', 0))}
    if 'weight' in row:
        call['action_weight'] = np.asarray(row['weight'], np.float32)[cut:]
    return call


def teacher_segments(rows, close_id, markers, counts=None, history='keep'):
    """Runs of consecutive usable calls of one invocation whose prompts extend the previous call's full text (the
    calls ``exp0_sft.chains_of`` chains). Each call before the last of a run records its ``suffix``: the tokens between
    its reply and the next generation prompt's end (the tool output and the next turn's opening).

    ``history`` 'keep' (the default): the recorded prompts, the teacher's earlier reasoning included (a segment that
    starts mid-invocation reasons after the teacher's trace). 'empty': every earlier call of the invocation (usable or
    not) appears in a later call's prompt with an empty think block (``</think>`` and its action); the history the
    template renders before the invocation's first call (the runtime's opening turn) is kept either way."""
    by_invocation = collections.defaultdict(list)
    for row in rows:
        by_invocation[row.get('invocation', '')].append(row)
    runs = []
    for group in by_invocation.values():
        group.sort(key=lambda row: row.get('index', 0))
        chains = []                                   # rows whose prompts extend the previous row's full text
        for row in group:
            prompt, sample = np.asarray(row['prompt']), np.asarray(row['sample'])
            if chains:
                last = chains[-1][-1]
                full = np.concatenate([np.asarray(last['prompt']), np.asarray(last['sample'])])
                if len(prompt) >= len(full) and np.array_equal(prompt[:len(full)], full):
                    chains[-1].append(row)
                    continue
            chains.append([row])
        for chain in chains:
            prompts = [np.asarray(row['prompt'], np.int32) for row in chain]
            suffixes = [np.asarray(chain[i + 1]['prompt'][len(row['prompt']) + len(row['sample']):], np.int32)
                        for i, row in enumerate(chain[:-1])]
            if history == 'empty':
                context = list(prompts[0])
                for i, row in enumerate(chain):
                    prompts[i] = np.asarray(context, np.int32)
                    sample = np.asarray(row['sample'])
                    found = np.flatnonzero(sample == close_id) if close_id is not None else []
                    reply = [close_id, *sample[found[0] + 1:]] if len(found) else list(sample)
                    if i + 1 < len(chain):
                        context = context + [int(t) for t in reply] + [int(t) for t in suffixes[i]]
            current = []
            for i, row in enumerate(chain):
                call = teacher_call({**row, 'prompt': prompts[i]}, close_id, markers)
                if isinstance(call, str):
                    if counts is not None:
                        counts[f'call_skip_{call}'] += 1
                    if current:
                        runs.append(current)
                    current = []
                    continue
                if current:
                    current[-1]['suffix'] = suffixes[i - 1]
                current.append(call)
            if current:
                runs.append(current)
    return runs


def future_ids(context, calls, t, close_id, horizon=None, max_len=None):
    """(ids, action spans [(start, length)]) of the segment's rest after ``context`` (which ends with step t's closed
    think block): action t, its suffix, then each later step after an empty think block (``</think>``), at most
    ``horizon`` later steps (None: all) and only while the ids fit ``max_len`` (action t always)."""
    ids, spans = list(context), []
    last = len(calls) if horizon is None else min(len(calls), t + 1 + horizon)
    for s in range(t, last):
        part = ([close_id] if s > t else []) + [int(i) for i in calls[s]['action']]
        if s > t and max_len is not None and len(ids) + len(part) > max_len:
            break
        if s > t:
            ids.append(close_id)
        spans.append((len(ids), len(calls[s]['action'])))
        ids += [int(i) for i in calls[s]['action']]
        if s + 1 < last:
            ids += [int(i) for i in calls[s]['suffix']]
    return ids, spans


def discounted(rewards, gamma):
    """r_0 + gamma r_1 + gamma^2 r_2 ... (a missing reward counts 0)."""
    return sum((gamma ** i) * (r or 0.) for i, r in enumerate(rewards))


def rrl_advantages(returns, closed, min_std, unclosed='drop'):
    """Advantages of one group's samples (None: the group's reasoning does not train) and the reason. A sample that
    never closed its reasoning hit the token cap and produced no action: its score says the cap cut it, not that its
    reasoning was worse. ``unclosed`` 'drop' (default; DAPO's overlong filtering): it gets no advantage (None at its
    place) and the group is normalized over its closed samples alone; pushing a cut block down would teach "close
    earlier" directly (grpo-v4-run2's ``</think>`` hazard, DECISIONS 2026-11-19). 'floor' (before 2026-11-19): it ties
    with the group's worst closed sample and trains. A group whose returns spread less than ``min_std`` nats per token
    has nothing to rank: the normalization would turn noise-sized differences into full-size advantages."""
    scored = [r for r, c in zip(returns, closed) if c]
    if not scored:
        return None, 'none_closed'
    floor = min(scored)
    keep = [c or unclosed != 'drop' for c in closed]
    values = [r if c else floor for r, c, k in zip(returns, closed, keep) if k]
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    if len(values) < 2 or std < max(min_std, 1e-6):
        return None, 'low_spread'
    it = iter((v - mean) / std for v in values)
    return [next(it) if k else None for k in keep], None


def rrl_rows(name, context, sample, action, action_weight, reasoning_advantage, action_advantage):
    """One step's learner rows: the sampled think block (actor log-probabilities) and, after a closed one, the teacher
    action (no log-probabilities: ratio-free). ``advantage`` per row; a row ``weight`` 0 keeps tokens as context."""
    rows = [{'id': name, 'invocation': name, 'index': 0, 'prompt': np.asarray(context, np.int32),
             'sample': np.asarray(sample['ids'], np.int32), 'logp': np.asarray(sample['logp'], np.float32),
             'advantage': reasoning_advantage, 'rrl': 'reasoning'}]
    if sample['closed'] and action_advantage:
        row = {'id': name + '-a', 'invocation': name, 'index': 1,
               'prompt': np.concatenate([np.asarray(context, np.int32), rows[0]['sample']]),
               'sample': np.asarray(action, np.int32), 'advantage': action_advantage, 'rrl': 'action'}
        if action_weight is not None:
            row['weight'] = np.asarray(action_weight, np.float32)
        rows.append(row)
    return rows


def rrl_episodes(groups, reasoning_weight, action_weight, unclosed='drop'):
    """The learner items of the step's reasoning RL groups: [(episode, role)] (one episode per sample: its think block
    at ``reasoning_weight * A`` and the teacher action at ``action_weight``; a group without spread trains its actions
    only) and the telemetry counters."""
    items, stats, lengths, centred = [], collections.Counter(), [], []
    for group in groups:
        samples = group['samples']
        if group.get('objective') == 'twin-fkl':
            # myopic: every sample's think block (closed or cut by the cap: each position is an on-policy prefix) at
            # the forward KL to its twin, weight ``reasoning_weight``; no return-ranked advantage; the teacher action
            # after a closed block as before
            stats['groups'] += 1
            stats['groups_twin'] += 1
            stats['samples'] += len(samples)
            stats['unclosed'] += sum(not s['closed'] for s in samples)
            stats[f"source_{group['source']}"] += 1
            stats[f"mode_{group.get('mode', 'whole')}"] += 1
            stats[f"hard_{bool(group.get('hard'))}"] += 1
            closed = [s for s in samples if s['closed']]
            stats['reward_sum'] += sum(s['reward'] for s in closed)
            stats['reward_count'] += len(closed)
            stats['r0_sum'] += group.get('r0') or 0.
            stats['r0_count'] += group.get('r0') is not None
            for k, sample in enumerate(samples):
                if 'twin_ids' not in sample:
                    stats['twin_missing'] += 1
                    continue
                name = f"rrl-{group['id']}-{k}"
                rows = rrl_rows(name, group['context'], sample, group['action'], group.get('action_weight'), 0.,
                                action_weight)
                rows[0].update(twin_ids=sample['twin_ids'], twin_logp=sample['twin_logp'], twin_weight=reasoning_weight)
                items.append(({'task_id': group['task_id'], 'round': name, 'rows': rows, 'decisions': []}, RRL_ROLE))
                stats['reasoning_tokens'] += len(sample['ids'])
                stats['action_tokens'] += len(group['action']) if len(rows) > 1 else 0
                stats['twin_close_sum'] += sample.get('twin_close_mean', 0.) * len(sample['ids'])
            continue
        advantages, reason = rrl_advantages([s['return'] for s in samples], [s['closed'] for s in samples],
                                            group['min_std'], unclosed)
        stats['groups'] += 1
        stats['samples'] += len(samples)
        stats['unclosed'] += sum(not s['closed'] for s in samples)
        stats['high'] += int(any(s.get('high') for s in samples))
        stats[f"source_{group['source']}"] += 1
        stats[f"mode_{group.get('mode', 'whole')}"] += 1
        stats[f"hard_{bool(group.get('hard'))}"] += 1
        closed = [s for s in samples if s['closed']]
        stats['reward_sum'] += sum(s['reward'] for s in closed)
        stats['reward_count'] += len(closed)
        stats['r0_sum'] += group.get('r0') or 0.
        stats['r0_count'] += group.get('r0') is not None
        if len(closed) >= 2:
            r = [s['return'] for s in closed]
            m = sum(r) / len(r)
            stats['return_std_sum'] += math.sqrt(sum((x - m) ** 2 for x in r) / len(r))
            stats['return_std_groups'] += 1
            lengths += [len(s['ids']) for s in closed]
            centred += [x - m for x in r]
        if advantages is None:
            stats[f'groups_{reason}'] += 1
            if not action_weight or not closed:
                continue
            advantages = [0.] * len(samples)
        else:
            stats['groups_trained'] += 1
        for k, (sample, advantage) in enumerate(zip(samples, advantages)):
            if unclosed == 'drop' and not sample['closed']:
                stats['unclosed_dropped'] += 1     # cut by the token cap: neither ranked nor trained
                continue
            name = f"rrl-{group['id']}-{k}"
            rows = rrl_rows(name, group['context'], sample, group['action'], group.get('action_weight'),
                            reasoning_weight * advantage, action_weight)
            if not advantage and len(rows) == 1:
                continue
            items.append(({'task_id': group['task_id'], 'round': name, 'rows': rows, 'decisions': []}, RRL_ROLE))
            stats['reasoning_tokens'] += len(sample['ids'])
            stats['action_tokens'] += len(group['action']) if len(rows) > 1 else 0
    if len(lengths) >= 3 and np.std(lengths) > 0 and np.std(centred) > 0:
        # within-group return against reasoning length: a length pressure the term would put on reasoning shows here
        stats['length_return_corr'] = float(np.corrcoef(lengths, centred)[0, 1])
    return items, stats


def rrl_record(stats, worker_counts, ready):
    """train.jsonl ``reasoning_rl``: groups taken / trained / without spread / none closed / stale, hard groups,
    groups with a high-scoring sample, samples, unclosed, the mean immediate reward and no-reasoning reward r0 (nats
    per action token), the mean within-group return std, reasoning and action tokens, the within-group length-return
    correlation, sources, the worker's cumulative counters and its ready queue."""
    out = {k: int(v) for k, v in stats.items() if k.startswith(('groups', 'source_', 'mode_', 'samples', 'unclosed',
                                                                'high', 'hard_')) or k.endswith('_tokens')}
    if stats.get('reward_count'):
        out['reward_mean'] = round(stats['reward_sum'] / stats['reward_count'], 4)
    if stats.get('r0_count'):
        out['r0_mean'] = round(stats['r0_sum'] / stats['r0_count'], 4)
    if stats.get('return_std_groups'):
        out['return_std_mean'] = round(stats['return_std_sum'] / stats['return_std_groups'], 4)
    if 'length_return_corr' in stats:
        out['length_return_corr'] = round(stats['length_return_corr'], 3)
    if stats.get('samples'):
        out['reasoning_tokens_per_sample'] = round(stats.get('reasoning_tokens', 0) / stats['samples'], 1)
    if stats.get('groups_twin') and stats.get('reasoning_tokens'):
        # the twin's mean p(</think>) over the trained think blocks (the learner logs the policy's beside it)
        out['twin_close_mean'] = round(stats['twin_close_sum'] / stats['reasoning_tokens'], 5)
    if stats.get('twin_missing'):
        out['twin_missing'] = int(stats['twin_missing'])
    out['worker'] = dict(worker_counts)
    out['ready'] = ready
    return out


def rrl_task_ids(collector, band, candidates):
    """Teacher-demonstration tasks for the reasoning RL term: every task with demonstrations except those the policy
    already solves (current pass rate above the band: the run's latest group, else the measured base rate)."""
    rates = {k: (v['successes'], v['scored']) for k, v in dict(collector.base_rates).items() if v.get('scored')}
    rates.update(dict(collector.own))
    high = band[1]
    return [k for k in candidates if not (k in rates and rates[k][1] and rates[k][0] / rates[k][1] > high)]


class ReasoningRL:
    """The reasoning RL term's search (the comment above ``RRL_ROLE``). ``--reasoning-rl-workers`` threads run while
    collection runs (never during a held measurement on the training actor or the learner phase, when the actor is
    released) and keep up to twice ``--reasoning-rl-groups`` finished groups ready. A trajectory comes from this run's
    JOLT+ teacher successes first (``offer``: the guided run's actions under the unguided prompt), else from verified
    teacher demonstrations (``TeacherDemos``: tasks the policy does not already solve, least used first). Think blocks
    are drawn on SGLang at the run's sampling law (temperature 1, ``--top-p``, the export's banned tokens) with the
    fork's untruncated per-token log-probabilities (``natlang_top_k`` records, as the front captures rollouts), stopped
    at ``</think>``, the reply end or ``--reasoning-rl-max-tokens``; rewards are SGLang input logprobs. The segment's
    first prompt is prefilled once before its samples (SGLang's radix cache keeps LFM3's conv state only at track
    points, so concurrent copies would each prefill it), and each scoring forward leaves the next step's history
    (z, action, tool output) in the radix cache for the draws that continue from it."""

    def __init__(self, args, sglang_url, close_id, stop_ids, banned, markers, collector, demos, log=print, first_group=0,
                 twin=None):
        self.args, self.url, self.close_id, self.collector, self.demos, self.log = (
            args, sglang_url, close_id, collector, demos, log)
        # the privileged twin's turn (``twin_context``: header ids, footer ids, <|im_end|>, newline); None: no twin
        self.twin = twin if rrl_objective(args) == 'twin-fkl' else None
        if rrl_objective(args) == 'twin-fkl' and twin is None:
            raise ValueError('--reasoning-rl-objective twin-fkl needs the twin turn ids')
        self.stops = sorted({close_id, *stop_ids})
        self.stop_ids = set(stop_ids)
        self.banned, self.markers = sorted(banned), set(markers)   # Neuralese role markers: the front's protocol
        self.sources = [s for s in args.reasoning_rl_sources.split(',') if s]
        self.ready, self.sd_ready = [], []
        self.jolt, self.queued = collections.deque(maxlen=256), collections.deque()
        self.lock = threading.Lock()
        self.counts = collections.Counter()
        # seeded per resume (the resumed run's first group): one seed for every resume repeated its demo picks, rewrite
        # choices and sampling seeds after each restart
        self.rng = random.Random(f'{args.seed}:reasoning-rl:{first_group}')
        self.ids = 0
        self.demo_tasks = sorted(demos.by_task) if demos is not None else []

    def start(self, stop):
        for index in range(self.args.reasoning_rl_workers):
            threading.Thread(target=self._loop, args=(stop,), daemon=True, name=f'reasoning-rl-{index}').start()

    def segments(self, episode, source):
        """A teacher episode's searches, one per run of usable calls (``teacher_segments``: one per invocation, split
        only where a call cannot be a step), each from its first call to its end (``--reasoning-rl-calls`` N > 0 caps
        it, opt-in). With probability ``--reasoning-rl-rewrite-share`` a run is rewritten in parallel (``rewrite``),
        else walked step by step (``run``: mode whole). The search itself stops a walk whose history no longer leaves
        room to reason."""
        runs = teacher_segments(episode['rows'], self.close_id, self.markers, self.counts,
                                self.args.reasoning_rl_teacher_history)
        out = []
        for run in runs:
            calls = run[:self.args.reasoning_rl_calls] if self.args.reasoning_rl_calls > 0 else run
            mode = 'rewrite' if self.rng.random() < rrl_rewrite_share(self.args) else 'whole'
            if len(calls[0]['prompt']) + len(calls[0]['reply']) + MIN_REASONING_ROOM > self.args.max_sequence_tokens:
                self.counts['segment_too_long'] += 1
                continue
            out.append({'calls': calls, 'source': source, 'task_id': episode['task_id'], 'key': episode.get('round'),
                        'mode': mode})
        if not out:
            self.counts['episode_no_segment'] += 1
        self.counts['segment_calls'] += sum(len(seg['calls']) for seg in out)
        return out

    def offer(self, episodes):
        """A scored group's JOLT+ teacher successes become segments of this term (newest first)."""
        if 'jolt' not in self.sources:
            return
        for episode in episodes:
            if episode.get('role') == 'teacher' and episode.get('quality') == 1.0:
                segments = self.segments(episode, 'jolt')
                with self.lock:
                    self.jolt.extend(segments)
                    self.counts['jolt_offered'] += len(segments)

    def _next(self):
        with self.lock:
            if self.jolt:
                return self.jolt.pop()
            if self.queued:
                return self.queued.popleft()
        if self.demos is None or 'demos' not in self.sources:
            return None
        tasks = rrl_task_ids(self.collector, self.collector.args.pass_band, self.demo_tasks)
        if not tasks:
            return None
        stats = collections.Counter()
        episodes = self.demos.take(self.rng.sample(tasks, min(len(tasks), 64)), 1, stats, use='rrl_uses',
                                   prefix='rrl_demo')
        self.counts['demo_render_error'] += stats.get('rrl_demo_render_error', 0)
        segments = self.segments(episodes[0], 'demos') if episodes else []
        if len(segments) > 1:
            with self.lock:
                self.queued.extend(segments[1:])   # the episode's other invocations come next, after any JOLT+ one
        return segments[0] if segments else None

    def _idle(self):
        c = self.collector
        return (c.paused.is_set() or c.active_slots <= 0 or c.stop.is_set()
                or len(self.ready) >= 2 * self.args.reasoning_rl_groups)

    def _loop(self, stop):
        while not stop.is_set():
            if self._idle():
                stop.wait(2)
                continue
            segment = self._next()
            if segment is None:
                stop.wait(10)
                continue
            started = time.time()
            try:
                if segment.get('mode') == 'rewrite':
                    groups, carried = [], self.rewrite(segment)
                else:
                    groups, carried = self.run(segment, emit=self._emit)
            except Exception as error:   # a failed request loses the rest of this trajectory only
                if stop.is_set() or self.collector.stop.is_set():
                    # The run is stopping: the actor goes with the unit (or after train() returns) and takes the
                    # in-flight requests with it. Not an actor failure; the unfinished trajectory is dropped as any
                    # unfinished episode is (grpo-v4-run2: all 114 'request failed' events were these).
                    self.counts['abandoned_at_stop'] += 1
                    return
                self.counts['failures'] += 1
                if self.counts['failures'] <= 20 or self.counts['failures'] % 100 == 0:
                    self.log({'event': 'reasoning rl request failed', 'failures': self.counts['failures'],
                              'error': repr(error)[:400]})
                stop.wait(5)
                continue
            with self.lock:
                if carried is not None:
                    self.sd_ready.append(carried)
                self.counts[f"segments_{segment.get('mode', 'whole')}"] += 1
                self.counts['groups'] += len(groups)
                self.counts['actor_s'] += round(time.time() - started)

    def _emit(self, group):
        """A finished step's group is ready for the learner at once (a whole trajectory spans learner steps)."""
        with self.lock:
            self.ids += 1
            group.update(id=self.ids, min_std=self.args.reasoning_rl_min_std)
            self.ready.append(group)

    def _post(self, body):
        request = urllib.request.Request(self.url + '/generate', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=7200) as response:   # waits through a learner phase, as episodes do
            return json.loads(response.read())

    def _draw(self, context, room, seed):
        """One think block on ``context``: (ids, actor log-probabilities)."""
        from natlang_neuralese.serve.sglang_actor import TOP_K_KEY, decode_record
        params = {'temperature': 1.0, 'max_new_tokens': room, 'stop_token_ids': self.stops, 'skip_special_tokens': False,
                  'no_stop_trim': True, 'sampling_seed': seed, 'custom_params': {TOP_K_KEY: 1}}
        if self.args.top_p < 1:
            params['top_p'] = float(self.args.top_p)
        if self.banned:
            params['logit_bias'] = {str(t): -1e30 for t in self.banned}
        result = self._post({'input_ids': list(context), 'sampling_params': params})
        ids, records = result.get('output_ids'), (result.get('meta_info') or {}).get(TOP_K_KEY)
        if not isinstance(ids, list) or not isinstance(records, list) or len(ids) != len(records):
            raise RuntimeError('SGLang returned no per-token natlang_top_k records')
        return [int(i) for i in ids], [decode_record(r, 1)[0] for r in records]

    def _score(self, ids, spans):
        """Mean token log-probability of each action span of ``ids`` (one forward: SGLang input logprobs)."""
        start = spans[0][0] - 1
        result = self._post({'input_ids': ids, 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1},
                             'return_logprob': True, 'logprob_start_len': start})
        logps = [lp for lp, _, _ in result['meta_info']['input_token_logprobs']]
        if len(logps) != len(ids) - start:
            raise RuntimeError('SGLang returned the wrong number of input logprobs')
        out = []
        for position, length in spans:
            values = logps[position - start:position - start + length]
            if len(values) != length or any(v is None for v in values):
                raise RuntimeError('SGLang returned no logprob for an action token')
            out.append(float(sum(values) / length))
        return out

    def _twin(self, context, action, samples):
        """The privileged twin's next-token distributions over each sample's think block (``twin_distributions``): the
        policy itself on ``context`` with the teacher action given in a training guidance turn (``twin_context``), the
        same SGLang weights the draws came from, temperature 1 and no nucleus (input log-probabilities). The twin's
        prompt is prefilled once first (SGLang keeps LFM3's conv state only at track points, so concurrent copies would
        each prefill it); the samples then score on it in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        twin = self.twin
        action = [int(i) for i in action]
        if action and action[-1] == twin['im_end']:
            action = action[:-1]          # the footer closes the turn
        prompt, found = twin_context(context, twin['header'] + action, twin['footer'], twin['im_end'], twin['newline'])
        self.counts['twin_no_system_turn'] += int(not found)
        self._post({'input_ids': prompt, 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}})

        def one(sample):
            ids = list(sample['ids'])
            result = self._post({'input_ids': prompt + ids, 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1},
                                 'return_logprob': True, 'logprob_start_len': len(prompt) - 1,
                                 'top_logprobs_num': self.args.reasoning_rl_twin_top_k,
                                 'token_ids_logprob': [self.close_id]})
            return twin_distributions(result['meta_info'], self.close_id, len(ids))
        with ThreadPoolExecutor(max(1, len(samples))) as pool:
            for sample, (ids, logp, close) in zip(samples, pool.map(one, samples)):
                sample['twin_ids'], sample['twin_logp'] = ids, logp
                sample['twin_close_mean'] = float(close.mean()) if len(close) else 0.
                self.counts['twin_tokens'] += len(close)

    def closed_form(self, ids):
        """A think block as later history: as sampled when it closed; else without a trailing stop token and closed
        with ``</think>`` (context, never a sample)."""
        if ids and ids[-1] == self.close_id:
            return list(ids)
        return (list(ids[:-1]) if ids and ids[-1] in self.stop_ids else list(ids)) + [self.close_id]

    def _samples(self, context, calls, t, count, room):
        """``count`` think blocks drawn on ``context`` and scored with the segment's rest (``future_ids``)."""
        from concurrent.futures import ThreadPoolExecutor
        seeds = [self.rng.randrange(1, 2**31) for _ in range(count)]
        with ThreadPoolExecutor(count) as pool:
            drawn = list(pool.map(lambda seed: self._draw(context, room, seed), seeds))
        samples = []
        for ids, logp in drawn:
            self.counts['generated_tokens_whole'] += len(ids)
            if not ids:
                self.counts['empty_samples'] += 1
            elif self.markers & set(ids):   # a Neuralese marker in reasoning: the front's protocol, not this term's
                self.counts['marker_samples'] += 1
            elif not np.isfinite(logp).all():
                self.counts['nonfinite_samples'] += 1
            else:
                samples.append({'ids': ids, 'logp': logp, 'closed': ids[-1] == self.close_id})

        def score(sample):
            ids, spans = future_ids(context + self.closed_form(sample['ids']), calls, t, self.close_id,
                                    self.args.reasoning_rl_horizon or None, self.args.max_sequence_tokens)
            return self._score(ids, spans)
        with ThreadPoolExecutor(max(1, len(samples))) as pool:
            for sample, rewards in zip(samples, pool.map(score, samples)):
                sample['reward'] = rewards[0]
                sample['return'] = discounted(rewards, self.args.reasoning_rl_gamma)
                sample['high'] = sample['closed'] and rewards[0] >= self.args.reasoning_rl_high_logp
        self.counts['samples'] += len(samples)
        self.counts['closed'] += sum(s['closed'] for s in samples)
        return samples

    def run(self, segment, emit=None):
        """Search one trajectory (blocking): (learner groups, the carried chain as an SD episode or None). Each group
        records the policy version its draws started on, and goes to ``emit`` as soon as its step is done."""
        a, calls = self.args, segment['calls']
        prompt = [int(i) for i in calls[0]['prompt']]
        ids, spans = future_ids(prompt + [self.close_id], calls, 0, self.close_id, None, self.args.max_sequence_tokens)
        r0 = self._score(ids, spans)      # the teacher-forced pass without reasoning: every step's difficulty at once
        calls = calls[:len(r0)]           # steps past the context limit even without reasoning never fit
        beams = [{'context': prompt, 'path': [], 'return': 0.}]
        groups = []
        for t, call in enumerate(calls):
            hard = r0[t] < a.reasoning_rl_hard_logp
            budget = a.reasoning_rl_max_samples if hard else a.reasoning_rl_samples
            candidates = []
            for beam in beams:
                # the reasoning room shrinks as the carried history grows; below MIN_REASONING_ROOM the trajectory ends
                room = min(a.reasoning_rl_max_tokens, a.max_sequence_tokens - len(beam['context']) - len(call['action']) - 1)
                if room < min(MIN_REASONING_ROOM, a.reasoning_rl_max_tokens):
                    self.counts['context_full'] += 1
                    continue
                version = self.collector.version
                samples, drawn = [], 0
                if self.twin is not None:
                    budget = a.reasoning_rl_wave   # myopic: every sample trains; no search for a high-scoring one
                while drawn < budget:   # waves until one scores high (stop early) or the budget is spent
                    wave = min(a.reasoning_rl_wave, budget - drawn)
                    samples += self._samples(beam['context'], calls, t, wave, room)
                    drawn += wave
                    if any(s['high'] for s in samples):
                        break
                if not samples:
                    continue
                if self.twin is not None:
                    self._twin(beam['context'], call['action'], samples)
                self.counts['hard_samples' if hard else 'samples_ordinary'] += len(samples)
                groups.append({'task_id': segment['task_id'], 'key': segment['key'], 'source': segment['source'],
                               'mode': segment.get('mode', 'whole'), 'objective': rrl_objective(a),
                               'step': t, 'hard': hard, 'r0': r0[t], 'context': np.asarray(beam['context'], np.int32),
                               'action': call['action'], 'action_weight': call.get('action_weight'),
                               'samples': samples, 'version': version})
                if emit is not None:
                    emit(groups[-1])
                candidates += [(beam, s) for s in samples]
            if not candidates:
                break
            # carry the best: closed first, then by return; the history the next step reasons on
            candidates.sort(key=lambda item: (item[1]['closed'], item[1]['return']), reverse=True)
            beams = []
            for beam, sample in candidates[:max(1, a.reasoning_rl_beam)]:
                context = beam['context'] + self.closed_form(sample['ids']) + [int(i) for i in call['action']]
                if t + 1 < len(calls):
                    context += [int(i) for i in call['suffix']]
                beams.append({'context': context, 'path': beam['path'] + [(beam['context'], sample)],
                              'return': beam['return'] + sample['reward']})
        carried = (self.carried(segment, max(beams, key=lambda b: b['return'])['path'])
                   if beams and groups and rrl_carry_sd(a) else None)
        return groups, carried

    def carried(self, segment, path):
        """The best chain as an SD episode: each step's think block (trained when it closed and scored high, else
        context) and the teacher action after it (trained). None without a high-scoring step."""
        if not any(sample['high'] for _, sample in path):
            return None
        rows = []
        name = f"rrl-carried-{segment['key']}-{self.rng.randrange(2**31)}"
        for t, ((context, sample), call) in enumerate(zip(path, segment['calls'])):
            z = self.closed_form(sample['ids'])
            reasoning = {'id': f'{name}-{t}', 'invocation': name, 'index': 2 * t,
                         'prompt': np.asarray(context, np.int32), 'sample': np.asarray(z, np.int32),
                         'logp': np.asarray(list(sample['logp'][:len(z) - 1]) + [sample['logp'][-1] if sample['closed']
                                                                                 else 0.], np.float32)}
            if not sample['high']:   # (an unclosed block is never high: its forced close is never trained)
                reasoning['weight'] = np.zeros(len(z), np.float32)
            action = {'id': f'{name}-{t}-a', 'invocation': name, 'index': 2 * t + 1,
                      'prompt': np.concatenate([reasoning['prompt'], reasoning['sample']]),
                      'sample': np.asarray(call['action'], np.int32), 'teacher': True}
            if call.get('action_weight') is not None:
                action['weight'] = np.asarray(call['action_weight'], np.float32)
            rows += [reasoning, action]
        self.counts['carried_chains'] += 1
        return {'task_id': segment['task_id'], 'round': name, 'rows': rows, 'decisions': [], 'reward': 1.,
                'quality': 1., 'role': 'sd', 'sd_source': 'search'}

    def _rewrite_score(self, calls, t, think):
        ids, spans = future_ids([int(i) for i in calls[t]['prompt']] + list(think), calls, t, self.close_id,
                                self.args.reasoning_rl_horizon or None, self.args.max_sequence_tokens)
        return discounted(self._score(ids, spans), self.args.reasoning_rl_gamma)

    def rewrite(self, segment):
        """Parallel rewrite (TRICE's MCMC-EM over rationales, STaR rationalization): every position keeps its recorded
        history, so the think blocks of all positions are independent and are rewritten together. Seeded with the
        teacher's own think block at each position; each of ``--reasoning-rl-rewrite-rounds`` rounds cuts every
        position's block at a random point (just before its ``</think>``: an extension) and draws the student's
        continuation (at most ``--reasoning-rl-rewrite-tokens``; one that does not close is no candidate), all
        positions batched; each candidate is scored by one teacher-forced forward (its action, then up to
        ``--reasoning-rl-horizon`` later actions after empty think blocks, ``--reasoning-rl-gamma``) and kept when it
        scores better (``--reasoning-rl-rewrite-accept`` best) or by a Metropolis step at
        ``--reasoning-rl-rewrite-temperature``. A cut only ever keeps a prefix, so a block is the teacher's first tokens
        then the student's, each student token drawn on exactly the tokens before it. The result is an SD episode
        (source ``rewrite``): per position whose block now scores better than the teacher's seed, the student's tokens
        are the target, with the actor log-probabilities they were drawn with, and the surviving teacher prefix is
        prompt (never trained: training teacher reasoning cut held-v3 by 16 points). None when no position improved."""
        from concurrent.futures import ThreadPoolExecutor
        a, calls = self.args, segment['calls']
        pool_size = max(1, min(len(calls), 16))
        with ThreadPoolExecutor(pool_size) as pool:
            current = [list(int(i) for i in c['think']) for c in calls]
            logps = [[None] * len(think) for think in current]          # None: a teacher token
            seed = list(pool.map(lambda t: self._rewrite_score(calls, t, current[t]), range(len(calls))))
            score = list(seed)

            def propose(t):
                prompt, think = [int(i) for i in calls[t]['prompt']], current[t]
                cut = self.rng.randrange(len(think))                    # think ends with </think>: cut before it at most
                room = min(a.reasoning_rl_rewrite_tokens,
                           a.max_sequence_tokens - len(prompt) - cut - len(calls[t]['action']) - 1)
                if room < 1:
                    return None
                ids, logp = self._draw(prompt + think[:cut], room, self.rng.randrange(1, 2**31))
                self.counts['generated_tokens_rewrite'] += len(ids)
                if not ids or ids[-1] != self.close_id or self.markers & set(ids) or not np.isfinite(logp).all():
                    self.counts['rewrite_unclosed' if ids and ids[-1] != self.close_id else 'rewrite_invalid'] += 1
                    return None
                candidate = think[:cut] + ids
                return candidate, logps[t][:cut] + list(logp), self._rewrite_score(calls, t, candidate)
            for _ in range(a.reasoning_rl_rewrite_rounds):
                for t, proposal in enumerate(pool.map(propose, range(len(calls)))):
                    if proposal is None:
                        continue
                    candidate, logp, value = proposal
                    self.counts['rewrite_proposals'] += 1
                    if a.reasoning_rl_rewrite_accept == 'metropolis':
                        accept = math.log(max(self.rng.random(), 1e-300)) < (value - score[t]) / max(
                            a.reasoning_rl_rewrite_temperature, 1e-9)
                    else:
                        accept = value > score[t]
                    if accept:
                        current[t], logps[t], score[t] = candidate, logp, value
                        self.counts['rewrite_accepted'] += 1
        rows = []
        name = f"rrl-rewrite-{segment['key']}-{self.rng.randrange(2**31)}"
        for t, call in enumerate(calls):
            teacher = next((i for i, lp in enumerate(logps[t]) if lp is not None), len(logps[t]))
            if score[t] <= seed[t] or teacher == len(current[t]):
                continue
            self.counts['rewrite_improved'] += 1
            self.counts['rewrite_gain_milli'] += round(1000 * (score[t] - seed[t]))
            self.counts['rewrite_student_tokens'] += len(current[t]) - teacher
            rows.append({'id': f'{name}-{t}', 'invocation': f'{name}-{t}', 'index': 0,
                         'prompt': np.asarray([int(i) for i in call['prompt']] + current[t][:teacher], np.int32),
                         'sample': np.asarray(current[t][teacher:], np.int32),
                         'logp': np.asarray(logps[t][teacher:], np.float32)})
        if not rows:
            return None
        return {'task_id': segment['task_id'], 'round': name, 'rows': rows, 'decisions': [], 'reward': 1.,
                'quality': 1., 'role': 'sd', 'sd_source': 'rewrite'}

    def save(self, path):
        """The ready groups and carried chains to ``path`` (``sd_buffer_save``), so a stop does not discard up to
        2 x --reasoning-rl-groups finished searches (grpo-v4-run2 stopped 14 times in 40 steps)."""
        if path is None:
            return
        with self.lock:
            ready, carried = list(self.ready), list(self.sd_ready)
        sd_buffer_save(path, [{'version': g['version'], 'group': g} for g in ready]
                       + [{'version': c.get('version', -1), 'carried': c} for c in carried])

    def restore(self, path, step):
        """The saved groups and chains sampled before ``step`` policy versions were reached (a policy the resumed
        state never served is gone: its samples are not this run's); returns how many."""
        items = sd_buffer_load(path, step + 1)   # actor versions 0..step are the resumed run's policies
        with self.lock:
            self.ready += [item['group'] for item in items if 'group' in item]
            self.sd_ready += [item['carried'] for item in items if 'carried' in item]
        return len(items)

    def take(self, step, count, max_staleness):
        """The step's groups: the oldest ready ones within ``max_staleness`` policy versions, at most ``count``; staler
        ones are dropped. Also the carried chains finished since (for the SD buffer). Returns (groups, dropped, chains)."""
        with self.lock:
            fresh = [g for g in self.ready if step - g['version'] <= max_staleness]
            dropped = len(self.ready) - len(fresh)
            taken, self.ready = fresh[:count], fresh[count:]
            chains, self.sd_ready = self.sd_ready, []
            self.counts['stale_dropped'] += dropped
        return taken, dropped, chains


class ReplayCorpus:
    """A fixed corpus of per-call chains trained in every step beside the rollouts (``--replay-corpus``): the operator
    rows (scripts/neuralese_data/operator_rows.py merge) or the whole-graph Neuralese episodes (sft_sources
    neuralese-graph). A unit is all chains of one ``task`` key (an operator case: its operator calls and the consumer
    importing them through pass-through producer documents; or one Neuralese episode). Step ``n`` takes units
    ``[n*K, n*K+K)`` of an endless sequence of seeded per-epoch shuffles, so a resume continues without saved state.
    The chains are ratio-free (role ``replay``: no actor log-probabilities, rho = 1) at the corpus weight; their rows'
    own weights (the operator seed weight) multiply the term, and the KL term applies as to every chain.

    With a ``tokenizer`` and ``think=context`` (the default), a row whose target opens with the template's empty think
    block (a seed or exact answer written without reasoning) has that block at weight 0 (``exp0_sft.mask_empty_think``):
    operators-v1 trained ``</think>`` at the first position of all its 11,558 rows. ``think=train`` opts out."""

    def __init__(self, spec, held, seed, log=print, tokenizer=None):
        import pickle
        name, _, rest = spec.partition('=')
        path, *options = rest.split(',')
        if not name or not path:
            raise SystemExit(f'--replay-corpus is NAME=PKL[,weight=W][,units=K][,think=context|train], not {spec!r}')
        self.name, self.path, self.weight, self.units, self.think = name, path, 0.05, 8, 'context'
        for option in options:
            key, _, value = option.partition('=')
            if key == 'weight':
                self.weight = float(value)
            elif key == 'units':
                self.units = int(value)
            elif key == 'think' and value in ('context', 'train'):
                self.think = value
            else:
                raise SystemExit(f'--replay-corpus option {option!r}: weight=W, units=K or think=context|train')
        with open(path, 'rb') as handle:
            chains = pickle.load(handle)['chains']
        import exp0_sft
        self.think_masked = (exp0_sft.mask_empty_think(chains, tokenizer)
                             if tokenizer is not None and self.think == 'context' else 0)
        # nothing after the end of turn trains (operators-v1 rows end in the template's separator after <|im_end|>)
        self.after_end_masked = exp0_sft.mask_after_end(chains, tokenizer) if tokenizer is not None else 0
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        groups = collections.defaultdict(list)
        for chain in chains:
            groups[str(chain['task'])].append(chain)
        dropped = [key for key in groups if key in held]
        for key in dropped:
            del groups[key]
        self.keys = sorted(groups)
        self.groups = groups
        self.seed = seed
        if not self.keys or self.weight <= 0 or self.units <= 0:
            raise SystemExit(f'--replay-corpus {name}: no units, or weight/units not positive')
        log({'event': 'replay corpus', 'name': name, 'path': path, 'sha256': self.sha256, 'units': len(self.keys),
             'chains': sum(len(v) for v in groups.values()), 'held_dropped': len(dropped), 'weight': self.weight,
             'units_per_step': self.units, 'think': self.think if tokenizer is not None else 'train (no tokenizer)',
             'think_masked_rows': self.think_masked, 'after_end_masked_rows': self.after_end_masked})

    def take(self, step):
        """The units of ``step``: [(key, [chains])], each chain a fresh copy of its corpus chain's fields."""
        import random as _random
        out, count = [], len(self.keys)
        for position in range(step * self.units, (step + 1) * self.units):
            epoch, index = divmod(position, count)
            order = list(self.keys)
            _random.Random(f'{self.seed}:{self.name}:{epoch}').shuffle(order)
            key = order[index]
            out.append((key, [dict(chain) for chain in self.groups[key]]))
        return out


TEACHER_FIELDS = ('groups', 'episodes', 'successes', 'episodes_trained', 'calls_student_form',
                  'privileged_tokens_removed', 'policy_calls_removed')
SD_FIELDS = ('added', 'episodes', 'tokens', 'buffer', 'skip_imports', 'own_episodes', 'own_tokens', 'search_added',
             'search_episodes', 'search_tokens', 'rewrite_added', 'rewrite_episodes', 'rewrite_tokens', 'teacher_episodes',
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


ACTOR_MISMATCH_NATS = 5.        # |learner logp - actor's recorded logp| of a sampled token beyond this: an engine mismatch
ACTOR_MISMATCH_FLAG_PER_M = 10.  # ordinary sampled tokens below -ACTOR_MISMATCH_NATS per million loss tokens that flag


def actor_mismatch_counts(log_rho, consumer=None, nats=ACTOR_MISMATCH_NATS):
    """Per chain: sampled tokens whose learner logp lies more than ``nats`` below (``below``) or above (``above``) the
    actor's recorded logp, ``below`` again outside calls that read imported Neuralese state (``below_ordinary``), and the
    smallest log rho. The 2026-11-03 overlapped-scheduler defect drew single tokens from a wrong distribution (actor
    p 0.53, learner -24 nats) in 2-6% of turns, i.e. ~15-45 such tokens per million; the learner gives them weight ~0
    (rho), so only this count shows them."""
    low = log_rho < -nats
    out = {'below': int(low.sum()), 'above': int((log_rho > nats).sum()),
           'min': float(log_rho.min()) if log_rho.numel() else 0.}
    out['below_ordinary'] = int((low & ~consumer).sum()) if consumer is not None else out['below']
    return out


def actor_mismatch_summary(sums, tokens):
    """(train.jsonl ``actor_mismatch``, flag): the step's engine-mismatch tokens (``actor_mismatch_counts``) and the
    rate of ordinary ones per million loss tokens; the flag (``actor_mismatch_shift``) is set at
    ACTOR_MISMATCH_FLAG_PER_M with at least 3 tokens, the cue to rerun with ``--actor-overlap off`` and compare."""
    per_m = 1e6 * sums['mismatch_below_ordinary'] / max(1, tokens)
    summary = {'below': sums['mismatch_below'], 'above': sums['mismatch_above'],
               'below_ordinary': sums['mismatch_below_ordinary'], 'below_ordinary_per_m': round(per_m, 2),
               'min_log_rho': round(sums.get('mismatch_min', 0.), 3), 'nats': ACTOR_MISMATCH_NATS}
    return summary, bool(sums['mismatch_below_ordinary'] >= 3 and per_m >= ACTOR_MISMATCH_FLAG_PER_M)


def actor_overlap(args):
    """Whether the run's own SGLang actor runs the overlapped scheduler (``--actor-overlap``). ``auto``: on unless held
    evaluations measure on this actor (no ``--held-remote`` and either periodic evals or a start-policy measurement),
    whose pinned eval engine (``engine.json`` expect) serves without it."""
    if args.actor_overlap != 'auto':
        return args.actor_overlap == 'on'
    held_here = not args.held_remote and (args.held_every_hours > 0 or bool(getattr(args, 'held_extra', None))
                                          or not (args.skip_base_held or args.held_baseline))
    return not held_here


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


def episode_unit_counts(episodes, decision_weight, decision_cap=8.):
    """``episodes`` {key: (trained tokens, trained calls, trained decisions)} -> {key: (the episode's loss units, the units
    of each of its decisions)}: its tokens plus its decisions times ``decision_unit``."""
    return {key: (tokens + decisions * decision_unit(tokens, calls, decision_weight, decision_cap),
                  decision_unit(tokens, calls, decision_weight, decision_cap))
            for key, (tokens, calls, decisions) in episodes.items()}


def loss_scales(episodes, norm, decision_weight, decision_cap=8.):
    """Per-episode loss scale factors: ``episodes`` {key: (trained tokens, trained calls, trained decisions)} ->
    {key: (factor of each token term, factor of each decision term)}. An episode's units are its tokens plus its
    decisions times ``decision_unit``. ``episode``: every episode with units weighs 1 / episodes and its terms share
    that weight by units (A_i times the mean over its units, averaged over episodes; a chain's factor is
    1 / (episode units * episodes), so accumulating chain by chain is exact). ``token``: one factor 1 / all units."""
    units = episode_unit_counts(episodes, decision_weight, decision_cap)
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


def malformed_close_targets(chain, advantage, close_id, stop_ids):
    """Bool per target of ``chain`` (torch, or None when there is none): a reply's second and later ``</think>`` and
    the tokens after each up to the action (``exp0_sft.second_close_mask``) where the advantage is positive. Malformed
    output is never reinforced (``--second-close context``); a negative advantage still pushes it down. ``advantage``: a
    number or a per-token tensor (``token_advantage``)."""
    import torch
    import exp0_sft
    masks = [exp0_sft.second_close_mask(row['sample'], close_id, stop_ids) for _, _, row in chain['segs']]
    if all(m is None for m in masks):
        return None
    bad = torch.from_numpy(np.concatenate([np.ones(n, np.float32) if m is None else m
                                           for m, (_, n, _) in zip(masks, chain['segs'])]) == 0)
    if torch.is_tensor(advantage):
        return bad & (advantage.detach().cpu() > 0)
    return bad if advantage > 0 else torch.zeros_like(bad)


def decision_token_kl(token_logp, ref_logp, beta, estimator, token_scale):
    """The KL term on a sampled decision's chosen option, token by token (``--decision-token-kl``, default): those tokens
    are what the actor emitted, and the option softmax (the decision's policy and KL terms) does not see any token all
    options share (``</think>``, quotes, ``<|im_end|>``), so only this anchors their absolute probabilities, as the KL
    term anchors every sampled chain token (same beta, same per-token loss factor; grpo-v4-run2's policy drifted 6-13
    nats on ``<|im_end|>`` after short answers by step 12, DECISIONS 2026-11-19)."""
    return beta * kl_term(ref_logp - token_logp, estimator).sum() * token_scale


def kl_stats(sums, d, prefix=''):
    """Telemetry of one batch of ``d`` (ref - policy logp): the k3 KL estimate, the k2 mean the loss trains (d^2 / 2),
    the largest single-token k3 term (k3 = e^d - d - 1 grows exponentially in d: one token where the policy sits 14
    nats below the start policy adds e^14 = 1.2M and alone made grpo-v4-run2 step 13's kl_mean 1.14 while |d| stayed
    at 0.058; ``kl_k3_max_share`` says how much of the k3 sum one token is), |d|, its max, the most positive d (policy
    below the reference) and the units past 5 nats. ``kl_n`` counts the values: the logged means divide by it, not by the
    step's loss units, because a sampled decision adds its option-softmax d and one d per chosen-option token for its
    one unit (the 2026-11-22 LoRA bench's step-6 kl_mean of 860 was ~170 values over 24 units). ``prefix`` also sums
    k2 and the count under ``<prefix>kl_k2`` / ``<prefix>kl_n`` (``decision_``: the decision-level option-softmax KL)."""
    d = d.detach()
    k3 = d.exp() - d - 1
    if prefix:
        sums[prefix + 'kl_k2'] += (0.5 * d * d).sum().item()
        sums[prefix + 'kl_n'] += d.numel()
    sums['kl_n'] += d.numel()
    sums['kl'] += k3.sum().item()
    sums['kl_k2'] += (0.5 * d * d).sum().item()
    if d.numel():
        sums['kl_k3_max'] = max(sums['kl_k3_max'], k3.max().item())
        sums['ref_dlogp_pos_max'] = max(sums['ref_dlogp_pos_max'], d.max().item())
    sums['ref_abs_dlogp'] += d.abs().sum().item()
    sums['ref_dlogp_over_5'] += int((d.abs() > 5).sum())
    sums['ref_dlogp_max'] = max(sums['ref_dlogp_max'], d.abs().max().item() if d.numel() else 0.)


def head_rows(weight, ids, trainable=None):
    """FP32 rows of the effective tied head at ``ids``: the frozen head, with the learner's trainable Neuralese rows
    (``trainable``: ``(token_ids, rows)``) in place at their ids (differentiable in the rows), as ``lfm3_fast.linear_nll``
    computes the logits the loss uses."""
    import torch
    out = weight[ids].float()
    if trainable is None:
        return out
    token_ids, rows = trainable
    match = ids.unsqueeze(-1) == token_ids.to(ids.device)
    if not bool(match.any()):
        return out
    which = match.to(torch.int64).argmax(-1)
    return torch.where(match.any(-1).unsqueeze(-1), rows.float()[which], out)


def head_override(trainable_rows):
    """``head_rows``' ``trainable`` argument of an ``Lfm3Learner``'s rows (None without trainable rows)."""
    return None if trainable_rows is None else (trainable_rows.token_ids, trainable_rows.rows)


def close_probability(h, positions, weight, targets, target_logp, close_id, chunk=4096, trainable=None):
    """p(``</think>``) at each target row (no grad): exp(z_close - z_target + log p(target)), i.e. the softmax's own
    normalizer recovered from the target's logit and the log-probability the loss already computed (two dot products a
    row, no second softmax over the vocabulary). ``h[positions]`` are the hidden rows that predict ``targets``; the head
    is the effective one (``head_rows``: trained Neuralese rows in place)."""
    import torch
    with torch.no_grad():
        w_close = head_rows(weight, torch.tensor([close_id], device=weight.device), trainable)[0]
        out = []
        for a in range(0, positions.shape[0], chunk):
            rows = h[positions[a:a + chunk]].float()
            z_y = (rows * head_rows(weight, targets[a:a + chunk], trainable)).sum(-1)
            out.append((rows @ w_close - z_y + target_logp[a:a + chunk].detach().float()).exp().clamp(max=1.))
        return torch.cat(out) if out else torch.zeros(0, device=h.device)


def close_push(coefficient, targets, p_close, close_id, mask):
    """First-order push of one term on the ``</think>`` logit over the ``mask`` positions (reasoning tokens):
    sum_t c_t (1[y_t = close] - p_close_t), c_t the term's -dLoss/dlogp(y_t) (policy: weight * A * token factor; KL:
    beta * d for k2). Positive: the step raises the close logit where the policy reasons. On-policy the unweighted sum is
    zero in expectation, so a term's push is its correlation of c with reasoning that closes earlier or later than the
    policy's own hazard predicts (grpo-v4-run2 2026-11-20: the probe's hazard sits at the points right after the policy
    names its next action)."""
    import torch
    with torch.no_grad():
        delta = (targets == close_id).float() - p_close
        return float((coefficient.float() * delta)[mask].sum())


def _twin_fkl_chunk(rows, weight, ids, teacher_logp, valid, targets, target_logp, row_ids=None, row_values=None):
    """Forward KL of one chunk (see ``twin_forward_kl``); ``rows`` the hidden rows predicting ``targets``."""
    import torch
    trainable = None if row_ids is None else (row_ids, row_values)
    rows = rows.float()
    z_y = (rows * head_rows(weight, targets, trainable)).sum(-1)
    z = torch.einsum('nd,nkd->nk', rows, head_rows(weight, ids, trainable))
    log_student = z - (z_y - target_logp.float())[:, None]           # log p_S(v) = z_v - log Z, log Z = z_y - log p_S(y)
    p_teacher = teacher_logp.float().exp() * valid
    p_student = log_student.clamp(max=0.).exp() * valid
    teacher_tail = (1. - p_teacher.sum(-1)).clamp(min=0.)
    student_tail = (1. - p_student.sum(-1)).clamp(min=1e-12)
    inner = torch.where(valid > 0, p_teacher * (teacher_logp.float() - log_student), torch.zeros_like(log_student))
    tail = torch.where(teacher_tail > 0, teacher_tail * (teacher_tail.clamp(min=1e-12).log() - student_tail.log()),
                       torch.zeros_like(teacher_tail))
    return inner.sum(-1) + tail


def twin_forward_kl(h, positions, weight, targets, target_logp, ids, teacher_logp, chunk=1024, trainable=None):
    """Token-level forward KL(twin || policy) at each target row (reasoning RL's myopic objective, Strauss, Fowl & Razin
    arXiv 2611.23397 eq. 3 with D = FKL): the teacher is the policy's privileged twin, given as its top-k ids and
    log-probabilities (``ids``/``teacher_logp`` (n, k), ``-inf`` pads). The KL is exact on the partition {the k ids, the
    rest}: the twin's mass outside its top k is matched as one bucket, so tokens the twin did not list are neither
    pushed to zero (a renormalized top-k teacher would sharpen the student) nor ignored. The student's log-probability of
    an id needs no second softmax: ``z_v - z_y + log p(y)`` from the target's logit and the log-probability the loss
    already computed (as ``close_probability``), so the cost is k dot products a row, chunk-checkpointed. Gradient on a
    listed logit: p_S(v) - p_T(v) (the paper's eq. 4 for FKL: it vanishes only where the two agree)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    valid = torch.isfinite(teacher_logp).float()
    safe = torch.where(valid > 0, teacher_logp, torch.zeros_like(teacher_logp))
    out = []
    for a in range(0, positions.shape[0], chunk):
        part = slice(a, a + chunk)
        args = (h[positions[part]], weight, ids[part].long(), safe[part], valid[part], targets[part], target_logp[part],
                *(trainable or (None, None)))
        out.append(checkpoint(_twin_fkl_chunk, *args, use_reentrant=False) if torch.is_grad_enabled()
                   else _twin_fkl_chunk(*args))
    return torch.cat(out) if out else torch.zeros(0, device=h.device)


def twin_rows(chain, device):
    """(ids (n, k), twin log-probabilities (n, k), scale (n,)) of ``chain``'s n targets for ``twin_forward_kl``, or None
    when no row carries a twin. Rows with ``twin_ids``/``twin_logp`` (reasoning RL's sampled think blocks) get their
    row's ``twin_weight`` as scale; every other target has scale 0 and ``-inf`` pads."""
    import torch
    rows = [(length, row) for _, length, row in chain['segs']]
    if not any('twin_ids' in row for _, row in rows):
        return None
    width = max(np.asarray(row['twin_ids']).shape[1] for _, row in rows if 'twin_ids' in row)
    ids, logp, scale = [], [], []
    for length, row in rows:
        if 'twin_ids' in row:
            i, lp = np.asarray(row['twin_ids'], np.int64), np.asarray(row['twin_logp'], np.float32)
            pad = width - i.shape[1]
            ids.append(np.pad(i, ((0, 0), (0, pad))))
            logp.append(np.pad(lp, ((0, 0), (0, pad)), constant_values=-np.inf))
            scale.append(np.full(length, float(row.get('twin_weight', 1.)), np.float32))
        else:
            ids.append(np.zeros((length, width), np.int64))
            logp.append(np.full((length, width), -np.inf, np.float32))
            scale.append(np.zeros(length, np.float32))
    return (torch.from_numpy(np.concatenate(ids)).to(device), torch.from_numpy(np.concatenate(logp)).to(device),
            torch.from_numpy(np.concatenate(scale)).to(device))


def twin_context(context, header, footer, im_end, newline):
    """The privileged twin's prompt: ``context`` with one system turn (``header`` ids, the teacher action, ``footer``
    ids) inserted after the leading system turn (its ``<|im_end|>`` and the newline after it), where the runtime places
    a training guidance turn (rl/privileged_context ``PrivilegedRendering``); at the start when there is none. Returns
    (ids, whether a leading turn was found)."""
    context = [int(i) for i in context]
    at, found = 0, False
    if im_end in context:
        at, found = context.index(im_end) + 1, True
        if at < len(context) and context[at] == newline:
            at += 1
    return context[:at] + list(header) + list(footer) + context[at:], found


def twin_distributions(meta, close_id, count):
    """(ids (n, k+1) int32, log-probabilities (n, k+1) float32, twin p(``</think>``) (n,)) of the last ``count`` input
    positions from an SGLang ``/generate`` answer with ``top_logprobs_num`` k and ``token_ids_logprob`` [close]: each
    position's top k, plus ``</think>`` when the top k does not list it (a ``-inf`` pad otherwise), so the close
    decision is always matched exactly."""
    tops = (meta.get('input_top_logprobs') or [])[-count:] if count else []
    closes = (meta.get('input_token_ids_logprobs') or [])[-count:] if count else []
    if len(tops) != count or len(closes) != count:
        raise RuntimeError('SGLang returned no input top logprobs for the twin')
    k = max((len(t or ()) for t in tops), default=0)
    ids = np.zeros((count, k + 1), np.int32)
    logp = np.full((count, k + 1), -np.inf, np.float32)
    close = np.zeros(count, np.float32)
    for n, (top, extra) in enumerate(zip(tops, closes)):
        entries = [(float(lp), int(i)) for lp, i, *_ in (top or ()) if lp is not None]
        close_lp = next((float(lp) for lp, i, *_ in (extra or ()) if int(i) == close_id and lp is not None), None)
        if close_lp is None:
            raise RuntimeError('SGLang returned no </think> logprob for the twin')
        if all(i != close_id for _, i in entries):
            entries.append((close_lp, close_id))
        for j, (lp, i) in enumerate(entries[:k + 1]):
            ids[n, j], logp[n, j] = i, lp
        close[n] = math.exp(close_lp)
    return ids, logp, close


def policy_weight(rho, actor_logp, advantage, rho_cap, tail_floor):
    """(weight of the policy term, tail mask): the truncated importance ratio, zero where the actor sampled a tail token
    (its sampling logp below ``tail_floor``) in an episode with a positive advantage. Untruncated T=1 sampling draws
    such tokens about once in 200 (p < e^-7: 0.5% of sampled tokens); raising them is the largest-norm part of a
    positive episode's gradient and never the reason it succeeded. The token stays context and keeps its KL term; a
    negative advantage still pushes it down. ``tail_floor`` 0 or None: no mask. ``advantage``: a number or a per-token
    tensor (``token_advantage``)."""
    import torch
    weight = rho.clamp(max=rho_cap)
    if torch.is_tensor(advantage):
        if not tail_floor:
            return weight, torch.zeros_like(weight, dtype=torch.bool)
        masked = (actor_logp < tail_floor) & (advantage > 0)
        return weight.masked_fill(masked, 0.), masked
    if not tail_floor or advantage <= 0:
        return weight, torch.zeros_like(weight, dtype=torch.bool)
    masked = actor_logp < tail_floor
    return weight.masked_fill(masked, 0.), masked


RATIO_FREE_ROLES = ('sd_teacher', 'replay')   # chains without actor log-probabilities (rho = 1)


def student_sampled_targets(chain):
    """Per target of ``chain`` (bool array): whether the student sampled it, so that its likelihood is the one under its
    top-p nucleus. Not sampled by the student: teacher demonstrations (``sd_teacher``), the reasoning-RL term's teacher
    actions (rows ``rrl == 'action'``, and the carried search chains' action rows marked ``teacher``) and replay-corpus rows that are not the student's own (operator seeds, consumers'
    dataset answers; ``exp0_sft.student_sampled`` reads the corpus chain's source). Everything else (episodes, SD replay
    of own and JOLT+ successes, reasoning-RL reasoning, own Neuralese graph rows) was sampled by the actor."""
    import exp0_sft
    role = chain.get('role')
    if role == 'sd_teacher':
        return np.zeros(sum(n for _, n, _ in chain['segs']), bool)
    if role == 'replay':   # the corpus's own flags (per row or per target: filled-reasoning rows), else its source
        return exp0_sft.sampled_targets(chain)
    return np.concatenate([np.full(n, row.get('rrl') != 'action' and not row.get('teacher')) for _, n, row in chain['segs']])


def token_advantage(chain, device):
    """The chain's advantage per token: its ``advantage``, times a row's own ``advantage`` where rows carry one (the
    reasoning RL term: a think block at its group-normalized return, the teacher action after it at the action weight;
    ``--advantage-control`` still acts on the chain's factor). A plain number when no row carries one."""
    import torch
    if not any('advantage' in row for _, _, row in chain['segs']):
        return chain['advantage']
    return torch.from_numpy(np.concatenate([np.full(length, row.get('advantage', 1.), np.float32)
                                            for _, length, row in chain['segs']])).to(device) * chain['advantage']


def behaviour_logp(chain, logp):
    """(log-probabilities of the law that produced the chain's tokens, per-token row weights or None) for the policy
    term. An actor sample: the actor's recorded logp. A teacher demonstration (role ``sd_teacher``) has none: the
    learner's own (detached) logp stands in, so rho = 1 (a ratio-free cross-entropy at its advantage) and the tail floor
    reads the learner's probability; its rows' weights (an empty think block at 0) multiply the term. A chain mixing
    the two (the reasoning RL term and its carried chains: sampled think blocks, then teacher actions without logp)
    takes each row's own law, and its rows' weights."""
    import torch
    rows = [(length, row) for _, length, row in chain['segs']]
    if chain.get('role') not in RATIO_FREE_ROLES and all('logp' in row for _, row in rows):
        return torch.from_numpy(np.concatenate([row['logp'] for _, row in rows])).to(logp.device).float(), None
    import exp0_sft
    if chain.get('role') not in RATIO_FREE_ROLES:
        parts, start, learner = [], 0, logp.detach().float()
        for length, row in rows:
            parts.append(torch.from_numpy(np.asarray(row['logp'], np.float32)).to(logp.device) if 'logp' in row
                         else learner[start:start + length])
            start += length
        weights = exp0_sft.chain_weights(chain, logp.device) if any('weight' in row for _, row in rows) else None
        return torch.cat(parts), weights
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


def style_shift(work, style, ratio=0.75, from_step=4):
    """The ratio of this step's reasoning tokens per call to the run's first logged step at or after `from_step`, and
    whether it is outside [ratio, 1/ratio]. A logged flag, never a stop. The first steps train on the groups that finish
    first after launch (short tasks, little reasoning), so they are no baseline (grpo-v4-run1 steps 2-3 flagged 1.9x
    with the policy unchanged)."""
    first = None
    path = work / 'train.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            if '"reasoning_tokens_per_call"' in line:
                row = json.loads(line)
                if (row.get('step') or 0) >= from_step:
                    first = row.get('reasoning_tokens_per_call')
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
    step, and the KL to the start policy jumped after it (DECISIONS 2026-11-14). ``groups_with_spread`` may be fractional
    (``spread_data``: a group counts by the loss units it trains). The factor scales the gradient before the clip too
    (``scale_and_clip``, opt-out ``--no-data-scaled-gradient``): AdamW's moments carry a step's gradient into the next
    steps, so an LR factor alone does not shrink a small step's total push. ``reference`` 0: always 1."""
    if not reference:
        return 1.
    return min(1., groups_with_spread / reference)


def scale_and_clip(parameters, scale, clip):
    """(pre-clip norm of the step's mean-loss gradient, norm after ``scale`` and before the clip): multiplies every
    gradient by ``scale`` (the step's data factor, ``lr_data_scale``), then clips to ``clip``."""
    import torch
    parameters = [p for p in parameters if p.grad is not None]
    norm = torch.nn.utils.clip_grad_norm_(parameters, math.inf).item()
    if scale != 1. and math.isfinite(norm):
        for p in parameters:
            p.grad.mul_(scale)
    return norm, torch.nn.utils.clip_grad_norm_(parameters, clip).item()


def spread_data(group_units, reference_units):
    """The step's groups with reward spread, each counted by the loss units it trains (``group_units``: one total per
    such group, ``episode_unit_counts``) over ``reference_units``, at most 1; ``reference_units`` 0: each counts 1.
    lr_data_scale's measure of the data behind a step. Under ``--loss-norm token`` a step's loss is the mean over its
    own units, so a step of a few groups of sampled decisions (1 unit each, ~8-24 a group) puts a whole AdamW step on a
    handful of distinct choices that a mixed step weighs 1/10,000 each: the 2026-11-22 LoRA bench's decision-only steps
    had grad_norm 65 and 23 against 0.2-0.3, moved those decisions' option scores 2-3 nats in one step at LR 8e-6 and,
    through the moments, ~7.5 nats by step 6 (DECISIONS 2026-11-22 23:30)."""
    if not reference_units:
        return float(len(group_units))
    return sum(min(1., units / reference_units) for units in group_units)


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


def save_step_chains(directory, step, chains, scales, keep):
    """The step's exact learner batch (``--save-step-chains``): every chain with its advantage, role, row weights and
    actor log-probabilities, and its per-token loss factor (``token_scale``), pickled and zstd-compressed to
    ``directory/step-NNNNN.pkl.zst``; only the newest ``keep`` stay. Offline per-term diagnostics (the gradient of each
    term against a behaviour direction, DECISIONS 2026-11-19) then run on the batches the run actually trained, which
    the reasoning-RL samples and the SD buffer otherwise never leave memory for. Returns the file's size in MB."""
    import pickle
    import zstandard
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'step-{step:05d}.pkl.zst'
    record = {'step': step, 'chains': [{**chain, 'token_scale': scales[chain['episode']][0]} for chain in chains]}
    data = zstandard.ZstdCompressor(level=3, threads=-1).compress(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
    partial = path.with_suffix('.part')
    partial.write_bytes(data)
    os.replace(partial, path)
    registry_files(directory, [path])
    for old in sorted(directory.glob('step-*.pkl.zst'))[:-keep] if keep > 0 else ():
        old.unlink()
    return round(len(data) / 2**20, 1)


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


def reference_snapshot(learner, bundles=None, rows=None):
    """The KL reference as leaves of the policy's own structure: a clone of every factor (B = 0 at a fresh start: the
    start policy) and of the trainable rows, each requiring grad so the reference pass compiles the policy's training
    graph (they are never stepped). ``bundles``/``rows``: a resumed reference (``load_learner_state``); a reference
    saved as "bare base" (``()``) gets zero B factors in the live structure."""
    import torch
    from natlang_neuralese.train.lfm3_learner import LayerAdapterBundle, LinearFactorLeaf, TrainableRows

    def leaf(source, zero_b=False):
        b = torch.zeros_like(source.b) if zero_b else source.b
        return LinearFactorLeaf(a=source.a.detach().clone().requires_grad_(True),
                                b=b.detach().clone().requires_grad_(True), scale=float(source.scale))
    own = bundles if bundles else learner.bundles
    zero_b = not bundles and bundles is not None    # a resumed bare-base reference
    out = tuple(LayerAdapterBundle(layer_idx=bundle.layer_idx, layer_kind=bundle.layer_kind,
                                   **{name: leaf(value, zero_b) for name, value in bundle.iter_leaves()})
                for bundle in own)
    source = rows if rows is not None else learner.trainable_rows
    out_rows = None if source is None else TrainableRows(
        token_ids=source.token_ids.clone(), rows=source.rows.detach().clone().requires_grad_(True),
        marker_ids=source.marker_ids, override_ids=source.override_ids)
    return out, out_rows


def reference_tensors(bundles, rows):
    tensors = [t for bundle in bundles or () for _, value in bundle.iter_leaves() for t in (value.a, value.b)]
    return tensors + ([rows.rows] if rows is not None else [])


def refuse_legacy_resume(resume_dir, state_root, source_sha):
    """A resume must restore this run's own LoRA learner state on the same base: refuse a state saved against another
    ``--checkpoint`` (its factors would attach to the wrong base) and a run that holds only the full-FT
    ``training-state.safetensors`` (it would silently start again at step 0)."""
    manifest_path = Path(resume_dir) / 'learner_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('base_weights_sha256') != source_sha:
            raise SystemExit(f'{resume_dir} was trained on base {manifest.get("base_weights_sha256") or "(unrecorded)"}, '
                             f'not on --checkpoint {source_sha}: resume with the checkpoint it was trained on.')
        return
    legacy = [path for path in (Path(state_root) / 'training-state.safetensors', Path(resume_dir))
              if path.is_file() and path.suffix == '.safetensors']
    if legacy:
        raise SystemExit(f'{legacy[0]} is a full-FT training state, which the LoRA learner cannot resume: start a new '
                         'run from a merged checkpoint (--checkpoint) without --resume.')
    if Path(resume_dir).exists() and any(Path(resume_dir).iterdir()):
        raise SystemExit(f'{resume_dir} holds no learner_manifest.json: pass --resume-state with a learner-state '
                         'directory, or drop --resume to start fresh.')


def actor_command(args, work, source_sha):
    """(command, url) of the run's SGLang actor: ``serve_native_checkpoint.py`` with the pinned held engine's serve args
    (``engine.json`` beside the held manifest), its store under WORK/actor. Used by ``train`` and by
    ``expert_iteration.py collect``."""
    engine = json.loads((Path(args.held_manifest).parent / 'engine.json').read_text())['serve_args']
    store = Path(work) / 'actor'
    store.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(HERE / 'serve_native_checkpoint.py'), '--store-dir', str(store),
               '--checkpoint', args.checkpoint, '--checkpoint-sha256', source_sha, '--base', args.base,
               '--port', str(args.actor_port), '--prompt-rendering', 'full', *engine,
               '--hf-export-dir', str(store / 'sglang-hf'),
               *(['--sglang-memory-saver'] if args.actor_release else []),
               *(['--sglang-overlap-schedule'] if actor_overlap(args) else []), *args.actor_arg]
    return command, f'http://127.0.0.1:{args.actor_port}'


def train(args):
    import torch
    import behaviour_probe
    import exp0_sft
    from natlang_neuralese.train import lfm3_fast as lfm3_fast_train
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
        handle.write(json.dumps(run, default=str) + '\n')
    log_line(log, {'event': 'start', 'code': run['code'], 'eligible_tasks': len(tasks), 'excluded_ids': excluded})

    t0 = time.time()
    actor_ready = {}
    actor_process = []

    def launch_actor():
        # The training actor is a child of this unit (one ledger claim covers learner and actor, so neither waits on the
        # other's admission), served with the pinned held-v2 eval engine (held-v2/engine.json serve_args): training and
        # held measurements sample with the same arithmetic.
        try:
            command, url = actor_command(args, work, source_sha)
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
    # Shared LoRA+rows learner core (lfm3opti.md §5): frozen BF16 base + FP32 LoRA factors + trainable sparse rows. The
    # base is the start checkpoint: its digest binds every policy id (effective_policy_sha256) and the learner state.
    from natlang_neuralese.train.lfm3_learner import (
        AdapterSpec, Lfm3Learner, OOMRetryPlan, run_step_with_oom_retry,
    )
    from natlang_neuralese.train.learner_state import load_learner_state, save_learner_state
    spec = AdapterSpec.for_rl(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        trainable_rows=not args.no_trainable_rows,
        gate_rank=args.gate_rank,
    )
    state_dir = Path(args.state_dir) / 'learner-state'
    branch_root = Path(args.checkpoint_dir or Path(args.state_dir) / 'checkpoints')
    resume_dir = Path(args.resume_state) if args.resume_state else state_dir
    if args.resume and not args.resume_state and not (resume_dir / 'learner_manifest.json').exists() and args.branch_states:
        candidates = sorted(p for p in branch_root.glob('state-step-*') if (p / 'learner_manifest.json').exists())
        if candidates:
            resume_dir = candidates[-1]
    if args.resume:
        refuse_legacy_resume(resume_dir, Path(args.state_dir), source_sha)
    learner = Lfm3Learner.from_checkpoint(
        args.checkpoint, args.base, spec, device='cuda', base_weights_sha256=source_sha,
        fp8_base=True if args.fp8_base else None, compile_layers=not args.no_compile,
    )
    model = learner.model
    special = learner.special_ids
    native_rows = learner.native_rows
    # only what a pruned checkpoint needs stays on the host (the source backbone is 3.4 GB)
    state = {k: learner.checkpoint_state[k] for k in ('schema', 'model_config', 'interface_state', 'step')}
    learner.state = state
    import gc
    gc.collect()
    drop_cache([args.checkpoint, *Path(args.base).glob('*.safetensors')])
    log_line(log, {'event': 'learner model loaded', 'meminfo_gib': meminfo(), 'load_s': round(time.time() - t0, 1),
                   'adapter_spec': spec.to_dict(), 'fp8_base': learner.fp8_base,
                   'trainable_params': sum(p.numel() for p in learner.trainable_parameters())})
    forward = lfm3_fast_train.hidden
    pack_tokens = args.pack_tokens if args.pack_tokens > 0 else lfm3_fast_train.resident_tokens(
        model, args.activation_gb, parameterization='lora')
    oom_plan = OOMRetryPlan(pack_tokens=pack_tokens, activation_gib=args.activation_gb)
    compile_health = lfm3_fast_train.CompileHealthTracker()

    # The KL reference is the run's start policy: B = 0 (the bare base) with the start rows at a fresh start
    # (async-train §7.1), a frozen snapshot of it carried in the learner state across resumes.
    reference_active = args.kl_beta > 0
    ref_bundles, ref_rows = reference_snapshot(learner) if reference_active else (None, None)
    parameters = learner.trainable_parameters()
    optimizer = learner.build_optimizer(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    step, cursor, first_group = 0, 0, 0
    consumed_groups = set()
    if args.resume and (resume_dir / 'learner_manifest.json').exists():
        resumed = load_learner_state(resume_dir, learner, optimizer=optimizer, restore_rng=True,
                                     verify_base_sha=source_sha)
        manifest = resumed['manifest']
        step = int(manifest.get('learner_step', 0))
        ledger = manifest.get('rl_rollout_ledger') or {}
        cursor = int(ledger.get('cursor', 0))
        first_group = int(ledger.get('next_group', 0))
        consumed_groups = set(int(g) for g in (ledger.get('consumed_groups') or ()))
        if reference_active:
            if not manifest.get('has_ref_factors'):
                raise SystemExit(f'{resume_dir} holds no KL reference (saved without --kl-beta): resume with --kl-beta 0 '
                                 'or start a new run.')
            ref_bundles, ref_rows = reference_snapshot(learner, resumed['ref_bundles'], resumed['ref_rows'])
        if (resumed.get('trainer_state') or {}).get('oom_plan'):
            oom_plan = OOMRetryPlan.from_dict(resumed['trainer_state']['oom_plan'])
        release_host_memory()
        log_line(log, {'event': 'resumed', 'step': step, 'cursor': cursor, 'next_group': first_group,
                       'consumed_groups': len(consumed_groups), 'state': str(resume_dir),
                       'base_weights_sha256': manifest.get('base_weights_sha256')})
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
    weights_dir = Path('/dev/shm') / f'natlang-{args.unit_prefix}-actor-weights'   # owned by WeightSync
    if not args.frozen_producer_table and producer_adapter_keys(state['interface_state']):
        raise SystemExit(f'the checkpoint has a producer adapter {producer_adapter_keys(state["interface_state"])}: '
                         'the producer table is not the embedding (use --frozen-producer-table)')
    release_tags = [t for t in args.actor_release_tags.split(',') if t] if (
        args.actor_release and actor_thread is not None and not args.no_actor_sync) else []
    sentinel_ids = None
    if not args.no_publication_sentinel:
        from tokenizers import Tokenizer
        sentinel_ids = Tokenizer.from_file(str(Path(args.base) / 'tokenizer.json')).encode(
            SENTINEL_TEXT, add_special_tokens=False).ids
    config = state['model_config']
    extra_actors = parse_extra_actors(args.extra_actor, args.slots)
    sync = WeightSync(learner, args.actor_export, None if args.no_actor_sync else sglang_url, weights_dir,
                      front_url=None if args.no_actor_sync else args.actor,
                      producer_from_embedding=not args.frozen_producer_table, base_sha256=source_sha,
                      start_policy={'policy_id': source_sha, 'version': 0},
                      interface_ids=(config['marker_ids'], config.get('token_override_ids') or ()),
                      tied=native_rows['tied_head'], sentinel_ids=sentinel_ids,
                      extra_actors=() if args.no_actor_sync else extra_actors,
                      on_actor=lambda front, version: collector.actor_versions.__setitem__(front, version))
    drop_cache([Path(args.actor_export) / 'model.safetensors'])
    last_state_save_info = None
    if args.replay:
        collector = ReplayCollector(args, work, first_group=first_group)
    else:
        collector = Collector(args, tasks, work, source_sha, runtime, first_group=first_group, cursor=cursor)
        collector.variants = load_variants(args, tasks, work)
    teacher_demos = None
    sd_teacher_on = args.sd_teacher_weight > 0 and args.sd_teacher_episodes > 0
    reasoning_rl_on = args.reasoning_rl_weight > 0 and args.reasoning_rl_groups > 0 and not args.replay
    if not args.replay and (sd_teacher_on or (reasoning_rl_on and 'demos' in args.reasoning_rl_sources.split(','))):
        teacher_demos = TeacherDemos(args.sd_teacher_source or default_sd_teacher_sources(), collector.by_id, args.held_manifest,
                                     renderer_tokenizer(args.actor_export, work / 'sd-teacher-tokenizer'),
                                     args.sd_teacher_max_per_task, args.seed, log=lambda r: log_line(log, r),
                                     uses_path=work / 'teacher-demo-uses.json', take_seed=f'{args.seed}:{first_group}',
                                     drifted=args.sd_teacher_admit_drifted == 'on')
    replay_corpora = []
    if args.replay_corpus and args.no_neuralese_passthrough:
        raise SystemExit('--replay-corpus consumers import their producer documents: drop --no-neuralese-passthrough')
    if args.replay_corpus:
        from natlang_neuralese.rl.held_eval import exclusion_ids
        held = set(exclusion_ids(args.held_manifest))
        from transformers import AutoTokenizer
        corpus_tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
        replay_corpora = [ReplayCorpus(spec, held, args.seed, log=lambda r: log_line(log, r), tokenizer=corpus_tokenizer)
                          for spec in args.replay_corpus]
    collector.version = step
    if extra_actors and not args.replay:
        collector.actor_versions.update(extra_actor_start_versions(extra_actors, source_sha, step))
        log_line(log, {'event': 'extra actors', 'actors': extra_actors, 'versions': collector.actor_versions})
    if args.resume and not args.replay:
        reused, left_out = collector.reuse_scored(step, args.max_staleness, args.memetic_max_staleness)
        if consumed_groups:
            # Never double-train a group already recorded in rl_rollout_ledger.
            for g in consumed_groups:
                d = work / 'groups' / f'g{g:06d}'
                if d.is_dir():
                    (d / RETIRED).touch()
        if reused or left_out:
            log_line(log, {'event': 'reused groups', 'reused': len(reused), 'left_out': left_out[:50],
                           'left_out_count': len(left_out)})
    from natlang_neuralese.common.sampling import rl_sampling_controls
    (work / 'sampling-config.json').write_text(json.dumps(rl_sampling_controls(args.top_p, capture=True)) + '\n')
    versions_live = {'actor': step}
    probes = probe_chains(args) if args.probe_turns else []
    think_close = None
    second_close_stops = set()
    try:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_file(str(Path(args.base) / 'tokenizer.json'))
        think_close = _tokenizer.token_to_id('</think>')
        second_close_stops = {i for i in (_tokenizer.token_to_id(t) for t in exp0_sft.SECOND_CLOSE_STOPS) if i is not None}
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
            logp = lfm3_fast_train.token_logprobs(
                model, ids, torch.from_numpy(positions).cuda(),
                torch.from_numpy(targets.astype(np.int64)).cuda(),
                adapters=learner.bundles, trainable_rows=learner.trainable_rows, fp8_base=learner.fp8_base,
            )
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
    strict_series = ['--strict-series'] if args.held_strict_series else []
    step0 = work / 'held' / 'step-00000'
    if (step0 / 'result.json').exists() and not args.held_baseline:
        measured = json.loads((step0 / 'result.json').read_text())
        old = (measured.get('series') or {}).get('key')
        law = measured.get('sampling') or LEGACY_SAMPLING
        if (args.held_strict_series and old != run['series']['key']) or law != sampling_law_of(args.top_p):
            aside = work / 'held' / (f'step-00000-series-{old}' if old != run['series']['key'] else
                                     f'step-00000-top-p-{law["top_p"]}')
            step0.rename(aside)
            log_line(log, {'event': 'held series changed', 'old_series': old, 'series': run['series']['key'],
                           'old_sampling': law, 'sampling': sampling_law_of(args.top_p), 'moved_to': str(aside)})
    if step:
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
                           **behaviour_probe.summary(json.loads(behaviour_start_path.read_text()), probe=behaviour),
                           'seconds': round(time.time() - t, 1)})
        else:
            log_line(log, {'event': 'behaviour probe: no start measurement (resumed); copy the start run\'s '
                                    'behaviour-start.json to compare'})
    if (behaviour is not None and behaviour.get('end_calls') and behaviour_start_path.exists() and reference_active
            and not json.loads(behaviour_start_path.read_text()).get('end_logp')):
        t = time.time()
        try:
            with learner.reference_view(ref_bundles, ref_rows):
                end = behaviour_probe.measure_end(model, behaviour)
        except Exception as error:
            end = None
            log_line(log, {'event': 'behaviour probe start: end of turn not measured', 'error': repr(error)[:300]})
        if end is not None:
            start_record = json.loads(behaviour_start_path.read_text())
            start_record.update(end)
            behaviour_start_path.write_text(json.dumps(start_record) + '\n')
            log_line(log, {'event': 'behaviour probe start: end of turn from the reference weights',
                           'calls': len(end['end_logp']), 'end_logp': round(float(np.mean(end['end_logp'])), 4),
                           'seconds': round(time.time() - t, 1)})
    log_line(log, {'event': 'learner ready', 'init_s': round(time.time() - t0, 1), 'meminfo_gib': meminfo(),
                   'cuda_allocated_gib': round(torch.cuda.memory_allocated() / 2**30, 1),
                   'cuda_reserved_gib': round(torch.cuda.memory_reserved() / 2**30, 1)})
    if args.held_every_hours > 0 and not (work / 'held' / 'step-00000' / 'result.json').exists() \
            and not args.skip_base_held and not args.held_baseline and not args.held_remote:
        collector.paused.set()            # the start policy's held measurement goes first (held() resumes collection)
    slots = [threading.Thread(target=collector.slot, args=(index,), daemon=True)
             for index in range(0 if args.replay else args.slots)]
    for thread in slots:
        thread.start()

    checkpoint_root = Path(args.checkpoint_dir or Path(args.state_dir) / 'checkpoints')   # off the busy NVMe by default
    from natlang_neuralese.train.checkpoint_offload import discard
    # every trained episode is kept (compressed) beside the checkpoints: the w5 lesson (AGENTS.md: pruned bundles lost 85
    # successful episodes); '-' keeps none
    rollouts_archive = None if args.rollouts_dir == '-' else Path(args.rollouts_dir or checkpoint_root.parent / 'rollouts')
    registry_run_records(args, work, rollouts_archive, collector, teacher_demos, replay_corpora, len(tasks))
    checkpoints = sorted(checkpoint_root.glob('step-*'))
    held_steps = set(json.loads((work / 'held-steps.json').read_text())) if (work / 'held-steps.json').exists() else set()
    stop = threading.Event()
    # A unit stop signals every process of the cgroup, so the child actor goes at once: slots stop starting episodes
    # (each would fail in seconds on 'fetch failed' and burn its whole group as infrastructure errors, 97 episodes in
    # the two run2 restarts of 2026-11-13); the learner finishes the step it is in.
    signal.signal(signal.SIGTERM, lambda *_: (stop.set(), collector.stop.set()))
    reasoning_rl = None
    # finished reasoning-RL searches persist across a stop (ReasoningRL.save); NATLANG_RRL_READY_PERSIST=0 opts out
    rrl_ready_path = work / 'reasoning-rl-ready.pkl' if os.environ.get('NATLANG_RRL_READY_PERSIST', '1') != '0' else None
    if reasoning_rl_on:
        if think_close is None:
            raise SystemExit('--reasoning-rl-weight needs the </think> id (see the log); --reasoning-rl-weight 0 opts out')
        from tokenizers import Tokenizer
        served = Tokenizer.from_file(str(Path(args.actor_export) / 'tokenizer.json'))
        stop_ids = [i for i in (served.token_to_id('<|im_end|>'), served.token_to_id('<|endoftext|>')) if i is not None]
        export_manifest = Path(args.actor_export) / 'natlang-export.json'
        banned = (json.loads(export_manifest.read_text()).get('serving') or {}).get('banned_token_ids', []) \
            if export_manifest.exists() else []
        from natlang_neuralese.model.shared_state import ROLE_MARKERS
        markers = [i for i in (served.token_to_id(m) for m in ROLE_MARKERS) if i is not None]   # Neuralese role markers
        twin = twin_turn_ids(served, think_close) if args.reasoning_rl_objective == 'twin-fkl' else None
        reasoning_rl = ReasoningRL(args, sglang_url, think_close, stop_ids, banned, markers, collector, teacher_demos,
                                   log=lambda r: log_line(log, r), first_group=first_group, twin=twin)
        if args.resume and rrl_ready_path is not None:
            restored = reasoning_rl.restore(rrl_ready_path, step)
            if restored:
                log_line(log, {'event': 'reasoning rl ready restored', 'items': restored})
        reasoning_rl.start(stop)
        log_line(log, {'event': 'reasoning rl', 'sources': reasoning_rl.sources, 'demo_tasks': len(reasoning_rl.demo_tasks),
                       'stop_ids': reasoning_rl.stops, 'banned': banned, 'markers': markers,
                       'rewrite_share_effective': rrl_rewrite_share(args), 'carry_sd_effective': rrl_carry_sd(args),
                       **({'twin_header_tokens': len(twin['header'])} if twin else {}),
                       **{k: v for k, v in vars(args).items() if k.startswith('reasoning_rl_')}})

    def write_native(step_number):
        """The step's pruned native checkpoint, from the bytes the actor is about to serve (the staging slot) and the
        snapshot's trained rows: what Pop's held evaluations and the probes load is exactly the published policy."""
        directory = checkpoint_root / f'step-{step_number:05d}'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'checkpoint.pt'
        policy_id = sync.publishing['policy_id']
        provenance = {'parameterization': 'lora+rows', 'adapter_spec': spec.to_dict(), 'effective_policy_sha256': policy_id,
                      'base_weights_sha256': source_sha}
        write_pruned_checkpoint({**state, 'interface_state': sync.interface(state['interface_state'])}, native_rows,
                                special, path, source_sha, sync.backbone(), learner_step=step_number, extra=provenance)
        from natlang_neuralese.common.hashing import sha256_file_hex
        sha = sha256_file_hex(path)
        (directory / 'sha256.txt').write_text(sha + '\n')
        (directory / 'meta.json').write_text(json.dumps({
            'step': step_number, 'learner_step': step_number, 'source_step': int(state['step']),
            'sha256': sha, 'effective_policy_sha256': policy_id, 'actor_weights_sha256': sync.publishing['weights_sha256'],
            'tensor_identity': sync.publishing['tensor_identity'], 'source_sha256': source_sha, 'code': run['code'],
        }) + '\n')
        checkpoints.append(directory)
        for old in checkpoints[:-2]:
            number = int(old.name.split('-')[1])
            if number not in held_steps and number > args.keep_checkpoints_until and old.exists():
                discard(old)
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
                # every complete earlier result of the run orders the episodes (held_eval.Dispatch)
                prior = [item for path in held_priors(work / 'held' / 'step-00000', work / 'held', out)
                         for item in ('--duration-prior', path)]
                code = subprocess.run([*python, 'measure', '--manifest', args.held_manifest, *prior, '--endpoint',
                                       f'{args.actor}={args.held_workers}', '--policy-id', policy, '--top-p', str(args.top_p), '--samples', str(samples),
                                       '--out', str(out), '--label', f'grpo-{name or f"step-{step_number}"}',
                                       '--max-model-requests', str(args.max_model_requests)],
                                      cwd=REPO, env=pinned_env(), stdout=handle, stderr=subprocess.STDOUT).returncode
                base = Path(args.held_baseline) if args.held_baseline else work / 'held' / 'step-00000' / 'result.json'
                # Only complete measurements pair: a stop mid-measurement also exits 0 with a partial result.json
                # (grpo-v4-run2 held/step-00038: 13 of 512 episodes, with a compare a judge would read as the step's).
                (out / 'compare-vs-step0.json').unlink(missing_ok=True)
                if (step_number or name) and code == 0 and held_complete(base) and held_complete(out):
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
                sync.publish_start(0, lambda *_: None, source_sha)
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
                sync.stage_native(path)
                sync.publish(0, on_published, policy=sha)
                sync.wait()
                if not sync.last.get('ok'):
                    raise RuntimeError(f'loading {path} into the actor failed: {sync.last}')
                (work / 'held').mkdir(exist_ok=True)
                (work / 'held' / f'{name}.source').write_text(path + '\n')
                held(0, args.held_samples_extra, name=name, policy=sha, resume_collection=False)
        finally:
            sync.publish_start(0, on_published, source_sha)   # the learner's own (start) weights back on the actor
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
        collector processes run on the DGX, against Pop's endpoint) into ``held/step-N`` with ``compare-vs-step0.json``.

        Each evaluation runs as its own DGX ledger unit (``pop_rl_actor.py held-eval``, ``--held-remote-gb``), so a
        restart of this run neither kills its episodes nor its copy: the restarted loop adopts a unit still running
        (``held/step-N/pop-eval.json``) and starts no other evaluation on Pop meanwhile (grpo-v4-run2: 14 restarts in 40
        steps cut 8 of 10 evaluations and 3 copies). ``--held-remote-inline`` runs them inside this process as before.
        While an evaluation runs, the newest checkpoint not yet measured is copied to Pop (``pop_rl_actor.py prefetch``;
        ``--no-pop-prefetch`` opts out), so the next one starts without its 265-520 s copy. Every complete earlier result
        of the run orders the next evaluation's episodes (``--duration-prior``, held_eval.Dispatch)."""
        remote_profile = args.held_remote or 'pop'
        from pop_rl_actor import pop_sglang, serving_endpoint
        held_root, tool = work / 'held', [sys.executable, str(HERE / 'pop_rl_actor.py')]
        base_out = Path(args.held_base) if args.held_base else held_root / f'step-00000-{remote_profile}'
        python = [sys.executable, '-m', 'natlang_neuralese.rl.held_eval']
        # Remote engine settings: the KV dtype is part of the attested engine (export format), so the start policy's
        # result is kept only if it was measured with the same one and the same attested max running sequences.
        pop_engine = ['--host-profile', remote_profile, '--kv-cache-dtype', args.pop_kv_cache_dtype,
                      '--max-running', str(args.pop_max_running)]
        pop_format = pop_sglang(args.pop_kv_cache_dtype, args.pop_max_running, profile=remote_profile)[1]

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

        if args.held_remote_inline and remote_profile != 'pop':
            raise SystemExit('--held-remote-inline serves one Pop actor; Colab G4 evaluations run detached '
                             '(pop_rl_actor.py held-eval shards them over the sessions)')

        def pop_endpoint(policy):
            return serving_endpoint(policy, pop_format, profile=remote_profile)

        def priors(out):
            return held_priors(base_out, held_root, out)

        def inflight():
            return pop_eval_inflight([base_out, *sorted(held_root.glob('step-*'))])

        def other_pop_evaluation():
            """An active held evaluation unit on this machine that is not this run's (Pop is one GPU: its serve would stop
            that evaluation's actor)."""
            out = subprocess.run(['systemctl', '--user', 'list-units', '*popheld*', '--state=active,activating', '--plain',
                                  '--no-legend'], capture_output=True, text=True).stdout
            mine = f'{args.unit_prefix}-popheld-'
            return [line.split()[0] for line in out.splitlines() if line.strip() and not line.startswith(mine)]

        def wait_detached(out, record):
            """Wait for a detached evaluation unit; its outcome (``pop-eval-done.json``), or a failure if it ended
            without one (two checks 60 s apart: a ledger waiter hands over to the unit in between)."""
            done, gone = out / 'pop-eval-done.json', 0
            while not stop.is_set():
                if done.exists():
                    return json.loads(done.read_text())
                gone = 0 if unit_active(record['unit']) else gone + 1
                if gone >= 3:
                    return {'returncode': 1, 'serve_seconds': None, 'served': (out / 'run.json').exists(),
                            'error': f'unit {record["unit"]} ended without an outcome (journalctl --user -u {record["unit"]})'}
                stop.wait(30)
            return None

        def evaluate_detached(step_number, out, policy, serve):
            out.mkdir(parents=True, exist_ok=True)
            (out / 'pop-eval-done.json').unlink(missing_ok=True)
            unit = f'{args.unit_prefix}-popheld-{step_number:05d}-{int(time.time())}'
            measure = ['--manifest', str(args.held_manifest), '--policy-id', policy, '--top-p', str(args.top_p),
                       '--samples', str(args.held_samples), '--out', str(out), '--label', f'grpo-pop-step-{step_number}',
                       '--allow-other-engine', '--max-model-requests', str(args.max_model_requests)]
            for path in priors(out):
                measure += ['--duration-prior', path]
            compare = ([str(base_out / 'result.json'), str(out / 'result.json'), '--out',
                        str(out / 'compare-vs-step0.json'), *strict_series]
                       if step_number and held_complete(base_out) else None)
            spec = {'unit': unit, 'policy': policy, 'format': pop_format, 'host_profile': remote_profile,
                    'serve': [str(x) for x in serve], 'measure': measure,
                    'slots': args.held_remote_slots, 'compare': compare, 'log': str(work / 'held-pop.log'),
                    'done': str(out / 'pop-eval-done.json')}
            (out / 'pop-eval-spec.json').write_text(json.dumps(spec, indent=1) + '\n')
            subprocess.run(['/usr/bin/python3', str(REPO / 'scripts/memory_ledger.py'), 'run', '--unit', unit,
                            '--budget-gb', str(args.held_remote_gb), '--priority', 'medium', '--no-resumable',
                            '--wait', '86400', '--detach', '--workdir', str(REPO), '--',
                            '/usr/bin/env', f'NATLANG_REPO={REPO}', f'PYTHONPATH={REPO}/training/neuralese:{REPO}',
                            'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'HF_HUB_OFFLINE=1',
                            sys.executable, str(HERE / 'pop_rl_actor.py'), 'held-eval', '--spec', str(out / 'pop-eval-spec.json')],
                           check=True, timeout=600)
            record = {'unit': unit, 'policy': policy, 'step': step_number, 'started': time.time()}
            (out / 'pop-eval.json').write_text(json.dumps(record) + '\n')
            return record

        def report(step_number, out, policy, outcome, started):
            summary = {}
            if (out / 'result.json').exists():
                result = json.loads((out / 'result.json').read_text())
                summary = {k: result.get(k) for k in ('ran', 'scored', 'successes', 'loop_rate', 'ran_out_rate', 'complete')}
            if (out / 'compare-vs-step0.json').exists():
                summary['compare'] = json.loads((out / 'compare-vs-step0.json').read_text())
            log_line(log, {'event': 'held eval', 'remote': remote_profile, 'step': step_number, 'policy': policy,
                           'returncode': outcome.get('returncode'), 'seconds': round(time.time() - started),
                           'serve_seconds': outcome.get('serve_seconds'), 'detached': True, **summary})
            if not outcome.get('served', True):
                raise RuntimeError(f'Pop does not serve {policy[:12]} (see {work}/held-pop.log)'
                                   + (f': {outcome["error"]}' if outcome.get('error') else ''))

        def evaluate(step_number, out, policy, serve):
            if not args.held_remote_inline:
                record = evaluate_detached(step_number, out, policy, serve)
                outcome = wait_detached(out, record)
                if outcome is not None:
                    report(step_number, out, policy, outcome, record['started'])
                return
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
                # every complete earlier result's per-task wall times order this measurement (held_eval.Dispatch)
                prior = [item for path in priors(out) for item in ('--duration-prior', path)]
                measuring.set()
                code = subprocess.run([*python, 'measure', '--manifest', args.held_manifest, *prior, '--endpoint',
                                       f'{endpoint}={args.held_remote_slots}', '--policy-id', policy, '--top-p', str(args.top_p), '--samples',
                                       str(args.held_samples), '--out', str(out), '--label', f'grpo-pop-step-{step_number}',
                                       '--allow-other-engine', '--max-model-requests', str(args.max_model_requests)],
                                      cwd=REPO, env=pinned_env(), stdout=handle,
                                      stderr=subprocess.STDOUT).returncode
                measuring.clear()
                (out / 'compare-vs-step0.json').unlink(missing_ok=True)   # complete measurements only (as held())
                if step_number and code == 0 and held_complete(base_out) and held_complete(out):
                    subprocess.run([*python, 'compare', str(base_out / 'result.json'), str(out / 'result.json'), '--out',
                                    str(out / 'compare-vs-step0.json'), *strict_series], cwd=REPO, env=pinned_env(), stdout=handle,
                                   stderr=subprocess.STDOUT)
            summary = {}
            if (out / 'result.json').exists():
                result = json.loads((out / 'result.json').read_text())
                summary = {k: result.get(k) for k in ('ran', 'scored', 'successes', 'loop_rate', 'ran_out_rate', 'complete')}
            if (out / 'compare-vs-step0.json').exists():
                summary['compare'] = json.loads((out / 'compare-vs-step0.json').read_text())
            log_line(log, {'event': 'held eval', 'remote': remote_profile, 'step': step_number, 'policy': policy, 'returncode': code,
                           'seconds': round(time.time() - started), 'serve_seconds': round(serving), **summary})

        def measured_steps():
            return {int(d.name.split('-')[1]) for d in held_root.glob('step-*') if d.name.split('-')[1].isdigit()
                    and held_complete(d)}

        last_start, last_step = 0., max(measured_steps() - {0}, default=0)
        partial_tried = set()
        prefetched = set()

        def prefetch_loop():
            """While an evaluation runs, copy the newest checkpoint not yet measured to Pop (the next evaluation's)."""
            while not stop.wait(300):
                busy = inflight() is not None or measuring.is_set()
                ready = sorted(d for d in checkpoint_root.glob('step-*') if (d / 'sha256.txt').exists()
                               and (d / 'checkpoint.pt').exists() and int(d.name.split('-')[1]) > last_step)
                if not busy or not ready:
                    continue
                sha = (ready[-1] / 'sha256.txt').read_text().strip()
                if sha in prefetched:
                    continue
                with (work / 'held-pop.log').open('a') as handle:
                    code = subprocess.run([*tool, 'prefetch', '--checkpoint', str(ready[-1] / 'checkpoint.pt'), '--policy-id',
                                           sha, '--step', ready[-1].name.split('-')[1]], cwd=REPO, env=pinned_env(),
                                          stdout=handle, stderr=subprocess.STDOUT, timeout=4 * 3600).returncode
                if code == 0:
                    prefetched.add(sha)
        measuring = threading.Event()   # an inline evaluation is running (detached ones: inflight())
        if args.pop_prefetch and remote_profile == 'pop':   # copies to Pop's disk
            threading.Thread(target=prefetch_loop, daemon=True).start()
        while not stop.is_set():
            try:
                running = inflight()
                if running:   # a detached evaluation started before this run (re)started: adopt it, start none beside
                    out, record = running
                    if record.get('step'):
                        held_steps.add(record['step'])
                        last_start, last_step = record['started'], max(last_step, record['step'])
                    log_line(log, {'event': 'held eval adopted', 'remote': remote_profile, 'step': record.get('step'),
                                   'unit': record['unit']})
                    outcome = wait_detached(out, record)
                    if outcome is not None:
                        report(record.get('step') or 0, out, record['policy'], outcome, record['started'])
                    continue
                if not args.held_remote_inline and remote_profile == 'pop' and other_pop_evaluation():
                    stop.wait(300)   # another run's evaluation holds Pop
                    continue
                if args.held_base and not held_complete(base_out):
                    # measured by its own job (e.g. the pin's base on Pop while this run starts): wait, never serve
                    # another policy on Pop meanwhile; checkpoints keep being kept for their evaluations
                    stop.wait(300)
                    continue
                if args.held_base and not base_engine_matches():
                    raise RuntimeError(f'--held-base {base_out} was measured on another engine or sampling law than {pop_format}')
                if not base_engine_matches():
                    aside = held_root / f'step-00000-{remote_profile}-{time.strftime("%Y%m%d%H%M")}-other-engine'
                    base_out.rename(aside)
                    log_line(log, {'event': 'held eval', 'remote': remote_profile, 'moved_start_policy_result': str(aside),
                                   'reason': f'measured on another Pop engine than {pop_format}'})
                if not held_complete(base_out):
                    evaluate(0, base_out, source_sha, ['start', '--checkpoint', args.checkpoint, '--policy-id', source_sha,
                                                       *pop_engine])
                    if not held_complete(base_out):
                        raise RuntimeError(f'the start policy measurement ended without a result (see {work}/held-pop.log)')
                ready = sorted(d for d in checkpoint_root.glob('step-*') if (d / 'sha256.txt').exists()
                               and int(d.name.split('-')[1]) > last_step)
                # A measurement a restart interrupted (held/step-N with run.json, no result) resumes first and in place
                # (held_eval measure resumes by manifest, policy and sampling), so a resume never waits for an
                # evaluation or discards its episodes. NATLANG_HELD_RESUME_PARTIAL=0 opts out.
                partial = [d for d in ready if os.environ.get('NATLANG_HELD_RESUME_PARTIAL', '1') != '0'
                           and d.name not in partial_tried and (held_root / d.name / 'run.json').exists()
                           and not held_complete(held_root / d.name)]
                if partial or (ready and time.time() - last_start >= args.held_every_hours * 3600):
                    directory = (partial or ready)[-1]
                    partial_tried.add(directory.name)   # one attempt: a failing resume never blocks newer evaluations
                    step_number = int(directory.name.split('-')[1])
                    held_steps.add(step_number)          # kept by write_native's pruning from now on
                    (work / 'held-steps.json').write_text(json.dumps(sorted(held_steps)) + '\n')
                    if (directory / 'checkpoint.pt').exists():
                        sha = (directory / 'sha256.txt').read_text().strip()
                        previous_start, last_start, last_step = last_start, time.time(), step_number
                        try:
                            evaluate(step_number, held_root / f'step-{step_number:05d}', sha,
                                     ['serve-checkpoint', '--checkpoint', str(directory / 'checkpoint.pt'), '--policy-id',
                                      sha, '--commit', run['code']['git_commit'], '--step', str(step_number), *pop_engine])
                        except Exception:
                            # Pop never served it (rsync or serve failed): the newest checkpoint is tried at the next
                            # retry instead of an --held-every-hours interval later (grpo-v4-run2 lost up to an hour
                            # of Pop time per 'Pop does not serve' failure).
                            if not (held_root / f'step-{step_number:05d}' / 'run.json').exists():
                                last_start = previous_start
                                held_steps.discard(step_number)   # never measured: write_native's pruning may drop it
                                (work / 'held-steps.json').write_text(json.dumps(sorted(held_steps)) + '\n')
                            raise
            except Exception as error:   # Pop or the network failed: the run trains on, the evaluation retries later
                log_line(log, {'event': 'held eval failed', 'remote': remote_profile, 'error': repr(error)[:500]})
                stop.wait(540)
            stop.wait(60)
    if args.held_remote:
        threading.Thread(target=pop_held_loop, daemon=True).start()

    # actor tokens/s while the learner computes (learner-phase slots) and while it waits (all slots), cumulative
    phase_totals = collections.Counter()
    phase_mark = {'time': time.time(), 'tokens': actor_generated(args.actor)}
    # self-distillation replay of recent rare successes (``sd_add``/``sd_take``; --sd-weight). It persists in
    # WORK/sd-buffer.pkl so a resume keeps the window's successes (grpo-v4-run2 resumed every 1-6 steps against a 4-step
    # window and lost most of them); NATLANG_SD_BUFFER_PERSIST=0 opts out.
    sd_buffer_path = work / 'sd-buffer.pkl' if os.environ.get('NATLANG_SD_BUFFER_PERSIST', '1') != '0' else None
    sd_buffer = sd_buffer_load(sd_buffer_path, step) if args.resume and args.sd_weight > 0 else []
    if sd_buffer:
        log_line(log, {'event': 'sd buffer restored', 'episodes': len(sd_buffer)})
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
        spread_keys = []     # per group with spread: its trained episodes' keys (spread_data)
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
            if reasoning_rl is not None:   # JOLT+ teacher successes: their actions are reasoning RL targets
                reasoning_rl.offer(episodes)
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
            spread_keys.append([])
            for index, (episode, advantage) in enumerate(zip(episodes, advantages)):
                if advantage is None:
                    continue
                if args.call_fraction < 1:
                    episode = subsample_calls(episode, args.call_fraction,
                                              random.Random(f'{args.seed}:{step}:{group}:{index}'), stats)
                kept = exp0_sft.chains_of([episode], max_len=args.max_sequence_tokens)
                stats['calls_too_long_dropped'] += len(episode['rows']) - sum(len(c['segs']) for c in kept)
                key = len(episode_units)
                spread_keys[-1].append(key)
                episode_advantage[key] = advantage
                episode_units[key] = [sum(n for c in kept for _, n, _ in c['segs']), sum(len(c['segs']) for c in kept), 0]
                for chain in kept:
                    chain['advantage'] = advantage
                    chain['episode'] = key
                    chain['role'] = episode.get('role', 'student')
                    # rho per staleness (``rho_by_staleness``) covers the students' own sampled episodes only: a
                    # JOLT+ teacher attempt trains in student form without its guidance, so its ratio is no actor drift
                    chain['stale'] = rho_stale(chain['role'], stale)
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
            replays += [(episode, args.sd_weight, 'sd', episode.get('sd_source', 'own'))
                        for episode in sd_take(sd_buffer, step, args, stats)]
        if teacher_demos is not None and sd_teacher_on:
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
        rrl_stats = None
        if reasoning_rl is not None:
            # reasoning RL against fixed teacher actions (ReasoningRL): each sample's reasoning at its group-normalized
            # action likelihood, the teacher action after it as a ratio-free cross-entropy
            rrl_groups, rrl_dropped, carried = reasoning_rl.take(step, args.reasoning_rl_groups, args.max_staleness)
            rrl_items, rrl_stats = rrl_episodes(rrl_groups, args.reasoning_rl_weight, args.reasoning_rl_action_weight,
                                                 args.reasoning_rl_unclosed)
            rrl_stats['groups_stale'] = rrl_dropped
            rrl_stats['carried_chains'] = len(carried)
            for episode, role in rrl_items:
                kept = exp0_sft.chains_of([episode], max_len=args.max_sequence_tokens)
                if not kept:
                    rrl_stats['too_long'] += 1
                    continue
                key = len(episode_units)
                episode_advantage[key] = 1.          # the rows carry their own advantages (token_advantage)
                episode_units[key] = [sum(n for c in kept for _, n, _ in c['segs']), sum(len(c['segs']) for c in kept), 0]
                for chain in kept:
                    chain.update(advantage=1., episode=key, role=role)
                    chains.append(chain)
            if args.sd_weight > 0:   # the searches' carried chains join the self-distillation replay (source search)
                sd_buffer += [{'episode': episode, 'version': step, 'uses': 0} for episode in carried]
                for episode in carried:
                    stats[f"sd_{episode['sd_source']}_added"] += 1
        replay_record = {}
        for corpus in replay_corpora:
            entry = replay_record.setdefault(corpus.name, collections.Counter())
            for _, unit in corpus.take(step):
                kept = [c for c in unit if len(c['ids']) + exp0_sft.source_tokens(c) <= args.max_sequence_tokens]
                if len(kept) < len(unit):
                    entry['chains_too_long'] += len(unit) - len(kept)
                if not kept:
                    continue
                key = len(episode_units)
                episode_advantage[key] = corpus.weight
                episode_units[key] = [sum(n for c in kept for _, n, _ in c['segs']), sum(len(c['segs']) for c in kept), 0]
                entry['units'] += 1
                entry['chains'] += len(kept)
                entry['imports'] += sum(len(c.get('imports') or ()) for c in kept)
                entry['loss_tokens'] += episode_units[key][0]
                entry['forward_tokens'] += sum(len(c['ids']) + exp0_sft.source_tokens(c) for c in kept)
                for chain in kept:
                    chain.update(advantage=corpus.weight, episode=key, role='replay', replay_corpus=corpus.name)
                    chains.append(chain)
        if args.advantage_control != 'none':
            control_advantages([*chains, *decisions], episode_advantage, args.advantage_control,
                               random.Random(args.seed + step))
        # Loss normalization (loss_scales; --loss-norm token by default: one sum over the step's units / its units, a
        # sampled decision counting decision_unit units). spread_data counts each group with spread by its units.
        loss_tokens = sum(length for c in chains for _, length, _ in c['segs'])
        scales = loss_scales(episode_units, args.loss_norm, args.decision_weight, args.decision_unit_cap)
        unit_counts = episode_unit_counts(episode_units, args.decision_weight, args.decision_unit_cap)
        step_data = spread_data([sum(unit_counts[k][0] for k in keys) for keys in spread_keys],
                                args.lr_reference_group_units)
        episodes_trained = sum(1 for token_scale, decision_scale in scales.values() if token_scale or decision_scale)
        phase['lower'] = time.time() - t
        # The decoded imported states of the step's calls (lowering read them for their token ids) are not held through
        # the learner phase (up to the step's whole set of blocks, GBs of host memory): the learner reads each chain's
        # blocks again when it computes that chain (``chain_states``).
        imported_state_mb = sum(t.numel() * t.element_size() for block in collector.states.cache.values()
                                for layer in block.layers for t in layer.values() if hasattr(t, 'numel')) / 2**20
        collector.states.cache.clear()
        record = {'step': step + 1, 'groups': len(groups), 'groups_with_spread': spread,
                  'spread_data': round(step_data, 4),
                  'advantage_control': args.advantage_control if args.advantage_control != 'none' else None,
                  'replay_groups': [min(groups), max(groups)] if args.replay and groups else None,
                  'spread_fraction': round(spread / max(1, len(groups)), 3), 'dropped_stale_groups': dropped_stale,
                  'staleness': dict(collections.Counter(staleness)), 'episodes': len(rewards),
                  'reward_mean': round(sum(rewards) / max(1, len(rewards)), 4), 'sequences': len(chains),
                  'loss_tokens': loss_tokens, 'decisions': len(decisions), 'lowering': dict(stats),
                  'loss_norm': args.loss_norm, 'decision_weight': args.decision_weight, 'episodes_trained': episodes_trained,
                  'rollouts_archive_free_gib': round(shutil.disk_usage(rollouts_archive).free / 2**30, 1)
                  if rollouts_archive is not None and rollouts_archive.exists() else None,
                  'actor_version': versions_live['actor'], 'imported_state_mb': round(imported_state_mb, 1),
                  'neuralese_passthrough': not args.no_neuralese_passthrough,
                  'producer_document_tokens': sum(exp0_sft.source_tokens(c) for c in chains),
                  **({'replay_corpora': {name: dict(entry) for name, entry in replay_record.items()}}
                     if replay_corpora else {}),
                  **({'memetic_groups': memetic_groups, 'memetic': memetic_step_record(memetic_groups, memetic_history)}
                     if memetic_groups or args.memetic_share > 0 else {}),
                  'dynamic_sampling': dict(getattr(collector, 'dynamic', {})) or None,   # cumulative since this process started
                  **restore_telemetry(stats, args)}
        if rrl_stats is not None:
            record['reasoning_rl'] = rrl_record(rrl_stats, reasoning_rl.counts, len(reasoning_rl.ready))
        # rollouts only: replay chains are fixed data, and the reasoning RL samples are drawn on teacher prompts
        style = reasoning_style([c for c in chains if c.get('role') not in ('replay', RRL_ROLE)],
                                think_close)
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
        cache_release = {}                # per cache release (free_cuda_cache): what gc freed, allocated, reserved
        peak_events = {}                  # per pass: the leaf batches where the allocated peak grew by >= 1 GB

        def note_peak(name, tokens, chains_in_batch):
            peak = torch.cuda.max_memory_allocated() / 2**30      # monotone within a phase (phase_peak resets it)
            events = peak_events.setdefault(name, [])
            if len(events) < 16 and (not events or peak >= events[-1]['allocated_gib'] + 1):
                events.append({'allocated_gib': round(peak, 1), 'batch_tokens': int(tokens), 'chains': chains_in_batch})

        def phase_peak(name):
            cuda_peaks[name] = {'allocated': round(torch.cuda.max_memory_allocated() / 2**30, 1),
                                'reserved': round(torch.cuda.max_memory_reserved() / 2**30, 1), **process_memory_gb()}
            torch.cuda.reset_peak_memory_stats()
        collector.active_slots = args.learner_phase_slots
        # episodes still running when the learner starts (they finish while it computes: their collectors and KV are the
        # memory the learner phase shares with them)
        record['episodes_in_flight'] = {'learner_start': collector.running}
        learn_started, learn_tokens0 = time.time(), actor_generated(args.actor)
        if release_tags:
            # the actor's KV, weights and CUDA graphs go back to the system for this phase; episodes in flight wait in its
            # queues (case_seconds counts the wait) and continue after the step's weight update (WeightSync.publish)
            free0 = torch.cuda.mem_get_info()[0]
            phase['actor_release'] = sync.release(release_tags)
            record['actor_release_freed_gib'] = round((torch.cuda.mem_get_info()[0] - free0) / 2**30, 1)
        swap0, mem0 = vmstat(), meminfo()
        progress_mark = time.time()
        if phase_mark['tokens'] is not None and learn_tokens0 is not None:
            phase_totals['collect_s'] += learn_started - phase_mark['time']
            phase_totals['collect_tokens'] += learn_tokens0 - phase_mark['tokens']
        sums = collections.Counter()
        stale_buckets = collections.defaultdict(lambda: {'tokens': 0, 'rho_sum': 0.0, 'abs_log_rho_sum': 0.0, 'rho_capped': 0})
        rho_pre_max, rho_post_max, forward_tokens = 0., 0., 0
        teacher_log_rho = []   # per-chain arrays of the teacher tokens' log rho (``teacher_rho_summary``)
        rng = random.Random(args.seed + step)
        rng.shuffle(chains)
        if args.save_step_chains > 0:
            record['step_chains_mb'] = save_step_chains(work / 'step-chains', step + 1, chains, scales,
                                                        args.save_step_chains)
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

        def run_tree(leaf, backward=True, *, adapters=None, trainable_rows=None, cur_pack=None, cur_act=None):
            return lfm3_fast_train.run_prefix_tree(
                model, nodes, ids_of, leaf, tokens_of,
                cur_pack if cur_pack is not None else oom_plan.pack_tokens,
                cur_act if cur_act is not None else oom_plan.activation_gib,
                backward, mode=tree_mode if share_prefix_tokens > 0 else 'levels',
                batch_tokens=args.tree_batch_tokens, retained_gb=args.prefix_retained_gb,
                adapters=adapters, trainable_rows=trainable_rows, fp8_base=learner.fp8_base,
            )

        refs = {}
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

            def scores_of(net, tokens=False, *, adapters=None, trainable_rows=None):
                h = forward(net, ids, lengths, imports, adapters=adapters, trainable_rows=trainable_rows,
                            fp8_base=learner.fp8_base)
                token = -exp0_sft.token_nll(net, h, positions, targets, trainable_rows=trainable_rows)
                scores = torch.zeros(options, device='cuda', dtype=token.dtype).index_add(0, owner, token)
                return (scores, token, owner) if tokens else scores
            return ids, counts, scores_of

        ref_scores_of = []
        ref_tokens_of = []
        if reference_active:
            # The reference pass runs exactly the policy's training graph: grad mode, leaves that require grad (the
            # reference's own, never stepped; its outputs detached and never backpropagated), the same checkpointing.
            # A no-grad pass compiles other kernels, whose BF16 rounding alone put KL at 5e-4 (|dlogp| ~0.03) with
            # identical weights (smoke 2026-11-12); the bare-base path is another graph too.
            t = time.time()

            def reference_leaves(items, run):
                ids, positions, targets, imports, lengths, counts = batch_inputs(model, items)
                logp = -exp0_sft.token_nll(model, run(ids, lengths, imports), positions, targets,
                                           trainable_rows=ref_rows).detach()
                for (index, _), part in zip(items, logp.split(counts)):
                    refs[index] = part
                note_peak('reference', len(ids), len(items))
            with torch.enable_grad():
                run_tree(reference_leaves, backward=False, adapters=ref_bundles, trainable_rows=ref_rows)
                for batch in batches:
                    ids, _, scores_of = decision_inputs(batch)
                    exp0_sft.set_checkpointing(model, len(ids), oom_plan.activation_gib, parameterization='lora')
                    ref_scores, ref_tokens, _ = scores_of(model, tokens=True, adapters=ref_bundles, trainable_rows=ref_rows)
                    ref_scores_of.append(ref_scores.detach())
                    ref_tokens_of.append(ref_tokens.detach())
            for leaf in reference_tensors(ref_bundles, ref_rows):
                leaf.grad = None
            torch.cuda.synchronize()
            phase['reference_forward'] = phase.get('reference_forward', 0.) + time.time() - t
            phase_peak('reference')
            if not args.keep_learner_cache:
                cache_release['reference'] = free_cuda_cache(not args.no_learner_gc)
        record['episodes_in_flight']['policy_start'] = collector.running
        policy_started = time.time()

        def run_policy_and_decisions(cur_pack, cur_act):
            nonlocal rho_pre_max, rho_post_max, forward_tokens, progress_mark
            sums.clear()
            stale_buckets.clear()
            teacher_log_rho.clear()
            rho_pre_max = rho_post_max = 0.
            forward_tokens = 0
            step_refs = dict(refs)

            def train_leaves(items, run):
                nonlocal rho_pre_max, rho_post_max, forward_tokens, progress_mark
                ids, positions, targets, imports, lengths, counts = batch_inputs(model, items)
                h = run(ids, lengths, imports)
                if args.top_p < 1:
                    nll, nucleus_nll, outside = exp0_sft.token_nll_nucleus(
                        model, h, positions, targets, args.top_p, trainable_rows=learner.trainable_rows)
                    logp_rows, policy_rows = -nll, -nucleus_nll
                    sums['outside_nucleus'] += int(outside.sum())
                    sums['nucleus_log_mass'] += (logp_rows.detach() - policy_rows.detach()).sum().item()
                    if args.non_student_loss == 'full':
                        sampled = torch.from_numpy(np.concatenate([student_sampled_targets(chains[index])
                                                                   for index, _ in items])).to(nll.device)
                        policy_rows = torch.where(sampled, policy_rows, logp_rows)
                        sums['non_student_tokens'] += int((~sampled).sum())
                else:
                    logp_rows = policy_rows = -exp0_sft.token_nll(
                        model, h, positions, targets, trainable_rows=learner.trainable_rows)
                p_close_rows = (close_probability(h, positions, model.lm_head.weight, targets, logp_rows, think_close,
                                                  trainable=head_override(learner.trainable_rows))
                                if args.close_push and think_close is not None else None)
                loss = 0.
                for k, ((index, _), logp, policy_logp) in enumerate(zip(items, logp_rows.split(counts), policy_rows.split(counts))):
                    chain = chains[index]
                    actor, row_weight = behaviour_logp(chain, logp)
                    token_scale = scales[chain['episode']][0]
                    ref = step_refs.pop(index, None)
                    with torch.no_grad():
                        log_rho = logp.detach() - actor
                        rho = log_rho.clamp(max=20).exp()
                        advantage = token_advantage(chain, logp.device)
                        weight, tail = policy_weight(rho, actor, advantage, args.rho_cap,
                                                     None if chain.get('role') == 'replay' else args.tail_logp_floor)
                        if row_weight is not None:
                            weight = weight * row_weight
                        if args.second_close == 'context' and think_close is not None:
                            bad = malformed_close_targets(chain, advantage, think_close, second_close_stops)
                            if bad is not None:
                                weight = weight.masked_fill(bad.to(weight.device), 0.)
                                sums['second_close_masked'] += int(bad.sum())
                        mass = token_scale * abs(advantage) * weight * (1 - policy_logp.detach().exp())
                        thinking = torch.from_numpy(np.concatenate([think_mask(row['sample'], think_close)
                                                                    for _, _, row in chain['segs']])).cuda()
                        sums['mass_think'] += mass[thinking].sum().item()
                        sums['mass_post_think'] += mass[~thinking].sum().item()
                        sums['tail_masked'] += int(tail.sum())
                        if p_close_rows is not None:
                            chain_targets, chain_p_close = targets.split(counts)[k], p_close_rows.split(counts)[k]
                            sums['close_push:' + (chain.get('role') or 'student')] += close_push(
                                token_scale * advantage * weight, chain_targets, chain_p_close, think_close, thinking)
                    loss_terms = -(weight * advantage * policy_logp)
                    if ref is not None:
                        d = ref - logp
                        loss_terms = loss_terms + args.kl_beta * kl_term(d, args.kl_estimator)
                        kl_stats(sums, d)
                        if p_close_rows is not None:
                            pull = d.detach() if args.kl_estimator == 'k2' else d.detach().exp() - 1
                            sums['close_push:kl'] += close_push(token_scale * args.kl_beta * pull, chain_targets,
                                                                chain_p_close, think_close, thinking)
                    loss = loss + loss_terms.sum() * token_scale
                    twin = twin_rows(chain, logp.device) if chain.get('role') == RRL_ROLE else None
                    if twin is not None:
                        t_ids, t_logp, t_scale = twin
                        on = t_scale > 0
                        chain_positions, chain_targets = positions.split(counts)[k], targets.split(counts)[k]
                        fkl = twin_forward_kl(h, chain_positions[on], model.lm_head.weight, chain_targets[on], logp[on],
                                              t_ids[on], t_logp[on], trainable=head_override(learner.trainable_rows))
                        loss = loss + (t_scale[on] * fkl).sum() * token_scale
                        with torch.no_grad():
                            sums['twin_fkl'] += fkl.detach().sum().item()
                            sums['twin_tokens'] += int(on.sum())
                            teacher_close = torch.where(t_ids[on] == think_close, t_logp[on].exp(),
                                                        torch.zeros_like(t_logp[on])).sum(-1)
                            sums['twin_close_teacher'] += teacher_close.sum().item()
                            if p_close_rows is not None:
                                student_close = p_close_rows.split(counts)[k][on]
                                sums['twin_close_student'] += student_close.sum().item()
                                sums['close_push:reasoning_rl_twin'] += float(
                                    (token_scale * t_scale[on] * (teacher_close - student_close)).sum())
                    if chain.get('role') == RRL_ROLE:
                        start = 0
                        for _, length, row in chain['segs']:
                            part = slice(start, start + length)
                            start += length
                            if row.get('rrl') == 'action':
                                sums['rrl_action_logp'] += logp[part].detach().sum().item()
                                sums['rrl_action_tokens'] += length
                            elif row.get('rrl') == 'reasoning':
                                sums['rrl_reasoning_abs_log_rho'] += log_rho[part].abs().sum().item()
                                sums['rrl_reasoning_tokens'] += length
                    if chain.get('role') == 'replay':
                        sums[f"replay_logp:{chain['replay_corpus']}"] += logp.detach().sum().item()
                        sums[f"replay_tokens:{chain['replay_corpus']}"] += logp.numel()
                    sums['rho'] += rho.sum().item()
                    if chain.get('role') == 'teacher':
                        teacher_log_rho.append(log_rho.float().cpu().numpy())
                    sums['rho_capped'] += (rho > args.rho_cap).sum().item()
                    sums['abs_log_rho'] += log_rho.abs().sum().item()
                    if chain.get('stale') is not None:
                        sb = stale_buckets[int(chain['stale'])]
                        sb['tokens'] += int(logp.numel())
                        sb['rho_sum'] += float(rho.sum().item())
                        sb['abs_log_rho_sum'] += float(log_rho.abs().sum().item())
                        sb['rho_capped'] += int((rho > args.rho_cap).sum().item())
                    consumer = None
                    if chain.get('imports'):
                        consumer = torch.from_numpy(np.concatenate([np.full(n, bool(row.get('imports')))
                                                                    for _, n, row in chain['segs']])).cuda()
                        sums['consumer_tokens'] += int(consumer.sum())
                        sums['consumer_abs_log_rho'] += log_rho.abs()[consumer].sum().item()
                    if chain.get('role') not in RATIO_FREE_ROLES:
                        mismatch = actor_mismatch_counts(log_rho, consumer)
                        for key in ('below', 'above', 'below_ordinary'):
                            sums['mismatch_' + key] += mismatch[key]
                        sums['mismatch_min'] = min(sums.get('mismatch_min', 0.), mismatch['min'])
                    sums['logp'] += logp.detach().sum().item()
                    rho_pre_max = max(rho_pre_max, rho.max().item())
                    rho_post_max = max(rho_post_max, weight.max().item())
                loss.backward()
                forward_tokens += len(ids)
                note_peak('policy', len(ids), len(items))
                del h, logp_rows, policy_rows, loss, p_close_rows
                if time.time() - progress_mark > 300:
                    progress_mark = time.time()
                    log_line(log, {'event': 'learner progress', 'step': step + 1, 'leaf_tokens': forward_tokens,
                                   'policy_s': round(time.time() - policy_started), 'meminfo_gib': meminfo(),
                                   'swap_pages': {k: v - swap0[k] for k, v in vmstat().items()}})

            prefix_info = run_tree(
                train_leaves, adapters=learner.bundles, trainable_rows=learner.trainable_rows,
                cur_pack=cur_pack, cur_act=cur_act,
            )
            torch.cuda.synchronize()
            phase['policy_forward_backward'] = time.time() - policy_started
            phase_peak('policy')
            dec_records = []
            for batch_index, batch in enumerate(batches):
                ids, counts, scores_of = decision_inputs(batch)
                t = time.time()
                ref_scores = ref_scores_of[batch_index] if reference_active else None
                exp0_sft.set_checkpointing(model, len(ids), cur_act, parameterization='lora')
                scores, token_logp, owner = scores_of(
                    model, tokens=True, adapters=learner.bundles, trainable_rows=learner.trainable_rows)
                ref_tokens = ref_tokens_of[batch_index] if reference_active else None
                terms = []
                offsets = np.cumsum([0] + counts[:-1]).tolist()
                # every decision of a forward (decision_batches) reads that forward's option scores
                for item, k, start in ((item, k, start) for same, k, start in zip(batch, counts, offsets) for item in same):
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
                        dec_records.append(decision_record(
                            item, scores[start:start + k].detach().float().cpu().numpy(), row['scores'],
                            None if ref_scores is None else ref_scores[start:start + k].float().cpu().numpy(), log_rho.item()))
                    term = -(weight * item['advantage'] * logpi)
                    if ref_scores is not None:
                        d = torch.log_softmax(ref_scores[start:start + k] / row['temperature'], 0)[row['chosen']] - logpi
                        term = term + args.kl_beta * kl_term(d, args.kl_estimator)
                        kl_stats(sums, d, 'decision_')
                    term = term * decision_scale
                    if ref_tokens is not None and args.decision_token_kl:
                        chosen = owner == start + row['chosen']
                        term = term + decision_token_kl(token_logp[chosen], ref_tokens[chosen], args.kl_beta, args.kl_estimator,
                                                        scales[item['episode']][0])
                        kl_stats(sums, ref_tokens[chosen] - token_logp[chosen], 'decision_token_')
                        sums['decision_token_kl_tokens'] += int(chosen.sum())
                    terms.append(term)
                    sums['decision_rho'] += rho.item()
                    sums['decision_abs_log_rho'] += log_rho.abs().item()
                    sums['decision_logpi'] += logpi.item()
                    rho_pre_max = max(rho_pre_max, rho.item())
                    rho_post_max = max(rho_post_max, weight.item())
                torch.stack(terms).sum().backward()
                forward_tokens += len(ids)
                del scores, terms, token_logp
                torch.cuda.synchronize()
                phase['decision_forward_backward'] = phase.get('decision_forward_backward', 0.) + time.time() - t
            return prefix_info, dec_records

        record['prefix_tree'], decision_records = run_step_with_oom_retry(run_policy_and_decisions, learner, oom_plan)
        forward_tokens = computed_tokens
        cuda_ids.clear()
        record.update(chain_tokens=chain_tokens, shared_prefix_tokens_saved=chain_tokens - computed_tokens)
        if batches:
            phase_peak('decisions')
        if decision_records:
            (work / 'decisions').mkdir(exist_ok=True)
            with (work / 'decisions' / f'step-{step + 1:05d}.jsonl').open('w') as handle:
                handle.writelines(json.dumps(r) + '\n' for r in decision_records)
        record['decision_diagnostics'] = decision_summary(decision_records)
        t = time.time()
        data_scale = lr_data_scale(step_data, args.lr_reference_spread)
        # The data factor scales the gradient too, before the clip: AdamW's moments carry a step's gradient into the next
        # ones at their own LR, so an LR factor alone let a decision-only step's clipped gradient (norm 1 against 0.2-0.3)
        # keep pushing its few decisions through the following chain steps (bench 2026-11-22: their option scores kept
        # moving while their share of those steps' gradient was 1e-4). grad_norm stays the norm of the step's mean loss.
        grad_norm, grad_norm_applied = scale_and_clip(parameters, data_scale if args.data_scaled_gradient else 1.,
                                                      args.clip)
        record['grad_norm_applied'] = round(grad_norm_applied, 5)
        if not math.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            sync.restore(step)
            collector.active_slots = args.slots
            log_line(log, {**record, 'event': 'non-finite gradient: update skipped', 'grad_norm': repr(grad_norm),
                           'loss_sums': {k: v for k, v in sums.items() if isinstance(v, float)}})
            continue
        fam_grads = learner.family_grad_norms()
        cosine = moment_cosine(parameters, optimizer)
        lr = args.lr * min(1., (step + 1) / args.warmup) * data_scale
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        phase['optimizer'] = time.time() - t
        phase_peak('optimizer')
        cache_release['collect'] = free_cuda_cache(not args.no_learner_gc)
        if args.no_learner_gc:
            release_host_memory()
        record['cuda_reserved_collect_gib'] = round(torch.cuda.memory_reserved() / 2**30, 1)
        record['cuda_cache_release'] = cache_release
        record['family_grad_norms'] = fam_grads
        record['oom_retries'] = oom_plan.retries
        record['effective_policy_sha256'] = learner.effective_policy_sha256()
        torch.cuda.reset_peak_memory_stats()
        if probes:
            record['probe_learner_nll'] = round(probe_learner_nll(), 5)
        if behaviour is not None:
            t = time.time()
            start = json.loads(behaviour_start_path.read_text()) if behaviour_start_path.exists() else None
            record['behaviour'] = behaviour_probe.summary(behaviour_probe.measure(model, behaviour), start, behaviour)
            record['behaviour_shift'] = behaviour_probe.shifted(record['behaviour'])
            phase['behaviour_probe'] = time.time() - t
            phase_peak('behaviour_probe')
            if record['behaviour_shift']:
                print(f"behaviour shift at step {step + 1}: {record['behaviour']}", flush=True)
        step += 1
        consumed_groups.update(int(g) for g in groups)
        collector.active_slots = args.slots
        learn_ended, learn_tokens1 = time.time(), actor_generated(args.actor)
        if learn_tokens0 is not None and learn_tokens1 is not None:
            phase_totals['learn_s'] += learn_ended - learn_started
            phase_totals['learn_tokens'] += learn_tokens1 - learn_tokens0
        phase_mark.update(time=learn_ended, tokens=learn_tokens1)
        # publication of this step's policy (WeightSync: snapshot now, merge/bind/attest in the background)
        t = time.time()
        sync.snapshot()                   # waits for the previous publication
        phase['snapshot'] = time.time() - t
        if sync.last and not sync.last.get('ok'):
            # the publish thread already retried once and served the last good version again on every route
            raise RuntimeError(f"publishing version {sync.last.get('version')} to the actor failed twice; the actor serves "
                               f"version {sync.last.get('rolled_back_to')} again: {sync.last.get('answer')}")
        held_due = args.held_every_hours > 0 and not args.held_remote and time.time() - last_held > args.held_every_hours * 3600
        save_native = (step % args.save_every == 0 or (args.max_steps and step == args.max_steps) or held_due
                       or step <= args.keep_checkpoints_until)
        sync.publish(step, on_published, also=(lambda s=step: write_native(s)) if save_native else None)
        if getattr(args, 'export_factors_dir', None):
            from natlang_neuralese.train.learner_state import export_adapter_factors
            t_fe = time.time()
            fe_dir = Path(args.export_factors_dir) / f'step-{step:05d}'
            fe_info = export_adapter_factors(
                learner, fe_dir, base_checkpoint_sha256=source_sha, learner_step=step,
                round_id=f'grpo-step-{step:05d}', objective_version='natlang_grpo_v1',
            )
            record['factor_export'] = {
                'dir': str(fe_dir), 'sha256': fe_info['adapter_factors_sha256'],
                'bytes': fe_info['adapter_factors_bytes'], 'seconds': round(time.time() - t_fe, 2),
            }
        branch = branch_root / f'state-step-{step:05d}' if args.branch_states else None
        if branch is not None or (args.state_every and step % args.state_every == 0):
            if branch is not None:
                for old in sorted(branch_root.glob('state-step-*'))[:-(args.branch_states - 1) or None]:
                    discard(old)
            t_sv = time.time()
            target_state_dir = branch or state_dir
            save_learner_state(
                target_state_dir, learner, optimizer=optimizer, learner_step=step,
                source_checkpoint_id=str(args.checkpoint), base_weights_sha256=source_sha,
                objective_version='natlang_grpo_v1', trainer_state={'oom_plan': oom_plan.to_dict()},
                optimizer_config={'name': 'adamw', 'lr': args.lr, 'weight_decay': args.weight_decay,
                                  'betas': [0.9, 0.99], 'warmup': args.warmup},
                batch_cursor={'step': step, 'cursor': collector.cursor, 'next_group': collector.next_group},
                rl_rollout_ledger={'cursor': collector.cursor, 'next_group': collector.next_group,
                                   'consumed_groups': sorted(consumed_groups),
                                   'actor_version': versions_live['actor']},
                ref_bundles=ref_bundles, ref_rows=ref_rows,
            )
            last_state_save_info = {'path': str(target_state_dir), 'seconds': round(time.time() - t_sv, 1)}
        learner_seconds = sum(v for k, v in phase.items()
                              if k in ('reference_forward', 'policy_forward_backward', 'decision_forward_backward', 'optimizer'))
        record['compile_health'] = compile_health.record_step(learner_seconds, is_warmup=(step == 1))
        n_dec = max(1, len(decisions))
        kl_n = max(1, sums['kl_n'])
        mass_total = sums['mass_think'] + sums['mass_post_think'] + sums['mass_decision']
        elapsed = time.time() - run['started']
        record.update(
            loss_tokens=loss_tokens, forward_tokens=forward_tokens, lr=lr, lr_data_scale=round(data_scale, 4),
            grad_norm=round(grad_norm, 4),
            kl_mean=round(sums['kl'] / kl_n, 7), ref_abs_dlogp_mean=round(sums['ref_abs_dlogp'] / kl_n, 6),
            decision_kl_k2_mean=round(sums['decision_kl_k2'] / sums['decision_kl_n'], 7) if sums['decision_kl_n'] else None,
            decision_token_kl_k2_mean=round(sums['decision_token_kl_k2'] / sums['decision_token_kl_n'], 7)
            if sums['decision_token_kl_n'] else None,
            second_close_masked=int(sums['second_close_masked']), decision_token_kl_tokens=int(sums['decision_token_kl_tokens']),
            close_push={key.split(':', 1)[1]: round(1e6 * value, 3) for key, value in sums.items()
                        if key.startswith('close_push:')} or None,
            kl_k2_mean=round(sums['kl_k2'] / kl_n, 7), kl_k3_max_share=round(sums['kl_k3_max'] / sums['kl'], 4) if sums['kl'] > 0 else None,
            ref_dlogp_pos_max=round(sums['ref_dlogp_pos_max'], 3),
            ref_dlogp_max=round(sums['ref_dlogp_max'], 3), ref_dlogp_over_5=sums['ref_dlogp_over_5'], kl_estimator=args.kl_estimator,
            mean_logp=round(sums['logp'] / max(1, loss_tokens), 4),
            grad_moment_cosine=None if cosine is None else round(cosine, 5),
            top_p=args.top_p, non_student_tokens=int(sums['non_student_tokens']), outside_nucleus_fraction=round(sums['outside_nucleus'] / max(1, loss_tokens), 6),
            nucleus_mass_mean=round(math.exp(sums['nucleus_log_mass'] / max(1, loss_tokens)), 5),
            tail_logp_floor=args.tail_logp_floor, tail_masked_fraction=round(sums['tail_masked'] / max(1, loss_tokens), 6),
            tail_masked_decisions=sums['tail_masked_decisions'],
            grad_mass_share={k: round(sums['mass_' + k] / mass_total, 4) for k in ('think', 'post_think', 'decision')}
            if think_close is not None and mass_total > 0 else None,
            decision_rho_mean=round(sums['decision_rho'] / n_dec, 4), decision_abs_log_rho_mean=round(sums['decision_abs_log_rho'] / n_dec, 5),
            decision_mean_logpi=round(sums['decision_logpi'] / n_dec, 4),
            decision_abs_dscore_mean=round(sums['decision_abs_dscore'] / n_dec, 4),
            teacher_rho=teacher_rho_summary(teacher_log_rho, args.rho_cap),
            **({'reasoning_rl': {**record['reasoning_rl'],
                                 'learner_action_logp': round(sums['rrl_action_logp'] / max(1, sums['rrl_action_tokens']), 4),
                                 'reasoning_abs_log_rho': round(sums['rrl_reasoning_abs_log_rho']
                                                                / max(1, sums['rrl_reasoning_tokens']), 5),
                                 **({'twin_fkl_mean': round(sums['twin_fkl'] / sums['twin_tokens'], 5),
                                     'twin_tokens': int(sums['twin_tokens']),
                                     'twin_p_close_twin': round(sums['twin_close_teacher'] / sums['twin_tokens'], 6),
                                     'twin_p_close_policy': round(sums['twin_close_student'] / sums['twin_tokens'], 6)}
                                    if sums['twin_tokens'] else {})}}
               if 'reasoning_rl' in record else {}),
            replay_mean_logp={key.split(':', 1)[1]: round(sums[key] / max(1, sums['replay_tokens:' + key.split(':', 1)[1]]), 4)
                              for key in list(sums) if key.startswith('replay_logp:')} or None,
            rho_mean=round(sums['rho'] / max(1, loss_tokens), 4), rho_pre_cap_max=round(rho_pre_max, 3),
            rho_post_cap_max=round(rho_post_max, 3), rho_capped_fraction=round(sums['rho_capped'] / max(1, loss_tokens), 5),
            abs_log_rho_mean=round(sums['abs_log_rho'] / max(1, loss_tokens), 5),
            rho_by_staleness=rho_by_staleness_summary(stale_buckets),
            consumer_tokens=sums['consumer_tokens'],
            consumer_abs_log_rho_mean=round(sums['consumer_abs_log_rho'] / max(1, sums['consumer_tokens']), 5),
            **dict(zip(('actor_mismatch', 'actor_mismatch_shift'), actor_mismatch_summary(sums, loss_tokens))),
            actor_overlap=actor_overlap(args) if not args.replay else None,
            learner_tokens_per_s=round(forward_tokens / max(1e-6, learner_seconds)),
            actor_generated_tokens=collector.generated, actor_episodes=collector.episodes,
            actor_generated_tokens_per_s=round(collector.generated / max(1., elapsed)),
            phase_s={k: round(v, 1) for k, v in phase.items()}, previous_sync=sync.last,
            actor_tokens_per_s_learn_phase=round(phase_totals['learn_tokens'] / phase_totals['learn_s']) if phase_totals['learn_s'] else None,
            actor_tokens_per_s_collect_phase=round(phase_totals['collect_tokens'] / phase_totals['collect_s']) if phase_totals['collect_s'] else None,
            learner_phase_slots=args.learner_phase_slots,
            learner_swap_pages={k: v - swap0[k] for k, v in vmstat().items()}, meminfo_learn_start_gb=mem0,
            previous_state_save=last_state_save_info, cuda_peak_gib=max([v['allocated'] for v in cuda_peaks.values()], default=None),
            cuda_peak_by_phase_gib=cuda_peaks, cuda_peak_events=peak_events, host_pinned_gib=host_pinned_gb(),
            outstanding_groups=collector.outstanding, elapsed_s=round(elapsed))
        record.update(grad_spike(work, grad_norm))
        if record.get('grad_spike'):
            print(f"grad spike: grad_norm {grad_norm:.3g} is {record['grad_norm_vs_median']}x the run's median", flush=True)
        if not args.keep_rollouts and args.block_gc_grace >= 0:
            t = time.time()
            with collector.lock:
                oldest = min([info['created'] for info in collector.groups.values()], default=time.time())
            files, freed = collect_blocks(collector.states.directory, oldest - args.block_gc_grace)
            record.update(blocks_removed=files, blocks_freed_mb=round(freed / 2**20, 1),
                          block_gc_s=round(time.time() - t, 2))
        log_line(log, record)
        if teacher_demos is not None:
            teacher_demos.save()
        if args.sd_weight > 0:
            sd_buffer_save(sd_buffer_path, sd_buffer)
        if reasoning_rl is not None:
            reasoning_rl.save(rrl_ready_path)
        if save_native and held_due:
            sync.wait()
            held(step, args.held_samples)
            last_held = time.time()
            phase_mark.update(time=time.time(), tokens=actor_generated(args.actor))
    sync.wait()
    collector.stop.set()
    stop.set()
    log_line(log, {'event': 'stopped', 'step': step})


def build_parser():
    """The command line (``train``); ``expert_iteration.py collect`` parses the same flags, so its collector, actor and
    episodes run with exactly grpo's defaults."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--work', required=True)
    t.add_argument('--checkpoint', required=True, help='start policy (native checkpoint); also the KL reference')
    t.add_argument('--checkpoint-sha256')
    t.add_argument('--base', default=BASE)
    t.add_argument('--actor', help='natlang front of an SGLang actor serving --checkpoint (default: a child actor of this '
                   'process with the held-v2 engine arguments)')
    t.add_argument('--extra-actor', action='append', default=[], metavar='URL[=SLOTS]',
                   help='additional BF16 actor front URL[=SLOTS] for multi-actor GRPO collection (async-train §7.1)')
    t.add_argument('--export-factors-dir', help='directory to export ~130 MB FP32 r=32 LoRA factors per step for WAN '
                   'shipping to DGX/G4 actors (async-train §7.1)')
    t.add_argument('--actor-port', type=int, default=8460, help='port of the child actor (its SGLang server: +1000)')
    t.add_argument('--actor-arg', action='append', default=[], help='extra serve_native_checkpoint.py argument')
    t.add_argument('--sglang', help='its SGLang server (default: actor port + 1000)')
    t.add_argument('--actor-store', help="the actor's native block store (--store-dir; default WORK/actor): consumer calls "
                   'read their imported Neuralese state from it')
    t.add_argument('--actor-export', help="the actor's HF export dir (layout and frozen tables; default WORK/actor/sglang-hf)")
    t.add_argument('--pool', action='append', help='task pool cases.jsonl (repeatable; default: the registry role grpo-pool)')
    t.add_argument('--held-manifest', help='held-out manifest (default: the registry role held-eval, held-v3)')
    t.add_argument('--exclude-manifest', action='append', default=[], help='more held manifests to exclude')
    t.add_argument('--task-ids', help='JSON list: train only on these pool ids (smoke)')
    t.add_argument('--group', type=int, default=8)
    t.add_argument('--slots', type=int, default=96, help='concurrent episodes on the actor (its default runs 128 sequences)')
    t.add_argument('--min-groups', type=int, default=24,
                   help='groups per step at least (run2 at 12: about 75 episodes with a nonzero advantage per step, '
                        'consecutive updates did not add up; DECISIONS 2026-11-13)')
    t.add_argument('--actor-release', action=argparse.BooleanOptionalAction, default=True,
                   help='release the actor\'s GPU memory (SGLang release_memory_occupation of --actor-release-tags) for '
                        'each learner phase and resume it with the step\'s weight update: the GB10\'s unified memory goes '
                        'to the learner while it computes, to the actor while it collects. The run\'s own actor runs '
                        'with --sglang-memory-saver. Episodes in flight wait in the paused actor (--case-seconds counts '
                        'the wait). --no-actor-release keeps the actor resident')
    t.add_argument('--actor-overlap', choices=('auto', 'on', 'off'), default='auto',
                   help="SGLang's overlapped scheduler on the run's own actor: the next decode step's CPU scheduling "
                        'runs while the GPU computes the current one (+8-11%% output tok/s measured at fork dbc51ddb72, '
                        'GRPO_STEP_THROUGHPUT_20261117.md). auto (default): on unless held evals measure on this actor '
                        '(their pinned engine serves without it). The 2026-11-03 one-step wrong-distribution defect did '
                        'not reproduce at the fork pin (0/192 turns); train.jsonl actor_mismatch counts such tokens every '
                        'step and actor_mismatch_shift flags them: off opts out')
    t.add_argument('--actor-release-tags', default='kv_cache,weights,cuda_graph',
                   help='SGLang memory types released (the weights are reloaded from the mapped file on resume)')
    t.add_argument('--no-learner-gc', action='store_true',
                   help='release the learner\'s CUDA cache before the garbage collection as before (default: gc.collect '
                        'first, so blocks freed by cycle collection are returned too; free_cuda_cache)')
    t.add_argument('--keep-learner-cache', action='store_true',
                   help='keep PyTorch\'s cached blocks across learner phases (default: empty the cache after the reference '
                        'pass and before the AdamW moments are read back, so the gradients and the moments do not stack '
                        'on the previous pass\'s reserve: grpo-v4-run1 step 2 optimizer 74 GB reserved, 31.5 allocated)')
    t.add_argument('--learner-phase-slots', type=int, default=0,
                   help='collector slots that take new episodes while the learner computes (in-flight episodes always '
                        'finish). 0: none. The GB10 time-slices the actor and the learner, and a few-sequence decode costs '
                        'nearly a full batch\'s GPU time: run1\'s 4 learner-phase slots decoded at 16 tok/s per sequence '
                        '(34 alone) and slowed the learner (GRPO_STEP_THROUGHPUT_20261117.md)')
    t.add_argument('--max-groups', type=int, default=48,
                   help='groups per step at most: about 27k forward tokens per episode (smoke), so 48 groups is ~60-90 min of learner')
    t.add_argument('--max-outstanding', type=int, default=0, help='groups in flight or waiting (default max-groups + slots/group)')
    t.add_argument('--max-staleness', type=int, default=2)
    t.add_argument('--case-seconds', type=int, default=7200,
                   help='wall-clock safety net only (a wall stop is unscored); it includes the time an episode waits in a '
                        'released actor during learner phases (--actor-release: up to ~40 min per step), hence 2 h')
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
    t.add_argument('--teacher-attempts', type=int, default=2,
                   help='JOLT+: privileged teacher attempts of the same task for a group whose students succeeded at most '
                        '--teacher-max-successes times, trained in student form (async-train §8: default 2; 0 opts out)')
    t.add_argument('--teacher-max-successes', type=int, default=0)
    t.add_argument('--teacher-fact-kinds', default=None, help='comma list of fact kinds that guide a teacher (default: rl/jolt.py '
                   'TEACHER_FACT_KINDS, no answers); "all" keeps every kind')
    t.add_argument('--guidance-answers', choices=('withhold', 'show'), default='withhold',
                   help="JOLT+ guidance never carries the result (default withhold: reference/trajectory facts show no "
                        "returned value, facts holding an answer value are left out, the runtime rejects written guidance "
                        "that gives one; rl/privileged_context.py case_facts); show opts out")
    t.add_argument('--guidance-leaks', choices=('drop', 'train'), default='drop',
                   help='a JOLT+ teacher attempt whose replies quote or name its guidance, or whose guidance carried the '
                        'answer, trains nowhere (default drop: episode term, SD, reasoning RL; rl/jolt.py guidance_leak); '
                        'train opts out')
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
    t.add_argument('--sd-teacher-admit-drifted', choices=('on', 'off'), default='on',
                   help='teacher successes whose replay display drifted (sft_sources.drifted_success) join the exact ones')
    t.add_argument('--sd-teacher-source', action='append', default=None,
                   help='sft_sources WORK whose extracted, replayed teacher successes are the demonstrations (repeatable; '
                        'default: the registry role grpo-sd-teacher)')
    t.add_argument('--reasoning-rl-weight', type=float, default=0.0,
                   help='reasoning RL against fixed teacher actions (ReasoningRL): a search along teacher trajectory '
                        'segments samples the student\'s own think blocks; this scales their training term, the forward '
                        'KL from the privileged twin (the policy given the teacher action; --reasoning-rl-objective '
                        'twin-fkl, the default) or their group-normalized return (sequence). Default 0: off, the '
                        'privileged twin\'s KL term that async-train §8 keeps off (JOLT+ guided successes carry its '
                        'signal); 1.0 opts in')
    t.add_argument('--reasoning-rl-action-weight', type=float, default=0.05,
                   help='the teacher action after each closed sampled think block as a ratio-free cross-entropy (0: the '
                        'reasoning term alone)')
    t.add_argument('--reasoning-rl-objective', choices=RRL_OBJECTIVES, default='twin-fkl',
                   help='how sampled think blocks train: twin-fkl (default; 2026-11-20) the token-level forward KL from '
                        'the privileged twin (the policy given the teacher action) at every position, myopic, without '
                        'the future agreement bias of a sequence-level reward (arXiv 2611.23397); sequence (before): '
                        'the group-normalized return (the teacher actions\' likelihood after the block). '
                        '--reasoning-rl-weight scales either')
    t.add_argument('--reasoning-rl-twin-top-k', type=int, default=32,
                   help='twin-fkl: the twin\'s top-k log-probabilities per position (plus </think>); the rest of its '
                        'mass is matched as one bucket')
    t.add_argument('--reasoning-rl-carry-sd', choices=('on', 'off'), default=None,
                   help='a walk\'s best-return chain joins the SD replay (source search); default on for sequence, off '
                        'for twin-fkl (choosing the best-return chain is the sequence-level selection)')
    t.add_argument('--reasoning-rl-rewrite-share', type=float, default=None,
                   help='share of teacher invocations rewritten in parallel (ReasoningRL.rewrite: SD data of the '
                        'best-scoring rewrites); the rest are walked step by step on the student\'s own carried '
                        'reasoning. Default 0.3 for --reasoning-rl-objective sequence, 0 for twin-fkl')
    t.add_argument('--reasoning-rl-rewrite-rounds', type=int, default=4, help='rewrite rounds')
    t.add_argument('--reasoning-rl-rewrite-tokens', type=int, default=1024,
                   help='tokens of one rewrite continuation at most (it must close its think block)')
    t.add_argument('--reasoning-rl-rewrite-accept', choices=('best', 'metropolis'), default='best',
                   help='a rewrite candidate is kept when it scores better (best) or by a Metropolis step')
    t.add_argument('--reasoning-rl-rewrite-temperature', type=float, default=0.05,
                   help='the Metropolis step\'s temperature (nats per action token)')
    t.add_argument('--reasoning-rl-calls', type=int, default=0,
                   help='steps of a teacher invocation searched at most (0: all, from its first call; 1: single decisions)')
    t.add_argument('--reasoning-rl-horizon', type=int, default=4,
                   help='later actions in a think block\'s return (scored in its one forward; 0: all of the rest)')
    t.add_argument('--reasoning-rl-wave', type=int, default=2,
                   help='think blocks drawn at once; a step stops drawing once one scores high')
    t.add_argument('--reasoning-rl-samples', type=int, default=4, help='think blocks per step at most (ordinary step)')
    t.add_argument('--reasoning-rl-max-samples', type=int, default=12,
                   help='think blocks per step at most where the action without reasoning is unlikely (hard step)')
    t.add_argument('--reasoning-rl-high-logp', type=float, default=-0.1,
                   help='a think block scores high when its action\'s mean token log-probability is at least this')
    t.add_argument('--reasoning-rl-hard-logp', type=float, default=-1.0,
                   help='a step is hard when its action\'s mean token log-probability without reasoning is below this')
    t.add_argument('--reasoning-rl-gamma', type=float, default=1.0,
                   help='discount of later actions\' rewards in a think block\'s return (0: the next action only)')
    t.add_argument('--reasoning-rl-beam', type=int, default=1,
                   help='best think blocks carried forward as history per step (1: greedy)')
    t.add_argument('--reasoning-rl-groups', type=int, default=32, help='step groups trained per learner step at most')
    t.add_argument('--reasoning-rl-max-tokens', type=int, default=8192,
                   help='reasoning tokens per sample at most (a sample that has not closed by then ties with its '
                        'group\'s worst)')
    t.add_argument('--second-close', choices=('context', 'train'), default='context',
                   help="a reply's second and later </think> and the tokens after each up to the action: context (default; "
                        'never reinforced at a positive advantage, in every learner row: episodes, SD, reasoning RL, replay, '
                        'teacher demonstrations; exp0_sft.second_close_mask); train: as any token (before 2026-11-19)')
    t.add_argument('--no-close-push', dest='close_push', action='store_false',
                   help="train.jsonl close_push: each term's first-order push on the </think> logit over the reasoning "
                        'tokens it trains (policy roles and the KL term, x1e6; grpo.close_push). Positive raises the '
                        'behaviour probe\'s close hazard; on by default')
    t.add_argument('--no-decision-token-kl', dest='decision_token_kl', action='store_false',
                   help="sampled decisions: no per-token KL on the chosen option's tokens (only the KL of the option "
                        'softmax, which ignores the tokens all options share; before 2026-11-19)')
    t.add_argument('--truncated-episodes', choices=('drop', 'train'), default='drop',
                   help="an episode with a reply cut by a token cap (finish_reason 'length'): drop (default) leaves it out of "
                        "its group's advantages and of training; train: it scores and trains like any other (before "
                        '2026-11-19). The call budget (--max-model-requests) is an outcome either way')
    t.add_argument('--budget-stop-episodes', choices=('drop', 'train'), default='train',
                   help='an episode that ended on the model-request budget (--max-model-requests; outcome '
                        'budget_exhaustion): train (default) scores it 0 like any failure; drop leaves it out of its '
                        'group\'s advantages and of training, as overlong filtering does for a token cap '
                        '(rollout_math.budget_stopped_rollout). Held evaluation counts it as a failure either way')
    t.add_argument('--reasoning-rl-unclosed', choices=('drop', 'floor'), default='drop',
                   help='a reasoning-RL sample that hit --reasoning-rl-max-tokens without closing: drop (default) leaves it '
                        "out of its group's advantages and of training (overlong filtering: a cut block pushed down teaches "
                        '"close earlier"); floor: it ties with the worst closed sample and trains (before 2026-11-19)')
    t.add_argument('--reasoning-rl-min-std', type=float, default=0.01,
                   help='a group whose returns (nats per action token) spread less trains only its actions')
    t.add_argument('--reasoning-rl-workers', type=int, default=8,
                   help='trajectories searched in parallel (each a thread; their draws batch on the actor)')
    t.add_argument('--reasoning-rl-teacher-history', choices=('keep', 'empty'), default='keep',
                   help='the teacher\'s earlier calls in a segment\'s history: the recorded teacher reasoning (default), '
                        'or empty think blocks')
    t.add_argument('--reasoning-rl-sources', default='jolt,demos',
                   help='teacher trajectories: jolt (this run\'s JOLT+ teacher successes), demos (TeacherDemos)')
    t.add_argument('--replay-corpus', action='append', default=None, metavar='NAME=PKL[,weight=W][,units=K]',
                   help='a fixed corpus of per-call chains (exp0_sft format: operator rows from operator_rows.py merge, '
                        'whole-graph Neuralese episodes from sft_sources neuralese-graph) trained in every step as a '
                        'ratio-free cross-entropy term of weight W (default 0.05), K units per step (default 8; a unit '
                        'is the chains of one task key, consumers with their pass-through producer documents); '
                        'repeatable; held tasks are dropped; think=train keeps a target\'s leading empty think block as '
                        'loss (default context: weight 0, ReplayCorpus)')
    t.add_argument('--memetic-tasks',
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
    t.add_argument('--lr', type=float, default=1e-4,
                   help='LoRA RL learning rate (lfm3opti.md §5 default 1e-4 for r=32, alpha=32)')
    t.add_argument('--lora-rank', type=int, default=32, help='LoRA rank on all 9 LFM3 projection families (default 32)')
    t.add_argument('--lora-alpha', type=float, default=None, help='LoRA alpha (default: equal to --lora-rank)')
    t.add_argument('--gate-rank', type=int, default=None, help='optional attention gate_proj rank (<= 16; default frozen)')
    t.add_argument('--no-trainable-rows', action='store_true', help='freeze Neuralese marker/override embedding rows')
    t.add_argument('--no-publication-sentinel', action='store_true',
                   help='skip the sentinel log-probability check after each actor update (WeightSync._sentinel)')
    t.add_argument('--fp8-base', action='store_true', help='FP8 frozen-base GEMMs (Blackwell; off by default: lfm3_learner, NATLANG_FP8_BASE=1 also opts in)')
    t.add_argument('--no-compile', action='store_true', help='disable torch.compile on fast layers')
    t.add_argument('--warmup', type=int, default=2)
    t.add_argument('--lr-reference-spread', type=int, default=12,
                   help='a step\'s LR is scaled by its groups with reward spread over this, at most 1 (lr_data_scale; '
                        'grpo-v2-run1 step 3 trained 2 such groups at full LR); 0: off')
    t.add_argument('--no-data-scaled-gradient', dest='data_scaled_gradient', action='store_false',
                   help='apply lr_data_scale to the LR only, not also to the gradient before the clip (AdamW moments then '
                        'carry a small step\'s full-size gradient into the next steps)')
    t.add_argument('--lr-reference-group-units', type=float, default=4096.,
                   help='for that LR scale a group with spread counts by the loss units it trains over this, at most 1 '
                        '(spread_data: tokens plus decision units; a group of 8 sampled decisions is ~8 units, so a step '
                        'of decision groups alone no longer takes a whole AdamW step, LoRA bench 2026-11-22); 0: each '
                        'group with spread counts 1')
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
    t.add_argument('--non-student-loss', choices=('full', 'nucleus'), default='full',
                   help='with --top-p < 1: targets the student never sampled (teacher demonstrations, the reasoning-RL '
                        'teacher actions, non-own replay-corpus rows; grpo.student_sampled_targets) train the full-softmax '
                        "log-probability (default), the student's own samples the nucleus one; nucleus: every target the "
                        'nucleus one (before 2026-11-18)')
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
                        'samples (sampled-token logp change), entropy, </think> hazard, and (v2) p(<|im_end|>) after short answers '
                        '(end_logp, end_logp_change, reclose_p); logged as "behaviour", flagged as behaviour_shift; "-": off')
    t.add_argument('--save-step-chains', type=int, default=4,
                   help="keep the exact learner batch of each of the last N steps in WORK/step-chains/step-N.pkl.zst (chains, "
                        'advantages, roles, row weights, actor logp, per-token loss factor; tens of MB a step) for offline '
                        'per-term diagnostics; 0: off')
    t.add_argument('--branch-states', type=int, default=0,
                   help='keep the learner state (LoRA factors, rows, AdamW, RNG, ledger; ~0.5 GB at r=32) of each of the '
                        'last N steps in CHECKPOINT_DIR/state-step-N/, so a control can --resume-state from any of them; 0: off')
    t.add_argument('--resume-state', help='with --resume: start from this learner-state directory (learner_manifest.json; '
                   'e.g. a --branch-states directory of another run) instead of STATE_DIR/learner-state')
    t.add_argument('--keep-checkpoints-until', type=int, default=0,
                   help='save a native checkpoint after every step up to this step and never prune those (probes, '
                        'grafting, held evals of early steps); 0: off')
    t.add_argument('--activation-gb', '--activation-gib', dest='activation_gb', type=float, default=22.,
                   help='saved activations (GiB) per packed batch; layers beyond are recomputed. 22 on the GB10 beside the actor: '
                        'full-FT 12 plus its freed masters and grads minus the second tmpfs weight file (DECISIONS 2026-11-22)')
    t.add_argument('--max-steps', type=int, default=0)
    t.add_argument('--save-every', type=int, default=2)
    t.add_argument('--state-every', type=int, default=1,
                   help='save the learner state (factors, rows, AdamW, RNG, consumed-group ledger) every N steps')
    t.add_argument('--state-dir', default=resolve_str('data_hdd', 'natlang-grpo-v1'))
    t.add_argument('--checkpoint-dir', help='pruned native checkpoints (default STATE_DIR/checkpoints). Native checkpoints '
                   'and FP32 states aimed at the external disk are written to the NVMe and moved in the background '
                   '(checkpoint_offload.staged_write; NATLANG_EXTERNAL_STAGING=0 writes them directly)')
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
    t.add_argument('--held-remote', choices=('pop', 'colab_g4'), help="measure held on a remote GPU (Pop or Colab G4) in parallel with training instead of "
                   'pausing collection on the training actor (pop_held_loop)')
    t.add_argument('--held-remote-gb', type=float, default=5.,
                   help="with --held-remote: ledger budget of each evaluation's own DGX unit (its collector processes "
                        'share node processes; grpo-v5 step 1 at 20 slots peaked at 3.6 GB, and 11 GB kept step 3 queued '
                        'beside the learner)')
    t.add_argument('--held-remote-inline', action='store_true',
                   help='with --held-remote: run each evaluation inside this process (before 2026-11-20; a restart kills '
                        'it) instead of its own ledger unit that survives restarts')
    t.add_argument('--no-pop-prefetch', dest='pop_prefetch', action='store_false',
                   help='with --held-remote pop: do not copy the next checkpoint to Pop while an evaluation runs')
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
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == 'train':
        args.top_p = check_top_p(args.top_p)
        from natlang_neuralese.common.cuda_context import ensure_context
        ensure_context()   # GB10: a context can fail to create under host-memory fragmentation; retried
        args.pool = args.pool or default_pools()
        args.held_manifest = args.held_manifest or default_held()
        args.memetic_tasks = args.memetic_tasks or default_memetic_tasks()
        # a split group (dynamic sampling) holds as few as 2 slots: room for slots / 2 groups in flight
        args.max_outstanding = args.max_outstanding or args.max_groups + -(-args.slots // (
            2 if args.dynamic_sampling_max_spread > 0 else args.group))
        train(args)


if __name__ == '__main__':
    main()
