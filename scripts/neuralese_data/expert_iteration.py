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
  cycle     the driver: pin -> Pop held of the base and the start policy (in the background) -> select -> collect (with
            filled-reasoning generation on the same actor) -> frontier -> teacher frontier campaign -> extract, replay,
            filled build, frontier build -> Ambient SFT (exp0_sft default) -> Pop held of the candidate -> verdict, each
            stage resumable (DIR/stages/<name>.done) and pinned (one commit for actor, harness, scorer, learner and eval).

    expert_iteration.py select   --out TASKS.json [--rates-groups GLOB ...] [--frontier OLD.json ...]
    expert_iteration.py collect  --work W --checkpoint C --tasks TASKS.json [--k 4] [--hours 6] [--slots 128]
    expert_iteration.py frontier --out F.json --collect W [--collect W2 ...] --policy SHA
    expert_iteration.py cycle    --dir DIR --start CKPT --commit SHA [...]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import signal
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


def log(record):
    print(json.dumps({'time': round(time.time(), 1), **record}, default=str), flush=True)


# ---- rates and tiers (pure, tested) ----------------------------------------------------------------------------------
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
            rates[row['task_id']][0] += row['reward'] == 1.0
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


# ---- collect ---------------------------------------------------------------------------------------------------------
def frontier_collector(base):
    class FrontierCollector(base):
        """grpo's Collector over a fixed task order: one group of ``--group`` samples per task, in order, no new group
        after the deadline; tasks this WORK already scored are skipped (resume)."""

        def setup(self, order, deadline):
            self.order = collections.deque(t for t in order if t in self.by_id and t not in self.own)
            self.deadline = deadline
            self._learnable()         # reads --pass-rates once: the dynamic-sampling prior

        def _pick(self):
            return self.by_id[self.order.popleft()]

        def _take(self):
            if self.paused.is_set():
                return None
            with self.lock:
                if (not self.pending and self.order and time.time() < self.deadline
                        and self.outstanding < self.args.max_outstanding):
                    self._new_group()
                return self.pending.popleft() if self.pending else None

        def finished(self):
            with self.lock:
                return not self.pending and self.outstanding == 0 and (not self.order or time.time() >= self.deadline)
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
    g = grpo.build_parser().parse_args([
        'train', '--work', str(work), '--checkpoint', args.checkpoint, '--checkpoint-sha256', sha,
        '--group', str(args.k), '--slots', str(args.slots), '--task-ids', args.tasks, '--actor-port', str(args.actor_port),
        '--dynamic-sampling-max-spread', str(args.dynamic_sampling_max_spread), '--held-every-hours', '0',
        '--no-actor-release', '--actor-overlap', 'on', '--memetic-share', '0', '--exclude-family-regex',
        args.exclude_family_regex, '--unit-prefix', args.unit_prefix, '--seed', str(args.seed),
        *(['--unpinned'] if args.unpinned else []), *[f'--actor-arg={a}' for a in args.actor_arg]])
    g.top_p = check_top_p(g.top_p)
    g.pool = args.pool or grpo.default_pools()
    g.held_manifest = args.held_manifest or grpo.default_held()
    g.max_outstanding = args.slots // 2 + 8
    tasks, excluded = grpo.load_tasks(g, work)
    code = grpo.git_pin()
    pin = grpo.check_pin(work, code['git_commit'], args.unpinned)
    from natlang_neuralese.rl.held_eval import series
    run = {'schema': 'natlang.ei-collect/1', 'started': time.time(), 'code': code, 'pin': pin, 'series': series(),
           'argv': sys.argv, 'checkpoint': args.checkpoint, 'checkpoint_sha256': sha, 'k': args.k, 'hours': args.hours,
           'tasks_file': args.tasks, 'tasks_sha256': sha256_file_hex_cached(Path(args.tasks)), 'eligible': len(tasks),
           'excluded_ids': excluded, 'provenance': code_provenance()}
    with (work / 'runs.jsonl').open('a') as handle:
        handle.write(json.dumps(run, default=str) + '\n')
    actor = None
    if args.actor:      # an actor already serving this policy (smokes, a shared engine): its attested sha must match
        url = args.actor.rstrip('/')
        wait_actor(url, sha, None, None, timeout=60)
        sglang = None
    else:
        command, url = grpo.actor_command(g, work, sha)
        with (work / 'actor.log').open('a') as handle:
            actor = subprocess.Popen(command, cwd=REPO, env=grpo.pinned_env(), stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        wait_actor(url, sha, actor, work / 'actor.log')
        sglang = url.rsplit(':', 1)[0] + ':' + str(args.actor_port + 1000)

    def stop_actor(*_):
        if actor is not None and actor.poll() is None:
            os.killpg(actor.pid, signal.SIGTERM)
    import atexit
    atexit.register(stop_actor)
    g.actor = url
    (work / 'actor.json').write_text(json.dumps({'url': url, 'sglang': sglang, 'checkpoint_sha256': sha,
                                                 'export': str(work / 'actor' / 'sglang-hf'),
                                                 'pid': actor.pid if actor else None}) + '\n')
    (work / 'sampling-config.json').write_text(json.dumps(rl_sampling_controls(g.top_p, capture=True)) + '\n')
    existing = [int(d.name[1:]) for d in (work / 'groups').glob('g*') if d.name[1:].isdigit()]
    collector = frontier_collector(grpo.Collector)(g, tasks, work, sha, REPO / 'ts-host',
                                                   first_group=max(existing, default=-1) + 1)
    collector.setup(order, time.time() + args.hours * 3600)
    log({'event': 'collect start', 'tasks': len(collector.order), 'already_scored': len(collector.own), 'k': args.k,
         'slots': args.slots, 'actor': url})
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: (stop.set(), collector.stop.set()))
    slots = [threading.Thread(target=collector.slot, args=(i,), daemon=True) for i in range(args.slots)]
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
               'stopped': stop.is_set()}
    (work / 'collect.json').write_text(json.dumps({**summary, 'run': run}, indent=1, default=str) + '\n')
    log(summary)
    if not stop.is_set():
        record_collection(work)


def record_collection(work):
    """The collection in the data registry (rollouts the SFT build and teacher campaign read)."""
    from natlang_neuralese.common import data_registry as dr
    dr.record_dataset(dr.dataset_id(work, 'grpo-rollouts'), work, kind='rollouts',
                      include=['groups/*/rollouts.jsonl', 'groups/*/rewards.jsonl', 'groups/*/score.json',
                               'groups/*/cases.jsonl', 'groups/*/versions.json', '*.json', 'runs.jsonl'],
                      license='project-generated (own LFM3 policy samples)',
                      format='expert_iteration collect WORK: groups/gNNNNNN/{rollouts,rewards}.jsonl (grpo group format)',
                      producer={'script': 'scripts/neuralese_data/expert_iteration.py collect'})


# ---- frontier --------------------------------------------------------------------------------------------------------
def frontier(args):
    from natlang_neuralese.rl.held_eval import exclusion_ids
    import grpo
    dirs, policies = [], set()
    for work in map(Path, args.collect):
        runs = [json.loads(l) for l in (work / 'runs.jsonl').open() if l.strip()]
        policies |= {r['checkpoint_sha256'] for r in runs}
        dirs += sorted(work.glob('groups/g*'))
    if policies != {args.policy}:
        raise SystemExit(f'collections served {sorted(policies)}, not the frontier policy {args.policy}')
    rates = group_outcomes(dirs)
    held = set(exclusion_ids(args.held_manifest or grpo.default_held(), [{'id': t} for t in rates]))
    leaked = sorted(set(rates) & held)
    if leaked:
        raise SystemExit(f'{len(leaked)} held tasks in the collections (e.g. {leaked[0]})')
    tasks = {t: {'successes': s, 'scored': n, 'tier': tier(s, n, args.solved, args.min_scored)}
             for t, (s, n) in sorted(rates.items())}
    data = {'schema': FRONTIER_SCHEMA, 'policy': args.policy, 'collections': [str(Path(c).resolve()) for c in args.collect],
            'thresholds': {'solved': args.solved, 'min_scored': args.min_scored},
            'tiers': dict(collections.Counter(v['tier'] for v in tasks.values())), 'tasks': tasks}
    Path(args.out).write_text(json.dumps(data, indent=0) + '\n')
    log({'event': 'frontier', 'tasks': len(tasks), 'tiers': data['tiers']})


# ---- cycle -----------------------------------------------------------------------------------------------------------
TEACHER_POOLS = [OPS / 'teacher-gen-20261115' / p for p in ('pool-v1', 'pool-v2-rich', 'pool-progtune-decisions')] + [
    OPS / 'teacher-gen-20261119' / 'gap-v1']
# the teacher campaign's teachers (DECISIONS 2026-11-21 teacher yield): Haiku retires itself while the Anthropic account
# has no credit; Luna at medium effort (low failed most reading families); GPT-6.1 Sol, the stronger model on the same
# Codex subscription, only on tasks another teacher failed (min_failed); family routing and --attempts in teacher_gen
TEACHERS = [
    {'model_id': 'claude-haiku-5-5', 'provider': 'anthropic', 'reasoning_effort': 'medium', 'slots': 4},
    {'model_id': 'gpt-6-luna', 'provider': 'openai-codex', 'reasoning_effort': 'medium', 'slots': 4},
    {'model_id': 'gpt-6.1-sol', 'provider': 'openai-codex', 'reasoning_effort': 'medium', 'slots': 2, 'min_failed': 1},
    {'model_id': 'nvidia/nemotron-3-ultra-550b-a55b', 'endpoint': 'https://integrate.api.nvidia.com/v1',
     'api_key_env': 'NVIDIA_API_KEY', 'slots': 1}]


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
        self.collect_dir, self.held = self.dir / 'collect', self.dir / 'held'
        self.work = Path(args.sft_work)
        self.mix = self.dir / 'mix'

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

    def wait(self, unit, poll=60):
        time.sleep(20)
        while self.active(unit):
            time.sleep(poll)

    def ledger(self, unit, budget, priority, command, log_path, detach=False, workdir=None):
        """``command`` (argv) as ledger unit ``unit``; waits for it unless ``detach``; a running unit is joined."""
        if not self.active(unit):
            shell = ' '.join(map(_quote, command)) + f' >> {_quote(str(log_path))} 2>&1'
            env = ['/usr/bin/env', *[f'{k}={self.env[k]}' for k in ('NATLANG_REPO', 'PYTHONPATH', 'HF_HUB_OFFLINE',
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

    def pop_held(self, name, checkpoint):
        """One held-v3 measurement on Pop (ops/grpo-v4/pop-held.sh: the pinned harness, 2 samples)."""
        out = self.held / name
        result = out / 'result.json'
        if result.exists() and json.loads(result.read_text()).get('complete'):
            return out
        mode = ['base', str(out), str(self.a.held_samples)] if checkpoint == BASE_CKPT else \
            ['ckpt', str(out), str(self.a.held_samples), str(checkpoint)]
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(['/bin/sh', str(OPS / 'grpo-v4' / 'pop-held.sh'), str(self.pin), *mode],
                       stdout=(out / 'driver.log').open('a'), stderr=subprocess.STDOUT)
        return out

    def popbase_stage(self):
        """Base and start policy on Pop, in the background (Pop is idle while the DGX collects and trains)."""
        unit = f'{self.a.name}-popbase'
        if not self.active(unit):
            subprocess.run(['systemd-run', '--user', '--unit', unit, '--collect', '--setenv', f'PATH={os.environ["PATH"]}',
                            *[f'--setenv={k}={self.env[k]}' for k in ('NATLANG_REPO', 'PYTHONPATH')],
                            *self.py('expert_iteration.py', 'pop-held', '--dir', self.dir, '--pin', self.pin,
                                     '--start', self.a.start, '--held-samples', self.a.held_samples)], check=True)
        self.note(stage='popbase', started=unit)

    def select_stage(self):
        groups = [str(OPS / 'grpo-v*' / '*' / 'groups' / 'g*'), str(OPS / 'ei' / '*' / 'collect' / 'groups' / 'g*')]
        self.ledger(f'{self.a.name}-select', 6, 'medium',
                    self.py('expert_iteration.py', 'select', '--out', self.dir / 'tasks.json',
                            *[x for g in groups for x in ('--rates-groups', g)], *self.a.select_args.split()),
                    self.dir / 'select.log')
        if not (self.dir / 'tasks.json').exists():
            raise SystemExit(f'select failed: {self.dir}/select.log')
        self.mark('select', **json.loads((self.dir / 'tasks.report.json').read_text()))

    def collect_stage(self):
        unit = f'{self.a.name}-collect'
        budget = self.a.kv_gb + 24 + round(self.a.slots * 0.3)
        self.ledger(unit, budget, 'high',
                    self.py('expert_iteration.py', 'collect', '--work', self.collect_dir, '--checkpoint', self.a.start,
                            '--checkpoint-sha256', self.start_sha, '--tasks', self.dir / 'tasks.json',
                            '--k', self.a.k, '--hours', self.a.collect_hours, '--slots', self.a.slots,
                            '--unit-prefix', unit, '--actor-arg=--sglang-kv-gb', f'--actor-arg={self.a.kv_gb}',
                            '--actor-arg=--sglang-max-running', f'--actor-arg={self.a.slots}'),
                    self.dir / 'collect.log', detach=True)
        if self.a.fill:
            self.fill_alongside(unit)
        self.wait(unit)
        summary = json.loads((self.collect_dir / 'collect.json').read_text())
        if summary.get('stopped'):
            raise SystemExit('collection stopped before its end: rerun the cycle to resume it')
        self.mark('collect', **{k: v for k, v in summary.items() if k != 'run'})

    def fill_alongside(self, collect_unit):
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
        if actor['checkpoint_sha256'] != self.start_sha:
            raise SystemExit('collect actor serves another policy')
        self.ledger(f'{self.a.name}-fill', 6, 'medium', self.fill_command(actor), self.dir / 'fill.log', detach=True)
        self.note(stage='fill', started=f'{self.a.name}-fill', actor=actor['url'])

    def fill_command(self, actor):
        """filled_reasoning.py generate on the collect actor: the seed WORKs then the grpo-sd-teacher role's (first
        WORK wins a key), teacher episodes of the tasks the student fails first (the select stage's rates: failed, then
        unmeasured, then by rate; the frontier mix trains teacher trajectories there), and a fixed share of the actor
        (``--fill-inflight`` draws in flight while collection keeps every slot busy, up to twice that when it does not;
        spare-slot-only draws would starve beside a 128-slot collection)."""
        from natlang_neuralese.common import data_registry
        works = list(dict.fromkeys([*map(str, self.a.seed_work), *data_registry.role_paths('grpo-sd-teacher')]))
        rates = self.dir / 'tasks.rates.json'
        return self.py('filled_reasoning.py', 'generate', '--out', self.a.fill, '--sglang', actor['sglang'],
                       '--front', actor['url'], '--render-tokenizer', self.work / 'tokenizer', '--export', actor['export'],
                       '--polite', '--max-running', self.a.slots, '--min-inflight', self.a.fill_inflight,
                       '--max-inflight', 2 * self.a.fill_inflight, '--workers', self.a.fill_inflight,
                       *[x for w in works for x in ('--work', w)],
                       *(('--prefer-rates', rates) if rates.exists() else ()))

    def stop_unit(self, unit):
        subprocess.run([*LEDGER, 'cancel', '--unit', unit])
        for suffix in ('-waiter', ''):
            subprocess.run(['systemctl', '--user', 'stop', f'{unit}{suffix}.service'], stderr=subprocess.DEVNULL)

    def frontier_stage(self):
        self.ledger(f'{self.a.name}-frontier', 4, 'medium',
                    self.py('expert_iteration.py', 'frontier', '--out', self.dir / 'frontier.json',
                            '--collect', self.collect_dir, '--policy', self.start_sha), self.dir / 'frontier.log')
        data = json.loads((self.dir / 'frontier.json').read_text())
        self.mark('frontier', tiers=data['tiers'])

    def teacher_stage(self):
        """The policy's new failures feed the teacher campaign (teacher_gen gap tiers: student_rare first, from this
        collection's outcomes; every earlier campaign covers its successes)."""
        if self.a.no_teacher_campaign:
            return self.mark('teacher', skipped=True)
        work = self.dir / 'teacher'
        work.mkdir(exist_ok=True)
        covered = [*TEACHER_POOLS, *sorted(OPS.glob('ei/*/teacher'))]
        rare = [t for t, v in json.loads((self.dir / 'frontier.json').read_text())['tasks'].items()
                if v['tier'] in ('never',) or v['successes'] / max(1, v['scored']) <= 0.25]
        if not rare:     # teacher_gen --limit 0 means every task
            return self.mark('teacher', frontier_tasks=0)
        command = [PY, str(self.pin / 'scripts/neuralese_data/teacher_gen.py'),
                   '--work', str(work), '--pause-below-available-gb', '25', '--exclude-family-regex', '(?!)',
                   *[x for c in covered if Path(c) != work and (Path(c) / 'manifest.jsonl').exists()
                     for x in ('--covered-by', str(c))],
                   '--student-groups', str(self.collect_dir / 'groups' / 'g*'), '--rare-rate', '0.25',
                   '--limit', str(len(rare)), '--attempts', '3', *[x for t in TEACHERS for x in ('--teacher', json.dumps(t))]]
        script = work / 'launch.sh'
        script.write_text('#!/bin/bash\n# expert-iteration teacher frontier (expert_iteration.py teacher_stage)\nset -u\n'
                          f'set -a; for f in nvidia gemini anthropic; do . {Path.home()}/.config/natlang/$f.env; done; set +a\n'
                          f'cd {self.pin}\nexec ' + ' '.join(map(_quote, command)) + '\n')
        script.chmod(0o755)
        # CPU/API-bound, ~4 GB at 20 slots: medium so medium-priority smokes and evals do not preempt it (each preemption
        # kills its running paid episodes); a high-priority training run still can
        self.ledger(f'{self.a.name}-teacher', 6, 'medium', ['/bin/bash', str(script)], work / 'launch.log', detach=True)
        self.mark('teacher', frontier_tasks=len(rare), work=str(work))

    def work_tokenizer(self):
        self.work.mkdir(parents=True, exist_ok=True)
        if not (self.work / 'tokenizer').exists():
            os.symlink(root('data_nvme') / 'sft-sources-20261118/tokenizer', self.work / 'tokenizer')

    def data_stage(self):
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
        name = f'episodes-{self.a.name}'
        if not (w / f'{name}-extract.json').exists():
            steps.append(self.py('sft_sources.py', 'extract', '--work', w, '--rl',
                                 f'{self.collect_dir}:{self.collect_dir / "rollouts"}', '--out-name', name))
        steps.append(self.py('sft_sources.py', 'replay', '--work', w, '--workers', self.a.replay_workers,
                             '--runtime-repo', self.pin))
        if self.a.fill and (Path(self.a.fill) / 'fills.jsonl').exists():
            steps.append(self.py('filled_reasoning.py', 'build', '--out', self.a.fill, '--corpus', self.dir / 'filled'))
        lines = [' '.join(map(_quote, s)) for s in steps]
        if not self.a.no_own_neuralese_graph:
            lines.append(' '.join(map(_quote, self.own_neuralese_graph_command()))
                         + " || echo 'own neuralese-graph failed: the build runs without it' >&2")
        script = self.dir / 'data.sh'
        script.write_text('set -e\n' + ''.join(line + '\n' for line in lines))
        self.ledger(f'{self.a.name}-data', 8, 'medium', ['/bin/sh', str(script)], self.dir / 'data.log')
        self.mark('data')

    def own_neuralese_graph_command(self):
        """The verified own successes that carry Neuralese state (every consumer call that read a native import) as
        whole-graph rows (``sft_sources neuralese-graph``). ``build`` skips these episodes (no text replay can carry
        the state), so without this part a cycle's mix holds no consumer call of the student's own: held-v3 after
        sft-v5 (DECISIONS 2026-11-21 "ran-out") showed the student calling Neuralese consumers twice as often while
        they loop ~7.6 calls each. ``--no-own-neuralese-graph`` opts out."""
        return self.py('sft_sources.py', 'neuralese-graph', '--work', self.work, '--out', self.dir / 'neuralese-graph-own',
                       '--exclude-run', 'grpo-v2/', '--exclude-run', 'grpo-v4/run1')

    def build_stage(self):
        aux = list(self.a.aux)
        if (self.dir / 'filled' / 'chains.pkl').exists():
            aux.append(f'filled={self.dir / "filled" / "chains.pkl"},share=all,task_tokens=40000,replaces=teacher')
        own_graph = self.dir / 'neuralese-graph-own' / 'chains.pkl'
        if not self.a.no_own_neuralese_graph and own_graph.exists():
            aux.append(f'own-neuralese-graph={own_graph},share={self.a.own_neuralese_graph_share},task_tokens=40000')
        command = self.py('sft_sources.py', 'build', '--work', self.work, '--out', self.mix, '--tokenizer',
                          self.work / 'tokenizer', '--frontier', self.dir / 'frontier.json', '--allow-stale-teacher',
                          '--october-share', '0', '--exclude-run', 'grpo-v2/', '--exclude-run', 'grpo-v4/run1',
                          '--teacher-forward-budget', self.a.teacher_budget, '--own-loss-share', self.a.own_share,
                          '--max-task-target-tokens', '40000', '--val-episodes', '40',
                          *[x for p in TEACHER_POOLS for x in ('--teacher-pool', p)],
                          *[x for a in aux for x in ('--aux', a)],
                          *[x for b in self.a.prefer_unseen for x in ('--prefer-unseen', b)])
        self.ledger(f'{self.a.name}-build', 24, 'medium', command, self.dir / 'build.log')
        sha = subprocess.check_output(['sha256sum', str(self.mix / 'chains.pkl')], text=True).split()[0]
        (self.mix / 'chains.sha256').write_text(sha + '\n')
        build = json.loads((self.mix / 'build.json').read_text())
        self.mark('build', chains_sha256=sha, episodes=build['episodes'], tasks=build['tasks'],
                  forward_tokens=build['forward_tokens'], parts=build['mix']['parts'])

    def sft_stage(self):
        run = self.dir / 'sft'
        unit = f'{self.a.name}-sft'
        if not (run / 'checkpoint.pt').exists() and not self.active(unit):
            sha = (self.mix / 'chains.sha256').read_text().split()[0]
            env = {**os.environ, 'CHAINS': str(self.mix / 'chains.pkl'), 'CHAINS_SHA256': sha, 'CKPT': self.a.start,
                   'SHA': self.start_sha, 'WORK': str(run), 'UNIT': unit, 'LR': self.a.lr, 'WARMUP': '8',
                   'BLT': '32000', 'ACT': '16', 'BUDGET': str(self.a.sft_budget)}
            vals = [v for v in ('val-own.pkl', 'val-teacher.pkl') if (self.mix / v).exists()]
            if vals:
                env['VAL'] = str(self.mix / vals[0])
            env['EXTRA'] = ' '.join([*(f'--val-chains {self.mix / v}' for v in vals[1:]),
                                     '--val-every 20' if vals else '', self.a.sft_extra])
            subprocess.run(['/bin/sh', str(OPS / 'grpo-v4' / 'launch-sft.sh'), str(self.pin)], env=env, check=True,
                           stdout=(self.dir / 'sft-launch.log').open('a'), stderr=subprocess.STDOUT)
        self.wait(unit, poll=120)
        if not (run / 'checkpoint.pt').exists():
            raise SystemExit(f'no SFT checkpoint: {run}/train.log')
        sha = subprocess.check_output(['sha256sum', str(run / 'checkpoint.pt')], text=True).split()[0]
        (run / 'sha256.txt').write_text(sha + '\n')
        self.mark('sft', checkpoint=str(run / 'checkpoint.pt'), sha256=sha)

    def held_stage(self):
        while self.active(f'{self.a.name}-popbase'):     # Pop serves one policy at a time
            time.sleep(300)
        candidate = self.pop_held('candidate', self.dir / 'sft' / 'checkpoint.pt')
        if not json.loads((candidate / 'result.json').read_text()).get('complete'):
            raise SystemExit('candidate held measurement incomplete: rerun the cycle to resume it')
        self.mark('held')

    def compare(self, a, b, out):
        subprocess.run([PY, '-m', 'natlang_neuralese.rl.held_eval', 'compare', str(a / 'result.json'),
                        str(b / 'result.json'), '--out', str(out)], env=self.env, check=True,
                       stdout=subprocess.DEVNULL)
        return json.loads(out.read_text())

    def verdict_stage(self):
        """The declared criteria (DECISIONS): vs the start policy measured on this pin, and vs the base."""
        base, start, candidate = self.held / 'base', self.held / ('base' if self.a.start == BASE_CKPT else 'start'), \
            self.held / 'candidate'
        vs_start = self.compare(start, candidate, self.dir / 'compare-vs-start.json')
        vs_base = self.compare(base, candidate, self.dir / 'compare-vs-base.json')
        d, p = vs_start['primary']['delta'], vs_start['primary']['permutation_p']
        ran_out = vs_start.get('ran_out_rate_delta') or 0
        if d < -self.a.fail_pts / 100 and p < 0.05:
            verdict = 'failure'
        elif d >= self.a.success_pts / 100 and ran_out <= 0.02:
            verdict = 'success'
        else:
            verdict = 'neutral'
        adopt = verdict == 'success' or (verdict == 'neutral' and d > 0 and ran_out <= 0.02)
        result = {'verdict': verdict, 'adopt': adopt, 'vs_start': {'delta': d, 'p': p, 'ran_out_delta': ran_out},
                  'vs_base': {'delta': vs_base['primary']['delta'], 'p': vs_base['primary']['permutation_p']},
                  'next_start': str(self.dir / 'sft' / 'checkpoint.pt') if adopt else self.a.start}
        (self.dir / 'verdict.json').write_text(json.dumps(result, indent=1) + '\n')
        if verdict == 'success' and vs_base['primary']['delta'] > 0:
            ckpt = self.dir / 'sft' / 'checkpoint.pt'
            (OPS / 'current-policy.json').write_text(json.dumps({
                'schema': 'natlang.current-policy-pointer/1',
                'scope': 'the best policy on held-v3 so far; read by the lfm-latest native source. An experiment that '
                         'meets its declared success criteria rewrites this file (DECISIONS records it).',
                'checkpoint': {'path': str(ckpt), 'sha256': (self.dir / 'sft' / 'sha256.txt').read_text().strip(),
                               'bytes': ckpt.stat().st_size},
                'label': self.a.name, 'why': f'{self.a.name}: {json.dumps(result)}', 'updated': time.time()},
                indent=1) + '\n')
        self.mark('verdict', **result)

    def run(self):
        self.note(stage='cycle', start=self.a.start, start_sha=self.start_sha, commit=self.a.commit)
        stages = [('pin', self.pin_stage), ('popbase', self.popbase_stage), ('select', self.select_stage),
                  ('collect', self.collect_stage), ('frontier', self.frontier_stage), ('teacher', self.teacher_stage),
                  ('data', self.data_stage), ('build', self.build_stage), ('sft', self.sft_stage),
                  ('held', self.held_stage), ('verdict', self.verdict_stage)]
        for name, stage in stages:
            if self.done(name):
                continue
            if name == 'data' and self.a.fill:
                self.stop_unit(f'{self.a.name}-fill')
            stage()


def _quote(text):
    import shlex
    return shlex.quote(str(text))


def pop_held(args):
    """Background unit of ``cycle``: the base, then (if it is not the base) the start policy, on Pop."""
    a = argparse.Namespace(dir=args.dir, pin=args.pin, start=args.start, start_sha=BASE_SHA, commit='x' * 40,
                           held_samples=args.held_samples, sft_work='/dev/null')
    c = Cycle(a)
    c.pin = Path(args.pin)
    c.pop_held('base', BASE_CKPT)
    if args.start != BASE_CKPT:
        c.pop_held('start', args.start)


def cycle(args):
    Cycle(args).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def pool_flags(p):
        p.add_argument('--pool', action='append', help='default: the registry role grpo-pool')
        p.add_argument('--held-manifest', help='default: grpo.default_held() (held-v3)')
        p.add_argument('--exclude-family-regex', default='(?!)', help="families left out (default: none, as grpo-v5)")
        p.add_argument('--seed', type=int, default=20261120)
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
    c.add_argument('--actor', help='use this running actor front (its attested policy must be --checkpoint) instead '
                   'of launching one')
    c.add_argument('--actor-arg', action='append', default=[])
    c.add_argument('--unit-prefix', default='ei-collect')
    c.add_argument('--unpinned', action='store_true')
    f = sub.add_parser('frontier')
    f.add_argument('--out', required=True)
    f.add_argument('--collect', action='append', required=True)
    f.add_argument('--policy', required=True, help='sha256 of the policy every collection served')
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
    y.add_argument('--select-args', default='', help='more select flags (e.g. "--limit 3000")')
    y.add_argument('--k', type=int, default=4)
    y.add_argument('--collect-hours', type=float, default=6.)
    y.add_argument('--slots', type=int, default=128)
    y.add_argument('--kv-gb', type=int, default=12)
    y.add_argument('--fill', help='filled_reasoning.py OUT directory: generate on the collect actor, build into the mix')
    y.add_argument('--fill-inflight', type=int, default=16,
                   help='filled-reasoning draws kept in flight on the collect actor (twice that when it has spare slots)')
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
    y.add_argument('--lr', default='5e-6')
    y.add_argument('--sft-budget', type=int, default=80)
    y.add_argument('--sft-extra', default='')
    y.add_argument('--held-samples', type=int, default=2)
    y.add_argument('--success-pts', type=float, default=2.)
    y.add_argument('--fail-pts', type=float, default=3.)
    h = sub.add_parser('pop-held')
    h.add_argument('--dir', required=True)
    h.add_argument('--pin', required=True)
    h.add_argument('--start', required=True)
    h.add_argument('--held-samples', type=int, default=2)
    args = parser.parse_args()
    {'select': select, 'collect': collect, 'frontier': frontier, 'cycle': cycle, 'pop-held': pop_held}[args.command](args)


if __name__ == '__main__':
    main()
