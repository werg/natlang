#!/usr/bin/env python3
"""Shared memory ledger for jobs on the DGX's unified memory.

On the GB10 CUDA allocations come from the same 121 GB as everything else, systemd's MemoryMax does not see
them, and the kernel's OOM killer ranks processes by resident pages, which leave CUDA memory out. A single
experiment can therefore take the teacher campaign or the vLLM server down with it. This tool makes every
heavy job declare a budget and enforces it from outside the job:

  run      admit a job against the ledger and launch it as a user unit: kill priority by class
           (experiment/collection/service), MemoryMax on host memory, NATLANG_CUDA_MEMORY_GB for the job's own
           CUDA cap. Refuses (or with --wait, waits) while the start would leave less than the reserve free.
  adopt    register a unit started before the ledger and raise its kill priority.
  release-cache  drop clean page cache of large files under the data roots. On the GB10 a CUDA allocation can only
           use MemFree: page cache is counted as available but is not reclaimed for the GPU, so stale cache of
           read-once checkpoints and corpora makes model loads fail with "CUDA error: out of memory" while
           MemAvailable looks ample. `run` and `guard` call it when MemFree is short.
  status   claims with measured use (cgroup memory plus each process's CUDA memory from nvidia-smi).
  guard    loop: under memory pressure (available below twice the floor) stop the unit most over its budget,
           and below the floor the lowest-priority admitted unit; without pressure an over-budget unit keeps
           running and its budget is raised to what it uses. Admission learns each job family's measured peak. Only units admitted through this ledger are ever stopped.

OVERSUBSCRIPTION AND PRIORITIES (owner 2026-10-11). Admission is by real headroom: MemAvailable (which already excludes what
running jobs use) minus what running claims still reserve, minus the job's budget, must leave the reserve. The sum of
claims may exceed the machine. A running claim reserves its unspent budget in full while it ramps up (RAMP_SECONDS) or
when it is fixed (`--fixed-claim`, alias `--hold-budget`: late spikes such as the first eval or a checkpoint write),
half of it for a GPU unit (CUDA bursts can reach the floor before the guard reacts; CUDA use is nvidia-smi per pid and
counts in the unit's `used`, with its cgroup and attached docker container), a quarter for a CPU unit.
Priority (`--priority high|medium|low`; default medium, services high): training and GPU end-to-end runs high, smokes
and evals medium, data generation, rendering, check-main and other CPU jobs low. It sets (1) the order of the shared
waiting queue (higher first, then earlier; only the head is admitted), (2) the kernel OOM score of the unit (high 200,
medium 600, low 900: user units cannot go below the user manager's own 100, and with no systemd-oomd the kernel is the
instant safety net), (3) the guard's victim order under pressure (lowest priority first; a unit over its own budget
goes first), and (4) preemption: when the head of the queue does not fit, the guard gracefully stops the lowest-priority
resumable units until it does. Units are resumable by default (tests, checks and data jobs are rerunnable; `--no-resumable` only for long non-checkpointed jobs; services and adopted units are protected); a
stopped resumable unit (preemption or pressure, not an over-budget stop) re-enters the queue with the spec it was
launched from and the guard relaunches it when room returns. `status` lists claims with priority, budget, use and
reservation, and the queue with the units that would be preempted for each entry and the non-preemptible units blocking it (`blocked_by`). A stopped unit's docker containers (named by the command, `run --container`, a `docker run --name` client in its cgroup, or the unit name) are stopped by name too, and a requeued unit is not relaunched while one still runs. The guard's kill threshold is still
the original budget.
"""
import argparse
import contextlib
import shutil
import fcntl
import mmap
import json
import re
import os
import shlex
import subprocess
import sys
import threading
import time

STATE = os.path.expanduser(os.environ.get('NATLANG_MEMORY_LEDGER', '~/.local/state/natlang/memory-ledger.json'))
# Higher is killed first by the kernel. Unprivileged units can only raise their score, so protection of the
# campaign and the vLLM server (score 0) comes from every admitted job ranking above them.
CLASSES = {'experiment': 900, 'collection': 600, 'service': 300}
GIB = 2**30
# Oversubscription (owner 2026-10-11): admission is by real headroom (MemAvailable already excludes what running jobs use),
# not by the sum of claims. A running claim reserves only part of its unspent budget: all of it while ramping up or when
# fixed (``--fixed-claim``), half for a GPU unit (CUDA allocations come in bursts and can reach the floor before the guard
# reacts), a quarter for a CPU unit. Priorities decide who waits, who is preempted and who the guard stops first.
RAMP_SECONDS = 300
RESERVE_FRACTION = {'gpu': 0.5, 'cpu': 0.25}
PRIORITIES = {'low': 1, 'medium': 2, 'high': 3}
PRIORITY_NAMES = {v: k for k, v in PRIORITIES.items()}
# Kernel OOM score adjust by priority: the safety net when memory really runs out faster than the guard polls.
PRIORITY_OOM = {'high': 200, 'medium': 600, 'low': 900}
DEFAULT_PRIORITY = {'experiment': 'medium', 'collection': 'medium', 'service': 'high'}
WAITER_TTL = 90  # a waiting `run` refreshes its queue entry every 30 s; older entries are abandoned
REQUEUE_DELAY = 120  # seconds after a preemption/pressure stop before the unit may be relaunched
# A guard stop gives the victim its declared grace (``run --stop-seconds``) to write its full checkpoint (latents and
# optimizer: ~4 min for Mellum's 47 GB state), and kills at once only below this emergency level (under the floor).
EMERGENCY_GB = 3
# Stop grace scales with the job (owner 2026-10-10: "the guard should be different for smaller models"): the bytes
# its family last checkpointed (the shared writer logs every write, train/trajectory_state.py) over the measured write
# throughput, times a margin, plus a few seconds to reach a step boundary. A family that never checkpointed (tests,
# probes, no state) gets the minimum. An explicit ``run --stop-seconds`` overrides.
CHECKPOINT_WRITES = os.path.expanduser(os.environ.get('NATLANG_CHECKPOINT_WRITES',
                                                      '~/.local/state/natlang/checkpoint-writes.jsonl'))
GRACE_MIN, GRACE_MARGIN, GRACE_SLACK = 5.0, 2.0, 15.0
DEFAULT_THROUGHPUT = 150e6  # bytes/s when nothing was measured yet (v3: 46.8 GB in ~250 s, ~190 MB/s)


def mem_available():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing from /proc/meminfo')


def mem_total():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('MemTotal missing from /proc/meminfo')


def mem_free():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('MemFree:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('MemFree missing from /proc/meminfo')


CACHE_ROOTS = ['~/natlang', '~/data', '~/natlang-data-nvme', '~/.cache', '/mnt/external/natlang-development-data',
               '/mnt/external/bgkit-data/models']


def reclaimable():
    """Memory the kernel counts as available but not free: roughly the clean page cache a release can drop."""
    return mem_available() - mem_free()


def release_cache(roots=None, min_bytes=16 << 20):
    """posix_fadvise(DONTNEED) on every file of at least ``min_bytes`` under the roots: drops only clean cached
    pages (dirty pages and mapped pages in use stay), so running jobs at worst re-read from disk. One walk at a
    time: the roots include a slow disk with millions of small files, and concurrent walks only thrash it."""
    roots = roots or [r for r in os.environ.get('NATLANG_CACHE_ROOTS', '').split(':') if r] or CACHE_ROOTS
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    lock = open(STATE + '.release.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return {'skipped': 'another cache release is running'}
    before, files = mem_free(), 0
    for root in roots:
        for directory, subdirs, names in os.walk(os.path.expanduser(root)):
            # Hidden directories (.git, sync histories) and package trees hold many small files and no data a job reads.
            subdirs[:] = [d for d in subdirs if not d.startswith('.') and d not in ('node_modules', 'proc')]
            for name in names:
                path = os.path.join(directory, name)
                try:
                    if os.path.islink(path) or os.path.getsize(path) < min_bytes:
                        continue
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        files += 1
                    finally:
                        os.close(fd)
                except OSError:
                    pass
    walked = mem_free()
    ballooned = balloon_reclaim()
    lock.close()
    return {'files': files, 'free_before_gb': round(before / GIB, 1), 'free_after_walk_gb': round(walked / GIB, 1),
            'ballooned_gb': ballooned, 'free_after_gb': round(mem_free() / GIB, 1)}


def inactive_file():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('Inactive(file):'):
                return int(line.split()[1]) * 1024
    return 0


def balloon_reclaim(keep_available=16 * GIB, limit=24 * GIB):
    """Evict page cache the walk cannot reach (files outside the roots, container layers): touch anonymous memory
    one GiB at a time, which makes the kernel reclaim inactive file pages, then free it. CUDA on the GB10 never
    triggers that reclaim itself. Bounded by the inactive file cache, and stops while MemAvailable stays above
    ``keep_available`` (twice the guard's default floor). Returns the GiB touched."""
    size = min(limit, inactive_file() - 2 * GIB, mem_available() - keep_available)
    chunks = []
    try:
        while len(chunks) * GIB < size and mem_available() > keep_available:
            chunk = mmap.mmap(-1, GIB)
            for offset in range(0, GIB, mmap.PAGESIZE):
                chunk[offset] = 1
            chunks.append(chunk)
    finally:
        for chunk in chunks:
            chunk.close()
    return len(chunks)


def gpu_usage():
    """CUDA memory per pid from nvidia-smi; empty when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    usage = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            usage[int(parts[0])] = int(parts[1]) * 2**20
    return usage


def unit_state(unit):
    out = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'ControlGroup'],
                         capture_output=True, text=True).stdout
    fields = dict(line.split('=', 1) for line in out.splitlines() if '=' in line)
    return fields.get('ActiveState', 'inactive'), fields.get('ControlGroup', '')


def cgroup_usage(root, gpu):
    """Host memory charged to a cgroup plus CUDA memory of its processes; None when it is gone."""
    try:
        host = int(open(os.path.join(root, 'memory.current')).read())
        pids = [int(p) for p in open(os.path.join(root, 'cgroup.procs')).read().split()]
    except OSError:
        return None
    # Page cache charged to the cgroup is reclaimable, not the job's working set: a job writing a 47 GB checkpoint
    # otherwise reads 10+ GB over its real use. Shared memory (also counted as file) stays.
    try:
        stat = dict(line.split() for line in open(os.path.join(root, 'memory.stat')))
        host -= max(0, int(stat.get('file', 0)) - int(stat.get('shmem', 0)))
    except (OSError, ValueError):
        pass
    # Anonymous memory a process freed with MADV_FREE (torch's mimalloc CPU allocator on the GB10 build) stays charged
    # as anon until the kernel reclaims it, but it is reclaimable like page cache: a 12B model's freed host staging
    # read ~22 GB over the job's working set (plans/mellum-port.md "Warm-up step profile").
    host -= sum(lazyfree_bytes(pid) for pid in pids)
    return max(0, host) + sum(gpu.get(pid, 0) for pid in pids)


PROC = '/proc'


def lazyfree_bytes(pid):
    """``LazyFree`` of a process (``smaps_rollup``): freed anonymous pages the kernel may drop without writeback; 0
    when unreadable."""
    try:
        with open(os.path.join(PROC, str(pid), 'smaps_rollup')) as f:
            for line in f:
                if line.startswith('LazyFree:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def container_cgroup(command, unit=None):
    """The cgroup of the container a unit runs: its processes are Docker's, not the unit's, so the unit's own cgroup
    shows only the docker client. The container is the one its command names (``docker start -a NAME``,
    ``docker run --name NAME``), else one named like the unit (a wrapper script that runs ``docker run``)."""
    name = container_name(command) or (unit.removesuffix('.service') if unit else None)
    if not name:
        return None
    out = subprocess.run(['docker', 'inspect', '-f', '{{.Id}}', name],
                         capture_output=True, text=True).stdout.strip()
    return f'/sys/fs/cgroup/system.slice/docker-{out}.scope' if out else None


def unit_usage(unit, gpu, command=None):
    """Host memory charged to the unit's cgroup plus CUDA memory of its processes (and of its attached container)."""
    active, cgroup = unit_state(unit)
    if active not in ('active', 'activating', 'reloading') or not cgroup:
        return None
    used = cgroup_usage('/sys/fs/cgroup' + cgroup, gpu)
    if used is None:
        return None
    container = container_cgroup(command, unit)
    return used + ((cgroup_usage(container, gpu) or 0) if container else 0)


@contextlib.contextmanager
def ledger():
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.load(open(STATE)) if os.path.exists(STATE) else {'schema': 'natlang.memory-ledger/1', 'claims': {}, 'events': []}
        yield state
        state['events'] = state['events'][-500:]
        tmp = STATE + '.tmp'
        json.dump(state, open(tmp, 'w'), indent=2)
        os.replace(tmp, STATE)


def container_name(command):
    """The container a unit's command attaches to or runs (``docker start -a NAME`` or ``docker run --name NAME``):
    stopping the unit only ends the docker client, so the container must be stopped by name."""
    text = ' '.join(command) if isinstance(command, list) else str(command or '')
    found = re.search(r'docker start (?:-a|--attach) (\S+)', text) or \
        re.search(r'docker run\b.*?--name[ =](\S+)', text)
    return found.group(1).strip("'\"") if found else None


def family(unit):
    """A unit's job family: its name without trailing time/hash suffixes (natlang-foo-test-210556 → natlang-foo-test)."""
    return re.sub(r'(-[0-9a-f]{4,})+$', '', unit.removesuffix('.service'))


def learned_budget(state, unit, budget):
    """The admission budget: at least the family's measured peak plus 10% (guessed budgets were the main cause of
    over-budget stops and of refused admissions)."""
    peak = state.get('family_peaks', {}).get(family(unit), 0)
    return max(budget, min(int(peak * 1.1), mem_total() - 2 * 8 * GIB))


def unit_cuda(unit, gpu, command=None):
    """CUDA memory held by the unit's processes and its attached container (from nvidia-smi's per-pid list)."""
    active, cgroup = unit_state(unit)
    if active not in ('active', 'activating', 'reloading') or not cgroup:
        return 0
    roots = ['/sys/fs/cgroup' + cgroup]
    container = container_cgroup(command, unit)
    if container:
        roots.append(container)
    total = 0
    for root in roots:
        with contextlib.suppress(OSError, ValueError):
            total += sum(gpu.get(int(p), 0) for p in open(os.path.join(root, 'cgroup.procs')).read().split())
    return total


def live_claims(state, gpu):
    """Claims whose unit still runs, with measured use and the highest use measured so far; claims of finished
    units are released."""
    live = {}
    for unit, claim in list(state['claims'].items()):
        used = unit_usage(unit, gpu, claim.get('command'))
        if used is None and time.time() - claim['admitted'] > 30:  # give systemd a moment to start the unit
            if claim.get('peak'):  # the family's measured peak sizes its next admission
                peaks = state.setdefault('family_peaks', {})
                peaks[family(unit)] = max(peaks.get(family(unit), 0), claim['peak'])
            del state['claims'][unit]
            state['events'].append({'time': time.time(), 'event': 'released', 'unit': unit})
            continue
        if gpu and not claim.get('gpu') and unit_cuda(unit, gpu, claim.get('command')):
            claim['gpu'] = True
        claim['peak'] = max(claim.get('peak', 0), used or 0)
        claim.setdefault('peak_since', time.time())
        live[unit] = dict(claim, used=used or 0)
    return live


def fixed_claim(claim):
    """A claim that always reserves its whole unspent budget (``--fixed-claim``, older name ``--hold-budget``)."""
    return bool(claim.get('hold_budget') or claim.get('fixed_claim'))


def priority_of(claim):
    """Numeric priority (1 low, 2 medium, 3 high) from ``run --priority``, else the class default."""
    return PRIORITIES[claim.get('priority') or DEFAULT_PRIORITY.get(claim.get('class'), 'medium')]


def resumable(claim):
    """Whether the guard may stop the unit for a higher priority and relaunch it later: declared at ``run``, default
    true for experiments and collections (their commands resume from checkpoints), false for services."""
    declared = claim.get('resumable')
    if declared is None:  # undeclared: tests, checks and jobs the ledger launched are rerunnable; services and adopted units are not
        return claim.get('class') != 'service' and not claim.get('adopted')
    return bool(declared)  # without a launch spec it is stopped and logged, and its owner restarts it


def outstanding(claim, now=None):
    """What a running claim still reserves beyond its use: the unspent budget times its reserve fraction (whole while
    ramping up or fixed)."""
    now = now or time.time()
    unspent = max(0, claim['budget'] - claim['used'])
    if fixed_claim(claim) or now - claim['admitted'] < RAMP_SECONDS:
        return unspent
    return int(unspent * RESERVE_FRACTION['gpu' if claim.get('gpu') else 'cpu'])


def headroom(state, gpu):
    live = live_claims(state, gpu)
    return mem_available() - sum(outstanding(c) for c in live.values()), live


def blocked_by(live, wanted=None, count=3):
    """Names of what keeps a waiting job out, for a refusal message: with ``wanted`` the units it cannot preempt, else
    the claims reserving the most memory."""
    if wanted:
        rows = non_preemptible(live, wanted, count)
        return ' (not preemptible: ' + ', '.join(f'{r["unit"]} {r["held_gb"]} GiB, {r["reason"]}' for r in rows) + ')' \
            if rows else ''
    top = [(u, c) for u, c in sorted(live.items(), key=lambda item: -outstanding(item[1]))[:count]
           if outstanding(c) >= GIB]
    return ' (blocked by ' + ', '.join(f'{u} {outstanding(c) / GIB:.0f} GiB reserved of {c["budget"] / GIB:.0f} '
                                        f'claimed' for u, c in top) + ')' if top else ''


def waiting_order(state, now=None):
    """The queue of jobs waiting for room, head first: higher priority first, then the earlier request. Entries of
    waiting ``run`` calls expire when not refreshed; requeued units (preempted or stopped under pressure) stay."""
    now = now or time.time()
    entries = [(u, w) for u, w in state.get('waiting', {}).items()
               if w.get('requeued') or now - w.get('seen', 0) < WAITER_TTL]
    return sorted(entries, key=lambda item: (-PRIORITIES[item[1]['priority']], item[1]['since'], item[0]))


def fits(free, budget, reserve):
    """Real-headroom admission: free is MemAvailable minus what running claims still reserve."""
    return free - budget >= reserve


def preemption_plan(live, free, wanted, graces=None):
    """The running units to stop (in order) so a waiting job of ``wanted`` = {priority, budget, reserve} fits: only
    resumable units of strictly lower priority, the lowest priority first, within it the one freeing the most per
    second of stop grace. Empty when it already fits or when stopping every candidate would not make it fit."""
    need = wanted['budget'] + wanted['reserve'] - free
    if need <= 0:
        return []
    graces = graces or {}
    candidates = sorted((u for u, c in live.items() if resumable(c) and priority_of(c) < PRIORITIES[wanted['priority']]),
                        key=lambda u: (priority_of(live[u]), -(live[u]['used'] / (graces.get(u, GRACE_MIN) + 1)),
                                       -live[u]['admitted']))
    plan, freed = [], 0
    for unit in candidates:
        plan.append(unit)
        freed += live[unit]['used'] + outstanding(live[unit])
        if freed >= need:
            return plan
    return []


def non_preemptible(live, wanted, count=4):
    """Running units a waiting job cannot preempt (equal or higher priority, or not resumable) that hold memory it needs,
    biggest first, with the reason: shown in ``status`` and in a refusal so the owner sees what blocks the head."""
    rows = []
    for unit, c in live.items():
        held = c['used'] + outstanding(c)
        if held < GIB:
            continue
        if not resumable(c):
            rows.append((held, unit, 'not resumable'))
        elif priority_of(c) >= PRIORITIES[wanted['priority']]:
            rows.append((held, unit, f'priority {PRIORITY_NAMES[priority_of(c)]} is not below {wanted["priority"]}'))
    return [{'unit': u, 'held_gb': round(h / GIB, 1), 'reason': why} for h, u, why in sorted(rows, reverse=True)[:count]]


def requeue_spec(state, unit, now=None, containers=()):
    """Put a stopped resumable unit back in the waiting queue with the spec it was launched from."""
    claim = state['claims'].get(unit, {})
    now = now or time.time()
    if not resumable(claim):
        return False
    if not claim.get('spec'):
        state['events'].append({'time': now, 'event': 'stopped without launch spec; owner must restart', 'unit': unit})
        return False
    state.setdefault('waiting', {})[unit] = {'priority': PRIORITY_NAMES[priority_of(claim)], 'since': now,
                                             'requeued': True, 'not_before': now + REQUEUE_DELAY,
                                             'budget': claim['budget'], 'reserve': claim['spec'].get('reserve', 8 * GIB),
                                             'spec': claim['spec']}
    if containers:
        state['waiting'][unit]['containers'] = list(containers)
    state['events'].append({'time': now, 'event': 'requeued', 'unit': unit})
    return True


def claim_for(spec, now=None):
    return {'budget': spec['budget'], 'class': spec['cls'], 'admitted': now or time.time(),
            'command': spec['command'], 'host_max': spec.get('host_max_gb'), 'priority': spec['priority'],
            'resumable': spec['resumable'], 'spec': spec,
            **({'stop_seconds': spec['stop_seconds'], 'stop_seconds_explicit': True}
               if spec.get('stop_seconds') is not None else {}),
            **({'hold_budget': True} if spec.get('hold_budget') else {})}


def launch(spec):
    """Start the admitted job as a user unit. Its kernel OOM score follows its priority (user units cannot go below
    the user manager's own score, 100 on the DGX), so when memory really runs out the kernel takes low priority first."""
    unit, budget = spec['unit'], spec['budget']
    host_max = spec.get('host_max_gb') or budget / GIB
    subprocess.run(['systemctl', '--user', 'reset-failed', unit], capture_output=True)
    command = ['systemd-run', '--user', '--unit', unit, '-p', f'MemoryMax={int(host_max * GIB)}',
               '-p', 'MemorySwapMax=0', '-p', f'OOMScoreAdjust={PRIORITY_OOM[spec["priority"]]}',
               '-p', f'OOMPolicy={spec["oom_policy"]}',
               '--working-directory', os.path.abspath(spec['workdir']),
               '-E', f'NATLANG_CUDA_MEMORY_GB={round(budget / GIB, 1)}', '-E', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True']
    # The user manager's PATH lacks ~/.local/bin, so units got /usr/bin/node v18 (no crypto.randomUUID; ts-host scripts
    # need Node >= 22) while shells get v24 (2026-10-11). Units see the shell's toolchain (scripts/natlang_env.sh).
    if not any(env.startswith('PATH=') for env in spec['env']):
        command += ['-E', f"PATH={os.path.expanduser('~/.local/bin')}:{os.environ.get('PATH', '/usr/bin:/bin')}"]
    for env in spec['env']:
        command += ['-E', env]
    # A stopped unit takes its container with it (an orphaned one held 36 GB outside the ledger): the one its command
    # names, else one named like the unit (wrapper scripts that `docker run --name <unit>`; '-' ignores none).
    container = container_name(spec['command']) or unit.removesuffix('.service')
    # --stop-seconds: a planned stop (systemctl stop) waits this long for the job to checkpoint before its container
    # is killed (a 47 GB latent+optimizer checkpoint takes minutes). The guard's pressure stops stay short.
    if spec.get('stop_seconds') is not None:
        stop_seconds = spec['stop_seconds']
    else:
        writes = checkpoint_writes()
        stop_seconds = max(30.0, grace_seconds(unit, {}, family_state_bytes(writes), write_throughput(writes)))
    command += ['-E', f'NATLANG_LEDGER_UNIT={unit}', '-p', f'TimeoutStopSec={int(stop_seconds) + 30}',
                '-p', f'ExecStopPost=-/usr/bin/docker stop -t {int(stop_seconds)} ' + ' '.join(
                    dict.fromkeys([container, *spec.get('containers', ())]))]
    return subprocess.run(command + spec['command']).returncode


def run(args):
    budget = int(args.budget_gb * GIB)
    reserve = int(args.reserve_gb * GIB)
    unit = args.unit if args.unit.endswith('.service') else args.unit + '.service'
    priority = args.priority or DEFAULT_PRIORITY[args.cls]
    # Before admission: a claim whose unit has not started within 30 s is released by the next ledger call, and a
    # cache walk takes minutes. Walk only when there is cache to drop; skip it when another walk is running.
    # In a thread: the walk takes minutes and the waiter must stay in the queue (refreshed every loop) meanwhile, or the
    # guard cannot see it and preempts nothing.
    if mem_free() < budget + reserve and reclaimable() >= GIB:
        threading.Thread(target=lambda: print(json.dumps({'released_cache': release_cache()}), file=sys.stderr),
                         daemon=True).start()
    deadline = time.time() + args.wait
    since = time.time()
    while True:
        with ledger() as state:
            free, live = headroom(state, gpu_usage())
            if unit in live:
                raise SystemExit(f'{unit} already holds a claim')
            learned = learned_budget(state, unit, budget)
            if learned > budget:
                print(json.dumps({'budget_raised_to_measured_peak_gb': round(learned / GIB, 1),
                                  'requested_gb': args.budget_gb, 'family': family(unit)}), file=sys.stderr)
                budget = learned
            waiting = state.setdefault('waiting', {})
            now = time.time()
            waiting[unit] = {'priority': priority, 'since': waiting.get(unit, {}).get('since', since), 'seen': now,
                             'budget': budget, 'reserve': reserve}
            head = waiting_order(state, now)[0][0]
            # Head of the priority queue only: a lower job does not slip in ahead of a higher one that waits for room
            # (the guard preempts lower units for the head when real headroom is short).
            if head == unit and fits(free, budget, reserve):
                spec = {'unit': unit, 'budget': budget, 'host_max_gb': args.host_max_gb, 'cls': args.cls,
                        'oom_policy': args.oom_policy, 'workdir': os.path.abspath(args.workdir), 'env': args.env,
                        'stop_seconds': args.stop_seconds, 'command': args.command, 'priority': priority,
                        'resumable': args.resumable if args.resumable is not None else args.cls != 'service',
                        'hold_budget': args.hold_budget, 'reserve': reserve, 'containers': args.container}
                state['claims'][unit] = claim_for(spec)
                waiting.pop(unit, None)
                state['events'].append({'time': now, 'event': 'admitted', 'unit': unit, 'budget': budget,
                                        'free': free, 'priority': priority})
                break
            blockers = blocked_by(live, {'priority': priority}) + ('' if head == unit else f' (queue head: {head})')
        if time.time() >= deadline:
            with ledger() as state:
                state.get('waiting', {}).pop(unit, None)
            raise SystemExit(f'not admitted: {free / GIB:.1f} GiB free after reservations{blockers}, '
                             f'{args.budget_gb} GiB requested, {args.reserve_gb} GiB reserve')
        time.sleep(30)
    if launch(spec):
        with ledger() as state:
            state['claims'].pop(unit, None)
        raise SystemExit(1)
    print(json.dumps({'unit': unit, 'budget_gb': round(budget / GIB, 1), 'class': args.cls, 'priority': priority,
                      'free_gb': round(free / GIB, 1)}))


def main_command(unit):
    """argv of the unit's main process, or None."""
    pid = subprocess.run(['systemctl', '--user', 'show', '-p', 'MainPID', '--value', unit],
                         capture_output=True, text=True).stdout.strip()
    with contextlib.suppress(OSError, ValueError):
        argv = open(f'/proc/{int(pid)}/cmdline', 'rb').read().split(b'\0')
        return [a.decode() for a in argv if a] or None
    return None


def adopt(args):
    """Register a unit that is already running (started before the ledger) and raise its kill priority."""
    unit = args.unit if args.unit.endswith('.service') else args.unit + '.service'
    active, cgroup = unit_state(unit)
    if active != 'active':
        raise SystemExit(f'{unit} is not running')
    for pid in open('/sys/fs/cgroup' + cgroup + '/cgroup.procs').read().split():
        with contextlib.suppress(OSError):
            open(f'/proc/{pid}/oom_score_adj', 'w').write(str(PRIORITY_OOM[args.priority or DEFAULT_PRIORITY[args.cls]]))
    with ledger() as state:
        # Re-adopting a claimed unit keeps its command: it names the container a `docker start -a` unit is charged for.
        # A fresh adoption takes the unit's own command line, so an attached container is measured too.
        # It also keeps the admission time (the victim order) and the measured peak.
        previous = state['claims'].get(unit, {})
        command = previous.get('command')
        if not command or command == ['(adopted)']:
            command = main_command(unit) or ['(adopted)']
        state['claims'][unit] = {'budget': int(args.budget_gb * GIB), 'class': args.cls, 'adopted': True,
                                 **({'priority': args.priority} if args.priority else {}),
                                 'admitted': previous.get('admitted', time.time()), 'command': command, 'host_max': None,
                                 **{key: previous[key] for key in ('peak', 'peak_since') if key in previous},
                                 **({'hold_budget': True} if args.hold_budget else {})}
        state['events'].append({'time': time.time(), 'event': 'adopted', 'unit': unit,
                                **({'hold_budget': True} if args.hold_budget else {})})


def status(args):
    with ledger() as state:
        free, live = headroom(state, gpu_usage())
        queue = waiting_order(state)
        graces = {}
        plans = {}
        for position, (name, wait) in enumerate(queue):
            plans[name] = preemption_plan(live, free, wait, graces)
        claims = {u: {'class': c['class'], 'priority': PRIORITY_NAMES[priority_of(c)],
                      'budget_gb': round(c['budget'] / GIB, 1), 'used_gb': round(c['used'] / GIB, 1),
                      'peak_gb': round(c.get('peak', 0) / GIB, 1), 'reserved_gb': round(outstanding(c) / GIB, 1),
                      'resumable': resumable(c), 'gpu': bool(c.get('gpu')),
                      **({'fixed_claim': True} if fixed_claim(c) else {}),
                      'command': shlex.join(c['command'])[:160]} for u, c in live.items()}
    print(json.dumps({'available_gb': round(mem_available() / GIB, 1), 'headroom_gb': round(free / GIB, 1),
                      'claims': claims,
                      'waiting': [{'position': n + 1, 'unit': name, 'priority': w['priority'],
                                   'budget_gb': round(w['budget'] / GIB, 1), 'requeued': bool(w.get('requeued')),
                                   'would_preempt': plans[name],
                                   'blocked_by': non_preemptible(live, w)} for n, (name, w) in enumerate(queue)]}, indent=2))


def victim(live, floor_breached, overshoot, pressure=None, grace=None):
    """The unit to stop, only under memory pressure (available below twice the floor; ``pressure`` defaults to
    ``floor_breached``): first the unit most over its budget by the overshoot factor, then, below the floor, the
    lowest-priority newest. A unit over its budget while memory is plentiful is not stopped (the guard raises its
    budget to what it uses instead): over 3 days, 7 of 9 over-budget stops happened with >20 GB still available,
    so they protected nothing and cost the work."""
    pressure = floor_breached if pressure is None else pressure
    over = [(c['used'] / c['budget'], u) for u, c in live.items() if c['used'] > c['budget'] * overshoot]
    if over and (pressure or floor_breached):
        return max(over)[1], 'over budget under memory pressure'
    if floor_breached and live:
        # Class priority first; within a class, the unit freeing the most memory per second of stop grace (small,
        # stateless jobs such as tests and probes before big long-running trainers), then the newest.
        grace = grace or {}
        return max(live, key=lambda u: (-priority_of(live[u]), CLASSES[live[u]['class']],
                                        live[u]['used'] / (grace.get(u, GRACE_MIN) + 1),
                                        live[u]['admitted'])), 'free memory below floor'
    return None, None


def escalation(pending, now, available, emergency, active):
    """What the guard does with a graceful stop in progress: ``done`` when the unit has exited, ``kill`` when the
    victim's grace has run out or available memory fell below the emergency level, else ``wait`` (the victim is
    writing its checkpoint)."""
    if not active:
        return 'done'
    if available < emergency:
        return 'kill'
    if now >= pending['deadline']:
        return 'kill'
    return 'wait'


def unit_containers(unit, command=None, extra=()):
    """Every docker container a unit runs: named by its command, recorded at launch (``run --container``), named in the
    command line of a docker client inside its cgroup (a wrapper script's ``docker run --rm --name X``), or named like the
    unit. Stopping the unit only ends the client (and a shell waiting on its foreground client runs its trap only after
    that client exits), so the containers must be stopped by name."""
    names = [container_name(command), *extra, unit.removesuffix('.service')]
    _, cgroup = unit_state(unit)
    if cgroup:
        with contextlib.suppress(OSError, ValueError):
            for pid in open('/sys/fs/cgroup' + cgroup + '/cgroup.procs').read().split():
                with contextlib.suppress(OSError):
                    text = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
                    names.append(container_name(text))
    return list(dict.fromkeys(n for n in names if n))


def container_running(name):
    out = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', name], capture_output=True, text=True)
    return out.stdout.strip() == 'true'


def as_list(containers):
    return [containers] if isinstance(containers, str) else list(containers)


def stop_containers(containers, seconds=10):
    """Stop (not wait for) every named container still running: a preempted unit's work must free its memory, and a
    requeued unit must not start a duplicate beside an orphan."""
    for name in as_list(containers):
        if container_running(name):
            subprocess.Popen(['docker', 'stop', '-t', str(seconds), name], stderr=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL)


def terminate(unit, container):
    """Ask a unit's job to checkpoint and exit: SIGTERM to the container's main process (``docker kill -s TERM``
    does not stop the container itself) and to the unit."""
    for name in as_list(container):
        subprocess.Popen(['docker', 'kill', '--signal', 'SIGTERM', name], stderr=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL)
    subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGTERM', unit])


def kill(unit, container):
    for name in as_list(container):
        subprocess.Popen(['docker', 'kill', name], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', unit])
    subprocess.Popen(['systemctl', '--user', 'stop', unit], stderr=subprocess.DEVNULL)


def checkpoint_writes(path=None, limit=2000):
    path = path or CHECKPOINT_WRITES
    try:
        lines = open(path).read().splitlines()[-limit:]
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def family_state_bytes(writes):
    """The largest recent checkpoint each job family wrote (per family of the writing unit)."""
    sizes = {}
    for row in writes:
        if row.get('unit'):
            name = family(row['unit'] if row['unit'].endswith('.service') else row['unit'] + '.service')
            sizes[name] = max(sizes.get(name, 0), int(row.get('bytes', 0)))
    return sizes


def write_throughput(writes):
    """Median measured bytes per second of large writes (>= 256 MiB; small ones are dominated by latency)."""
    rates = sorted(row['bytes'] / row['seconds'] for row in writes
                   if row.get('bytes', 0) >= 256 << 20 and row.get('seconds', 0) > 0)
    return rates[len(rates) // 2] if rates else DEFAULT_THROUGHPUT


def grace_seconds(unit, claim, sizes, throughput):
    """Seconds a stopping unit gets to write its full state: explicit ``stop_seconds``, else its family's measured
    checkpoint size over the measured throughput."""
    if claim.get('stop_seconds_explicit'):
        return float(claim['stop_seconds'])
    state = sizes.get(family(unit), 0)
    return max(GRACE_MIN, GRACE_SLACK + GRACE_MARGIN * state / throughput) if state else GRACE_MIN


def guard(args):
    floor = int(args.floor_gb * GIB)
    emergency = int(getattr(args, 'emergency_gb', EMERGENCY_GB) * GIB)
    stopped = {}
    pending = {}  # unit -> {'deadline', 'container'}: graceful stops in progress
    next_release, release = 0, None
    while True:
        # CUDA allocates only from MemFree, and the warm-up's memory preflight refuses an update when MemFree is
        # short: release clean page cache well above the floor, one walk at a time, at idle I/O priority, and only
        # when there is cache worth dropping (MemFree is also short when jobs simply use it). A walk takes minutes on
        # the external disk: after one that freed little, wait half an hour before the next.
        if release is not None and release.poll() is not None:
            try:
                result = json.loads(release.communicate()[0] or '{}')
                gained = (result.get('free_after_gb', 0) - result.get('free_before_gb', 0)) * GIB
            except (ValueError, AttributeError):
                gained = 0
            next_release = time.time() + (60 if gained >= GIB else 1800)
            release = None
        if mem_free() < 2 * floor and reclaimable() >= 2 * GIB and release is None and time.time() >= next_release:
            idle = ['ionice', '-c3'] if shutil.which('ionice') else []
            # Its own transient unit: the release balloons anonymous memory a GiB at a time, which inside the
            # guard's 256 MB cgroup got the guard itself OOM-killed (2026-10-09 22:04).
            own_unit = ['systemd-run', '--user', '--quiet', '--pipe', '--wait', '--collect',
                        '-p', 'MemoryMax=26G', '-p', 'MemorySwapMax=0'] if shutil.which('systemd-run') else []
            release = subprocess.Popen(own_unit + idle + [sys.executable, os.path.abspath(__file__), 'release-cache'],
                                       stdout=subprocess.PIPE, text=True)
        with ledger() as state:
            all_live = live_claims(state, gpu_usage())
            live = {u: c for u, c in all_live.items() if time.time() - stopped.get(u, 0) > 60}
            available = mem_available()
            free = available - sum(outstanding(c) for c in all_live.values())
            writes = checkpoint_writes()
            sizes, throughput = family_state_bytes(writes), write_throughput(writes)
            graces = {name: grace_seconds(name, state['claims'].get(name, {}), sizes, throughput) for name in live}
            unit, reason = victim(live, available < floor, args.overshoot, pressure=available < 2 * floor, grace=graces)
            for name, claim in live.items():  # overshoot without pressure: the budget becomes what it uses
                if not unit and claim['used'] > claim['budget'] * args.overshoot and name in state['claims']:
                    raised = int(claim['used'] * 1.1)
                    state['claims'][name]['budget'] = raised
                    state['events'].append({'time': time.time(), 'event': 'budget raised', 'unit': name,
                                            'used': claim['used'], 'budget': raised, 'available': available})
            if pending:
                # One graceful stop at a time: while a victim checkpoints, the guard watches it instead of picking
                # another (cache release above continues).
                unit, reason = None, None
            to_launch = None
            queue = waiting_order(state)
            if queue and not pending:
                head, wanted = queue[0]
                if not unit and not fits(free, wanted['budget'], wanted['reserve']):
                    # A waiting higher priority job: stop the lowest priority resumable units (one graceful stop at a
                    # time; the next cycle re-plans with what that stop freed).
                    plan = preemption_plan({u: c for u, c in live.items() if u not in stopped or
                                            time.time() - stopped[u] > 60}, free, wanted, graces)
                    if plan:
                        unit, reason = plan[0], f'preempted for {head} ({wanted["priority"]})'
                elif (not unit and wanted.get('requeued') and time.time() >= wanted['not_before']
                      and head not in state['claims'] and unit_state(head)[0] not in ('active', 'activating', 'deactivating')
                      and fits(free, wanted['budget'], wanted['reserve'])
                      and not any(container_running(n) for n in wanted.get('containers', ()))):
                    state['claims'][head] = claim_for(wanted['spec'])
                    state['waiting'].pop(head)
                    state['events'].append({'time': time.time(), 'event': 'relaunched', 'unit': head, 'free': free})
                    to_launch = wanted['spec']
            command = state['claims'].get(unit, {}).get('command') if unit else None
            state_claim_containers = state['claims'].get(unit, {}).get('spec', {}).get('containers', ()) if unit else ()
            containers = unit_containers(unit, command, state_claim_containers) if unit else []
            if unit and (reason.startswith('preempted') or reason.startswith('free memory below floor')):
                requeue_spec(state, unit, containers=containers)  # a resumable unit comes back when room returns
            grace = graces.get(unit, GRACE_MIN) if unit else 0
            if unit:
                state['events'].append({'time': time.time(), 'event': 'stopped', 'unit': unit, 'reason': reason,
                                        'grace': round(grace, 1),
                                        'used': live[unit]['used'], 'budget': live[unit]['budget'],
                                        'available': mem_available()})
        if to_launch and launch(to_launch):
            with ledger() as state:
                state['claims'].pop(to_launch['unit'], None)
                state['events'].append({'time': time.time(), 'event': 'relaunch failed', 'unit': to_launch['unit']})
        if unit:
            print(json.dumps({'stopped': unit, 'reason': reason, 'used_gb': round(live[unit]['used'] / GIB, 1)}),
                  flush=True)
            # Killing a `docker start -a NAME` unit only detaches its client; the container keeps running and keeps
            # its memory. Signal the job inside, then give it its grace to write its full state.
            container = containers
            terminate(unit, container)
            pending[unit] = {'deadline': time.time() + grace, 'container': container}
            stopped[unit] = time.time()
        for name in list(pending):
            action = escalation(pending[name], time.time(), mem_available(), emergency,
                                unit_state(name)[0] in ('active', 'activating', 'deactivating'))
            if action == 'wait':
                continue
            if action == 'done':
                stop_containers(pending[name]['container'])  # the unit ended; a detached container may live on
            if action == 'kill':
                kill(name, pending[name]['container'])
                with ledger() as state:
                    state['events'].append({'time': time.time(), 'event': 'killed after grace', 'unit': name,
                                            'available': mem_available(),
                                            'reason': 'emergency' if mem_available() < emergency else 'grace over'})
            pending.pop(name)
        if args.once:
            return
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run', help='admit and launch a job')
    r.add_argument('--unit', required=True)
    r.add_argument('--budget-gb', type=float, required=True, help='host plus CUDA memory the job may use')
    r.add_argument('--host-max-gb', type=float, default=None, help='MemoryMax for host memory (default: the budget)')
    r.add_argument('--oom-policy', choices=['stop', 'continue', 'kill'], default='stop',
                   help='when the kernel kills a process of the unit for memory: stop the unit (default), or continue '
                        '(a runner of many child processes, such as a test runner, records the kill and goes on)')
    r.add_argument('--class', dest='cls', choices=sorted(CLASSES), default='experiment')
    r.add_argument('--priority', choices=sorted(PRIORITIES, key=PRIORITIES.get),
                   help='high: training and GPU end-to-end runs; medium: smokes and evals; low: data generation, rendering, '
                        'check-main and other CPU jobs (default: medium, services high). Higher waits ahead, preempts '
                        'lower resumable units, is stopped last under pressure and gets a lower kernel OOM score')
    r.add_argument('--resumable', action=argparse.BooleanOptionalAction, default=None,
                   help='the command resumes from its own checkpoints, so the ledger may stop it for a higher priority job '
                        'or under pressure and relaunch it from the queue (default: yes except services)')
    r.add_argument('--reserve-gb', type=float, default=8, help='free memory that must remain after admission (the guard floor; owner: use the memory we have)')
    r.add_argument('--wait', type=float, default=0, help='seconds to wait for admission')
    r.add_argument('--hold-budget', '--fixed-claim', action='store_true',
                   help='always count the whole unspent budget, even after the job settles (working set grows later)')
    r.add_argument('--stop-seconds', type=float, default=None,
                   help='seconds the job gets to write its full state when stopped (planned or by the guard); default: '
                        'its family\'s measured checkpoint size over the measured write throughput')
    r.add_argument('--container', action='append', default=[],
                   help='name of a docker container the job runs outside the unit (repeatable); stopped with the unit. '
                        'Containers named by `docker run --name` in the command or its wrapper script are found anyway')
    r.add_argument('--workdir', default='.')
    r.add_argument('--env', action='append', default=[])
    r.add_argument('command', nargs=argparse.REMAINDER)
    a = sub.add_parser('adopt', help='register an already running unit')
    a.add_argument('--unit', required=True)
    a.add_argument('--budget-gb', type=float, required=True)
    a.add_argument('--class', dest='cls', choices=sorted(CLASSES), default='experiment')
    a.add_argument('--priority', choices=sorted(PRIORITIES, key=PRIORITIES.get))
    a.add_argument('--hold-budget', '--fixed-claim', action='store_true', help='always count the whole unspent budget')
    sub.add_parser('status')
    sub.add_parser('release-cache', help='drop clean page cache of large files under the data roots')
    g = sub.add_parser('guard', help='enforce budgets and the free-memory floor')
    g.add_argument('--floor-gb', type=float, default=8)
    g.add_argument('--overshoot', type=float, default=1.15)
    g.add_argument('--interval', type=float, default=5)
    g.add_argument('--emergency-gb', type=float, default=EMERGENCY_GB,
                   help='below this available memory a stopping unit is killed at once (no checkpoint grace)')
    g.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.action == 'run':
        if args.command[:1] == ['--']:
            args.command = args.command[1:]
        if not args.command:
            parser.error('run needs a command after --')
    {'run': run, 'adopt': adopt, 'status': status, 'guard': guard,
     'release-cache': lambda _: print(json.dumps(release_cache()))}[args.action](args)


if __name__ == '__main__':
    sys.exit(main())
