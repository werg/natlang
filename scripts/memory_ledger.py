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

Outstanding demand of a running claim is its budget minus what it already uses, so a job still ramping up is
counted at its full budget and a job at its peak is counted once, through MemAvailable. A claim that has run for
SETTLE_SECONDS, and been measured that long, has shown its working set: from then on it is counted up to the highest
use the ledger has measured (the guard measures every few seconds), not its whole budget. The guard's floor still covers a later spike, and
its first victim is the newest experiment. A claim made with --hold-budget (a job whose working set grows on its own
schedule, or that checks MemFree before each step) always counts its whole unspent budget.
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
import time

STATE = os.path.expanduser(os.environ.get('NATLANG_MEMORY_LEDGER', '~/.local/state/natlang/memory-ledger.json'))
# Higher is killed first by the kernel. Unprivileged units can only raise their score, so protection of the
# campaign and the vLLM server (score 0) comes from every admitted job ranking above them.
CLASSES = {'experiment': 900, 'collection': 600, 'service': 300}
GIB = 2**30
SETTLE_SECONDS = 1800


def mem_available():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing from /proc/meminfo')


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
    return host + sum(gpu.get(pid, 0) for pid in pids)


def container_cgroup(command):
    """The cgroup of the container a `docker start -a NAME` unit attaches to: its processes are Docker's, not the
    unit's, so the unit's own cgroup shows only the attached client."""
    text = ' '.join(command) if isinstance(command, list) else str(command or '')
    match = re.search(r'docker start (?:-a|--attach) (\S+)', text)
    if not match:
        return None
    out = subprocess.run(['docker', 'inspect', '-f', '{{.Id}}', match.group(1).strip("'\"")],
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
    container = container_cgroup(command)
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
    return max(budget, int(peak * 1.1))


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
        claim['peak'] = max(claim.get('peak', 0), used or 0)
        claim.setdefault('peak_since', time.time())
        live[unit] = dict(claim, used=used or 0)
    return live


def outstanding(claim, now=None):
    """What a claim may still take: its unspent budget, or, once it has settled (run and been measured for
    SETTLE_SECONDS), up to its measured peak."""
    settled = not claim.get('hold_budget') and \
        (now or time.time()) - max(claim['admitted'], claim.get('peak_since', time.time())) >= SETTLE_SECONDS
    ceiling = min(claim['budget'], claim.get('peak', 0)) if settled else claim['budget']
    return max(0, ceiling - claim['used'])


def headroom(state, gpu):
    live = live_claims(state, gpu)
    return mem_available() - sum(outstanding(c) for c in live.values()), live


def run(args):
    budget = int(args.budget_gb * GIB)
    reserve = int(args.reserve_gb * GIB)
    unit = args.unit if args.unit.endswith('.service') else args.unit + '.service'
    # Before admission: a claim whose unit has not started within 30 s is released by the next ledger call, and a
    # cache walk takes minutes. Walk only when there is cache to drop; skip it when another walk is running.
    if mem_free() < budget + reserve and reclaimable() >= GIB:
        print(json.dumps({'released_cache': release_cache()}), file=sys.stderr)
    deadline = time.time() + args.wait
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
            if free - budget >= reserve:
                state['claims'][unit] = {'budget': budget, 'class': args.cls, 'admitted': time.time(),
                                         'command': args.command, 'host_max': args.host_max_gb,
                                         **({'hold_budget': True} if args.hold_budget else {})}
                state['events'].append({'time': time.time(), 'event': 'admitted', 'unit': unit, 'budget': budget,
                                        'free': free})
                break
        if time.time() >= deadline:
            raise SystemExit(f'not admitted: {free / GIB:.1f} GiB free after outstanding claims, '
                             f'{args.budget_gb} GiB requested, {args.reserve_gb} GiB reserve')
        time.sleep(30)
    host_max = args.host_max_gb or max(args.budget_gb, budget / GIB)
    command = ['systemd-run', '--user', '--unit', unit, '-p', f'MemoryMax={int(host_max * GIB)}',
               '-p', 'MemorySwapMax=0', '-p', f'OOMScoreAdjust={CLASSES[args.cls]}', '-p', f'OOMPolicy={args.oom_policy}',
               '--working-directory', os.path.abspath(args.workdir),
               '-E', f'NATLANG_CUDA_MEMORY_GB={round(budget / GIB, 1)}', '-E', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True']
    for env in args.env:
        command += ['-E', env]
    container = container_name(args.command)
    if container:  # a stopped unit takes its container with it (an orphaned one held 36 GB outside the ledger)
        command += ['-p', f'ExecStopPost=-/usr/bin/docker stop -t 30 {container}']
    result = subprocess.run(command + args.command)
    if result.returncode:
        with ledger() as state:
            state['claims'].pop(unit, None)
        raise SystemExit(result.returncode)
    print(json.dumps({'unit': unit, 'budget_gb': round(budget / GIB, 1), 'class': args.cls, 'free_gb': round(free / GIB, 1)}))


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
            open(f'/proc/{pid}/oom_score_adj', 'w').write(str(CLASSES[args.cls]))
    with ledger() as state:
        # Re-adopting a claimed unit keeps its command: it names the container a `docker start -a` unit is charged for.
        # A fresh adoption takes the unit's own command line, so an attached container is measured too.
        # It also keeps the admission time (the victim order) and the measured peak.
        previous = state['claims'].get(unit, {})
        command = previous.get('command')
        if not command or command == ['(adopted)']:
            command = main_command(unit) or ['(adopted)']
        state['claims'][unit] = {'budget': int(args.budget_gb * GIB), 'class': args.cls,
                                 'admitted': previous.get('admitted', time.time()), 'command': command, 'host_max': None,
                                 **{key: previous[key] for key in ('peak', 'peak_since') if key in previous},
                                 **({'hold_budget': True} if args.hold_budget else {})}
        state['events'].append({'time': time.time(), 'event': 'adopted', 'unit': unit,
                                **({'hold_budget': True} if args.hold_budget else {})})


def status(args):
    with ledger() as state:
        free, live = headroom(state, gpu_usage())
    print(json.dumps({'available_gb': round(mem_available() / GIB, 1), 'headroom_gb': round(free / GIB, 1),
                      'claims': {u: {'class': c['class'], 'budget_gb': round(c['budget'] / GIB, 1),
                                     'used_gb': round(c['used'] / GIB, 1), 'peak_gb': round(c.get('peak', 0) / GIB, 1),
                                     'outstanding_gb': round(outstanding(c) / GIB, 1),
                                     **({'hold_budget': True} if c.get('hold_budget') else {}),
                                     'command': shlex.join(c['command'])[:160]} for u, c in live.items()}}, indent=2))


def victim(live, floor_breached, overshoot, pressure=None):
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
        return max(live, key=lambda u: (CLASSES[live[u]['class']], live[u]['admitted'])), 'free memory below floor'
    return None, None


def guard(args):
    floor = int(args.floor_gb * GIB)
    stopped = {}
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
            release = subprocess.Popen(idle + [sys.executable, os.path.abspath(__file__), 'release-cache'],
                                       stdout=subprocess.PIPE, text=True)
        with ledger() as state:
            live = live_claims(state, gpu_usage())
            live = {u: c for u, c in live.items() if time.time() - stopped.get(u, 0) > 60}
            available = mem_available()
            unit, reason = victim(live, available < floor, args.overshoot, pressure=available < 2 * floor)
            for name, claim in live.items():  # overshoot without pressure: the budget becomes what it uses
                if not unit and claim['used'] > claim['budget'] * args.overshoot and name in state['claims']:
                    raised = int(claim['used'] * 1.1)
                    state['claims'][name]['budget'] = raised
                    state['events'].append({'time': time.time(), 'event': 'budget raised', 'unit': name,
                                            'used': claim['used'], 'budget': raised, 'available': available})
            command = state['claims'].get(unit, {}).get('command') if unit else None
            if unit:
                state['events'].append({'time': time.time(), 'event': 'stopped', 'unit': unit, 'reason': reason,
                                        'used': live[unit]['used'], 'budget': live[unit]['budget'],
                                        'available': mem_available()})
        if unit:
            print(json.dumps({'stopped': unit, 'reason': reason, 'used_gb': round(live[unit]['used'] / GIB, 1)}),
                  flush=True)
            subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGTERM', unit])
            # Killing a `docker start -a NAME` unit only detaches its client; the container keeps running and keeps
            # its memory. Stop the container itself.
            container = container_name(command)
            if container:
                subprocess.Popen(['docker', 'stop', '-t', '20', container])
            subprocess.Popen(['sh', '-c', f'sleep 20; systemctl --user stop {shlex.quote(unit)} 2>/dev/null'])
            stopped[unit] = time.time()
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
    r.add_argument('--reserve-gb', type=float, default=8, help='free memory that must remain after admission (the guard floor; owner: use the memory we have)')
    r.add_argument('--wait', type=float, default=0, help='seconds to wait for admission')
    r.add_argument('--hold-budget', action='store_true',
                   help='always count the whole unspent budget, even after the job settles (working set grows later)')
    r.add_argument('--workdir', default='.')
    r.add_argument('--env', action='append', default=[])
    r.add_argument('command', nargs=argparse.REMAINDER)
    a = sub.add_parser('adopt', help='register an already running unit')
    a.add_argument('--unit', required=True)
    a.add_argument('--budget-gb', type=float, required=True)
    a.add_argument('--class', dest='cls', choices=sorted(CLASSES), default='experiment')
    a.add_argument('--hold-budget', action='store_true', help='always count the whole unspent budget')
    sub.add_parser('status')
    sub.add_parser('release-cache', help='drop clean page cache of large files under the data roots')
    g = sub.add_parser('guard', help='enforce budgets and the free-memory floor')
    g.add_argument('--floor-gb', type=float, default=8)
    g.add_argument('--overshoot', type=float, default=1.15)
    g.add_argument('--interval', type=float, default=5)
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
