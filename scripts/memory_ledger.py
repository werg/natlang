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
  guard    loop: stop a unit that exceeds its budget, and when free memory falls below the floor stop the
           lowest-priority admitted unit. Only units admitted through this ledger are ever stopped.

Outstanding demand of a running claim is its budget minus what it already uses, so a job still ramping up is
counted at its full budget and a job at its peak is counted once, through MemAvailable.
"""
import argparse
import contextlib
import fcntl
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


def release_cache(roots=None, min_bytes=16 << 20):
    """posix_fadvise(DONTNEED) on every file of at least ``min_bytes`` under the roots: drops only clean cached
    pages (dirty pages and mapped pages in use stay), so running jobs at worst re-read from disk."""
    roots = roots or [r for r in os.environ.get('NATLANG_CACHE_ROOTS', '').split(':') if r] or CACHE_ROOTS
    before, files = mem_free(), 0
    for root in roots:
        for directory, subdirs, names in os.walk(os.path.expanduser(root)):
            subdirs[:] = [d for d in subdirs if d not in ('.git', 'node_modules', 'proc')]
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
    return {'files': files, 'free_before_gb': round(before / GIB, 1), 'free_after_gb': round(mem_free() / GIB, 1)}


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


def live_claims(state, gpu):
    """Claims whose unit still runs, with measured use; claims of finished units are released."""
    live = {}
    for unit, claim in list(state['claims'].items()):
        used = unit_usage(unit, gpu, claim.get('command'))
        if used is None and time.time() - claim['admitted'] > 30:  # give systemd a moment to start the unit
            del state['claims'][unit]
            state['events'].append({'time': time.time(), 'event': 'released', 'unit': unit})
            continue
        live[unit] = dict(claim, used=used or 0)
    return live


def headroom(state, gpu):
    live = live_claims(state, gpu)
    outstanding = sum(max(0, c['budget'] - c['used']) for c in live.values())
    return mem_available() - outstanding, live


def run(args):
    budget = int(args.budget_gb * GIB)
    reserve = int(args.reserve_gb * GIB)
    unit = args.unit if args.unit.endswith('.service') else args.unit + '.service'
    deadline = time.time() + args.wait
    # Before admission: a claim whose unit has not started within 30 s is released by the next ledger call, and a
    # cache walk takes minutes.
    if mem_free() < budget + reserve:
        print(json.dumps({'released_cache': release_cache()}), file=sys.stderr)
    while True:
        with ledger() as state:
            free, live = headroom(state, gpu_usage())
            if unit in live:
                raise SystemExit(f'{unit} already holds a claim')
            if free - budget >= reserve:
                state['claims'][unit] = {'budget': budget, 'class': args.cls, 'admitted': time.time(),
                                         'command': args.command, 'host_max': args.host_max_gb}
                state['events'].append({'time': time.time(), 'event': 'admitted', 'unit': unit, 'budget': budget,
                                        'free': free})
                break
        if time.time() >= deadline:
            raise SystemExit(f'not admitted: {free / GIB:.1f} GiB free after outstanding claims, '
                             f'{args.budget_gb} GiB requested, {args.reserve_gb} GiB reserve')
        time.sleep(30)
    host_max = args.host_max_gb or args.budget_gb
    command = ['systemd-run', '--user', '--unit', unit, '-p', f'MemoryMax={int(host_max * GIB)}',
               '-p', 'MemorySwapMax=0', '-p', f'OOMScoreAdjust={CLASSES[args.cls]}',
               '--working-directory', os.path.abspath(args.workdir),
               '-E', f'NATLANG_CUDA_MEMORY_GB={args.budget_gb}', '-E', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True']
    for env in args.env:
        command += ['-E', env]
    result = subprocess.run(command + args.command)
    if result.returncode:
        with ledger() as state:
            state['claims'].pop(unit, None)
        raise SystemExit(result.returncode)
    print(json.dumps({'unit': unit, 'budget_gb': args.budget_gb, 'class': args.cls, 'free_gb': round(free / GIB, 1)}))


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
        command = state['claims'].get(unit, {}).get('command')
        if not command or command == ['(adopted)']:
            command = main_command(unit) or ['(adopted)']
        state['claims'][unit] = {'budget': int(args.budget_gb * GIB), 'class': args.cls, 'admitted': time.time(),
                                 'command': command, 'host_max': None}
        state['events'].append({'time': time.time(), 'event': 'adopted', 'unit': unit})


def status(args):
    with ledger() as state:
        free, live = headroom(state, gpu_usage())
    print(json.dumps({'available_gb': round(mem_available() / GIB, 1), 'headroom_gb': round(free / GIB, 1),
                      'claims': {u: {'class': c['class'], 'budget_gb': round(c['budget'] / GIB, 1),
                                     'used_gb': round(c['used'] / GIB, 1),
                                     'command': shlex.join(c['command'])[:160]} for u, c in live.items()}}, indent=2))


def victim(live, floor_breached, overshoot):
    """The unit to stop: first any over its budget by the overshoot factor, then the lowest-priority newest."""
    over = [(c['used'] / c['budget'], u) for u, c in live.items() if c['used'] > c['budget'] * overshoot]
    if over:
        return max(over)[1], 'over budget'
    if floor_breached and live:
        return max(live, key=lambda u: (CLASSES[live[u]['class']], live[u]['admitted'])), 'free memory below floor'
    return None, None


def guard(args):
    floor = int(args.floor_gb * GIB)
    stopped = {}
    released = 0
    while True:
        if mem_free() < floor and time.time() - released > 300:
            released = time.time()
            subprocess.Popen([sys.executable, os.path.abspath(__file__), 'release-cache'])
        with ledger() as state:
            live = live_claims(state, gpu_usage())
            live = {u: c for u, c in live.items() if time.time() - stopped.get(u, 0) > 60}
            unit, reason = victim(live, mem_available() < floor, args.overshoot)
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
            container = re.search(r'docker start (?:-a|--attach) (\S+)',
                                  ' '.join(command) if isinstance(command, list) else str(command or ''))
            if container:
                subprocess.Popen(['docker', 'stop', '-t', '20', container.group(1).strip("'\"")])
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
    r.add_argument('--class', dest='cls', choices=sorted(CLASSES), default='experiment')
    r.add_argument('--reserve-gb', type=float, default=8, help='free memory that must remain after admission (the guard floor; owner: use the memory we have)')
    r.add_argument('--wait', type=float, default=0, help='seconds to wait for admission')
    r.add_argument('--workdir', default='.')
    r.add_argument('--env', action='append', default=[])
    r.add_argument('command', nargs=argparse.REMAINDER)
    a = sub.add_parser('adopt', help='register an already running unit')
    a.add_argument('--unit', required=True)
    a.add_argument('--budget-gb', type=float, required=True)
    a.add_argument('--class', dest='cls', choices=sorted(CLASSES), default='experiment')
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
