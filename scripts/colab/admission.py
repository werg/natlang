#!/usr/bin/env python3
"""VRAM and host-RAM admission for Colab discrete GPUs (lfm3opti.md §2, async-train.md §1).

A Colab G4 VM has separate GPU VRAM (96 GB) and host RAM (~170 GB). ``run`` admits a command when its claim fits
both pools beside the live claims of earlier admitted commands, records the claim in a claims file and runs the
command; the claim lasts while the admitting process (or its child) lives and is removed when the command ends. The
file lock is held only to read and write the claims (never while the command runs), so a long-lived actor and the
SFT learner beside it (MPS: learner + capped actor) are admitted together whenever their claims fit.

Capacity is the probed total less a reserve per pool. A command that finds no live claim also needs the probed free
memory (a process the shim does not know about holds the rest); with live claims the claims are the accounting,
because an admitted SGLang engine reserves its pool at launch and the free probe then reads its claim as used.

A ``--preemptible`` claim is a tenant (program tuning between rounds, DECISIONS 2026-11-24 "program tuning on the A100"): a normal
claim that does not fit (a round's SFT learner, an actor served again) evicts tenants, newest first, until it fits
(SIGTERM to the tenant's process group, SIGKILL after ``EVICT_GRACE_SECONDS``), and while a normal claim waits no tenant
is admitted. A tenant restarts from its own saved state; the work it yields to never waits on it.

    scripts/colab/admission.py check --vram-gib 80 --host-ram-gib 32
    scripts/colab/admission.py run --vram-gib 34 --host-ram-gib 24 --label actor --wait 600 -- python ...
    scripts/colab/admission.py status
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

STATE_DIR = Path(os.environ.get('NATLANG_COLAB_ADMISSION_DIR', '/tmp/natlang-colab-admission'))
VRAM_RESERVE_GIB, HOST_RESERVE_GIB = 1.0, 4.0
EXIT_TIMEOUT = 75
EVICT_GRACE_SECONDS = 60.0


def _meminfo() -> dict:
    try:
        return {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    except (OSError, ValueError, IndexError):
        return {}


def free_host_ram_gib() -> float:
    return _meminfo().get('MemAvailable', 0) / 2**20


def total_host_ram_gib() -> float:
    return _meminfo().get('MemTotal', 0) / 2**20


def _nvidia(field: str, device: int) -> float | None:
    try:
        out = subprocess.check_output(['nvidia-smi', f'--id={device}', f'--query-gpu={field}', '--format=csv,noheader,nounits'],
                                      text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
        return float(out.splitlines()[0]) / 1024.0 if out else None
    except Exception:
        return None


def free_vram_gib(device: int = 0) -> float | None:
    return _nvidia('memory.free', device)


def total_vram_gib(device: int = 0) -> float | None:
    return _nvidia('memory.total', device)


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


@contextlib.contextmanager
def locked(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_claims(state_dir: Path) -> dict:
    """Live claims ({id: claim}); a claim whose admitting process and child are both gone is dropped."""
    path = state_dir / 'claims.json'
    try:
        claims = json.loads(path.read_text())
    except (OSError, ValueError):
        claims = {}
    return {k: c for k, c in claims.items() if alive(c.get('pid')) or alive(c.get('child'))}


def write_claims(state_dir: Path, claims: dict) -> None:
    tmp = state_dir / 'claims.json.tmp'
    tmp.write_text(json.dumps(claims, indent=1, sort_keys=True) + '\n')
    os.replace(tmp, state_dir / 'claims.json')


def read_waiters(state_dir: Path) -> dict:
    """Normal (non-preemptible) requests waiting for admission ({id: request}) whose process lives."""
    try:
        waiters = json.loads((state_dir / 'waiting.json').read_text())
    except (OSError, ValueError):
        waiters = {}
    return {k: w for k, w in waiters.items() if alive(w.get('pid'))}


def write_waiters(state_dir: Path, waiters: dict) -> None:
    tmp = state_dir / 'waiting.json.tmp'
    tmp.write_text(json.dumps(waiters, indent=1, sort_keys=True) + '\n')
    os.replace(tmp, state_dir / 'waiting.json')


def pending_normal_requests(claims: dict, proc_root: Path = Path('/proc')) -> list[int]:
    """PIDs of ``admission.py run`` processes without ``--preemptible`` that hold no claim: normal requests waiting
    for admission. Read from the process table, so a waiting request of any version of this script counts (a round's
    learner admitted by an older checkout writes no waiting record and evicts nobody: tenants yield to it themselves)."""
    held = {int(c.get('pid') or 0) for c in claims.values()} | {int(c.get('child') or 0) for c in claims.values()}
    found = []
    for entry in proc_root.iterdir() if proc_root.is_dir() else ():
        if not entry.name.isdigit() or int(entry.name) in held or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        words = [w.decode(errors='replace') for w in argv if w]
        script = next((i for i, w in enumerate(words) if w.endswith('admission.py')), None)
        if script is None or 'run' not in words[script + 1:]:
            continue
        own = words[script + 1:words.index('--')] if '--' in words else words[script + 1:]
        if '--preemptible' not in own:
            found.append(int(entry.name))
    return sorted(found)


def evictions(vram_gib: float, host_ram_gib: float, claims: dict, **probes) -> list[str] | None:
    """The preemptible claims (newest first) a normal request evicts to fit, ``[]`` when it fits as is, ``None`` when
    evicting every tenant would not make it fit (pure; ``probes`` are ``decide``'s keyword arguments). With tenants
    left the claims are the accounting; with none left the probed free memory must also hold the request, and the
    tenants' memory counts as free once they are gone."""
    ok, _ = decide(vram_gib, host_ram_gib, claims, **probes)
    if ok:
        return []
    tenants = sorted((k for k, c in claims.items() if c.get('preemptible')),
                     key=lambda k: -float(claims[k].get('started', 0)))
    chosen, rest = [], dict(claims)
    for key in tenants:
        chosen.append(key)
        freed = rest.pop(key)
        probes = {**probes, 'free_vram': (probes.get('free_vram') or 0) + float(freed.get('vram_gib', 0)),
                  'free_host': (probes.get('free_host') or 0) + float(freed.get('host_ram_gib', 0))}
        if decide(vram_gib, host_ram_gib, rest, **probes)[0]:
            return chosen
    return None


def evict(claim: dict, grace_seconds: float = EVICT_GRACE_SECONDS) -> None:
    """Stop a tenant: SIGTERM to its admission process's group when it leads one (tenants start under setsid), else
    to the process (which forwards it to its command); SIGKILL after ``grace_seconds`` in a background thread."""
    pid = int(claim.get('pid') or 0)
    if not pid:
        return
    try:
        group = os.getpgid(pid) == pid and pid != os.getpgrp()
    except OSError:
        return
    kill = (lambda sig: os.killpg(pid, sig)) if group else (lambda sig: os.kill(pid, sig))

    def finish():
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline and alive(pid):
            time.sleep(1.0)
        with contextlib.suppress(OSError):
            if alive(pid):
                kill(signal.SIGKILL)
    with contextlib.suppress(OSError):
        kill(signal.SIGTERM)
        threading.Thread(target=finish, daemon=True).start()


def decide(vram_gib: float, host_ram_gib: float, claims: dict, *, total_vram, total_host, free_vram, free_host,
           vram_reserve_gib: float = VRAM_RESERVE_GIB, host_reserve_gib: float = HOST_RESERVE_GIB) -> tuple[bool, dict]:
    """Whether a claim of (``vram_gib``, ``host_ram_gib``) fits beside ``claims`` (pure; the probes are arguments)."""
    claimed_vram = sum(float(c.get('vram_gib', 0)) for c in claims.values())
    claimed_host = sum(float(c.get('host_ram_gib', 0)) for c in claims.values())
    reasons = []
    if vram_gib > 0:
        if total_vram is None:
            reasons.append('no GPU found')
        elif claimed_vram + vram_gib > total_vram - vram_reserve_gib:
            reasons.append(f'vram: {claimed_vram:.1f} GiB claimed + {vram_gib:.1f} > {total_vram - vram_reserve_gib:.1f}')
        elif not claims and (free_vram or 0) < vram_gib + vram_reserve_gib:
            reasons.append(f'vram: {free_vram} GiB free < {vram_gib + vram_reserve_gib:.1f} (an unadmitted process)')
    if host_ram_gib > 0:
        if claimed_host + host_ram_gib > total_host - host_reserve_gib:
            reasons.append(f'host_ram: {claimed_host:.1f} GiB claimed + {host_ram_gib:.1f} > {total_host - host_reserve_gib:.1f}')
        elif not claims and free_host < host_ram_gib + host_reserve_gib:
            reasons.append(f'host_ram: {free_host:.1f} GiB free < {host_ram_gib + host_reserve_gib:.1f}')
    info = {'admitted': not reasons, 'claimed_vram_gib': round(claimed_vram, 2), 'claimed_host_ram_gib': round(claimed_host, 2),
            'request': {'vram_gib': vram_gib, 'host_ram_gib': host_ram_gib}, 'live_claims': sorted(claims),
            'total_vram_gib': total_vram, 'free_vram_gib': free_vram, 'free_host_ram_gib': round(free_host, 2),
            'reasons': reasons}
    return not reasons, info


def check_admission(vram_gib: float, host_ram_gib: float, *, device: int = 0, state_dir: Path = STATE_DIR,
                    vram_probe=None, host_probe=None, total_vram_probe=None, total_host_probe=None):
    claims = read_claims(state_dir)
    return decide(vram_gib, host_ram_gib, claims,
                  total_vram=(total_vram_probe or (lambda: total_vram_gib(device)))(),
                  total_host=(total_host_probe or total_host_ram_gib)(),
                  free_vram=(vram_probe or (lambda: free_vram_gib(device)))(),
                  free_host=(host_probe or free_host_ram_gib)())


def wait_and_run(command: list[str], *, vram_gib: float, host_ram_gib: float, label: str = '',
                 wait_seconds: float = 600.0, poll_seconds: float = 5.0, device: int = 0, state_dir: Path = STATE_DIR,
                 workdir: str | None = None, probes: dict | None = None, preemptible: bool = False,
                 beside_label: str | None = None, proc_root: Path = Path('/proc'), watch_seconds: float = 5.0) -> int:
    """Admit, record the claim, run ``command`` with the lock released, drop the claim when it ends. A normal
    request evicts the tenants it needs gone (``evictions``) and keeps tenants out while it waits; a ``preemptible``
    one is a tenant."""
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    claim_id = f'{label or Path(command[0]).name}-{os.getpid()}'
    evicted = set()
    try:
        while True:
            with locked(state_dir):
                claims = read_claims(state_dir)
                ok, info = check_admission(vram_gib, host_ram_gib, device=device, state_dir=state_dir, **(probes or {}))
                waiters = read_waiters(state_dir)
                if preemptible and ok:
                    pending = sorted(waiters) or pending_normal_requests(claims, proc_root)
                    beside = [k for k, c in claims.items() if not c.get('preemptible')
                              and str(c.get('label') or '').startswith(beside_label or '')]
                    if pending:
                        ok, info = False, {**info, 'reasons': [f'normal requests wait: {pending}']}
                    elif beside_label is not None and not beside:
                        ok, info = False, {**info, 'reasons': [f'no live claim labelled {beside_label}*']}
                if ok:
                    claims[claim_id] = {'pid': os.getpid(), 'child': None, 'vram_gib': vram_gib,
                                        'host_ram_gib': host_ram_gib, 'label': label, 'preemptible': preemptible,
                                        'command': ' '.join(command)[:300], 'started': time.time()}
                    write_claims(state_dir, claims)
                    break
                if not preemptible:
                    waiters[claim_id] = {'pid': os.getpid(), 'vram_gib': vram_gib, 'host_ram_gib': host_ram_gib,
                                         'label': label}
                    write_waiters(state_dir, waiters)
                    p = probes or {}
                    victims = evictions(vram_gib, host_ram_gib, claims,
                                        total_vram=(p.get('total_vram_probe') or (lambda: total_vram_gib(device)))(),
                                        total_host=(p.get('total_host_probe') or total_host_ram_gib)(),
                                        free_vram=(p.get('vram_probe') or (lambda: free_vram_gib(device)))(),
                                        free_host=(p.get('host_probe') or free_host_ram_gib)())
                    for key in victims or []:
                        if key not in evicted:
                            evicted.add(key)
                            print(json.dumps({'event': 'colab admission evicts a tenant', 'claim': claim_id,
                                              'tenant': key, 'label': claims[key].get('label')}), flush=True)
                            evict(claims[key])
            if time.monotonic() >= deadline:
                print(json.dumps({'event': 'colab admission timeout', **info}), file=sys.stderr, flush=True)
                return EXIT_TIMEOUT
            time.sleep(min(poll_seconds, max(0.1, deadline - time.monotonic())))
    finally:
        if not preemptible:
            with locked(state_dir):
                waiters = read_waiters(state_dir)
                if waiters.pop(claim_id, None) is not None:
                    write_waiters(state_dir, waiters)
    print(json.dumps({'event': 'colab admission granted', 'claim': claim_id, **info}), flush=True)
    proc = None

    def forward(signum, _frame):
        if proc is not None and proc.poll() is None:
            if preemptible:      # the tenant's command leads its own session: its whole group
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signum)
            else:
                proc.send_signal(signum)
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, forward)
    stop_watch = threading.Event()

    def watch():
        # a tenant yields as soon as a normal request waits (whatever version of this script admits it): its whole
        # process group (an SGLang actor's server and schedulers too) gets SIGTERM, SIGKILL after the grace period
        while not stop_watch.wait(watch_seconds):
            with locked(state_dir):
                live = read_claims(state_dir)
                pending = sorted(read_waiters(state_dir)) or pending_normal_requests(live, proc_root)
            if pending and proc is not None and proc.poll() is None:
                print(json.dumps({'event': 'colab tenant yields', 'claim': claim_id, 'to': pending}), flush=True)
                evict({'pid': proc.pid})
                return
    try:
        proc = subprocess.Popen(command, cwd=workdir, start_new_session=preemptible)
        with locked(state_dir):
            claims = read_claims(state_dir)
            if claim_id in claims:
                claims[claim_id]['child'] = proc.pid
                write_claims(state_dir, claims)
        if preemptible:
            threading.Thread(target=watch, daemon=True).start()
        return proc.wait()
    finally:
        stop_watch.set()
        with locked(state_dir):
            claims = read_claims(state_dir)
            claims.pop(claim_id, None)
            write_claims(state_dir, claims)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--state-dir', type=Path, default=STATE_DIR)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('check', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--vram-gib', type=float, default=0.0)
        p.add_argument('--host-ram-gib', type=float, default=4.0)
        p.add_argument('--device', type=int, default=0)
    sub.add_parser('status')
    r = sub.choices['run']
    r.add_argument('--label', default='')
    r.add_argument('--wait', type=float, default=600.0)
    r.add_argument('--poll', type=float, default=5.0)
    r.add_argument('--workdir', default=None)
    r.add_argument('--preemptible', action='store_true',
                   help='a tenant: it yields to any waiting normal request (and normal claims that do not fit evict it)')
    r.add_argument('--beside-label', help='a tenant is admitted only beside a live normal claim with this label prefix '
                                          '(actor-: never while the session trains a round)')
    r.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(json.dumps({**read_claims(args.state_dir), **{f'waiting:{k}': w for k, w in read_waiters(args.state_dir).items()}},
                         indent=1, sort_keys=True))
        return 0
    if args.action == 'check':
        ok, info = check_admission(args.vram_gib, args.host_ram_gib, device=args.device, state_dir=args.state_dir)
        print(json.dumps(info, sort_keys=True))
        return 0 if ok else 1
    cmd = list(args.command)
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit('admission.py run requires a command after --')
    return wait_and_run(cmd, vram_gib=args.vram_gib, host_ram_gib=args.host_ram_gib, label=args.label,
                        wait_seconds=args.wait, poll_seconds=args.poll, device=args.device, state_dir=args.state_dir,
                        workdir=args.workdir, preemptible=args.preemptible, beside_label=args.beside_label)


if __name__ == '__main__':
    sys.exit(main())
