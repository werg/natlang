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
                 workdir: str | None = None, probes: dict | None = None) -> int:
    """Admit, record the claim, run ``command`` with the lock released, drop the claim when it ends."""
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    claim_id = f'{label or Path(command[0]).name}-{os.getpid()}'
    while True:
        with locked(state_dir):
            claims = read_claims(state_dir)
            ok, info = check_admission(vram_gib, host_ram_gib, device=device, state_dir=state_dir, **(probes or {}))
            if ok:
                claims[claim_id] = {'pid': os.getpid(), 'child': None, 'vram_gib': vram_gib, 'host_ram_gib': host_ram_gib,
                                    'label': label, 'command': ' '.join(command)[:300], 'started': time.time()}
                write_claims(state_dir, claims)
                break
        if time.monotonic() >= deadline:
            print(json.dumps({'event': 'colab admission timeout', **info}), file=sys.stderr, flush=True)
            return EXIT_TIMEOUT
        time.sleep(min(poll_seconds, max(0.1, deadline - time.monotonic())))
    print(json.dumps({'event': 'colab admission granted', 'claim': claim_id, **info}), flush=True)
    proc = None

    def forward(signum, _frame):
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, forward)
    try:
        proc = subprocess.Popen(command, cwd=workdir)
        with locked(state_dir):
            claims = read_claims(state_dir)
            if claim_id in claims:
                claims[claim_id]['child'] = proc.pid
                write_claims(state_dir, claims)
        return proc.wait()
    finally:
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
    r.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(json.dumps(read_claims(args.state_dir), indent=1, sort_keys=True))
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
                        workdir=args.workdir)


if __name__ == '__main__':
    sys.exit(main())
