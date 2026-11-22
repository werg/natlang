#!/usr/bin/env python3
"""Separate VRAM GiB and host RAM GiB admission shim for Colab discrete GPUs (lfm3opti.md §2).

Unlike the DGX GB10 unified-memory ``memory_ledger.py``, Colab G4 VMs have separate GPU VRAM (96 GiB)
and host system RAM (~170 GiB). This shim checks both pools independently before admitting a command,
coordinates concurrent processes via a local file lock, and exits with the child command's status code.

    scripts/colab/admission.py check --vram-gib 80 --host-ram-gib 32
    scripts/colab/admission.py run --vram-gib 80 --host-ram-gib 32 --wait 600 -- python ...
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LOCK_PATH = Path(os.environ.get('NATLANG_COLAB_ADMISSION_LOCK', '/tmp/natlang-colab-admission.lock'))


def free_host_ram_gib() -> float:
    """Available host system RAM in GiB from /proc/meminfo."""
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                kb = int(line.split()[1])
                return kb / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def free_vram_gib(device: int = 0) -> float | None:
    """Free GPU VRAM in GiB via nvidia-smi (falls back to torch.cuda.mem_get_info). Returns None if no GPU."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', f'--id={device}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        ).strip()
        if out:
            mib = float(out.splitlines()[0].strip())
            return mib / 1024.0
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info(device)
            return free_bytes / float(2**30)
    except Exception:
        pass
    return None


def check_admission(
    vram_gib: float,
    host_ram_gib: float,
    *,
    device: int = 0,
    vram_reserve_gib: float = 1.0,
    host_reserve_gib: float = 4.0,
    vram_probe=None,
    host_probe=None,
) -> tuple[bool, dict]:
    """Check whether both discrete VRAM and host RAM satisfy the requested budgets + reserves."""
    avail_vram = (vram_probe() if vram_probe is not None else free_vram_gib(device))
    avail_host = (host_probe() if host_probe is not None else free_host_ram_gib())
    need_vram = max(0.0, float(vram_gib)) + max(0.0, float(vram_reserve_gib)) if vram_gib > 0 else 0.0
    need_host = max(0.0, float(host_ram_gib)) + max(0.0, float(host_reserve_gib)) if host_ram_gib > 0 else 0.0
    vram_ok = (need_vram == 0.0) or (avail_vram is not None and avail_vram >= need_vram)
    host_ok = (need_host == 0.0) or (avail_host >= need_host)
    reasons = []
    if not vram_ok:
        reasons.append(f'vram {avail_vram} GiB < required {need_vram:.1f} GiB')
    if not host_ok:
        reasons.append(f'host_ram {avail_host:.1f} GiB < required {need_host:.1f} GiB')
    info = {
        'admitted': vram_ok and host_ok,
        'avail_vram_gib': None if avail_vram is None else round(avail_vram, 2),
        'need_vram_gib': round(need_vram, 2),
        'avail_host_ram_gib': round(avail_host, 2),
        'need_host_ram_gib': round(need_host, 2),
        'reasons': reasons,
    }
    return info['admitted'], info


def wait_and_run(
    command: list[str],
    *,
    vram_gib: float,
    host_ram_gib: float,
    wait_seconds: float = 600.0,
    poll_seconds: float = 5.0,
    device: int = 0,
    lock_path: Path = DEFAULT_LOCK_PATH,
    workdir: str | None = None,
) -> int:
    """Wait until VRAM and host RAM are admissible under lock, then execute ``command``."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    with lock_path.open('a+') as handle:
        while True:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                ok, info = check_admission(vram_gib, host_ram_gib, device=device)
                if ok:
                    print(json.dumps({'event': 'colab admission granted', **info}), flush=True)
                    proc = subprocess.run(command, cwd=workdir)
                    return proc.returncode
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
            if time.monotonic() >= deadline:
                print(json.dumps({'event': 'colab admission timeout', **info}), file=sys.stderr, flush=True)
                return 75
            time.sleep(min(poll_seconds, max(0.1, deadline - time.monotonic())))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('check', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--vram-gib', type=float, default=0.0)
        p.add_argument('--host-ram-gib', type=float, default=4.0)
        p.add_argument('--device', type=int, default=0)
        p.add_argument('--lock', type=Path, default=DEFAULT_LOCK_PATH)
    r = sub.choices['run']
    r.add_argument('--wait', type=float, default=600.0)
    r.add_argument('--poll', type=float, default=5.0)
    r.add_argument('--workdir', default=None)
    r.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == 'check':
        ok, info = check_admission(args.vram_gib, args.host_ram_gib, device=args.device)
        print(json.dumps(info, sort_keys=True))
        return 0 if ok else 1
    cmd = list(args.command)
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit('admission.py run requires a command after --')
    return wait_and_run(
        cmd, vram_gib=args.vram_gib, host_ram_gib=args.host_ram_gib,
        wait_seconds=args.wait, poll_seconds=args.poll, device=args.device,
        lock_path=args.lock, workdir=args.workdir,
    )


if __name__ == '__main__':
    sys.exit(main())
