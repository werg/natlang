#!/usr/bin/env python3
"""One screen of health for a pinned GRPO run, for heartbeat checks: `scripts/run_status.py RUNDIR [--unit UNIT]`.

Prints the run unit's state, the last learner step and its age, steps per hour, the latest held results, the pin it runs,
GPU, memory and disk, then one line per problem. Exits 1 when anything is red. Read-only; host python, no repo package
imports (the machine roots come from common/paths.py, loaded by file).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def machine_root(name: str) -> Path:
    """A logical machine root (``common/paths.py``, stdlib-only, loaded by file so host python needs no repo path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'natlang_paths', Path(__file__).resolve().parents[1] / 'training/neuralese/natlang_neuralese/common/paths.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.root(name)


STEP_STALE_H = 2.0       # no learner step for this long while the unit runs: red
DISK_RED = 0.95          # filesystem use fraction


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ''


def unit_props(unit: str) -> dict[str, str]:
    out = sh('systemctl', '--user', 'show', unit, '-p', 'ActiveState,SubState,NRestarts,ActiveEnterTimestamp,'
             'MemoryCurrent,MemoryMax,ExecStart')
    return dict(line.split('=', 1) for line in out.splitlines() if '=' in line)


def gb(value: str) -> str:
    return f'{int(value) / 2**30:.1f}G' if value.isdigit() else '-'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('run', type=Path)
    ap.add_argument('--unit', help='systemd user unit of the run (default: the unit whose command line names RUNDIR)')
    args = ap.parse_args()
    run, red, now = args.run.resolve(), [], time.time()

    unit = args.unit
    if not unit:
        listing = sh('systemctl', '--user', 'list-units', '--type=service', '--state=running', '--no-legend', '--plain')
        for name in (line.split()[0] for line in listing.splitlines() if line.strip()):
            if f'--work {run}' in unit_props(name).get('ExecStart', ''):
                unit = name
                break
    props = unit_props(unit) if unit else {}
    state = f"{props.get('ActiveState', '?')}/{props.get('SubState', '?')}" if unit else 'no running unit names this run'
    print(f'unit     {unit or "-"}  {state}  restarts {props.get("NRestarts", "-")}  since {props.get("ActiveEnterTimestamp", "-")}'
          f'  mem {gb(props.get("MemoryCurrent", ""))}/{gb(props.get("MemoryMax", ""))}')
    running = props.get('ActiveState') == 'active'
    if unit and not running:
        red.append(f'unit {unit} is {state}')
    exec_start = props.get('ExecStart', '')
    pin = next((part for part in exec_start.split() if '/pin-' in part and part.endswith('grpo.py')), '')
    if pin:
        print(f'pin      {pin.split("/scripts/")[0]}')
    stale_pins = [p.name for p in run.glob('pin-*') if p.is_dir() and pin and str(p) not in pin]
    if stale_pins:
        print(f'         stale pins (not in the unit command): {", ".join(sorted(stale_pins))}')

    rows = []
    train = run / 'train.jsonl'
    if train.exists():
        for line in train.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    steps = [r for r in rows if 'step' in r and 'event' not in r]
    if steps:
        last = steps[-1]
        age_h = (now - train.stat().st_mtime) / 3600
        print(f"step     {last['step']}  reward {last.get('reward_mean', 0):.3f}  spread {last.get('spread_fraction', 0):.2f}"
              f"  KL {last.get('kl_mean', 0):.2g}  learner {last.get('learner_tokens_per_s', '-')} tok/s"
              f"  actor {last.get('actor_generated_tokens_per_s', '-')} tok/s  train.jsonl written {age_h:.1f} h ago")
        if running and age_h > STEP_STALE_H:
            red.append(f'no train.jsonl write for {age_h:.1f} h while the unit runs (held eval on the DGX, or stuck)')
    else:
        print('step     none yet')
    starts = [i for i, r in enumerate(rows) if r.get('event') == 'start']
    since = rows[starts[-1]:] if starts else rows
    print(f'launches {len(starts)}  steps since last launch {sum(1 for r in since if "step" in r and "event" not in r)}')

    held = [r for r in rows if r.get('event') == 'held eval']
    for r in held[-3:]:
        rate = r['successes'] / r['scored'] if r.get('scored') else 0
        print(f"held     step {r['step']}: {r.get('successes')}/{r.get('scored')} ({rate:.1%})  loop {r.get('loop_rate')}"
              f"  {r.get('seconds', 0) / 3600:.1f} h  complete {r.get('complete')}")
    for compare in sorted((run / 'held').glob('step-*/compare*.json'))[-2:]:
        try:
            c = json.loads(compare.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        delta = c.get('delta', c.get('mean_task_delta'))
        print(f"compare  {compare.parent.name}: delta {delta}  p {c.get('p_value', c.get('permutation_p'))}")
    in_progress = [d.name for d in (run / 'held').glob('step-*') if d.is_dir() and not (d / 'result.json').exists()]
    if in_progress:
        print(f'         held in progress: {", ".join(sorted(in_progress))}')

    gpu = sh('nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', '--format=csv,noheader')
    mem = {k: int(v.split()[0]) for k, v in (l.split(':', 1) for l in Path('/proc/meminfo').read_text().splitlines())
           if k in ('MemAvailable', 'SwapTotal', 'SwapFree')}
    swap_used = (mem.get('SwapTotal', 0) - mem.get('SwapFree', 0)) / 2**20
    print(f"machine  GPU {gpu or '-'}  MemAvailable {mem.get('MemAvailable', 0) / 2**20:.0f}G  swap used {swap_used:.1f}G")
    for mount in ('/', str(machine_root('data_hdd')), '/dev/shm'):
        try:
            usage = shutil.disk_usage(mount)
        except OSError:
            continue
        frac = usage.used / usage.total
        print(f'disk     {mount:<14} {frac:.0%} used, {usage.free / 2**30:.0f}G free')
        if frac > DISK_RED:
            red.append(f'{mount} is {frac:.0%} full')
    compactor = sh('systemctl', '--user', 'is-active', 'natlang-gb10-compactor')
    if compactor != 'active':
        red.append(f'natlang-gb10-compactor is {compactor or "unknown"} (CUDA contexts fail under fragmentation)')

    for problem in red:
        print(f'RED      {problem}')
    print('OK' if not red else f'{len(red)} problem(s)')
    return 1 if red else 0


if __name__ == '__main__':
    sys.exit(main())
