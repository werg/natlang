#!/usr/bin/env python3
"""Dispatch the asynchronous in-run task evaluations (eval/task_cadence.py ``task_eval.async``; coordinator
2026-10-13): each request a trainer leaves in ``<stage>/task-eval/async/step-N/`` becomes its own resumable ledger unit
(medium priority) that restores the step's weights-only snapshot, runs the held subset and appends its row to the
stage's task-eval.jsonl. The trainer runs in a container without the host's systemd, so launching is this host-side
dispatcher's job.

  python3 scripts/task_eval_dispatch.py --once              # one pass over runs/
  python3 scripts/task_eval_dispatch.py --every 120         # keep dispatching (run it as a small ledger service)

Crisp plumbing (process launch, retry count); where a job runs is the placement policy's decision. CoreWeave sandboxes
do not yet take eval snapshots (a 24 GB snapshot transfer per evaluation is not built), so every job is placed on
this machine's ledger, which admits it when memory allows. A request whose job ended without a row (a hard crash) is
relaunched up to --max-attempts times, then recorded as an ineligible evaluation (the trainer counts it) and its
snapshot deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from natlang_neuralese.common.paths import root as logical_root  # noqa: E402

PATTERNS = ('*/*/task-eval/async/step-*/request.json', '*/*/*/task-eval/async/step-*/request.json')


def requests(runs: Path) -> list[Path]:
    found = set()
    for pattern in PATTERNS:
        found.update(path.parent for path in runs.glob(pattern))
    return sorted(found)


def unit_name(request_dir: Path, request: dict) -> str:
    stage = hashlib.sha256(request['stage_out'].encode()).hexdigest()[:8]
    return f"taskeval-{stage}-{request['step']:07d}"


def unit_state(unit: str) -> str:
    """active | waiting (queued in the ledger) | none."""
    active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit + '.service']).returncode == 0
    if active:
        return 'active'
    try:
        status = json.loads(subprocess.run([sys.executable, str(ROOT / 'scripts' / 'memory_ledger.py'), 'status'],
                                           capture_output=True, text=True, check=True).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return 'unknown'
    waiting = status.get('waiting') or []
    names = set(waiting) if isinstance(waiting, dict) else {w.get('unit') for w in waiting}
    return 'waiting' if unit + '.service' in names else 'none'


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def budget_gb(request_dir: Path) -> float | None:
    """Host + CUDA memory of a job: the BF16 weights, the collector's KV cache and node workers. None while the
    snapshot is still being pruned by the trainer (its size is not known yet)."""
    weights = request_dir / 'weights.pt'
    if not weights.exists():
        full = request_dir / 'full.pt'
        if full.exists() and time.time() - full.stat().st_mtime < 1800:
            return None  # the trainer's pruning child is still writing weights.pt
        return None if not full.exists() else full.stat().st_size / 2**30 / 2 * 1.3 + 8
    return round(weights.stat().st_size / 2**30 * 1.3 + 8, 1)


def launch(request_dir: Path, request: dict, unit: str, budget: float, dry_run: bool) -> dict:
    repo = logical_root('repo')
    stage_out = Path(request['stage_out'])
    log = stage_out / 'task-eval' / 'async-logs' / f"step-{request['step']:07d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    python = repo / '.venv-neuralese' / 'bin' / 'python'
    command = [sys.executable, str(repo / 'scripts' / 'memory_ledger.py'), 'run', '--unit', unit,
               '--budget-gb', str(budget), '--reserve-gb', '2', '--class', 'experiment', '--priority', 'medium',
               '--resumable', '--wait', str(7 * 86400), '--workdir', str(repo), '--',
               'env', 'HF_HUB_OFFLINE=1', f'NATLANG_REPO={repo}', f'PYTHONPATH={repo / "training" / "neuralese"}',
               'sh', '-c', f'exec {python} -m natlang_neuralese.eval.task_cadence --async-request {request_dir} '
                           f'>> {log} 2>&1']
    if dry_run:
        return {'unit': unit, 'command': command}
    with open(log, 'a') as handle:
        process = subprocess.Popen(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    return {'unit': unit, 'launcher_pid': process.pid, 'budget_gb': budget, 'log': str(log)}


def dispatch(runs: Path, *, max_attempts: int = 3, dry_run: bool = False) -> list[dict]:
    from natlang_neuralese.eval.task_cadence import TaskEval
    events = []
    for request_dir in requests(runs):
        try:
            request = json.loads((request_dir / 'request.json').read_text())
        except (OSError, json.JSONDecodeError):
            continue
        unit = unit_name(request_dir, request)
        record_path = request_dir / 'dispatch.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {'attempts': 0}
        state = unit_state(unit)
        if state in ('active', 'waiting', 'unknown') or alive(record.get('launcher_pid')):
            continue
        evaluator = TaskEval(request['spec'], Path(request['stage_out']), None,
                             prompt_rendering=request['prompt_rendering'])
        if evaluator.landed(request['step']) is not None:
            import shutil
            shutil.rmtree(request_dir, ignore_errors=True)  # the row landed; a leftover snapshot goes
            events.append({'event': 'cleaned', 'dir': str(request_dir)})
            continue
        if record['attempts'] >= max_attempts:
            evaluator.record_failure(request['step'], f'asynchronous evaluation job ended without a result '
                                                      f'{record["attempts"]} times (see {record.get("log")})')
            import shutil
            shutil.rmtree(request_dir, ignore_errors=True)
            events.append({'event': 'failed', 'dir': str(request_dir), 'attempts': record['attempts']})
            continue
        budget = budget_gb(request_dir)
        if budget is None:
            continue
        launched = launch(request_dir, request, unit, budget, dry_run)
        record = {**record, **launched, 'attempts': record['attempts'] + 1, 'placement': 'dgx-ledger',
                  'launched_at': time.time()}
        if not dry_run:
            temp = record_path.with_name('dispatch.json.pending')
            temp.write_text(json.dumps(record, indent=1) + '\n')
            temp.replace(record_path)
        events.append({'event': 'launched', 'dir': str(request_dir), **record})
    return events


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=None, help='runs directory (default: the repo\'s runs/)')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--every', type=float, default=120.0)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    runs = args.runs or logical_root('repo') / 'runs'
    while True:
        for event in dispatch(runs, max_attempts=args.max_attempts, dry_run=args.dry_run):
            print(json.dumps(event), flush=True)
        if args.once:
            return 0
        time.sleep(args.every)


if __name__ == '__main__':
    raise SystemExit(main())
