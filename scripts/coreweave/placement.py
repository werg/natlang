#!/usr/bin/env python3
"""Job placement: the crisp facts and decision, and the bridge to the natural-language policy.

The policy is applications/ops-placement/placeJob.nl (DECOMPOSITION.md there). This module computes exactly what
the policy reads:

- `job_needs`: the job's needs, from its file;
- `fleet_facts`: the DGX memory ledger, this machine's CoreWeave launch records, the settings and the budget;
- `crisp_decision`: the crisp implementation of the same numbered steps.

In `nl` or `shadow` mode it runs `node ts-host/scripts/placement-decide.mjs`, the same subprocess bridge as
storage_retention.py to retention-classify.mjs. `cwctl.py place` writes the served decision into the job's
`placement`, and `cwctl.py launch` refuses a job whose decision is not "coreweave".
"""
from __future__ import annotations

import calendar
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import job as jobs  # noqa: E402

SETTINGS = REPO / 'training' / 'coreweave-placement.json'
PRIORITY = {'train': 'high', 'eval': 'medium', 'smoke': 'medium', 'probe': 'medium', 'render': 'low'}
RANK = {'low': 0, 'medium': 1, 'high': 2}


def settings(path: Path = SETTINGS) -> dict:
    return json.loads(path.read_text())


def job_needs(job: dict, config: dict) -> dict:
    """The JobNeeds of applications/ops-placement/types.ts. ``needs`` in the job file supplies the measured values."""
    needs = job.get('needs', {})
    hours = float(needs.get('hours', job['lifetime_hours']))
    price = config.get('gpu_hour_price_usd')
    cost = jobs.cost_estimate({**job, 'lifetime_hours': hours}, price)['max_total']
    return {'id': job['id'], 'kind': job['kind'],
            'gpu_memory_gb': float(needs.get('gpu_memory_gb', 0 if not job['resources'].get('gpu') else 90)),
            'hours': hours, 'needs_teacher': bool(needs.get('teacher', False)),
            'lineage_owner': needs.get('lineage_owner', 'dgx'),
            'moves_active_lineage': bool(needs.get('moves_active_lineage', False)), 'coreweave_cost_usd': cost}


def dgx_facts(kind: str, ledger_status: dict | None = None) -> dict:
    """Headroom and the queue ahead of a job of this kind's priority, from `memory_ledger.py status`."""
    if ledger_status is None:
        out = subprocess.run(['python3', str(REPO / 'scripts' / 'memory_ledger.py'), 'status'], capture_output=True,
                             text=True, check=True).stdout
        ledger_status = json.loads(out)
    rank = RANK[PRIORITY.get(kind, 'medium')]
    ahead = [w for w in ledger_status.get('waiting', []) if RANK.get(w.get('priority', 'medium'), 1) >= rank]
    return {'dgx_headroom_gb': round(float(ledger_status.get('headroom_gb', 0)), 1), 'dgx_waiting_ahead': len(ahead)}


def coreweave_facts(config: dict, records: list[dict], credential: bool, bucket: bool, now: float | None = None) -> dict:
    """Readiness, active sandboxes and budget from this machine's launch records (cwctl.py sandboxes.jsonl).

    Active = launched within its lifetime and not stopped since. A sandbox that completed early still counts until
    its lifetime ends: an overestimate, which only makes the parallel limit stricter.
    """
    now = time.time() if now is None else now
    stamp = lambda r: calendar.timegm(time.strptime(r['at'], '%Y-%m-%dT%H:%M:%SZ'))
    stopped = {a for r in records if r.get('event') == 'stop' for a in r.get('args', [])}
    launched = [r for r in records if r.get('event') == 'launched']
    active = [r for r in launched if r['sandbox_id'] not in stopped and stamp(r) + r['lifetime_hours'] * 3600 > now]
    week = [r for r in launched if stamp(r) > now - 7 * 86400]
    budget = config.get('weekly_budget_usd')
    spent = sum((r.get('cost') or {}).get('max_total') or 0 for r in week)
    gpu_ok = any(r.get('event') == 'gpu_smoke_ok' for r in records)
    return {'coreweave_ready': bool(credential and bucket and gpu_ok),
            'coreweave_gpu_memory_gb': float(config['gpu_memory_gb']),
            'coreweave_fit_margin_gb': float(config['fit_margin_gb']),
            'coreweave_active': len(active), 'coreweave_max_parallel': int(config['max_parallel']),
            'coreweave_budget_left_usd': None if budget is None else round(budget - spent, 2)}


def crisp_decision(job: dict, fleet: dict) -> dict:
    """The crisp implementation of placeJob.nl's steps (keep the two in the same order)."""
    if job['lineage_owner'] == 'pop':
        return {'where': 'pop', 'reason': 'Pop owns this lineage'}
    if job['moves_active_lineage']:
        return {'where': 'ask', 'reason': 'moving the active lineage of a line is the owner\'s decision (to CoreWeave)'}
    fits_dgx = job['gpu_memory_gb'] <= fleet['dgx_headroom_gb']
    if job['needs_teacher']:
        return {'where': 'dgx' if fits_dgx else 'queue-dgx', 'reason': 'needs the teacher on DGX'}
    if fits_dgx and fleet['dgx_waiting_ahead'] == 0:
        return {'where': 'dgx', 'reason': 'fits the DGX headroom now at no extra cost'}
    limit = fleet['coreweave_gpu_memory_gb'] - fleet['coreweave_fit_margin_gb']
    budget, cost = fleet['coreweave_budget_left_usd'], job['coreweave_cost_usd']
    blockers = [why for bad, why in (
        (not fleet['coreweave_ready'], 'CoreWeave is not ready'),
        (job['gpu_memory_gb'] > limit, f'needs {job["gpu_memory_gb"]} GB, more than one sandbox GPU allows ({limit} GB)'),
        (fleet['coreweave_active'] >= fleet['coreweave_max_parallel'],
         f'{fleet["coreweave_active"]} sandboxes already run (limit {fleet["coreweave_max_parallel"]})'),
        (budget is not None and (cost is None or cost > budget), f'cost {cost} USD against {budget} USD left this week'),
    ) if bad]
    if not blockers:
        price = '' if cost is None else f', at most {cost} USD'
        return {'where': 'coreweave', 'reason': f'DGX cannot take it now (headroom {fleet["dgx_headroom_gb"]} GB, '
                                                f'{fleet["dgx_waiting_ahead"]} waiting ahead){price}'}
    return {'where': 'queue-dgx', 'reason': blockers[0]}


def mode_of(config: dict, override: str | None = None) -> str:
    mode = override or os.environ.get('NATLANG_CW_PLACEMENT') or config.get('mode', 'crisp')
    if mode not in ('crisp', 'nl', 'shadow'):
        raise ValueError(f'placement mode is crisp, nl or shadow, got {mode!r}')
    return mode


def decide(items: list[dict], mode: str, config: dict, model: str | None = None, server: str | None = None,
           shadow_log: Path | None = None) -> list[dict]:
    """Served decisions for items [{id, job, fleet, crisp}]. crisp mode calls no model and no node."""
    if mode == 'crisp':
        return [{'id': item['id'], **item['crisp'], 'served': 'crisp'} for item in items]
    model, server = model or config.get('model'), server or config.get('server')
    if not model:
        raise ValueError('nl and shadow placement need a model: set "model" in training/coreweave-placement.json or --model')
    with tempfile.TemporaryDirectory(prefix='placement-') as tmp:
        source, out = Path(tmp) / 'items.json', Path(tmp) / 'decisions.json'
        source.write_text(json.dumps({'items': items}))
        log = shadow_log or Path.home() / '.local/state/natlang/coreweave/placement-shadow.jsonl'
        command = ['node', str(REPO / 'ts-host' / 'scripts' / 'placement-decide.mjs'), '--input', str(source),
                   '--out', str(out), '--mode', mode, '--shadow-log', str(log), '--server', server, '--model', model]
        subprocess.run(command, check=True, cwd=REPO / 'ts-host')
        return json.loads(out.read_text())['placed']
