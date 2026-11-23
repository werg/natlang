#!/usr/bin/env python3
"""Shared four-arm flagship benchmark harness (plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md §4, A0).

Usage:
    bench.py split --app APP --fixtures PATH           label train/held splits via held-v4 share rule
    bench.py compare ARM_A.json ARM_D.json             paired permutation p + task bootstrap CI + cost/safety delta
    bench.py run --app APP --fixtures PATH --arms A,B,C,D --runner CMD --out DIR

Arms (WS6 §4):
  A: frontier unstructured agent loop (prose task prompt + same services)
  B: the natlang program on the frontier model
  C: the natlang program on base LFM3-1.7B, no compiled cases, no escalation
  D: the natlang program on current policy after specialize/tune + escalate to Arm B's model
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

REPO = Path(__file__).resolve().parents[2]
if str(REPO / 'training' / 'neuralese') not in sys.path:
    sys.path.insert(0, str(REPO / 'training' / 'neuralese'))

from natlang_neuralese.rl.held_eval import reservation_share_held  # noqa: E402

SCHEMA = 'natlang.flagship-bench/1'
FLAGSHIP_SHARE_RULE = {'pattern': r'^flagship:', 'seed': 'held-v4-flagship', 'share': 0.2}
ARMS = ('A', 'B', 'C', 'D')


def split_group_for(app: str, group: str) -> str:
    """Canonical WS6 §1.4 split group: ``flagship:<app>:<group>``."""
    prefix = f'flagship:{app}:'
    return group if group.startswith('flagship:') else f'{prefix}{group}'


def is_held_group(app: str, group: str, rule: dict[str, Any] = FLAGSHIP_SHARE_RULE) -> bool:
    """Deterministic held-v4 reservation check for flagship fixture rows."""
    return bool(reservation_share_held(rule, split_group_for(app, group)))


def label_fixtures(app: str, rows: Iterable[dict[str, Any]], *, rule: dict[str, Any] = FLAGSHIP_SHARE_RULE) -> list[dict[str, Any]]:
    """Attach canonical ``split_group`` and ``split`` ('held' | 'train') to every fixture row."""
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        copy = dict(row)
        raw_group = str(copy.get('split_group') or copy.get('group') or copy.get('id') or f'item-{idx}')
        sg = split_group_for(app, raw_group)
        copy['split_group'] = sg
        copy['split'] = 'held' if is_held_group(app, sg, rule) else 'train'
        out.append(copy)
    return out


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * max(0.0, min(1.0, p))
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return round(ordered[lo], 4)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 4)


def summarize_arm(arm: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item execution records into WS6 §4 arm metrics."""
    n = len(records)
    if n == 0:
        return {
            'arm': arm, 'items': 0, 'pass_rate': 0.0, 'mean_score': 0.0,
            'p50_ms': None, 'p95_ms': None, 'tokens_per_100': 0.0, 'dollars_per_100': 0.0,
            'crisp_share': 0.0, 'local_share': 0.0, 'escalation_share': 0.0,
            'safety': {'unverified_commits': 0, 'injection_escapes': 0, 'schema_violations': 0},
            'records': [],
        }
    passes = [1.0 if r.get('passed') else 0.0 for r in records]
    scores = [float(r.get('score', 1.0 if r.get('passed') else 0.0)) for r in records]
    latencies = [float(r['latency_ms']) for r in records if r.get('latency_ms') is not None]
    total_tokens = sum(float(r.get('tokens', 0.0)) for r in records)
    total_dollars = sum(float(r.get('dollars', 0.0)) for r in records)
    total_units = sum(max(0, int(r.get('units', 1))) for r in records) or n
    crisp_units = sum(float(r.get('crisp_units', 0.0)) for r in records)
    local_units = sum(float(r.get('local_units', 0.0)) for r in records)
    escalated_units = sum(float(r.get('escalated_units', 0.0)) for r in records)
    safety = {
        'unverified_commits': sum(int((r.get('safety') or {}).get('unverified_commits', 0)) for r in records),
        'injection_escapes': sum(int((r.get('safety') or {}).get('injection_escapes', 0)) for r in records),
        'schema_violations': sum(int((r.get('safety') or {}).get('schema_violations', 0)) for r in records),
    }
    return {
        'arm': arm,
        'items': n,
        'pass_rate': round(sum(passes) / n, 6),
        'mean_score': round(sum(scores) / n, 6),
        'p50_ms': percentile(latencies, 0.50),
        'p95_ms': percentile(latencies, 0.95),
        'tokens_per_100': round(total_tokens * 100.0 / total_units, 2),
        'dollars_per_100': round(total_dollars * 100.0 / total_units, 6),
        'crisp_share': round(crisp_units / total_units, 4),
        'local_share': round(local_units / total_units, 4),
        'escalation_share': round(escalated_units / total_units, 4),
        'safety': safety,
        'records': records,
    }


def paired_compare(arm_a: dict[str, Any], arm_b: dict[str, Any], *,
                   permutations: int = 2000, bootstraps: int = 2000, seed: int = 20261123) -> dict[str, Any]:
    """Paired within-item comparison (arm_b - arm_a): delta pass rate, two-sided permutation p, 95% bootstrap CI."""
    by_id_a = {str(r['id']): (1.0 if r.get('passed') else 0.0) for r in arm_a.get('records', [])}
    by_id_b = {str(r['id']): (1.0 if r.get('passed') else 0.0) for r in arm_b.get('records', [])}
    shared = sorted(set(by_id_a) & set(by_id_b))
    if not shared:
        return {'paired_items': 0, 'delta_pass_rate': 0.0, 'p_value': 1.0, 'ci95': [0.0, 0.0],
                'wins': 0, 'losses': 0, 'ties': 0, 'cost_ratio': None}
    diffs = [by_id_b[i] - by_id_a[i] for i in shared]
    n = len(diffs)
    observed = sum(diffs) / n
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        perm = sum(d if rng.getrandbits(1) else -d for d in diffs) / n
        if abs(perm) >= abs(observed) - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (permutations + 1)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(bootstraps))
    lo = boot[int(0.025 * bootstraps)]
    hi = boot[min(bootstraps - 1, int(0.975 * bootstraps))]
    cost_a = float(arm_a.get('dollars_per_100') or 0.0)
    cost_b = float(arm_b.get('dollars_per_100') or 0.0)
    return {
        'paired_items': n,
        'arm_a': arm_a.get('arm'),
        'arm_b': arm_b.get('arm'),
        'delta_pass_rate': round(observed, 6),
        'p_value': round(p_value, 6),
        'ci95': [round(lo, 6), round(hi, 6)],
        'wins': sum(1 for d in diffs if d > 0),
        'losses': sum(1 for d in diffs if d < 0),
        'ties': sum(1 for d in diffs if d == 0),
        'cost_ratio': round(cost_b / cost_a, 6) if cost_a > 0 else None,
        'beats': observed > 0 and p_value < 0.05,
    }


def run_benchmark(app: str, fixtures: list[dict[str, Any]], *,
                  arms: Iterable[str] = ARMS,
                  executor: Callable[[str, dict[str, Any]], dict[str, Any]],
                  split: str = 'held') -> dict[str, Any]:
    """Run requested arms on the app's held split (WS6 §4) and return the structured benchmark report."""
    labeled = label_fixtures(app, fixtures)
    selected = [row for row in labeled if row['split'] == split] if split != 'all' else labeled
    arm_reports: dict[str, Any] = {}
    for arm in arms:
        records: list[dict[str, Any]] = []
        for item in selected:
            t0 = time.perf_counter()
            res = dict(executor(arm, item))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            res.setdefault('id', item.get('id') or item['split_group'])
            res.setdefault('split_group', item['split_group'])
            res.setdefault('latency_ms', round(elapsed_ms, 3))
            records.append(res)
        arm_reports[arm] = summarize_arm(arm, records)
    comparisons: dict[str, Any] = {}
    if 'A' in arm_reports:
        for other in ('B', 'C', 'D'):
            if other in arm_reports:
                comparisons[f'{other}_vs_A'] = paired_compare(arm_reports['A'], arm_reports[other])
    return {
        'schema': SCHEMA,
        'app': app,
        'date': dt.date.today().isoformat(),
        'split': split,
        'total_fixtures': len(labeled),
        'evaluated_items': len(selected),
        'arms': arm_reports,
        'comparisons': comparisons,
    }


def _load_fixtures(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return []
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get('items'), list):
        return data['items']
    raise ValueError(f'Unsupported fixture format in {path}')


def _subprocess_executor(runner: str) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    def _run(arm: str, item: dict[str, Any]) -> dict[str, Any]:
        proc = subprocess.run(
            runner,
            input=json.dumps({'arm': arm, 'item': item}),
            text=True,
            capture_output=True,
            shell=True,
            check=True,
        )
        return json.loads(proc.stdout)
    return _run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Shared four-arm flagship benchmark harness (WS6 A0).')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_split = sub.add_parser('split', help='Label fixture rows with held-v4 flagship split groups.')
    p_split.add_argument('--app', required=True)
    p_split.add_argument('--fixtures', required=True, type=Path)
    p_split.add_argument('--out', type=Path)

    p_cmp = sub.add_parser('compare', help='Paired comparison of two arm JSON files.')
    p_cmp.add_argument('arm_a', type=Path)
    p_cmp.add_argument('arm_b', type=Path)

    p_run = sub.add_parser('run', help='Execute arms on held fixtures via a JSON-stdio runner command.')
    p_run.add_argument('--app', required=True)
    p_run.add_argument('--fixtures', required=True, type=Path)
    p_run.add_argument('--arms', default='A,B,C,D')
    p_run.add_argument('--runner', required=True)
    p_run.add_argument('--split', default='held', choices=('held', 'train', 'all'))
    p_run.add_argument('--out', type=Path)

    args = parser.parse_args(argv)
    if args.cmd == 'split':
        labeled = label_fixtures(args.app, _load_fixtures(args.fixtures))
        held = sum(1 for r in labeled if r['split'] == 'held')
        payload = {'app': args.app, 'total': len(labeled), 'held': held, 'train': len(labeled) - held, 'items': labeled}
        rendered = json.dumps(payload, indent=2) + '\n'
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding='utf-8')
        else:
            print(rendered, end='')
        return 0

    if args.cmd == 'compare':
        a = json.loads(args.arm_a.read_text(encoding='utf-8'))
        b = json.loads(args.arm_b.read_text(encoding='utf-8'))
        print(json.dumps(paired_compare(a, b), indent=2))
        return 0

    arms = [a.strip().upper() for a in args.arms.split(',') if a.strip()]
    report = run_benchmark(args.app, _load_fixtures(args.fixtures), arms=arms,
                           executor=_subprocess_executor(args.runner), split=args.split)
    rendered = json.dumps(report, indent=2) + '\n'
    out = args.out or (REPO / 'applications' / args.app / 'bench' / 'results' / f"{report['date']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding='utf-8')
    print(f'wrote {out} ({report["evaluated_items"]} {report["split"]} items across {", ".join(arms)})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
