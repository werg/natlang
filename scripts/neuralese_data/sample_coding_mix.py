#!/usr/bin/env python3
"""Diverse, verified sample of generic coding trajectories for the LFM3 coding mix (policy ``coding_mix``).

Owner 2026-10-28: a small coding mix (~10% of joint-SFT supervised tokens) from the SWE-rebench OpenHands records
(harness-bench-swe-rebench-openhands-pi-records-bulk-20261011-v1, pi-replayed native records), rendered only for the
sampled subset. Selection is by whole trajectory (its records render into whole-trajectory chains):
  - train split, outcome resolved, training_admission approved;
  - at most ``--per-repo`` trajectories per repository, chosen by sha256(salt + trajectory id), so the sample spreads
    over all repositories instead of the few large ones; near-duplicates (same instance) count once;
  - the mixture builder later trims the rendered rows to the declared share (round-robin over families).
Pass 1 parses each record; pass 2 copies the selected trajectories' lines unchanged.

  sample_coding_mix.py --records DIR/shards --out runs/lfm3-coding-mix/records.jsonl --per-repo 2
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SALT = 'lfm3-coding-mix-v1:'
TRAJ = re.compile(r'"trajectory_id": "([^"]+)"')


def scan(path: str) -> dict:
    """{trajectory: [repo, instance, ok]}: ok when every record is train, resolved and approved."""
    found = {}
    with open(path, 'rb') as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            ref = row.get('source_ref') or {}
            ok = (row.get('split') == 'train' and (row.get('outcome') or {}).get('resolved') is True
                  and (row.get('training_admission') or {}).get('approved') is True)
            entry = found.setdefault(ref['trajectory_id'], [ref.get('repo'), ref.get('instance_id'), True])
            entry[2] = entry[2] and ok
    return found


def copy_selected(args) -> int:
    path, chosen, out = args
    n = 0
    with open(path, 'rb') as stream, open(out, 'wb') as sink:
        for raw in stream:
            tid = TRAJ.search(raw[:6000].decode('utf-8', 'replace'))
            if tid and tid.group(1) in chosen:
                sink.write(raw)
                n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-repo', type=int, default=2)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    files = sorted(str(f) for f in a.records.glob('*/records.jsonl'))
    trajectories = {}
    with ProcessPoolExecutor(a.workers) as pool:
        for found in pool.map(scan, files):
            for tid, (repo, instance, ok) in found.items():
                prior = trajectories.get(tid)
                trajectories[tid] = [repo, instance, ok and (prior is None or prior[2])]
    by_repo = collections.defaultdict(list)
    for tid, (repo, instance, ok) in trajectories.items():
        if ok:
            by_repo[repo].append((hashlib.sha256((SALT + tid).encode()).hexdigest(), instance, tid))
    chosen = set()
    for repo, items in by_repo.items():
        instances = set()
        for _, instance, tid in sorted(items):
            if instance in instances:
                continue
            instances.add(instance)
            chosen.add(tid)
            if len(instances) >= a.per_repo:
                break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    parts = [str(a.out.parent / f'part-{i:02d}.jsonl') for i in range(len(files))]
    with ProcessPoolExecutor(a.workers) as pool:
        counts = list(pool.map(copy_selected, [(f, chosen, o) for f, o in zip(files, parts)]))
    with open(a.out, 'wb') as sink:
        for part in parts:
            with open(part, 'rb') as stream:
                for raw in stream:
                    sink.write(raw)
            Path(part).unlink()
    summary = {'schema': 'natlang.coding-mix-sample/1', 'records_dir': str(a.records), 'files': len(files),
               'trajectories_seen': len(trajectories),
               'trajectories_eligible': sum(1 for v in trajectories.values() if v[2]),
               'repositories': len(by_repo), 'per_repo': a.per_repo, 'salt': SALT,
               'trajectories_chosen': len(chosen), 'records_written': sum(counts), 'out': str(a.out)}
    a.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
