#!/usr/bin/env python3
"""Assemble the procedural TRAINING-split RL task curriculum from per-family build.mjs outputs.

Inputs are the shards written by ts-host/scripts/inline-curriculum/build.mjs (--split train, hints kept) and their
`.report.json` files. Steps, all recorded in the receipt:

  1. a record failing the collector reference replay is absent from its shard (build.mjs); a group with any such
     failure is dropped whole, and every failure id and reason is kept in the receipt;
  2. a group whose visible-world fingerprints (scripts/replenish_luna_held_shapes.world) all occurred already is a
     duplicate world and is dropped whole (fixed banks saturate), so every kept group is a distinct world;
  2b. a group whose source groups intersect the current held/quality projections or whose ids hit a decision hold
     is dropped whole (dataset-sourced families);
  3. every kept row gets `curriculum.rl = {level, family, basis, score, reason}` from a structural score of the
     reference execution (no live-model qualification is claimed, so `qualification` stays absent: the production
     selector accepts only None or 'qualified', and 'qualified' denotes a live window);
  4. rows are written family by family in generation order.

  assemble_rl_train_curriculum.py --gen-dir DIR --out runs/natlang-rl-train-curriculum-lfm3-20261028
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve  # noqa: E402
from replenish_luna_held_shapes import world  # noqa: E402  (the one visible-world fingerprint)

LEVEL_THRESHOLDS = (6.5, 8.5, 11.0)  # score <= 6.5: level 0, <= 8.5: 1, <= 11: 2, else 3


def visible_chars(sem: dict) -> int:
    return len(json.dumps({k: v for k, v in sem.items() if k not in ('expected', 'expected_files', 'oracle', 'files_oracle')}))


def score(row: dict) -> tuple[float, dict]:
    """Structural work of the reference execution: more required steps, child judgments, loops and evidence = harder."""
    cur = row['curriculum']
    sem = row['semantics']
    ref = cur.get('reference', {})
    children = len(ref.get('children', []))
    parts = {
        'minimum_sequence': len(cur.get('minimum_sequence', [])),
        'root_actions': max(0, len(ref.get('root', [])) - 2),
        'child_calls': min(children, 48) / 6.0,
        'iterate_required': 3.0 if cur.get('iterate') == 'required' else 0.0,
        'inline_required': 1.0 if cur.get('inline') == 'required' else 0.0,
        'services': float(min(len(sem.get('services', {})), 3)),
        'extra_files': float(min(max(0, len(sem.get('files', {})) - 1), 3)),
        'evidence_items': min(len(cur.get('evidence', {}).get('world', [])), 24) / 6.0,
        'folder_files': min(len(sem.get('folder_files', {})), 24) / 6.0,
        # Visible world size (everything the model reads or can read, excluding the gold expectations and oracles).
        'world_size': min(4.0, max(0.0, math.log2(max(1, visible_chars(sem)) / 1000.0))),
        'followup': 1.0 if cur.get('mode') == 'followup' else 0.0,
    }
    return sum(parts.values()), parts


def level_of(total: float, hinted: bool) -> int:
    level = sum(total > t for t in LEVEL_THRESHOLDS)
    # An explicit technique hint in the instructions makes the same world easier to practise.
    return max(0, level - 1) if hinted else level


HELD = resolve('data_nvme', 'e2e-current-protection-bundle-20261015-v1/held-union/full/source-groups/text-data.jsonl')
QUALITY = resolve('data_nvme', 'e2e-current-protection-bundle-20261015-v1/quality-groups.jsonl')


def group_set(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text().splitlines():
        if line.strip():
            out.update(json.loads(line).get('source_groups', []))
    return out


def decision_hold_ids() -> set[str]:
    out: set[str] = set()
    for path in sorted((ROOT / 'training/decision-holds').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip() and isinstance(json.loads(line).get('id'), str):
                out.add(json.loads(line)['id'])
    return out


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen-dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', default='s81028')
    a = ap.parse_args()
    order = [line.split()[0] for name in ('plan.txt', 'plan2.txt') if (a.gen_dir / name).exists()
             for line in (a.gen_dir / name).read_text().splitlines() if line.strip()]
    held_groups, quality_groups = group_set(HELD), group_set(QUALITY)
    hold_ids = decision_hold_ids()
    seen_worlds: set[str] = set()
    rows_out: list[dict] = []
    receipt: dict = {'families': {}, 'dropped': {}}
    ids: set[str] = set()
    for family in order:
        path = a.gen_dir / f'{family}.jsonl'
        if not (a.gen_dir / f'{family}.jsonl.report.json').exists():
            receipt['families'][family] = {'generated_records': 0, 'generated_groups': 0, 'failed_records': 0,
                                           'failure_reasons': [], 'failed_ids': [], 'kept_groups': 0,
                                           'no_output': 'generator crashed or produced no report'}
            continue
        report = json.loads((a.gen_dir / f'{family}.jsonl.report.json').read_text())
        rows = read_rows(path) if path.exists() else []
        failed_ids = {item['id'] for item in report['failures']}
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[row['curriculum']['split_group']].append(row)
        # build.mjs omits failing records from the shard; a family with any failure is handled explicitly, never silently.
        # A failed record leaves its group incomplete; a group with any failed member is dropped whole (the group's
        # shape string sits between colons in every member id, so a shard member of the same shape is matched).
        failure_problems = {item['id']: item['problems'] for item in report['failures']}
        stats = Counter()
        kept_groups = 0
        for group, members in groups.items():
            marker = f":{members[0]['curriculum']['family']}:{members[0]['curriculum']['shape']}:"
            if any(marker in failed for failed in failed_ids):
                stats['groups_dropped_incomplete_after_failure'] += 1
                continue
            # Source-backed groups are dropped whole when any member touches protected held/quality source groups or
            # a decision hold (procedural groups carry only their own s81028 group id).
            if any(set(m.get('source_groups', [])) & (held_groups | quality_groups)
                   or ({m['id']} | set(m.get('source_ids', []))) & hold_ids for m in members):
                stats['groups_dropped_protected_source'] += 1
                continue
            fps = tuple(sorted({world(m) for m in members}))
            if all(fp in seen_worlds for fp in fps):
                stats['groups_dropped_duplicate_world'] += 1
                continue
            if any(fp in seen_worlds for fp in fps):
                stats['groups_partial_world_overlap_kept'] += 1
            seen_worlds.update(fps)
            kept_groups += 1
            for m in members:
                if m['id'] in ids:
                    raise SystemExit(f'duplicate id {m["id"]}')
                ids.add(m['id'])
                total, parts = score(m)
                hinted = bool(m['curriculum'].get('hinted_of'))
                lvl = level_of(total, hinted)
                m['curriculum']['rl'] = {
                    'level': lvl, 'family': m['curriculum']['family'],
                    'basis': 'reference-verified-procedural-train-world',
                    'score': round(total, 3), 'hinted_twin': hinted,
                    'reason': 'Structural score of the exact reference execution (steps, child judgments, iteration, '
                              'evidence); static procedural world, no live-model qualification.'}
                m['generation'] = {**m.get('generation', {}), 'teacher_generation': False,
                                   'method': 'procedural world generation with exact collector oracle; no sampled trajectory',
                                   'rl_curriculum': 'natlang-rl-train-curriculum-lfm3-20261028', 'seed': a.seed}
                rows_out.append(m)
        receipt['families'][family] = {
            'generated_records': len(rows), 'generated_groups': len(groups), 'failed_records': len(failed_ids),
            'failure_reasons': sorted({p for ps in failure_problems.values() for p in ps}),
            'failed_ids': sorted(failed_ids),
            **dict(stats), 'kept_groups': kept_groups}
    a.out.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows_out)
    (a.out / 'cases.jsonl').write_text(text)
    by_family_level = defaultdict(Counter)
    for r in rows_out:
        by_family_level[r['curriculum']['family']][r['curriculum']['rl']['level']] += 1
    receipt.update({
        'cases': len(rows_out), 'cases_sha256': hashlib.sha256(text.encode()).hexdigest(), 'bytes': len(text.encode()),
        'groups': len({r['curriculum']['split_group'] for r in rows_out}),
        'level_counts': dict(sorted(Counter(r['curriculum']['rl']['level'] for r in rows_out).items())),
        'by_family_level': {f: dict(sorted(c.items())) for f, c in by_family_level.items()},
        'level_thresholds': list(LEVEL_THRESHOLDS)})
    (a.out / 'assembly-receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: receipt[k] for k in ('cases', 'groups', 'cases_sha256', 'level_counts')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
