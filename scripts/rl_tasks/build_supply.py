#!/usr/bin/env python3
"""Assemble a GRPO task-supply source from admitted authored batches (plans/neuralese/TASK_SUPPLY_20261118.md).

  build_supply.py OUT_DIR --min-worker 205 [--max-worker N] [--drop-related custody.json]

Rows: train-split rows of batches by workers in [min, max] admitted to natlang-rl-authoring/admitted (admission log) whose
ids are in none of the registry's grpo-pool sources. --drop-related leaves out rows custody_check.py found related to a
protected world. Writes OUT_DIR/cases.jsonl and OUT_DIR/selection.json (inputs with sha256, counts per batch/family) and
records OUT_DIR in the data registry (kind rl-tasks, inputs, producer commit). After its final checks pass, the source
enters training with `scripts/data_registry.py promote ID --role grpo-pool` (a registry commit; the next pin trains it).
Moved into the repository from natlang-rl-authoring/round10 (2026-11-20), where it was the one producer of the
natlang-rl-task-supply-* pools outside git.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO)]
from natlang_neuralese.common import data_registry  # noqa: E402
from natlang_neuralese.common.data_registry import assert_registry_writable, registry_owner_repo  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha  # noqa: E402
from natlang_neuralese.common.paths import root  # noqa: E402

AUTHORING = root('data_hdd') / 'natlang-rl-authoring'
LICENSE = ('project-authored: tasks written by Claude subagents under the natlang authoring briefs, qualified by '
           'reference, negative and alias replay')


def resolve_output_dir(out_arg) -> Path:
    """Resolve OUT_DIR, routing relative ``runs/...`` paths under the canonical registry owner repo,
    guarding against immutable code pins, and refusing non-primary git worktrees."""
    owner = registry_owner_repo()
    raw = Path(out_arg)
    if not raw.is_absolute() and raw.parts and raw.parts[0] == 'runs':
        out = (owner / raw).resolve()
    else:
        out = raw.expanduser().resolve()
    out = assert_registry_writable(out)
    for parent in (out, *out.parents):
        git_marker = parent / '.git'
        if git_marker.is_file():
            raise ValueError(f'refusing to write supply output inside non-primary git worktree {parent}: {out}')
        if git_marker.is_dir():
            break
    return out


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--min-worker', type=int, required=True)
    ap.add_argument('--max-worker', type=int, default=10 ** 6)
    ap.add_argument('--drop-related')
    ap.add_argument('--authoring', default=str(AUTHORING), help='authoring root (admitted/ inside)')
    args = ap.parse_args(argv)
    out = resolve_output_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    admitted = Path(args.authoring) / 'admitted'
    import grpo
    pools = [p for p in grpo.default_pools() if Path(p).resolve() != (out / 'cases.jsonl').resolve()]
    seen = set()
    for p in pools:   # every pool but the one being built
        seen |= {r['id'] for r in rows(p)}
    dropped = set(json.loads(Path(args.drop_related).read_text())['related']) if args.drop_related else set()
    batch_of = {}
    for e in rows(admitted / 'admissions.jsonl'):
        for i in e.get('admitted_case_ids', []):
            batch_of[i] = e['batch']
    picked, per_batch = [], collections.Counter()
    for r in rows(admitted / 'cases.jsonl'):
        b = batch_of.get(r['id'], '')
        try:
            w = int(b.split('/')[0].lstrip('w'))
        except ValueError:
            continue
        if args.min_worker <= w <= args.max_worker and r.get('split') == 'train' and r['id'] not in seen | dropped:
            seen.add(r['id'])
            picked.append(r)
            per_batch[b] += 1
    with open(out / 'cases.jsonl', 'w') as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    inputs = [admitted / 'cases.jsonl', admitted / 'admissions.jsonl', *map(Path, pools)]
    report = {'schema': 'natlang.rl-task-supply-selection/1', 'rows': len(picked), 'workers': [args.min_worker, args.max_worker],
              'batches': dict(sorted(per_batch.items())), 'dropped_protected_related': sorted(dropped),
              'drop_related_from': args.drop_related,
              'families': dict(collections.Counter(r['family'] for r in picked).most_common()),
              'inputs': {str(p): sha(p) for p in inputs}, 'cases_sha256': sha(out / 'cases.jsonl')}
    (out / 'selection.json').write_text(json.dumps(report, indent=1) + '\n')
    record = data_registry.record_dataset(
        data_registry.dataset_id(out, 'rl-tasks'), out, kind='rl-tasks', include=['cases.jsonl', 'selection.json'],
        format='natlang.program/2 task JSONL (cases.jsonl) + selection.json', inputs=[str(p) for p in inputs],
        license=LICENSE, note=f'authored workers {args.min_worker}-{args.max_worker}: {len(picked)} train rows')
    print(json.dumps({k: report[k] for k in ('rows', 'batches')} | {'registry_id': record['id'],
                     'admit': f"scripts/data_registry.py promote {record['id']} --role grpo-pool"}))


if __name__ == '__main__':
    sys.exit(main())
