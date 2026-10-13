#!/usr/bin/env python3
"""Choose diverse, same-state repair handoffs from exact replay results.

The source must be build-handoffs.mjs output. Selection does not admit any
teacher answer to training; accepted continuations need native decision review.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def select(source, held, output, *, per_family=3, limit=24):
    if output.exists():
        raise FileExistsError(output)
    held_groups = {group for row in rows(held) for group in row.get('source_groups', [])}
    candidates = list(rows(source))
    if not candidates:
        raise ValueError('no replay-checked handoffs')
    seen_parents, seen_groups = set(), set()
    counts = collections.Counter()
    selected = []
    reasons = collections.Counter()
    # Round-robin across families so the largest failure family cannot fill
    # the packet before smaller families are considered.
    by_family = collections.defaultdict(list)
    for row in candidates:
        handoff = row.get('handoff', {})
        if (row.get('version') != 'natlang.program/2' or row.get('split') != 'train'
                or handoff.get('version') != 'natlang.handoff/2'
                or handoff.get('kind') not in ('failed_action', 'wrong_result')
                or not handoff.get('source', {}).get('trajectory_id')
                or not row.get('source_groups')):
            reasons['invalid_or_nontrain'] += 1
            continue
        if held_groups.intersection(row['source_groups']):
            reasons['held_source_group'] += 1
            continue
        by_family[row['family']].append(row)
    for family in by_family:
        by_family[family].sort(key=lambda row: (row['handoff']['source']['trajectory_id'],
                                                 row['handoff']['source']['index'], row['id']))
    while len(selected) < limit and any(by_family.values()):
        for family in sorted(by_family, key=lambda name: (counts[name], name)):
            if not by_family[family] or len(selected) >= limit:
                continue
            row = by_family[family].pop(0)
            if counts[family] >= per_family:
                reasons['family_cap'] += 1
                by_family[family].clear()
                continue
            parent = row['handoff']['source']['trajectory_id']
            groups = set(row['source_groups'])
            if parent in seen_parents:
                reasons['parent_duplicate'] += 1
                continue
            if groups.intersection(seen_groups):
                reasons['source_group_duplicate'] += 1
                continue
            selected.append(row)
            counts[family] += 1
            seen_parents.add(parent)
            seen_groups.update(groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {'schema': 'natlang.repair-handoff-selection/1',
                'source': str(source.resolve()), 'source_sha256': sha256_file_hex(source),
                'held': str(held.resolve()), 'held_sha256': sha256_file_hex(held),
                'output_sha256': sha256_file_hex(output), 'candidates': len(candidates),
                'selected': len(selected), 'selected_by_family': dict(counts),
                'skipped': dict(reasons), 'per_family': per_family, 'limit': limit,
                'training_admission': False,
                'selection': 'round-robin families; one replay-checked site per failed train trajectory and source group'}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-family', type=int, default=3)
    parser.add_argument('--limit', type=int, default=24)
    args = parser.parse_args()
    if args.per_family < 1 or args.limit < 1:
        parser.error('per-family and limit must be positive')
    print(json.dumps(select(args.source, args.held, args.out,
                            per_family=args.per_family, limit=args.limit)))


if __name__ == '__main__':
    main()
