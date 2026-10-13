#!/usr/bin/env python3
"""Collect exact-success repair parents from individually pinned provider results.

Use for an alternate teacher whose runs span several small queue campaigns.
Only a mechanical, held-disjoint review candidate is written; native-turn and
preference training still require independent quality review.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_luna_accepted_parents import rows, sha256, world


def collect(source, held, paths, model, output):
    if output.exists():
        raise FileExistsError(output)
    source = source.resolve()
    held = held.resolve()
    originals = {row['id']: row for row in rows(source)}
    held_rows = rows(held)
    held_groups = {group for row in held_rows for group in row.get('source_groups', [])}
    held_worlds = {world(row) for row in held_rows}
    seen_ids, seen_worlds, accepted, rejected = set(), set(), [], []
    receipts = []
    for path in paths:
        path = path.resolve()
        result_rows = rows(path)
        if len(result_rows) != 1:
            raise ValueError(f'expected one exact result row: {path}: {len(result_rows)}')
        result = result_rows[0]
        original = result['task']['program_ir']
        if (original['id'] not in originals or originals[original['id']] != original or
                original['id'] in seen_ids):
            raise ValueError(f'provider result source mismatch or duplicate: {path}')
        if result.get('provenance', {}).get('model') != model:
            raise ValueError(f'provider model mismatch: {path}')
        if original.get('split') != 'train' or held_groups.intersection(original.get('source_groups', [])):
            raise ValueError(f'provider result crosses held split: {path}')
        key = world(original)
        if key in held_worlds or key in seen_worlds:
            raise ValueError(f'provider result duplicates held/selected world: {path}')
        outcome = result['outcome']
        exact = (outcome.get('accepted') is True and
                 outcome.get('oracle', {}).get('accepted') is True and
                 outcome.get('checks') and all(value is True for value in outcome['checks'].values()))
        if exact:
            accepted.append(result)
            seen_worlds.add(key)
        else:
            rejected.append({'source_id': original['id'], 'reasons': outcome.get('rejection_reasons', []),
                             'status': outcome.get('status')})
        seen_ids.add(original['id'])
        receipts.append({'path': str(path), 'sha256': sha256(path), 'source_id': original['id'],
                         'exact_accepted': bool(exact)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for parent in accepted:
            stream.write(json.dumps(parent, ensure_ascii=False, separators=(',', ':')) + '\n')
    review = {'schema': 'natlang.repair-provider-parent-review/1', 'training_admission': False,
              'source': str(source), 'source_sha256': sha256(source),
              'held': str(held), 'held_sha256': sha256(held),
              'model': model, 'results': receipts,
              'accepted': len(accepted), 'rejected': rejected,
              'candidate_sha256': sha256(output),
              'decision': 'Exact outcomes and held/source identity only; native decision quality not approved.'}
    output.with_suffix('.review.json').write_text(json.dumps(review, indent=2) + '\n')
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--result', action='append', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.source, args.held, args.result, args.model, args.out)))


if __name__ == '__main__':
    main()
