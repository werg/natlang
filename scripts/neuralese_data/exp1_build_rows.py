#!/usr/bin/env python3
"""Experiment 1 (RFT): the verified successes of a collection as training and validation chains for `exp0_sft.py train`.

Declared settings (recorded in stats.json): only student episodes that succeeded (`reward.quality == 1`), on train tasks,
with NO repeated actions (`reward.execution_efficiency.repeated_actions == 0`: a looping success would teach the loop);
exact duplicate episodes dropped; per-call rows read from the captured sampling (`lower_rollout_turns.episode_turns`);
`--val-fraction` of the split groups (all samples of a group together) held out as the validation set.

    exp1_build_rows.py --rollouts collect/rollouts.jsonl --cases cases.jsonl --out-dir DIR
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
from exp0_sft import chains_of  # noqa: E402
from lower_rollout_turns import episode_turns  # noqa: E402


def group_of(row):
    return ((row.get('curriculum') or {}).get('split_group')) or row['id']


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--rollouts', type=Path, required=True)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--val-fraction', type=float, default=0.1)
    ap.add_argument('--keep-repeated', action='store_true', help='keep successes with repeated actions (report only)')
    args = ap.parse_args()
    groups = {}
    for line in args.cases.read_text().splitlines():
        row = json.loads(line)
        groups[row['id']] = group_of(row)
    stats, episodes, seen = collections.Counter(), [], set()
    with args.rollouts.open('rb') as handle:
        for line in handle:
            row = json.loads(line)
            stats['rollouts'] += 1
            reward = row.get('reward') or {}
            if reward.get('quality') != 1 or row.get('jolt_role', 'student') != 'student':
                continue
            stats['successes'] += 1
            if row.get('split') != 'train' or row.get('unscored') or not row.get('turns'):
                stats['skip_not_train_or_unscored'] += 1
                continue
            repeated = ((reward.get('execution_efficiency') or {}).get('repeated_actions')) or 0
            if repeated:
                stats['successes_with_repeated_actions'] += 1
                if not args.keep_repeated:
                    continue
            identity = (row['task_id'], hashlib.sha256(json.dumps([t.get('target') for t in row['turns']], sort_keys=True).encode()).hexdigest())
            if identity in seen:
                stats['skip_duplicate_episode'] += 1
                continue
            seen.add(identity)
            rows = episode_turns(row, stats)
            if not rows:
                stats['skip_no_usable_calls'] += 1
                continue
            stats['episodes_kept'] += 1
            stats['calls_kept'] += len(rows)
            stats['calls_total'] += len(row['turns'])
            episodes.append({'round': 'exp1', 'task_id': row['task_id'], 'sample': row['sample'], 'family': row['family'],
                             'group': groups[row['task_id']], 'rows': rows})
    held = {g for g in {e['group'] for e in episodes}
            if int(hashlib.sha256(('exp1-val|' + g).encode()).hexdigest(), 16) % 1000 < args.val_fraction * 1000}
    train = [e for e in episodes if e['group'] not in held]
    val = [e for e in episodes if e['group'] in held]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, eps in (('train', train), ('val', val)):
        chains = chains_of(eps)
        with (args.out_dir / f'{name}.pkl').open('wb') as handle:
            pickle.dump({'chains': chains, 'meta': {'episodes': len(eps), 'tasks': len({e['task_id'] for e in eps})}}, handle, protocol=4)
        stats[f'{name}_episodes'] = len(eps)
        stats[f'{name}_tasks'] = len({e['task_id'] for e in eps})
        stats[f'{name}_chains'] = len(chains)
        stats[f'{name}_sequence_tokens'] = int(sum(len(c['ids']) for c in chains))
        stats[f'{name}_loss_tokens'] = int(sum(length for c in chains for _, length, _ in c['segs']))
    (args.out_dir / 'stats.json').write_text(json.dumps(dict(stats), indent=1) + '\n')
    print(json.dumps(dict(stats)))


if __name__ == '__main__':
    main()
