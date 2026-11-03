#!/usr/bin/env python3
"""Effective own-output loss weight of flagged eval spans: the declared CE objective, not the raw row weights.

Input is the flagged JSONL of audit_eval_import_targets.py. Each flagged span's weight is read from
`graph_weights` (runtime-authored opening turns, failed actions and context get zero there).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.jsonio import open_jsonl_text  # noqa: E402
from natlang_neuralese.train.native_supervision import load_policy  # noqa: E402
from natlang_neuralese.train.own_output_supervision import graph_weights, policy_objective  # noqa: E402


def effective(row, objective, spans):
    weights, _ = graph_weights(row, objective)
    offsets, total = [], 0
    for node in row['nodes']:
        offsets.append(total)
        total += len(node['token_ids'])
    return [{'node': node, 'start': start, 'stop': stop, 'effective_tokens': sum(1 for w in weights[offsets[node] + start:offsets[node] + stop] if w > 0),
             'effective_weight': float(sum(weights[offsets[node] + start:offsets[node] + stop]))} for node, start, stop in spans]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flagged', type=Path, action='append', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    objective = policy_objective(load_policy({'path': str(args.policy), 'sha256': args.policy_sha256}))
    wanted = defaultdict(lambda: defaultdict(set))  # leaf -> id -> {(node, start, stop)}
    for path in args.flagged:
        for line in path.open():
            event = json.loads(line)
            if event['kind'] == 'supervised' and (event['syntax_fail'] or any(flag.startswith('start-') for flag in event['flags'])):
                wanted[event['leaf']][event['id']].add((event['node'], event['start'], event['stop']))
    with args.out.open('x') as sink:
        for leaf, ids in wanted.items():
            with open_jsonl_text(leaf) as stream:
                for line in stream:
                    head = line[:600]
                    row_id = next((i for i in ids if i in head), None)
                    if row_id is None:
                        continue
                    row = json.loads(line)
                    if row['id'] != row_id:
                        continue
                    for item in effective(row, objective, sorted(ids[row_id])):
                        sink.write(json.dumps({'leaf': leaf, 'id': row_id, **item}) + '\n')
            sink.flush()


if __name__ == '__main__':
    main()
