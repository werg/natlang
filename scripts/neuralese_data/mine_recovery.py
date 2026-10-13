#!/usr/bin/env python3
"""Recovery supervision source 1: mine natural recoveries from rendered native-turn corpora (selection only).

plans/neuralese/RECOVERY_SUPERVISION.md. Streams ``text-data.jsonl`` files (one pass, bounded memory) and selects
(rendering, document id, assistant-turn index) pairs whose supervised turn follows a no-progress observation and has an
action different from the previous action (classifier: natlang_neuralese.train.recovery_mining, exact, no model).
Supervised turns: program-header and full rows supervise their last assistant turn; whole-trajectory rows supervise
every assistant turn. The manifest holds selections, not copies.

Held-split guard: a train selection from a document that shares a source group with any held (non-train) document of
any scanned file is dropped (counted); held-split selections are kept, marked ``split: test``, and are never train data.

Usage (CPU, under the memory ledger, low priority):
  mine_recovery.py --out-dir DIR --input RENDERING=PATH [--input ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.train.recovery_mining import RULES, parse_turns, recovery_rules  # noqa: E402

SCHEMA = 'natlang.recovery-selection/1'
ALL_TURNS = {'whole-trajectory'}


def family(row: dict) -> str:
    for group in row.get('source_groups') or ():
        if group.startswith('inline-curriculum:'):
            parts = group.split(':')
            if len(parts) > 1:
                return parts[1]
    return str(row.get('corpus') or 'unknown')


def candidates(rendering: str, row: dict):
    """Yield (turn index, rules, action sha) for the supervised turns of one row that are natural recoveries."""
    turns = parse_turns(row['text'])
    indices = range(1, len(turns)) if rendering in ALL_TURNS else ([len(turns) - 1] if len(turns) > 1 else [])
    for index in indices:
        rules = recovery_rules(turns, index)
        if rules:
            yield index, rules, hashlib.sha256(turns[index].action.encode()).hexdigest()[:16]


def scan(inputs: dict[str, Path], spool: Path):
    held_groups: set[str] = set()
    stats = {name: Counter() for name in inputs}
    with open(spool, 'w') as out:
        for name, path in inputs.items():
            rendering = name.split(':')[0]
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    stats[name]['rows'] += 1
                    groups = list(row.get('source_groups') or ())
                    train = row.get('split') == 'train'
                    stats[name]['train_rows' if train else 'held_rows'] += 1
                    if not train:
                        held_groups.update(groups)
                    for index, rules, action_sha in candidates(rendering, row):
                        out.write(json.dumps({'input': name, 'rendering': rendering, 'id': row['id'], 'turn': index,
                                              'rules': rules, 'split': 'train' if train else 'test',
                                              'family': family(row), 'corpus': row.get('corpus'),
                                              'groups': groups, 'action_sha': action_sha}) + '\n')
    return held_groups, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--input', action='append', required=True, metavar='RENDERING[:TAG]=PATH')
    args = ap.parse_args()
    inputs = {}
    for spec in args.input:
        name, _, path = spec.partition('=')
        inputs[name] = Path(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spool = args.out_dir / 'candidates.spool.jsonl'
    held_groups, stats = scan(inputs, spool)
    manifest = args.out_dir / 'recovery-selection.jsonl'
    counts = {'by_rule': Counter(), 'by_family': Counter(), 'by_split': Counter(), 'by_input': Counter(),
              'by_rule_split': Counter(), 'by_rendering_split': Counter()}
    dropped = Counter()
    decisions = defaultdict(set)
    with open(spool) as f, open(manifest, 'w') as out:
        for line in f:
            sel = json.loads(line)
            if sel['split'] == 'train' and held_groups.intersection(sel['groups']):
                dropped[sel['input']] += 1
                continue
            sel.pop('groups')
            out.write(json.dumps(sel) + '\n')
            counts['by_split'][sel['split']] += 1
            counts['by_input'][sel['input']] += 1
            counts['by_rendering_split'][f"{sel['rendering']}/{sel['split']}"] += 1
            counts['by_family'][f"{sel['family']}/{sel['split']}"] += 1
            for rule in sel['rules']:
                counts['by_rule'][rule] += 1
                counts['by_rule_split'][f"{rule}/{sel['split']}"] += 1
            decisions[sel['split']].add(sel['id'].removeprefix('whole-trajectory:'))
    spool.unlink()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary = {'schema': SCHEMA, 'manifest': manifest.name, 'manifest_sha256': digest,
               'inputs': {k: str(v) for k, v in inputs.items()}, 'rules': list(RULES),
               'held_groups': len(held_groups), 'dropped_train_shared_held_group': dict(dropped),
               'distinct_decisions': {k: len(v) for k, v in decisions.items()},
               'rows': {k: dict(v) for k, v in stats.items()},
               **{k: dict(v.most_common()) for k, v in counts.items()}}
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('by_rule', 'by_split', 'by_rendering_split', 'distinct_decisions',
                                              'dropped_train_shared_held_group')}, indent=1))


if __name__ == '__main__':
    main()
