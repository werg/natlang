#!/usr/bin/env python3
"""Reproduce exact protected/held exclusions from the bounded v3 assembly sample."""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from natlang_neuralese.common.custody import ProtectedWorlds, world_identities


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20261031)
    ap.add_argument('--cap-override', default='codebase-py=7725,codebase-natlang=1432')
    a = ap.parse_args()
    caps = {k: int(v) for k, v in (item.split('=') for item in a.cap_override.split(',') if item)}
    files = sorted(a.build.glob('*/verified/cases.jsonl'))
    by_source = {p.parent.parent.name: read(p) for p in files}
    held = [r for rows in by_source.values() for r in rows if r.get('split') in ('dev', 'confirmation')]
    own_held = ProtectedWorlds.from_cases({'v3-held': ('pending', held)})
    protected = ProtectedWorlds.load()
    blocked = protected.identities | own_held.identities
    held_index = collections.defaultdict(list)
    for row in held:
        for identity in world_identities(row, evidence_items=False):
            held_index[identity].append({'id': row['id'], 'split': row['split'], 'source': row['source']})
    records = []
    counts = collections.Counter()
    for source, cases in sorted(by_source.items()):
        selected = list(cases)
        random.Random(a.seed).shuffle(selected)
        cap = caps.get(source, 4000 if source == 'reasoning-gym' else 1500)
        selected = [r for r in selected if r.get('split') == 'train'][:cap]
        for row in selected:
            intersections = world_identities(row, evidence_items=False) & blocked
            if not intersections:
                continue
            kinds = sorted({kind for kind, _ in intersections})
            held_matches = {}
            for identity in intersections:
                for match in held_index.get(identity, []):
                    held_matches[match['id']] = match
            rec = {'source_package': source, 'case_id': row['id'], 'source_groups': row.get('source_groups', []),
                   'reason_kinds': kinds, 'matched_identities': [list(x) for x in sorted(intersections)],
                   'related_v3_held_cases': sorted(held_matches.values(), key=lambda x: x['id'])}
            records.append(rec)
            counts[(source, ','.join(kinds))] += 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    summary = {'rows': len(records), 'by_source_reason': {f'{s}:{k}': n for (s, k), n in counts.items()},
               'output': str(a.out)}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
