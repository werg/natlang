#!/usr/bin/env python3
"""Exact decontamination verifier: native training rows against the held task packet's visible worlds.

A training row collides when it is related to a held case under the shared world custody
(common.custody.world_identities: ids, source ids, source/split/pair groups, world bindings, the rendered-world and
visible-world fingerprints), or names the held seed. Prints one JSON receipt; exit 1 when anything collides.

  check_held_world_collisions.py --held runs/natlang-held-task-packet-v3-20261012/cases.jsonl SOURCE.jsonl [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.custody import ProtectedWorlds, world_identities  # noqa: E402


def ir_of(row: dict) -> dict | None:
    task = row.get('task') or {}
    ir = task.get('program_ir') if isinstance(task, dict) else None
    if isinstance(ir, dict):
        return ir
    return row if 'semantics' in row and 'family' in row else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--held', required=True, type=Path)
    p.add_argument('--held-seed', default='s61012')
    p.add_argument('sources', nargs='+', type=Path)
    a = p.parse_args()
    held = [json.loads(s) for s in a.held.read_text().splitlines() if s.strip()]
    protected = ProtectedWorlds.from_cases({str(a.held): ('', held)})
    receipt = {'held': str(a.held), 'held_cases': len(held), 'sources': {}}
    bad = 0
    for src in a.sources:
        stats = {'rows': 0, 'programs': 0, 'no_program_ir': 0, 'collisions': [], 'held_seed_mentions': 0}
        seen = set()
        with src.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                stats['rows'] += 1
                if a.held_seed in line:
                    stats['held_seed_mentions'] += 1
                ir = ir_of(row)
                if ir is None:
                    stats['no_program_ir'] += 1
                    continue
                pid = ir.get('id') or json.dumps(ir, sort_keys=True)[:200]
                if pid in seen:
                    continue
                seen.add(pid)
                stats['programs'] += 1
                shared = world_identities(ir, evidence_items=False) & protected.identities
                if shared:
                    stats['collisions'].append([pid, sorted(map(list, shared))])
        bad += len(stats['collisions']) + stats['held_seed_mentions']
        receipt['sources'][str(src)] = stats
    receipt['collisions'] = bad
    print(json.dumps(receipt, indent=1))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
