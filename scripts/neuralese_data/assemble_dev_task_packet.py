#!/usr/bin/env python3
"""Assemble the protected development task packet from per-family build.mjs outputs.

Selection rule (the held packet's, extended to 8 per shape): per shape, take the first 8 split groups in generation
order and, from the k-th group, the (k mod group size)-th record, so every selected case comes from a distinct world
(group) and no group contributes twice. Groups named by --exclude-group are skipped. Related variants of a group are therefore all on the development side
(unselected variants of selected groups are not used anywhere else).

  assemble_dev_task_packet.py --gen-dir DIR --out runs/natlang-dev-task-packet-v1-20261028 [--per-shape 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SHAPES = ['workbench_tickets', 'edit_stream', 'logic_entailment_exception', 'logic_proof_verifier',
          'relational_multihop_qualifier', 'relational_late_argmax', 'relational_dynamic_snapshot',
          'actor_route_replan', 'actor_greenhouse', 'iterate_schedule_repair', 'iterate_frontier',
          'folder_criteria_reducer', 'composed_process', 'live_inventory', 'idempotent_retry', 'contract_diagnosis']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen-dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--per-shape', type=int, default=8)
    ap.add_argument('--exclude-group', action='append', default=[],
                    help='split_group (or case id) skipped during selection, e.g. after a separation-scan finding')
    a = ap.parse_args()
    rows, index, selection = [], {}, {}
    for shape in SHAPES:
        groups: dict[str, list[dict]] = {}
        for line in (a.gen_dir / f'{shape}.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r['curriculum']['split_group'] in a.exclude_group or r['id'] in a.exclude_group:
                    continue
                groups.setdefault(r['curriculum']['split_group'], []).append(r)
        if len(groups) < a.per_shape:
            raise SystemExit(f'{shape}: only {len(groups)} groups for {a.per_shape} cases')
        chosen = [members[k % len(members)] for k, members in enumerate(list(groups.values())[:a.per_shape])]
        selection[shape] = {'groups_available': len(groups), 'rows': [len(rows) + i for i in range(len(chosen))],
                            'ids': [c['id'] for c in chosen]}
        rows.extend(chosen)
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids)
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'cases.jsonl').write_text(text)
    print(json.dumps({'cases': len(rows), 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'selection': {
        k: {'groups_available': v['groups_available'], 'rows': v['rows']} for k, v in selection.items()}}))
    (a.out / 'selection.json').write_text(json.dumps(selection, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
