#!/usr/bin/env python3
"""Captured rollouts with text-writer guidance -> ``natlang.guidance-case/1`` records (data/guidance_cases.py).

    build_guidance_cases.py --rollouts ROLLOUTS.jsonl --facts FACTS.json --source NAME [--out-root DIR]

FACTS.json maps each rollout's ``task_id`` to ``{facts: [{id, kind, text, invocation?, match?}], split_groups: [...],
family?}``: the world's verified privileged facts (rl/privileged_context.case_facts, under its custody check) and its
split groups. A rollout's own ``outcome`` field ``{success, evidence}`` (written by the collector from the task oracle) is
required; rollouts without one are omitted, never assumed successful. Writes
OUT/guidance/<source>.guidance-cases.jsonl, <source>.omissions.jsonl and <source>.manifest.json (last), the layout
build_operator_graphs.py renders with ``--cases OUT``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training' / 'neuralese'), str(ROOT)]
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402

OUT_ROOT = data_path('data_hdd', 'natlang-compaction-operators-20261103', 'cases')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--rollouts', type=Path, required=True)
    parser.add_argument('--facts', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    args = parser.parse_args(argv)
    from natlang_neuralese.data.guidance_cases import cases_from_rollout
    from natlang_neuralese.rl.privileged_context import Fact
    worlds = json.loads(args.facts.read_text())
    out = args.out_root / 'guidance'
    out.mkdir(parents=True, exist_ok=True)
    cases_path = out / f'{args.source}.guidance-cases.jsonl'
    omissions_path = out / f'{args.source}.omissions.jsonl'
    counts = {'rollouts': 0, 'cases': 0, 'checked_seeds': 0, 'mechanical_seeds': 0, 'omissions': {}}

    def omit(record):
        key = record['reason'].split(':')[0]
        counts['omissions'][key] = counts['omissions'].get(key, 0) + 1
        omissions.write(json.dumps(record) + '\n')

    with open(cases_path, 'w') as cases, open(omissions_path, 'w') as omissions, open(args.rollouts) as stream:
        for line in stream:
            if not line.strip():
                continue
            rollout = json.loads(line)
            counts['rollouts'] += 1
            world = worlds.get(rollout.get('task_id'))
            if world is None:
                omit({'rollout': rollout.get('id'), 'reason': 'no-world-facts'})
                continue
            if not isinstance(rollout.get('outcome'), dict) or 'success' not in rollout['outcome']:
                omit({'rollout': rollout.get('id'), 'reason': 'no-checked-outcome'})
                continue
            facts = [Fact(item['id'], item['kind'], item['text'], item.get('invocation'), tuple(item.get('match') or ()))
                     for item in world['facts']]
            made, skipped = cases_from_rollout(rollout, facts, outcome=rollout['outcome'],
                                               split_groups=world['split_groups'], family=world.get('family'),
                                               lineage={'source': args.source,
                                                        'rollouts_path': str(args.rollouts)})
            for record in skipped:
                omit({'rollout': rollout.get('id'), **record})
            for case in made:
                cases.write(json.dumps(case, ensure_ascii=False, separators=(',', ':')) + '\n')
                counts['cases'] += 1
                counts['checked_seeds' if case['guidance'] else 'mechanical_seeds'] += 1
    manifest = {'schema': 'natlang.guidance-cases-build/1', 'source': args.source,
                'inputs': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.rollouts, args.facts)},
                'cases': {'path': str(cases_path), 'sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest()},
                'counts': counts, 'admission': 'cases only; graph rendering and training admission are separate'}
    (out / f'{args.source}.manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(counts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
