#!/usr/bin/env python3
"""Select exact greenhouse teachers that actually ran iterateOn steps."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from curate_mellum_recovery_proxy import rows, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--materialized', type=Path)
    args = parser.parse_args()
    source = list(rows(args.source))
    counts = Counter((row['curriculum']['variant'], 'hinted' in row['id']) for row in source)
    if (len(source) != 48 or len({row['id'] for row in source}) != 48 or
            any(row['curriculum']['family'] != 'actor_greenhouse' or
                row['curriculum'].get('family_version') != 3 for row in source) or
            counts != {(variant, hinted): 6 for variant in
                       ('normal', 'sun_arrives', 'long', 'heater_broken') for hinted in (False, True)}):
        raise ValueError('expected 48 verified greenhouse v3 cases, six per variant and hint arm')
    source_by_id = {row['id']: row for row in source}
    held = list(rows(args.held))
    held_ids = {row['id'] for row in held}
    held_groups = {group for row in held for group in row.get('source_groups', [])}
    held_sources = {item for row in held for item in row.get('source_ids', [])}
    seen, selected, decisions = set(), [], []
    for file in sorted(args.campaign.glob('structured-worker-*/case-*/results.jsonl')):
        result_rows = list(rows(file))
        if len(result_rows) != 1:
            raise ValueError(f'{file}: expected one terminal row')
        result = result_rows[0]
        program = result['task']['program_ir']
        ident = program['id']
        if ident not in source_by_id or ident in seen or program != source_by_id[ident]:
            raise ValueError(f'{file}: duplicate or changed source {ident}')
        seen.add(ident)
        if (ident in held_ids or held_groups.intersection(program.get('source_groups', [])) or
                held_sources.intersection(program.get('source_ids', [])) or program.get('split') != 'train'):
            raise ValueError(f'{file}: protected or nontrain source {ident}')
        reasons = []
        if (result['outcome'].get('accepted') is not True or
                result['outcome'].get('oracle', {}).get('level') != 'exact'):
            reasons.append('exact_oracle_or_status')
        if not any(event.get('kind') == 'iteration_step' for event in
                   result['outcome'].get('execution_graph') or []):
            reasons.append('iterateOn_not_executed')
        decisions.append({'id': ident, 'variant': program['curriculum']['variant'],
                          'hinted': 'hinted' in ident, 'approved': not reasons,
                          'reasons': reasons, 'result_sha256': sha(file)})
        if not reasons:
            selected.append(result)
    if seen != set(source_by_id):
        raise ValueError(f'incomplete campaign: {len(seen)}/48 terminal rows')
    selected.sort(key=lambda row: row['task']['program_ir']['id'])
    selected_ids = {row['task']['program_ir']['id'] for row in selected}
    args.out.mkdir(parents=True, exist_ok=True)
    parent_file = args.out / 'accepted-parents.jsonl'
    selected_source = args.out / 'selected.ir.jsonl'
    parent_file.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                                   for row in selected))
    selected_source.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                                       for row in source if row['id'] in selected_ids))
    review = {
        'schema': 'natlang.greenhouse-iteration-selection/1',
        'source': {'path': str(args.source), 'sha256': sha(args.source), 'cases': len(source)},
        'held': {'path': str(args.held), 'sha256': sha(args.held), 'source_group_overlap': 0},
        'parents': {'path': str(parent_file), 'sha256': sha(parent_file), 'approved': len(selected),
                    'by_variant': dict(Counter(row['task']['program_ir']['curriculum']['variant'] for row in selected))},
        'selected_source': {'path': str(selected_source), 'sha256': sha(selected_source)},
        'decisions': decisions,
        'next': 'Materialize native turns, review, render tokenizer twins, then apply global protected filters.',
    }
    if args.materialized:
        accepted_parents = {row['id'] for row in selected}
        approved, turn_counts = [], Counter()
        for turn in rows(args.materialized):
            turn_counts['all'] += 1
            if turn.get('teacher_trajectory_id') not in accepted_parents:
                raise ValueError(f"materialized turn has unselected parent: {turn.get('id')}")
            if turn.get('split') != 'train' or held_groups.intersection(turn.get('source_groups', [])):
                raise ValueError(f"materialized turn crosses held/train boundary: {turn.get('id')}")
            if (turn.get('training_admission', {}).get('approved') is not True or
                    turn.get('outcome', {}).get('accepted') is not True):
                turn_counts['failed_or_unexecuted'] += 1
                continue
            approved.append(turn)
            turn_counts['approved'] += 1
        if {turn['teacher_trajectory_id'] for turn in approved} != accepted_parents:
            raise ValueError('one or more approved parents have no approved native decision')
        out = args.out / 'approved-turns.jsonl'
        out.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                               for row in approved))
        review['materialized'] = {'path': str(args.materialized), 'sha256': sha(args.materialized),
                                  'decision_counts': dict(turn_counts)}
        review['approved_turns'] = {'path': str(out), 'sha256': sha(out), 'count': len(approved)}
        review['next'] = 'Render tokenizer twins and apply global protected filters.'
    (args.out / 'selection-review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'terminal': len(seen), 'approved': len(selected),
                      'by_variant': review['parents']['by_variant']}))


if __name__ == '__main__':
    main()
