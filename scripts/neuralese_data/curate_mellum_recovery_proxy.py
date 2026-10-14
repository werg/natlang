#!/usr/bin/env python3
"""Select exact, process-following train-only recovery trajectories.

This is an exact admission check for the three constructed curriculum families,
not a semantic judge. Native decision materialization and a separate turn review
follow this parent selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


FAMILIES = {
    'relational_dynamic_snapshot',
    'logic_entailment_exception',
    'iterate_frontier',
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def actions(result: dict) -> str:
    return '\n'.join(str(call.get('arguments', {}).get('code', ''))
                     for turn in result.get('trajectory', []) if turn.get('phase') == 'action'
                     for call in turn.get('assistant', {}).get('calls', [])
                     if call.get('tool') == 'eval')


def decide(result: dict) -> list[str]:
    program = result['task']['program_ir']
    family = program['curriculum']['family']
    code = actions(result)
    graph = result['outcome'].get('execution_graph') or []
    problems = []
    if result['outcome'].get('accepted') is not True or result['outcome'].get('oracle', {}).get('level') != 'exact':
        problems.append('exact_oracle')
    if family == 'relational_dynamic_snapshot' and 'changes.pending()' not in code:
        problems.append('live_pending_events_unread')
    if family == 'logic_entailment_exception' and not ('facts.pages()' in code and 'facts.page(' in code):
        problems.append('fact_pages_unread')
    if family == 'iterate_frontier':
        if not any(item.get('kind') == 'iteration_step' for item in graph):
            problems.append('iterateOn_not_executed')
        if 'graph.links(' not in code:
            problems.append('graph_links_unread')
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--held', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--materialized', type=Path,
                        help='review a finished native decision materialization and write approved-turns.jsonl')
    args = parser.parse_args()
    source = list(rows(args.source))
    if len(source) != 144 or Counter(row['curriculum']['family'] for row in source) != {f: 48 for f in FAMILIES}:
        raise ValueError('expected 48 verified fresh cases per recovery family')
    source_by_id = {row['id']: row for row in source}
    if len(source_by_id) != len(source):
        raise ValueError('duplicate source IDs')
    held = list(rows(args.held))
    held_ids = {row['id'] for row in held}
    held_groups = {group for row in held for group in row.get('source_groups', [])}
    held_sources = {item for row in held for item in row.get('source_ids', [])}
    selected = []
    reviewed = []
    seen = set()
    for file in sorted(args.campaign.glob('recovery-v2-worker-*/case-*/results.jsonl')):
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
        problems = decide(result)
        reviewed.append({'id': ident, 'family': program['curriculum']['family'],
                         'variant': program['curriculum']['variant'], 'approved': not problems,
                         'reasons': problems, 'result_sha256': sha(file)})
        if not problems:
            selected.append(result)
    if seen != set(source_by_id):
        raise ValueError(f'incomplete campaign: {len(seen)}/144 terminal rows')
    selected.sort(key=lambda row: row['task']['program_ir']['id'])
    selected_ids = {row['task']['program_ir']['id'] for row in selected}
    args.out.mkdir(parents=True, exist_ok=True)
    parent_file = args.out / 'accepted-parents.jsonl'
    selected_source = args.out / 'selected.ir.jsonl'
    parent_file.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in selected))
    selected_source.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                                       for row in source if row['id'] in selected_ids))
    review = {
        'schema': 'natlang.mellum-recovery-proxy-selection/1',
        'source': {'path': str(args.source), 'sha256': sha(args.source), 'cases': len(source)},
        'held': {'path': str(args.held), 'sha256': sha(args.held), 'source_group_overlap': 0},
        'parents': {'path': str(parent_file), 'sha256': sha(parent_file), 'approved': len(selected),
                    'by_family': dict(Counter(row['task']['program_ir']['curriculum']['family'] for row in selected))},
        'selected_source': {'path': str(selected_source), 'sha256': sha(selected_source)},
        'decisions': reviewed,
        'next': 'Materialize native decisions, review successful turns, render both tokenizer twins, and apply global protected filters.',
    }
    if args.materialized:
        selected_trajectories = {parent['id'] for parent in selected}
        approved = []
        turn_counts = Counter()
        for turn in rows(args.materialized):
            turn_counts['all'] += 1
            if turn.get('teacher_trajectory_id') not in selected_trajectories:
                raise ValueError(f"materialized turn has unselected parent: {turn.get('id')}")
            if turn.get('split') != 'train' or held_groups.intersection(turn.get('source_groups', [])):
                raise ValueError(f"materialized turn crosses held/train boundary: {turn.get('id')}")
            if turn.get('training_admission', {}).get('approved') is not True or turn.get('outcome', {}).get('accepted') is not True:
                turn_counts['failed_or_unexecuted'] += 1
                continue
            approved.append(turn)
            turn_counts['approved'] += 1
        if set(turn['teacher_trajectory_id'] for turn in approved) != selected_trajectories:
            raise ValueError('one or more approved parents have no approved native decision')
        out = args.out / 'approved-turns.jsonl'
        out.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in approved))
        review['materialized'] = {'path': str(args.materialized), 'sha256': sha(args.materialized),
                                  'decision_counts': dict(turn_counts)}
        review['approved_turns'] = {'path': str(out), 'sha256': sha(out), 'count': len(approved)}
        review['next'] = 'Render both tokenizer twins and apply global protected filters in the next union.'
    review_file = args.out / 'selection-review.json'
    review_file.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'terminal': len(seen), 'approved': len(selected),
                      'by_family': dict(Counter(row['task']['program_ir']['curriculum']['family'] for row in selected)),
                      'parents_sha256': sha(parent_file)}))


if __name__ == '__main__':
    main()
