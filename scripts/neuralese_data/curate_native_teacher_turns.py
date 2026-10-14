#!/usr/bin/env python3
"""Curate exact-success teacher turns with explicit parent holds and split proof.

The native materializer's decision approval is necessary but not sufficient:
the caller records any semantic parent holds after reading the trajectories.
This writes an immutable review alongside the selected, successful train turns.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256  # noqa: E402


def rows(path: Path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_immutable(path: Path, body: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f'existing curated artifact differs: {path}')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                     suffix='.building', delete=False) as stream:
        pending = Path(stream.name)
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parents', 'selection-review', 'turns', 'semantic-holds', 'protected', 'out', 'review'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection_review.read_text())
    if selection.get('candidate_sha256') != sha256(args.parents) or selection.get('training_admission') is not False:
        raise ValueError('parent selection review does not bind an unadmitted exact candidate')
    semantic = json.loads(args.semantic_holds.read_text())
    if semantic.get('schema') != 'natlang.teacher-parent-semantic-holds/1' or semantic.get('source_sha256') != sha256(args.parents):
        raise ValueError('semantic holds do not bind the exact parent bytes')
    holds = {item['trajectory_id']: item for item in semantic['holds']}
    if len(holds) != len(semantic['holds']) or any(not item.get('reason') for item in holds.values()):
        raise ValueError('semantic holds need unique trajectory IDs and reasons')
    protected = list(rows(args.protected))
    protected_ids = {row['id'] for row in protected}
    protected_groups = {group for row in protected for group in row.get('source_groups', [])}
    protected_sources = {source for row in protected for source in row.get('source_ids', [])}
    parents = {}
    families = collections.Counter()
    for parent in rows(args.parents):
        program = parent['task']['program_ir']
        if (parent['id'] in parents or program['split'] != 'train' or
                parent['outcome'].get('accepted') is not True or
                parent['outcome'].get('oracle', {}).get('accepted') is not True or
                not parent['outcome'].get('checks') or
                any(value is not True for value in parent['outcome']['checks'].values()) or
                program['id'] in protected_ids or
                protected_groups.intersection(program.get('source_groups', [])) or
                protected_sources.intersection(program.get('source_ids', []))):
            raise ValueError(f'parent is not an exact, source-disjoint train success: {program["id"]}')
        parents[parent['id']] = program
        if parent['id'] in holds:
            if holds[parent['id']]['program_id'] != program['id']:
                raise ValueError(f'semantic hold program mismatch: {parent["id"]}')
        else:
            families[program['family']] += 1
    if set(holds) - set(parents):
        raise ValueError('semantic holds name unknown trajectory IDs')
    counts = collections.Counter()
    target_tools = collections.Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.out.parent,
                                     prefix=args.out.name + '.', suffix='.building', delete=False) as stream:
        pending = Path(stream.name)
        for turn in rows(args.turns):
            counts['materialized'] += 1
            trajectory = turn['teacher_trajectory_id']
            if trajectory not in parents:
                raise ValueError(f'native turn lacks an exact parent: {trajectory}')
            if trajectory in holds:
                counts['semantic_parent_hold'] += 1
                continue
            if turn.get('training_admission', {}).get('approved') is not True:
                counts['unapproved_decision'] += 1
                continue
            if (turn.get('split') != 'train' or turn.get('outcome', {}).get('accepted') is not True or
                    protected_groups.intersection(turn.get('source_groups', []))):
                raise ValueError(f'native turn has invalid outcome or held overlap: {turn["id"]}')
            stream.write(json.dumps(turn, ensure_ascii=False, separators=(',', ':')) + '\n')
            counts['approved'] += 1
            calls = turn.get('target', {}).get('tool_calls') or []
            target_tools[calls[0]['function']['name'] if calls else '<text>'] += 1
        stream.flush()
        os.fsync(stream.fileno())
    if args.out.exists():
        if sha256(args.out) != sha256(pending):
            pending.unlink()
            raise ValueError(f'existing curated output differs: {args.out}')
        pending.unlink()
    else:
        os.replace(pending, args.out)
    review = {'schema': 'natlang.curated-native-teacher-turns/1', 'training_admission': True,
              'source': {'parents': str(args.parents), 'sha256': sha256(args.parents),
                         'selection_review': str(args.selection_review), 'selection_review_sha256': sha256(args.selection_review),
                         'turns': str(args.turns), 'turns_sha256': sha256(args.turns)},
              'semantic_holds': {'path': str(args.semantic_holds), 'sha256': sha256(args.semantic_holds),
                                 'parents': len(holds)},
              'protected': {'path': str(args.protected), 'sha256': sha256(args.protected),
                            'program_overlap': 0, 'source_group_overlap': 0, 'source_id_overlap': 0},
              'parents': {'exact_accepted': len(parents), 'admitted': len(parents) - len(holds),
                          'by_family': dict(sorted(families.items()))},
              'decisions': dict(counts), 'target_tools': dict(sorted(target_tools.items())),
              'output': {'path': str(args.out), 'sha256': sha256(args.out), 'rows': counts['approved']},
              'decision': 'Only successful, native-approved teacher decisions from exact train parents without '
                          'semantic holds enter this corpus. Failed proposals, replay context, and held parents stay out.'}
    write_immutable(args.review, (json.dumps(review, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({'parents': review['parents'], 'decisions': review['decisions'],
                      'output_sha256': review['output']['sha256']}))


if __name__ == '__main__':
    main()
