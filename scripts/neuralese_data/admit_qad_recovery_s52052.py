#!/usr/bin/env python3
"""Curate the pinned train-only QAD student handoff repairs for native SFT.

The source package is immutable evidence.  Only exact successful teacher
decisions are copied; student replay and failed proposals remain context in
those rows and never become targets.  The composed-process repair is held:
its teacher used an arbitrary 10,000-step loop in place of the requested
bounded iterateOn process, despite returning the exact answer in this case.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402

SOURCE = ROOT / 'runs/qad-student-recovery-teacher-20261014-v1'
PARENTS = SOURCE / 'accepted-parents.jsonl'
TURNS = SOURCE / 'native-turns.review-candidates.jsonl'
HELD = Path('/srv/storage/natlang-artifacts/qad-base-vs-15000-matched48-w1-20261014-v1/trained15000/task-eval/subset.jsonl')
OUT = SOURCE / 'approved-turns.jsonl'
REVIEW = SOURCE / 'admission-review.json'
HELD_FAMILY = 'curriculum_composed_process'


def rows(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parents = list(rows(PARENTS))
    held = list(rows(HELD))
    protected_ids = {row['id'] for row in held}
    protected_groups = {group for row in held for group in row.get('source_groups', [])}
    protected_sources = {source for row in held for source in row.get('source_ids', [])}
    admitted = {}
    family_counts = Counter()
    for parent in parents:
        program = parent['task']['program_ir']
        assert parent['outcome']['accepted'] is True and program['split'] == 'train'
        assert program['id'] not in protected_ids
        assert not protected_groups.intersection(program.get('source_groups', []))
        assert not protected_sources.intersection(program.get('source_ids', []))
        if program['family'] == HELD_FAMILY:
            continue
        admitted[parent['id']] = parent
        family_counts[program['family']] += 1
    assert len(parents) == 10 and len(admitted) == 9 and len(family_counts) == 9
    counts = Counter()
    selected = []
    for turn in rows(TURNS):
        counts['all'] += 1
        if turn['teacher_trajectory_id'] not in admitted:
            counts['held_family'] += 1
            continue
        if turn['training_admission'].get('approved') is not True:
            counts['unapproved_decision'] += 1
            continue
        assert turn['split'] == 'train' and turn['outcome']['accepted'] is True
        assert turn['teacher_trajectory_id'] in admitted
        assert not protected_groups.intersection(turn.get('source_groups', []))
        selected.append(turn)
    assert counts['all'] == 95 and len(selected) == 57, (counts, len(selected))
    OUT.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
    review = {
        'schema': 'natlang.qad-student-recovery-admission/1',
        'training_admission': True,
        'source': {'parents': str(PARENTS.relative_to(ROOT)), 'sha256': sha256_file_hex(PARENTS)},
        'materialized': {'turns': str(TURNS.relative_to(ROOT)), 'sha256': sha256_file_hex(TURNS)},
        'held_eval': {'sha256': sha256_file_hex(HELD), 'source_group_overlap': 0, 'source_id_overlap': 0},
        'approved_turns': {'path': str(OUT.relative_to(ROOT)), 'sha256': sha256_file_hex(OUT), 'count': len(selected)},
        'parents': {'exact_accepted': len(parents), 'admitted': len(admitted), 'held': 1,
                    'admitted_families': dict(sorted(family_counts.items()))},
        'decision_counts': dict(counts),
        'decision': 'Nine exact-oracle train-only teacher continuations are admitted. '
                    'Admit only successful native teacher decisions; replayed student prefix and '
                    'failed/unexecuted proposals are excluded as targets. The one composed-process '
                    'continuation is held for violating the requested iterateOn algorithm with an '
                    'arbitrary 10,000-step loop, despite its exact final answer. Preference-pair '
                    'review remains separate; only 7 of 10 raw candidate pairs were causally verified.',
    }
    REVIEW.write_text(json.dumps(review, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'parents': len(admitted), 'turns': len(selected), 'families': len(family_counts),
                      'sha256': sha256_file_hex(OUT)}, sort_keys=True))


if __name__ == '__main__':
    main()
