#!/usr/bin/env python3
"""Pin exact structural quality holds for native teacher decisions.

This is a verifier of authored task requirements against executed actions, not
an oracle for semantic correctness. It emits source-group witnesses that the
native union builder can use as --quarantine-reference. A group-level witness
may also exclude a good sibling; the receipt counts that collateral explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.jsonio import open_jsonl_text


NL_CALL = re.compile(r'\bnl\s*(?:<|\.|`|\()')


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def write_once(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != body:
            raise ValueError(f'immutable quality witness changed: {path}')
        return
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.building', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        if path.read_text() != body:
            raise ValueError(f'immutable quality witness changed concurrently: {path}')
        return
    os.replace(temporary, path)


def action_code(turn: dict) -> str:
    codes = []
    for call in turn.get('target', {}).get('tool_calls') or ():
        try:
            arguments = json.loads(call['function']['arguments'])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(arguments, dict) and isinstance(arguments.get('code'), str):
            codes.append(arguments['code'])
    return '\n'.join(codes)


def hold_reason(family: str, code: str, inline: str | None = None,
                iterate: str | None = None) -> str | None:
    has_nl = bool(NL_CALL.search(code))
    has_iterate = 'iterateOn' in code
    # The task's declared process requirements apply to every curriculum
    # family. Family checks below retain stricter requirements for older
    # source generations whose metadata did not yet express the contract.
    if iterate == 'required' and not has_iterate:
        return 'declared_iterateOn_missing'
    if inline == 'required' and not has_nl:
        return 'declared_inline_nl_missing'
    if family == 'curriculum_composed_process' and not has_iterate:
        return 'required_iterateOn_missing'
    if family == 'curriculum_edit_stream' and (not has_iterate or not has_nl):
        return 'required_stepwise_nl_iterateOn_missing'
    if family == 'curriculum_actor_greenhouse' and not has_iterate:
        return 'required_greenhouse_iterateOn_missing'
    if family == 'curriculum_iterate_schedule_repair' and not has_iterate:
        return 'required_schedule_iterateOn_missing'
    if family == 'curriculum_folder_criteria_reducer' and not has_nl:
        return 'required_folder_nl_criterion_missing'
    if family == 'curriculum_workbench_tickets' and not has_nl:
        return 'required_inline_nl_missing'
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--turns', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()

    parents = {}
    with open_jsonl_text(args.turns) as stream:
        for line in stream:
            if not line.strip():
                continue
            turn = json.loads(line)
            if (turn.get('training_admission', {}).get('approved') is not True or
                    turn.get('split') != 'train'):
                raise ValueError('input must contain approved train decisions only')
            parent_id = turn['teacher_trajectory_id']
            family = turn['task_family']
            # Compact native decisions may omit the source task. Family rules
            # still apply; declared process requirements are available only
            # when the task is carried in this representation.
            curriculum = turn.get('task', {}).get('program_ir', {}).get('curriculum') or {}
            requirements = (curriculum.get('inline'), curriculum.get('iterate'))
            groups = tuple(turn['source_groups'])
            parent = parents.setdefault(parent_id, {'family': family, 'groups': groups,
                                                    'requirements': requirements, 'codes': []})
            if (parent['family'] != family or parent['groups'] != groups or
                    parent['requirements'] != requirements):
                raise ValueError(f'parent source identity changed across decisions: {parent_id}')
            parent['codes'].append(action_code(turn))

    held = {}
    by_reason = Counter()
    all_families = Counter()
    for parent_id, parent in parents.items():
        all_families[parent['family']] += 1
        reason = hold_reason(parent['family'], '\n'.join(parent['codes']), *parent['requirements'])
        if reason:
            held[parent_id] = reason
            by_reason[reason] += 1
    groups = {group for parent_id in held for group in parents[parent_id]['groups']}
    collateral = sorted(parent_id for parent_id, parent in parents.items()
                        if parent_id not in held and groups.intersection(parent['groups']))
    witnesses = [dict(id='source-quality-quarantine:teacher-shape:' + hashlib.sha256(group.encode()).hexdigest()[:20],
                      split='test', source_groups=[group],
                      text='Structural teacher action review: source group held from training; see companion review receipt.')
                 for group in sorted(groups)]
    body = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in witnesses)
    write_once(args.out, body)
    receipt = {
        'schema': 'natlang.teacher-action-shape-quarantine/1',
        'training_admission': False,
        'source': str(args.turns), 'source_sha256': digest(args.turns),
        'quarantine': str(args.out), 'quarantine_sha256': digest(args.out),
        'parents': len(parents), 'held_parents': len(held),
        'held_by_reason': dict(sorted(by_reason.items())),
        'families': dict(sorted(all_families.items())),
        'source_groups': len(groups),
        'collateral_unflagged_parents': len(collateral),
        'collateral_unflagged_family_counts': dict(sorted(Counter(parents[pid]['family'] for pid in collateral).items())),
        'held_parent_ids': dict(sorted(held.items())),
        'collateral_parent_ids': collateral,
        'decision': 'Exclude these source groups from any union using this teacher cohort. Structural checks are '
                    'necessary but do not certify the remaining parents. Inspect the collateral and consider a '
                    'source-level rerender to recover sound siblings.',
    }
    write_once(args.review, json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('parents', 'held_parents', 'held_by_reason',
                                                    'source_groups', 'collateral_unflagged_parents',
                                                    'quarantine_sha256')}, sort_keys=True))


if __name__ == '__main__':
    main()
