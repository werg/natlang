#!/usr/bin/env python3
"""Create a scheduling-only RL task-source successor from the admitted v1 set."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/decision-rich-train-20261016-v1'
OUTPUT = ROOT / 'runs/decision-rich-train-20261016-v2-curriculum'
LEVEL_ZERO = {'inline-curriculum:authored-bounded-decisions-v1:decision_skill_catalog:s7001-3:catalog'}
LEVEL_ONE = {
    'inline-curriculum:authored-bounded-decisions-v1:decision_skill_catalog:s7001-0:catalog',
    'inline-curriculum:authored-bounded-decisions-v1:decision_skill_catalog:s7001-1:catalog',
    'inline-curriculum:authored-bounded-decisions-v1:decision_skill_catalog:s7001-2:catalog',
}
LEVEL_TWO = {
    'inline-curriculum:port_record_calls:trajectory_continuation_coding-a200353b9248c54eeac7:iterate-records/1',
    'inline-curriculum:port_record_calls:trajectory_continuation_coding-8dc678c5f1e7463b663e:iterate-records/1',
    'inline-curriculum:port_record_calls:trajectory_continuation_coding-4e3d920afc01e871dea8:iterate-records/1',
    'inline-curriculum:port_record_calls:trajectory_continuation_coding-4c5d87886d7bebdf6ff2:iterate-records/1',
    'inline-curriculum:port_record_calls:trajectory_continuation_research-8472c675b149b593f2e9:iterate-records/1',
    'inline-curriculum:port_record_calls:trajectory_continuation_research-1a5a62c380d7ae7a17e6:iterate-records/1',
    'inline-curriculum:port_record_calls:trajectory_continuation_research-b877f79d0ce4393f8d25:iterate-records/1',
    'inline-curriculum:port_record_calls:trajectory_continuation_research-d95d1403b45a64c4dc86:iterate-records/1',
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(row: dict) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def classify(row: dict) -> tuple[int, str]:
    task_id = row['id']
    if task_id in LEVEL_ZERO:
        objective = row.get('semantics', {}).get('inputs', {}).get('task', {}).get('objective')
        expected = row.get('semantics', {}).get('expected')
        if objective != 'Return the integer 7 supplied explicitly by the caller.' or expected.get('id') != 'none':
            raise ValueError('level-zero task no longer matches its exact no-work oracle')
        curriculum = row.get('curriculum', {})
        if curriculum.get('minimum_sequence') or len(curriculum.get('reference', {}).get('children', [])) != 1:
            raise ValueError('level-zero task acquired extra execution work')
        return 0, 'One exact no-work routing choice: the caller supplies 7 and the oracle chooses none.'
    if task_id in LEVEL_ONE:
        curriculum = row.get('curriculum', {})
        if (row.get('family') != 'curriculum_decision_skill_catalog'
                or len(curriculum.get('reference', {}).get('children', [])) != 1
                or len(curriculum.get('reference', {}).get('root', [])) != 3):
            raise ValueError('level-one skill-routing task changed its one-child oracle shape')
        return 1, 'One exact skill-routing judgment and one bound skill read.'
    if task_id in LEVEL_TWO:
        curriculum = row.get('curriculum', {})
        reference = curriculum.get('reference', {})
        if (row.get('family') != 'curriculum_port_record_calls'
                or len(reference.get('children', [])) != 1
                or len(reference.get('root', [])) != 2
                or len(curriculum.get('evidence', {}).get('retrieved', [])) != 1):
            raise ValueError('level-two port task is not an exact one-record/one-child task')
        return 2, 'One admitted source record, one exact child answer, and one-item iterate/writeback.'
    return 3, 'Retained full-breadth task; no simpler execution claim is made.'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()

    from natlang_neuralese.rl.task_source import resolve_task_source

    source_info = resolve_task_source(source / 'task-source-manifest.json')
    old_manifest = json.loads((source / 'task-source-manifest.json').read_text())
    old_audit = json.loads(source_info['audit'].read_text())
    old_rows = read_rows(source_info['cases'])
    if len(old_rows) != 128:
        raise ValueError('expected the complete reviewed 128-task v1 source')

    assigned = []
    new_rows = []
    levels = Counter()
    for old in old_rows:
        row = copy.deepcopy(old)
        level, reason = classify(row)
        row.setdefault('curriculum', {})['rl'] = {
            'level': level,
            'family': row.get('family', 'unspecified'),
            'scheduling_only': True,
            'reason': reason,
        }
        levels[level] += 1
        assigned.append({'id': row['id'], 'level': level, 'reason': reason,
                         'source_groups': row.get('source_groups', [])})
        # The only changed path is scheduling metadata.
        stripped = copy.deepcopy(row)
        stripped['curriculum'].pop('rl')
        if canonical(stripped) != canonical(old):
            raise ValueError(f'non-scheduling task content changed for {row["id"]}')
        new_rows.append(row)

    expected_low = LEVEL_ZERO | LEVEL_ONE | LEVEL_TWO
    actual_low = {item['id'] for item in assigned if item['level'] < 3}
    if expected_low != actual_low:
        raise ValueError(f'exact level map differs: absent={sorted(expected_low - actual_low)}, unexpected={sorted(actual_low - expected_low)}')
    if len({row['id'] for row in new_rows}) != len(new_rows) or any(row.get('split') != 'train' for row in new_rows):
        raise ValueError('curriculum union contains duplicate identities or non-train rows')

    # Reuse the source-bound current protection sets from the verified v1 audit,
    # checking their exact bytes and the current decision-hold file inventory.
    current = old_audit['current_protection']
    held_path = Path(current['held_groups_path']).resolve(strict=True)
    quality_path = Path(current['quality_groups_path']).resolve(strict=True)
    if sha(held_path) != current['held_groups_sha256'] or sha(quality_path) != current['quality_groups_sha256']:
        raise ValueError('the v1 held/quality protection inputs changed; refresh their closure before building')
    held_groups = {group for row in read_rows(held_path) for group in row.get('source_groups', [])}
    quality_groups = {group for row in read_rows(quality_path) for group in row.get('source_groups', [])}
    hold_dir = ROOT / 'training/decision-holds'
    hold_files = sorted(hold_dir.glob('*.jsonl'))
    hold_ids: set[str] = set()
    hold_bindings = {}
    for hold_file in hold_files:
        hold_bindings[hold_file.name] = sha(hold_file)
        for hold in read_rows(hold_file):
            if isinstance(hold.get('id'), str):
                hold_ids.add(hold['id'])
    row_ids = {row['id'] for row in new_rows}
    source_ids = {source_id for row in new_rows for source_id in row.get('source_ids', [])}
    all_ids = row_ids | source_ids
    all_groups = {group for row in new_rows for group in row.get('source_groups', [])}
    held_intersection = sorted(all_groups & held_groups)
    quality_intersection = sorted(all_groups & quality_groups)
    decision_intersection = sorted(all_ids & hold_ids)
    if held_intersection or quality_intersection or decision_intersection:
        raise ValueError('new task union intersects current held, quality, or decision-hold protection')

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output}')
    case_path = output / 'cases.jsonl'
    with case_path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in new_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    for name in ('broad-selection-proof.json',):
        (output / name).write_bytes((source / name).read_bytes())

    audit = copy.deepcopy(old_audit)
    audit['training_admitted'] = False
    audit['union'] = {
        **audit['union'], 'path': str(case_path), 'sha256': sha(case_path), 'rows': len(new_rows),
    }
    audit['curriculum_schedule'] = {
        'policy': 'explicit scheduling metadata only; exact task inputs, prompts, source identities, executable references and expected outputs are unchanged',
        'level_counts': {str(level): levels[level] for level in sorted(levels)},
        'assignments': assigned,
        'source_manifest': str(source / 'task-source-manifest.json'),
        'source_manifest_sha256': sha(source / 'task-source-manifest.json'),
        'source_cases_sha256': source_info['cases_sha256'],
        'current_decision_hold_files': hold_bindings,
        'current_protection_intersections': {
            'decision_ids': decision_intersection,
            'held_groups': held_intersection,
            'quality_groups': quality_intersection,
        },
        'all_128_original_rows_retained': True,
        'non_scheduling_payloads_exactly_preserved': True,
    }
    audit_path = output / 'union-audit.json'
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    manifest = {
        **old_manifest,
        'id': 'decision-rich-train-20261016-v2-curriculum',
        'cases': 'cases.jsonl',
        'cases_sha256': sha(case_path),
        'source_audit': 'union-audit.json',
        'source_audit_sha256': sha(audit_path),
        'broad_selection_proof': 'broad-selection-proof.json',
        'broad_selection_proof_sha256': sha(output / 'broad-selection-proof.json'),
        'derived_from_task_source': {
            'id': old_manifest['id'],
            'manifest_sha256': sha(source / 'task-source-manifest.json'),
            'cases_sha256': source_info['cases_sha256'],
        },
        'curriculum_levels': {str(level): levels[level] for level in sorted(levels)},
        'curriculum_policy': 'Scheduling-only exact-ID annotations; original 128 train tasks and their full breadth remain intact.',
    }
    manifest_path = output / 'task-source-manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'cases_sha256': sha(case_path), 'rows': len(new_rows),
        'levels': {str(level): levels[level] for level in sorted(levels)},
        'decision_hold_files': hold_bindings,
        'protection_intersections': audit['curriculum_schedule']['current_protection_intersections'],
    }, indent=2))


if __name__ == '__main__':
    main()
