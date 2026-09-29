#!/usr/bin/env python3
"""Measure the reducer target on admitted decisions after the final corpus audit."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

DIRECTORY_FAMILIES = {'folder_triage', 'folder_index', 'folder_edit', 'folder_find', 'folder_extract',
                      'folder_mixed', 'directory_target_criteria', 'directory_reducer_apply',
                      'source_workbench', 'source_commitpack', 'source_tatqa', 'source_musique',
                      'source_swesmith', 'source_qasper', 'source_scifact'}


def modality(row):
    if row.get('task_modality'):
        return row['task_modality']
    program = row.get('task', {}).get('program_ir', {})
    if program.get('task_modality'):
        return program['task_modality']
    semantics = program.get('semantics', {})
    root = semantics.get('files', {}).get(semantics.get('root'), '')
    if '\nkind: directory-reducer\n' in root:
        return 'directory-reducer'
    family = row.get('task_family', program.get('family', ''))
    identifier = str(row.get('program_id', ''))
    if any(family == 'curriculum_' + item or identifier.startswith('inline-curriculum:' + item + ':')
           for item in DIRECTORY_FAMILIES):
        return 'directory-reducer'
    if identifier.startswith('inline-curriculum:source_treedst:'):
        return 'tree-edit'
    return 'other'


def audit(paths, target=0.25):
    if not 0 < target < 1:
        raise ValueError('reducer share must be between zero and one')
    counts, families = Counter(), Counter()
    held = 0
    seen = set()
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('split', 'train') != 'train' or row.get('training_admission', {}).get('approved') is False:
                    held += 1
                    continue
                if row['id'] in seen:
                    raise ValueError('duplicate decision id in mix audit: ' + row['id'])
                seen.add(row['id'])
                counts[modality(row)] += 1
                families[row.get('task_family', row.get('family', 'unknown'))] += 1
    total = sum(counts.values())
    reducers = counts['directory-reducer'] + counts['file-reducer']
    deficit = max(0, math.ceil((target * total - reducers) / (1 - target)))
    return {'version': 'natlang.training_mix_audit/1', 'measure': 'unique admitted train decisions',
            'target_reducer_share': target, 'train_decisions': total, 'reducer_decisions': reducers,
            'reducer_share': reducers / total if total else 0, 'by_modality': dict(counts),
            'by_task_family': dict(families), 'heldout_or_unapproved': held,
            'additional_reducer_decisions_needed': deficit,
            'target_met': bool(total) and reducers / total >= target,
            'policy': 'Expand verified supply; do not duplicate decisions or silently downsample other tasks to meet the target.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reducer-share', type=float, default=0.25)
    parser.add_argument('--require-target', action='store_true')
    args = parser.parse_args()
    report = audit(args.input, args.reducer_share)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_suffix(args.output.suffix + '.pending')
    staged.write_text(json.dumps(report, indent=2) + '\n')
    staged.replace(args.output)
    print(json.dumps(report))
    if args.require_target and not report['target_met']:
        raise SystemExit('Reducer mix shortfall: generate at least ' + str(report['additional_reducer_decisions_needed']) + ' more usable reducer decisions.')


if __name__ == '__main__':
    main()
