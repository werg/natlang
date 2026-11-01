#!/usr/bin/env python3
"""Seal completed RL collection with explicit stale-source and incomplete-group withdrawals.

The source round is immutable. This emits a separate rollouts projection and a proof
consumed by rl.amended_collection; it never edits score, queue, packet, or rollout bytes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import iter_jsonl_rows, read_jsonl, write_bytes_durable, write_json_durable
from natlang_neuralese.rl.task_source import TOOLACE_SCHEMA_CONTRACT, _canonical_row_sha256

REQUIRED = {'collection-binding.json', 'score.json', 'rollouts.jsonl', 'packet.jsonl',
            'queue.jsonl', 'journal.jsonl', 'preparation.json', 'sampling-config.json',
            'selected-cases.jsonl'}


def bind(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex(path), 'bytes': path.stat().st_size}


def build(round_dir: Path, cases_path: Path, task_id: str, output: Path) -> dict:
    round_dir = round_dir.resolve(strict=True)
    cases_path = cases_path.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f'withdrawal output already exists: {output}')
    binding = json.loads((round_dir / 'collection-binding.json').read_text())
    score = json.loads((round_dir / 'score.json').read_text())
    if binding.get('schema') != 'natlang.rl-collection-binding/1':
        raise ValueError('round lacks the immutable RL collection binding')
    if not score.get('collection_complete'):
        raise ValueError('round is not complete; do not fabricate an interrupted or completed collection')
    if sha256_file_hex(cases_path) != binding.get('cases_sha256'):
        raise ValueError('the exact original cases packet bound by the collection is required')
    selected = list(read_jsonl(round_dir / 'selected-cases.jsonl'))
    packet = list(read_jsonl(round_dir / 'packet.jsonl'))
    rows = int(binding['rows']); group_size = int(binding['group'])
    teacher_attempts = binding.get('jolt_teacher_attempts', 0)
    if type(teacher_attempts) is not int or teacher_attempts < 0:
        raise ValueError('collection binding has an invalid privileged teacher attempt count')
    samples_per_group = group_size + teacher_attempts
    if (len(selected) != rows or len(packet) != rows
            or [row.get('id') for row in selected] != [row.get('id') for row in packet]
            or any(json.dumps(a, sort_keys=True, separators=(',', ':')) !=
                   json.dumps(b, sort_keys=True, separators=(',', ':')) for a, b in zip(selected, packet))):
        raise ValueError('selected rows and collector packet are not identical')
    selected_ids = [row.get('id') for row in selected]
    if len(set(selected_ids)) != rows or task_id not in selected_ids:
        raise ValueError('withdrawal target is absent or duplicated in selected rows')
    group_index = selected_ids.index(task_id)
    target = selected[group_index]
    external = ((target.get('curriculum') or {}).get('external') or {})
    if external.get('source') != 'toolace':
        raise ValueError('the requested withdrawal is not a ToolACE task')
    if TOOLACE_SCHEMA_CONTRACT in target.get('source_revisions', []):
        raise ValueError('the requested ToolACE task already carries the current schema contract')
    source_matches = [row for _, row in iter_jsonl_rows(cases_path) if row.get('id') == task_id]
    if len(source_matches) != 1 or _canonical_row_sha256(source_matches[0]) != _canonical_row_sha256(target):
        raise ValueError('selected ToolACE task does not match its unique source row')
    queue = list(read_jsonl(round_dir / 'queue.jsonl'))
    if len(queue) != rows * samples_per_group:
        raise ValueError('round queue is incomplete')
    task_for_group = {}
    files = {name: bind(round_dir / name) for name in REQUIRED}
    result_entries = {}
    for entry in queue:
        group, sample = entry.get('group'), entry.get('sample')
        if type(group) is not int or type(sample) is not int or not (0 <= group < rows and 0 <= sample < samples_per_group):
            raise ValueError('round queue contains an invalid group/sample index')
        if entry.get('index') != group or entry.get('task_id') != selected[group].get('id'):
            raise ValueError('round queue task identities differ from selected rows')
        if ((sample < group_size and entry.get('jolt_role') == 'teacher')
                or (sample >= group_size and entry.get('jolt_role') != 'teacher')):
            raise ValueError('round queue student and privileged teacher roles are inconsistent')
        key = (group, sample)
        if key in result_entries:
            raise ValueError('round queue repeats a sample')
        task_for_group.setdefault(group, entry['task_id'])
        path = Path(entry['output']).resolve(strict=True)
        rel = path.relative_to(round_dir).as_posix()
        files[rel] = bind(path)
        result_entries[key] = entry
    expected_samples = {(g, s) for g in range(rows) for s in range(samples_per_group)}
    if set(result_entries) != expected_samples:
        raise ValueError('round queue does not contain every expected sample')
    groups = score.get('groups')
    if not isinstance(groups, list) or len(groups) != rows:
        raise ValueError('complete score omits selected groups')
    rollout_rows = list(read_jsonl(round_dir / 'rollouts.jsonl'))
    expected_samples = {(g, s) for g in range(rows) for s in range(samples_per_group)}
    if (len(rollout_rows) != len(expected_samples)
            or len({(r.get('group'), r.get('sample')) for r in rollout_rows}) != len(rollout_rows)
            or {(r.get('group'), r.get('sample')) for r in rollout_rows} != expected_samples
            or score.get('rollouts') != len(rollout_rows)
            or score.get('scored') != sum(r.get('reward') is not None for r in rollout_rows)):
        raise ValueError('completed score totals do not match the exact rollout rows')
    rollout_by_sample = {(r.get('group'), r.get('sample')): r for r in rollout_rows}
    for group in range(rows):
        if groups[group].get('task_id') != selected[group].get('id'):
            raise ValueError('score group identity differs from selected packet')
        student_rows = [rollout_by_sample[(group, sample)] for sample in range(group_size)]
        if sum(row.get('reward') is not None for row in student_rows) != groups[group].get('scored'):
            raise ValueError('group score differs from its student rollout rows')
        outcomes = groups[group].get('outcomes')
        if (not isinstance(outcomes, list) or len(outcomes) != group_size
                or Counter(student_rows[sample].get('outcome') for sample in range(group_size))
                   != Counter(outcomes)):
            raise ValueError('group outcome labels differ from exact student rollout rows')
    if score.get('block_bundle'):
        bundle = Path(score['block_bundle']).resolve(strict=True)
        if bundle != (round_dir / 'blocks/manifest.json').resolve(strict=True):
            raise ValueError('score names an unexpected native bundle manifest')
        files['blocks/manifest.json'] = bind(bundle)
    # Retain only fully scored student groups. An unresolved execution is
    # preserved as custody evidence and withdrawn as unknown, never as negative.
    from natlang_neuralese.rl.task_source import validate_current_materialized_task_contract
    kept = []
    withdrawals = [{'group': group_index, 'task_id': task_id,
        'row_sha256': _canonical_row_sha256(target), 'reason': 'stale-toolace-schema-contract',
        'required_schema_contract': TOOLACE_SCHEMA_CONTRACT}]
    journal = list(read_jsonl(round_dir / 'journal.jsonl'))
    rollout_rows = [json.loads(line) for line in (round_dir / 'rollouts.jsonl').read_text().splitlines() if line.strip()]
    for group in range(rows):
        if group == group_index:
            continue
        case = selected[group]
        if groups[group].get('scored') == group_size:
            if any(rollout_by_sample[(group, sample)].get('outcome') == 'unresolved'
                   for sample in range(group_size)):
                raise ValueError('fully scored group contains an unresolved student rollout')
            if any((rollout_by_sample[(group, sample)].get('jolt_role') != 'teacher'
                    or rollout_by_sample[(group, sample)].get('outcome') != 'task_success'
                    or (rollout_by_sample[(group, sample)].get('reward') or {}).get('quality') != 1
                    or not rollout_by_sample[(group, sample)].get('turns'))
                   for sample in range(group_size, samples_per_group)):
                raise ValueError('retained privileged teacher row is not a successful bound trajectory')
            validate_current_materialized_task_contract(case)
            kept.append(group)
            continue
        if (type(groups[group].get('scored')) is not int or groups[group]['scored'] >= group_size
                or groups[group].get('outcomes', []).count('unresolved') != 1
                or len(groups[group].get('outcomes', [])) != group_size
                or sum(outcome != 'unresolved' for outcome in groups[group].get('outcomes', [])) != groups[group]['scored']):
            raise ValueError('partly scored group lacks an explicit unresolved outcome')
        unresolved = []
        for sample in range(group_size):
            rollout = next((row for row in rollout_rows if row.get('group') == group and row.get('sample') == sample), None)
            if rollout is None or rollout.get('outcome') != 'unresolved':
                continue
            entry = result_entries[(group, sample)]
            finished = [event for event in journal if event.get('event') == 'finish' and event.get('key') == entry.get('key')]
            if len(finished) != 1:
                raise ValueError('unresolved sample does not have one terminal journal record')
            error_states = [state.get('terminal_error') for state in
                ((finished[0].get('output_accounting') or {}).get('job_states') or []) if state.get('terminal_error')]
            if len(error_states) != 1:
                raise ValueError('unresolved sample does not have one exact collector error')
            error = error_states[0]
            error_path = Path(error.get('path', '')).resolve(strict=True)
            try:
                error_rel = error_path.relative_to(round_dir).as_posix()
            except ValueError as exc:
                raise ValueError('unresolved error artifact escapes the source collection') from exc
            error_ref = bind(error_path);files[error_rel] = error_ref
            error_row = json.loads(error_path.read_text())
            if (error_ref['sha256'] != error.get('sha256') or error_row.get('error') != error.get('error')
                    or error_row.get('source_sha256') != binding.get('cases_sha256')
                    or error_row.get('program_ir_sha256') != error.get('program_ir_sha256')
                    or error.get('source_sha256') != binding.get('cases_sha256')
                    or error.get('record_binding_verified') is not True
                    or 'neuralese-native-grad-invalid' not in error.get('error', '')
                    or Path(entry['output']).stat().st_size != 0 or rollout.get('reward') is not None
                    or rollout.get('turns') or rollout.get('unscored') != 'unresolved'):
                raise ValueError('unresolved sample does not match the exact native custody error')
            unresolved.append({'sample': sample, 'error_path': error_rel, 'error_sha256': error_ref['sha256'],
                'error': error.get('error'), 'program_ir_sha256': error.get('program_ir_sha256'),
                'source_sha256': error.get('source_sha256'), 'record_binding_verified': True})
        if not unresolved:
            raise ValueError('partly scored group has no exact unscored execution record')
        withdrawals.append({'group': group, 'task_id': case['id'], 'row_sha256': _canonical_row_sha256(case),
            'reason': 'incomplete-native-execution-group', 'unresolved_samples': unresolved})
    withdrawals.sort(key=lambda item: item['group'])
    kept.sort()
    if set(kept) | {item['group'] for item in withdrawals} != set(range(rows)):
        raise ValueError('selected groups are not fully partitioned into retained/withdrawn groups')
    rollouts = (round_dir / 'rollouts.jsonl').read_bytes().splitlines(keepends=True)
    kept_lines = []
    observed = set()
    for raw in rollouts:
        if not raw.strip():
            continue
        row = json.loads(raw)
        key = (row.get('group'), row.get('sample'))
        if key not in expected_samples or row.get('task_id') != selected[key[0]].get('id'):
            raise ValueError('rollout row does not bind to a selected queue sample')
        if key in observed:
            raise ValueError('rollout rows repeat a sample')
        observed.add(key)
        if key[0] in kept:
            kept_lines.append(raw if raw.endswith(b'\n') else raw + b'\n')
    if observed != expected_samples:
        raise ValueError('complete rollout file omits one or more scored samples')
    output.mkdir(parents=True)
    projected = output / 'retained-rollouts.jsonl'
    write_bytes_durable(projected, b''.join(kept_lines))
    proof = {'schema': 'natlang.completed-collection-withdrawal/1', 'out': str(round_dir),
        'groups': kept, 'withdrawals': withdrawals, 'files': files,
        # The exact selected source packet is what collection-binding hashes.
        # Preserve this historical packet separately from the corrected source
        # that will be used by the next collection.
        'source_cases': bind(round_dir / 'selected-cases.jsonl'), 'filtered_rollouts': bind(projected)}
    proof_path = output / 'proof.json'
    write_json_durable(proof_path, proof)
    return {'schema': 'natlang.completed-toolace-withdrawal-build/1', 'proof': bind(proof_path),
        'projection': bind(projected), 'round': str(round_dir), 'round_score_sha256': files['score.json']['sha256'],
        'withdrawn': withdrawals, 'retained_groups': kept, 'retained_rollouts': len(kept_lines),
        'collection_report_preserved': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    receipt = build(args.round_dir, args.cases, args.task_id, args.out)
    write_json_durable(args.out / 'receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
