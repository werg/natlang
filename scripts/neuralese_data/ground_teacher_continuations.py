#!/usr/bin/env python3
"""Make an explicit recorded-output extraction task from source imitation IR.

The original source remains historical SFT. This derivative supplies its exact
recorded answer as evidence and asks for that recorded answer; it never grades
an independently chosen future action against an arbitrary source action.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex, sha256_hex
from natlang_neuralese.common.jsonio import iter_jsonl_rows, write_json_durable


CONTRACT = 'natlang.recorded-continuation-extraction/1'


def ground(row):
    if row.get('split') != 'train' or not row.get('source_groups'):
        raise ValueError('teacher extraction requires protected train-only source groups')
    result = copy.deepcopy(row)
    if row.get('family') != 'curriculum_port_record_calls':
        return result, False
    semantics = result['semantics']
    expected = semantics['expected']
    task = json.loads(semantics['folder_files']['task.json'])
    changed = set()
    for path in task['items']:
        key = Path(path).stem
        target = expected[key]
        if not isinstance(target, str):
            raise ValueError('recorded assistant target must be a string')
        item = json.loads(semantics['folder_files'][path])
        item['source_request'] = item['request']
        item['request'] = ('Return the exact recorded_continuation.assistant_message string from the evidence. '
                           'Preserve its characters, line breaks and tool-call formatting.')
        item['evidence'] += '\n\n' + json.dumps({'recorded_continuation': {
            'source_record_id': item['source_record_id'], 'assistant_message': target}}, ensure_ascii=False)
        item['answer_format'] = 'Return only the exact recorded assistant message string.'
        serialized = json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n'
        semantics['folder_files'][path] = serialized
        semantics['expected_files'][path] = serialized
        changed.add(path)
        children = result['curriculum']['reference']['children']
        matches = [child for child in children if child.get('source_binding', {}).get('path') == path]
        if len(matches) != 1:
            raise ValueError('recorded item must have exactly one bound reference child')
        child = matches[0]
        child['source_binding']['sha256'] = sha256_hex(serialized.encode())
        # The existing reference already reads item.evidence then returns the
        # source target; the derivative makes that target uniquely available.
        calls = child['calls']
        if calls[-1] != ['return_result', {'status': 'success', 'value': target}]:
            raise ValueError('recorded child target differs from its exact oracle')
    if len(changed) != len(expected):
        raise ValueError('recorded continuation item/oracle coverage differs')
    result['id'] += ':recorded-extraction-v1'
    result['curriculum']['variant'] += '/recorded-extraction-v1'
    result['teacher_derivation'] = {'schema': CONTRACT, 'source_program_id': row['id'],
        'source_program_sha256': canonical_json_sha256_hex(row), 'items': len(changed),
        'claim': 'exact recorded-output extraction; no independent environment-success claim',
        'training_admission': False}
    result['generation'] = {**result.get('generation', {}), 'target_values_visible_to_model': True,
        'teacher_contract': CONTRACT, 'independent_world_credit': 0}
    return result, True


def prepare(source, output):
    if output.exists():
        raise ValueError('refusing to overwrite a teacher source derivative')
    output.mkdir(parents=True)
    counts = {'rewritten': 0, 'unchanged_other_families': 0}
    destination = output / 'cases.jsonl'
    with destination.open('w') as stream:
        for _, row in iter_jsonl_rows(source):
            derived, changed = ground(row)
            counts['rewritten' if changed else 'unchanged_other_families'] += 1
            stream.write(json.dumps(derived, ensure_ascii=False, separators=(',', ':')) + '\n')
    receipt = {'schema': CONTRACT, 'source': str(source), 'source_sha256': sha256_file_hex(source),
        'cases': {'path': str(destination), 'sha256': sha256_file_hex(destination)}, **counts,
        'protection': 'original train split/source groups/source IDs remain exact; no new source admission',
        'training_admission': False, 'required': 'actual reference replay and live teacher measurement before outcome admission'}
    write_json_durable(output / 'derivation.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out)))


if __name__ == '__main__':
    main()
