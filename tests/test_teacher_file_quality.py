import copy
import json
from pathlib import Path

import pytest
from scripts.neuralese_data import teacher_file_quality as quality


def captured():
    task = {'output_contract': quality.OUTPUT_CONTRACT, 'output_file': 'answers.json',
            'items': ['items/one.json']}
    inputs = {'task.json': json.dumps(task), 'items/one.json': '{"request":"preserve"}'}
    expected = {'one': 'Action: use lamp'}
    return {'id': 'teacher:test', 'task': {'program_ir': {'id': 'program:test',
        'split': 'train', 'source_groups': ['group:test'], 'family': 'curriculum_port_record_calls',
        'semantics': {'expected': expected, 'folder_files': inputs,
            'expected_files': {**inputs, 'answers.json': json.dumps(expected, separators=(',', ':'))},
            'files_oracle': {'compare': 'exact'}}}},
        'outcome': {'status': 'done', 'accepted': False, 'value': expected,
            'files': {**inputs, 'answers.json': json.dumps(expected, indent=2)},
            'files_check': {'accepted': False}, 'oracle': {'accepted': True},
            'checks': {'files': False, 'answer': True, 'effects': True, 'file_return_consistency': True},
            'quality_pending': [], 'rejection_reasons': ['files']},
        'trajectory': [{'raw_model_text': 'unchanged'}], 'provenance': {'actor': 'old-base-teacher'}}


def test_real_comparator_repair_preserves_original_and_all_execution_fields():
    row = captured(); original = copy.deepcopy(row); binding = {'path': '/original', 'sha256': 'a' * 64}
    derived = quality.continue_quality(row, binding)
    assert row == original and derived['outcome']['accepted'] is True
    assert derived['task'] == original['task']
    assert derived['trajectory'] == original['trajectory'] and derived['provenance'] == original['provenance']
    quality.verify_quality_continuation(derived, original, binding)
    derived['trajectory'][0]['raw_model_text'] = 'tampered'
    with pytest.raises(ValueError, match='differs'):
        quality.verify_quality_continuation(derived, original, binding)


@pytest.mark.parametrize('change', ['wrong_answer', 'changed_input', 'extra_file', 'byte_contract', 'held',
                                  'other_check', 'other_rejection', 'quality_pending', 'wrong_gold_file'])
def test_real_verifier_rejects_every_non_format_failure(change):
    row = captured(); semantics = row['task']['program_ir']['semantics']
    if change == 'wrong_answer': row['outcome']['value'] = {'one': 'different'}
    elif change == 'changed_input': row['outcome']['files']['items/one.json'] += ' '
    elif change == 'extra_file': row['outcome']['files']['extra'] = 'new'
    elif change == 'byte_contract':
        task = json.loads(semantics['folder_files']['task.json']); task['output_contract'] = 'Write exact bytes.'
        semantics['folder_files']['task.json'] = json.dumps(task)
    elif change == 'held': row['task']['program_ir']['split'] = 'test'
    elif change == 'other_check': row['outcome']['checks']['answer'] = False
    elif change == 'other_rejection': row['outcome']['rejection_reasons'].append('answer')
    elif change == 'quality_pending': row['outcome']['quality_pending'] = ['review']
    elif change == 'wrong_gold_file': semantics['expected_files']['answers.json'] = '{"one":"different"}'
    with pytest.raises(ValueError): quality.continue_quality(row, {'path': '/original', 'sha256': 'a' * 64})


def test_future_program_derivative_changes_only_file_oracle():
    row = captured()['task']['program_ir']; original = copy.deepcopy(row)
    derived = quality.json_file_program(row)
    assert row == original
    derived['semantics']['files_oracle'] = original['semantics']['files_oracle']
    assert derived == original


def test_wrong_actual_file_answer_is_not_format_equivalence():
    row = captured(); row['outcome']['files']['answers.json'] = '{"one":"wrong action"}'
    with pytest.raises(ValueError, match='contract verification'):
        quality.continue_quality(row, {'path': '/original', 'sha256': 'a' * 64})


def test_actual_converter_reconstructs_view_from_bound_immutable_original(tmp_path):
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import verify_original_row
    original = captured(); source = tmp_path / 'result.jsonl'
    source.write_text(json.dumps(original) + '\n')
    binding = {'path': str(source), 'sha256': quality.sha256_file_hex(source)}
    view = quality.continue_quality(original, binding)
    assert verify_original_row(view, binding) == source
    view['task']['program_ir']['semantics']['expected']['one'] = 'teacher-produced replacement'
    with pytest.raises(ValueError): verify_original_row(view, binding)
    source.write_text(json.dumps(original) + ' \n')
    with pytest.raises(ValueError, match='artifact changed'): verify_original_row(original, binding)


def test_same_comparator_rebuild_preserves_historical_provenance_but_rechecks_admission(monkeypatch):
    original = captured(); binding = {'path': '/original', 'sha256': 'a' * 64}
    view = quality.continue_quality(original, binding)
    canonical = quality.file_verdict
    def rebuilt(row):
        verdict, _ = canonical(row)
        return verdict, {'files': 1, 'inventory_sha256': 'b' * 64}
    monkeypatch.setattr(quality, 'file_verdict', rebuilt)
    quality.verify_quality_continuation(view, original, binding)
    assert view['source_quality_continuation']['verifier']['inventory_sha256'] != 'b' * 64
    view['outcome']['new_unknown_authority'] = True
    with pytest.raises(ValueError, match='differs'):
        quality.verify_quality_continuation(view, original, binding)
    del view['outcome']['new_unknown_authority']
    def rejects(row):
        verdict, identity = rebuilt(row)
        return {**verdict, 'accepted': False}, identity
    monkeypatch.setattr(quality, 'file_verdict', rejects)
    with pytest.raises(ValueError, match='contract verification'):
        quality.verify_quality_continuation(view, original, binding)


@pytest.mark.parametrize('value', [{'files': True, 'inventory_sha256': 'a' * 64},
    {'files': 1, 'inventory_sha256': 'not-a-hash'}, {'files': 1, 'inventory_sha256': 'a' * 64, 'passed': True}])
def test_historical_verifier_must_be_strictly_typed(value):
    original = captured(); binding = {'path': '/original', 'sha256': 'a' * 64}
    view = quality.continue_quality(original, binding)
    view['source_quality_continuation']['verifier'] = value
    with pytest.raises(ValueError, match='historical verifier'):
        quality.verify_quality_continuation(view, original, binding)
