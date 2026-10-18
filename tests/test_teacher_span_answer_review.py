import copy
import json

import pytest
from scripts.neuralese_data import teacher_file_quality as quality


def binding(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': quality.sha256_file_hex(path)}


def write(path, value):
    path.write_text(json.dumps(value) + '\n')
    return binding(path)


def fixture(tmp_path, mutate=None):
    checks = {key: True for key in ('seeded_failure_requirement', 'expected_status',
        'honest_stop', 'effects', 'files', 'world', 'file_return_consistency')}
    checks['answer'] = False
    row = {'version': 'natlang.teacher_trajectory.native/1', 'id': 'teacher:review',
        'task': {'program_ir': {'version': 'natlang.program/2', 'id': 'qa:one',
            'split': 'train', 'family': 'curriculum_web_research',
            'source_groups': ['train:question-one'], 'gold_sources': [],
            'semantics': {'inputs': {'question': 'Which county?'},
                'expected': 'Anderson County', 'files': {'research.nl': 'Return the county.'},
                'oracle': {'level': 'span', 'normalization': 'qa', 'threshold': .8,
                    'context': {'question': 'Which county?'},
                    'rubric': 'Accept the same entity in another phrasing.'}}}},
        'outcome': {'status': 'done', 'accepted': False,
            'value': 'Anderson County, Tennessee (the Coal Creek region)',
            'files': {}, 'effects': {}, 'checks': checks,
            'oracle': {'accepted': False, 'level': 'span', 'score': .5,
                'needs_review': True, 'verdict': 'answer equivalence needs independent review'},
            'quality_pending': ['answer_needs_review'], 'rejection_reasons': ['answer']},
        'trajectory': [{'captured_response': 'unchanged complete context'}]}
    if mutate:
        mutate(row)
    original = write(tmp_path / 'raw.jsonl', row)
    program = row['task']['program_ir']; sem = program['semantics']; out = row['outcome']
    record = {'key': 'one', 'raw_output': original, 'source_row_id': row['id'],
        'program_id': program['id'], 'program_ir_sha256': quality.canonical_json_sha256_hex(program),
        'family': program['family'], 'source_groups': program['source_groups'],
        'source_gold': program.get('gold_sources'), 'question': sem['inputs']['question'],
        'expected': sem['expected'], 'declared_oracle': sem['oracle'],
        'declared_instruction_files': sem['files'], 'captured_actual': out['value'],
        'captured_oracle': out['oracle'], 'captured_checks': out['checks'],
        'captured_file_check': out.get('files_check'), 'training_admission': False}
    inventory = {'schema': 'natlang.source-owned-answer-equivalence-review-inventory/1',
        'training_admission': False, 'parent_release': False, 'records': [record]}
    inventory_ref = write(tmp_path / 'inventory.json', inventory)
    review = {'schema': 'natlang.teacher-span-answer-semantic-review/1',
        'reviewer': 'independent-root-source-meaning-review', 'training_admission': False,
        'reason_verbatim': 'The captured answer names the same county.', 'inventory': inventory_ref,
        'approved': [{'row_id': row['id'], 'raw_output_sha256': original['sha256'],
            'program_ir_sha256': record['program_ir_sha256'], 'accepted': True,
            'reason': 'The captured answer names the same county.'}]}
    return row, original, review, write(tmp_path / 'review.json', review)


def test_real_span_oracle_and_actual_converter_preserve_complete_capture(tmp_path):
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import verify_original_row
    row, raw, _, authority = fixture(tmp_path); before = copy.deepcopy(row)
    view = quality.continue_span_answer_quality(row, raw, review_reference=authority)
    assert row == before
    assert view['outcome']['oracle'] == {'accepted': True, 'level': 'span', 'score': .5,
        'verdict': 'The captured answer names the same county.'}
    assert view['outcome']['quality_pending'] == []
    assert view['trajectory'] == row['trajectory']
    assert view['task'] == row['task']
    assert view['outcome']['value'] == row['outcome']['value']
    assert verify_original_row(view, raw).as_posix() == raw['path']
    view['trajectory'][0]['captured_response'] = 'rewritten context'
    with pytest.raises(ValueError, match='differs from exact original evaluation view'):
        verify_original_row(view, raw)


@pytest.mark.parametrize('field', ['files', 'effects', 'world', 'honest_stop', 'file_return_consistency'])
def test_review_cannot_override_nonanswer_failure(tmp_path, field):
    row, raw, _, authority = fixture(tmp_path,
        lambda row: row['outcome']['checks'].__setitem__(field, False))
    with pytest.raises(ValueError, match='only a completed answer-equivalence review'):
        quality.continue_span_answer_quality(row, raw, review_reference=authority)


@pytest.mark.parametrize('change', ['held', 'quiesced', 'exact', 'pending', 'already_derived'])
def test_review_cannot_change_eligibility(tmp_path, change):
    def mutate(row):
        if change == 'held': row['task']['program_ir']['split'] = 'test'
        elif change == 'quiesced': row['outcome']['status'] = 'quiesced'
        elif change == 'exact': row['task']['program_ir']['semantics']['oracle']['level'] = 'exact'
        elif change == 'pending': row['outcome']['quality_pending'].append('files')
        else: row['source_quality_continuation'] = {'schema': quality.SPAN_REVIEW_SCHEMA}
    row, raw, _, authority = fixture(tmp_path, mutate)
    with pytest.raises(ValueError):
        quality.continue_span_answer_quality(row, raw, review_reference=authority)


@pytest.mark.parametrize('change', ['answer', 'question', 'reference', 'rubric', 'program', 'groups', 'trajectory'])
def test_changed_capture_rejected_even_with_authentic_raw_binding(tmp_path, change):
    row, raw, _, authority = fixture(tmp_path)
    sem = row['task']['program_ir']['semantics']
    if change == 'answer': row['outcome']['value'] = 'A different county'
    elif change == 'question': sem['inputs']['question'] = 'A different question'
    elif change == 'reference': sem['expected'] = 'A different county'
    elif change == 'rubric': sem['oracle']['rubric'] = 'Always accept'
    elif change == 'program': row['task']['program_ir']['id'] = 'another:program'
    elif change == 'groups': row['task']['program_ir']['source_groups'] = ['held:question']
    else: row['trajectory'].append({'new_target': 'invented'})
    with pytest.raises(ValueError, match='complete original captured row'):
        quality.continue_span_answer_quality(row, raw, review_reference=authority)


@pytest.mark.parametrize('change', ['wrong_reviewer', 'admission', 'negative', 'duplicate', 'empty_reason'])
def test_invalid_approval_rejected(tmp_path, change):
    row, raw, review, _ = fixture(tmp_path)
    if change == 'wrong_reviewer': review['reviewer'] = 'self'
    elif change == 'admission': review['training_admission'] = True
    elif change == 'negative': review['approved'][0]['accepted'] = False
    elif change == 'duplicate': review['approved'].append(copy.deepcopy(review['approved'][0]))
    else: review['approved'][0]['reason'] = ''
    authority = write(tmp_path / 'review.json', review)
    with pytest.raises(ValueError):
        quality.continue_span_answer_quality(row, raw, review_reference=authority)


def test_rebound_inventory_cannot_change_declared_meaning(tmp_path):
    row, raw, review, _ = fixture(tmp_path)
    inventory = json.loads((tmp_path / 'inventory.json').read_text())
    inventory['records'][0]['declared_oracle']['rubric'] = 'Unrelated policy'
    review['inventory'] = write(tmp_path / 'inventory.json', inventory)
    authority = write(tmp_path / 'review.json', review)
    with pytest.raises(ValueError, match='source/rubric binding differs'):
        quality.continue_span_answer_quality(row, raw, review_reference=authority)


def test_raw_and_review_bytes_are_authenticated(tmp_path):
    row, raw, _, authority = fixture(tmp_path)
    (tmp_path / 'review.json').write_text('{}')
    with pytest.raises(ValueError):
        quality.continue_span_answer_quality(row, raw, review_reference=authority)
    row, raw, _, authority = fixture(tmp_path)
    (tmp_path / 'raw.jsonl').write_text(json.dumps(row) + ' \n')
    with pytest.raises(ValueError, match='raw file changed'):
        quality.continue_span_answer_quality(row, raw, review_reference=authority)


def test_current_oracle_must_reproduce_original_score(tmp_path, monkeypatch):
    row, raw, _, authority = fixture(tmp_path)
    wrong = copy.deepcopy(row['outcome']['oracle']); wrong['score'] = .6
    monkeypatch.setattr(quality, 'span_answer_verdict', lambda *args:
        ({'legacy': wrong, 'verdict': {'accepted': True, 'level': 'span', 'score': .6}, 'calls': 1}, {}))
    with pytest.raises(ValueError, match='fresh original span comparison'):
        quality.continue_span_answer_quality(row, raw, review_reference=authority)


def test_normal_converter_still_rejects_current_held_groups(tmp_path):
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import validate_teacher_row
    row, raw, _, authority = fixture(tmp_path)
    view = quality.continue_span_answer_quality(row, raw, review_reference=authority)
    with pytest.raises(ValueError, match='train-only source-group proof'):
        validate_teacher_row(view, {'selected_ids': {'qa:one'},
            'train_groups': {'train:question-one'}, 'held_groups': {'train:question-one'}})
