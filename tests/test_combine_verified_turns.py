import json

import pytest

from scripts.combine_verified_turns import combine


def turn(id, role, approved=True):
    return {'version': 'natlang.teacher_training_turn.native/1', 'id': id,
            'program_id': id.split(':')[0], 'source_groups': [id.split(':')[0]],
            'source_ref': {'source_row_sha256': 'a' * 64},
            'messages': [{'role': 'system', 'content': 'test'}],
            'target': {'role': 'assistant', 'content': 'done'},
            'provenance': {'collection_role': role}, 'outcome': {'accepted': True},
            'training_admission': {'kind': 'exact-native-runtime-oracle', 'approved': approved},
            'trace_admission': {'admitted': True}}


def test_combines_verified_student_and_teacher_without_rewriting(tmp_path):
    student, teacher, output = (tmp_path / name for name in ('student.jsonl', 'teacher.jsonl', 'out.jsonl'))
    student.write_text(json.dumps(turn('a:0', 'student')) + '\n')
    teacher.write_text(json.dumps(turn('b:0', 'teacher', False)) + '\n' +
                       json.dumps(turn('b:1', 'teacher')) + '\n')
    result = combine(student, teacher, output)
    assert result['rows'] == 3
    assert result['approved_decisions'] == 2
    assert combine(student, teacher, output) == result
    assert len(output.read_text().splitlines()) == 3


def test_no_approved_data_or_duplicate_decision_fails(tmp_path):
    student, teacher, output = (tmp_path / name for name in ('student.jsonl', 'teacher.jsonl', 'out.jsonl'))
    student.write_text('')
    teacher.write_text(json.dumps(turn('a:0', 'teacher', False)) + '\n')
    with pytest.raises(ValueError, match='no approved'):
        combine(student, teacher, output)
    assert not output.exists()
    student.write_text(json.dumps(turn('a:0', 'student')) + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        combine(student, teacher, output)
