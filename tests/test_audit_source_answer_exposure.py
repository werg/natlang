import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('exposure', Path(__file__).parents[1] /
                                            'scripts/audit_source_answer_exposure.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(initial, expected=None):
    return {'semantics': {'expected': expected or {'answer': 42}, 'folder_files': {
        'task.json': '{"output_path":"decision.json"}', 'decision.json': initial}}}


def test_exact_answer_in_declared_output():
    findings = module.audit_record(record('{ "answer": 42 }'))
    assert findings[0]['reasons'] == ['declared-output-initially-contains-exact-final-result']


def test_placeholders_and_partial_answers_are_not_exact_gold():
    assert not module.audit_record(record('{}'))
    assert not module.audit_record(record('{"answer":0}'))


def test_json_booleans_are_not_numbers():
    assert not module.same_content('{"answer":true}', '{"answer":1}')


def test_unchanged_files_are_review_flags_not_admission_decisions():
    source = record('{}')
    source['semantics']['expected_files'] = {'decision.json': '{}'}
    assert module.audit_record(source)[0]['disposition'] == 'requires-source-review'


def test_gold_elsewhere_or_substring_is_not_declared_output_match():
    source = record('{}')
    source['semantics']['folder_files']['reference.json'] = '{"answer":42}'
    assert not module.audit_record(source)


def test_unchanged_context_files_are_not_output_exposure():
    source = record('{}')
    source['semantics']['expected_files'] = {'task.json': source['semantics']['folder_files']['task.json']}
    assert not module.audit_record(source)
