"""TEST-only scripted review exchanges. No provider call or teacher registration."""
import copy
import json
import pytest
import live_teacher_sources as live


def review_fixture():
    actual = {'goal': 'Explain why the unchanged record cannot proceed', 'artifacts': {},
              'requestedReturn': {'reason': 'Review withdrawn'},
              'attemptedArguments': [{'path': '/effects/0/args', 'service': 'owned', 'method': 'apply',
                                      'arguments': ['old', {'notice': 'shift-board'}]}]}
    expected = {'goal': actual['goal'], 'requestedReturnRequired': True, 'evidenceRoleLayout': 'native-reference/1',
                'observations': {'current': {'notice': 'paper'}, 'effects': [{'service': 'owned', 'method': 'apply',
                    'result': {'status': 'changed'}, 'attemptedArgumentsPath': '/effects/0/args'}]}}
    source_input = {'actual': actual, 'expected': expected, 'rubric': 'Judge the requested answer against actual native state'}
    grade = {'accepted': True, 'verdict': 'TEST-only independent reviewer fixture'}
    review = {'schema': 'natlang.live-workflow-review/1', 'teacher': 'TEST-only:teacher', 'model': 'TEST-only:reviewer',
              'input': copy.deepcopy(source_input), 'verdict': copy.deepcopy(grade), 'exchange': {
                'request': {'messages': [{'role': 'system', 'content': 'TEST-only'}, {'role': 'user', 'content': json.dumps({
                    'rubric': source_input['rubric'], 'reference': expected, 'candidate': actual})}]},
                'reply': {'calls': [['grade', copy.deepcopy(grade)]],
                          'raw_calls': [{'function': {'name': 'grade', 'arguments': json.dumps(grade)}}],
                          'raw_response': {'provider': 'TEST-only', 'model': 'reviewer'}}}}
    return review, source_input


def test_new_reference_roles_bind_exact_declared_input_request_provider_and_raw_grade():
    review, source_input = review_fixture()
    before = copy.deepcopy(review)
    assert live.semantic_review_roles_bound(review, source_input)
    assert review == before


@pytest.mark.parametrize('mutation', ['missing-declaration', 'changed-native-state', 'changed-answer', 'changed-attempt',
    'changed-request', 'changed-role-layout', 'candidate-native-facts', 'forged-verdict', 'forged-raw-grade',
    'wrong-provider', 'same-teacher', 'missing-reply', 'optional-wrong-return', 'malformed-declaration'])
def test_reference_role_forgery_fails_closed(mutation):
    review, source_input = review_fixture()
    if mutation == 'missing-declaration': source_input = None
    if mutation == 'changed-native-state': review['input']['expected']['observations']['current']['notice'] = 'shift-board'
    if mutation == 'changed-answer': review['input']['actual']['requestedReturn'] = {'reason': 'Proceed'}
    if mutation == 'changed-attempt': review['input']['actual']['attemptedArguments'][0]['arguments'] = []
    if mutation == 'changed-request': review['exchange']['request']['messages'][1]['content'] = '{}'
    if mutation == 'changed-role-layout': review['input']['expected']['evidenceRoleLayout'] = 'unknown'
    if mutation == 'candidate-native-facts':
        review['input']['actual']['observations'] = review['input']['expected'].pop('observations')
        source_input = copy.deepcopy(review['input'])
    if mutation == 'forged-verdict': review['verdict']['verdict'] = 'Different unissued verdict'
    if mutation == 'forged-raw-grade': review['exchange']['reply']['raw_calls'][0]['function']['arguments'] = json.dumps({'accepted': False, 'verdict': 'Rejected'})
    if mutation == 'wrong-provider': review['exchange']['reply']['raw_response']['provider'] = 'foreign'
    if mutation == 'same-teacher': review['teacher'] = review['model']
    if mutation == 'missing-reply': review['exchange'].pop('reply')
    if mutation == 'optional-wrong-return':
        review['input']['expected']['requestedReturnRequired'] = False
        source_input = copy.deepcopy(review['input'])
    if mutation == 'malformed-declaration': source_input = {}
    assert not live.semantic_review_roles_bound(review, source_input)


def test_declared_legacy_input_stays_exact_without_changing_old_undeclared_schema():
    review = {'input': {'actual': {'old': 'data'}, 'expected': {'goal': 'old'}}}
    assert live.semantic_review_roles_bound(review)
    assert live.semantic_review_roles_bound(review, copy.deepcopy(review['input']))
    assert not live.semantic_review_roles_bound(review, {'actual': {'old': 'other'}})
