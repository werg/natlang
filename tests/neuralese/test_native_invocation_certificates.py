"""Native partial scopes bind original failed packets; no synthetic fixture enters a registry."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from live_teacher_sources import verify_native_invocation_certificate, verified_training_scope


def fixture(root):
    def bind(name, value, raw=False):
        data = value.encode() if raw else (json.dumps(value, ensure_ascii=False) + '\n').encode()
        (root / name).write_bytes(data)
        return {'path': name, 'sha256': hashlib.sha256(data).hexdigest()}
    def ref(value):
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        return {'complete': True, 'hash': hashlib.sha256(text.encode()).hexdigest(), 'bytes': len(text.encode())}
    goal = 'Review the actual current question.'
    history = {'complete': True, 'messages': []}
    answer = {'status': 'no-question', 'message': None}
    body = 'Return the actual native question judgment.\n'
    source = '---\nreturns: DurableQuestionJudgment\n---\n' + body
    definition = {'id': 'nl:reviewDurableQuestion/reviewQuestion/interpretQuestion.nl', 'source': 'reviewDurableQuestion/reviewQuestion/interpretQuestion.nl',
                  'returns': 'DurableQuestionJudgment', 'instructions': ref(body)}
    record = {'call_id': 'question', 'parent_call_id': None, 'definition': definition, 'outcome': 'done', 'effects': [],
              'inputs': {'goal': ref(goal), 'history': ref(history)}, 'output': ref(answer)}
    records = {'question': record}
    proof = {'goal': goal, 'diagnostics': {'history': history}, 'actualResult': {'question': answer},
             'checks': {'exactIndividualPublications': True, 'exactQuestionDisposition': True, 'exactParticipantDisposition': False}}
    outcome = {'passed': False, 'reason': 'Incorrect independent participant result'}
    raw = {'result': outcome, 'execution': {'diagnostics': {'natlangTeacher': {'calls': [{'callId': 'question', 'record': record}]}, 'openGoalVerification': proof}}}
    certificate = {'schema': 'natlang.independent-native-return/1', 'contract': 'gad-question-judgment/1',
                   'definition': definition, 'inputs': record['inputs'], 'output': record['output'], 'expectedInputs': {'goal': goal, 'history': history},
                   'expectedInputJson': {'goal': json.dumps(goal), 'history': json.dumps(history, separators=(',', ':'))},
                   'expectedReturn': answer, 'expectedReturnJson': json.dumps(answer, separators=(',', ':')), 'instructionsJson': json.dumps(body),
                   'checker': bind('contract.nl', source, True), 'originalPacket': bind('original.json', raw),
                   'nativeBinding': {'returnPath': ['actualResult', 'question'], 'inputPaths': {'goal': ['goal'], 'history': ['diagnostics', 'history']},
                                     'checkNames': ['exactIndividualPublications', 'exactQuestionDisposition']}}
    scope = {'original_outcome': bind('outcome.json', outcome), 'assessor': bind('assessor.mjs', 'source-bound native assessor', True)}
    invocation = {'call_id': 'question', 'certificate': bind('certificate.json', certificate)}
    return records, scope, invocation, certificate, raw, bind


def test_native_certificate_accepts_only_its_independent_complete_result(tmp_path):
    records, scope, invocation, _, _, _ = fixture(tmp_path)
    verify_native_invocation_certificate(invocation, records, scope, tmp_path)


@pytest.mark.parametrize('mutation', ['native-return', 'native-input', 'unknown-contract', 'remove-required-check', 'failed-check',
    'wrong-contract-source', 'contract-bytes', 'original-passed', 'original-call', 'effects', 'assessor-missing'])
def test_native_certificate_rejects_forged_scope(tmp_path, mutation):
    records, scope, invocation, certificate, raw, bind = fixture(tmp_path)
    if mutation == 'native-return': certificate['expectedReturn']['status'] = 'current'
    elif mutation == 'native-input': certificate['expectedInputs']['goal'] = 'Other goal'
    elif mutation == 'unknown-contract': certificate['contract'] = 'arbitrary-successful-call'
    elif mutation == 'remove-required-check': certificate['nativeBinding']['checkNames'] = ['exactIndividualPublications']
    elif mutation == 'failed-check': raw['execution']['diagnostics']['openGoalVerification']['checks']['exactQuestionDisposition'] = False
    elif mutation == 'wrong-contract-source': certificate['contract'] = 'gad-participant-judgment/1'
    elif mutation == 'contract-bytes': certificate['checker'] = bind('different.nl', '---\nreturns: string\n---\nSay something else.\n', True)
    elif mutation == 'original-passed': raw['result']['passed'] = True
    elif mutation == 'original-call': raw['execution']['diagnostics']['natlangTeacher']['calls'][0]['record'] = {**records['question'], 'outcome': 'error'}
    elif mutation == 'effects': records['question']['effects'] = [{'service': 'unexpected', 'method': 'write'}]
    else: del scope['assessor']
    certificate['originalPacket'] = bind('original.json', raw)
    invocation['certificate'] = bind('certificate.json', certificate)
    with pytest.raises(ValueError): verify_native_invocation_certificate(invocation, records, scope, tmp_path)


@pytest.mark.parametrize('mutation', [None, 'downgrade', 'delete-certificate', 'delete-assessor', 'downgrade-delete-certificate'])
def test_native_partial_protocol_is_explicit_and_cannot_downgrade(tmp_path, mutation):
    records, scope, invocation, _, _, _ = fixture(tmp_path)
    scope.update(schema='natlang.training-scope/2', invocations=[invocation])
    invocation.update(checks={'nativeCompleteReturn': True}, evidence=[{'kind': 'workspace', **invocation['certificate']}])
    if mutation in {'downgrade', 'downgrade-delete-certificate'}: scope['schema'] = 'natlang.training-scope/1'
    if mutation in {'delete-certificate', 'downgrade-delete-certificate'}: del invocation['certificate']
    if mutation == 'delete-assessor': del scope['assessor']
    verification = {'status': 'partial', 'training_scope': scope}
    if mutation:
        with pytest.raises(ValueError, match='scope|assessor|downgrade'):
            verified_training_scope(verification, records, tmp_path)
    else:
        assert verified_training_scope(verification, records, tmp_path) == {'question': None}
