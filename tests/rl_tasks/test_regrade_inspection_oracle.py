import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.rl_tasks.regrade_inspection_oracle import regrade, qualify_converted_regrade, _source_facts


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


@pytest.fixture
def bound_regrade(tmp_path):
    policy_text = ('Select and cite historical evidence. Return the sorted unique union of the latest inspection date, '
                   'scoreless follow-up dates relevant to the action, and all dates supporting repeatedHighRisk.')
    policy_path = tmp_path / 'chooseNextQuestion.nl'
    policy_path.write_text(policy_text)
    database = tmp_path / 'train.sqlite'
    connection = sqlite3.connect(database)
    connection.executescript('''
        CREATE TABLE inspections(business_id INTEGER, date TEXT, type TEXT, score INTEGER);
        CREATE TABLE violations(business_id INTEGER, date TEXT, violation_type_id TEXT, risk_category TEXT, description TEXT);
    ''')
    connection.executemany('INSERT INTO inspections VALUES(?,?,?,?)', [
        (489, '2016-07-21', 'Routine', 90),
        (489, '2016-07-29', 'Reinspection/Followup', None),
    ])
    connection.executemany('INSERT INTO violations VALUES(?,?,?,?,?)', [
        (489, '2016-07-21', '103103', 'High Risk', 'fixture row A'),
        (489, '2016-07-29', '103103', 'High Risk', 'fixture row B'),
    ])
    connection.commit(); connection.close()
    facts = _source_facts(database, 489)
    correct_dates = sorted(set([facts['latest_inspection_date'], *facts['scoreless_followup_dates'],
                                *facts['recurrent_high_risk_dates']]))
    original = {'businessId': 489, 'recommendationDates': ['2016-07-21', '2016-07-29']}
    corrected = {**original, 'recommendationDates': correct_dates}
    policy_sha = sha(policy_path.read_bytes())
    case = {
        'id': 'case-489', 'split': 'train',
        'source_groups': ['sql-db:bird:food_inspection', 'ext-unit:bird:db:food_inspection'],
        'curriculum': {'external': {'upstream_id': 'deep-review-489', 'program_revision': 'source-v5'}},
        'semantics': {'expected': original, 'oracle': 'exact', 'files': {
            'reviewEstablishment/recommendReview/chooseNextQuestion.nl': policy_text},
            'services': {}},
    }
    case['curriculum']['reference'] = {'root': [['return_result', {'status': 'success', 'value': original}]]}
    case_path = tmp_path / 'cases.jsonl'
    case_path.write_text(json.dumps(case, ensure_ascii=False) + '\n')
    raw = {
        'id': 'trajectory-1', 'task': {'program_ir': case},
        'provenance': {'model': 'gpt-6-luna', 'provider': 'openai-codex',
                       'transport': 'pi-provider', 'collection_role': 'teacher'},
        'outcome': {'status': 'done', 'accepted': False, 'rejection_reasons': ['answer'],
                    'oracle': {'accepted': False, 'level': 'exact'},
                    'checks': {'seeded_failure_requirement': True, 'expected_status': True,
                               'honest_stop': True, 'effects': True, 'files': True,
                               'world': True, 'file_return_consistency': True, 'answer': False},
                    'value': corrected},
        'trajectory': [],
    }
    raw_path = tmp_path / 'raw.json'
    raw_path.write_text(json.dumps(raw, ensure_ascii=False) + '\n')
    raw_sha = sha(raw_path.read_bytes())
    proof = {
        'schema': 'natlang.oracle-regrade-proof/1',
        'disposition': 'retain original raw rejection unchanged; corrected independent regrade is not a training row and is only consumable by a converter that verifies this proof and all bindings',
        'source_row': {'path': str(case_path.resolve()), 'sha256': sha(case_path.read_bytes()),
                       'row_sha256': sha(canonical(case)), 'program_revision': 'source-v5',
                       'upstream_id': 'deep-review-489'},
        'original_oracle': {'kind': 'exact', 'expected_sha256': sha(canonical(original)),
                            'expected': original, 'raw_result_accepted': False,
                            'raw_result_sha256': raw_sha, 'raw_result_path': str(raw_path.resolve()),
                            'rejection_reasons': ['answer']},
        'corrected_oracle': {'kind': 'exact', 'version': 'inspection-review-date-union/1',
                             'expected_sha256': sha(canonical(corrected)), 'expected': corrected,
                             'accepted_under_corrected_oracle': True},
        'independent_source_check': {'dataset': 'BIRD food_inspection train',
            'database_path': str(database.resolve()), 'database_sha256': sha(database.read_bytes()),
            'business_id': 489, 'query_facts': facts},
        'policy_binding': {'operation': 'reviewEstablishment.nl', 'policy_path': str(policy_path.resolve()),
            'policy_sha256': policy_sha, 'policy_text': 'the sorted unique union of the latest inspection date, '
                'scoreless follow-up dates relevant to the action, and all dates supporting repeatedHighRisk'},
    }
    proof_path = tmp_path / 'proof.json'; proof_path.write_text(json.dumps(proof, ensure_ascii=False))
    return raw_path, case_path, proof_path, sha(proof_path.read_bytes())


def test_regrade_is_separate_from_immutable_historical_rejection(bound_regrade):
    raw_path, case_path, proof_path, proof_sha = bound_regrade
    derived, receipt = regrade(raw_result_path=raw_path, case_path=case_path, proof_path=proof_path,
                              expected_proof_sha256=proof_sha)
    assert derived['historical_outcome']['accepted'] is False
    assert derived['historical_outcome']['rejection_reasons'] == ['answer']
    assert derived['outcome']['accepted'] is True
    assert derived['outcome']['rejection_reasons'] == []
    assert derived['oracle_regrade']['admission_decision'] == 'pending-native-graph-conversion'
    assert receipt['current_admission_decision'] == 'pending-native-graph-conversion'


@pytest.mark.parametrize('tamper', ['raw', 'case', 'proof'])
def test_regrade_fails_closed_on_any_bound_artifact_change(bound_regrade, tamper):
    raw_path, case_path, proof_path, proof_sha = bound_regrade
    target = {'raw': raw_path, 'case': case_path, 'proof': proof_path}[tamper]
    target.write_bytes(target.read_bytes() + b' ')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        regrade(raw_result_path=raw_path, case_path=case_path, proof_path=proof_path,
                expected_proof_sha256=proof_sha)


def test_regrade_rejects_raw_failure_beyond_answer_only(bound_regrade):
    raw_path, case_path, proof_path, proof_sha = bound_regrade
    raw = json.loads(raw_path.read_text())
    raw['outcome']['checks']['effects'] = False
    raw_path.write_text(json.dumps(raw) + '\n')
    proof = json.loads(proof_path.read_text())
    proof['original_oracle']['raw_result_sha256'] = sha(raw_path.read_bytes())
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='other than the original answer'):
        regrade(raw_result_path=raw_path, case_path=case_path, proof_path=proof_path,
                expected_proof_sha256=sha(proof_path.read_bytes()))


def test_regrade_rejects_corrected_gold_that_disagrees_with_source(bound_regrade):
    raw_path, case_path, proof_path, proof_sha = bound_regrade
    proof = json.loads(proof_path.read_text())
    proof['corrected_oracle']['expected']['recommendationDates'] = ['2016-07-29']
    proof['corrected_oracle']['expected_sha256'] = sha(canonical(proof['corrected_oracle']['expected']))
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='date union'):
        regrade(raw_result_path=raw_path, case_path=case_path, proof_path=proof_path,
                expected_proof_sha256=sha(proof_path.read_bytes()))


@pytest.fixture
def converted_regrade(bound_regrade, tmp_path):
    raw_path, case_path, proof_path, proof_sha = bound_regrade
    derived, receipt = regrade(raw_result_path=raw_path, case_path=case_path, proof_path=proof_path,
                              expected_proof_sha256=proof_sha)
    derived_path = tmp_path / 'derived.jsonl'
    derived_path.write_text(json.dumps(derived, ensure_ascii=False) + '\n')
    conversion_path = tmp_path / 'conversion.json'
    conversion_path.write_text(json.dumps(receipt, ensure_ascii=False))
    turns_path = tmp_path / 'turns.jsonl'
    turns = []
    for index in range(13):
        turns.append({
            'id': f"{derived['id']}:decision:{index:04d}", 'split': 'train',
            'source_groups': ['sql-db:bird:food_inspection', 'ext-unit:bird:db:food_inspection'],
            'teacher_trajectory_id': derived['id'],
            'source_ref': {'trajectory_id': derived['id']},
            'training_admission': {'approved': True}, 'trace_admission': {'admitted': True},
            'outcome': {'accepted': True, 'oracle': {'verification_epoch': 'inspection-review-date-union/1'}},
        })
    turns_path.write_text(''.join(json.dumps(row) + '\n' for row in turns))
    return raw_path, case_path, proof_path, proof_sha, derived_path, conversion_path, turns_path


def test_converted_regrade_requires_and_binds_all_native_turns(converted_regrade):
    raw, case, proof, proof_sha, derived, conversion, turns = converted_regrade
    receipt = qualify_converted_regrade(raw_result_path=raw, case_path=case, proof_path=proof,
        expected_proof_sha256=proof_sha, derived_path=derived, conversion_receipt_path=conversion,
        native_turns_path=turns)
    assert receipt['current_admission_decision'] == 'eligible-for-source-admission'
    assert receipt['turn_count'] == 13
    assert receipt['historical_outcome_preserved']['accepted'] is False
    assert receipt['bindings']['raw_result_sha256'] == sha(raw.read_bytes())


@pytest.mark.parametrize('tamper', ['derived', 'conversion', 'turns', 'turn-source', 'turn-group'])
def test_converted_regrade_fails_closed_on_current_admission_tampering(converted_regrade, tamper):
    raw, case, proof, proof_sha, derived, conversion, turns = converted_regrade
    if tamper == 'derived':
        derived.write_text(derived.read_text().replace('inspection-review-date-union/1', 'wrong-version'))
    elif tamper == 'conversion':
        obj = json.loads(conversion.read_text()); obj['proof_sha256'] = '0' * 64
        conversion.write_text(json.dumps(obj))
    else:
        rows = [json.loads(line) for line in turns.read_text().splitlines()]
        if tamper == 'turns':
            rows.pop()
        elif tamper == 'turn-source':
            rows[0]['teacher_trajectory_id'] = 'other-trajectory'
        else:
            rows[0]['source_groups'] = ['sql-db:bird:food_inspection']
        turns.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        qualify_converted_regrade(raw_result_path=raw, case_path=case, proof_path=proof,
            expected_proof_sha256=proof_sha, derived_path=derived, conversion_receipt_path=conversion,
            native_turns_path=turns)
