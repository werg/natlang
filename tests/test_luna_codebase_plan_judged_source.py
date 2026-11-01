import hashlib
import json

import pytest

from scripts.rl_tasks.build_reviewed_luna_plan import _verify_declared_reference_audits, _verify_rows


def test_judged_source_qualification_binds_intake_without_claiming_behavior(tmp_path):
    source = tmp_path / 'cases.jsonl'
    row = {'id': 'judged-1', 'split': 'train', 'source_groups': ['loghub-system:HDFS'],
           'semantics': {'oracle': {'level': 'judged', 'rubric': 'Assess source evidence.'}}}
    source.write_text(json.dumps(row) + '\n')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    intake = {'admission_eligible': True, 'judged_cases_path': str(source.resolve()),
              'judged_cases_sha256': source_sha}
    intake_sha = hashlib.sha256(json.dumps(intake, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    qualification = {'schema': 'natlang.judged-source-qualification/1',
                     'judged_cases_path': str(source.resolve()), 'judged_cases_sha256': source_sha,
                     'cases_sha256': source_sha, 'intake_receipt_sha256': intake_sha,
                     'behaviorally_verified': False}
    qualification_path = tmp_path / 'source-qualification.json'
    qualification_path.write_text(json.dumps(qualification))
    rows, proof = _verify_rows(source, qualification_path, intake_receipt=intake)
    assert rows == [row]
    assert proof['behaviorally_verified'] is False

    qualification['behaviorally_verified'] = True
    qualification_path.write_text(json.dumps(qualification))
    with pytest.raises(ValueError, match='does not bind'):
        _verify_rows(source, qualification_path, intake_receipt=intake)


def test_judged_source_requires_matching_cleared_intake(tmp_path):
    source = tmp_path / 'cases.jsonl'
    row = {'id': 'judged-1', 'split': 'train', 'source_groups': ['loghub-system:HDFS'],
           'semantics': {'oracle': {'level': 'judged', 'rubric': 'Assess source evidence.'}}}
    source.write_text(json.dumps(row) + '\n')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    intake = {'admission_eligible': True, 'judged_cases_path': str(source.resolve()),
              'judged_cases_sha256': source_sha}
    intake_sha = hashlib.sha256(json.dumps(intake, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    qualification = {'schema': 'natlang.judged-source-qualification/1',
                     'judged_cases_path': str(source.resolve()), 'judged_cases_sha256': source_sha,
                     'cases_sha256': source_sha, 'intake_receipt_sha256': intake_sha,
                     'behaviorally_verified': False}
    qualification_path = tmp_path / 'source-qualification.json'
    qualification_path.write_text(json.dumps(qualification))
    intake['admission_eligible'] = False
    with pytest.raises(ValueError, match='does not bind'):
        _verify_rows(source, qualification_path, intake_receipt=intake)


def _reference_fixture(tmp_path, *, second_row=False):
    source = tmp_path / 'cases.jsonl'
    row = {'id': 'case-a', 'split': 'train', 'source_groups': ['contractnli:train:34'],
           'source_revisions': ['source@rev1'],
           'curriculum': {'external': {'upstream_id': 'nda-34'}},
           'semantics': {'root': 'reviewAgreement.nl', 'inputs': {'request': {'agreementId': '34'}},
                         'oracle': {'level': 'judged', 'rubric': 'Check source claims.'}}}
    rows = [row]
    if second_row:
        rows.append({**row, 'id': 'case-b', 'source_groups': ['contractnli:train:146'],
                     'curriculum': {'external': {'upstream_id': 'nda-146'}},
                     'semantics': {'root': 'reviewAgreement.nl', 'inputs': {'request': {'agreementId': '146'}},
                                   'oracle': {'level': 'judged', 'rubric': 'Check source claims.'}}})
    source.write_text(''.join(json.dumps(item, sort_keys=True) + '\n' for item in rows))
    program_ir = json.loads(json.dumps(row))
    program_ir_sha = hashlib.sha256(json.dumps(program_ir, sort_keys=True).encode()).hexdigest()
    result = {'task': {'program_ir': program_ir},
              'provenance': {'program_ir_sha256': program_ir_sha}}
    result_path = tmp_path / 'capture.result.json'
    result_path.write_text(json.dumps(result, sort_keys=True) + '\n')
    audit = {'case_id': row['id'], 'world': 'nda-34', 'source_groups': row['source_groups'],
             'source_row_program_ir_sha256': program_ir_sha,
             'program_ir_sha256': program_ir_sha,
             'task_program_ir_sha256': program_ir_sha,
             'result': {'path': str(result_path),
                        'sha256': hashlib.sha256(result_path.read_bytes()).hexdigest()}}
    audit_path = tmp_path / 'reference-audit.json'
    audit_path.write_text(json.dumps(audit, sort_keys=True) + '\n')
    verification = {'reference_audits': [{'path': str(audit_path),
                                           'sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest()}],
                    'reference_coverage': 'all_candidates'}
    return source, rows, result_path, audit_path, verification


def test_declared_reference_audit_binds_captured_task_result_and_exact_source_row(tmp_path):
    source, rows, _, _, verification = _reference_fixture(tmp_path)
    _verify_declared_reference_audits(source, rows, verification)


def test_declared_reference_audit_rejects_swapped_source_row(tmp_path):
    source, rows, _, _, verification = _reference_fixture(tmp_path)
    rows[0]['semantics']['inputs']['request']['agreementId'] = '146'
    with pytest.raises(ValueError, match='exact candidate source row'):
        _verify_declared_reference_audits(source, rows, verification)


def test_declared_reference_audit_rejects_modified_capture_bytes(tmp_path):
    source, rows, result_path, _, verification = _reference_fixture(tmp_path)
    result_path.write_text(result_path.read_text() + 'tampered\n')
    with pytest.raises(ValueError, match='captured result bytes'):
        _verify_declared_reference_audits(source, rows, verification)


def test_reference_capture_cannot_bind_another_task_under_the_same_case_id(tmp_path):
    source, rows, result_path, audit_path, verification = _reference_fixture(tmp_path)
    result = json.loads(result_path.read_text())
    result['task']['program_ir']['semantics']['inputs']['request']['agreementId'] = '146'
    program_ir = result['task']['program_ir']
    result['provenance']['program_ir_sha256'] = hashlib.sha256(json.dumps(program_ir, sort_keys=True).encode()).hexdigest()
    result_path.write_text(json.dumps(result, sort_keys=True) + '\n')
    audit = json.loads(audit_path.read_text())
    audit['program_ir_sha256'] = result['provenance']['program_ir_sha256']
    audit['source_row_program_ir_sha256'] = result['provenance']['program_ir_sha256']
    audit['task_program_ir_sha256'] = result['provenance']['program_ir_sha256']
    audit['result']['sha256'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit, sort_keys=True) + '\n')
    verification['reference_audits'][0]['sha256'] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='exact candidate source row'):
        _verify_declared_reference_audits(source, rows, verification)


def test_full_reference_coverage_rejects_missing_candidate_case(tmp_path):
    source, rows, _, _, verification = _reference_fixture(tmp_path, second_row=True)
    with pytest.raises(ValueError, match='full reference coverage'):
        _verify_declared_reference_audits(source, rows, verification)
