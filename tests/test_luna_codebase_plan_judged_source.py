import hashlib
import json

import pytest

from scripts.rl_tasks.build_reviewed_luna_plan import _verify_rows


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
