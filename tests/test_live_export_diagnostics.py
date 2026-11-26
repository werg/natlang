import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/neuralese_data'))
from live_export_diagnostics import export_rejection_diagnostic


def fixture(tmp_path):
    directory = tmp_path / 'case'
    folder = directory / 'st_owned'
    folder.mkdir(parents=True)
    raw = folder / 'trajectory-case-full.json.gz'
    raw.write_bytes(b'actual packet')
    (directory / 'source.json').write_bytes(b'actual source')
    row = {'schema': 'vibestudio.teacher-export-rejection/1', 'scenario': 'case', 'runId': 'st_owned',
           'trajectory': {'path': raw.name, 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()},
           'pinsSha256': hashlib.sha256(b'actual source').hexdigest(),
           'error': {'name': 'Error', 'message': 'Error: Independent content review did not pass'}, 'interrupted': False}
    return directory, raw, row


def test_reads_exact_matching_latest_diagnostic_without_granting_admission(tmp_path):
    directory, raw, row = fixture(tmp_path)
    path = raw.parent / 'export-rejections.jsonl'
    second = {**row, 'error': {'message': 'Error: Native evidence does not bind'}, 'interrupted': True}
    path.write_text(json.dumps(row) + '\n' + json.dumps(second) + '\n')
    actual, errors = export_rejection_diagnostic(directory, raw, 'case')
    assert actual == second and errors == []
    assert 'passed' not in actual and 'task_completed' not in actual


def test_stale_raw_source_run_and_case_are_never_authoritative(tmp_path):
    directory, raw, row = fixture(tmp_path)
    path = raw.parent / 'export-rejections.jsonl'
    changes = [{'scenario': 'other'}, {'runId': 'st_foreign'}, {'pinsSha256': 'f' * 64},
               {'trajectory': {'path': raw.name, 'sha256': 'a' * 64}}, {'error': {'message': ''}}]
    for change in changes:
        path.write_text(json.dumps({**row, **change}) + '\n')
        actual, errors = export_rejection_diagnostic(directory, raw, 'case')
        assert actual is None and errors


def test_malformed_tail_does_not_erase_matching_evidence_or_become_capture_failure(tmp_path):
    directory, raw, row = fixture(tmp_path)
    (raw.parent / 'export-rejections.jsonl').write_text(json.dumps(row) + '\n{"interrupted')
    actual, errors = export_rejection_diagnostic(directory, raw, 'case')
    assert actual == row and errors == ['Malformed export diagnostic at export-rejections.jsonl:2']


def test_old_capture_without_diagnostics_is_unchanged(tmp_path):
    directory, raw, _ = fixture(tmp_path)
    assert export_rejection_diagnostic(directory, raw, 'case') == (None, [])


def test_actual_campaign_outcome_surfaces_export_failure_without_changing_native_verdict(tmp_path):
    import gzip
    import vibestudio_campaign
    directory, raw, row = fixture(tmp_path)
    packet = {'result': {'passed': True}, 'execution': {'diagnostics': {'natlangTeacher': {}}}}
    with gzip.open(raw, 'wt') as stream:
        json.dump(packet, stream)
    row['trajectory']['sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    (raw.parent / 'export-rejections.jsonl').write_text(json.dumps(row) + '\n')
    result = vibestudio_campaign.outcomes(directory, ['case'], None, {}, {'case': {'taskId': 'task'}})['case']
    assert result['status'] == 'export-rejected'
    assert result['reason'] == row['error']['message']
    assert result['export_rejection'] == row
    packet['result'] = {'passed': False, 'reason': 'Required artifact is absent'}
    with gzip.open(raw, 'wt') as stream:
        json.dump(packet, stream)
    row['trajectory']['sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    (raw.parent / 'export-rejections.jsonl').write_text(json.dumps(row) + '\n')
    result = vibestudio_campaign.outcomes(directory, ['case'], None, {}, {'case': {'taskId': 'task'}})['case']
    assert result['status'] == 'test-failed' and result['reason'] == 'Required artifact is absent'
    assert not vibestudio_campaign.infrastructure_failure(result, {})


def test_actual_host_writer_is_read_by_campaign_with_exact_utf8_and_byte_identity(tmp_path, monkeypatch):
    import os
    import subprocess
    import pytest
    module = os.environ.get('VIBESTUDIO_HOST_EXPORT_DIAGNOSTICS_MODULE')
    if not module:
        pytest.skip('Set actual Host diagnostic module for the cross-repository boundary control')
    directory, raw, _ = fixture(tmp_path)
    monkeypatch.delenv('VIBESTUDIO_EXPORT_FAILURE_DIAGNOSTICS', raising=False)
    code = '''import {pathToFileURL} from 'node:url';
const {recordExportRejection}=await import(pathToFileURL(process.argv[1]).href);
await recordExportRejection({...JSON.parse(process.argv[2]),error:new Error('Native review: 実際\\nretained "bytes"')});'''
    subprocess.run(['node', '--input-type=module', '-e', code, module,
                    json.dumps({'directory': str(directory), 'scenario': 'case', 'runId': 'st_owned',
                                'pins': str(directory / 'source.json'), 'trajectory': str(raw)})], check=True)
    row, errors = export_rejection_diagnostic(directory, raw, 'case')
    assert errors == [] and row['error']['message'] == 'Error: Native review: 実際\nretained "bytes"'


def review_fixture(tmp_path):
    directory, raw, row = fixture(tmp_path)
    pins = directory/'source.json'
    pins.write_text(json.dumps({'teacher':'test-only:teacher'}))
    row['pinsSha256'] = hashlib.sha256(pins.read_bytes()).hexdigest()
    actual = {'schema':'natlang.live-workflow-review-failure/1','model':'test-only:reviewer','teacher':'test-only:teacher',
              'input':{'actual':{'goal':'actual goal'}},'complete':False,
              'failure':{'kind':'provider-transport','phase':'provider-turn','error':{'name':'Error','message':'fetch failed'}}}
    path = directory/'review-failed.json'
    path.write_text(json.dumps(actual))
    row['reviewFailure'] = {k:actual[k] for k in ['schema','model','teacher','failure']}
    row['reviewFailure']['capture'] = {'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (raw.parent/'export-rejections.jsonl').write_text(json.dumps(row)+'\n')
    return directory, raw, row, path, actual


def test_transport_review_pending_is_source_and_capture_bound(tmp_path):
    directory, raw, _, _, _ = review_fixture(tmp_path)
    row, errors = export_rejection_diagnostic(directory,raw,'case')
    assert not errors and row['review_transport_failure'] is True
    assert 'task_completed' not in row


def test_changed_review_capture_is_not_an_operational_excuse(tmp_path):
    directory, raw, _, path, _ = review_fixture(tmp_path)
    path.write_text('{}')
    row, errors = export_rejection_diagnostic(directory,raw,'case')
    assert row is None and errors


def test_assessment_error_and_foreign_teacher_cannot_claim_transport(tmp_path):
    directory, raw, row, path, actual = review_fixture(tmp_path)
    actual['failure']['phase'] = 'assessment'
    path.write_text(json.dumps(actual))
    row['reviewFailure']['capture']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (raw.parent/'export-rejections.jsonl').write_text(json.dumps(row)+'\n')
    result, errors = export_rejection_diagnostic(directory,raw,'case')
    assert not errors and result['review_transport_failure'] is False
    actual['teacher'] = 'foreign';path.write_text(json.dumps(actual))
    row['reviewFailure']['capture']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (raw.parent/'export-rejections.jsonl').write_text(json.dumps(row)+'\n')
    result, errors = export_rejection_diagnostic(directory,raw,'case')
    assert result is None and errors
