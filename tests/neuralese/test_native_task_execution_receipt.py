"""Known model failures are observations; exports and quality stay separate."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.eval.native_state import task_execution_receipt, validate_task_execution_receipt
from natlang_neuralese.eval.task_packet import score_rows, accounted


IDENTITY = {'checkpoint_sha256': 'a'*64, 'tokenizer_fingerprint': 'b'*64,
            'embedding_weight_sha256': 'c'*64, 'architecture': 'lfm2'}


def score_fixture(tmp_path):
    sources = [{'id': f'case-{i}'} for i in range(3)]
    terminal = {'status': 'incomplete', 'output_accounting': {
        'manifest_valid': True, 'source_sha256': 'd'*64, 'expected_jobs': 1,
        'job_states': [{'state': 'failed', 'digest': 'e'*64, 'terminal_error': {
            'record_binding_verified': True, 'program_ir_sha256': 'e'*64,
            'source_sha256': 'd'*64, 'sha256': 'f'*64,
            'error': 'Error: model stream error: '+json.dumps({
                'code': 'neuralese-native-state-open',
                'message': 'neuralese-native-state-open: generation stopped inside a produce span'})}}]}}
    raw = score_rows(sources, [{'task': {'program_ir': {'id': 'case-0'}},
                              'outcome': {'status': 'done', 'accepted': True}}],
                     {'zero': {'status': 'complete'}, 'one': terminal,
                      'two': {'status': 'resource_limit', 'resource_limit_reason': 'model_request_budget'}},
                     {0: 'zero', 1: 'one', 2: 'two'})
    assert accounted(raw) and not raw['complete']
    summary = {'schema': 'natlang.in-run-task-eval/1', 'step': 12,
        'run': str(tmp_path), 'cases': 3, 'case_indices': [0, 3, 6],
        'packet_sha256': '1'*64, 'outcome_counts': raw['outcome_counts'],
        'complete': raw['complete'], 'eligible': accounted(raw), 'task_successes': raw['task_successes']}
    score = {**summary, 'cases_detail': raw['cases']}
    path = tmp_path/'task-score.json';path.write_text(json.dumps(score))
    return summary, score, path


def test_fully_accounted_failures_preserve_missing_exports_and_success_count(tmp_path):
    summary, score, path = score_fixture(tmp_path)
    original = path.read_bytes()
    receipt = task_execution_receipt(summary, IDENTITY, checkpoint_sha256='a'*64)
    assert receipt['schema'] == 'natlang.native-task-execution-receipt/2'
    assert receipt['passed'] and receipt['complete'] and receipt['eligible']
    assert receipt['result_export_complete'] is False
    assert receipt['task_successes'] == 1
    assert receipt['outcome_counts']['model_incomplete'] == 1
    assert receipt['outcome_counts']['budget_exhaustion'] == 1
    assert receipt['task_successes_are_qualification_barrier'] is False
    assert validate_task_execution_receipt(receipt, IDENTITY)
    assert path.read_bytes() == original


@pytest.mark.parametrize('failure', ['unresolved', 'infrastructure_error'])
def test_unknown_or_infrastructure_outcomes_still_block_execution_readiness(tmp_path, failure):
    summary, score, path = score_fixture(tmp_path)
    summary['outcome_counts']['model_incomplete'] = 0
    summary['outcome_counts'][failure] = 1
    score.update(summary);score['cases_detail'][1]['outcome'] = failure
    path.write_text(json.dumps(score))
    receipt = task_execution_receipt(summary, IDENTITY, checkpoint_sha256='a'*64)
    assert not receipt['passed'] and not receipt['complete']
    assert not validate_task_execution_receipt(receipt, IDENTITY)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'counts', 'summary', 'historical', 'indices', 'packet'])
def test_incomplete_or_inconsistent_evidence_cannot_pass(tmp_path, mutation):
    summary, score, path = score_fixture(tmp_path)
    if mutation == 'missing':
        score['cases_detail'].pop()
    elif mutation == 'duplicate':
        score['cases_detail'][1]['id'] = score['cases_detail'][0]['id']
    elif mutation == 'counts':
        score['cases_detail'][1]['outcome'] = 'task_success'
    elif mutation == 'summary':
        summary['task_successes'] = 2
    elif mutation == 'historical':
        summary['historical'] = True;score.update(summary)
    elif mutation == 'indices':
        summary['case_indices'] = [0, 0, 6];score.update(summary)
    else:
        summary['packet_sha256'] = None;score.update(summary)
    path.write_text(json.dumps(score))
    receipt = task_execution_receipt(summary, IDENTITY, checkpoint_sha256='a'*64)
    assert not receipt['passed']


def test_receipt_rechecks_exact_score_bytes_and_current_identity(tmp_path):
    summary, score, path = score_fixture(tmp_path)
    receipt = task_execution_receipt(summary, IDENTITY, checkpoint_sha256='a'*64)
    assert not validate_task_execution_receipt(receipt, {**IDENTITY, 'checkpoint_sha256': '2'*64})
    path.write_text(json.dumps(score)+'\n')
    assert not validate_task_execution_receipt(receipt, IDENTITY)
    receipt['score_sha256'] = sha256_file_hex(path)
    assert validate_task_execution_receipt(receipt, IDENTITY)
    receipt['result_export_complete'] = True
    assert not validate_task_execution_receipt(receipt, IDENTITY)


def test_old_receipts_keep_their_original_export_completion_contract():
    old = {'schema': 'natlang.native-task-execution-receipt/1', 'native_identity': IDENTITY,
           'passed': True, 'complete': False, 'eligible': True}
    assert not validate_task_execution_receipt(old, IDENTITY)
    old['complete'] = True
    assert validate_task_execution_receipt(old, IDENTITY)


@pytest.mark.parametrize('bad', [None, [], 'receipt', {'schema': 'unknown'}])
def test_malformed_receipts_fail_closed(bad):
    assert not validate_task_execution_receipt(bad, IDENTITY)


def test_unbound_or_historical_current_checkpoint_cannot_be_substituted(tmp_path):
    summary, _, _ = score_fixture(tmp_path)
    assert not task_execution_receipt(summary, None, checkpoint_sha256='a'*64)['passed']
    assert not task_execution_receipt(summary, IDENTITY, checkpoint_sha256='2'*64)['passed']


def test_raw_readiness_accepts_bound_v2_but_rejects_modified_score(tmp_path):
    from test_native_readiness import ready
    from natlang_neuralese.train.native_readiness import validate_ready
    path, readiness, requirement = ready(tmp_path, 'lfm')
    identity = readiness['native_identity']
    summary, _, score_path = score_fixture(path.parent)
    receipt = task_execution_receipt(summary, identity, checkpoint_sha256=identity['checkpoint_sha256'])
    task_path = Path(readiness['task_execution_receipt']['path'])
    task_path.write_text(json.dumps(receipt))
    readiness['task_execution_receipt']['sha256'] = sha256_file_hex(task_path)
    path.write_text(json.dumps(readiness))
    assert validate_ready(path, requirement)['model_id'] == 'lfm'
    score_path.write_text('{}')
    with pytest.raises(ValueError, match='task execution proof'):
        validate_ready(path, requirement)
