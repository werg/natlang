import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'start_reviewed_openrouter_successor.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_worker_plan(tmp_path, *, launcher_name='start_openrouter_teacher.py', status_override=None):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import digest as file_digest

    worker = tmp_path / 'worker-1'
    worker.mkdir(parents=True, exist_ok=True)
    queue = worker / 'worker-1.queue.jsonl'
    queue.write_text('{"key":"a"}\n{"key":"b"}\n')
    journal = worker / 'journal.jsonl'
    launcher = worker / launcher_name
    launcher.write_text('# test worker launcher\n')
    supervisor = worker / 'run_bonsai_queue.py'
    supervisor.write_text('# test supervisor\n')
    config = worker / 'provider-request-config.json'
    config.write_text('{}\n')
    dep = worker / 'dependency.js'
    dep.write_text('export {}\n')
    dep_pins = worker / 'provider-dependency-pins.json'
    dep_pins.write_text(json.dumps({'files': {str(dep): file_digest(dep)}}) + '\n')
    native = worker / 'native-proof.json'
    runtime = worker / 'runtime'
    runtime.mkdir()
    runtime_file = runtime / 'dist' / 'teacher.js'
    runtime_file.parent.mkdir()
    runtime_file.write_text('export {}\n')
    runtime_manifest = {'files': {'dist/teacher.js': file_digest(runtime_file)}, 'symlinks': []}
    (runtime / 'frozen-runtime.json').write_text(json.dumps(runtime_manifest) + '\n')
    runtime_manifest_sha = file_digest(runtime / 'frozen-runtime.json')
    native.write_text(json.dumps({'model_calls': 0, 'runtime_manifest_sha256': runtime_manifest_sha,
                                  'counts': {'cases': 2, 'admitted': 2, 'denied': 0}, 'rejected': []}) + '\n')
    pins = {str(path): file_digest(path) for path in [queue, launcher, supervisor, config, dep_pins, native]}
    plan = {
        'approved': True, 'root_approved': True, 'provider': 'openrouter',
        'model': 'stealth/space-bunny-alpha', 'model_concurrency': 1, 'cases': 2,
        'queue': str(queue), 'journal': str(journal), 'status_file': status_override or str(worker / 'worker-status.json'),
        'launcher': str(launcher), 'supervisor': str(supervisor), 'runtime': str(runtime),
        'runtime_manifest_sha256': runtime_manifest_sha, 'provider_request_config': str(config),
        'provider_dependency_pins': str(dep_pins), 'native_review': str(native),
        'native_review_sha256': file_digest(native), 'pins': pins,
    }
    plan_path = worker / 'root-approved-openrouter-plan.json'
    plan_path.write_text(json.dumps(plan) + '\n')
    successor = {'plan': str(plan_path), 'plan_sha256': file_digest(plan_path), 'queue': str(queue),
                 'journal': str(journal), 'status_file': str(worker / 'worker-status.json'),
                 'launcher_log': str(worker / 'launcher.log'), 'cases': 2}
    return plan, successor, plan_path, launcher


def test_preflight_creates_output_parents_before_claim(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import preflight_successor_outputs

    plan = tmp_path / 'campaign' / 'plan.json'
    successor = {
        'launcher_log': str(plan.parent / 'worker' / 'launcher.log'),
        'journal': str(plan.parent / 'worker' / 'jobs' / 'journal.jsonl'),
        'status_file': str(plan.parent / 'worker-status.json'),
    }
    preflight_successor_outputs(successor)
    assert Path(successor['launcher_log']).parent.is_dir()
    assert Path(successor['journal']).parent.is_dir()
    assert Path(successor['status_file']).parent.is_dir()
    assert Path(successor['launcher_log']).is_file()


def test_output_parent_failure_does_not_claim_handoff(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import digest as file_digest

    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('file')
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    record = campaign / 'launch.json'
    _worker_plan, successor, worker_plan_path, _launcher = make_worker_plan(campaign)
    successor['launcher_log'] = str(blocked / 'launcher.log')
    authority = campaign / 'authority.json'
    authority.write_text(json.dumps({'additional_teachers': {'space_bunny': {
        'queue': successor['queue'], 'journal': successor['journal'], 'status_file': successor['status_file']}}}))
    plan_path = campaign / 'plan.json'
    helper = Path(__file__).resolve().parents[1] / 'scripts' / 'start_reviewed_openrouter_successor.py'
    plan = {
        'root_approved': True, 'successor': successor,
        'predecessor': {'queue': successor['queue'], 'journal': successor['journal'],
                        'status_file': successor['status_file']},
        'authority': str(authority), 'cwd': str(campaign), 'launch_record': str(record),
        'artifact_hashes': {str(helper): file_digest(helper), str(worker_plan_path): file_digest(worker_plan_path)},
    }
    plan_path.write_text(json.dumps(plan))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan_path), '--sha256', digest(plan_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not record.exists()
    assert not record.with_suffix('.lock').exists()


def test_successor_uses_approved_worker_plan_launcher_and_exact_case_count(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import validate_worker_plan

    _worker, successor, _path, launcher = make_worker_plan(tmp_path)
    worker_plan, plan_path, resolved_launcher = validate_worker_plan(successor)
    assert worker_plan['cases'] == 2
    assert plan_path == Path(successor['plan'])
    assert resolved_launcher == launcher


def test_successor_rejects_rollover_helper_as_worker_launcher(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import validate_worker_plan, digest as file_digest

    _worker_plan, successor, plan_path, _launcher = make_worker_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    wrong = plan_path.parent / 'start_reviewed_openrouter_successor.py'
    wrong.write_text('# wrapper, not worker\n')
    plan['launcher'] = str(wrong)
    plan['pins'][str(wrong)] = file_digest(wrong)
    plan_path.write_text(json.dumps(plan) + '\n')
    successor['plan_sha256'] = file_digest(plan_path)
    with pytest.raises(ValueError, match='must be start_openrouter_teacher.py'):
        validate_worker_plan(successor)


def test_successor_rejects_semantic_pin_keys_and_status_path_mismatch(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import validate_worker_plan, digest as file_digest

    _worker, successor, plan_path, _launcher = make_worker_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan['pins'] = {'worker_queue': plan['pins'][plan['queue']]}
    plan_path.write_text(json.dumps(plan) + '\n')
    successor['plan_sha256'] = file_digest(plan_path)
    with pytest.raises(ValueError, match='absolute'):
        validate_worker_plan(successor)

    _worker, successor, plan_path, _launcher = make_worker_plan(tmp_path / 'other')
    successor['status_file'] = str(Path(successor['status_file']).parent / 'wrong-status.json')
    with pytest.raises(ValueError, match='status_file'):
        validate_worker_plan(successor)


def test_mutable_authority_status_journal_and_symlink_aliases_cannot_be_pinned(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import validate_path_hash_map

    status = tmp_path / 'worker-status.json'
    status.write_text('{}\n')
    with pytest.raises(ValueError, match='mutable authority/status/journal'):
        validate_path_hash_map({str(status): digest(status)}, 'worker pin', [status])
    alias = tmp_path / 'status-alias.json'
    alias.symlink_to(status)
    with pytest.raises(ValueError, match='mutable authority/status/journal'):
        validate_path_hash_map({str(alias): digest(status)}, 'rollover pin', [status])

    control = tmp_path / 'control'
    control.mkdir()
    control_file = control / 'drain.json'
    control_file.write_text('{}\n')
    with pytest.raises(ValueError, match='mutable authority/status/journal'):
        validate_path_hash_map({str(control_file): digest(control_file)}, 'worker pin',
                               protected_dirs=[control])


def test_fresh_authority_binding_does_not_carry_predecessor_proof_or_counts(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import validate_worker_plan, new_authority_binding

    _worker, successor, plan_path, _launcher = make_worker_plan(tmp_path)
    worker_plan, _, _ = validate_worker_plan(successor)
    binding = new_authority_binding(worker_plan, successor, 12345, plan_path)
    assert binding['queue'] == worker_plan['queue']
    assert binding['cases'] == 2
    assert binding['actual_results_seen'] == 0
    assert binding['completed_cases'] == 0
    assert binding['native_review'] == worker_plan['native_review']
    assert 'old_native_review' not in binding
    assert 'old_actual_results_seen' not in binding


def test_predecessor_review_preserves_explicit_failed_finish_without_calling_it_success(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import verify_predecessor_terminal

    queue = tmp_path / 'queue.jsonl'
    queue.write_text('{"key":"ok"}\n{"key":"failed"}\n')
    failed = {'event': 'finish', 'key': 'failed', 'status': 'incomplete', 'exit_code': 0,
              'output_accounting': {'complete': False, 'exact_result_rows': 0}}
    journal = tmp_path / 'journal.jsonl'
    journal.write_text('\n'.join([
        json.dumps({'event': 'finish', 'key': 'ok', 'status': 'complete', 'exit_code': 0,
                    'output_accounting': {'complete': True}}),
        json.dumps(failed),
    ]) + '\n')
    predecessor = {'queue': str(queue), 'queue_sha256': digest(queue), 'journal': str(journal)}
    event_sha = hashlib.sha256(json.dumps(failed, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    with pytest.raises(ValueError, match='needs agent review'):
        verify_predecessor_terminal(predecessor)
    with pytest.raises(ValueError, match='nonempty'):
        verify_predecessor_terminal(predecessor, {})
    result = verify_predecessor_terminal(predecessor, {'failed': event_sha})
    assert result == {
        'expected_keys': 2,
        'successful_keys': 1,
        'reviewed_failed_keys': ['failed'],
        'reviewed_failed_event_sha256': {'failed': event_sha},
        'reviewed_failed_events': {'failed': failed},
        'all_keys_accounted': True,
        'disposition': 'finished_with_reviewed_failures',
    }


def test_reviewed_failure_map_must_match_exactly_the_saved_incomplete_keys(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import verify_predecessor_terminal

    queue = tmp_path / 'queue.jsonl'
    queue.write_text('{"key":"failed"}\n')
    event = {'event': 'finish', 'key': 'failed', 'status': 'incomplete', 'exit_code': 0,
             'output_accounting': {'complete': False}}
    journal = tmp_path / 'journal.jsonl'
    journal.write_text(json.dumps(event) + '\n')
    predecessor = {'queue': str(queue), 'queue_sha256': digest(queue), 'journal': str(journal)}
    event_sha = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with pytest.raises(ValueError, match='exactly'):
        verify_predecessor_terminal(predecessor, {'failed': event_sha, 'unassigned': '0' * 64})
    with pytest.raises(ValueError, match='needs agent review'):
        verify_predecessor_terminal(predecessor, {'failed': '0' * 64})
