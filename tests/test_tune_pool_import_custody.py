import hashlib
import json
from pathlib import Path
import pytest

from scripts.neuralese_data import verify_tune_pool_import as custody


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _campaign(tmp_path: Path):
    root = tmp_path / 'campaign'
    source_row = {'version': 'natlang.program/2', 'id': 'tune:program:root',
                  'semantics': {'root': 'main.nl', 'inputs': {'value': 'x'}, 'expected': 'X'},
                  'source_groups': ['world-a'], 'split': 'train',
                  'tune': {'schema': 'natlang.tune-pool-cases/v1', 'program': 'program', 'buildHash': 'build',
                           'scaffold': 'scaffold-independent-of-student', 'split': 'split', 'root': 'root',
                           'partition': 'train', 'caseIndex': 0}}
    source = _write(root / 'source.jsonl', (json.dumps(source_row) + '\n').encode())
    runtime = root / 'runtime'
    receipt = _write(runtime / 'frozen-runtime.json', b'{"files":[]}\n')
    output = root / 'slot-01/results/case-000000.results.jsonl'
    manifest_path = Path(str(output) + '.manifest.json')
    queue_path = root / 'slot-01/queue/case-000000.jsonl'
    provenance = {'program_ir_sha256': _canonical(source_row), 'model': 'gpt-6-luna', 'provider': 'openai-codex',
                  'execution_plans': {'enabled': True}}
    row = {'version': 'natlang.teacher_trajectory.native/1', 'task': {'program_ir': source_row},
           'provenance': provenance, 'execution_identity': {'schema': 'natlang.teacher_execution_identity/1',
             'execution_run_id': 'run-1'}, 'trajectory': [{'phase': 'action'}], 'outcome': {'status': 'done', 'accepted': True}}
    output_bytes = (json.dumps(row, separators=(',', ':')) + '\n').encode()
    _write(output, output_bytes)
    manifest = {'version': 'natlang.teacher_batch.native/1', 'range': {'start': 0, 'count': 1}, 'completed': 1,
                'missing': [], 'output_sha256': _sha(output_bytes), 'source_sha256': _sha(source.read_bytes()),
                'source': str(source)}
    manifest_bytes = (json.dumps(manifest) + '\n').encode()
    _write(manifest_path, manifest_bytes)
    queue = {'key': 'campaign-case-000000', 'index': 0, 'seed': 17, 'count': 1, 'source': str(source),
             'source_row_sha256': _sha((json.dumps(source_row) + '\n').encode()), 'output': str(output)}
    queue_bytes = (json.dumps(queue) + '\n').encode()
    _write(queue_path, queue_bytes)
    cases = [{'index': 0, 'seed': 17, 'source_row_sha256': _sha((json.dumps(source_row) + '\n').encode())}]
    campaign_id = 'campaign'
    identity = {'campaign_id': campaign_id, 'campaign_root': str(root.resolve()), 'source': str(source.resolve()),
                'source_sha256': _sha(source.read_bytes()), 'runtime': str(runtime.resolve()),
                'runtime_receipt_sha256': _sha(receipt.read_bytes()), 'ledger': str((root / 'dispatch/claims.jsonl').resolve()),
                'launch_record': str((root / 'dispatch/launch.json').resolve()), 'slots': 1, 'cases': cases}
    identity['identity_sha256'] = _canonical(identity)
    claim_id = 'campaign-case-000000'
    ledger = [
        {'event': 'campaign_open', 'identity': identity},
        {'event': 'claim', 'identity_sha256': identity['identity_sha256'], 'claim_id': claim_id, 'index': 0,
         'source': identity['source'], 'runtime': identity['runtime'], 'queue': str(queue_path.resolve()),
         'queue_sha256': _sha(queue_bytes), 'slot': 1},
        {'event': 'terminal', 'identity_sha256': identity['identity_sha256'], 'claim_id': claim_id, 'index': 0,
         'queue': str(queue_path.resolve()), 'queue_sha256': _sha(queue_bytes), 'exit_code': 0,
         'output_accounting': {'complete': True, 'manifest_valid': True, 'exact_result_rows': 1, 'output_rows': 1,
                               'output_error': None, 'output_row_errors': [], 'output_sha256': _sha(output_bytes)}}]
    ledger_bytes = ''.join(json.dumps(item) + '\n' for item in ledger).encode()
    _write(Path(identity['ledger']), ledger_bytes)
    launch = {'status': 'running', 'identity': identity}
    launch_bytes = (json.dumps(launch) + '\n').encode()
    _write(Path(identity['launch_record']), launch_bytes)
    plan = root / 'plan.json'
    _write(plan, b'{"reviewed":true}\n')
    return root, plan, output, identity, row, source_row


def _bind_fixture_dispatcher(monkeypatch, identity, plan=None):
    class FixtureDispatcher:
        @staticmethod
        def validate_plan(path, expected_sha256):
            raw_identity = {key: value for key, value in identity.items() if key != 'identity_sha256'}
            return plan or {'provider': 'openai-codex', 'model_id': 'gpt-6-luna'}, raw_identity

        @staticmethod
        def _verify_ledger_identity(events, bound):
            return None

        @staticmethod
        def _verify_ledger_events(events, bound):
            return None

    monkeypatch.setattr(custody, '_load_bound_dispatcher', lambda path: FixtureDispatcher)


def test_verified_campaign_import_binds_exact_row_source_actor_and_runtime(tmp_path, monkeypatch):
    root, plan, output, identity, _, _ = _campaign(tmp_path)
    _bind_fixture_dispatcher(monkeypatch, identity)
    proof = custody.verify_import(output, snapshot_dir=root / 'custody-snapshots')
    assert proof['plan']['sha256'] == _sha(plan.read_bytes())
    assert proof['selected_rows'][0]['result_file']['sha256'] == _sha(output.read_bytes())
    assert proof['selected_rows'][0]['result_line_sha256'] == _sha(output.read_bytes())
    assert proof['selected_rows'][0]['source_row_sha256'] == identity['cases'][0]['source_row_sha256']
    assert proof['selected_rows'][0]['actor']['provider'] == 'openai-codex'
    assert proof['selected_rows'][0]['actor']['runtime_receipt_sha256'] == _sha((root / 'runtime/frozen-runtime.json').read_bytes())
    for name in ('claim_ledger', 'launch_record'):
        evidence = proof[name]
        assert evidence['observed']['path'].endswith('claims.jsonl' if name == 'claim_ledger' else 'launch.json')
        assert evidence['observed']['prefix']['sha256'] == evidence['snapshot']['sha256']
        assert Path(evidence['snapshot']['path']).read_bytes() == Path(evidence['observed']['path']).read_bytes()


def test_closed_evidence_snapshot_stays_exact_when_live_campaign_advances(tmp_path, monkeypatch):
    root, _, output, identity, _, _ = _campaign(tmp_path)
    _bind_fixture_dispatcher(monkeypatch, identity)
    proof = custody.verify_import(output, snapshot_dir=root / 'custody-snapshots')
    ledger = proof['claim_ledger']
    launch = proof['launch_record']
    Path(ledger['observed']['path']).write_bytes(Path(ledger['observed']['path']).read_bytes() + b'{"event":"later"}\n')
    Path(launch['observed']['path']).write_text('{"status":"finished","later":true}\n')
    assert _sha(Path(ledger['snapshot']['path']).read_bytes()) == ledger['snapshot']['sha256']
    assert _sha(Path(launch['snapshot']['path']).read_bytes()) == launch['snapshot']['sha256']
    assert ledger['observed']['sha256'] != _sha(Path(ledger['observed']['path']).read_bytes())


def test_directory_input_binds_only_terminal_result_files_present_in_its_closed_case_cohort(tmp_path, monkeypatch):
    root, _, output, identity, _, _ = _campaign(tmp_path)
    _bind_fixture_dispatcher(monkeypatch, identity)
    selected = custody._selected_result_files

    def selected_then_new_result(path, campaign_root):
        files = selected(path, campaign_root)
        _write(campaign_root / 'slot-02/results/case-000099.results.jsonl', b'{"later":"not selected"}\n')
        return files

    monkeypatch.setattr(custody, '_selected_result_files', selected_then_new_result)
    proof = custody.verify_import(root, campaign_root=root, snapshot_dir=root / 'custody-snapshots')
    assert proof['input']['kind'] == 'directory'
    assert proof['input']['files'] == [{'path': str(output.resolve()), 'bytes': output.stat().st_size,
                                        'sha256': _sha(output.read_bytes())}]


def test_detached_result_file_keeps_original_campaign_output_and_row_bindings(tmp_path, monkeypatch):
    root, _, output, identity, _, _ = _campaign(tmp_path)
    _bind_fixture_dispatcher(monkeypatch, identity)
    detached = _write(tmp_path / 'merged-results.jsonl', output.read_bytes())
    proof = custody.verify_import(detached, campaign_root=root, snapshot_dir=root / 'custody-snapshots')
    row = proof['selected_rows'][0]
    assert proof['input'] == {'kind': 'file', 'path': str(detached.resolve()),
                              'bytes': detached.stat().st_size, 'sha256': _sha(detached.read_bytes())}
    assert row['input'] == {'path': str(detached.resolve()), 'line': 1}
    assert row['result_file']['path'] == str(output.resolve())
    assert row['result_line_sha256'] == _sha(detached.read_bytes())


def test_verified_campaign_import_rejects_changed_output_and_unclaimed_rows(tmp_path, monkeypatch):
    _, _, output, identity, row, _ = _campaign(tmp_path)
    _bind_fixture_dispatcher(monkeypatch, identity)
    output.write_bytes(output.read_bytes() + (json.dumps(row) + '\n').encode())
    import pytest
    with pytest.raises(ValueError, match='differ|changed'):
        custody.verify_import(output, snapshot_dir=output.parent / 'custody-snapshots')


def test_pinned_dispatcher_artifact_change_after_review_is_rejected(tmp_path, monkeypatch):
    _, _, output, identity, _, _ = _campaign(tmp_path)
    root = output.parents[2]
    dispatcher_path = _write(root / 'scripts/run_reviewed_luna_dispatcher.py', b'original dispatcher bytes')
    pins = {str(dispatcher_path): _sha(dispatcher_path.read_bytes())}
    def reviewed_then_changed(path, digest):
        dispatcher_path.write_bytes(b'changed dispatcher bytes')
        return {'artifact_hashes': pins}, identity
    _bind_fixture_dispatcher(monkeypatch, identity, {'artifact_hashes': pins})
    class ChangedAfterReview:
        @staticmethod
        def validate_plan(path, digest):
            reviewed_plan, reviewed_identity = reviewed_then_changed(path, digest)
            return reviewed_plan, {key: value for key, value in reviewed_identity.items() if key != 'identity_sha256'}
        _verify_ledger_identity = staticmethod(lambda events, bound: None)
        _verify_ledger_events = staticmethod(lambda events, bound: None)
    monkeypatch.setattr(custody, '_load_bound_dispatcher', lambda path: ChangedAfterReview)
    import pytest
    with pytest.raises(ValueError, match='artifact changed after plan validation'):
        custody.verify_import(output, snapshot_dir=output.parent / 'custody-snapshots')


def test_dispatcher_rejects_mismatched_canonical_dispatcher_hash(tmp_path, monkeypatch):
    canonical_root = tmp_path / 'canonical'
    scripts = canonical_root / 'scripts'
    scripts.mkdir(parents=True)
    dispatcher_path = _write(scripts / 'run_reviewed_luna_dispatcher.py', b'actual canonical dispatcher')
    runner_path = _write(scripts / 'run_bonsai_queue.py', b'actual queue runner')
    source_path = _write(canonical_root / 'source.jsonl', b'{"semantics":{"oracle":"exact"}}\n')
    runtime_root = canonical_root / 'runtime'
    runtime_root.mkdir()
    receipt_path = _write(runtime_root / 'frozen-runtime.json', b'{"files":[]}\n')
    candidate = Path(custody.REPOSITORY_ROOT) / 'scripts/run_reviewed_luna_dispatcher.py'
    expected_dispatcher_sha = _sha(candidate.read_bytes())
    plan = {
        'schema': 'natlang.reviewed_luna_dispatch_plan/1', 'root_approved': True,
        'campaign_id': 'hash-mismatch', 'provider': 'openai-codex', 'model_id': 'gpt-6-luna',
        'model_concurrency': 1, 'slots': 1, 'execution_plans': True,
        'cwd': str(canonical_root), 'supervisor': str(runner_path),
        'source': str(source_path), 'runtime': str(runtime_root),
        'dispatcher_sha256': expected_dispatcher_sha,
        'source_sha256': _sha(source_path.read_bytes()),
        'supervisor_sha256': _sha(runner_path.read_bytes()),
        'runtime_receipt_sha256': _sha(receipt_path.read_bytes()),
        'artifact_hashes': {
            str(dispatcher_path): expected_dispatcher_sha,
            str(source_path): _sha(source_path.read_bytes()),
            str(runner_path): _sha(runner_path.read_bytes()),
            str(receipt_path): _sha(receipt_path.read_bytes()),
        },
    }
    plan_path = _write(canonical_root / 'plan.json', json.dumps(plan).encode())
    with pytest.raises(ValueError, match='canonical dispatcher bytes differ'):
        custody._load_bound_dispatcher(plan_path)
