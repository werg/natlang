import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'coreweave'))
import recovery_snapshots as snapshots


def save(path, value):
    path.write_text(json.dumps(value))


def test_ambiguous_capture_reuses_persisted_request_id_and_polls_without_new_capture(tmp_path):
    requests = []

    def capture(request):
        requests.append(request)
        if len(requests) == 1:
            raise TimeoutError('committed RPC, reply lost')
        return 'snap1'

    inspect = lambda _: {'source_sandbox_id': 'source', 'status': 'CREATING', 'size_bytes': 10}
    delete = lambda _: pytest.fail('no deletion before verified restore')
    with pytest.raises(TimeoutError):
        snapshots.run_once(tmp_path, 'source', capture=capture, inspect=inspect, delete=delete, now=0)
    snapshots.run_once(tmp_path, 'source', capture=capture, inspect=inspect, delete=delete, now=1)
    result = snapshots.run_once(tmp_path, 'source', capture=capture, inspect=inspect, delete=delete, now=2)
    assert requests[0] == requests[1] and len(requests) == 2
    assert result['pending']['snapshot_id'] == 'snap1'
    assert result['restore_verified'] is False


def test_ready_snapshot_is_not_restore_proof_and_foreign_snapshot_is_rejected(tmp_path):
    inspect = lambda _: {'source_sandbox_id': 'source', 'status': 'READY'}
    result = snapshots.run_once(tmp_path, 'source', capture=lambda _: 'snap1', inspect=inspect,
                                delete=lambda _: pytest.fail('unexpected deletion'), now=0)
    assert result['snapshots'][0]['status'] == 'READY'
    assert not result['restore_verified']
    with pytest.raises(ValueError, match='different sandbox'):
        snapshots.run_once(tmp_path, 'other', capture=None, inspect=None, delete=None)


def test_retention_requires_newer_verified_restore_and_protects_current_snapshots(tmp_path):
    state = {'sandbox_id': 'source', 'snapshots': [
        {'snapshot_id': f'snap{i}', 'status': 'READY', 'requested_at': i} for i in range(6)],
        'last_capture_at': 100, 'restore_verified': False}
    save(tmp_path / 'state.json', state)
    deleted = []
    snapshots.run_once(tmp_path, 'source', capture=None, inspect=None, delete=deleted.append, now=101)
    assert deleted == []
    save(tmp_path / 'verified-restore.json', {
        'snapshot_id': 'snap3', 'source_sandbox_id': 'source', 'destination_sandbox_id': 'restored',
        'passed': True, 'protected_snapshot_ids': ['snap1'],
        'verified_files': [{'path': '/work/checkpoint.pt', 'bytes': 3, 'sha256': 'a' * 64}]})

    def delete(snapshot_id):
        plan = json.loads((tmp_path / 'deletion-plan.json').read_text())
        assert snapshot_id in plan['delete_snapshot_ids']
        deleted.append(snapshot_id)

    result = snapshots.run_once(tmp_path, 'source', capture=None, inspect=None, delete=delete, now=102)
    assert deleted == ['snap0', 'snap2']
    assert result['restore_verified'] is True


def test_restore_proof_checks_actual_mount_origin_and_exact_destination_bytes():
    files = [{'path': '/work/checkpoint.pt', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}]
    result = snapshots.verify_restored_files('source', 'snap', 'dest', files,
        inspect_destination=lambda: ['snap'], execute=lambda _: files)
    assert result['passed'] is True
    with pytest.raises(ValueError, match='not created'):
        snapshots.verify_restored_files('source', 'snap', 'dest', files,
            inspect_destination=lambda: ['other'], execute=lambda _: files)
    with pytest.raises(ValueError, match='differ'):
        snapshots.verify_restored_files('source', 'snap', 'dest', files,
            inspect_destination=lambda: ['snap'], execute=lambda _: [])
    with pytest.raises(ValueError, match='separate destination'):
        snapshots.verify_restored_files('source', 'snap', 'source', files,
            inspect_destination=lambda: ['snap'], execute=lambda _: files)


def test_restore_hash_program_reads_actual_files_and_rejects_changed_content(tmp_path):
    import subprocess
    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'abc')
    # The transport invokes exactly this program on the restored mount. Exercise
    # its real hashing locally while retaining the public /work path contract.
    files = [{'path': '/work/checkpoint.pt', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}]

    def execute(argv):
        mapped = json.loads(argv[-1])
        mapped[0]['path'] = str(path)
        result = subprocess.run([*argv[:-1], json.dumps(mapped)], capture_output=True, text=True, check=True)
        actual = json.loads(result.stdout)
        actual[0]['path'] = '/work/checkpoint.pt'
        return actual

    assert snapshots.verify_restored_files('source', 'snap', 'dest', files,
        inspect_destination=lambda: ['snap'], execute=execute)['passed']
    path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='differ'):
        snapshots.verify_restored_files('source', 'snap', 'dest', files,
            inspect_destination=lambda: ['snap'], execute=execute)
