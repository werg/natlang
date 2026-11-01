import hashlib
import os
import pytest
from scripts.coreweave import rl_cloud as cloud


def setup_upload(tmp_path, monkeypatch):
    path = tmp_path / 'full-state.bin'
    path.write_bytes(b'complete checkpoint bytes')
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    part = {'name': 'part0', 'offset': 0, 'bytes': len(data)}
    calls = {'digest': 0, 'put': 0, 'commit': 0}
    def initial(source):
        calls['digest'] += 1
        assert source.read_bytes() == data
        return digest, [part]
    def put(url, headers, source, offset, size, metadata):
        calls['put'] += 1
        assert source.read_bytes()[offset:offset + size] == data
        return True
    def commit(*args): calls['commit'] += 1
    monkeypatch.setattr(cloud.sandbox, 'digest_parts', initial)
    monkeypatch.setattr(cloud.sandbox, 'put_part', put)
    monkeypatch.setattr(cloud, 'missing_parts', lambda *args: [part])
    monkeypatch.setattr(cloud, 'upload_urls', lambda *args: {'part0': {'url': 'unused', 'headers': {}}})
    monkeypatch.setattr(cloud, 'commit_parts', commit)
    return path, digest, data, calls, initial, put, commit


def test_complete_upload_hashes_source_once_and_keeps_result(tmp_path, monkeypatch):
    path, digest, data, calls, *_ = setup_upload(tmp_path, monkeypatch)
    assert cloud.put_local({}, path) == (digest, len(data))
    assert calls == {'digest': 1, 'put': 1, 'commit': 1}


@pytest.mark.parametrize('phase', ['digest', 'put', 'commit'])
def test_changed_source_cannot_publish_success(tmp_path, monkeypatch, phase):
    path, _, data, calls, initial, put, commit = setup_upload(tmp_path, monkeypatch)
    def replace():
        other = path.with_suffix('.replacement')
        other.write_bytes(data)  # Identical bytes still require a new checked read.
        os.replace(other, path)
    if phase == 'digest':
        def changed(source):
            result = initial(source); replace(); return result
        monkeypatch.setattr(cloud.sandbox, 'digest_parts', changed)
    elif phase == 'put':
        def changed(*args):
            result = put(*args); replace(); return result
        monkeypatch.setattr(cloud.sandbox, 'put_part', changed)
    else:
        def changed(*args): commit(*args); replace()
        monkeypatch.setattr(cloud, 'commit_parts', changed)
    with pytest.raises(RuntimeError, match='changed during upload'):
        cloud.put_local({}, path)
    if phase != 'commit': assert calls['commit'] == 0


def test_committed_parts_still_check_current_source_identity(tmp_path, monkeypatch):
    path, _, data, calls, *_ = setup_upload(tmp_path, monkeypatch)
    def existing(*args):
        path.write_bytes(data + b'changed')
        return []
    monkeypatch.setattr(cloud, 'missing_parts', existing)
    with pytest.raises(RuntimeError, match='changed during upload'):
        cloud.put_local({}, path)
    assert calls['put'] == calls['commit'] == 0
