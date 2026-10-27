import hashlib

import pytest

from scripts.coreweave.verified_download import verify_download_blocks


def identity(payload):
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def test_one_remote_stream_yields_verified_local_recovery_without_replacing_parent(tmp_path):
    payload = b'earned optimizer, RNG and exposure\n' * 1000
    parent = tmp_path / 'previous.pt'
    parent.write_bytes(b'last independently verified parent')
    target = tmp_path / 'next.pt'
    consumed = []

    def stream():
        for offset in range(0, len(payload), 317):
            consumed.append(offset)
            yield payload[offset:offset + 317]

    result = verify_download_blocks(stream(), identity(payload), destination=target, sync_bytes=1024)
    assert result['local_copy']['committed'] is True
    assert target.read_bytes() == payload
    assert parent.read_bytes() == b'last independently verified parent'
    assert len(consumed) > 100
    assert not list(tmp_path.glob('.owner-verified-*'))


@pytest.mark.parametrize('corruption', ['changed', 'short', 'long', 'transport'])
def test_bad_or_interrupted_remote_bytes_never_publish_a_local_candidate(tmp_path, corruption):
    payload = b'full earned state'
    target = tmp_path / 'next.pt'

    def stream():
        if corruption == 'transport':
            yield payload[:4]
            raise ConnectionError('interrupted remote GET')
        yield {'changed': b'X' + payload[1:], 'short': payload[:-1], 'long': payload + b'X'}[corruption]

    with pytest.raises((ValueError, ConnectionError)):
        verify_download_blocks(stream(), identity(payload), destination=target, sync_bytes=1)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_optional_copy_space_failure_still_requires_complete_remote_verification(tmp_path):
    payload = b'earned state'
    result = verify_download_blocks(iter([payload]), identity(payload),
        destination=tmp_path / 'next.pt', reserve_bytes=1 << 100)
    assert result['sha256'] == identity(payload)['sha256']
    assert result['local_copy']['committed'] is False
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='SHA/size'):
        verify_download_blocks(iter([b'bad']), identity(payload),
            destination=tmp_path / 'next.pt', reserve_bytes=1 << 100)


def test_existing_candidate_is_not_overwritten_or_credited_as_verified(tmp_path):
    target = tmp_path / 'next.pt'
    target.write_bytes(b'candidate under another reader')
    payload = b'new remote object'
    result = verify_download_blocks(iter([payload]), identity(payload), destination=target)
    assert result['local_copy']['committed'] is False
    assert target.read_bytes() == b'candidate under another reader'


def test_local_writeback_failure_does_not_interrupt_remote_hash_check(tmp_path, monkeypatch):
    import os
    payload = b'first part and remaining remote bytes'
    consumed = []
    progress = []

    def stream():
        for part in [payload[:7], payload[7:]]:
            consumed.append(part)
            yield part

    def no_space(_fd):
        raise OSError('writeback failed')

    monkeypatch.setattr(os, 'fdatasync', no_space)
    result = verify_download_blocks(stream(), identity(payload), destination=tmp_path / 'next.pt',
        sync_bytes=1, on_progress=progress.append)
    assert b''.join(consumed) == payload
    assert result['sha256'] == identity(payload)['sha256']
    assert result['local_copy']['committed'] is False
    assert progress == [7, len(payload)]
    assert not list(tmp_path.iterdir())
