import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/coreweave'))
import fetch_native_eval_export as fetcher


class FakeResponse:
    def __init__(self, status, body, *, headers=None, error_after=None):
        self.status_code = status
        self.headers = {'Content-Length': str(len(body)), **(headers or {})}
        self.body = body
        self.error_after = error_after

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, _chunk_size):
        if self.error_after is None:
            if self.body:
                yield self.body
            return
        yield self.body[:self.error_after]
        raise OSError('interrupted fixture stream')


class FakeRequests:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class FakeApi:
    def run(self, _run):
        return self

    def file(self, _name):
        return SimpleNamespace(url='https://private.invalid/scoped')


def receipt(data):
    identity = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return {'private': True, 'role': 'mellum', 'identity': identity,
            'independent_verification': dict(identity), 'run': 'entity/project/run', 'name': 'export.bin'}


def fetch_with(destination, data, response, **kwargs):
    os.environ['WANDB_API_KEY'] = 'fixture-secret'
    requests = FakeRequests([response])
    result = fetcher.fetch(receipt(data), destination, requests_client=requests, wandb_api=FakeApi(), **kwargs)
    return result, requests


def test_partial_stream_resumes_from_verified_prefix_with_exact_range(tmp_path):
    data = b'abcdef'
    destination = tmp_path / 'export.bin'
    os.environ['WANDB_API_KEY'] = 'fixture-secret'
    interrupted = FakeRequests([FakeResponse(200, data, error_after=3)])
    with pytest.raises(OSError):
        fetcher.fetch(receipt(data), destination, requests_client=interrupted, wandb_api=FakeApi())
    pending = destination.with_name(destination.name + '.pending')
    assert pending.read_bytes() == data[:3]
    assert not destination.exists()

    response = FakeResponse(206, data[3:], headers={'Content-Range': 'bytes 3-5/6'})
    result, requests = fetch_with(destination, data, response)
    assert requests.calls[0][1]['headers'] == {'Range': 'bytes=3-'}
    assert destination.read_bytes() == data
    assert not pending.exists()
    assert result['resumed_bytes'] == 3


def test_ignored_range_200_restarts_at_byte_zero(tmp_path):
    data = b'complete export'
    destination = tmp_path / 'export.bin'
    pending = destination.with_name(destination.name + '.pending')
    pending.write_bytes(b'wrong prefix')
    result, requests = fetch_with(destination, data, FakeResponse(200, data))
    assert requests.calls[0][1]['headers']['Range'] == f"bytes={len(b'wrong prefix')}-"
    assert destination.read_bytes() == data
    assert result['sha256'] == hashlib.sha256(data).hexdigest()


def test_malformed_206_does_not_change_partial_prefix(tmp_path):
    data = b'abcdef'
    destination = tmp_path / 'export.bin'
    pending = destination.with_name(destination.name + '.pending')
    pending.write_bytes(data[:2])
    requests = FakeRequests([FakeResponse(206, data[2:], headers={'Content-Range': 'bytes 1-5/6'})])
    with pytest.raises(ValueError, match='range response'):
        fetcher.fetch(receipt(data), destination, requests_client=requests, wandb_api=FakeApi())
    assert pending.read_bytes() == data[:2]
    assert not destination.exists()


def test_complete_pending_is_verified_and_published_without_network(tmp_path):
    data = b'complete export'
    destination = tmp_path / 'export.bin'
    pending = destination.with_name(destination.name + '.pending')
    pending.write_bytes(data)

    class NoNetwork:
        def get(self, *_args, **_kwargs):
            raise AssertionError('complete pending file should publish without network')

    result = fetcher.fetch(receipt(data), destination, requests_client=NoNetwork(), wandb_api=FakeApi())
    assert destination.read_bytes() == data
    assert not pending.exists()
    assert result['resumed_bytes'] == len(data)


def test_verified_destination_wins_over_partial_and_skips_network(tmp_path):
    data = b'complete export'
    destination = tmp_path / 'export.bin'
    destination.write_bytes(data)
    pending = destination.with_name(destination.name + '.pending')
    pending.write_bytes(b'stale')

    class NoNetwork:
        def get(self, *_args, **_kwargs):
            raise AssertionError('verified destination should take precedence')

    result = fetcher.fetch(receipt(data), destination, requests_client=NoNetwork(), wandb_api=FakeApi())
    assert destination.read_bytes() == data
    assert pending.read_bytes() == b'stale'
    assert result['reused_destination'] is True


def test_invalid_full_pending_restarts_and_bad_final_hash_is_not_published(tmp_path):
    data = b'abcdef'
    destination = tmp_path / 'export.bin'
    pending = destination.with_name(destination.name + '.pending')
    pending.write_bytes(b'badbad')
    wrong = FakeResponse(200, b'xxxxxx')
    requests = FakeRequests([wrong])
    with pytest.raises(ValueError, match='byte/hash mismatch'):
        fetcher.fetch(receipt(data), destination, requests_client=requests, wandb_api=FakeApi())
    assert requests.calls[0][1]['headers'] == {}
    assert not pending.exists()
    assert not destination.exists()


def test_verified_existing_destination_is_never_overwritten(tmp_path):
    destination = tmp_path / 'export.bin'
    destination.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='existing private export destination'):
        fetcher.fetch(receipt(b'expected'), destination, requests_client=FakeRequests([]), wandb_api=FakeApi())
    assert destination.read_bytes() == b'wrong'
