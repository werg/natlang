"""The atomic checkpoint's digest comes from actual written bytes, without rereads."""
import builtins
import io
import os
from pathlib import Path

import pytest
import torch

from natlang_neuralese.common import hashing
from natlang_neuralese.train import trajectory_state as state


def test_full_optimizer_checkpoint_has_both_verified_cache_entries(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path/'hashes.jsonl'))
    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter])
    parameter.square().sum().backward(); optimizer.step()
    saved = {'weights': parameter.detach().clone(), 'optimizer': optimizer.state_dict(),
             'rng': torch.get_rng_state(), 'exposure': {'actual-row': 3}}
    path = tmp_path/'checkpoint.pt'
    state._atomic_checkpoint(path, saved)
    expected = hashing.sha256_file_hex(path)
    loaded = torch.load(path, mmap=True, weights_only=True)
    assert torch.equal(loaded['weights'], saved['weights'])
    assert torch.equal(loaded['rng'], saved['rng'])
    assert torch.equal(loaded['optimizer']['state'][0]['exp_avg'], optimizer.state[parameter]['exp_avg'])
    assert loaded['exposure'] == saved['exposure']
    original = builtins.open
    def no_body_read(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and Path(file).resolve() == path.resolve():
            raise AssertionError('a committed writer digest must avoid another checkpoint read')
        return original(file, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', no_body_read)
    assert hashing.sha256_file_hex_cached(path, trust_archive_receipt=False) == expected
    assert hashing.sha256_file_hex_cached(path) == expected


def test_next_atomic_save_replaces_both_digest_identities(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path/'hashes.jsonl'))
    path = tmp_path/'checkpoint.pt'
    state._atomic_checkpoint(path, {'v': torch.ones(128)})
    first = hashing.sha256_file_hex_cached(path, trust_archive_receipt=False)
    state._atomic_checkpoint(path, {'v': torch.zeros(128)})
    second = hashing.sha256_file_hex(path)
    assert second != first
    assert hashing.sha256_file_hex_cached(path, trust_archive_receipt=False) == second
    assert hashing.sha256_file_hex_cached(path) == second


def test_mutated_same_size_same_mtime_file_cannot_reuse_writer_digest(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path/'hashes.jsonl'))
    path = tmp_path/'checkpoint.pt'
    state._atomic_checkpoint(path, {'v': torch.ones(128)})
    before = path.stat(); first = hashing.sha256_file_hex_cached(path, trust_archive_receipt=False)
    with path.open('r+b') as stream:
        stream.seek(100); stream.write(b'changed')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    actual = hashing.sha256_file_hex(path)
    assert first != actual
    assert hashing.sha256_file_hex_cached(path, trust_archive_receipt=False) == actual


@pytest.mark.parametrize('mode', ['partial', 'seek'])
def test_digest_writer_rejects_incomplete_or_nonsequential_writes(tmp_path, mode):
    path = tmp_path/'partial'
    with path.open('wb') as stream:
        if mode == 'seek':
            writer = state._CheckpointDigestWriter(stream)
            writer.write(b'valid'); stream.seek(0)
            with pytest.raises(OSError):writer.write(b'bad')
        else:
            class Partial:
                def tell(self):return stream.tell()
                def write(self, data):return stream.write(data[:2])
            writer = state._CheckpointDigestWriter(Partial())
            with pytest.raises(OSError):writer.write(b'truncated')
        assert writer.bytes == (5 if mode == 'seek' else 0)


def test_digest_writer_syncs_bounded_windows_and_preserves_exact_bytes(tmp_path, monkeypatch):
    path = tmp_path/'windows.bin'
    sync_positions = []
    drop_positions = []
    real_fdatasync = os.fdatasync
    with path.open('wb') as stream:
        writer = state._CheckpointDigestWriter(stream, sync_window_bytes=8)
        def record_sync(fd):
            real_fdatasync(fd)
            sync_positions.append(os.fstat(fd).st_size)
        monkeypatch.setattr(os, 'fdatasync', record_sync)
        monkeypatch.setattr(state, 'drop_file_cache', lambda fd: drop_positions.append(os.fstat(fd).st_size))
        assert writer.write(b'1234567') == 7
        assert writer.write(b'89xyz123') == 8
        assert writer.write(b'456789') == 6
        stream.flush()
        assert writer.digest.digest() == __import__('hashlib').sha256(b'123456789xyz123456789').digest()
        assert sync_positions == [8, 16]
        assert drop_positions == [8, 16]
    assert path.read_bytes() == b'123456789xyz123456789'


def test_atomic_torch_save_crosses_windows_byte_exactly_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path/'hashes.jsonl'))
    monkeypatch.setattr(state, 'CHECKPOINT_SYNC_WINDOW_BYTES', 1024)
    sync_count = 0
    real_fdatasync = os.fdatasync
    def record_sync(fd):
        nonlocal sync_count
        sync_count += 1
        return real_fdatasync(fd)
    monkeypatch.setattr(os, 'fdatasync', record_sync)
    dropped = []
    monkeypatch.setattr(state, 'drop_file_cache', lambda fd: dropped.append(os.fstat(fd).st_size))
    parameter = torch.nn.Parameter(torch.arange(4096, dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter])
    parameter.square().sum().backward(); optimizer.step()
    saved = {'weights_bf16': parameter.detach().to(torch.bfloat16),
        'weights_fp32': parameter.detach(), 'optimizer': optimizer.state_dict(),
        'rng': torch.get_rng_state(), 'exposure': {'row-a': 4}}
    expected_stream = io.BytesIO()
    torch.save(saved, state._CheckpointDigestWriter(expected_stream, sync_window_bytes=0))
    path = tmp_path/'checkpoint.pt'
    state._atomic_checkpoint(path, saved)
    assert path.read_bytes() == expected_stream.getvalue()
    loaded = torch.load(path, mmap=True, weights_only=True)
    assert torch.equal(loaded['weights_bf16'], saved['weights_bf16'])
    assert torch.equal(loaded['weights_fp32'], saved['weights_fp32'])
    assert torch.equal(loaded['optimizer']['state'][0]['exp_avg'], optimizer.state[parameter]['exp_avg'])
    assert torch.equal(loaded['rng'], saved['rng']) and loaded['exposure'] == saved['exposure']
    assert sync_count >= 2 and len(dropped) == sync_count + 1


def test_window_sync_failure_keeps_old_checkpoint_and_removes_pending(tmp_path, monkeypatch):
    path = tmp_path/'checkpoint.pt'
    path.write_bytes(b'old committed bytes')
    before = path.read_bytes()
    monkeypatch.setattr(state, 'CHECKPOINT_SYNC_WINDOW_BYTES', 1024)
    real_fdatasync = os.fdatasync
    calls = 0
    def fail_second(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('fixture fdatasync failure')
        return real_fdatasync(fd)
    monkeypatch.setattr(os, 'fdatasync', fail_second)
    with pytest.raises((OSError, RuntimeError)) as error:
        state._atomic_checkpoint(path, {'large': torch.arange(16384, dtype=torch.float32)})
    errors = []
    pending_errors = [error.value]
    while pending_errors:
        item = pending_errors.pop()
        if item in errors:
            continue
        errors.append(item)
        pending_errors.extend(child for child in (item.__cause__, item.__context__) if child is not None)
    assert any('fixture fdatasync failure' in str(item) for item in errors)
    assert path.read_bytes() == before
    assert not path.with_suffix('.pending').exists()
