"""The atomic checkpoint's digest comes from actual written bytes, without rereads."""
import builtins
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
