"""Whole-model seeds must not require whole-model serialization buffers."""
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from natlang_neuralese.train.diloco.protocol import save_tensors


def test_streamed_seed_casts_bounded_chunks_and_preserves_metadata(tmp_path, monkeypatch):
    source = torch.arange(5_000_000, dtype=torch.float32).to(torch.bfloat16)
    tensors = {'large': source, 'scalar': torch.tensor(3.), 'empty': torch.empty(0, 3),
               'transposed': torch.arange(12.).reshape(3, 4).t()}
    original_to = torch.Tensor.to
    casts = []

    def observed_to(self, *args, **kwargs):
        casts.append(self.numel() * 4)
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', observed_to)
    syncs = []
    original_sync = os.fdatasync

    def observed_sync(fd):
        syncs.append(os.lseek(fd, 0, os.SEEK_CUR))
        original_sync(fd)

    monkeypatch.setattr(os, 'fdatasync', observed_sync)
    path = tmp_path / 'seed.safetensors'
    save_tensors(path, tensors, {'version': '0'}, dtype=torch.float32)
    assert casts and max(casts) <= 16 << 20
    assert syncs and syncs[0] == 16 << 20
    loaded = load_file(path)
    for name, value in tensors.items():
        assert loaded[name].dtype == torch.float32
        assert torch.equal(loaded[name], value.float())
    with safe_open(path, framework='pt') as handle:
        assert handle.metadata() == {'version': '0'}


def test_failed_seed_write_preserves_existing_seed(tmp_path, monkeypatch):
    path = tmp_path / 'seed.safetensors'
    save_tensors(path, {'value': torch.tensor([1.])})
    original = path.read_bytes()

    def fail(_fd):
        raise OSError('injected sync failure')

    monkeypatch.setattr(os, 'fsync', fail)
    with pytest.raises(OSError, match='injected sync failure'):
        save_tensors(path, {'value': torch.tensor([2.])})
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ['seed.safetensors']


def test_streamed_seed_round_trips_integer_boolean_and_empty(tmp_path):
    tensors = {str(dtype): torch.tensor([0, 1], dtype=dtype) for dtype in (
        torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.float16, torch.bfloat16, torch.float32, torch.float64)}
    tensors['empty'] = torch.empty(0)
    path = tmp_path / 'seed.safetensors'
    save_tensors(path, tensors)
    loaded = load_file(path)
    assert all(torch.equal(value, loaded[name]) for name, value in tensors.items())
