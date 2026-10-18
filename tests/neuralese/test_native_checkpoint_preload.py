from pathlib import Path
import copy
import pytest
import torch
from test_native_actor_view import source
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.model.native_model import load_native_checkpoint
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): exact(x, y)
    else: assert a == b


def test_both_native_ports_one_decode_exact_full_state_sparse_rows_and_rng(source, monkeypatch):
    engine, path, state, signature = source
    reference = load_native_checkpoint(path, device='cpu')
    original = CheckpointPolicy.load; calls = []
    def load(p, *a, **kw):
        calls.append(Path(p).resolve()); return original(p, *a, **kw)
    monkeypatch.setattr(CheckpointPolicy, 'load', staticmethod(load))
    rng = torch.get_rng_state().clone()
    owned = CheckpointPolicy.preload(path, expected_sha256=sha256_file_hex_cached(path))
    assert torch.equal(torch.get_rng_state(), rng)
    actual = load_native_checkpoint(path, device='cpu', preloaded_checkpoint=owned)
    assert calls == [path.resolve()]
    assert owned._state is None
    assert actual[2:4] == reference[2:4]
    exact(actual[4], state)
    assert actual[4]['training']['sft_progress'] == state['training']['sft_progress']
    actual_params = dict(actual[0].named_parameters())
    for name, value in reference[0].named_parameters(): exact(actual_params[name], value)
    exact(actual[0].input_rows, reference[0].input_rows)
    assert hasattr(actual[0], 'output_rows') == hasattr(reference[0], 'output_rows')
    if hasattr(actual[0], 'output_rows'):
        exact(actual[0].output_rows, reference[0].output_rows)
    with pytest.raises(ValueError, match='already consumed'):
        load_native_checkpoint(path, device='cpu', preloaded_checkpoint=owned)
    assert calls == [path.resolve()]


@pytest.mark.parametrize('negative', ['wrong-path', 'participant', 'changed-source', 'readonly'])
def test_preload_rejects_wrong_authority_without_another_decode(source, tmp_path, negative):
    engine, path, state, signature = source
    owned = CheckpointPolicy.preload(path, expected_sha256=sha256_file_hex_cached(path))
    if negative == 'wrong-path':
        other = tmp_path / 'same-bytes-other.pt'; other.write_bytes(path.read_bytes())
        with pytest.raises(ValueError, match='path or participant'):
            load_native_checkpoint(other, device='cpu', preloaded_checkpoint=owned)
    elif negative == 'participant':
        with pytest.raises(ValueError, match='path or participant'):
            load_native_checkpoint(path, participant='wrong', device='cpu', preloaded_checkpoint=owned)
    elif negative == 'changed-source':
        path.touch()
        with pytest.raises(ValueError, match='source changed'):
            load_native_checkpoint(path, device='cpu', preloaded_checkpoint=owned)
    else:
        with pytest.raises(ValueError, match='exact full-state restore'):
            load_native_checkpoint(path, device='cpu', allow_weights_only=True, preloaded_checkpoint=owned)
    assert owned._state is not None
