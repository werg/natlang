"""Soft states exported as llama.cpp native state spans carry the exact per-layer state (plan §3.3, §7)."""
import dataclasses

import numpy as np
import pytest
import torch

from test_tune_rows import real_engine, real_lfm
from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.soft_export import _gguf, export_state_gguf


@real_lfm
def test_state_export_holds_exact_keys_values_and_gated_inputs(tmp_path):
    engine, _ = real_engine(tmp_path)
    text = 'Answer only with the capital city.'
    block = engine.store.get_native_state(engine.native_write_text(text)['id']).block
    path = tmp_path / 'state.gguf'
    export_state_gguf(block, path, state_id='nzs1_x', text_sha256='0' * 64, role_tokens=engine.role_tokens)
    reader = _gguf().GGUFReader(str(path))
    field = lambda key: reader.fields[key]
    value = lambda key: field(key).parts[field(key).data[0]]
    assert bytes(value('general.type')).decode() == 'natlang.native_state'
    assert int(value('natlang.state.n_tokens')[0]) == len(block.token_ids)
    assert int(value('natlang.state.consume_open')[0]) == engine.role_tokens.consume_open
    assert bytes(value('natlang.state.checkpoint_sha256')).decode() == block.checkpoint_sha256
    tensors = {tensor.name: np.array(tensor.data) for tensor in reader.tensors}
    count = len(block.token_ids)
    for index, layer in enumerate(block.layers):
        if layer['kind'] == 'attention':
            # rows of n_embd_k_gqa, head-major: llama.cpp's K/V layout; keys stay before RoPE
            assert np.array_equal(tensors[f'blk.{index}.state_k'], layer['raw_k'].float().reshape(count, -1).numpy())
            assert np.array_equal(tensors[f'blk.{index}.state_v'], layer['v'].float().reshape(count, -1).numpy())
            assert f'blk.{index}.state_conv' not in tensors
        else:
            assert layer['kind'] == 'finite_convolution'
            assert np.array_equal(tensors[f'blk.{index}.state_conv'], layer['gated_inputs'].float().numpy())
    assert len(tensors) == sum(2 if layer['kind'] == 'attention' else 1 for layer in block.layers)
    with pytest.raises(TuneMathError, match='supports lfm2'):
        export_state_gguf(dataclasses.replace(block, architecture='mellum'), tmp_path / 'other.gguf', state_id='nzs1_x',
                          text_sha256='0' * 64, role_tokens=engine.role_tokens)
