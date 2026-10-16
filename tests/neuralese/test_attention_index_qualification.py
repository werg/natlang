"""The paired diagnostic uses the real backward while retaining no training mutation."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.neuralese_data.qualify_native_attention_index import diagnostic_optimizer, paired_case, tensor_digest
from natlang_neuralese.model import shared_state
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from test_shared_state import tiny
from test_native_optimizer_checkpointing import examples, assert_exact


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_real_paired_backward_preserves_weights_optimizer_rng_and_dispatch(family):
    model = shared_state.SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4)
    if family == 'mellum':
        model.port.hf.natlang_precision = 'bf16'
    # Actual saved-parent qualifier loader freezes everything first.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = diagnostic_optimizer(model)
    weights = copy.deepcopy(model.state_dict())
    state = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    original = shared_state.forward_graph
    result = paired_case(model, optimizer, examples()[0], SimpleNamespace(forward_graph=original),
        chunk_tokens=3, checkpoint_sha256='a'*64, source_binding={})
    assert result['passed'] and result['gradient_parameters'] > 0
    assert any(name.startswith('backbone.') for name in result['observed_gradient_names'])
    assert any(name.startswith('interface.') for name in result['observed_gradient_names'])
    assert result['old_seconds'] > 0 and result['new_seconds'] > 0
    assert_exact(weights, model.state_dict())
    assert_exact(state, optimizer.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert shared_state.forward_graph is original
    assert all(parameter.grad is None for parameter in model.parameters())


def test_exact_digest_distinguishes_dtype_shape_and_bytes():
    value = torch.arange(6, dtype=torch.float32)
    assert tensor_digest(value) != tensor_digest(value.reshape(2, 3))
    assert tensor_digest(value) != tensor_digest(value + 1)
    with pytest.raises(ValueError, match='contiguous'):
        tensor_digest(value.reshape(2, 3).T)


def test_real_reference_output_mismatch_fails_and_restores_dispatch():
    model = shared_state.SharedStateModel(tiny('lfm'), (58, 59, 60, 61), rank=4)
    optimizer, _ = make_native_optimizer(model, lr=3e-6)
    original = shared_state.forward_graph
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        return {**result, 'h_final': result['h_final'] + .125}
    result = paired_case(model, optimizer, examples()[0], SimpleNamespace(forward_graph=changed),
        chunk_tokens=3, checkpoint_sha256='a'*64, source_binding={})
    assert not result['passed']
    assert shared_state.forward_graph is original
    assert all(parameter.grad is None for parameter in model.parameters())
