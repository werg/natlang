"""Actual backward/durability checks for the production arithmetic observer."""
import copy

import pytest
import torch

from scripts.neuralese_data.qualify_native_attention_arithmetic import paired_arithmetic_case
from scripts.neuralese_data.qualify_native_attention_index import diagnostic_optimizer
from natlang_neuralese.model import shared_state
from natlang_neuralese.model.native_precision import set_native_attention_precision
from test_shared_state import tiny
from test_native_optimizer_checkpointing import examples,assert_exact


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_real_arithmetic_observer_compares_all_adjoints_without_updates(tmp_path,family):
    model=shared_state.SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    set_native_attention_precision(model,'ieee')
    optimizer=diagnostic_optimizer(model)
    weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict())
    rng=torch.get_rng_state().clone()
    result=paired_arithmetic_case(model,optimizer,examples()[0],atol=1e-6,rtol=1e-5,
        checkpoint_sha256='a'*64,source_binding={},chunk_tokens=3,temporary_directory=tmp_path)
    assert result['passed'] and result['named_gradients']
    assert not result['missing_gradient_names']
    assert any(name.startswith('backbone.') for name in result['named_gradients'])
    assert any(name.startswith('interface.') for name in result['named_gradients'])
    assert result['seconds']['candidate']>0 and result['seconds']['reference']>0
    assert result['optimizer_updates']==0 and not result['task_quality_qualified']
    assert_exact(weights,model.state_dict());assert_exact(state,optimizer.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    assert model.native_attention_precision==model.port.native_attention_precision=='ieee'
    assert all(value.grad is None for value in model.parameters())
    assert not list(tmp_path.iterdir())


def test_real_candidate_gradient_mismatch_rejects_and_restores(tmp_path):
    model=shared_state.SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    set_native_attention_precision(model,'ieee');optimizer=diagnostic_optimizer(model)
    def changed(module,args,output):
        if module.native_attention_precision=='tf32x3':
            h=output['h_final']
            # Same values, wrong adjoint. Comparing forward/loss alone misses it.
            return {**output,'h_final':h*2-h.detach()}
    hook=model.register_forward_hook(changed)
    try:
        result=paired_arithmetic_case(model,optimizer,examples()[0],atol=1e-6,rtol=1e-5,
            checkpoint_sha256='a'*64,source_binding={},chunk_tokens=3,temporary_directory=tmp_path)
    finally:hook.remove()
    assert not result['passed'] and result['h_final']['passed'] and result['loss']['passed']
    assert any(not value['passed'] for value in result['named_gradients'].values())
    assert not result['seconds']
    assert model.native_attention_precision==model.port.native_attention_precision=='ieee'
    assert all(value.grad is None for value in model.parameters())
    assert not list(tmp_path.iterdir())
