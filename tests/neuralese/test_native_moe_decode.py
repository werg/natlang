"""Routed decode reuses native row tiles without changing real pairs or math."""
import torch
import pytest
from unittest.mock import patch
from natlang_neuralese.maple import fused_moe as fused

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA MoE qualification')


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('clamp',[None,.05])
def test_routed_tile_preserves_real_expert_outputs_and_every_gradient(dtype,clamp):
    torch.manual_seed(901)
    x=(torch.randn(2,64,device='cuda')*.03).to(dtype).requires_grad_()
    up=(torch.randn(4,64,64,device='cuda')*.02).to(dtype).requires_grad_()
    down=(torch.randn(4,64,32,device='cuda')*.02).to(dtype).requires_grad_()
    index=torch.tensor([[0,3],[3,1]],device='cuda')
    weights=torch.randn(2,2,device='cuda').softmax(-1).requires_grad_()
    parameters=(x,up,down,weights)
    old=fused.invariant_dense_experts(x,index,weights,up,down,clamp)
    coefficient=torch.randn_like(old)
    old_grad=torch.autograd.grad((old*coefficient).sum(),parameters)
    # Counterfactual differentiable 16-row path checks all expert contributions,
    # even though real gradient-enabled production retains its64-row Plan.
    with patch.object(fused,'_invariant_row_block',lambda rows:16):
        new=fused.invariant_dense_experts(x,index,weights,up,down,clamp)
        new_grad=torch.autograd.grad((new*coefficient).sum(),parameters)
    assert torch.equal(old,new)
    for i,(a,b) in enumerate(zip(old_grad,new_grad)):
        assert torch.equal(a,b),(dtype,clamp,i,(a-b).abs().max().item())


def test_real_mellum_expert_geometry_and_default_training_plan():
    torch.manual_seed(902)
    up=torch.randn(64,3584,2304,device='cuda',dtype=torch.bfloat16)*.02
    down=torch.randn(64,2304,1792,device='cuda',dtype=torch.bfloat16)*.02
    for tokens in [1,2]:
        x=torch.randn(tokens,2304,device='cuda',dtype=torch.bfloat16)*.03
        index=torch.stack([torch.randperm(64,device='cuda')[:8] for _ in range(tokens)])
        weights=torch.randn(tokens,8,device='cuda').softmax(-1)
        with torch.inference_mode():
            actual=fused.invariant_dense_experts(x,index,weights,up,down,None)
            with patch.object(fused,'_invariant_row_block',lambda rows:64):
                expected=fused.invariant_dense_experts(x,index,weights,up,down,None)
        assert torch.equal(actual,expected)
    assert fused._invariant_row_block(1)==64
    with torch.no_grad():assert fused._invariant_row_block(1)==64
    with torch.inference_mode():
        assert fused._invariant_row_block(8)==16
        assert fused._invariant_row_block(16)==16
        assert fused._invariant_row_block(24)==64


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_expert_tile_keeps_native_adapted_producer_reader_captures(tmp_path,family,monkeypatch):
    from test_native_serving_adapters import engine_fixture,nonzero_adapter
    from test_native_adapted_rl import capture_actor
    monkeypatch.setenv('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cuda')
    engine=engine_fixture(tmp_path,family,service_template=True)
    adapter,_=nonzero_adapter(engine);entries=[{'id':adapter.id,'scale':.75}]
    actual_id,actual=capture_actor(engine,entries,monkeypatch)
    with patch.object(fused,'_invariant_row_block',lambda rows:64):
        expected_id,expected=capture_actor(engine,entries,monkeypatch)
    assert actual_id==expected_id
    assert actual==expected
    engine.store.close()
