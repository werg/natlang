"""The decode row tile changes padding only: exact arithmetic, not tolerance."""
import pytest
import torch
import triton
from types import SimpleNamespace
from natlang_neuralese.maple import fused_moe as fused

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA kernel qualification')


def plan(rows,block):
    return SimpleNamespace(size=rows,blocks=triton.cdiv(rows,block),block=block)


@pytest.mark.parametrize('dtype,precision',[(torch.float32,'ieee'),(torch.float32,'bf16x6'),
    (torch.float32,'tf32x3'),(torch.float32,'tf32'),(torch.float32,'bf16'),(torch.bfloat16,'ieee')])
def test_inference_linear_row_tile_preserves_forward_and_input_adjoint(dtype,precision):
    torch.manual_seed(721)
    for n,k,rows in [(4,32,1),(32,4,2),(63,65,7),(63,65,16),(63,65,17),
                     (256,128,1),(256,128,65),(1024,1024,1),
                     (2304,2304,1),(1792,2304,1),(2304,1792,2),(1792,2304,16),
                     (256,2304,1),(64000,1024,1)]:
        x=(torch.randn(rows,k,device='cuda')*.07).to(dtype)
        w=(torch.randn(n,k,device='cuda')*.02).to(dtype)
        gradient=(torch.randn(rows,n,device='cuda')*.03).to(dtype)
        for a,transpose in [(x,False),(gradient,True)]:
            old=fused._dense_mm(a,None,w[None],plan(rows,64),transpose,precision=precision,single_expert=True)
            new=fused._dense_mm(a,None,w[None],plan(rows,16),transpose,precision=precision,single_expert=True)
            assert torch.equal(old,new),(dtype,precision,n,k,rows,transpose,(old-new).abs().max().item())
    assert fused._invariant_forward_plan(1).block==64
    with torch.inference_mode():
        assert fused._invariant_forward_plan(1).block==16
        assert fused._invariant_forward_plan(17).block==64


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_decode_tile_preserves_whole_fanout_graph_and_every_parameter_gradient(family,monkeypatch):
    from test_shared_state import tiny,graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    port=tiny(family).cuda()
    if family=='mellum':port.bfloat16()
    model=SharedStateModel(port,(58,59,60,61),rank=4)
    torch.nn.init.normal_(model.adapter.up.weight,std=.03)
    parameters=tuple(p for p in model.parameters() if p.requires_grad)
    result=model(graph())['h_final']
    coefficients=torch.randn_like(result)
    old=torch.autograd.grad((result*coefficients).sum(),parameters,allow_unused=True)
    # Exercise the exact inference forward tile in a differentiable whole
    # producer/two-consumer graph; production training retains its64 tile.
    monkeypatch.setattr(fused,'_invariant_forward_plan',lambda rows:plan(rows,16))
    candidate=model(graph())['h_final']
    assert torch.equal(result,candidate)
    new=torch.autograd.grad((candidate*coefficients).sum(),parameters,allow_unused=True)
    for index,(a,b) in enumerate(zip(old,new)):
        if a is None:assert b is None
        else:assert torch.equal(a,b),(family,index,(a-b).abs().max().item())
