"""Inference removes bookkeeping/initialization only, keeping the shared math."""
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch
from natlang_neuralese.maple import fused_moe as fused

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA inference qualification')


@pytest.mark.parametrize('dtype,precision',[(torch.float32,'ieee'),(torch.float32,'bf16x6'),
    (torch.float32,'tf32x3'),(torch.float32,'tf32'),(torch.float32,'bf16'),(torch.bfloat16,'ieee')])
def test_poisoned_single_expert_outputs_are_fully_written_and_exact(dtype,precision,monkeypatch):
    torch.manual_seed(781)
    original_empty=torch.empty
    def poisoned(*args,**kwargs):
        value=original_empty(*args,**kwargs)
        if value.is_cuda and value.is_floating_point():value.fill_(float('nan'))
        return value
    for n,k,rows in [(4,32,1),(32,4,2),(63,65,17),(1792,2304,1),(2304,1792,16),(64000,1024,1)]:
        x=torch.randn(rows,k,device='cuda',dtype=dtype)*.03
        weight=torch.randn(n,k,device='cuda',dtype=dtype)*.02
        with torch.inference_mode():
            expected=fused._InvariantLinear.apply(x,weight,precision,None)
            with patch.object(torch,'empty',poisoned):
                actual=fused.invariant_linear(x,weight,precision=precision)
            assert torch.isfinite(actual).all()
            assert torch.equal(actual,expected),(dtype,precision,n,k,rows)


def test_inference_cast_and_autograd_dispatch_contracts(monkeypatch):
    x=torch.randn(3,128,device='cuda',dtype=torch.bfloat16)
    weight=torch.randn(64,128,device='cuda',dtype=torch.float32,requires_grad=True)
    original=fused._InvariantLinear.apply
    calls=[]
    def tracked(*args):calls.append(True);return original(*args)
    monkeypatch.setattr(fused._InvariantLinear,'apply',tracked)
    with torch.inference_mode():
        actual=fused.invariant_linear(x,weight,forward_storage_dtype=torch.bfloat16,
            cotangent_anchors=(torch.ones((),device='cuda'),))
        expected=original(x,weight,'ieee',torch.bfloat16,torch.ones((),device='cuda'))
        assert torch.equal(actual,expected)
    assert calls==[]
    with torch.no_grad():fused.invariant_linear(x,weight,forward_storage_dtype=torch.bfloat16)
    assert len(calls)==1
    result=fused.invariant_linear(x,weight,forward_storage_dtype=torch.bfloat16)
    assert len(calls)==2
    result.float().square().mean().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_fp32_output_keeps_the_accumulator_in_inference_only():
    """``output_dtype`` (FP32 head logits, shared_state.logits): the fixed-tile product leaves its FP32 accumulator
    unrounded (rounding it gives the BF16 product), in inference and, straight through the rounded product's gradient,
    under autograd."""
    torch.manual_seed(5)
    x=(torch.randn(19,256,device='cuda')*.5).to(torch.bfloat16)
    weight=(torch.randn(4000,256,device='cuda')*.5).to(torch.bfloat16)
    exact=x.double()@weight.double().t()
    with torch.inference_mode():
        rounded=fused.invariant_linear(x,weight)
        wide=fused.invariant_linear(x,weight,output_dtype=torch.float32)
        again=fused.invariant_linear(x[:7],weight,output_dtype=torch.float32)
    assert rounded.dtype==torch.bfloat16 and wide.dtype==torch.float32
    assert torch.equal(wide.to(torch.bfloat16),rounded)
    assert torch.equal(again,wide[:7])          # still batch invariant
    assert (wide.double()-exact).abs().max()<(rounded.double()-exact).abs().max()/8
    graph=fused.invariant_linear(x,weight,output_dtype=torch.float32)
    assert graph.dtype==torch.float32 and torch.equal(graph,wide)
    xg=x.clone().requires_grad_()
    trained=fused.invariant_linear(xg,weight,output_dtype=torch.float32)
    assert torch.equal(trained.detach(),wide)     # training reads the same logits, straight through the rounded product
    probe=torch.randn_like(trained)
    (trained*probe).sum().backward()
    xr=x.clone().requires_grad_()
    (fused.invariant_linear(xr,weight).float()*probe).sum().backward()
    assert torch.equal(xg.grad,xr.grad)


@pytest.mark.parametrize('family',['lfm'])
def test_adapted_native_producer_reader_receipts_match_autograd_forward(tmp_path,family,monkeypatch):
    from test_native_serving_adapters import engine_fixture,nonzero_adapter
    from test_native_adapted_rl import capture_actor
    monkeypatch.setenv('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cuda')
    engine=engine_fixture(tmp_path,family,service_template=True)
    adapter,_=nonzero_adapter(engine)
    entries=[{'id':adapter.id,'scale':.75}]
    actual_id,actual=capture_actor(engine,entries,monkeypatch)
    def reference(x,w,*,precision='ieee',forward_storage_dtype=None,cotangent_anchors=(),allow_plain_fastpath=True,
                  output_dtype=None):
        out=fused._InvariantLinear.apply(x,w,precision,forward_storage_dtype,*cotangent_anchors)
        if output_dtype is None or output_dtype==out.dtype:return out
        with torch.no_grad():   # FP32 head logits: the fixed-tile product unrounded (fused._widened)
            weight=w.to(forward_storage_dtype) if forward_storage_dtype is not None else w
            return fused._invariant_linear_forward(x,weight,precision,output_dtype)
    monkeypatch.setattr(fused,'invariant_linear',reference)
    expected_id,expected=capture_actor(engine,entries,monkeypatch)
    assert actual_id==expected_id
    assert actual==expected
    engine.store.close()


def test_grouped_padding_and_gradient_path_keep_zero_initialization(monkeypatch):
    import triton
    torch.manual_seed(17)
    x=torch.randn(1,65,device='cuda')
    weights=torch.randn(3,63,65,device='cuda')
    index=torch.tensor([[0,1]],device='cuda')
    grouped=fused.Plan(index,3,1,64)
    zeros=torch.zeros;empty=torch.empty;calls=[]
    def tracked(*args,**kwargs):calls.append(torch.is_inference_mode_enabled());return zeros(*args,**kwargs)
    def poisoned(*args,**kwargs):
        value=empty(*args,**kwargs)
        if value.is_cuda and value.is_floating_point():value.fill_(float('nan'))
        return value
    with patch.object(torch,'zeros',tracked),patch.object(torch,'empty',poisoned):
        with torch.inference_mode():
            result=fused._dense_mm(x,grouped.rows,weights,grouped,False,single_expert=False)
        assert torch.count_nonzero(result[-64:])==0
        assert calls==[True]
        single=SimpleNamespace(size=1,blocks=triton.cdiv(1,64),block=64)
        fused._dense_mm(x,None,weights[:1],single,False,single_expert=True)
        assert calls==[True,False]
        # Even an explicitly invoked adjoint inside an inference scope keeps
        # the old allocation unless this is the declared forward output.
        with torch.inference_mode():
            fused._dense_mm(x[:,:63],None,weights[:1],single,True,single_expert=True)
        assert calls==[True,False,True]
