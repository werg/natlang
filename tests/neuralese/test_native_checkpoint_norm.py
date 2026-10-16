"""A compiler fallback cannot alter checkpoint saved-tensor structure."""
import pytest
import torch
from test_shared_state import tiny,graph
from natlang_neuralese.model.shared_state import forward_graph


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_native_checkpoint_survives_compiled_norm_fallback(monkeypatch):
    from natlang_neuralese.maple import model as maple
    port=tiny('mellum').cuda();port.checkpoint_layers=True
    plan=graph();inputs=torch.randn(1,plan.token_count,port.embedding_weight.shape[1],device='cuda',requires_grad=True)
    compiled=torch.compile(maple._rms_norm,dynamic=True)
    fallback=False
    def dispatch(*args):
        return maple._rms_norm(*args) if fallback else compiled(*args)
    monkeypatch.setattr(maple,'_rms_norm_compiled',dispatch)
    output=forward_graph(port,plan,embeddings=inputs)['h_final']
    fallback=True
    torch.autograd.grad(output.square().sum(),(inputs,*(p for p in port.parameters() if p.requires_grad)),allow_unused=True)


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_checkpoint_stable_norm_exact_outputs_all_adjoints_and_fused_dispatch(monkeypatch,family,dtype):
    from natlang_neuralese.maple import model as maple
    from natlang_neuralese.maple import norm_kernels
    port=tiny(family).to(device='cuda',dtype=dtype);plan=graph()
    inputs=torch.randn(1,plan.token_count,port.embedding_weight.shape[1],device='cuda',dtype=dtype,requires_grad=True)
    upstream=torch.randn_like(inputs);parameters=(inputs,*(p for p in port.parameters() if p.requires_grad))
    calls=[];original=norm_kernels.fused_rms_norm
    def fused(*args):
        calls.append(args[0].dtype);return original(*args)
    monkeypatch.setattr(norm_kernels,'fused_rms_norm',fused)
    # A saturated compiler cannot affect checkpointed FP32 replay.
    monkeypatch.setattr(maple,'_rms_norm_compiled',lambda *args:pytest.fail('dynamic compiler reached a checkpointed norm'))
    port.checkpoint_layers=True
    actual=forward_graph(port,plan,embeddings=inputs)['h_final']
    forward_calls=len(calls)
    grad=torch.autograd.grad((actual*upstream).sum(),parameters,allow_unused=True)
    if family=='mellum' and dtype==torch.bfloat16:
        assert forward_calls>0 and len(calls)>forward_calls  # Both original and actual recompute fused.
    port.checkpoint_layers=False
    with maple.eager_rms_norm(compiled_only=True):
        expected=forward_graph(port,plan,embeddings=inputs)['h_final']
    expected_grad=torch.autograd.grad((expected*upstream).sum(),parameters,allow_unused=True)
    assert torch.equal(actual,expected)
    for value,reference in zip(grad,expected_grad):
        assert (value is None)==(reference is None)
        if value is not None:assert torch.equal(value,reference)
    assert maple._compiled_eager_depth.get()==0


def test_compiled_only_scope_is_nested_exception_safe_and_context_local():
    from contextvars import copy_context,Context
    from natlang_neuralese.maple import model as maple
    with pytest.raises(RuntimeError):
        with maple.eager_rms_norm(compiled_only=True):
            assert maple._compiled_eager_depth.get()==1
            assert Context().run(maple._compiled_eager_depth.get)==0
            with maple.eager_rms_norm(compiled_only=True):assert maple._compiled_eager_depth.get()==2
            raise RuntimeError('restore scope')
    assert maple._compiled_eager_depth.get()==0


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('mode',['tf32','bf16x6'])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA native normalization')
def test_native_norm_rule_matches_graph_and_real_incremental_serving(monkeypatch,dtype,mode):
    from natlang_neuralese.maple import model as maple
    from natlang_neuralese.model.shared_state import SharedStateModel,native_step,Invocation,compile_graph
    from natlang_neuralese.model.native_precision import set_native_linear_precision
    torch.manual_seed(93)
    model=SharedStateModel(tiny('mellum').to(device='cuda',dtype=dtype),(60,61,62,63),rank=4).cuda()
    set_native_linear_precision(model,mode)
    monkeypatch.setattr(maple,'_rms_norm_compiled',lambda *args:pytest.fail('native path used dynamic norm compilation'))
    plan=compile_graph([Invocation('task',(2,3,4,5))])
    ids=torch.tensor([[2,3,4,5]],device='cuda')
    first=native_step(model,ids[:,:2]);second=native_step(model,ids[:,2:],first['cache'])
    expected=torch.cat([first['h_final'],second['h_final']],dim=1)
    for checkpointed in [False,True]:
        model.port.checkpoint_layers=checkpointed
        flat=model(plan)['h_final'];cached=model(plan,reference=True)['h_final']
        torch.testing.assert_close(flat,cached,atol=5e-6,rtol=2e-5)
        torch.testing.assert_close(flat,expected,atol=5e-6,rtol=2e-5)
        torch.testing.assert_close(model.logits(flat),model.logits(expected),atol=5e-6,rtol=2e-5)
    # This is a native per-model contract, not a process-wide compiler disable.
    calls=[]
    monkeypatch.setattr(maple,'_rms_norm_compiled',lambda *args:(calls.append(True),maple._rms_norm(*args))[1])
    maple.RMSNorm(64,1e-5).cuda()(torch.randn(1,2,64,device='cuda'))
    assert calls
