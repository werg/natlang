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
