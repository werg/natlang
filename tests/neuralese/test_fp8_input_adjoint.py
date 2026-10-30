"""Shared expert-adjoint precision is explicit, durable, and numerically checked."""
import pytest
import torch
from natlang_neuralese.train.fp8_backend import (
    FP8ExpertAdapter, expert_input_gradient_precision, expert_precision_transition)


def adapter(mode='fp8'):
    value=object.__new__(FP8ExpertAdapter)
    value.input_gradient_precision=mode;value.precision_transition=None
    value.experts=['model.layers.0.mlp.experts'];value.row_counts={'calls':0,'rows':0,'padded_rows':0}
    value.runtime={};value.has_executed=False
    return value


def test_expert_precision_is_bound_and_legacy_state_has_explicit_fp8_meaning():
    baseline=adapter();saved=baseline.extra_state()
    assert saved['input_gradient_precision']=='fp8'
    selected=adapter('bf16');selected.restore_extra_state(saved)
    assert selected.input_gradient_precision=='bf16'
    assert selected.precision_transition=={'from':'fp8','to':'bf16','stateless':True}
    resumed=adapter('bf16');resumed.restore_extra_state(selected.extra_state())
    assert resumed.extra_state()==selected.extra_state()
    legacy={k:v for k,v in saved.items() if k not in {'input_gradient_precision','precision_transition'}}
    selected.restore_extra_state(legacy)
    assert selected.precision_transition['from']=='fp8'
    for bad in ({**saved,'input_gradient_precision':'unknown'}, {**saved,'backend':'wrong'},
                {**saved,'module_paths':['different']}):
        with pytest.raises(ValueError):selected.restore_extra_state(bad)
    inherited=adapter('fp8');inherited.precision_explicit=False
    inherited.restore_extra_state(selected.extra_state())
    assert inherited.input_gradient_precision=='bf16'
    assert expert_input_gradient_precision('current-bf16-dx')=='bf16'
    old={'kind':'triton-experts','recipe':'current','amax_history_len':16}
    new={**old,'recipe':'current-bf16-dx'}
    assert expert_precision_transition(old,new)['to']=='bf16'
    assert expert_precision_transition(old,{**new,'kind':'te-fp8'}) is None
    assert expert_precision_transition(old,{**new,'amax_history_len':4}) is None


def test_checkpoint_captures_actual_adjoint_mode_and_restores_outer_scope():
    from natlang_neuralese.maple.model import DENSE_MOE
    chosen=adapter('bf16');seen=[];original=dict(DENSE_MOE)
    x=torch.tensor([2.,3.],requires_grad=True)
    def function(value):
        seen.append(DENSE_MOE['fp8_input_gradient_precision'])
        return value.sin().square()
    with chosen.autocast():y=chosen.checkpoint(function,x)
    assert DENSE_MOE==original
    y.sum().backward()
    assert seen==['bf16','bf16'] and DENSE_MOE==original
    assert not chosen.has_executed


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
@pytest.mark.parametrize('counts',[(5,0,7),(129,1,0)])
def test_bf16_input_adjoint_matches_actual_quantized_weight_oracle_and_preserves_forward_dw(counts):
    from natlang_neuralese.maple.model import DENSE_MOE
    from natlang_neuralese.maple.expert_gemm import BlockTable,Fp8ExpertProjection
    from natlang_neuralese.maple.fp8_experts import rowwise_quant
    torch.manual_seed(43)
    rows=sum(counts);width,out_width=80,96
    offsets=torch.tensor(counts,device='cuda',dtype=torch.int32).cumsum(0).int()
    table=BlockTable(offsets,rows)
    a=torch.randn(rows,width,device='cuda',dtype=torch.bfloat16)
    w=(torch.randn(3,out_width,width,device='cuda')*.05).bfloat16()
    g=torch.randn(rows,out_width,device='cuda',dtype=torch.bfloat16)
    outputs={};original=dict(DENSE_MOE)
    try:
        for mode in ('fp8','bf16'):
            DENSE_MOE['fp8_input_gradient_precision']=mode
            x=a.detach().clone().requires_grad_();weight=w.detach().clone().requires_grad_()
            y=Fp8ExpertProjection.apply(x,None,None,1,weight,table,offsets,None)
            # Backward must use the captured forward mode, not ambient state.
            DENSE_MOE['fp8_input_gradient_precision']='fp8' if mode=='bf16' else 'bf16'
            dx,dw=torch.autograd.grad(y,(x,weight),g)
            outputs[mode]=(y.detach(),dx,dw)
        assert torch.equal(outputs['fp8'][0],outputs['bf16'][0])
        assert torch.equal(outputs['fp8'][2],outputs['bf16'][2])
        q,scale=rowwise_quant(w.flatten(0,1));q=q.view_as(w);scale=scale.view(3,out_width)
        expected=torch.empty_like(a);start=0;exact=[];roundoff=[]
        for expert,count in enumerate(counts):
            stop=start+count
            dequantized=q[expert].double()*scale[expert,:,None].double()
            # The chosen kernel's only additional weight rounding is explicit.
            expected[start:stop]=(g[start:stop].double() @ dequantized.bfloat16().double()).bfloat16()
            exact.append(g[start:stop].double() @ dequantized)
            # BF16 products are exact FP32 operands; bound accumulation error
            # through cancellation using the standard gamma_n absolute bound.
            unit=torch.finfo(torch.float32).eps/2
            gamma=out_width*unit/(1-out_width*unit)
            roundoff.append(gamma*(g[start:stop].double().abs() @ dequantized.bfloat16().double().abs()))
            start=stop
        actual=outputs['bf16'][1]
        up=(torch.nextafter(expected,torch.full_like(expected,float('inf')))-expected).abs()
        down=(torch.nextafter(expected,torch.full_like(expected,float('-inf')))-expected).abs()
        allowance=torch.maximum(up,down).double()+torch.cat(roundoff)
        assert ((actual.double()-expected.double()).abs()<=allowance).all()
        oracle=torch.cat(exact).float()
        relative=float((actual.float()-oracle).norm()/oracle.norm())
        fp8_relative=float((outputs['fp8'][1].float()-oracle).norm()/oracle.norm())
        assert relative<.003 and relative<fp8_relative/4
    finally:DENSE_MOE.clear();DENSE_MOE.update(original)
