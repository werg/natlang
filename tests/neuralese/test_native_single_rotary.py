import copy
import pytest
import torch
from torch.utils.checkpoint import checkpoint
from natlang_neuralese.model import shared_state
from test_shared_state import tiny, graph


@pytest.mark.parametrize('family',['lfm','lfm3','qwen','qwen35'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('recompute',[False,True])
def test_single_rotary_matches_pair_for_real_ports_and_gradients(family,dtype,recompute,monkeypatch):
    port=tiny(family).to(dtype=dtype)
    layer=next(i for i in range(port.num_layers) if port.is_attention(i))
    monkeypatch.setattr(shared_state,'_ROTARY_QUERY_CHUNK',11)
    torch.manual_seed(512)
    positions=torch.tensor([(i*7)%61 for i in range(43)],dtype=torch.long)
    original=torch.randn(1,3,len(positions),8,dtype=dtype)
    def execute(use_new):
        value=original.detach().clone().requires_grad_()
        scale=torch.randn(1,3,1,8,dtype=dtype, generator=torch.Generator().manual_seed(922)).requires_grad_()
        tensor=value*scale
        fn=(lambda t:shared_state._rotate(port,t,positions,layer)) if use_new else (
            lambda t:shared_state._rotate_pair(port,t,t,positions,layer)[0])
        out=checkpoint(fn,tensor,use_reentrant=False) if recompute else fn(tensor)
        grad=torch.randn(out.shape,dtype=dtype,generator=torch.Generator().manual_seed(111))
        named=dict(zip(('value','scale'),torch.autograd.grad(out,(value,scale),grad)))
        return out,named
    actual,actual_grads=execute(True);expected,expected_grads=execute(False)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for name in actual_grads:torch.testing.assert_close(actual_grads[name],expected_grads[name],atol=0,rtol=0,msg=name)


@pytest.mark.parametrize('family',['lfm','lfm3','qwen','qwen35'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_rotary_tables_ignore_dummy_shape_for_real_ports(family,dtype):
    port=tiny(family).to(dtype=dtype);positions=torch.tensor([0,1,4,17,63])
    query=torch.randn(1,2,5,8,dtype=dtype)
    actual=shared_state._rotary_tables(port,query,positions)
    expected=port._rope(query.new_empty(1,len(positions),port.embedding_weight.shape[1]),positions[None])
    def compare(a,b):
        if isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            for x,y in zip(a,b):compare(x,y)
        else:torch.testing.assert_close(a,b,atol=0,rtol=0)
    compare(actual,expected)


@pytest.mark.parametrize('family',['lfm','lfm3','qwen','qwen35'])
def test_rotary_table_dummy_is_one_element_for_supported_ports(family,monkeypatch):
    port=tiny(family);query=torch.randn(1,2,17,8);positions=torch.arange(17)
    original=torch.Tensor.new_empty;shapes=[]
    def capture(tensor,*shape,**kwargs):
        if tensor is query:shapes.append(shape)
        return original(tensor,*shape,**kwargs)
    monkeypatch.setattr(torch.Tensor,'new_empty',capture)
    shared_state._rotary_tables(port,query,positions)
    assert shapes==[(1,)]


def test_custom_rotary_override_keeps_original_pair_dispatch(monkeypatch):
    port=tiny('qwen'); original=port._apply_rope; calls=[]
    def custom(layer,q,k,cos,sin):
        calls.append((q.shape,k.shape))
        return original(layer,q,k,cos,sin)
    monkeypatch.setattr(port,'_apply_rope',custom)
    value=torch.randn(1,2,5,8);positions=torch.arange(5)
    actual=shared_state._rotate(port,value,positions,0)
    assert calls==[(value.shape,value.shape)]
    torch.testing.assert_close(actual,original(0,value,value,*port._rope(
        value.new_empty(1,5,port.embedding_weight.shape[1]),positions[None]))[0],atol=0,rtol=0)


@pytest.mark.parametrize('family',['lfm','lfm3','qwen','qwen35'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_full_graph_different_query_key_positions_keep_named_gradients(family,dtype,monkeypatch):
    port=tiny(family).to(dtype=dtype);model=shared_state.SharedStateModel(port,tuple(range(60,64)),rank=4).to(dtype=dtype)
    reference=copy.deepcopy(model); plan=graph()
    monkeypatch.setattr(shared_state,'_ROTARY_QUERY_CHUNK',3)
    actual=model(plan)['h_final']
    monkeypatch.setattr(shared_state,'_rotate',lambda p,t,pos,layer:shared_state._rotate_pair(p,t,t,pos,layer)[0])
    expected=reference(plan)['h_final']
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    weight=torch.randn_like(actual)
    (actual*weight).sum().backward();(expected*weight).sum().backward()
    for (name,left),(other,right) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other and (left.grad is None)==(right.grad is None),name
        if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,atol=0,rtol=0,msg=name)
