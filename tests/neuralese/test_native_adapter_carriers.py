"""Exact forward storage and FP32 coefficient adjoints through real native ports."""
from dataclasses import replace
import os

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_native_adapted_rl import capture_actor
from natlang_neuralese.model.native_adapters import create_adapter, target_parameters, resolve_adapters, effective_identity, run_native
from natlang_neuralese.rl.native_replay import compile_replay, action_logps


def exercise_carrier(tmp_path,monkeypatch,family,target,*,recipe=None,lfm_bf16=False,duplicate=False):
    torch.manual_seed(1701)
    engine=engine_fixture(tmp_path,family,service_template=True)
    if lfm_bf16:
        from natlang_neuralese.serve import load_native_engine
        path,device=engine.native_checkpoint,engine.device
        engine.store.close()
        engine=load_native_engine(path,device=device,dtype=torch.bfloat16)
    backend=None
    if recipe is not None:
        from natlang_neuralese.train.fp8_backend import make_backend
        backend=make_backend('triton-experts',engine.shared_model.port.hf,device=engine.device,recipe=recipe)
    available=target_parameters(engine.shared_model)
    name=next(name for name in available if name.endswith(target))
    adapter=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    artifact=replace(adapter.artifact,tensors={name:torch.full_like(adapter.artifact.tensors[name],.04)})
    adapter=engine.store.put_native_tensor_artifact(artifact)
    entries=[{'id':adapter.id,'scale':.75}]
    if duplicate:entries.extend([{'id':adapter.id,'scale':-.2},{'id':adapter.id,'scale':.1}])
    identity=effective_identity(engine,resolve_adapters(engine,entries))
    from contextlib import nullcontext
    with backend.autocast() if backend else nullcontext():
        _,turns=capture_actor(engine,entries,monkeypatch)
    engine.shared_model.port.checkpoint_layers=True
    replay=compile_replay(turns,identity=identity,roles=engine.role_tokens,
        fetch_block=lambda ident:engine.store.get_native_state(ident).block)
    coefficient=adapter.artifact.tensors[name].clone().requires_grad_(True)
    originals=dict(engine.shared_model.named_modules())
    original_params=dict(engine.shared_model.named_parameters())
    reader=[i for i,item in enumerate(replay.actions) if item['turn']=='reader']
    def execute(reference):
        result=engine.shared_model(replay.plan,reference=reference,retain_states=False)
        logits=engine.shared_model.logits(result['h_final'])
        logps,_=action_logps(engine.shared_model,result['h_final'][0],replay.actions,chunk_tokens=2)
        gradient=torch.autograd.grad(-logps[reader].mean(),coefficient)[0]
        return result['h_final'].detach(),logits.detach(),logps.detach(),gradient.detach()
    def run(reference,precision):
        before=backend.row_counts['calls'] if backend else 0
        with backend.autocast() if backend else nullcontext():
            result=run_native(engine,lambda:execute(reference),entries,leaves={adapter.id:{name:coefficient}},
                gradient_precision=precision)
        if backend:assert backend.row_counts['calls']>before
        return result
    baseline=run(False,'storage')
    actual=run(False,'fp32')
    expected=run(True,'fp32')
    for observed,old in zip(actual[:3],baseline[:3]):assert torch.equal(observed,old)
    torch.testing.assert_close(actual[2],expected[2],rtol=2e-5,atol=3e-6)
    torch.testing.assert_close(actual[3],expected[3],rtol=2e-5,atol=3e-6)
    relative=(actual[3]-expected[3]).norm()/expected[3].norm()
    assert relative<2e-5 and actual[3].norm()>0
    assert all(dict(engine.shared_model.named_modules())[name] is value for name,value in originals.items())
    assert all(dict(engine.shared_model.named_parameters())[name] is value for name,value in original_params.items())
    engine.store.close()


@pytest.mark.parametrize('family,target',[('lfm','q_proj.weight'),('lfm','feed_forward.w1.weight'),
    ('mellum','q_proj.weight'),('mellum','v_proj.weight'),('mellum','experts.gate_up'),('mellum','experts.down')])
def test_real_native_carriers_preserve_forward_and_cached_coefficient_adjoint(tmp_path,monkeypatch,family,target):
    return exercise_carrier(tmp_path,monkeypatch,family,target)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual CUDA FP8 kernels')
@pytest.mark.parametrize('recipe',['current-bf16-dx','current'])
@pytest.mark.parametrize('target',['q_proj.weight','experts.gate_up','experts.down'])
def test_executed_fp8_carriers_preserve_forward_and_cached_adjoint(tmp_path,monkeypatch,recipe,target):
    monkeypatch.setenv('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cuda')
    return exercise_carrier(tmp_path,monkeypatch,'mellum',target,recipe=recipe)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual grouped CUDA kernel')
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_grouped_carrier_weight_adjoint_against_exact_fp64_with_empty_odd_groups(dtype):
    from natlang_neuralese.maple.fp8_experts import grouped_weight_gradient
    torch.manual_seed(391)
    a=torch.randn(72,29,device='cuda',dtype=dtype)
    grad=torch.randn(72,37,device='cuda',dtype=dtype)
    ends=torch.tensor([0,35,72],device='cuda',dtype=torch.int32)
    actual=grouped_weight_gradient(grad,a,ends,(3,37,29))
    ac,gc=a.cpu().double(),grad.cpu().double()
    expected=torch.stack([gc[:0].T@ac[:0],gc[:35].T@ac[:35],gc[35:].T@ac[35:]])
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual.cpu().double(),expected,rtol=2e-5,atol=3e-6)
    assert (actual.cpu().double()-expected).norm()/expected.norm()<2e-6


def test_native_carrier_restores_aliases_hooks_and_weights_on_failure(tmp_path):
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    torch.manual_seed(1701)
    engine=engine_fixture(tmp_path,'mellum',service_template=True)
    name=next(name for name in target_parameters(engine.shared_model) if name.endswith('q_proj.weight'))
    adapter=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    modules=dict(engine.shared_model.named_modules())
    parameters=dict(engine.shared_model.named_parameters())
    module=engine.shared_model.get_submodule(name.removesuffix('.weight'))
    original_hook=lambda *args:None
    handle=module.register_forward_pre_hook(original_hook)
    def fail():
        current=engine.shared_model.get_submodule(name.removesuffix('.weight'))
        assert current is not module and current._forward_pre_hooks==module._forward_pre_hooks
        assert current.weight.dtype==torch.float32
        raise RuntimeError('carrier fixture rejected')
    with pytest.raises(RuntimeError,match='carrier fixture rejected'):
        run_native(engine,fail,[adapter.id],gradient_precision='fp32')
    assert all(dict(engine.shared_model.named_modules())[name] is value for name,value in modules.items())
    assert all(dict(engine.shared_model.named_parameters())[name] is value for name,value in parameters.items())
    handle.remove()
    row={'schema':'natlang.neuralese-native-graph/1','id':'carrier-service','split':'train',
        'source_groups':['carrier-service-fixture'],'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
        'nodes':[{'name':'answer','token_ids':[4,5,6,7],'target_weights':[0,1,1,1]}]}
    body={'schema':'natlang.native-gradient-request/1','graph':row,'adapters':[adapter.id],
        'adapter_gradient_precision':'fp32','target':{'kind':'adapter','id':adapter.id}}
    from test_native_adapted_rl import post_service
    response=post_service(engine,'/v1/neuralese/grad',body)
    artifact=engine.store.get_native_tensor_artifact(response['gradient']['id']).artifact
    assert artifact.provenance['adapter_gradient_precision']=='fp32'
    assert artifact.tensors[name].dtype==torch.float32 and artifact.tensors[name].norm()>0
    assert all(dict(engine.shared_model.named_modules())[name] is value for name,value in modules.items())
    engine.store.close()


def test_adapter_vjp_default_names_matches_explicit_all_names(tmp_path):
    """Default adapter selection must install the same widened cotangent routes."""
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    torch.manual_seed(1701)
    engine=engine_fixture(tmp_path,'mellum',service_template=True)
    name=next(name for name in target_parameters(engine.shared_model) if name.endswith('q_proj.weight'))
    initial=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    adapter=engine.store.put_native_tensor_artifact(replace(initial.artifact,
        tensors={name:torch.full_like(initial.artifact.tensors[name],.04)}))
    graph={'schema':'natlang.neuralese-native-graph/1','id':'adapter-vjp-default-targets',
        'split':'train','source_groups':['adapter-vjp-default-targets-fixture'],
        'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
        'nodes':[{'name':'answer','token_ids':[4,5,6,7],'target_weights':[0,1,1,1]}]}
    common={'schema':'natlang.native-gradient-request/1','graph':graph,
        'adapters':[adapter.id],'adapter_gradient_precision':'fp32'}
    def gradient(target):
        result=native_graph_gradient(engine,{**common,'target':target},None)
        artifact=engine.store.get_native_tensor_artifact(result['gradient']['id']).artifact
        assert artifact.provenance['target_names']==[name]
        return artifact.tensors[name]
    default=gradient({'kind':'adapter','id':adapter.id})
    explicit=gradient({'kind':'adapter','id':adapter.id,'names':[name]})
    assert torch.equal(default,explicit)
    assert torch.isfinite(default).all() and default.norm()>0
    engine.store.close()


def test_native_attention_compiler_contract_separates_actual_adjoint_and_inference_modes(monkeypatch):
    from natlang_neuralese.model import shared_state
    from natlang_neuralese.common import native_attention_delta
    monkeypatch.setattr(native_attention_delta,'install_native_delta_lowering',lambda:None)
    monkeypatch.setattr(torch,'compile',lambda fn,**kwargs:fn)
    monkeypatch.setattr(shared_state,'_flex',{})
    q=torch.randn(1,2,5,4);k=q.clone();v=q.clone()
    base=shared_state._compiled_flex(q,k,v,.5)
    assert shared_state._compiled_flex(q,k,v,.5) is base
    q_grad=shared_state._compiled_flex(q.requires_grad_(),k,v,.5)
    kv_grad=shared_state._compiled_flex(q.detach(),k.requires_grad_(),v.requires_grad_(),.5)
    with torch.no_grad():no_grad=shared_state._compiled_flex(q,k,v,.5)
    with torch.inference_mode():
        inf=torch.randn(1,2,5,4)
        inference=shared_state._compiled_flex(inf,inf.clone(),inf.clone(),.5)
    assert len({fn.__code__ for fn in (base,q_grad,kv_grad,no_grad,inference)})==5


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual grouped CUDA kernel')
@pytest.mark.parametrize('glue',['0','1'])
def test_grouped_carrier_bank_uses_actual_custom_projection_and_preserves_forward(tmp_path,monkeypatch,glue):
    from natlang_neuralese.maple.model import DENSE_MOE
    from natlang_neuralese.maple.fp8_experts import GroupedLinear
    from natlang_neuralese.model.shared_state import Invocation,compile_graph
    monkeypatch.setenv('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cuda')
    monkeypatch.setenv('NATLANG_MOE_GLUE',glue)
    monkeypatch.setitem(DENSE_MOE,'kernel','grouped')
    monkeypatch.setitem(DENSE_MOE,'fp8',None)
    engine=engine_fixture(tmp_path,'mellum',service_template=True)
    name=next(name for name in target_parameters(engine.shared_model) if name.endswith('experts.gate_up'))
    initial=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    adapter=engine.store.put_native_tensor_artifact(replace(initial.artifact,
        tensors={name:torch.full_like(initial.artifact.tensors[name],.04)}))
    coefficient=adapter.artifact.tensors[name].clone().requires_grad_(True)
    count=[0]
    original=GroupedLinear.forward
    def measured(*args):count[0]+=1;return original(*args)
    monkeypatch.setattr(GroupedLinear,'forward',staticmethod(measured))
    plan=compile_graph([Invocation('answer',(4,5,6,7))])
    def execute():
        result=engine.shared_model(plan,retain_states=False)
        logits=engine.shared_model.logits(result['h_final'])
        gradient=torch.autograd.grad(logits.square().mean(),coefficient)[0]
        return logits.detach(),gradient.detach()
    baseline=run_native(engine,execute,[adapter.id],leaves={adapter.id:{name:coefficient}})
    old=count[0]
    actual=run_native(engine,execute,[adapter.id],leaves={adapter.id:{name:coefficient}},gradient_precision='fp32')
    assert count[0]>old
    assert torch.equal(baseline[0],actual[0])
    assert actual[1].norm()>0 and torch.isfinite(actual[1]).all()
    engine.store.close()


@pytest.mark.parametrize('target',['feed_forward.w1.weight','feed_forward.w2.weight'])
def test_bf16_lfm_hf_ffn_carriers_preserve_stock_forward_and_cached_adjoint(tmp_path,monkeypatch,target):
    return exercise_carrier(tmp_path,monkeypatch,'lfm',target,lfm_bf16=True)


def test_ordered_repeated_coefficient_carriers_keep_each_forward_rounding_and_sum_adjoints(tmp_path,monkeypatch):
    return exercise_carrier(tmp_path,monkeypatch,'mellum','v_proj.weight',duplicate=True)
