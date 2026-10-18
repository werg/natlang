import copy
import weakref
import pytest
import torch
import torch.nn.functional as F
from test_shared_state import tiny
from natlang_neuralese.eval.native_gradients import compare_complete_gradients,production_inventory
from natlang_neuralese.model.shared_state import SharedStateModel,Invocation,ImportSpan,compile_graph
from natlang_neuralese.train.native_state_sft import make_native_optimizer


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('wrong_derivative',[False,True])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all_production_gradients_connected_rows_and_inputs_preserved(family,wrong_derivative,device):
    if device=='cuda'and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4).to(device)
    model.port.checkpoint_layers=False
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    with torch.no_grad():model.adapter.up.weight.normal_(std=.03)
    optimizer,named=make_native_optimizer(model,lr=3e-6,latent_ns='cubic5',in_backward=True)
    named,roles=production_inventory(model,named)
    for name,p in named:p.grad=torch.ones_like(p)
    before=[(p,p.grad,p.grad.clone(),p._version)for _,p in named]
    def forbidden(*args):raise AssertionError('qualification applied optimizer update')
    optimizer.latent._muon_update=forbidden
    state=copy.deepcopy(optimizer.state_dict())
    plan=compile_graph([Invocation('producer',(2,58,3,4,59),producing=((2,4),)),
        Invocation('consumer',(5,60,61,6,7),imports=(ImportSpan(2,'producer',2,4),))])
    ids=plan.tensor([t for node in plan.invocations for t in node.token_ids],model.input_rows.device)
    flat_input=model.embed(ids[None]);cached_input=model.embed(ids[None])
    flat=model(plan,embeddings=flat_input);cached=model(plan,embeddings=cached_input,reference=True)
    a=F.cross_entropy(model.logits(flat['h_final'][:,-1]).float(),torch.tensor([7],device=device))
    b=F.cross_entropy(model.logits(cached['h_final'][:,-1]).float(),torch.tensor([7],device=device))
    if wrong_derivative:b=b*.5+b.detach()*.5
    rng=torch.get_rng_state()
    report=compare_complete_gradients(a,b,named,flat_input,cached_input,atol=1e-5,rtol=1e-4)
    assert report['passed']is(not wrong_derivative)
    assert set(report['comparisons'])=={'input_embeddings',*(name for name,_ in named)}
    assert report['retained_flat_gradient_bytes']>report['largest_live_comparison_pair_bytes']
    assert report['retained_flat_gradient_device']=='cpu'and report['retained_flat_gradient_cuda_bytes']==0
    for role in('backbone','adapter','learned_input','learned_output'):
        assert roles[role]
        assert any(report['comparisons'][name].get('reference_l2',0)>0 for name in roles[role])
    assert report['comparisons']['input_embeddings']['reference_l2']>0
    assert torch.equal(rng,torch.get_rng_state())
    for p,previous,value,version in before:
        assert p.grad is previous and torch.equal(p.grad,value)and p._version==version
    assert optimizer.state_dict()==state


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA reference lifetime')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_reference_gpu_tensors_are_released_before_cached_forward_and_observer_is_exact_cpu(monkeypatch,dtype):
    parameter=torch.nn.Parameter(torch.linspace(.1,.5,128,device='cuda',dtype=dtype))
    flat_input=torch.ones(2,128,device='cuda',dtype=dtype,requires_grad=True)
    flat=(flat_input*parameter).square().mean()
    original=torch.autograd.grad;references=[];observed=[]
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        if not references:references.extend(weakref.ref(value)for value in result if value is not None)
        return result
    monkeypatch.setattr(torch.autograd,'grad',capture)
    def cached():
        assert len(references)==2 and all(ref()is None for ref in references)
        x=torch.ones_like(flat_input,requires_grad=True)
        return (x*parameter).square().mean(),x
    def observer(name,left,right):
        assert left.device.type==right.device.type=='cpu'
        assert left.dtype==right.dtype==dtype
        assert torch.equal(left,right)
        observed.append(name)
    rng=torch.cuda.get_rng_state().clone()
    report=compare_complete_gradients(flat,None,[('weight',parameter)],flat_input,None,
        atol=1e-5,rtol=1e-4,cached_factory=cached,gradient_observer=observer)
    assert report['passed']and set(observed)=={'input_embeddings','weight'}
    assert parameter.grad is None and torch.equal(torch.cuda.get_rng_state(),rng)


def test_inventory_cannot_omit_a_trainable_or_alias_a_parameter():
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    _,named=make_native_optimizer(model,lr=3e-6,in_backward=False)
    with pytest.raises(ValueError,match='omitted'):
        production_inventory(model,named[:-1])
    x=torch.ones(2,requires_grad=True);a=x.square().sum();b=x.square().sum()
    with pytest.raises(ValueError,match='unique'):
        compare_complete_gradients(a,b,[('a',x),('b',x)],x,x,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_runtime_qualifier_labels_complete_inventory_and_legacy_channel(family):
    from types import SimpleNamespace
    from natlang_neuralese.eval.native_state import qualify_existing
    from natlang_neuralese.eval.native_gradients import validate_complete_proof
    class Tokenizer:
        all_special_ids=[]
        def __call__(self,*args,**kwargs):return SimpleNamespace(input_ids=[2,3,4,5,6,7,8,9])
        def decode(self,*args,**kwargs):return 'lexical source'
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    model.port.checkpoint_layers=False
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    with torch.no_grad():model.adapter.up.weight.normal_(std=.03)
    optimizer,named=make_native_optimizer(model,lr=3e-6,in_backward=True)
    roles=SimpleNamespace(produce_open=58,produce_close=59,consume_open=60,consume_close=61)
    config={'architecture':model.port.config.model_type,'tokenizer_fingerprint':'b'*64}
    args=dict(checkpoint_sha256='a'*64,forward_atol=1e-4,gradient_atol=1e-5)
    legacy=qualify_existing(model,Tokenizer(),roles,config,**args)
    assert legacy['passed']and legacy['gradient_scope']=='adapter-channel-only'
    assert legacy['complete_gradient_qualification']is None
    report=qualify_existing(model,Tokenizer(),roles,config,gradient_parameters=named,**args)
    assert report['passed']and report['gradient_scope']=='complete-production-trainables'
    proof=report['complete_gradient_qualification']
    validate_complete_proof(proof,report['native_identity'],gradient_atol=1e-5)
    assert all(roles for roles in proof['roles'].values())


def test_cached_forward_is_lazy_after_flat_vjp_for_stateful_backend():
    state={'epoch':0};events=[]
    class Stateful(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):
            ctx.epoch=state['epoch'];events.append(('forward',ctx.epoch))
            return x.square()
        @staticmethod
        def backward(ctx,g):
            assert state['epoch']==ctx.epoch,'another forward mutated the saved backend history'
            events.append(('backward',ctx.epoch))
            return g*2
    parameter=torch.ones(2,requires_grad=True)
    flat_input=torch.ones(2,requires_grad=True)
    flat=(Stateful.apply(flat_input)*parameter).sum()
    def cached():
        assert events==[('forward',0),('backward',0)]
        state['epoch']=1
        x=torch.ones(2,requires_grad=True)
        return (Stateful.apply(x)*parameter).sum(),x
    report=compare_complete_gradients(flat,None,[('weight',parameter)],flat_input,None,
        atol=1e-5,rtol=1e-4,cached_factory=cached)
    assert report['passed']
    assert events==[('forward',0),('backward',0),('forward',1),('backward',1)]
