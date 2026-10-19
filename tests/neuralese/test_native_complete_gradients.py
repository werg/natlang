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
    flat_input.grad=torch.full_like(flat_input,.125)
    cached_input.grad=torch.full_like(cached_input,-.25)
    input_grads=[(flat_input,flat_input.grad,flat_input.grad.clone(),flat_input.grad._version),
        (cached_input,cached_input.grad,cached_input.grad.clone(),cached_input.grad._version)]
    flat=model(plan,embeddings=flat_input);cached=model(plan,embeddings=cached_input,reference=True)
    a=F.cross_entropy(model.logits(flat['h_final'][:,-1]).float(),torch.tensor([7],device=device))
    b=F.cross_entropy(model.logits(cached['h_final'][:,-1]).float(),torch.tensor([7],device=device))
    if wrong_derivative:b=b*.5+b.detach()*.5
    rng=torch.get_rng_state()
    report=compare_complete_gradients(a,b,named,flat_input,cached_input,atol=1e-5,rtol=1e-4)
    assert report['passed']is(not wrong_derivative)
    assert set(report['comparisons'])=={'input_embeddings',*(name for name,_ in named)}
    assert report['trainable_parameter_count']==len(named)
    assert report['derivative_comparison_count']==len(named)+1
    assert report['retained_flat_gradient_bytes']>report['largest_live_comparison_pair_bytes']
    assert report['retained_flat_gradient_device']=='cpu'and report['retained_flat_gradient_cuda_bytes']==0
    assert report['retained_cached_gradient_map_bytes']==0
    assert report['retained_cached_gradient_device']=='none'
    for role in('backbone','adapter','learned_input','learned_output'):
        assert roles[role]
        assert any(report['comparisons'][name].get('reference_l2',0)>0 for name in roles[role])
    assert report['comparisons']['input_embeddings']['reference_l2']>0
    assert torch.equal(rng,torch.get_rng_state())
    for p,previous,value,version in before:
        assert p.grad is previous and torch.equal(p.grad,value)and p._version==version
    for tensor,previous,value,version in input_grads:
        assert tensor.grad is previous and torch.equal(tensor.grad,value)
        assert tensor.grad._version==version
    assert optimizer.state_dict()==state


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA reference lifetime')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_reference_gradients_are_cpu_resident_and_cached_gradients_stream_to_observer(monkeypatch,dtype):
    import natlang_neuralese.eval.native_gradients as native_gradients
    parameter=torch.nn.Parameter(torch.linspace(.1,.5,128,device='cuda',dtype=dtype))
    flat_input=torch.ones(2,128,device='cuda',dtype=dtype,requires_grad=True)
    flat=(flat_input*parameter).square().mean()
    original=native_gradients._backward_gradients_to_cpu
    flat_refs=[];streamed_calls=[];observed=[]
    def capture(loss,names,parameters,input_tensor=None,*,on_gradient=None):
        result=original(loss,names,parameters,input_tensor,on_gradient=on_gradient)
        if not streamed_calls:
            assert on_gradient is None
            assert result and all(value.device.type=='cpu' for value in result.values() if value is not None)
            flat_refs.extend(weakref.ref(value)for value in result.values()if value is not None)
        else:
            assert callable(on_gradient)
            assert result=={}
        streamed_calls.append(on_gradient)
        return result
    monkeypatch.setattr(native_gradients,'_backward_gradients_to_cpu',capture)
    def cached():
        assert len(flat_refs)==2 and all(ref() is not None and ref().device.type=='cpu' for ref in flat_refs)
        x=torch.ones_like(flat_input,requires_grad=True)
        return (x*parameter).square().mean(),x
    def observer(name,left,right):
        assert left.device.type==right.device.type=='cpu'
        assert left.dtype==right.dtype==dtype
        assert torch.equal(left,right)
        observed.append(name)
    rng=torch.cuda.get_rng_state().clone()
    report=compare_complete_gradients(flat,None,[('weight',parameter)],flat_input,flat_input,
        atol=1e-5,rtol=1e-4,cached_factory=cached,gradient_observer=observer)
    assert report['passed']and set(observed)=={'input_embeddings','weight'}
    assert len(streamed_calls)==2 and callable(streamed_calls[1])
    assert report['retained_cached_gradient_map_bytes']==0
    assert parameter.grad is None and flat_input.grad is None
    assert torch.equal(torch.cuda.get_rng_state(),rng)


def test_inventory_cannot_omit_a_trainable_or_alias_a_parameter():
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    _,named=make_native_optimizer(model,lr=3e-6,in_backward=False)
    with pytest.raises(ValueError,match='omitted'):
        production_inventory(model,named[:-1])
    x=torch.ones(2,requires_grad=True);a=x.square().sum();b=x.square().sum()
    with pytest.raises(ValueError,match='unique'):
        compare_complete_gradients(a,b,[('a',x),('b',x)],x,x,atol=1e-5,rtol=1e-4)


def test_mellum_production_topology_counts_parameters_separately_from_input_derivative():
    from natlang_neuralese.maple.model import MapleConfig,MapleForCausalLM
    from natlang_neuralese.maple.maple_port import MaplePortBackbone
    from natlang_neuralese.train.backbone_policy import configure_backbone_training
    config=MapleConfig(vocab_size=64,hidden_size=16,num_hidden_layers=28,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4,
        moe_intermediate_size=8,num_experts=2,num_experts_per_tok=1,
        layer_types=['sliding_attention']*28,model_type='mellum',dense_experts=True)
    with torch.device('meta'):
        hf=MapleForCausalLM(config);hf.natlang_precision='bf16'
        model=SharedStateModel(MaplePortBackbone(hf,markers=False),(58,59,60,61),
            rank=4,token_override_ids=(23,24))
        backbone=[('backbone.'+name,p)for name,p in configure_backbone_training(model.port,'latent')]
        interface=[('interface.'+name,p)for name,p in model.named_parameters()
            if name.startswith(('input_rows','output_rows','extra_input_rows','extra_output_rows','adapter.'))
            and p.requires_grad]
        named,roles=production_inventory(model,backbone+interface)
    assert len(backbone)==28*11==308 and len(interface)==6
    assert len(named)==314 and len({'input_embeddings',*(name for name,_ in named)})==315
    assert all(roles.values())and all(p.device.type=='meta'for _,p in named)


def test_complete_gradient_observer_start_failure_closes_flat_cotangent_scope():
    import os
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope,active_cotangent_scope
    class FailingObserver:
        def begin_vjp_observation(self):raise RuntimeError('injected observer setup failure')
    model=torch.nn.Sequential(torch.nn.Linear(4,3,bias=True)).to(dtype=torch.bfloat16)
    original=model[0]
    named=list(model.named_parameters())
    inputs=torch.randn(2,4,dtype=torch.bfloat16,requires_grad=True)
    cotangents=ParameterCotangentScope.for_model(model,mode='vjp')
    cotangents.__enter__()
    loss=model(inputs).float().square().mean()
    scratch=cotangents.directory
    assert model[0] is not original
    with pytest.raises(RuntimeError,match='injected observer setup failure'):
        compare_complete_gradients(loss,None,named,inputs,None,atol=1e-5,rtol=1e-4,
            gradient_observer=FailingObserver(),flat_cotangent_scope=cotangents)
    assert active_cotangent_scope() is None
    assert model[0] is original
    assert not os.path.exists(scratch)


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('device,backend',[('cpu','cpu_file'),('cuda','cpu_file'),('cuda','device_fp32')])
def test_actual_runtime_qualifier_labels_complete_inventory_and_legacy_channel(family,device,backend):
    if device=='cuda'and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from types import SimpleNamespace
    from natlang_neuralese.eval.native_state import qualify_existing
    from natlang_neuralese.eval.native_gradients import validate_complete_proof
    class Tokenizer:
        all_special_ids=[]
        def __call__(self,*args,**kwargs):return SimpleNamespace(input_ids=[2,3,4,5,6,7,8,9])
        def decode(self,*args,**kwargs):return 'lexical source'
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4).to(device)
    model.native_cotangent_accumulator_backend=backend
    model.port.checkpoint_layers=False
    model.port.hf.to(dtype=torch.bfloat16)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    for parameter in model.port.parameters():parameter.requires_grad_(True)
    with torch.no_grad():model.adapter.up.weight.normal_(std=.03)
    optimizer,named=make_native_optimizer(model,lr=3e-6,in_backward=True)
    assert any(name.startswith('backbone.') and parameter.dtype is torch.bfloat16
               for name,parameter in named)
    roles=SimpleNamespace(produce_open=58,produce_close=59,consume_open=60,consume_close=61)
    config={'architecture':model.port.config.model_type,'tokenizer_fingerprint':'b'*64}
    args=dict(checkpoint_sha256='a'*64,forward_atol=1e-4,gradient_atol=1e-5)
    legacy=qualify_existing(model,Tokenizer(),roles,config,**args)
    assert legacy['passed']and legacy['gradient_scope']=='adapter-channel-only'
    assert legacy['complete_gradient_qualification']is None
    report=qualify_existing(model,Tokenizer(),roles,config,gradient_parameters=named,**args)
    assert report['passed']and report['gradient_scope']=='complete-production-trainables'
    proof=report['complete_gradient_qualification']
    assert proof['cotangent_accumulator_backends']=={'flat':backend,'cached':'cpu_file'}
    assert model.native_cotangent_accumulator_backend==backend
    validate_complete_proof(proof,report['native_identity'],gradient_atol=1e-5)
    assert proof['derivative_comparison_count']==proof['trainable_parameter_count']+1
    for key in ('trainable_parameter_count','derivative_comparison_count'):
        forged=copy.deepcopy(proof);forged[key]+=1
        with pytest.raises(ValueError,match='count differs'):
            validate_complete_proof(forged,report['native_identity'],gradient_atol=1e-5)
    historical=copy.deepcopy(proof)
    historical.pop('trainable_parameter_count');historical.pop('derivative_comparison_count')
    validate_complete_proof(historical,report['native_identity'],gradient_atol=1e-5)
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
