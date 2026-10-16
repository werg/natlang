"""Declared reference arithmetic, exact visibility, native double backward."""
import os

import pytest
import torch

from natlang_neuralese.common.native_higher_order import (
    active_reference_arithmetic,reference_arithmetic,reference_attention,reference_linear,
    capture_reference_execution)
from natlang_neuralese.data.native_graph import SCHEMA,parse_graph,graph_loss
from natlang_neuralese.model.shared_state import SharedStateModel,Invocation,ImportSpan,compile_graph
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from test_shared_state import tiny


def plan():
    return compile_graph((Invocation('writer',(1,2,3,4),producing=((1,3),)),
        Invocation('reader',(5,6,7),imports=(ImportSpan(source='writer',start=1,stop=3,at=1),))))


@pytest.mark.parametrize('window',[None,3])
def test_reference_attention_exact_mask_gqa_and_second_derivatives(window):
    torch.manual_seed(700)
    layout=plan();n=layout.token_count;slots=len(layout.slot_sources)
    values=[torch.randn(1,heads,count,2,dtype=torch.float64,requires_grad=True)
            for heads,count in ((2,n),(1,slots),(1,slots))]
    with reference_arithmetic(query_chunk_tokens=2) as mode:
        output=reference_attention(*values,layout,window,.7)
        q,k,v=values
        logits=(q @ k.repeat_interleave(2,1).transpose(-2,-1))*.7
        expected=logits.masked_fill(~layout.attention_mask(q.device,window),float('-inf')).softmax(-1) @ v.repeat_interleave(2,1)
        torch.testing.assert_close(output,expected,rtol=1e-13,atol=1e-13)
        assert torch.autograd.gradgradcheck(lambda *args:reference_attention(*args,layout,window,.7),
            values,atol=1e-5,rtol=1e-4)
        assert mode.counters['attention']>0
    assert active_reference_arithmetic() is None


def test_reference_linear_uses_live_named_values_with_real_double_backward():
    torch.manual_seed(701)
    x=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    w=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    with reference_arithmetic():
        torch.testing.assert_close(reference_linear(x,w),x @ w.T,rtol=0,atol=0)
        assert torch.autograd.gradgradcheck(reference_linear,(x,w),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('checkpointed',[False,True])
def test_both_native_ports_actual_all_parameter_second_backward_and_scope_restore(family,checkpointed):
    torch.manual_seed(702)
    device=os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu')
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4).to(device)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    make_native_optimizer(model,lr=1e-3,in_backward=False)
    with torch.no_grad():model.adapter.up.weight.normal_(std=.01)
    example=parse_graph({'schema':SCHEMA,'id':'native-order2','split':'train','source_groups':['order2-only-train'],
        'tokenizer_fingerprint':'fixture','nodes':[
            {'name':'writer','token_ids':[1,58,5,6,59],'producing':[[2,4]],'target_weights':[0,1,1,1,1]},
            {'name':'consumer','token_ids':[2,60,61,7,8],
             'imports':[{'source':'writer','start':2,'stop':4,'at':2}], 'target_weights':[0,1,0,1,1]}]},
        tokenizer_fingerprint='fixture')
    objects={name:(id(p),p.data_ptr(),p.requires_grad,p.detach().clone()) for name,p in model.named_parameters()}
    rng=torch.get_rng_state().clone()
    old_checkpoint=getattr(model.port,'checkpoint_layers',None)
    with torch.no_grad():
        baseline=model(example.plan,retain_states=False)['h_final'].clone()
    with reference_arithmetic(model,query_chunk_tokens=2) as mode:
        loss,_=graph_loss(model,example,checkpoint_layers=checkpointed)
        parameters=[p for p in model.parameters() if p.requires_grad]
        first=torch.autograd.grad(loss,parameters,create_graph=True,allow_unused=False)
        second=torch.autograd.grad(sum(g.square().sum() for g in first),parameters,allow_unused=False)
        assert all(torch.isfinite(g).all() for g in second)
        assert sum(float(g.float().square().sum()) for g in second)>1e-8
        assert mode.counters['attention']==(2 if family=='lfm' else 4)*(3 if checkpointed else 1)
        replay=capture_reference_execution(lambda:reference_linear(torch.ones(1,2),torch.ones(1,2)))
    with pytest.raises(RuntimeError,match='scope expired'):replay()
    assert active_reference_arithmetic() is None
    assert getattr(model.port,'checkpoint_layers',None)==old_checkpoint
    assert torch.equal(torch.get_rng_state(),rng)
    for name,p in model.named_parameters():
        ident,storage,requires,value=objects[name]
        assert id(p)==ident and p.data_ptr()==storage and p.requires_grad==requires
        torch.testing.assert_close(p,value,rtol=0,atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model(example.plan,retain_states=False)['h_final'],baseline,rtol=0,atol=0)


@pytest.mark.parametrize('algorithm',['sgd','adam'])
def test_repeated_functional_service_optimizer_matches_actual_named_state_and_hypergradient(algorithm):
    from natlang_neuralese.train.optim import functional_named_service_step,named_service_optimizer
    original=torch.tensor([.7,-.4,1.2],dtype=torch.float64,requires_grad=True)
    actual=torch.nn.Parameter(original.detach().clone())
    options={'name':algorithm,'learning_rate':.03,'weight_decay':.07,'momentum':.4,
             'beta1':.8,'beta2':.9,'eps':1e-7}
    optimizer=named_service_optimizer([('p',actual)],**options)
    value=original;state={}
    for index in range(3):
        target=torch.tensor([.2,.1,-.3],dtype=original.dtype)
        loss=(value-target).square().sum()/2
        gradient=torch.autograd.grad(loss,value,create_graph=True)[0]
        before=value.clone();old_state={key:v.clone() for key,v in state.items()}
        result,next_state=functional_named_service_step({'p':value},{'p':gradient},state,**options)
        torch.testing.assert_close(value,before,rtol=0,atol=0)
        for key,v in state.items():torch.testing.assert_close(v,old_state[key],rtol=0,atol=0)
        value=result['p'];state=next_state
        optimizer.zero_grad();((actual-target).square().sum()/2).backward();optimizer.step()
        torch.testing.assert_close(value,actual,rtol=0,atol=0)
        if algorithm=='adam':
            for prefix,name in (('m.','exp_avg'),('v.','exp_avg_sq'),('step.','step')):
                torch.testing.assert_close(state[prefix+'p'].reshape(-1),optimizer.state[actual][name].reshape(-1),rtol=0,atol=0)
        else:torch.testing.assert_close(state['momentum.p'],optimizer.state[actual]['momentum_buffer'],rtol=0,atol=0)
    hypergradient=torch.autograd.grad(value.square().sum(),original)[0]
    assert torch.isfinite(hypergradient).all() and hypergradient.norm()>0
    def run(x):
        values=x;moments={}
        for _ in range(3):
            g=torch.autograd.grad((values-target).square().sum()/2,values,create_graph=True)[0]
            updated,moments=functional_named_service_step({'p':values},{'p':g},moments,**options)
            values=updated['p']
        return values.square().sum()
    assert torch.autograd.gradcheck(run,(original,),eps=1e-5,atol=1e-6,rtol=1e-5)


def test_functional_adam_zero_gradient_history_has_finite_actual_hypergradient():
    from natlang_neuralese.train.optim import functional_named_service_step
    value=torch.tensor([0.,1.],dtype=torch.float64,requires_grad=True)
    gradient=torch.autograd.grad(value.square().sum()/2,value,create_graph=True)[0]
    updated,_=functional_named_service_step({'p':value},{'p':gradient},{},name='adam',learning_rate=.01)
    derivative=torch.autograd.grad(updated['p'].sum(),value)[0]
    assert torch.isfinite(derivative).all()
    torch.testing.assert_close(derivative[0],torch.tensor(1.-.01/1e-8,dtype=value.dtype),rtol=1e-14,atol=1e-8)


@pytest.mark.parametrize('family,target',[('lfm','q_proj.weight'),('lfm','feed_forward.w1.weight'),
    ('mellum','q_proj.weight'),('mellum','experts.gate_up')])
@pytest.mark.parametrize('precision',['storage','fp32'])
def test_actual_native_adapter_coefficient_second_derivative_and_object_isolation(tmp_path,family,target,precision):
    from dataclasses import replace
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.model.native_adapters import create_adapter,target_parameters,run_native
    engine=engine_fixture(tmp_path,family)
    model=engine.shared_model
    name=next(name for name in target_parameters(model) if name.endswith(target))
    adapter=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    adapter=engine.store.put_native_tensor_artifact(replace(adapter.artifact,
        tensors={key:torch.full_like(value,.04) for key,value in adapter.artifact.tensors.items()}))
    coefficient=adapter.artifact.tensors[name].to(engine.device).clone().requires_grad_(True)
    originals=dict(model.named_parameters());storage={name:p.data_ptr() for name,p in originals.items()}
    layout=plan()
    try:
        def execute():
            hidden=model(layout,retain_states=False)['h_final']
            loss=model.logits(hidden).float().square().mean()
            first=torch.autograd.grad(loss,coefficient,create_graph=True)[0]
            second=torch.autograd.grad(first.square().sum(),coefficient)[0]
            assert torch.isfinite(second).all() and second.norm()>0
        with reference_arithmetic(model):
            run_native(engine,execute,[adapter.id],leaves={adapter.id:{name:coefficient}},gradient_precision=precision)
        assert all(dict(model.named_parameters())[name] is p and p.data_ptr()==storage[name] for name,p in originals.items())
        assert active_reference_arithmetic() is None
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_distinct_producer_consumer_actors_have_cross_actor_second_derivative(tmp_path,family):
    from dataclasses import replace
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.model.native_adapters import create_adapter,target_parameters
    from natlang_neuralese.model.native_actor_graph import run_invocation_actors
    engine=engine_fixture(tmp_path,family)
    model=engine.shared_model
    name=next(name for name in target_parameters(model) if name.endswith('v_proj.weight'))
    base=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    actors=[engine.store.put_native_tensor_artifact(replace(base.artifact,tensors={name:
        torch.full_like(base.artifact.tensors[name],value)})) for value in (.04,-.02)]
    leaves={actor.id:{name:actor.artifact.tensors[name].to(engine.device).clone().requires_grad_(True)} for actor in actors}
    layout=plan()
    try:
        def execute(bank):
            result=model(layout,retain_states=False)
            # Only the reader has loss: producer derivatives must traverse the
            # imported state; no producer own-token objective can fake this.
            reader=result['h_final'][:,layout.offsets[1]:]
            loss=model.logits(reader).float().square().mean()
            first=torch.autograd.grad(loss,[leaves[a.id][name] for a in actors],create_graph=True)
            assert all(g.norm()>0 for g in first)
            cross=torch.autograd.grad(first[0].sum(),leaves[actors[1].id][name])[0]
            assert torch.isfinite(cross).all() and cross.norm()>0
        with reference_arithmetic(model):
            run_invocation_actors(engine,['writer','reader'],{'writer':[actors[0].id],'reader':[actors[1].id]},
                execute,leaves=leaves,gradient_precision='storage')
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_native_external_kv_and_hybrid_leaf_second_derivatives(tmp_path,family):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.serve.native_grad import _leaf_block
    engine=engine_fixture(tmp_path,family)
    try:
        capture=engine.native_write_text('t4 t5')
        original=engine.store.get_native_state(capture['id']).block
        block,leaves=_leaf_block(original)
        roles=engine.role_tokens
        example=parse_graph({'schema':SCHEMA,'id':'external-order2','split':'train',
            'source_groups':['order2-external-train'],'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
            'external_states':{'wire':{'id':capture['id'],'token_count':len(block.token_ids)}},
            'nodes':[{'name':'reader','token_ids':[4,roles.consume_open,roles.consume_close,7],
                'target_weights':[0,1,0,1],'imports':[{'source':'wire','start':0,'stop':len(block.token_ids),'at':2}]}]},
            tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint'])
        with reference_arithmetic(engine.shared_model):
            loss,_=graph_loss(engine.shared_model,example,external_blocks={'wire':block},
                external_identity={'wire':engine.native_identity})
            first=torch.autograd.grad(loss,list(leaves.values()),create_graph=True)
            second=torch.autograd.grad(sum(g.square().sum() for g in first),list(leaves.values()))
            assert all(torch.isfinite(g).all() for g in second)
            assert sum(float(g.square().sum()) for g in second)>0
        if family=='lfm':assert any(name.endswith('gated_inputs') for name in leaves)
        assert all(torch.equal(original.layers[i][key],block.layers[i][key])
                   for i,layer in enumerate(original.layers) for key in layer if key!='kind')
    finally:engine.store.close()


@pytest.mark.parametrize('state',[{'v.p':torch.ones(2)},
    {'m.p':torch.zeros(2),'v.p':-torch.ones(2),'step.p':torch.tensor([1.]),'optimizer.step':torch.tensor([1.])},
    {'m.p':torch.zeros(2),'v.p':torch.ones(2),'step.p':torch.tensor([3.]),'optimizer.step':torch.tensor([1.])}])
def test_functional_update_rejects_missing_or_forged_history_without_reset(state):
    from natlang_neuralese.train.optim import functional_named_service_step
    p=torch.ones(2,requires_grad=True)
    with pytest.raises(ValueError,match='history|complete named'):
        functional_named_service_step({'p':p},{'p':p*2},state,name='adam',learning_rate=.01)


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('algorithm',['sgd','adam'])
@pytest.mark.parametrize('heterogeneous',[False,True])
@pytest.mark.parametrize('file_offload',[False,True])
def test_repeated_real_native_updates_recompute_exact_earlier_actor_bindings(tmp_path,family,algorithm,heterogeneous,file_offload):
    from dataclasses import replace
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.model.native_adapters import create_adapter,target_parameters,run_native
    from natlang_neuralese.model.native_actor_graph import run_invocation_actors
    from natlang_neuralese.train.optim import functional_named_service_step
    engine=engine_fixture(tmp_path,family)
    model=engine.shared_model
    offload=None
    if file_offload:
        from natlang_neuralese.common.checkpoint_offload import configure_checkpoint_input_offload
        offload=configure_checkpoint_input_offload(model.port,enabled=True,directory=tmp_path/'saved-inputs')
    parameter_name=next(name for name in target_parameters(model) if name.endswith('v_proj.weight'))
    base=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[parameter_name]})
    initial=[torch.full_like(base.artifact.tensors[parameter_name],value).to(engine.device)
             for value in ((.04,-.02) if heterogeneous else (.04,))]
    roles=engine.role_tokens
    row={'schema':SCHEMA,'id':'derived-native-updates','split':'train','source_groups':['derived-train-only'],
        'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],'nodes':[
            {'name':'writer','token_ids':[4,roles.produce_open,5,6,roles.produce_close],
             'producing':[[2,4]],'target_weights':[0,0,0,0,0]},
            {'name':'reader','token_ids':[7,roles.consume_open,roles.consume_close,8,9],
             'imports':[{'source':'writer','start':2,'stop':4,'at':2}],'target_weights':[0,1,0,1,1]}]}
    example=parse_graph(row,tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint'])
    def run(checkpointed):
        roots=[value.clone().requires_grad_(True) for value in initial]
        current=roots;state={}
        def objective(values,finish):
            artifacts=[engine.store.put_native_tensor_artifact(replace(base.artifact,
                tensors={parameter_name:value.detach().cpu()})) for value in values]
            leaves={artifact.id:{parameter_name:value} for artifact,value in zip(artifacts,values)}
            def callback(*unused):
                loss,_=graph_loss(model,example,checkpoint_layers=checkpointed)
                return finish(loss)
            if heterogeneous:
                return run_invocation_actors(engine,['writer','reader'],
                    {'writer':[artifacts[0].id],'reader':[artifacts[1].id]},callback,
                    leaves=leaves,gradient_precision='fp32')
            return run_native(engine,callback,[artifacts[0].id],leaves=leaves,gradient_precision='fp32')
        with reference_arithmetic(model):
            for _ in range(2):
                gradients=objective(current,lambda loss:torch.autograd.grad(loss,current,create_graph=True))
                names=[str(index) for index in range(len(current))]
                updated,state=functional_named_service_step(dict(zip(names,current)),dict(zip(names,gradients)),state,
                    name=algorithm,learning_rate=.003,momentum=.4,eps=1e-7)
                current=[updated[name] for name in names]
            result=objective(current,lambda loss:(loss.detach(),torch.autograd.grad(loss,roots)))
        return result[0],tuple(value.detach() for value in result[1]),tuple(value.detach() for value in current)
    try:
        expected=run(False);actual=run(True)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        if offload is not None:
            stats=offload.snapshot()
            assert stats['checkpoint_calls']>0 and stats['packs']>0 and stats['unpacks']>0
        assert all(value.norm()>0 and torch.isfinite(value).all() for value in actual[1])
        for got,reference in zip(actual[1],expected[1]):
            relative=(got-reference).double().norm()/reference.double().norm()
            cosine=torch.nn.functional.cosine_similarity(got.double().flatten(),reference.double().flatten(),dim=0)
            assert relative<2e-5 and cosine>1.-1e-6
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_engine_cached_reference_forward_and_coefficient_hvp_matches_graph(tmp_path,family):
    from dataclasses import replace
    from test_native_serving_adapters import engine_fixture,nonzero_adapter
    from natlang_neuralese.model.native_adapters import run_native
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    coefficient=adapter.artifact.tensors[name].to(engine.device).clone().requires_grad_(True)
    tokens=(4,5,6,7)
    layout=compile_graph([Invocation('answer',tokens)])
    def execute():
        hidden=engine.shared_model(layout,retain_states=False)['h_final'][:,-1]
        def observe(states):
            logits=engine.shared_model.logits(states)
            loss=-logits.float().log_softmax(-1)[...,8].sum()
            grad=torch.autograd.grad(loss,coefficient,create_graph=True)[0]
            hvp=torch.autograd.grad(grad.square().sum(),coefficient,retain_graph=True)[0]
            return states.detach(),logits.detach(),grad.detach(),hvp.detach()
        expected=observe(hidden)
        cache=None
        for token in tokens:
            output=engine._native_forward([token],cache);cache=output['cache']
        actual=observe(output['h_final'][:,-1])
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        for got,ref in zip(actual[2:],expected[2:]):
            assert ref.norm()>0
            assert (got-ref).double().norm()/ref.double().norm()<2e-5
    try:
        with reference_arithmetic(engine.shared_model):
            run_native(engine,execute,[adapter.id],leaves={adapter.id:{name:coefficient}},gradient_precision='fp32')
    finally:engine.store.close()
