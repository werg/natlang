"""Real graph/update/outer-gradient tests for the internal finite DAG executor."""
from dataclasses import replace
import copy
import pytest
import torch
from test_native_serving_adapters import engine_fixture,nonzero_adapter
from natlang_neuralese.serve.native_derivative import execute_derivative_program
from natlang_neuralese.model.native_derivative_contract import PROGRAM
from natlang_neuralese.data.native_graph import SCHEMA


def program(engine,adapter,algorithm='adam'):
    roles=engine.role_tokens
    row={'schema':SCHEMA,'id':'actual-derivative-graph','split':'train','source_groups':['derivative-train-only'],
        'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],'nodes':[
            {'name':'writer','token_ids':[4,roles.produce_open,5,6,roles.produce_close],
             'producing':[[2,4]],'target_weights':[0,0,0,0,0]},
            {'name':'reader','token_ids':[7,roles.consume_open,roles.consume_close,8,9],
             'imports':[{'source':'writer','start':2,'stop':4,'at':2}],'target_weights':[0,1,0,1,1]}]}
    request={'schema':'natlang.native-gradient-request/1','graph':row,'adapters':[adapter.id],
        'adapter_gradient_precision':'fp32'}
    options={'name':algorithm,'learning_rate':.003,'momentum':.4,'eps':1e-5}
    return {'schema':PROGRAM,'arithmetic':{'schema':'natlang.native-reference-arithmetic/1','query_chunk_tokens':2},
        'inputs':{'a':{'kind':'adapter','id':adapter.id}},'nodes':[
            {'kind':'gradient','name':'g0','value':'a','request':request,'bindings':{}},
            {'kind':'optimizer','name':'a1','value':'a','gradient':'g0','optimizer':options,'optimizer_state':None},
            {'kind':'gradient','name':'g1','value':'a1','request':request,'bindings':{'a':'a1'}},
            {'kind':'optimizer','name':'a2','value':'a1','gradient':'g1','optimizer':options,'optimizer_state':'a1'}],
        'outer':{'request':request,'bindings':{'a':'a2'}},'wrt':['a']}


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('algorithm',['sgd','adam'])
def test_actual_finite_derivative_two_updates_and_numeric_direction(tmp_path,family,algorithm):
    engine=engine_fixture(tmp_path,family)
    # Finite differences check the continuous FP32 instance. BF16 storage
    # casts are discrete and need a same-arithmetic autograd oracle below.
    if family=='mellum':
        from natlang_neuralese.common.hashing import canonical_json_sha256_hex
        from natlang_neuralese.model.native_model import native_identity
        engine.shared_model.float()
        engine.checkpoint_sha256=canonical_json_sha256_hex({'diagnostic_parent':engine.checkpoint_sha256,'transform':'exact-fp32-promotion/1'})
        engine.native_identity=native_identity(engine.shared_model,engine.model_config,checkpoint_sha256=engine.checkpoint_sha256)
        engine.embedding_weight_sha256=engine.native_identity['embedding_weight_sha256']
    adapter,name=nonzero_adapter(engine)
    original=[(key,id(value),value.detach().clone()) for key,value in engine.shared_model.named_parameters()]
    value=adapter.artifact.tensors[name]
    direction=torch.linspace(-1.,1.,value.numel()).reshape_as(value)
    try:
        source=program(engine,adapter,algorithm)
        result=execute_derivative_program(engine,source,'program-owner')
        gradient=engine.store.get_native_tensor_artifact(result['gradients']['a']['id']).artifact.tensors[name]
        assert torch.isfinite(gradient).all() and gradient.norm()>0
        execution=engine.store.get_native_tensor_artifact(result['execution_id']).artifact
        assert execution.provenance['outputs']==result['outputs']
        assert engine.store.get_native_tensor_artifact(result['optimizer_states']['a2']).artifact.provenance['step']==2
        losses=[];epsilon=.002
        for sign in (-1,1):
            changed=engine.store.put_native_tensor_artifact(replace(adapter.artifact,tensors={name:value+sign*epsilon*direction}))
            probe=program(engine,changed,algorithm)
            losses.append(execute_derivative_program(engine,probe)['loss'])
        numeric=(losses[1]-losses[0])/(2*epsilon)
        analytic=float((gradient*direction).sum())
        assert abs(analytic-numeric)<max(2e-4,abs(numeric)*.04),(analytic,numeric)
        actual=dict(engine.shared_model.named_parameters())
        for key,ident,before in original:
            assert id(actual[key])==ident and torch.equal(actual[key],before)
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_program_rejects_stale_gradient_and_forward_reference(tmp_path,family):
    engine=engine_fixture(tmp_path,family)
    adapter,_=nonzero_adapter(engine)
    try:
        bad=program(engine,adapter)
        bad['nodes'][-1]['gradient']='g0'
        with pytest.raises(ValueError,match='another exact earlier value'):execute_derivative_program(engine,bad)
        bad=program(engine,adapter)
        bad['nodes'][0]['bindings']={'a':'a2'}
        with pytest.raises(ValueError,match='available values'):execute_derivative_program(engine,bad)
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('algorithm',['sgd','adam'])
def test_source_dtype_program_matches_explicit_updated_actor_autograd(tmp_path,family,algorithm):
    from natlang_neuralese.common.native_higher_order import reference_arithmetic
    from natlang_neuralese.model.native_adapters import run_native
    from natlang_neuralese.data.native_graph import parse_graph,graph_loss
    from natlang_neuralese.train.optim import functional_named_service_step
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    source=program(engine,adapter,algorithm)
    request=source['outer']['request']
    example=parse_graph(request['graph'],tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint'])
    root=adapter.artifact.tensors[name].to(engine.device).clone().requires_grad_(True)
    current=root;moments={}
    def evaluate(value,finish):
        actual=engine.store.put_native_tensor_artifact(replace(adapter.artifact,tensors={name:value.detach().cpu()}))
        return run_native(engine,lambda:finish(graph_loss(engine.shared_model,example)[0]),[actual.id],
            leaves={actual.id:{name:value}},gradient_precision='fp32')
    try:
        with reference_arithmetic(engine.shared_model,query_chunk_tokens=2):
            for _ in range(2):
                gradient=evaluate(current,lambda loss:torch.autograd.grad(loss,current,create_graph=True)[0])
                updated,moments=functional_named_service_step({name:current},{name:gradient},moments,**source['nodes'][1]['optimizer'])
                current=updated[name]
            expected_loss,expected=evaluate(current,lambda loss:(loss.detach(),torch.autograd.grad(loss,root)[0]))
        result=execute_derivative_program(engine,source)
        actual=engine.store.get_native_tensor_artifact(result['gradients']['a']['id']).artifact.tensors[name].to(engine.device)
        assert result['loss']==float(expected_loss)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        assert (actual-expected).double().norm()/expected.double().norm()<2e-5
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('target_kind',['model-parameters','native-state'])
def test_named_backbone_and_hybrid_state_programs_have_real_outer_gradients(tmp_path,family,target_kind):
    engine=engine_fixture(tmp_path,family,service_template=True)
    adapter,_=nonzero_adapter(engine)
    source=program(engine,adapter,'sgd')
    for objective in [source['outer'],*(node for node in source['nodes'] if node['kind']=='gradient')]:
        objective['request']=copy.deepcopy(objective['request'])
        objective['request'].pop('adapters')
    if target_kind=='model-parameters':
        name=next(name for name in dict(engine.shared_model.named_parameters()) if name.endswith('v_proj.weight'))
        source['inputs']={'a':{'kind':target_kind,'names':[name]}}
    else:
        ident=engine.native_write_text('t5 t6')['id']
        source['inputs']={'a':{'kind':target_kind,'id':ident}}
        for objective in [source['outer'],*(node for node in source['nodes'] if node['kind']=='gradient')]:
            graph=objective['request']['graph']
            graph['external_states']={'prior':{'id':ident,'token_count':2}}
            graph['nodes']=graph['nodes'][1:]
            graph['nodes'][0]['imports']=[{'source':'prior','start':0,'stop':2,'at':2}]
    try:
        result=execute_derivative_program(engine,source)
        tensors=engine.store.get_native_tensor_artifact(result['gradients']['a']['id']).artifact.tensors
        assert all(torch.isfinite(value).all() for value in tensors.values())
        assert sum(value.float().norm() for value in tensors.values())>0
        value_id=result['outputs']['a2']
        if target_kind=='native-state':
            from natlang_neuralese.model.native_state_derivation import verify_state_derivation
            value=engine.store.get_native_state(value_id).block
            verify_state_derivation(engine.store,value)
            assert value.derivation['step']==2
        else:
            assert engine.store.get_native_tensor_artifact(value_id).artifact.kind=='parameter-set'
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_unused_program_input_has_exact_zero_derivative_and_no_fake_update(tmp_path,family):
    engine=engine_fixture(tmp_path,family)
    adapter,_=nonzero_adapter(engine)
    source=program(engine,adapter,'sgd')
    for objective in [source['outer'],*(node for node in source['nodes'] if node['kind']=='gradient')]:
        objective['request']=copy.deepcopy(objective['request']);objective['request'].pop('adapters')
    try:
        result=execute_derivative_program(engine,source)
        gradient=engine.store.get_native_tensor_artifact(result['gradients']['a']['id']).artifact
        assert all(torch.count_nonzero(value)==0 for value in gradient.tensors.values())
        updated=engine.store.get_native_tensor_artifact(result['outputs']['a2']).artifact
        assert all(torch.equal(value,adapter.artifact.tensors[key]) for key,value in updated.tensors.items())
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_heterogeneous_meta_gradient_distinguishes_original_from_updated_actors(tmp_path,family):
    engine=engine_fixture(tmp_path,family)
    from natlang_neuralese.model.native_adapters import create_adapter,target_parameters
    name=next(key for key in target_parameters(engine.shared_model) if key.endswith('v_proj.weight'))
    base=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    source=engine.store.put_native_tensor_artifact(replace(base.artifact,tensors={name:torch.full_like(base.artifact.tensors[name],.04)}))
    target,_=nonzero_adapter(engine,'tiny')
    body=program(engine,source)
    body['inputs']['b']={'kind':'adapter','id':target.id};body['wrt']=['a','b']
    for objective in [body['outer'],*(node for node in body['nodes'] if node['kind']=='gradient')]:
        objective['request']=copy.deepcopy(objective['request']);objective['request'].pop('adapters')
        objective['request']['invocation_adapters']={'writer':[source.id],'reader':[target.id]}
    try:
        result=execute_derivative_program(engine,body)
        gradients={key:engine.store.get_native_tensor_artifact(value['id']).artifact for key,value in result['gradients'].items()}
        assert all(sum(value.float().norm() for value in gradient.tensors.values())>0 for gradient in gradients.values())
        origin=gradients['a'].provenance['native_derivative_target']
        assert origin['value_id']==source.id
        actor=gradients['a'].provenance['functional_actor']
        assert actor['invocation_adapters']['writer'][0]['id']==result['outputs']['a2']
        assert actor['invocation_adapters']['reader'][0]['id']==target.id
        from natlang_neuralese.model.native_adapters import optimize_adapter
        # The optimizer updates the original differentiated value, not the
        # current inner actor merely because it generated the outer objective.
        update=optimize_adapter(engine,gradients['a'],{'name':'sgd','learning_rate':.001},None)
        assert update['adapter']['id'] not in (source.id,result['outputs']['a2'])
        # Even another real, same-base actor cannot be substituted for the
        # actual evaluation recorded by the immutable outer execution.
        from natlang_neuralese.common.hashing import canonical_json_sha256_hex
        forged=copy.deepcopy(gradients['a'].provenance)
        forged['functional_actor']=copy.deepcopy(forged['functional_actor'])
        forged['functional_actor']['invocation_adapters']['writer'][0]['id']=source.id
        forged['functional_actor_sha256']=canonical_json_sha256_hex(forged['functional_actor'])
        with pytest.raises(ValueError,match='evaluated per-invocation|actual execution'):
            engine.store.put_native_tensor_artifact(replace(gradients['a'],provenance=forged))
        forged=copy.deepcopy(gradients['a'].provenance)
        forged['evaluation_inputs']['a']['value_id']=source.id
        with pytest.raises(ValueError,match='actual execution'):
            engine.store.put_native_tensor_artifact(replace(gradients['a'],provenance=forged))
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('algorithm',['sgd','adam'])
@pytest.mark.parametrize('state_kind',['attention','hybrid'])
def test_state_optimizer_cpu_transfer_retains_exact_outer_derivative(tmp_path,family,algorithm,state_kind):
    from natlang_neuralese.common.native_higher_order import reference_arithmetic
    from natlang_neuralese.model.native_adapters import run_native
    from natlang_neuralese.model.native_state_derivation import state_values,replace_state_values
    from natlang_neuralese.data.native_graph import parse_graph,graph_loss
    from natlang_neuralese.train.optim import functional_named_service_step
    engine=engine_fixture(tmp_path,family,service_template=True)
    adapter,_=nonzero_adapter(engine)
    ident=engine.native_write_text('t5 t6')['id'];block=engine.store.get_native_state(ident).block
    candidates=[name for name in state_values(block) if name.endswith('.v' if state_kind=='attention' else '.gated_inputs')]
    if not candidates:engine.store.close();pytest.skip('this backbone has no hybrid convolution state')
    name=candidates[0];source=program(engine,adapter,algorithm)
    source['inputs']={'a':{'kind':'native-state','id':ident,'names':[name]}}
    for objective in [source['outer'],*(node for node in source['nodes'] if node['kind']=='gradient')]:
        objective['request']=copy.deepcopy(objective['request']);objective['request'].pop('adapters')
        graph=objective['request']['graph'];graph['nodes']=graph['nodes'][1:]
        graph['external_states']={'prior':{'id':ident,'token_count':2}}
        graph['nodes'][0]['imports']=[{'source':'prior','start':0,'stop':2,'at':2}]
    example=parse_graph(source['outer']['request']['graph'],tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint'])
    root=state_values(block)[name].to(engine.device).clone().requires_grad_(True);current=root;moments={}
    def evaluate(value):
        return run_native(engine,lambda:graph_loss(engine.shared_model,example,
            external_blocks={'prior':replace_state_values(block,{name:value})},
            external_identity={'prior':engine.native_identity})[0],gradient_precision='fp32')
    try:
        with reference_arithmetic(engine.shared_model,query_chunk_tokens=2):
            for _ in range(2):
                gradient=torch.autograd.grad(evaluate(current),current,create_graph=True)[0]
                work=current.to(device='cpu',dtype=torch.float32)
                updated,moments=functional_named_service_step({name:work},{name:gradient.to(work)},moments,
                    **source['nodes'][1]['optimizer'])
                current=updated[name].to(root)
            expected_loss=evaluate(current);expected=torch.autograd.grad(expected_loss,root)[0]
        result=execute_derivative_program(engine,source)
        actual=engine.store.get_native_tensor_artifact(result['gradients']['a']['id']).artifact.tensors[name].to(engine.device)
        assert expected.double().norm()>0
        assert result['loss']==float(expected_loss.detach())
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        assert (actual-expected).double().norm()/expected.double().norm()<2e-5
    finally:engine.store.close()
