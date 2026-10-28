"""Inline conditional scores, one-forward integration and real derived updates."""
import copy
from types import SimpleNamespace
import pytest
import torch
from test_native_derivative_program import program
from test_native_serving_adapters import engine_fixture, nonzero_adapter
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.serve.native_grad import native_graph_gradient
from natlang_neuralese.serve.native_derivative import execute_derivative_program
from natlang_neuralese.serve.native_query_objective import NativeQueryObjective, SCHEMA


def query_spec(graph, terms):
    return {'kind':'native-query','schema':SCHEMA,'graph_sha256':canonical_json_sha256_hex(graph),'terms':terms}


def query_nll(graph):
    return query_spec(graph,[{'kind':'nll','reduction':'mean','actions':[
        {'query':5,'token_id':graph['nodes'][1]['token_ids'][1]},
        {'query':7,'token_id':8},{'query':8,'token_id':9}]}])


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_inline_native_loss_and_all_selected_adjoints_equal_existing_ce_one_forward(tmp_path,family,monkeypatch):
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    request=copy.deepcopy(program(engine,adapter)['outer']['request'])
    request['target']={'kind':'adapter','id':adapter.id}
    forwards=[]
    original=engine.shared_model.forward
    def counted(*args,**kwargs):
        forwards.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(engine.shared_model,'forward',counted)
    try:
        ce=native_graph_gradient(engine,request,None)
        ce_grad=engine.store.get_native_tensor_artifact(ce['gradient']['id']).artifact.tensors[name]
        request['objective']=query_nll(request['graph'])
        scored=native_graph_gradient(engine,request,None)
        query_grad=engine.store.get_native_tensor_artifact(scored['gradient']['id']).artifact.tensors[name]
        assert len(forwards)==2, 'one graph forward per complete objective'
        assert scored['loss']==pytest.approx(ce['loss'],abs=1e-6)
        torch.testing.assert_close(query_grad,ce_grad,rtol=2e-5,atol=3e-6)
        assert query_grad.double().norm()>0
        assert (query_grad-ce_grad).double().norm()/ce_grad.double().norm()<2e-5
        assert scored['objective']=='native-query-objective/1'
        bad=copy.deepcopy(request);bad['objective']['terms'][0]['actions'][0]['token_id']=0
        with pytest.raises(Exception,match='actual next own input token'):
            native_graph_gradient(engine,bad,None)
        bad=copy.deepcopy(request);bad['objective']['terms'][0]['actions']=[{'query':6,'token_id':engine.role_tokens.consume_close}]
        with pytest.raises(Exception,match='actual next own input token'):
            native_graph_gradient(engine,bad,None)
        assert len(forwards)==2, 'invalid coordinates reject before forward'
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('algorithm',['sgd','adam'])
def test_inline_query_two_updated_actors_preserve_hypergradient(tmp_path,family,algorithm):
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    original=program(engine,adapter,algorithm)
    queried=copy.deepcopy(original)
    for objective in [queried['outer'],*(node for node in queried['nodes'] if node['kind']=='gradient')]:
        objective['request']['objective']=query_nll(objective['request']['graph'])
    try:
        ce=execute_derivative_program(engine,original)
        query=execute_derivative_program(engine,queried)
        actual=engine.store.get_native_tensor_artifact(query['gradients']['a']['id']).artifact.tensors[name]
        expected=engine.store.get_native_tensor_artifact(ce['gradients']['a']['id']).artifact.tensors[name]
        assert query['loss']==pytest.approx(ce['loss'],abs=1e-6)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        assert (actual-expected).double().norm()/expected.double().norm()<2e-5
        for node in ('a1','a2'):
            a=engine.store.get_native_tensor_artifact(query['outputs'][node]).artifact.tensors[name]
            b=engine.store.get_native_tensor_artifact(ce['outputs'][node]).artifact.tensors[name]
            torch.testing.assert_close(a,b,rtol=2e-5,atol=3e-6)
    finally:engine.store.close()


@pytest.mark.parametrize('rule',['logLoss','brier','rps','expectedReward'])
def test_signed_query_decision_and_stopped_teacher_exact_manual_oracle(rule):
    graph={'schema':'natlang.neuralese-native-graph/1','id':'conditional-oracle','split':'train',
        'source_groups':['explicit-training-example'],'tokenizer_fingerprint':'f'*64,'nodes':[
            {'name':'option-a','token_ids':[0,1,2],'target_weights':[0,0,0]},
            {'name':'option-b','token_ids':[3,1,2],'target_weights':[0,0,0]}]}
    example=parse_graph(graph,tokenizer_fingerprint='f'*64)
    actions=lambda queries:[{'query':q,'token_id':t} for q,t in zip(queries,[1,2])]
    a,b=actions([0,1]),actions([3,4])
    term={'kind':'decision','options':[a,b],'rule':rule,'probabilities':[.3,.7],'rewards':[-1.,2.],'weight':-.4}
    spec=query_spec(graph,[term,{'kind':'distill','actions':a,'teacher_actions':b,'weight':.8},
                                {'kind':'nll','actions':a,'reduction':'sum','weight':.2}])
    objective=NativeQueryObjective(spec,graph,example,vocabulary=4)
    torch.manual_seed(32)
    h=torch.randn(6,4,requires_grad=True)
    model=SimpleNamespace(logits=lambda value:value)
    actual,_=objective(model,example,{'plan':example.plan,'h_final':h[None]})
    log=h.log_softmax(-1)
    scores=torch.stack([log[0,1]+log[1,2],log[3,1]+log[4,2]])
    # Proper scores use the public readout's log-softmax then exp; the
    # expected-reward branch uses softmax directly, as its existing contract.
    p=scores.softmax(0) if rule=='expectedReward' else scores.log_softmax(0).exp()
    goal=p.new_tensor([.3,.7])
    proper=(-(goal*scores.log_softmax(0)).sum() if rule=='logLoss' else
            ((p-goal)**2).sum() if rule=='brier' else
            ((p.cumsum(0)-goal.cumsum(0))[:-1]**2).sum() if rule=='rps' else
            -(p*p.new_tensor([-1.,2.])).sum())
    teacher=log[[3,4]].detach()
    kl=(teacher.exp()*(teacher-log[[0,1]])).sum(-1).mean()
    expected=-.4*proper+.8*kl-.2*(log[0,1]+log[1,2])
    # The objective uses the shared stable softmax numerics (common/stable_softmax.py): a
    # different FP32 operation order than this plain-PyTorch oracle, equal to a few ulps.
    assert actual.item()==pytest.approx(expected.item(),abs=1e-6)
    ga=torch.autograd.grad(actual,h,retain_graph=True)[0]
    ge=torch.autograd.grad(expected,h)[0]
    torch.testing.assert_close(ga,ge,rtol=2e-6,atol=2e-7)
    isolated=NativeQueryObjective(query_spec(graph,[spec['terms'][1]]),graph,example,vocabulary=4)
    loss,_=isolated(model,example,{'plan':example.plan,'h_final':h[None]})
    stopped=torch.autograd.grad(loss,h)[0]
    assert torch.count_nonzero(stopped[3:])==0
    assert stopped[:2].norm()>0


def test_query_masks_and_graph_binding_fail_before_forward():
    graph={'schema':'natlang.neuralese-native-graph/1','id':'masked','split':'train','source_groups':['train'],
        'tokenizer_fingerprint':'f'*64,'nodes':[{'name':'x','token_ids':[0,1],'target_weights':[0,0]}]}
    example=parse_graph(graph,tokenizer_fingerprint='f'*64)
    spec=query_spec(graph,[{'kind':'nll','reduction':'mean','actions':[{'query':0,'token_id':1,'banned_token_ids':[1]}]}])
    with pytest.raises(ValueError,match='excludes its target'):NativeQueryObjective(spec,graph,example,vocabulary=4)
    spec['graph_sha256']='0'*64
    with pytest.raises(ValueError,match='exact graph'):NativeQueryObjective(spec,graph,example,vocabulary=4)


@pytest.mark.parametrize('rule',['logLoss','brier','rps'])
def test_decision_teacher_exact_manual_gradient_stops_its_distinct_hidden_and_head(rule):
    from natlang_neuralese.common.decision_objective import decision_loss
    graph={'schema':'natlang.neuralese-native-graph/1','id':'decision-teacher','split':'train',
        'source_groups':['explicit-train'],'tokenizer_fingerprint':'f'*64,
        'nodes':[{'name':str(i),'token_ids':[0,1],'target_weights':[0,0]} for i in range(4)]}
    example=parse_graph(graph,tokenizer_fingerprint='f'*64)
    teacher_id='nnt1_'+'a'*52
    spec=query_spec(graph,[{'kind':'decision-distill','rule':rule,
        'options':[[{'query':0,'token_id':1}],[{'query':2,'token_id':1}]],
        'teacher_options':[[{'query':4,'token_id':1}],[{'query':6,'token_id':1}]],
        'teacher_readouts':[teacher_id,teacher_id]}])
    torch.manual_seed(581)
    hidden=torch.randn(8,3,requires_grad=True)
    student=torch.randn(3,4,requires_grad=True)
    teacher=torch.randn(3,4,requires_grad=True)
    model=SimpleNamespace(logits=lambda value:value@student)
    objective=NativeQueryObjective(spec,graph,example,vocabulary=4,
        resolve_readout=lambda ident:lambda value:value@teacher)
    loss,_=objective(model,example,{'plan':example.plan,'h_final':hidden[None]})
    scores=(hidden[[0,2]]@student).log_softmax(-1)[:,1]
    with torch.no_grad():goal=(hidden[[4,6]]@teacher).log_softmax(-1)[:,1].softmax(0)
    expected=decision_loss(scores,rule=rule,probabilities=goal.tolist())
    actual_grad=torch.autograd.grad(loss,(hidden,student,teacher),retain_graph=True,allow_unused=True)
    expected_grad=torch.autograd.grad(expected,(hidden,student))
    torch.testing.assert_close(loss,expected,rtol=2e-6,atol=2e-7)
    for actual,wanted in zip(actual_grad[:2],expected_grad):
        torch.testing.assert_close(actual,wanted,rtol=2e-6,atol=2e-7)
    assert actual_grad[0][:4].norm()>0
    assert torch.count_nonzero(actual_grad[0][4:])==0
    assert actual_grad[2] is None


def test_decision_teacher_dependency_inventory_retains_every_exact_readout():
    from natlang_neuralese.model.native_derivative_contract import query_readout_dependencies
    a,b='nnt1_'+'a'*52,'nnt1_'+'b'*52
    assert query_readout_dependencies({'kind':'native-query','terms':[{
        'kind':'decision-distill','readouts':[a,None],'teacher_readouts':[b,b]}]})==[a,b]
    with pytest.raises(ValueError,match='ordered list'):
        query_readout_dependencies({'kind':'native-query','terms':[{'teacher_readouts':'not-an-array'}]})


def test_distinct_teacher_readout_has_exact_analytic_student_adjoint_and_no_teacher_adjoint():
    graph={'schema':'natlang.neuralese-native-graph/1','id':'different-heads','split':'train','source_groups':['train'],
        'tokenizer_fingerprint':'f'*64,'nodes':[
            {'name':'student','token_ids':[0,1,2],'target_weights':[0,0,0]},
            {'name':'teacher','token_ids':[3,1,2],'target_weights':[0,0,0]}]}
    example=parse_graph(graph,tokenizer_fingerprint='f'*64)
    teacher_id='nnt1_'+'a'*52
    spec=query_spec(graph,[{'kind':'distill','actions':[{'query':0,'token_id':1},{'query':1,'token_id':2}],
        'teacher_actions':[{'query':3,'token_id':1},{'query':4,'token_id':2}],'teacher_readout':teacher_id}])
    torch.manual_seed(49)
    h=torch.randn(6,3,requires_grad=True)
    student=torch.randn(3,4,requires_grad=True)
    teacher=torch.randn(3,4,requires_grad=True)
    model=SimpleNamespace(logits=lambda value:value@student)
    objective=NativeQueryObjective(spec,graph,example,vocabulary=4,
        resolve_readout=lambda ident:(lambda value:value@teacher) if ident==teacher_id else None)
    value,_=objective(model,example,{'plan':example.plan,'h_final':h[None]})
    dh,dw,dt=torch.autograd.grad(value,(h,student,teacher),allow_unused=True)
    goal=(h.detach()[[3,4]]@teacher.detach()).softmax(-1)
    predicted=(h.detach()[[0,1]]@student.detach()).softmax(-1)
    delta=(predicted-goal)/2
    expected_h=torch.zeros_like(h);expected_h[[0,1]]=delta@student.detach().T
    torch.testing.assert_close(dh,expected_h,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(dw,h.detach()[[0,1]].T@delta,rtol=2e-6,atol=2e-7)
    assert dt is None and dh[3:].count_nonzero()==0
    wrong_goal=(h.detach()[[3,4]]@student.detach()).softmax(-1)
    assert (goal-wrong_goal).norm()>.1, 'different teacher head must materially change the target'


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_complete_readout_snapshot_is_distinct_and_restores_every_tensor(tmp_path,family):
    from natlang_neuralese.model.native_readout import snapshot_readout,bind_readout,readout_layout
    engine=engine_fixture(tmp_path,family)
    model=engine.shared_model
    named,layout=readout_layout(model)
    originals={name:(id(value),value.detach().clone()) for name,value in model.named_parameters()}
    hidden=torch.randn(2,model.port.output_weight.shape[1],device=engine.device,dtype=model.port.output_weight.dtype)
    before=model.logits(hidden).detach()
    values={name:value.detach().clone() for name,value in named.items()}
    head_name=next(name for name,value in named.items() if value is model.port.output_weight)
    values[head_name][1:3]*=-1
    try:
        record=snapshot_readout(engine,values=values)
        from natlang_neuralese.model.native_readout import readout_scope
        with readout_scope():
            read=bind_readout(engine,record.id)
            assert bind_readout(engine,record.id) is read
        assert bind_readout(engine,record.id) is not read, "no permanent engine cache retains old readout copies"
        after=read(hidden).detach()
        assert not torch.equal(after[:,1:3],before[:,1:3])
        assert torch.equal(model.logits(hidden),before)
        assert all(id(dict(model.named_parameters())[name])==ident and torch.equal(dict(model.named_parameters())[name],old)
                   for name,(ident,old) in originals.items())
        wrong=copy.deepcopy(record.artifact)
        wrong.provenance={'readout':{**layout,'norm_eps_hex':float(.9).hex()}}
        bad=engine.store.put_native_tensor_artifact(wrong)
        with pytest.raises(ValueError,match='exact base/layout'):bind_readout(engine,bad.id)
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_real_service_distillation_uses_explicit_teacher_head(tmp_path,family):
    from natlang_neuralese.model.native_readout import snapshot_readout,readout_layout
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    request=copy.deepcopy(program(engine,adapter)['outer']['request'])
    graph=request['graph']
    teacher=copy.deepcopy(graph['nodes'][1]);teacher['name']='teacher';teacher['token_ids'][0]=11
    teacher['target_weights']=[0]*len(teacher['token_ids']);graph['nodes'].append(teacher)
    named,_=readout_layout(engine.shared_model)
    values={name:value.detach().clone() for name,value in named.items()}
    head_name=next(name for name,value in named.items() if value is engine.shared_model.port.output_weight)
    values[head_name]*=-.8
    try:
        readout=snapshot_readout(engine,values=values)
        term={'kind':'distill','actions':[{'query':7,'token_id':8},{'query':8,'token_id':9}],
            'teacher_actions':[{'query':12,'token_id':8},{'query':13,'token_id':9}]}
        request['target']={'kind':'adapter','id':adapter.id}
        request['objective']=query_spec(graph,[term])
        current=native_graph_gradient(engine,request,None)
        request['objective']['terms'][0]['teacher_readout']=readout.id
        bound=native_graph_gradient(engine,request,None)
        before=engine.store.get_native_tensor_artifact(current['gradient']['id']).artifact.tensors[name]
        after=engine.store.get_native_tensor_artifact(bound['gradient']['id']).artifact.tensors[name]
        assert abs(bound['loss']-current['loss'])>1e-4
        assert (before-after).double().norm()>1e-6
        assert torch.isfinite(after).all() and after.norm()>0
        from natlang_neuralese.model.native_artifact import native_tensor_dependencies
        artifact=engine.store.get_native_tensor_artifact(bound['gradient']['id']).artifact
        assert readout.id in native_tensor_dependencies(artifact)
        bad=copy.deepcopy(artifact);bad.provenance['query_objective_json']+=' '
        with pytest.raises(ValueError,match='exact objective declaration'):native_tensor_dependencies(bad)
        engine.store.collect({bound['gradient']['id']})
        assert engine.store.get_native_tensor_artifact(readout.id) is not None
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_program_origin_rejects_jointly_rehashed_objective_and_other_root_rebinding(tmp_path,family):
    import json
    from natlang_neuralese.common.jsonio import canonical_json_str
    from natlang_neuralese.model.native_derivative_contract import verify_derivative_gradient_origin
    engine=engine_fixture(tmp_path,family)
    adapter,_=nonzero_adapter(engine)
    other,_=nonzero_adapter(engine,seed=1)
    source=program(engine,adapter)
    source['inputs']['b']={'kind':'adapter','id':other.id}
    for objective in [source['outer'],*(node for node in source['nodes'] if node['kind']=='gradient')]:
        objective['request']['objective']=query_nll(objective['request']['graph'])
    try:
        result=execute_derivative_program(engine,source)
        original=engine.store.get_native_tensor_artifact(result['outputs']['g0']).artifact
        forged=copy.deepcopy(original)
        spec=json.loads(forged.provenance['query_objective_json']);spec['terms'][0]['weight']=-.5
        forged.provenance['query_objective_json']=canonical_json_str(spec)
        forged.provenance['query_objective_sha256']=canonical_json_sha256_hex(spec)
        with pytest.raises(ValueError,match='declared query objective'):verify_derivative_gradient_origin(engine.store,forged)
        forged=copy.deepcopy(original)
        graph=json.loads(forged.provenance['graph_json']);graph['nodes'][0]['token_ids'][0]+=1
        forged.provenance['graph_json']=canonical_json_str(graph)
        forged.provenance['graph_sha256']=canonical_json_sha256_hex(graph)
        with pytest.raises(ValueError,match='declared objective graph'):verify_derivative_gradient_origin(engine.store,forged)
        forged=copy.deepcopy(original)
        forged.provenance['evaluation_inputs']['b']['value_id']=adapter.id
        with pytest.raises(ValueError,match='unchanged declared root'):verify_derivative_gradient_origin(engine.store,forged)
    finally:engine.store.close()
