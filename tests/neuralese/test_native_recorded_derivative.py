"""Recorded-objective envelopes keep real graph gradients in the native DAG."""
import copy
import pytest
import torch
from test_native_derivative_program import program
from test_native_serving_adapters import engine_fixture,nonzero_adapter
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.data.native_graph import SCHEMA
from natlang_neuralese.model.native_state_derivation import state_values
from natlang_neuralese.serve.native_derivative import execute_derivative_program
from natlang_neuralese.serve.native_query_objective import SCHEMA as QUERY_SCHEMA


def _envelope(root,kind,ident,graph,objective):
    turns=[{'authenticated_native_turn':'test-turn'}]
    events=[{'authenticated_graph_event':'test-event'}]
    terms=[{'recorded_term':'test-term'}]
    return {'schema':'natlang.native-recorded-objective-request/1','turns':turns,'events':events,
        'terms':terms,'targets':[{'root':root,'kind':kind,'id':ident}]}


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_recorded_envelopes_run_through_two_true_native_updates(tmp_path,family,monkeypatch):
    """The lowerer stub isolates wiring; graph/query execution and nested derivatives are real."""
    from natlang_neuralese.serve import native_recorded_objective
    from natlang_neuralese.serve.native_query_objective import SCHEMA as QUERY_SCHEMA
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    source=program(engine,adapter,'adam')
    base=copy.deepcopy(source['outer']['request'])
    graph=base['graph']
    objective={'kind':'native-query','schema':QUERY_SCHEMA,'graph_sha256':canonical_json_sha256_hex(graph),
        'terms':[{'kind':'nll','reduction':'mean','actions':[
            {'query':5,'token_id':graph['nodes'][1]['token_ids'][1]},
            {'query':7,'token_id':8},{'query':8,'token_id':9}]}]}
    recorded=_envelope('a','adapter',adapter.id,graph,objective)
    payload={key:recorded[key] for key in ('turns','events','terms')}
    graph['recorded_objective_sha256']=canonical_json_sha256_hex(payload)
    objective['graph_sha256']=canonical_json_sha256_hex(graph)
    def lower(_engine,request,*,owner=None):
        return {**copy.deepcopy(base),'graph':copy.deepcopy(graph),
            'objective':copy.deepcopy(objective),'recorded_objective_request':copy.deepcopy(request)}
    monkeypatch.setattr(native_recorded_objective,'lower_recorded_objective_request',lower)
    request=copy.deepcopy(recorded)
    derivative=copy.deepcopy(source)
    derivative['nodes'][0]={'kind':'gradient','name':'g0','request':request,'bindings':{},
        'targets':[{'name':'ga','value':'a'}]}
    derivative['nodes'][2]={'kind':'gradient','name':'g1','request':copy.deepcopy(request),
        'bindings':{'a':'a1'},'targets':[{'name':'gb','value':'a1'}]}
    derivative['nodes'][1]['gradient']='ga';derivative['nodes'][3]['gradient']='gb'
    derivative['outer']={'request':copy.deepcopy(request),'bindings':{'a':'a2'}}
    try:
        result=execute_derivative_program(engine,derivative)
        gradient=engine.store.get_native_tensor_artifact(result['gradients']['a']['id']).artifact
        assert gradient.provenance['native_derivative_target']['value_id']==adapter.id
        assert torch.isfinite(gradient.tensors[name]).all() and gradient.tensors[name].norm()>0
        assert result['outputs']['a1'] and result['outputs']['a2']
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_recorded_joint_targets_share_one_graph_backward_and_keep_state_id(tmp_path,family,monkeypatch):
    from natlang_neuralese.serve import native_recorded_objective
    from natlang_neuralese.serve.native_grad import native_recorded_gradient
    from natlang_neuralese.common.native_higher_order import reference_arithmetic
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    state_id=engine.native_write_text('t5 t6')['id'];block=engine.store.get_native_state(state_id).block
    source=program(engine,adapter)
    graph=copy.deepcopy(source['outer']['request']['graph'])
    graph['nodes']=graph['nodes'][1:]
    first=graph['nodes'][0]
    second=copy.deepcopy(first);second['name']='reader-again'
    second['imports']=[{'source':'prior-again','start':0,'stop':len(block.token_ids),'at':2}]
    graph['nodes']=[first,second]
    graph['external_states']={'prior':{'id':state_id,'token_count':len(block.token_ids)},
        'prior-again':{'id':state_id,'token_count':len(block.token_ids)}}
    graph['nodes'][0]['imports']=[{'source':'prior','start':0,'stop':len(block.token_ids),'at':2}]
    # Use the actual graph loss to test one backward across both exact targets.
    query={'kind':'native-query','schema':QUERY_SCHEMA,'graph_sha256':canonical_json_sha256_hex(graph),
        'terms':[{'kind':'nll','reduction':'sum','actions':[
            {'query':0,'token_id':graph['nodes'][0]['token_ids'][1]},
            {'query':3,'token_id':graph['nodes'][0]['token_ids'][4]},
            {'query':5,'token_id':graph['nodes'][1]['token_ids'][1]},
            {'query':8,'token_id':graph['nodes'][1]['token_ids'][4]}]}]}
    state_name=next(iter(state_values(block)))
    request={'schema':'natlang.native-recorded-objective-request/1',
        'turns':[{'authenticated_native_turn':'joint-target-turn'}],
        'events':[{'authenticated_graph_event':'joint-target-event'}],
        'terms':[{'kind':'recorded-objective-term'}],
        'targets':[{'root':'adapter','kind':'adapter','id':adapter.id,'names':[name]},
            {'root':'state','kind':'native-state','id':state_id,'names':[state_name]}],
        'native_state_modes':{state_id:'same-base-adapter-state'}}
    def lower(_engine,recorded,*,owner=None):
        return {**source['outer']['request'],'graph':copy.deepcopy(graph),'objective':copy.deepcopy(query),
            'native_state_modes':copy.deepcopy(recorded['native_state_modes']),
            'recorded_objective_request':copy.deepcopy(recorded)}
    monkeypatch.setattr(native_recorded_objective,'lower_recorded_objective_request',lower)
    count=[];original=engine.shared_model.forward
    def counted(*args,**kwargs):count.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(engine.shared_model,'forward',counted)
    try:
        with reference_arithmetic(engine.shared_model,query_chunk_tokens=2):
            result=native_recorded_gradient(engine,request,None)
        assert len(count)==1
        assert set(result['gradients'])=={'adapter','state'}
        adapter_grad=engine.store.get_native_tensor_artifact(result['gradients']['adapter']['id']).artifact
        state_grad=engine.store.get_native_tensor_artifact(result['gradients']['state']['id']).artifact
        assert adapter_grad.kind=='adapter-gradient' and adapter_grad.source_artifact_id==adapter.id
        assert state_grad.kind=='state-gradient' and state_grad.source_artifact_id==state_id
        assert adapter_grad.tensors[name].norm()>0
        assert all(torch.isfinite(value).all() for value in state_grad.tensors.values())
    finally:engine.store.close()
