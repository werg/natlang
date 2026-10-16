"""Equal immutable state IDs retain distinct actual source/use occurrences."""
import copy
import os
import pytest
import torch
from test_native_rl_replay import actor_rollout
from natlang_neuralese.rl.native_occurrences import graph_binding
from natlang_neuralese.rl.native_replay import compile_replay


def repeated_graph(family):
    model,roles,identity,block_id,block,original=actor_rollout(family)
    first=copy.deepcopy(original[1]);first['graph_turn']='first';first['written_nodes']={block_id:'first-write'}
    second=copy.deepcopy(first);second['graph_turn']='second';second['written_nodes']={block_id:'second-write'}
    reader=copy.deepcopy(original[0]);reader['graph_turn']='reader'
    one=reader['sampling']['prompt'][1]
    one.update(source_occurrence='first-write',stop_gradient=True)
    reader['sampling']['prompt']=[reader['sampling']['prompt'][0],one,
        {'kind':'tokens','token_ids':[61,60],'producing':[False,False]},
        {'kind':'native_state','id':block_id,'native_identity':identity,'source_occurrence':'second-write'},
        {'kind':'tokens','token_ids':[61],'producing':[False]}]
    nodes={name:{'node':name,'kind':'model_turn'} for name in ('first','second','reader')}
    for owner in ('first','second'):
        nodes[owner+'-write']={'node':owner+'-write','kind':'block_write','turn':owner,'block':block_id}
    for index,owner in enumerate(('first','second')):
        nodes[f'read-{index}']={'node':f'read-{index}','kind':'block_read','turn':'reader','block':block_id,
            **({'stop_gradient':True} if index==0 else {}),'inputs':[{'node':owner+'-write','block':block_id}]}
    reader.update(graph_binding('reader',reader['sampling'],nodes))
    return model,roles,identity,block_id,block,[first,second,reader],nodes


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_equal_state_ids_two_origins_live_and_constant_are_distinct_gradient_paths(family):
    model,roles,identity,block_id,block,turns,nodes=repeated_graph(family)
    device=os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu')
    model.port.hf.to(device);model.to(device)
    replay=compile_replay(turns,identity=identity,roles=roles,fetch_block=lambda _:block)
    reader=next(node for node in replay.plan.invocations if node.name=='reader')
    assert [(edge.source,edge.stop_gradient) for edge in reader.imports]==[('first',True),('second',False)]
    ids=torch.tensor([[token for node in replay.plan.invocations for token in node.token_ids]],device=device)
    raw=model.embed(ids).detach().requires_grad_(True)
    result=model(replay.plan,embeddings=raw)
    offset=replay.plan.offsets[[node.name for node in replay.plan.invocations].index('reader')]
    gradient=torch.autograd.grad(result['h_final'][:,offset:].float().square().mean(),raw)[0]
    first,second=replay.plan.offsets[:2]
    assert torch.count_nonzero(gradient[:,first:first+len(replay.plan.invocations[0].token_ids)])==0
    assert gradient[:,second:second+len(replay.plan.invocations[1].token_ids)].norm()>0
    assert len(turns[-1]['read_occurrences'])==2
    forged=copy.deepcopy(turns[-1]['sampling']);forged['prompt'][1]['source_occurrence']='second-write'
    with pytest.raises(ValueError,match='occurrence-matched'):
        graph_binding('reader',forged,nodes)
    ambiguous=copy.deepcopy(turns[-1]['sampling'])
    for segment in ambiguous['prompt']:
        if segment['kind']=='native_state':segment.pop('source_occurrence',None);segment.pop('stop_gradient',None)
    unflagged=copy.deepcopy(nodes);unflagged['read-0'].pop('stop_gradient')
    with pytest.raises(ValueError,match='explicit source-occurrence'):
        graph_binding('reader',ambiguous,unflagged)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_engine_duplicate_import_capture_scores_and_live_source_gradient(tmp_path,monkeypatch,family):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.serve.engine import GenerationRequest
    from natlang_neuralese.rl.native_replay import replay_logps
    from natlang_neuralese.model.native_adapters import run_native
    engine=engine_fixture(tmp_path,family)
    model=engine.shared_model
    with torch.no_grad():model.adapter.up.weight.normal_(std=.01)
    model.adapter.up.weight.requires_grad_(True)
    roles=engine.role_tokens
    selected=iter([roles.produce_open,5,6,roles.produce_close,2])
    def choose(seq):
        token=next(selected)
        return token,float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[token])
    monkeypatch.setattr(engine,'_native_token',choose)
    writer=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4 t5'}],temperature=.8,seed=17,max_tokens=8,rl_capture=True))
    block_id=writer['native_state']['blocks'][0]['id']
    selected=iter([7,2])
    reader=engine.generate(GenerationRequest(messages=[{'role':'user','content':[
        {'type':'native_state','id':block_id,'source_occurrence':'source','stop_gradient':True},
        {'type':'text','text':' t8 '}, {'type':'native_state','id':block_id,'source_occurrence':'source'}]}],
        temperature=.8,seed=18,max_tokens=8,rl_capture=True))
    captured=reader['x_natlang_rl_capture']
    segments=[segment for segment in captured['prompt'] if segment['kind']=='native_state']
    assert len(segments)==2 and segments[0]['source_occurrence']==segments[1]['source_occurrence']=='source'
    nodes={'source':{'node':'source','kind':'block_write','turn':'writer','block':block_id},
        'r0':{'node':'r0','kind':'block_read','turn':'reader','block':block_id,'stop_gradient':True,'inputs':[{'node':'source','block':block_id}]},
        'r1':{'node':'r1','kind':'block_read','turn':'reader','block':block_id,'inputs':[{'node':'source','block':block_id}]}}
    turns=[{'sampling':writer['x_natlang_rl_capture'],**graph_binding('writer',writer['x_natlang_rl_capture'],nodes)},
        {'sampling':captured,**graph_binding('reader',captured,nodes)}]
    replay=compile_replay(turns,identity=engine.native_identity,roles=roles,fetch_block=lambda key:engine.store.get_native_state(key).block)
    def execute():
        values,behavior=replay_logps(model,replay)
        torch.testing.assert_close(values.detach(),behavior,rtol=2e-5,atol=2e-6)
        index=[i for i,action in enumerate(replay.actions) if action['turn']=='reader']
        gradient=torch.autograd.grad(-values[index].mean(),model.adapter.up.weight)[0]
        assert gradient.norm()>0
    run_native(engine,execute)
