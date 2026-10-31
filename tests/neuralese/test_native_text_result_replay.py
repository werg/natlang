import copy
import hashlib

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.rl.episode_graph import whole_episode_turns
from natlang_neuralese.rl.native_replay import compile_replay, replay_logps, capture_invocation_actors, prepare_objective, NativePolicyObjective
from natlang_neuralese.model.native_adapters import run_native
from natlang_neuralese.model.native_actor_graph import run_invocation_actors


def test_text_result_writer_replay_preserves_policy_and_consumer_gradient(tmp_path, monkeypatch):
    engine=engine_fixture(tmp_path,'lfm')
    model=engine.shared_model
    with torch.no_grad(): model.adapter.up.weight.normal_(std=.01)
    model.adapter.up.weight.requires_grad_(True)
    roles=engine.role_tokens
    selected=iter([7,2])
    def choose(seq):
        token=next(selected)
        return token,float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[token])
    monkeypatch.setattr(engine,'_native_token',choose)
    writer=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4 t5'}],temperature=.8,seed=17,max_tokens=8,rl_capture=True))
    result=engine.native_write_text('t5 t6',value_type='Neuralese<string>')
    block=result['id']
    selected=iter([7,2])
    reader=engine.generate(GenerationRequest(messages=[{'role':'user','content':[{'type':'native_state','id':block,'source_occurrence':'w'}, {'type':'text','text':' t8'}]}],temperature=.8,seed=18,max_tokens=8,rl_capture=True))
    wc=writer['x_natlang_rl_capture']; rc=reader['x_natlang_rl_capture']
    rollout={'turns':[{'invocation_id':'producer','messages':[], 'target':'t5 t6','sampling':wc},
                      {'invocation_id':'reader','messages':[], 'target':'x','sampling':rc}],
             'execution_graph':[{'node':'producer#turn1','kind':'model_turn','call_id':'producer'},
               {'node':'reader#turn1','kind':'model_turn','call_id':'reader'},
               {'node':'w','kind':'block_write','turn':'producer#turn1','block':block,'producer':None,
                'source_kind':'typed-text-result','format':'native-state','source_capture':result['source_capture'],
                'native_identity':engine.native_identity,
                'text_body_sha256':hashlib.sha256(b't5 t6').hexdigest(),
                'inputs':[{'node':'producer#turn1','port':'result-source'}]},
               {'node':'r','kind':'block_read','turn':'reader#turn1','block':block,
                'inputs':[{'node':'w','block':block}]}]}
    turns=whole_episode_turns(rollout)
    fetch=lambda key:engine.store.get_native_state(key).block
    replay=compile_replay(turns,identity=engine.native_identity,roles=roles,fetch_block=fetch)
    assert [n.name for n in replay.plan.invocations]==['producer#turn1','w','reader#turn1']
    from natlang_neuralese.rl.privileged_scoring import graph_invocations
    assert [item['name'] for item in graph_invocations(replay,turns,root_calls={'producer'})] == ['producer#turn1','reader#turn1']
    assert len(replay.actions)==len(wc['samples'])+len(rc['samples'])
    actors=capture_invocation_actors(turns,engine.native_identity)
    assert actors['invocations']['w']['native_identity']==engine.native_identity
    def execute():
        values,behavior=replay_logps(model,replay)
        torch.testing.assert_close(values.detach(),behavior,rtol=2e-5,atol=2e-6)
        ids=torch.tensor([[token for n in replay.plan.invocations for token in n.token_ids]],device=next(model.parameters()).device)
        embeddings=model.embed(ids).detach().requires_grad_(True)
        out=model(replay.plan,embeddings=embeddings)
        reader_offset=replay.plan.offsets[2]
        grad=torch.autograd.grad(out['h_final'][:,reader_offset:].float().square().mean(),embeddings)[0]
        assert grad[:,replay.plan.offsets[1]:reader_offset].norm()>0
        assert grad[:,:replay.plan.offsets[1]].norm()==0
    run_native(engine,execute)
    actor_replay=compile_replay(turns,identity=engine.native_identity,roles=roles,fetch_block=fetch,invocation_actors=actors)
    def execute_actors(bank):
        values,behavior=replay_logps(model,actor_replay)
        torch.testing.assert_close(values.detach(),behavior,rtol=2e-5,atol=2e-6)
    run_invocation_actors(engine,list(actors['invocations']),{},execute_actors)
    from natlang_neuralese.serve.native_recorded_graph import compile_recorded_graph
    recorded=[]
    for index,turn in enumerate(turns):
        graph={key:copy.deepcopy(turn[key]) for key in ('graph_turn','written_nodes','read_bindings','written_occurrences','read_occurrences','text_result_writes') if key in turn}
        recorded.append({'native':{'schema':'natlang.native-recorded-turn/1','id':f'receipt-{index}',
            'retry':0,'graph':graph,'kind':'generation','sampling':turn['sampling']}})
    recorded_graph=compile_recorded_graph(recorded,rollout['execution_graph'],identity=engine.native_identity,
                                           roles=roles,fetch_block=fetch)
    assert [node.name for node in recorded_graph.replay.plan.invocations]==['producer#turn1','w','reader#turn1']
    assert recorded_graph.origin_bindings['receipt-0']['text_result_writes']==turns[0]['text_result_writes']
    import json
    from natlang_neuralese.data.native_graph import parse_graph
    episodes=[]
    for sample,reward in enumerate((1,0)):
        row=copy.deepcopy(rollout)
        row.update(task_id='task',sample=sample,split='train',group='group',reward={'quality':reward},
                   policy={'checkpoint_sha256':engine.native_identity['checkpoint_sha256']})
        for turn in row['turns']: turn['replay']={'raw_calls_retained':True}
        episodes.append(row)
    graphs,spec=prepare_objective(episodes,identity=engine.native_identity,roles=roles,fetch_block=fetch)
    assert len(graphs)==2 and len(graphs[0]['nodes'])==3
    assert next(iter(spec['rows'].values()))['text_result_writes'][0]['source_turn']=='producer#turn1'
    path=tmp_path/'objective.json'; path.write_text(json.dumps(spec))
    objective=NativePolicyObjective(path)
    example=parse_graph(graphs[0],tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint'])
    context={'examples':[example], 'row_offsets':{example.id:{'physical_start':0,'external_names':{}}}}
    def train():
        result=model(example.plan,retain_states=False)
        loss,detail=objective(model,result,context)
        loss.backward()
        assert detail['sampled_tokens']==len(wc['samples'])+len(rc['samples'])
        assert model.adapter.up.weight.grad is not None and model.adapter.up.weight.grad.norm()>0
    run_native(engine,train)
    forged=copy.deepcopy(turns); forged[0]['text_result_writes'][0]['text_body_sha256']='0'*64
    with pytest.raises(ValueError,match='text differs'):
        compile_replay(forged,identity=engine.native_identity,roles=roles,fetch_block=fetch)


def test_adapted_text_writer_keeps_its_exact_actor_receipt(tmp_path,monkeypatch):
    from test_native_invocation_actors import adapters
    engine=engine_fixture(tmp_path,'lfm',service_template=True)
    adapter,_=adapters(engine)
    entries=[adapter.id]
    chosen=iter([7,2])
    def choose(seq):
        token=next(chosen)
        return token,float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[token])
    monkeypatch.setattr(engine,'_native_token',choose)
    producer=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4'}],adapters=entries,
        temperature=.8,seed=3,max_tokens=8,rl_capture=True))
    written=engine.native_write_text('t5 t6',adapters=entries,value_type='Neuralese<string>')
    ident=written['id']; block=engine.store.get_native_state(ident).block
    assert written['source_capture']['adapter_binding']==block.adapter_binding and block.adapter_binding is not None
    chosen=iter([7,2])
    reader=engine.generate(GenerationRequest(messages=[{'role':'user','content':[{'type':'native_state','id':ident,'source_occurrence':'text-write'}, {'type':'text','text':'t8'}]}],
        adapters=entries,temperature=.8,seed=4,max_tokens=8,rl_capture=True))
    rollout={'turns':[{'invocation_id':'p','messages':[],'target':'t5 t6','sampling':producer['x_natlang_rl_capture']},
                       {'invocation_id':'r','messages':[],'target':'x','sampling':reader['x_natlang_rl_capture']}],
        'execution_graph':[{'node':'p#turn1','kind':'model_turn','call_id':'p'},{'node':'r#turn1','kind':'model_turn','call_id':'r'},
            {'node':'text-write','kind':'block_write','turn':'p#turn1','block':ident,'producer':None,'source_kind':'typed-text-result',
             'format':'native-state','native_identity':written['source_capture']['policy'],'source_capture':written['source_capture'],
             'text_body_sha256':hashlib.sha256(b't5 t6').hexdigest(),'inputs':[{'node':'p#turn1','port':'result-source'}]},
            {'node':'read','kind':'block_read','turn':'r#turn1','block':ident,'inputs':[{'node':'text-write','block':ident}]}]}
    turns=whole_episode_turns(rollout)
    actors=capture_invocation_actors(turns,engine.native_identity)
    fetch=lambda key:engine.store.get_native_state(key).block
    replay=compile_replay(turns,identity=engine.native_identity,roles=engine.role_tokens,fetch_block=fetch,invocation_actors=actors)
    def execute(bank):
        logps,behavior=replay_logps(engine.shared_model,replay)
        torch.testing.assert_close(logps.detach(),behavior,rtol=2e-5,atol=3e-6)
    run_invocation_actors(engine,list(actors['invocations']),{name:entries for name in actors['invocations']},execute)
    forged=copy.deepcopy(turns); forged[0]['text_result_writes'][0]['source_capture'].pop('adapter_binding')
    with pytest.raises(ValueError,match='actual adapter binding'):
        capture_invocation_actors(forged,engine.native_identity)
