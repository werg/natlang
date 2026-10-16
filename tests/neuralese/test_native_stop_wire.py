"""Per-use constants preserve native forward values and exact capture provenance."""
import copy
import pytest
import torch
from test_native_rl_replay import actor_rollout
from test_native_serving_adapters import engine_fixture, nonzero_adapter
from natlang_neuralese.rl.native_replay import compile_replay, replay_logps
from natlang_neuralese.rl.episode_graph import validate_import_gradient_uses
from natlang_neuralese.model.native_actor_graph import run_invocation_actors
from natlang_neuralese.model.shared_state import Invocation, compile_graph
from natlang_neuralese.serve.chat import render_messages, RequestError


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_stopped_capture_replay_retains_forward_and_blocks_only_source_gradient(family):
    model,roles,identity,block_id,block,turns=actor_rollout(family)
    def execute(stopped):
        changed=copy.deepcopy(turns)
        if stopped:changed[0]['sampling']['prompt'][1]['stop_gradient']=True
        replay=compile_replay(changed,identity=identity,roles=roles,fetch_block=lambda _:block)
        values,_=replay_logps(model,replay)
        selected=[i for i,action in enumerate(replay.actions) if action['turn']=='consumer']
        grad=torch.autograd.grad(-values[selected].mean(),model.adapter.up.weight)[0]
        return values.detach(),grad,replay
    live,live_grad,_=execute(False)
    stopped,stopped_grad,replay=execute(True)
    assert torch.equal(live,stopped)
    assert live_grad.norm()>0
    assert torch.count_nonzero(stopped_grad)==0
    assert replay.plan.invocations[1].imports[0].stop_gradient


def test_stop_capture_must_agree_with_actual_event_and_be_true_or_absent():
    event={'kind':'block_read','turn':'reader','block':'state','stop_gradient':True}
    capture={'prompt':[{'kind':'native_state','id':'state','stop_gradient':True}]}
    validate_import_gradient_uses('reader',capture,{'read':event})
    bad=copy.deepcopy(capture);bad['prompt'][0].pop('stop_gradient')
    with pytest.raises(ValueError,match='gradient use'):
        validate_import_gradient_uses('reader',bad,{'read':event})
    bad['prompt'][0]['stop_gradient']=False
    with pytest.raises(ValueError,match='true or absent'):
        validate_import_gradient_uses('reader',bad,{'read':event})


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('constant_first',[True,False])
def test_same_numeric_adapter_live_and_constant_uses_never_share_derivative_cache(tmp_path,family,constant_first):
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    model=engine.shared_model
    names=('constant','live') if constant_first else ('live','constant')
    plan=compile_graph([Invocation(key,(4,5,6)) for key in names])
    coefficient=adapter.artifact.tensors[name].clone().requires_grad_(True)
    bindings={key:[{'id':adapter.id,**({'stop_gradient':True} if key=='constant' else {})}] for key in names}
    def execute(bank,which):
        result=model(plan)
        hidden=result['h_final']
        offset={key:plan.offsets[i] for i,key in enumerate(names)}
        assert torch.equal(hidden[:,offset['constant']:offset['constant']+3],hidden[:,offset['live']:offset['live']+3])
        loss=hidden[:,offset[which]:offset[which]+3].float().square().mean()
        grad=torch.autograd.grad(loss,coefficient,allow_unused=True)[0]
        if which=='constant':assert grad is None or torch.count_nonzero(grad)==0
        else:assert grad.norm()>0
        assert bank.receipt()['invocations']['live']['native_identity']==bank.receipt()['invocations']['constant']['native_identity']
    for which in ('constant','live'):
        run_invocation_actors(engine,names,bindings,lambda bank:execute(bank,which),leaves={adapter.id:{name:coefficient}},gradient_precision='fp32')


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_engine_generation_and_contextual_capture_retain_constant_import(tmp_path,monkeypatch,family):
    from natlang_neuralese.serve.engine import GenerationRequest
    engine=engine_fixture(tmp_path,family)
    block=engine.native_write_text('t4 t5')
    messages=[{'role':'user','content':[{'type':'text','text':'t6 '},
        {'type':'native_state','id':block['id'],'stop_gradient':True}]}]
    monkeypatch.setattr(engine,'_native_token',lambda seq:(2,float(seq.logits[0].float().log_softmax(-1)[2])))
    result=engine.generate(GenerationRequest(messages=messages,temperature=1.,seed=32,max_tokens=2,rl_capture=True))
    imported=[part for part in result['x_natlang_rl_capture']['prompt'] if part['kind']=='native_state']
    assert len(imported)==1 and imported[0]['stop_gradient'] is True
    assert result['native_state']['imports'][0]['stop_gradient'] is True
    rendered=engine.render_request(messages,None)
    assert rendered.block_stop_gradients==[True]
    invalid=copy.deepcopy(messages);invalid[0]['content'][1]['stop_gradient']=False
    with pytest.raises(RequestError,match='bad-part'):
        engine.render_request(invalid,None)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_cached_splice_constant_blocks_kv_and_hybrid_adjoints_without_mutating_archive(family):
    from dataclasses import replace
    from natlang_neuralese.model.shared_state import native_step, splice_block
    model,roles,identity,block_id,block,_=actor_rollout(family)
    layers=[];leaves=[]
    for payload in block.layers:
        copied={key:(value.detach().clone().requires_grad_(True) if isinstance(value,torch.Tensor) and value.is_floating_point() else value)
                for key,value in payload.items()}
        layers.append(copied);leaves.extend(value for value in copied.values() if isinstance(value,torch.Tensor) and value.requires_grad)
    observed=replace(block,layers=tuple(layers))
    outputs=[]
    for stopped in (True,False):
        prefix=native_step(model,torch.tensor([[3,60]]))
        cache=splice_block(model,prefix['cache'],observed,**{key:value for key,value in identity.items() if key!='architecture'},stop_gradient=stopped)
        hidden=native_step(model,torch.tensor([[61,11]]),cache)['h_final']
        outputs.append(hidden.detach())
        gradients=torch.autograd.grad(hidden.float().square().mean(),leaves,allow_unused=True)
        magnitude=sum(float(value.norm()) for value in gradients if value is not None)
        assert magnitude==0 if stopped else magnitude>0
    assert torch.equal(*outputs)
    assert all(value.requires_grad for value in leaves)
