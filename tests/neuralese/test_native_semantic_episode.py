"""Whole actual heterogeneous-tokenizer episodes through live codec training."""
import copy
import pytest
import torch
from test_native_codec_replay import captured
from natlang_neuralese.rl.native_semantic_replay import compile_semantic_episode,semantic_episode_loss
from natlang_neuralese.rl.native_replay import action_logps
from natlang_neuralese.model.native_codec_graph import execute_codec_graph


def episode(turns,identity):
    wire=turns[0]['sampling']['writes'][0]['state_id']
    raw=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],'sampling':turn['sampling'],
          'replay':{'raw_calls_retained':True}} for turn in turns]
    events=[{'node':name+'#turn1','kind':'model_turn','call_id':name} for name in ('writer','reader')]+[
        {'node':'actual-write','kind':'block_write','turn':'writer#turn1','block':wire},
        {'node':'actual-read','kind':'block_read','turn':'reader#turn1','block':wire,
         'inputs':[{'node':'actual-write','block':wire}]}]
    return {'task_id':'actual-train-task','split':'train','sample':0,'policy':identity,
            'reward':{'quality':1},'turns':raw,'execution_graph':events}


@pytest.mark.parametrize('source,target',[('lfm','mellum'),('mellum','lfm')])
def test_actual_cross_model_episode_compilation_preserves_all_calls_and_codec_gradients(tmp_path,source,target):
    torch.manual_seed(718)
    participants,channel,_,blocks,turns=captured(tmp_path,source,target,return_participants=True)
    rollout=episode(turns,participants[source].identity)
    replays,dependencies,external=compile_semantic_episode(rollout,participants,channel=channel,fetch_block=blocks.__getitem__)
    assert set(replays)=={source,target} and len(dependencies)==1
    dependency=dependencies[0]
    assert dependency.source_node=='writer#turn1' and dependency.external_source is None
    assert dependency.source_occurrence=='actual-write' and dependency.target_participant==target
    assert not any(name.startswith('semantic-source-') for name in external)
    results,_=execute_codec_graph(participants,{name:replay.plan for name,replay in replays.items()},dependencies,
        channel=channel,external_blocks=external)
    for name,replay in replays.items():
        logps,behavior=action_logps(participants[name].model,results[name]['h_final'][0],replay.actions)
        torch.testing.assert_close(logps,behavior,rtol=2e-5,atol=3e-6)
    # Only the consumer's reward term: the real source receives its downstream
    # adjoint, rather than merely its own teacher-forced producer-action loss.
    reader=replays[target]
    logps,_=action_logps(participants[target].model,results[target]['h_final'][0],reader.actions)
    values=[participants[source].model.adapter.up.weight,*channel.vae.posterior.parameters()]
    gradients=torch.autograd.grad(-logps.mean(),values)
    assert all(value.norm()>0 for value in gradients)
    loss,detail=semantic_episode_loss(participants,replays,dependencies,channel=channel,external_blocks=external,
                                      advantage=1.,mode='rft')
    loss.backward()
    assert detail['sampled_tokens']==7 and set(detail['sampled_tokens_by_model'])=={source,target}
    assert all(participants[name].model.adapter.up.weight.grad.norm()>0 for name in replays)
    assert any(value.grad is not None and value.grad.norm()>0 for value in channel.vae.parameters())
    missing=copy.deepcopy(rollout);missing['turns']=missing['turns'][1:]
    with pytest.raises(ValueError,match='model-turn inventory'):
        compile_semantic_episode(missing,participants,channel=channel,fetch_block=blocks.__getitem__)


@pytest.mark.parametrize('target',['lfm','mellum'])
def test_explicit_current_codec_training_keeps_external_old_code_and_behavior_identity_immutable(tmp_path,target):
    from types import SimpleNamespace
    from natlang_neuralese.model.semantic_channel import SemanticChannel
    from natlang_neuralese.serve.store import native_state_id
    source='mellum' if target=='lfm' else 'lfm'
    participants,behavior,_,blocks,turns=captured(tmp_path,source,target,return_participants=True)
    rollout=episode(turns,participants[source].identity)
    rollout['turns']=rollout['turns'][1:]
    rollout['execution_graph']=[event for event in rollout['execution_graph']
        if event['node'] not in {'writer#turn1','actual-write'}]
    replays,dependencies,external=compile_semantic_episode(rollout,participants,channel=behavior,fetch_block=blocks.__getitem__)
    assert set(replays)=={target} and dependencies[0].source_node is None
    wire=turns[0]['sampling']['writes'][0]['state_id'];original_id=native_state_id(blocks[wire])
    current_vae=copy.deepcopy(behavior.vae)
    with torch.no_grad():next(current_vae.shared_decoder.parameters()).add_(.01)
    current_bindings={name:{**identity,'checkpoint_sha256':'d'*64} for name,identity in behavior.bindings.items()}
    current=SemanticChannel(current_vae,current_bindings,behavior.tokenizers,adapter_sha256='c'*64)
    actual={name:SimpleNamespace(model=value.model,tokenizer=value.tokenizer,identity=current_bindings[name])
            for name,value in participants.items()}
    execution={'schema':'natlang.native-semantic-policy-execution/1','parent_checkpoint_sha256':'d'*64,
        'parent_adapter_sha256':'c'*64,'parent_step':0,'step':2,'behavior_adapter_sha256':behavior.adapter_sha256}
    loss,details=semantic_episode_loss(actual,replays,dependencies,channel=current,external_blocks=external,
        advantage=1.,mode='rft',training_execution=execution,behavior_channel=behavior)
    gradients=torch.autograd.grad(loss,[*current_vae.posterior.parameters(),*current_vae.shared_decoder.parameters()],allow_unused=True)
    posterior=list(current_vae.posterior.parameters())
    assert all(value is None or torch.count_nonzero(value)==0 for value in gradients[:len(posterior)])
    assert any(value is not None and value.norm()>0 for value in gradients[len(posterior):])
    assert native_state_id(blocks[wire])==original_id
    assert blocks[wire].adapter_sha256==behavior.adapter_sha256
    assert all(item['training_execution']==execution for item in details['execution'] if item['kind']=='semantic-codec')
    with pytest.raises(ValueError,match='exactly bound'):
        current.adapt(blocks[wire],target_name=target,target_model=actual[target].model,
            target_tokenizer=actual[target].tokenizer,target_identity=actual[target].identity,allow_nearest=True)
