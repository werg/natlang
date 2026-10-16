"""Semantic capture compilation retains actual producer and codec gradients."""
import copy
import pytest
import torch
from test_native_codec_graph import case
from natlang_neuralese.model.shared_state import native_step,concatenate_captures,splice_block,RoleTokens
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.serve.store import native_state_id
from natlang_neuralese.rl.native_replay import compile_replay,replay_logps


def captured(tmp_path,family,target=None,return_participants=False,loaded=None):
    target=target or family
    if loaded is None:participants,channel,_,_=case(tmp_path,family,target)
    else:participants,channel=loaded
    participant=participants[family];model=participant.model;identity=participant.identity
    target_participant=participants[target];target_model=target_participant.model
    roles=RoleTokens(*model.marker_ids);device=model.input_rows.device
    def step(ids,cache=None,producing=False,active=model):
        return native_step(active,torch.tensor([ids],device=device),cache,producing=producing)
    def capture(prompt,active_identity=identity):
        return {'schema':'natlang.native-rl-sampling/1','policy':active_identity,'prompt':prompt,
            'samples':[],'writes':[],'sample_range_convention':'half-open','finish_reason':'stop',
            'sampling_law':{'temperature':.8,'seed':3,'guidance':False,'probabilities':'post-processing-normalized'}}
    def sample(out,record,tokens,active=model):
        chunks=[]
        for index,(token,producing) in enumerate(tokens):
            logp=active.logits(out['h_final'][:,-1]).float().div(.8).log_softmax(-1)[0,token]
            record['samples'].append({'index':index,'token_id':token,'logp':float(logp),
                'input_role':'produce' if producing else 'ordinary','terminal':token==2})
            if token!=2:
                out=step([token],out['cache'],producing,active)
                if producing:chunks.append(out)
        return out,chunks
    with torch.no_grad():
        writer=capture([{'kind':'tokens','token_ids':[7],'producing':[False]}])
        _,chunks=sample(step([7]),writer,[(roles.produce_open,False),(4,True),(5,True),(roles.produce_close,False),(2,False)])
        raw=NativeStateBlock(text=participant.tokenizer.decode([4,5],skip_special_tokens=False),
                             **identity,**concatenate_captures(chunks)).validate()
        raw_id=native_state_id(raw);message=channel.message(raw);wire_id=native_state_id(message)
        writer['writes']=[{'state_id':wire_id,'raw_state_id':raw_id,'begin_sample':0,'end_sample':4,
            'payload_begin':1,'payload_end':3,'native_identity':identity}]
        decoded,receipt=channel.adapt(message,target_name=target,target_model=target_model,
            target_tokenizer=target_participant.tokenizer,target_identity=target_participant.identity,allow_nearest=True,source_state_id=wire_id)
        observed_id=native_state_id(decoded)
        target_roles=RoleTokens(*target_model.marker_ids)
        reader=capture([{'kind':'tokens','token_ids':[7,target_roles.consume_open],'producing':[False,False]},
            {'kind':'native_state','id':observed_id,'source_state_id':wire_id,'native_identity':target_participant.identity,
             'semantic_adaptation':receipt,'source_occurrence':'actual-write'},
            {'kind':'tokens','token_ids':[target_roles.consume_close],'producing':[False]}],target_participant.identity)
        prefix=step([7,target_roles.consume_open],active=target_model);cache=splice_block(target_model,prefix['cache'],decoded,
            **{key:value for key,value in target_participant.identity.items() if key!='architecture'})
        sample(step([target_roles.consume_close],cache,active=target_model),reader,[(9,False),(2,False)],target_model)
    turns=[{'graph_turn':'writer','sampling':writer,'written_nodes':{wire_id:'actual-write'}},
        {'graph_turn':'reader','sampling':reader,'read_bindings':{wire_id:'actual-write'},'read_occurrences':[
            {'index':0,'id':observed_id,'source_state_id':wire_id,'node':'actual-read','source':'actual-write'}]}]
    blocks={raw_id:raw,wire_id:message,observed_id:decoded}
    return participants if return_participants else participant,channel,roles,blocks,turns


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_semantic_sample_capture_compiles_to_live_codec_not_external_observation(tmp_path,family):
    participant,channel,roles,blocks,turns=captured(tmp_path,family)
    replay=compile_replay(turns,identity=participant.identity,roles=roles,fetch_block=blocks.__getitem__,semantic_channel=channel)
    assert set(replay.external_blocks)=={'semantic-input:reader:0'}
    assert len(replay.semantic_dependencies)==1
    dependency=replay.semantic_dependencies[0]
    assert dependency.source_node=='writer' and dependency.target_participant=='policy'
    current,behavior=replay_logps(participant.model,replay,semantic_channel=channel)
    torch.testing.assert_close(current,behavior,rtol=2e-5,atol=3e-6)
    indices=[index for index,action in enumerate(replay.actions) if action['turn']=='reader']
    loss=-current[indices].mean()
    values=[participant.model.adapter.up.weight,*channel.vae.posterior.parameters(),*channel.vae.shared_decoder.parameters()]
    gradients=torch.autograd.grad(loss,values,allow_unused=True)
    assert gradients[0].norm()>0
    assert sum(float(value.norm()) for value in gradients[1:] if value is not None)>0
    with pytest.raises(ValueError,match='exact differentiable semantic channel'):
        compile_replay(turns,identity=participant.identity,roles=roles,fetch_block=blocks.__getitem__)


@pytest.mark.parametrize('field,value',[('adapter_sha256','f'*64),('posterior','sample'),('source_model','invented')])
def test_semantic_replay_rejects_changed_actual_codec_receipt(tmp_path,field,value):
    participant,channel,roles,blocks,turns=captured(tmp_path,'lfm')
    forged=copy.deepcopy(turns);forged[1]['sampling']['prompt'][1]['semantic_adaptation'][field]=value
    with pytest.raises(ValueError,match='actual decode receipt'):
        compile_replay(forged,identity=participant.identity,roles=roles,fetch_block=blocks.__getitem__,semantic_channel=channel)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_semantic_episode_admission_archive_and_shared_trainer_objective(tmp_path,family):
    import json
    from natlang_neuralese.rl.native_replay import prepare_objective,NativePolicyObjective
    from natlang_neuralese.rl.graph_replay import save_bundle
    from natlang_neuralese.serve.store import encode_native_state
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    participant,channel,roles,blocks,turns=captured(tmp_path,family)
    raw=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],'sampling':turn['sampling'],
          'replay':{'raw_calls_retained':True}} for turn in turns]
    wire=turns[0]['sampling']['writes'][0]['state_id']
    events=[{'node':f'{name}#turn1','kind':'model_turn','call_id':name} for name in ('writer','reader')]+[
        {'node':'actual-write','kind':'block_write','turn':'writer#turn1','block':wire},
        {'node':'actual-read','kind':'block_read','turn':'reader#turn1','block':wire,
         'inputs':[{'node':'actual-write','block':wire}]}]
    episodes=[{'task_id':'train-task','sample':sample,'split':'train','group':'g',
        'policy':{'checkpoint_sha256':participant.identity['checkpoint_sha256']},'reward':{'quality':reward},
        'turns':copy.deepcopy(raw),'execution_graph':events} for sample,reward in enumerate((1,0))]
    save_bundle(episodes,tmp_path/'bundle',policy_id=participant.identity['checkpoint_sha256'],
        fetch=lambda key:encode_native_state(blocks[key]))
    graphs,spec=prepare_objective(episodes,identity=participant.identity,roles=roles,fetch_block=blocks.__getitem__,
        semantic_channel=channel,block_bundle=tmp_path/'bundle'/'manifest.json')
    assert len(graphs)==2 and all(len(row['semantic_dependencies'])==1 for row in spec['rows'].values())
    path=tmp_path/'objective.json';path.write_text(json.dumps(spec))
    objective=NativePolicyObjective(path);objective.bind_semantic_channel(channel)
    examples=[parse_graph(row,tokenizer_fingerprint=participant.identity['tokenizer_fingerprint']) for row in graphs]
    # One packed batch contains both full episodes. All actual producers run and
    # all consumers reach their own source spans, even when wire IDs are equal.
    loss,details=packed_graph_loss(participant.model,examples,checkpoint_layers=True,objective=objective)
    assert details['episodes']==2 and details['sampled_tokens']==14
    assert sum(item['kind']=='semantic-codec' for item in details['codec_execution'])==2
    assert len([name for item in details['codec_execution'] if item['kind']=='native-graph-segment'
                for name in item['invocations']])==4
    loss.backward()
    assert all(value.grad is None or torch.isfinite(value.grad).all() for value in channel.vae.parameters())
    assert participant.model.adapter.up.weight.grad is not None
