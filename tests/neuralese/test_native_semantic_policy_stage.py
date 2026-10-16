"""Actual shared-parent semantic policy updates persist all model/codec state."""
import copy
import json
import os
from pathlib import Path
import pytest
import torch
from test_native_codec_replay import captured
from test_native_semantic_episode import episode
from test_native_codec_graph import case
from natlang_neuralese.train import native_semantic_stage as stage
from natlang_neuralese.train.native_semantic import SemanticParticipant
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from natlang_neuralese.model.semantic_channel import SemanticChannel
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.rl.graph_replay import save_bundle
from natlang_neuralese.serve.store import encode_native_state


def fixture(tmp_path):
    device=os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu')
    original,channel,_,_=case(tmp_path,'lfm','mellum')
    participants={name:SemanticParticipant(value.model,value.tokenizer,value.config,value.identity,
        make_native_optimizer(value.model,lr=1e-4,in_backward=False)[0]) for name,value in original.items()}
    from natlang_neuralese.train.fp8_backend import make_backend
    participants['mellum'].backend=make_backend('triton-experts',participants['mellum'].model.port.hf,device=device)
    participants['mellum'].model.port.activation_checkpoint=participants['mellum'].backend.checkpoint
    config={'schema':stage.CONFIG_SCHEMA,'backbone_lr':1e-4,'vae_lr':1e-3,'latent_width':4,'hidden_width':8,
        'models':{name:{'fp8_training':'triton-experts' if name=='mellum' else 'off'} for name in participants},
        'checkpoint_every':1,'eval_every':100,'phases':[{'id':kind,'kind':kind,'epochs':1} for kind in ('vae','joint','distill')]}
    vae_optimizer=stage._vae_optimizer(channel.vae,config)
    policy=CheckpointPolicy(tmp_path/'parent',prune_best=False)
    export=stage.save_shared(policy,participants,channel.vae,vae_optimizer,
        training={'step':0,'exposure':{},'config_history':[]},device=device)
    for name,value in participants.items():value.identity=export['models'][name]['native_identity']
    channel=SemanticChannel.load(tmp_path/'parent'/'semantic-adapter.json',device=device)
    _,_,_,blocks,turns=captured(tmp_path,'lfm','mellum',loaded=(participants,channel),return_participants=True)
    rollout=episode(turns,participants['lfm'].identity)
    rollouts=tmp_path/'rollouts.jsonl';rollouts.write_text(json.dumps(rollout)+'\n')
    save_bundle([rollout],tmp_path/'bundle',policy_id=export['checkpoint']['sha256'],fetch=lambda key:encode_native_state(blocks[key]))
    def binding(path):return {'path':str(path),'sha256':sha256_file_hex(path)}
    source={'rollouts':binding(rollouts),'block_bundle':binding(tmp_path/'bundle'/'manifest.json'),
            'adapter':binding(tmp_path/'parent'/'semantic-adapter.json')}
    # Full genuine source/consumer paired examples for the existing semantic
    # phases; the policy source remains the original full captured trajectory.
    text='t4';communication={'source_id':'fixture','writer_record_id':'writer','reader_record_id':'reader',
        'write_name':'payload','occurrence':'edge','source_binding':{'manifest_sha256':'a'*64}}
    payload=__import__('hashlib').sha256(text.encode()).hexdigest();communication_sha=canonical_json_sha256_hex(communication)
    row={'schema':'natlang.native-semantic-pair/1','id':'pair','split':'train','source_groups':['train:fixture'],
         'payload_sha256':payload,'communication':communication,'communication_sha256':communication_sha,'models':{}}
    for name,participant in participants.items():
        op,close,co,cc=participant.model.marker_ids
        encoding=participant.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoding['input_ids'];end=2+len(ids)
        graph={'schema':'natlang.neuralese-native-graph/1','id':name+':pair','split':'train',
            'source_groups':row['source_groups'],'tokenizer_fingerprint':participant.identity['tokenizer_fingerprint'],
            'nodes':[{'name':'writer','token_ids':[7,op,*ids,close],'target_weights':[0,1,*([1]*len(ids)),1],
                'producing':[[2,end]]},{'name':'reader','token_ids':[7,co,cc,9,2],
                'target_weights':[0,1,0,1,1],'imports':[{'source':'writer','start':2,'stop':end,'at':2}]}]}
        row['models'][name]={'graph':graph,'export':{'node':'writer','start':2,'stop':end,'text':text,
            'offsets':encoding['offset_mapping']},'consumer_nodes':['reader'],
            'witness':{'source_id':'fixture','payload_sha256':payload,'communication_sha256':communication_sha}}
    held=copy.deepcopy(row);held.update(id='held',split='test',source_groups=['test:fixture'])
    for name,view in held['models'].items():view['graph'].update(id=name+':held',split='test',source_groups=['test:fixture'])
    for role,value in [('pairs',row),('held_pairs',held)]:
        path=tmp_path/(role+'.jsonl');path.write_text(json.dumps(value)+'\n');config[role]=binding(path)
    config['policy_training']={'id':'policy','mode':'rft','epochs':2,'every_semantic_updates':1,
        'updates_per_refresh':1,'sources':[source]}
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    models=[{'model_id':name,'checkpoint':{**export['checkpoint'],'participant':name},'native_identity':value['native_identity']}
            for name,value in export['models'].items()]
    manifest=tmp_path/'participants.json';manifest.write_text(json.dumps({'schema':'natlang.native-shared-participants/1',
        'models':models,'signature':canonical_json_sha256_hex(models)}))
    return path,manifest,config,tmp_path/'parent'/'checkpoint.pt'


def assert_same(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert a==b


@pytest.mark.skipif(os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE')!='cuda',reason='actual production Mellum shared-stage loader requires FP8 CUDA')
def test_real_joint_policy_stage_stop_resume_preserves_both_models_codec_optimizers_and_visits(tmp_path):
    path,manifest,config,parent=fixture(tmp_path)
    torch.manual_seed(1973)
    whole=stage.run(path,manifest,tmp_path/'whole',resume=parent,device='cuda')
    assert whole['completed'] and whole['step']==5
    assert whole['policy_training_progress']['policy_obligations']['complete']
    assert list(whole['exposure']['policy'].values())==[{'episode':2}]
    torch.manual_seed(1973)
    partial=stage.run(path,manifest,tmp_path/'continued',resume=parent,max_updates=2,device='cuda')
    assert not partial['completed']
    resumed=stage.run(path,manifest,tmp_path/'continued',resume=tmp_path/'continued'/'checkpoint.pt',device='cuda')
    assert resumed['completed'] and resumed['step']==whole['step']
    a=CheckpointPolicy.load(tmp_path/'whole'/'checkpoint.pt');b=CheckpointPolicy.load(tmp_path/'continued'/'checkpoint.pt')
    for key in ('vae_state','vae_optimizer','participants'):assert_same(a[key],b[key])
    assert_same(a['training']['rng'],b['training']['rng'])
    assert a['training']['exposure']==b['training']['exposure']
    assert a['training']['phase_backend_execution']==b['training']['phase_backend_execution']
    assert a['training']['phase_backend_execution']['policy']['mellum']['calls']>0
    for name,participant in a['participants'].items():assert participant['global_step']==4
    from natlang_neuralese.train.native_semantic_policy import validate_policy_boundary
    export=json.loads((tmp_path/'whole'/'semantic-adapter.json').read_text())
    validate_policy_boundary(whole,export,config)
    forged=copy.deepcopy(whole);forged['policy_training_progress']['policy_obligations']['complete']=False
    with pytest.raises(ValueError,match='exposure provenance'):validate_policy_boundary(forged,export,config)
    # Completed semantic phases do not block new policy obligations. This is a
    # fresh process/load with no producer() call to turn on checkpointing first.
    config['policy_training']['epochs']=3
    config['policy_training']['clip_high']=.15
    path.write_text(json.dumps(config))
    extended=stage.run(path,manifest,tmp_path/'extended',resume=tmp_path/'whole'/'checkpoint.pt',device='cuda')
    assert extended['completed'] and extended['step']==6
    assert list(extended['exposure']['policy'].values())==[{'episode':3}]
    for phase in ('vae','joint','distill'):assert extended['exposure'][phase]==whole['exposure'][phase]
    assert extended['metrics']['held']['scope']=='full'
