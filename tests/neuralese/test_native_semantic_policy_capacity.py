"""Qualify real whole policy episodes, not an unrelated paired SFT loss."""
import copy
import json
import pytest
import torch
from test_native_codec_graph import case
from test_native_codec_replay import captured
from test_native_semantic_episode import episode
from natlang_neuralese.train.native_semantic import SemanticParticipant
from natlang_neuralese.train.native_semantic_stage import save_shared
from natlang_neuralese.train.native_semantic_policy import PolicySource
from natlang_neuralese.train.native_semantic_policy_capacity import PolicyQualification
from natlang_neuralese.train.native_semantic_capacity import ensure_phase_qualification
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.model.semantic_channel import SemanticChannel
from natlang_neuralese.rl.graph_replay import save_bundle
from natlang_neuralese.serve.store import encode_native_state
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.model.native_weight_carrier import active_cotangent_scope


def fixture(tmp_path,mode):
    original,channel,_,_=case(tmp_path,'lfm','mellum')
    participants={name:SemanticParticipant(value.model,value.tokenizer,value.config,value.identity,
        make_native_optimizer(value.model,lr=1e-4,in_backward=False)[0]) for name,value in original.items()}
    for value in participants.values():
        # Production load_participant binds these from the inherited native
        # checkpoint. This direct numerical fixture constructs the optimizer
        # itself, so record its exact known settings rather than relying on a
        # generic semantic-stage default.
        value.optimizer_options={'lr':1e-4,'weight_decay':.1,'latent_ns':'quintic',
            'learned_row_lr':None,'interface_row_lr':None}
        value.optimizer_state_storage='device'
        value.fp8_profile={'kind':'off','recipe':'delayed','amax_history_len':16}
    optimizer=torch.optim.AdamW(channel.vae.parameters(),lr=1e-3)
    parent=tmp_path/'parent'
    export=save_shared(CheckpointPolicy(parent,prune_best=False),participants,channel.vae,optimizer,
        training={'step':0,'exposure':{},'config_history':[]},device='cpu')
    for name,value in participants.items():
        value.identity=export['models'][name]['native_identity'];value.model.port.checkpoint_layers=True
    behavior=SemanticChannel.load(parent/'semantic-adapter.json',device='cpu')
    _,_,_,blocks,turns=captured(tmp_path,'lfm','mellum',loaded=(participants,behavior),return_participants=True)
    rollout=episode(turns,participants['lfm'].identity)
    episodes=[rollout]
    if mode=='grpo':
        rejected=copy.deepcopy(rollout);rejected.update(sample=1,reward={'quality':0});episodes.append(rejected)
    path=tmp_path/'episodes.jsonl';path.write_text(''.join(json.dumps(row)+'\n'for row in episodes))
    save_bundle(episodes,tmp_path/'bundle',policy_id=export['checkpoint']['sha256'],
        fetch=lambda key:encode_native_state(blocks[key]))
    def binding(path):return {'path':str(path),'sha256':sha256_file_hex(path)}
    declaration={'id':'policy','mode':mode,'epochs':1,'every_semantic_updates':1,'updates_per_refresh':1,
        'sources':[{'rollouts':binding(path),'block_bundle':binding(tmp_path/'bundle'/'manifest.json'),
                    'adapter':binding(parent/'semantic-adapter.json')}]}
    source=PolicySource(declaration);training={'step':7,'exposure':{}}
    source.synchronize(training)
    origin={'checkpoint_sha256':export['checkpoint']['sha256'],
        'adapter_sha256':sha256_file_hex(parent/'semantic-adapter.json'),'step':0,
        'models':{name:{'native_identity':entry['native_identity']}for name,entry in export['models'].items()}}
    return participants,channel.vae,optimizer,source,training,origin


@pytest.mark.parametrize('mode',['rft','grpo'])
def test_actual_policy_qualifies_all_model_vae_derivatives_and_whole_graph_without_earning_visits(tmp_path,mode):
    participants,vae,optimizer,source,training,origin=fixture(tmp_path,mode)
    objective=PolicyQualification(source,participants,vae,training,origin,'cpu')
    parameters=[p for entry in participants.values() for p in entry.model.parameters()]+list(vae.parameters())
    weights=[p.detach().clone()for p in parameters];before=copy.deepcopy(training)
    rng=torch.get_rng_state().clone()
    args=dict(phase='policy',options={},optimizer=optimizer,fetch_external=None,device='cpu',
        training=training,out=tmp_path,cache={},objective=objective)
    proof=ensure_phase_qualification(vae,participants,objective,**args)
    assert proof['passed'] and proof['profile']['objective']=='whole-policy-episode'
    assert proof['source_rows']==len(source.entries)
    assert len(proof['complete_gradients'])==(2 if mode=='grpo' else 1)
    assert proof['complete_gradients'][0]['complete_gradients']['passed']
    assert proof['whole_pair_capacity'][0]['retained_gradient_bytes']>0
    assert all(p.grad is None and torch.equal(p,w)for p,w in zip(parameters,weights))
    assert torch.equal(torch.get_rng_state(),rng) and active_cotangent_scope() is None
    assert {key:value for key,value in training.items()if key!='phase_qualifications'}==before
    # Proof resumes without replaying old episodes. Changed objectives cannot
    # borrow the previous proof, while completed exposure remains untouched.
    source.load=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unnecessary policy replay'))
    resumed=PolicyQualification(source,participants,vae,training,origin,'cpu')
    reused=ensure_phase_qualification(vae,participants,resumed,**{**args,'objective':resumed,'cache':{}})
    assert reused==proof
    source.declaration['clip_high']=.15
    changed=PolicyQualification(source,participants,vae,training,origin,'cpu')
    with pytest.raises(AssertionError,match='unnecessary policy replay'):
        ensure_phase_qualification(vae,participants,changed,**{**args,'objective':changed,'cache':{}})


def test_broken_policy_cached_reference_blocks_update_and_retains_actual_failure(tmp_path,monkeypatch):
    participants,vae,optimizer,source,training,origin=fixture(tmp_path,'rft')
    actual=participants['mellum'].model.forward
    def wrong(*args,**kwargs):
        result=actual(*args,**kwargs)
        if kwargs.get('reference'):result['h_final']=result['h_final']+.5
        return result
    monkeypatch.setattr(participants['mellum'].model,'forward',wrong)
    objective=PolicyQualification(source,participants,vae,training,origin,'cpu')
    before=copy.deepcopy(training);rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='gradient qualification failed'):
        ensure_phase_qualification(vae,participants,objective,phase='policy',options={},optimizer=optimizer,
            fetch_external=None,device='cpu',training=training,out=tmp_path,cache={},objective=objective)
    failure=json.loads(next((tmp_path/'phase-qualifications').glob('*.progress.json')).read_text())
    assert failure['status']=='failed' and failure['complete_gradients'][0]['passed'] is False
    assert failure['optimizer_updates']==failure['exposure_updates']==0
    assert not training['phase_qualifications']
    assert {key:value for key,value in training.items()if key!='phase_qualifications'}==before
    assert torch.equal(torch.get_rng_state(),rng) and active_cotangent_scope() is None
