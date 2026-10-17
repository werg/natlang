"""Production-rate ordinary gains retain actual updates without altering matrix storage."""
import copy
import json
import pytest
import torch
from test_native_full_updates import source
from natlang_neuralese.model.native_model import load_native_model,load_native_checkpoint,model_state,native_identity,CHECKPOINT_SCHEMA
from natlang_neuralese.model.native_parameter_precision import PROFILE,upgrade_norm_precision,norm_parameters
from natlang_neuralese.model.shared_state import Invocation,compile_graph
from natlang_neuralese.train.native_state_sft import make_native_optimizer


def test_production_size_norm_updates_and_numeric_optimizer_migration():
    from natlang_neuralese.train.optim import PortMuonAdamW
    def optimizer(p):
        matrix=torch.nn.Parameter(torch.zeros(16,16,dtype=torch.bfloat16))
        return PortMuonAdamW([('backbone.norm',p)],lr=3e-6,vocab_size=64416,force_adamw_ids={id(p)},
            latent=[('matrix',matrix,3e-6,1)],latent_optimizer='muonsr')
    p=torch.nn.Parameter(torch.ones(2304,dtype=torch.bfloat16));opt=optimizer(p)
    gradient=torch.linspace(-.1,.1,2304).bfloat16()
    for _ in range(8):p.grad=gradient;opt.step()
    assert torch.equal(p.detach(),torch.ones_like(p))
    saved=copy.deepcopy(opt.state_dict());old_moments=copy.deepcopy(opt.auxiliary.state[p]);rng=torch.get_rng_state().clone()
    promoted=torch.nn.Parameter(p.detach().float());repaired=optimizer(promoted)
    with pytest.raises(ValueError):repaired.load_state_dict(saved)
    repaired.load_state_dict(saved,allow_promoted_names=('backbone.norm',))
    assert torch.equal(torch.get_rng_state(),rng)
    for name,value in old_moments.items():
        assert torch.equal(repaired.auxiliary.state[promoted][name].float(),value.float())
    ref=torch.nn.Parameter(p.detach().float());reference=torch.optim.AdamW([ref],lr=3e-6,weight_decay=0)
    reference.state[ref]={k:v.float().clone() for k,v in old_moments.items()}
    promoted.grad=gradient.float();ref.grad=gradient.float();repaired.step();reference.step()
    assert torch.count_nonzero(promoted.detach()-p.float())==2304
    torch.testing.assert_close(promoted,ref,atol=0,rtol=0)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_declared_profile_real_graph_restore_and_raw_identity(tmp_path,family):
    from natlang_neuralese.rl.native_raw_policy import prepare_raw_policy
    dtype=source(tmp_path/'base',family)
    model,_,_,config=load_native_model(tmp_path/'base',dtype=dtype,rank=4)
    plan=compile_graph([Invocation('producer',(4,5,6),producing=((1,3),))])
    old={name:p.detach().clone() for name,p in model.port.hf.named_parameters()}
    versions={name:(id(p),p._version) for name,(_,p) in norm_parameters(model).items()}
    before=model(plan)['h_final'].detach()
    rng=torch.get_rng_state().clone();event=upgrade_norm_precision(model,config)
    assert torch.equal(rng,torch.get_rng_state())
    assert bool(event['promoted_parameters'])==(family=='mellum')
    for name,(_,p) in norm_parameters(model).items():
        assert id(p)==versions[name][0]
        assert p._version==versions[name][1]+(family=='mellum')
    for name,p in model.port.hf.named_parameters():torch.testing.assert_close(p.float(),old[name].float(),atol=0,rtol=0)
    after=model(plan)['h_final'];assert after.dtype==before.dtype
    torch.testing.assert_close(after,before,atol=0,rtol=0)
    opt,named=make_native_optimizer(model,lr=3e-6,in_backward=False)
    # Real graph loss drives the actual gains rather than assigned test gradients.
    model.logits(model(plan)['h_final']).float().square().mean().backward()
    gains={name:p.detach().clone() for name,(_,p) in norm_parameters(model).items()}
    opt.step()
    assert any(torch.count_nonzero(p.detach()-gains[name]) for name,(_,p) in norm_parameters(model).items())
    path=tmp_path/'parent.pt'
    from natlang_neuralese.train.loop import capture_training_rng_state
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),
        'training':{'schema':'natlang.native-state-sft/1','optimizer':opt.state_dict(),'rng':capture_training_rng_state('cpu')}},path)
    restored,_,_,actual,_=load_native_checkpoint(path)
    assert actual==config
    torch.testing.assert_close(restored(plan)['h_final'],model(plan)['h_final'],atol=0,rtol=0)
    assert all(p.dtype==torch.float32 for _,p in norm_parameters(restored).values())
    identity=native_identity(restored,actual,checkpoint_sha256='1'*64)
    # Raw schema validation is covered independently; metadata proxy must use identical profile identity.
    raw=prepare_raw_policy(path,base=tmp_path/'base',tokenizer_out=tmp_path/'tokenizer',require_optimizer=False)
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    assert raw['native_identity']==native_identity(restored,actual,checkpoint_sha256=sha256_file_hex_cached(path))


def test_bf16_hf_gain_profile_preserves_activation_dtype(tmp_path):
    source(tmp_path/'base','lfm')
    model,_,_,config=load_native_model(tmp_path/'base',dtype=torch.bfloat16,rank=4)
    plan=compile_graph([Invocation('answer',(4,5,6))])
    before=model(plan)['h_final'];upgrade_norm_precision(model,config)
    after=model(plan)['h_final'];assert after.dtype==before.dtype==torch.bfloat16
    torch.testing.assert_close(after,before,atol=0,rtol=0)


def test_upgrade_completed_trainer_preserves_exact_full_history(tmp_path):
    from natlang_neuralese.train import native_state_sft as trainer
    source(tmp_path/'base','lfm')
    _,_,_,config=load_native_model(tmp_path/'base',dtype=torch.bfloat16,rank=4)
    paths={}
    for split in ('train','test'):
        row={'schema':'natlang.neuralese-native-graph/1','id':split,'split':split,'source_groups':[split],
             'tokenizer_fingerprint':config['tokenizer_fingerprint'],'nodes':[{'name':'a','token_ids':[4,5,6],'target_weights':[0,1,1]}]}
        paths[split]=tmp_path/(split+'.jsonl');paths[split].write_text(json.dumps(row)+'\n')
    out=tmp_path/'out'
    argv=['--base',str(tmp_path/'base'),'--dtype','bfloat16','--rank','4','--device','cpu','--native-graphs',str(paths['train']),
          '--held-native-graphs',str(paths['test']),'--epochs','1','--eval-every','1','--checkpoint-every','1','--seed','23','--out',str(out),'--stage-id','foundation']
    assert trainer.main(argv)==0
    old=torch.load(out/'checkpoint.pt',weights_only=False)
    assert trainer.main(argv+['--native-ordinary-precision',PROFILE])==0
    new=torch.load(out/'checkpoint.pt',weights_only=False)
    assert new['model_config']['native_ordinary_precision']==PROFILE
    for key in ('step','global_step','exposure_counts','source_row_bindings','source_receipts'):
        assert new['training'][key]==old['training'][key]
    assert torch.equal(new['training']['rng']['torch_rng'],old['training']['rng']['torch_rng'])
    assert new['training']['rng']['python_rng']==old['training']['rng']['python_rng']
    assert new['training']['rng']['cuda_rng']==old['training']['rng']['cuda_rng']
    for name,value in old['backbone_state'].items():torch.testing.assert_close(new['backbone_state'][name].float(),value.float(),atol=0,rtol=0)
    assert new['training']['options']['ordinary_precision_history'][-1]['profile']==PROFILE
    # Subsequent update restores the declared profile without another transition.
    assert trainer.main(argv+['--epochs','2'])==0
    final=torch.load(out/'checkpoint.pt',weights_only=False)
    assert final['training']['step']==2
    assert len(final['training']['options']['ordinary_precision_history'])==1


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_composite_full_patch_preserves_declared_norm_promotion(tmp_path,family):
    from natlang_neuralese.model.native_composite import write_native_composite_checkpoint
    dtype=source(tmp_path/'base',family)
    original,_,_,config=load_native_model(tmp_path/'base',dtype=dtype,rank=4)
    parent_path=tmp_path/'parent.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(original)},parent_path)
    parent,_,_,config,_=load_native_checkpoint(parent_path)
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    identity=native_identity(parent,config,checkpoint_sha256=sha256_file_hex_cached(parent_path))
    updated=copy.deepcopy(parent);config=copy.deepcopy(config)
    upgrade_norm_precision(updated,config)
    binding=write_native_composite_checkpoint(parent_path,tmp_path/'composite',updated,config,
        base_identity=identity,parent_model=parent,updated_parameter_names=list(dict(updated.named_parameters())))
    restored,_,_,actual,_=load_native_checkpoint(binding['path'])
    assert actual==config
    assert all(p.dtype==torch.float32 for _,p in norm_parameters(restored).values())
    plan=compile_graph([Invocation('answer',(4,5,6))])
    torch.testing.assert_close(restored(plan)['h_final'],updated(plan)['h_final'],atol=0,rtol=0)
    # Interface-only publication cannot hide a backbone dtype transition behind unchanged tensor versions.
    with pytest.raises(ValueError,match='unchanged backbone|ordinary precision'):
        write_native_composite_checkpoint(parent_path,tmp_path/'bad',updated,config,base_identity=identity)


def test_profile_rejects_undeclared_storage_and_downgrade(tmp_path):
    dtype=source(tmp_path/'base','mellum')
    model,_,_,config=load_native_model(tmp_path/'base',dtype=dtype,rank=4)
    upgrade_norm_precision(model,config)
    gain=next(iter(norm_parameters(model).values()))[1]
    gain.data=gain.detach().bfloat16()
    with pytest.raises(ValueError,match='storage'):
        native_identity(model,config,checkpoint_sha256='1'*64)
    with pytest.raises(ValueError,match='unsupported|downgrade'):
        upgrade_norm_precision(model,config,'bf16')


def test_already_upgraded_checkpoint_composite_and_mislabeled_tensor_rejection(tmp_path):
    from natlang_neuralese.model.native_composite import write_native_composite_checkpoint
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    dtype=source(tmp_path/'base','mellum')
    model,_,_,config=load_native_model(tmp_path/'base',dtype=dtype,rank=4)
    upgrade_norm_precision(model,config)
    path=tmp_path/'upgraded.pt'
    checkpoint={'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model)}
    torch.save(checkpoint,path)
    parent,_,_,config,_=load_native_checkpoint(path)
    identity=native_identity(parent,config,checkpoint_sha256=sha256_file_hex_cached(path))
    updated=copy.deepcopy(parent);event=upgrade_norm_precision(updated,config)
    assert event['previous_profile']==event['profile']==PROFILE and not event['promoted_parameters']
    binding=write_native_composite_checkpoint(path,tmp_path/'composite',updated,config,
        base_identity=identity,parent_model=parent,updated_parameter_names=list(dict(updated.named_parameters())))
    restored,_,_,actual,_=load_native_checkpoint(binding['path'])
    assert actual==config
    for name,value in model_state(restored)['backbone_state'].items():
        torch.testing.assert_close(value,checkpoint['backbone_state'][name],atol=0,rtol=0)
    wrong=copy.deepcopy(checkpoint)
    gain=next(iter(norm_parameters(model)))
    wrong['backbone_state'][gain]=wrong['backbone_state'][gain].bfloat16()
    torch.save(wrong,tmp_path/'mislabeled.pt')
    with pytest.raises(ValueError,match='declared FP32 profile'):
        load_native_checkpoint(tmp_path/'mislabeled.pt')


@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl','native_offline_preference'])
def test_recipe_declares_norm_profile_handoff_and_fresh_capacity(tmp_path,kind):
    from natlang_neuralese.train.recipe import HANDLERS,validate_stage_parameters,direct_windowing_handoffs,stage_parameter_args
    assert 'native_ordinary_precision' in HANDLERS[kind]['parameters']
    validate_stage_parameters(kind,{'native_ordinary_precision':PROFILE})
    assert stage_parameter_args({'native_ordinary_precision':PROFILE})==['--native-ordinary-precision',PROFILE]
    with pytest.raises(ValueError,match='ordinary precision'):
        validate_stage_parameters(kind,{'native_ordinary_precision':'bf16'})
    old={'stages':[{'id':'foundation','kind':kind,'parameters':{'epochs':2}}]}
    new=copy.deepcopy(old);new['stages'][0]['parameters']['native_ordinary_precision']=PROFILE
    with pytest.raises(ValueError,match='fresh restored-model'):
        direct_windowing_handoffs(old,new,tmp_path)
    new['stages'][0]['parameters']['memory_qualification']={'path':'actual-profile-max-request.json','sha256':'a'*64}
    changes,_,_=direct_windowing_handoffs(old,new,tmp_path)
    assert changes['foundation']['native_ordinary_precision']==[None,PROFILE]
    for bad in ({'memory_qualification':True}, {'memory_qualification':{'path':'x','sha256':'bad'}}):
        with pytest.raises(ValueError,match='memory_qualification'):
            validate_stage_parameters(kind,bad)
    old_with_request=copy.deepcopy(old)
    old_with_request['stages'][0]['parameters']['memory_qualification']=new['stages'][0]['parameters']['memory_qualification']
    with pytest.raises(ValueError,match='fresh restored-model'):
        direct_windowing_handoffs(old_with_request,new,tmp_path)
    with pytest.raises(ValueError,match='norm storage upgrade'):
        direct_windowing_handoffs(new,old,tmp_path)
