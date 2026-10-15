"""Sparse existing-token repair preserves parents and learns both boundaries."""
import copy
from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.model.shared_state import SharedStateModel,Invocation,compile_graph,channel_checkpoint,restore_channel_checkpoint
from natlang_neuralese.model.native_model import upgrade_reasoning_tokens,native_identity
from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
from natlang_neuralese.data.native_graph import parse_graph,SCHEMA
from test_shared_state import tiny
from test_native_optimizer_checkpointing import assert_exact


def optimizer_rows(optimizer):
    state=optimizer.state_dict()
    result={}
    for key in ('muon','adamw','muonsr'):
        child=state.get(key)
        if child is None:continue
        ids=[index for group in child['param_groups'] for index in group['params']]
        names=[item['name'] for item in state['schema'] if item['optimizer']==key]
        result.update({name:child['state'].get(index,{}) for name,index in zip(names,ids)})
    return result


def training_examples():
    return [parse_graph({'schema':SCHEMA,'id':'reasoning-'+str(i),'split':'train','source_groups':['train'],
        'tokenizer_fingerprint':'fixture','nodes':[{'name':'answer',
        'token_ids':[1,54,5+i,7,55,9+i,2],'target_weights':[0,1,1,.5,1,1,1]}]},
        tokenizer_fingerprint='fixture') for i in range(2)]


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_explicit_reasoning_upgrade_preserves_logits_old_moments_and_rng_then_learns_both(family):
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    with torch.no_grad():
        model.port.embedding_weight[55].copy_(model.port.embedding_weight[54])
        if model.port.output_weight is not model.port.embedding_weight:
            model.port.output_weight[55].copy_(model.port.output_weight[54])
    config={'tokenizer_fingerprint':'fixture','architecture':str(model.port.config.model_type)}
    old_optimizer,_=make_native_optimizer(model,lr=.003,in_backward=False)
    batch=training_examples()
    old_optimizer.zero_grad(set_to_none=True)
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    loss,_=packed_graph_loss(model,batch,chunk_tokens=1)
    loss.backward();old_optimizer.step()
    old_state=copy.deepcopy(old_optimizer.state_dict())
    old_rows=copy.deepcopy(optimizer_rows(old_optimizer))
    old_parameters={name:value.detach().clone() for name,value in model.named_parameters()}
    plan=compile_graph([Invocation('probe',(1,54,5,55,2))])
    with torch.no_grad():expected=model.logits(model(plan)['h_final']).clone()
    rng=torch.get_rng_state().clone()
    before_identity=native_identity(model,config,checkpoint_sha256='1'*64)
    event=upgrade_reasoning_tokens(model,SimpleNamespace(get_vocab=lambda:{'<think>':54,'</think>':55}),config)
    assert event['added'] and model.marker_ids==(58,59,60,61) and model.token_override_ids==(54,55)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in old_parameters.items():assert_exact(value,dict(model.named_parameters())[name],name)
    with torch.no_grad():assert_exact(model.logits(model(plan)['h_final']),expected,'initial logits')
    assert native_identity(model,config,checkpoint_sha256='1'*64)!=before_identity
    optimizer,_=make_native_optimizer(model,lr=.003,in_backward=True)
    optimizer.load_state_dict(old_state,allow_added_names=('interface.extra_input_rows','interface.extra_output_rows'))
    current=optimizer_rows(optimizer)
    for name,value in old_rows.items():assert_exact(value,current[name],'old optimizer/'+name)
    for name in current.keys()-old_rows.keys():assert current[name]=={}
    initial=model.extra_input_rows.detach().clone()
    if not model.tied:initial_output=model.extra_output_rows.detach().clone()
    native_training_update(model,optimizer,batch,chunk_tokens=1,checkpoint_layers=True)
    assert (model.extra_input_rows.detach()!=initial).any(dim=1).all()
    if not model.tied:assert (model.extra_output_rows.detach()!=initial_output).any(dim=1).all()
    learned_output=model.extra_input_rows if model.tied else model.extra_output_rows
    assert not torch.equal(learned_output[0],learned_output[1])
    current=optimizer_rows(optimizer)
    for name in current.keys()-old_rows.keys():
        assert any(value.abs().sum()>0 for value in current[name].values() if isinstance(value,torch.Tensor))
    restored=SharedStateModel(tiny(family),(58,59,60,61),rank=4,token_override_ids=(54,55))
    restored.port.hf.load_state_dict(model.port.hf.state_dict())
    if family=='mellum':restored.port.hf.natlang_precision='bf16'
    restore_channel_checkpoint(restored,channel_checkpoint(model))
    with torch.no_grad():assert_exact(restored.logits(restored(plan)['h_final']),model.logits(model(plan)['h_final']),'roundtrip')


def test_optimizer_added_rows_requires_explicit_narrow_permission():
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    old,_=make_native_optimizer(model,lr=.003,in_backward=False)
    model.add_token_overrides((54,55))
    current,_=make_native_optimizer(model,lr=.003,in_backward=False)
    with pytest.raises(ValueError,match='differ'):
        current.load_state_dict(old.state_dict())
    with pytest.raises(ValueError,match='existing parameter contracts'):
        current.load_state_dict(old.state_dict(),allow_added_names=('not-the-new-rows',))


def test_actual_old_checkpoint_upgrade_shared_trainer_resume_and_serve_roundtrip(tmp_path,monkeypatch):
    import json
    from test_native_model import tiny_source
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.model.native_model import load_native_model,load_native_checkpoint,CHECKPOINT_SCHEMA,model_state
    base=tmp_path/'base';tiny_source(base)
    path=base/'tokenizer.json';raw=json.loads(path.read_text())
    raw['model']['vocab']['<think>']=raw['model']['vocab'].pop('t54')
    raw['model']['vocab']['</think>']=raw['model']['vocab'].pop('t55')
    path.write_text(json.dumps(raw))
    legacy,tokenizer,roles,config=load_native_model(base,dtype=torch.float32,rank=4,token_override_ids=())
    assert not legacy.token_override_ids
    fresh,_,_,fresh_config=load_native_model(base,dtype=torch.float32,rank=4)
    assert fresh.token_override_ids==(54,55)
    def row(name,split):
        return {'schema':SCHEMA,'id':name,'split':split,'source_groups':[name],
            'tokenizer_fingerprint':config['tokenizer_fingerprint'],'nodes':[{'name':'answer',
            'token_ids':[1,54,5,55,7,2],'target_weights':[0,1,1,1,1,1]}]}
    rows=[row('train0','train'),row('train1','train')]
    train=tmp_path/'train.jsonl';train.write_text(''.join(json.dumps(item)+'\n' for item in rows))
    held=tmp_path/'held.jsonl';held.write_text(json.dumps(row('held','test'))+'\n')
    common=['--base',str(base),'--device','cpu','--dtype','float32','--rank','4','--epochs','1',
        '--held-native-graphs',str(held),'--batch-graphs','1','--eval-every','1','--chunk-tokens','1',
        '--learning-rate','.003','--weight-decay','0']
    with monkeypatch.context() as patch:
        # Simulate the released old factory; actual shared trainer writes every
        # optimizer/RNG/cursor field, rather than fabricating training metadata.
        patch.setattr(trainer,'load_native_model',lambda *a,**kw:load_native_model(*a,**kw,token_override_ids=()))
        assert trainer.main(['--native-graphs',str(train),'--out',str(tmp_path/'old'),*common])==0
    parent=tmp_path/'old/checkpoint.pt'
    old=torch.load(parent,map_location='cpu',weights_only=False)
    restored,restored_tokenizer,_,old_config,_=load_native_checkpoint(parent)
    assert not restored.token_override_ids and old_config==old['model_config']
    plan=compile_graph([Invocation('probe',(1,54,5,55,2))])
    with torch.no_grad():baseline=restored.logits(restored(plan)['h_final'])
    detached=tmp_path/'old-exact.pt';torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':old_config,**model_state(restored)},detached)
    same,_,_,_,_=load_native_checkpoint(detached)
    with torch.no_grad():assert_exact(same.logits(same(plan)['h_final']),baseline,'old untouched load')
    from natlang_neuralese.model.native_composite import write_native_composite_checkpoint
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    old_identity=native_identity(restored,old_config,checkpoint_sha256=sha256_file_hex_cached(parent))
    upgraded=copy.deepcopy(restored);upgraded_config=copy.deepcopy(old_config)
    upgrade_reasoning_tokens(upgraded,restored_tokenizer,upgraded_config)
    with torch.no_grad():upgraded.extra_input_rows.add_(.02)
    first_composite=write_native_composite_checkpoint(parent,tmp_path/'first-upgrade',upgraded,upgraded_config,
        base_identity=old_identity,parent_model=restored,
        updated_parameter_names=list(dict(upgraded.named_parameters())))
    migrated,_,_,_,_=load_native_checkpoint(first_composite['path'])
    with torch.no_grad():assert_exact(migrated.logits(migrated(plan)['h_final']),upgraded.logits(upgraded(plan)['h_final']),'old-parent composite upgrade')
    rows.append(row('train2','train'));train.write_text(''.join(json.dumps(item)+'\n' for item in rows))
    argv=['--native-graphs',str(train),'--out',str(tmp_path/'new'),'--continue-from',str(parent),
          '--learn-reasoning-boundaries',*common]
    assert trainer.main(argv)==0
    updated=torch.load(tmp_path/'new/checkpoint.pt',map_location='cpu',weights_only=False)
    assert updated['training']['global_step']==old['training']['global_step']+1
    assert updated['training']['exposure_counts']=={'train0':1,'train1':1,'train2':1}
    assert updated['model_config']['token_override_ids']==[54,55]
    assert updated['training']['options']['token_row_upgrade_history'][-1]['added']
    live,_,_,live_config,_=load_native_checkpoint(tmp_path/'new/checkpoint.pt')
    assert live.token_override_ids==(54,55)
    assert native_identity(live,live_config,checkpoint_sha256='2'*64)['embedding_weight_sha256']
    from natlang_neuralese.model.native_composite import write_native_composite_checkpoint
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    joint_path=tmp_path/'new/checkpoint.pt'
    identity=native_identity(live,live_config,checkpoint_sha256=sha256_file_hex_cached(joint_path))
    with torch.no_grad():live.extra_input_rows.add_(.01)
    composite=write_native_composite_checkpoint(joint_path,tmp_path/'portable',live,live_config,base_identity=identity)
    portable,_,_,portable_config,_=load_native_checkpoint(composite['path'])
    assert portable.token_override_ids==(54,55)
    with torch.no_grad():assert_exact(portable.logits(portable(plan)['h_final']),live.logits(live(plan)['h_final']),'portable learned rows')
    before=copy.deepcopy(updated['training']['optimizer'])
    assert trainer.main(argv)==0
    resumed=torch.load(tmp_path/'new/checkpoint.pt',map_location='cpu',weights_only=False)
    assert_exact(resumed['training']['optimizer'],before,'noop resumed optimizer')
