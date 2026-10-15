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
        from natlang_neuralese.train.optim_restore import port_saved_slot_names
        names=port_saved_slot_names(state,key,child)
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


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_bf16_backbone_reasoning_master_rows_learn_at_production_lr(family):
    port=tiny(family).to(dtype=torch.bfloat16)
    if family=='mellum':port.hf.natlang_precision='bf16'
    model=SharedStateModel(port,(58,59,60,61),rank=4,sparse_row_dtype=torch.bfloat16)
    old_optimizer,_=make_native_optimizer(model,lr=1e-5,in_backward=False)
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    # Exercise every role as an input, not merely its full-softmax output gradient.
    examples=[parse_graph({'schema':SCHEMA,'id':'role-probe','split':'train','source_groups':['train'],
        'tokenizer_fingerprint':'fixture','nodes':[{'name':'answer','token_ids':[1,58,54,59,60,55,61,2],
        'target_weights':[0,1,1,1,1,1,1,1]}]},tokenizer_fingerprint='fixture')]
    old_loss,_=packed_graph_loss(model,examples,chunk_tokens=64)
    old_loss.backward();old_optimizer.step();old_optimizer.zero_grad(set_to_none=True)
    old_state=copy.deepcopy(old_optimizer.state_dict())
    old_moments=copy.deepcopy(optimizer_rows(old_optimizer))
    old_roles={name:getattr(model,name).detach().clone() for name in ('input_rows',) + (() if model.tied else ('output_rows',))}
    plan=compile_graph([Invocation('probe',(1,54,5,55,2))])
    ids=torch.tensor([[1,54,55,2]])
    with torch.no_grad():
        baseline=model.logits(model(plan)['h_final']).clone()
        embeddings=model.embed(ids).clone()
    config={}
    rng=torch.get_rng_state().clone()
    event=upgrade_reasoning_tokens(model,SimpleNamespace(get_vocab=lambda:{'<think>':54,'</think>':55}),config)
    assert torch.equal(torch.get_rng_state(),rng)
    assert model.input_rows.dtype==torch.float32
    for name,value in old_roles.items():assert torch.equal(getattr(model,name).to(value.dtype),value)
    assert model.extra_input_rows.dtype==torch.float32
    if not model.tied:assert model.extra_output_rows.dtype==torch.float32
    assert config['token_override_dtype']=='torch.float32'
    with torch.no_grad():
        assert torch.equal(model.logits(model(plan)['h_final']),baseline)
        assert torch.equal(model.embed(ids),embeddings)
    optimizer,_=make_native_optimizer(model,lr=1e-5,in_backward=False)
    optimizer.load_state_dict(old_state,allow_added_names=('interface.extra_input_rows','interface.extra_output_rows'),
        allow_promoted_names=tuple('interface.'+name for name in event['promoted_parameters']))
    for name,saved in old_moments.items():
        restored=optimizer_rows(optimizer)[name]
        assert restored.keys()==saved.keys()
        for key,value in saved.items():
            if isinstance(value,torch.Tensor):
                assert torch.equal(restored[key].to(value.dtype),value),(name,key)
                if name in {'interface.input_rows','interface.output_rows'} and value.ndim:
                    assert restored[key].dtype==torch.float32
            else:assert restored[key]==value
    names=('input_rows','extra_input_rows') if model.tied else ('input_rows','output_rows','extra_input_rows','extra_output_rows')
    initial={name:getattr(model,name).detach().clone() for name in names}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss,_=packed_graph_loss(model,examples,chunk_tokens=64)
        loss.backward()
        for name in names:assert getattr(model,name).grad.abs().sum()>0
        optimizer.step()
    for name in names:
        assert (getattr(model,name).detach()!=initial[name]).any(dim=1).all(),name
    state=channel_checkpoint(model)
    assert state['token_override_dtype']=='torch.float32'
    restored=SharedStateModel(tiny(family).to(dtype=torch.bfloat16),(58,59,60,61),rank=4,
                              token_override_ids=(54,55),token_override_dtype=torch.float32,sparse_row_dtype=torch.float32)
    restore_channel_checkpoint(restored,state)
    for name in names:assert torch.equal(getattr(restored,name),getattr(model,name))


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('malformation',['shape','dtype','declared_dtype'])
def test_sparse_row_restore_rejects_malformed_tensor_before_mutating_channel(family,malformation):
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4,token_override_ids=(54,55))
    baseline=channel_checkpoint(model)
    state=copy.deepcopy(baseline)
    state['input_rows'].add_(1)  # rejection must happen before even earlier valid tensors mutate
    if malformation=='shape':state['extra_input_rows']=state['extra_input_rows'][:1]
    elif malformation=='dtype':state['extra_input_rows']=state['extra_input_rows'].to(torch.float64)
    else:state['token_override_dtype']='torch.float64'
    with pytest.raises(ValueError,match='sparse token checkpoint.*dtype'):
        restore_channel_checkpoint(model,state)
    assert_exact(channel_checkpoint(model),baseline,'unchanged after rejection')


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('existing_reasoning',[False,True])
def test_sparse_master_upgrade_real_full_and_both_composite_scopes(tmp_path,family,existing_reasoning):
    import json
    from test_native_model import tiny_source
    from test_mellum_port import tiny_mellum
    from natlang_neuralese.model.native_model import load_native_model,load_native_checkpoint,model_state,CHECKPOINT_SCHEMA
    from natlang_neuralese.model.native_composite import write_native_composite_checkpoint
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    base=tmp_path/'base';base.mkdir()
    if family=='lfm':tiny_source(base)
    else:
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        tiny_mellum(base)
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {f't{i}':i for i in range(128)},unk_token='t0')),unk_token='t0',bos_token='t1',eos_token='t2',pad_token='t3')
        tokenizer.save_pretrained(base)
    path=base/'tokenizer.json';raw=json.loads(path.read_text())
    raw['model']['vocab']['<think>']=raw['model']['vocab'].pop('t54')
    raw['model']['vocab']['</think>']=raw['model']['vocab'].pop('t55')
    path.write_text(json.dumps(raw))
    dtype=torch.float32 if family=='lfm' else torch.bfloat16
    old,tokenizer,roles,config=load_native_model(base,dtype=dtype,rank=4,
        token_override_ids=(54,55) if existing_reasoning else (),sparse_row_dtype=dtype,token_override_dtype=dtype)
    config.pop('sparse_row_dtype')
    config.pop('token_override_dtype',None)
    payload={'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(old)}
    payload['interface_state'].pop('sparse_row_dtype')  # released old parent has no sparse dtype metadata
    payload['interface_state'].pop('token_override_dtype',None)
    parent=tmp_path/'old.pt';torch.save(payload,parent)
    old,tokenizer,roles,old_config,_=load_native_checkpoint(parent)
    assert old_config==config and old.input_rows.dtype==dtype
    if family=='lfm':
        cast_old,_,_,cast_config,_=load_native_checkpoint(parent,dtype=torch.bfloat16)
        assert cast_config['sparse_row_dtype']=='torch.float32'
        cast_path=tmp_path/'cast-old.pt'
        torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':cast_config,**model_state(cast_old)},cast_path)
        cast_again,_,_,_,_=load_native_checkpoint(cast_path)
        assert cast_again.input_rows.dtype==torch.float32
    plan=compile_graph([Invocation('probe',(1,roles.produce_open,54,roles.produce_close,
        roles.consume_open,55,roles.consume_close,2))])
    with torch.no_grad():baseline=old.logits(old(plan)['h_final']).clone()
    identity=native_identity(old,config,checkpoint_sha256=sha256_file_hex_cached(parent))
    for scope in ('native-interface','model-parameters'):
        if scope=='native-interface':live,_,_,new_config,_=load_native_checkpoint(parent)
        else:live,new_config=copy.deepcopy(old),copy.deepcopy(config)
        event=upgrade_reasoning_tokens(live,tokenizer,new_config)
        with torch.no_grad():
            assert torch.equal(live.logits(live(plan)['h_final']),baseline)
            live.extra_input_rows.add_(.001)
            if not live.tied:live.extra_output_rows.sub_(.002)
        options={} if scope=='native-interface' else {'parent_model':old,'updated_parameter_names':list(dict(live.named_parameters()))}
        publication=write_native_composite_checkpoint(parent,tmp_path/scope,live,new_config,base_identity=identity,**options)
        restored,_,_,restored_config,_=load_native_checkpoint(publication['path'])
        assert restored_config==new_config
        assert restored.input_rows.dtype==restored.extra_input_rows.dtype==torch.float32
        with torch.no_grad():assert torch.equal(restored.logits(restored(plan)['h_final']),live.logits(live(plan)['h_final']))
        # A later update must not replay the old-parent promotion on an already upgraded parent.
        next_identity=native_identity(restored,restored_config,checkpoint_sha256=publication['sha256'])
        with torch.no_grad():restored.extra_input_rows.add_(.001)
        again=write_native_composite_checkpoint(publication['path'],tmp_path/(scope+'-again'),restored,restored_config,
                                                base_identity=next_identity)
        again_model,_,_,_,_=load_native_checkpoint(again['path'])
        with torch.no_grad():assert torch.equal(again_model.logits(again_model(plan)['h_final']),restored.logits(restored(plan)['h_final']))
        full=tmp_path/(scope+'.pt');torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':new_config,**model_state(live)},full)
        full_model,_,_,full_config,_=load_native_checkpoint(full)
        assert full_config==new_config and full_model.extra_input_rows.dtype==torch.float32
        if family=='lfm':
            cast,_,_,_,_=load_native_checkpoint(full,dtype=torch.bfloat16)
            assert cast.port.embedding_weight.dtype==torch.bfloat16
            assert cast.input_rows.dtype==cast.extra_input_rows.dtype==torch.float32


def test_native_identity_rejects_mismatched_sparse_precision_config():
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4,token_override_ids=(54,55))
    config={'architecture':'lfm2','tokenizer_fingerprint':'fixture','sparse_row_dtype':'torch.bfloat16'}
    with pytest.raises(ValueError,match='sparse row configuration'):
        native_identity(model,config,checkpoint_sha256='a'*64)
    config['sparse_row_dtype']='torch.float32';config['token_override_dtype']='torch.bfloat16'
    with pytest.raises(ValueError,match='sparse token dtype'):
        native_identity(model,config,checkpoint_sha256='a'*64)
