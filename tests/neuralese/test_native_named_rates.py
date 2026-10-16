"""Exact named native optimizer slots survive adaptive rate groups and resume."""
import copy
import pytest
import torch
from natlang_neuralese.train.optim import PortMuonAdamW
from natlang_neuralese.train.optim_restore import set_learning_rates,OptimizerRestoreError


def mixed(order=('row_a','matrix','row_b')):
    params={name:torch.nn.Parameter(torch.ones((16,16) if name=='matrix' else (4,))) for name in order}
    optimizer=PortMuonAdamW([(name,params[name]) for name in order],lr=.001,vocab_size=999)
    return params,optimizer


def named_history(optimizer):
    return {optimizer._parameter_names[id(p)]:(group,child.state[p])
        for child in (optimizer.muon,optimizer.auxiliary,optimizer.latent) if child is not None
        for group in child.param_groups for p in group['params']}


def test_raw_port_named_slots_survive_schema_child_order_split_reorder_and_rates():
    params,opt=mixed()
    for index,p in enumerate(params.values()):p.grad=torch.full_like(p,index+1.)
    opt.step();opt.zero_grad()
    set_learning_rates(opt,params,{'row_b':.003})
    saved=copy.deepcopy(opt.state_dict())
    assert saved['schema'][0]['name']=='row_a'
    assert saved['param_names']['muon']==['matrix']
    assert saved['param_names']['adamw']==['row_a','row_b']
    changed,new=mixed(('row_b','matrix','row_a'))
    new.load_state_dict(saved)
    actual=named_history(new);expected=named_history(opt)
    for name in params:
        assert actual[name][0]['lr']==expected[name][0]['lr']
        for key,value in expected[name][1].items():
            if torch.is_tensor(value):assert torch.equal(actual[name][1][key],value),(name,key)
            else:assert actual[name][1][key]==value
    assert new.state_dict()['param_names']['adamw']==['row_b','row_a']
    for name in params:changed[name].data.copy_(params[name])
    for name in params:
        params[name].grad=torch.full_like(params[name],.4)
        changed[name].grad=torch.full_like(changed[name],.4)
    opt.step();new.step()
    for name in params:assert torch.equal(params[name],changed[name]),name


def test_legacy_unsplit_schema_is_recovered_but_ambiguous_split_is_rejected():
    params,opt=mixed();legacy=opt.state_dict();legacy.pop('param_names')
    _,new=mixed(('row_b','matrix','row_a'));new.load_state_dict(legacy)
    set_learning_rates(opt,params,{'row_b':.003})
    ambiguous=opt.state_dict();ambiguous.pop('param_names')
    for child in (ambiguous['muon'],ambiguous['adamw']):
        for group in child['param_groups']:group.pop('param_names',None)
    with pytest.raises(OptimizerRestoreError,match='historical split'):new.load_state_dict(ambiguous)


def test_latent_backward_hooks_follow_actual_parameter_after_group_reorder():
    def build():
        values=[torch.nn.Parameter(torch.arange(256).reshape(16,16).float()/1000+i) for i in (0,1)]
        optimizer=PortMuonAdamW([],lr=.001,vocab_size=999,
            latent=[(name,p,.001,1) for name,p in zip(('a','b'),values)],latent_optimizer='muonsr')
        return values,optimizer
    actual,opt=build();expected,reference=build()
    opt.latent.step_in_backward(gated=True)
    opt.latent.param_groups.reverse();opt.param_groups=opt._groups()
    set_learning_rates(opt,dict(zip(('a','b'),actual)),{'a':.002,'b':.005})
    set_learning_rates(reference,dict(zip(('a','b'),expected)),{'a':.002,'b':.005})
    with opt.latent.in_backward():(actual[0].sum()+2*actual[1].sum()).backward()
    expected[0].grad=torch.ones_like(expected[0]);expected[1].grad=torch.full_like(expected[1],2.)
    reference.step()
    for a,b in zip(actual,expected):assert torch.equal(a,b)
    for group in opt.latent.param_groups:assert group['qat_latent_lr']==group['lr']


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_native_updates_preserve_distinct_rows_through_resume_and_semantic_rate_changes(family):
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
    from natlang_neuralese.train.native_learning_rates import apply_native_learning_rates,LEARNED_ROWS
    from natlang_neuralese.train.native_semantic_stage import _adjust_lr
    def build():
        result=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
        if family=='mellum':result.port.hf.natlang_precision='bf16'
        return result
    model=build();optimizer,_=make_native_optimizer(model,lr=3e-6,learned_row_lr=.003,in_backward=True)
    before=model.input_rows.detach().clone()
    native_training_update(model,optimizer,examples(),checkpoint_layers=True,chunk_tokens=3)
    assert (before!=model.input_rows).any()
    saved=copy.deepcopy(optimizer.state_dict())
    resumed=build();resumed.port.hf.load_state_dict(model.port.hf.state_dict());resumed.load_state_dict(model.state_dict())
    target,_=make_native_optimizer(resumed,lr=1e-5,in_backward=True);target.load_state_dict(saved)
    old=named_history(target)
    moments={name:copy.deepcopy(history) for name,(_,history) in old.items()}
    apply_native_learning_rates(target,backbone_lr=1e-5)
    _adjust_lr(target,2e-5)
    for name,(group,history) in named_history(target).items():
        assert group['lr']==(.003 if name in LEARNED_ROWS else 2e-5),name
        for key,value in moments[name].items():
            if torch.is_tensor(value):assert torch.equal(value,history[key]),(name,key)
            else:assert value==history[key]
    apply_native_learning_rates(optimizer,backbone_lr=2e-5)
    rng=torch.get_rng_state()
    expected,_=native_training_update(model,optimizer,examples(),checkpoint_layers=True,chunk_tokens=3)
    torch.set_rng_state(rng)
    actual,_=native_training_update(resumed,target,examples(),checkpoint_layers=True,chunk_tokens=3)
    assert torch.equal(expected,actual)
    for key,value in model.state_dict().items():assert torch.equal(value,resumed.state_dict()[key]),key
    for key,value in model.port.hf.state_dict().items():assert torch.equal(value,resumed.port.hf.state_dict()[key]),key
    _adjust_lr(target,2e-5,learned_row_lr=.001)
    assert all(group['lr']==(.001 if name in LEARNED_ROWS else 2e-5)
               for name,(group,_) in named_history(target).items())


def test_actual_sft_cli_inherits_row_rate_on_resume_and_retains_exposure(tmp_path):
    import json
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.data.native_graph import SCHEMA
    from natlang_neuralese.train.native_state_sft import main
    base=tmp_path/'base';tiny_source(base)
    source_model,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    expected_rows={'interface.input_rows'} | (set() if source_model.tied else {'interface.output_rows'})
    def row(split):
        return {'schema':SCHEMA,'id':split,'split':split,'source_groups':[split],
            'tokenizer_fingerprint':config['tokenizer_fingerprint'],'nodes':[
                {'name':'writer','token_ids':[1,roles.produce_open,5,roles.produce_close],
                 'producing':[[2,3]],'target_weights':[0,1,1,1]},
                {'name':'reader','token_ids':[6,roles.consume_open,roles.consume_close,9],
                 'imports':[{'source':'writer','start':2,'stop':3,'at':2}], 'target_weights':[0,1,0,1]}]}
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl';out=tmp_path/'run'
    train.write_text(json.dumps(row('train'))+'\n');held.write_text(json.dumps(row('test'))+'\n')
    argv=['--base',str(base),'--native-graphs',str(train),'--held-native-graphs',str(held),'--out',str(out),
        '--device','cpu','--dtype','float32','--rank','4','--epochs','1','--batch-graphs','1',
        '--eval-every','1','--checkpoint-every','1','--chunk-tokens','2',
        '--forward-atol','1e-4','--gradient-atol','1e-5','--learned-row-learning-rate','.003']
    assert main(argv)==0
    first=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert first['exposure_counts']=={'train':1}
    argv[argv.index('--epochs')+1]='2';del argv[-2:]
    assert main(argv)==0
    second=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert second['exposure_counts']=={'train':2}
    rates={};histories={}
    for kind,names in second['optimizer']['param_names'].items():
        child=second['optimizer'][kind];groups=child['param_groups']
        slots=[slot for group in groups for slot in group['params']]
        for name,slot in zip(names,slots):
            rates[name]=next(group['lr'] for group in groups if slot in group['params'])
            histories[name]=child['state'].get(slot,{})
    assert all(rates[name]==.003 for name in expected_rows)
    assert second['step']==first['step']+1
    for name in expected_rows:
        assert histories[name] and any(torch.is_tensor(value) and value.count_nonzero() for value in histories[name].values())


def test_actual_three_phase_shared_checkpoint_retains_row_rates_and_named_slots(tmp_path,monkeypatch):
    import json
    from test_native_semantic_stage import _fixture
    from natlang_neuralese.train import native_semantic_stage as stage
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    path,manifest,config,_=_fixture(tmp_path,monkeypatch)
    config['learned_row_learning_rate']=.003;path.write_text(json.dumps(config))
    report=stage.run(path,manifest,tmp_path/'shared',device='cpu')
    assert report['completed'] and report['step']==3
    checkpoint=CheckpointPolicy.load(tmp_path/'shared'/'checkpoint.pt')
    for participant in checkpoint['participants'].values():
        saved=participant['optimizer'];found=set()
        for kind,names in saved['param_names'].items():
            groups=saved[kind]['param_groups'];slots=[slot for group in groups for slot in group['params']]
            for name,slot in zip(names,slots):
                rate=next(group['lr'] for group in groups if slot in group['params'])
                if name in {'interface.input_rows','interface.output_rows'}:
                    assert rate==.003;found.add(name)
        expected={entry['name'] for entry in saved['schema']
                  if entry['name'] in {'interface.input_rows','interface.output_rows'}}
        assert 'interface.input_rows' in found and found==expected


def test_serialized_names_cannot_alias_two_parameters_to_one_saved_slot():
    _,optimizer=mixed();saved=optimizer.state_dict()
    group=saved['adamw']['param_groups'][0]
    assert len(group['params'])==2
    group['params'][1]=group['params'][0]
    with pytest.raises(OptimizerRestoreError,match='slot-name inventory'):
        optimizer.load_state_dict(saved)


@pytest.mark.parametrize('rate',[0,-1,float('nan'),True,'0.003'])
def test_rate_declarations_reject_invalid_native_recipe_and_phase_values(tmp_path,monkeypatch,rate):
    from test_native_semantic_stage import _fixture
    from natlang_neuralese.train import native_semantic_stage as stage
    from natlang_neuralese.train.recipe import validate_stage_parameters
    for kind in ('native_state_sft','native_state_rl'):
        with pytest.raises(ValueError,match='learned_row_learning_rate'):
            validate_stage_parameters(kind,{'learned_row_learning_rate':rate})
    _,_,config,_=_fixture(tmp_path,monkeypatch)
    config['phases'][1]['options']={'learned_row_learning_rate':rate}
    with pytest.raises(ValueError,match='learned_row_learning_rate'):stage.validate_config(config)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_role_only_rate_override_preserves_reasoning_moments_rng_and_semantic_resume(family):
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples,assert_exact
    from test_native_reasoning_rows import training_examples
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
    from natlang_neuralese.train.native_learning_rates import apply_native_learning_rates,INTERFACE_ROWS,LEARNED_ROWS
    from natlang_neuralese.train.native_semantic_stage import _adjust_lr
    def build():
        model=SharedStateModel(tiny(family),(58,59,60,61),rank=4,token_override_ids=(54,55),
            sparse_row_dtype=torch.float32,token_override_dtype=torch.float32)
        if family=='mellum':model.port.hf.natlang_precision='bf16'
        return model
    model=build();optimizer,_=make_native_optimizer(model,lr=3e-6,learned_row_lr=3e-6,in_backward=True)
    batch=examples()+training_examples()
    native_training_update(model,optimizer,batch,checkpoint_layers=True,chunk_tokens=3)
    prior={name:copy.deepcopy(history) for name,(_,history) in named_history(optimizer).items()}
    weights={key:value.detach().clone() for key,value in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    apply_native_learning_rates(optimizer,backbone_lr=3e-6,interface_row_lr=.003)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in model.state_dict().items():assert torch.equal(weights[key],value)
    for name,(group,history) in named_history(optimizer).items():
        assert group['lr']==(.003 if name in INTERFACE_ROWS else 3e-6),name
        assert_exact(prior[name],history,name)
    assert any(name in LEARNED_ROWS-INTERFACE_ROWS for name in prior)
    saved=copy.deepcopy(optimizer.state_dict())
    resumed=build();resumed.load_state_dict(model.state_dict());resumed.port.hf.load_state_dict(model.port.hf.state_dict())
    target,_=make_native_optimizer(resumed,lr=3e-6,in_backward=True);target.load_state_dict(saved)
    _adjust_lr(target,3e-6)  # Subsequent shared phases inherit the actual two row rates.
    for name,(group,history) in named_history(target).items():
        assert group['lr']==(.003 if name in INTERFACE_ROWS else 3e-6),name
        assert_exact(prior[name],history,name)
    before=model.input_rows.detach().clone()
    rng=torch.get_rng_state().clone()
    reference,_=native_training_update(model,optimizer,batch,checkpoint_layers=True,chunk_tokens=3)
    torch.set_rng_state(rng)
    actual,_=native_training_update(resumed,target,batch,checkpoint_layers=True,chunk_tokens=3)
    assert torch.equal(reference,actual)
    assert (before!=model.input_rows).any()
    for key,value in model.state_dict().items():assert torch.equal(value,resumed.state_dict()[key]),key
    for key,value in model.port.hf.state_dict().items():assert torch.equal(value,resumed.port.hf.state_dict()[key]),key


@pytest.mark.parametrize('value',[0,-.001,float('inf'),float('nan'),True,'0.003'])
def test_declared_role_only_rate_requires_finite_positive_number(value):
    from natlang_neuralese.train.recipe import validate_stage_parameters
    with pytest.raises(ValueError,match='interface_row_learning_rate'):
        validate_stage_parameters('native_state_sft',{'interface_row_learning_rate':value})
