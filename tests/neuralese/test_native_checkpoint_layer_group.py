"""Grouped recomputation controls, exact capacity identity and real resume custody."""
import json
from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.train import native_state_sft as trainer


@pytest.mark.parametrize('requested,saved,expected',[(None,{},4),(None,{'checkpoint_layer_group_size':1},1),
    (None,{'checkpoint_layer_group_size':2},2),(4,{'checkpoint_layer_group_size':1},4),(1,{},1)])
def test_group_size_inherits_explicit_setting_and_defaults_legacy_to_four(requested,saved,expected):
    port=SimpleNamespace()
    assert trainer.configure_checkpoint_layer_group_size(port,requested,{'options':saved})==expected
    assert port.checkpoint_layer_group_size==expected


@pytest.mark.parametrize('value',[0,-1,True,1.5,'4',None])
def test_invalid_saved_group_size_is_rejected(value):
    with pytest.raises(ValueError,match='positive integer'):
        trainer.configure_checkpoint_layer_group_size(SimpleNamespace(),None,{'options':{'checkpoint_layer_group_size':value}})


@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl','native_offline_preference','native_coefficient_rl'])
@pytest.mark.parametrize('value',[None,1,2,4])
def test_recipe_exposes_group_size_for_shared_trainer(kind,value):
    from natlang_neuralese.train.recipe import HANDLERS,validate_stage_parameters,stage_parameter_args
    assert 'checkpoint_layer_group_size' in HANDLERS[kind]['parameters']
    validate_stage_parameters(kind,{'checkpoint_layer_group_size':value})
    if value is not None:assert stage_parameter_args({'checkpoint_layer_group_size':value})==['--checkpoint-layer-group-size',str(value)]


@pytest.mark.parametrize('value',[0,-1,True,1.5,'2'])
def test_recipe_rejects_invalid_group_size(value):
    from natlang_neuralese.train.recipe import validate_stage_parameters
    with pytest.raises(ValueError,match='checkpoint_layer_group_size'):
        validate_stage_parameters('native_state_sft',{'checkpoint_layer_group_size':value})


def test_capacity_identity_separates_group_sizes_and_preserves_legacy_default(monkeypatch):
    from natlang_neuralese.eval import native_memory
    monkeypatch.setattr(native_memory,'memory_runtime',lambda device:{'device':device})
    args=SimpleNamespace(batch_graphs=1,batch_tokens=65536,checkpoint_layers=True,chunk_tokens=1024,
        checkpoint_input_offload=False,device='cpu',fp8_training='off',fp8_recipe='delayed',fp8_amax_history_len=16)
    legacy=native_memory.capacity_identity({},args,None)
    args.checkpoint_layer_group_size=1
    assert native_memory.capacity_identity({},args,None)==legacy
    args.checkpoint_layer_group_size=2
    two=native_memory.capacity_identity({},args,None)
    args.checkpoint_layer_group_size=4
    four=native_memory.capacity_identity({},args,None)
    assert two['checkpoint_layer_group_size']==2 and four['checkpoint_layer_group_size']==4 and legacy!=two!=four


def test_real_resume_adopts_grouping_and_preserves_optimizer_rng_and_exposure(tmp_path):
    from test_native_model import tiny_source
    from test_native_relative_quantum import _same
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.data.native_graph import SCHEMA
    base=tmp_path/'base';tiny_source(base);_,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    row={'schema':SCHEMA,'id':'train','split':'train','source_groups':['train'],
         'tokenizer_fingerprint':config['tokenizer_fingerprint'],
         'nodes':[{'name':'answer','token_ids':[1,2,3,4,5],'target_weights':[0,1,1,1,1]}]}
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    train.write_text(json.dumps(row)+'\n');held.write_text(json.dumps({**row,'id':'held','split':'test','source_groups':['held']})+'\n')
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--native-graphs',str(train),
          '--held-native-graphs',str(held),'--epochs','3','--batch-graphs','1','--accumulate-batches','1','--eval-every','1','--seed','1','--checkpoint-every','1']
    initial=tmp_path/'initial';assert trainer.main([*args,'--out',str(initial),'--checkpoint-layer-group-size','1','--refresh-after-updates','1'])==0
    legacy=torch.load(initial/'checkpoint.pt',weights_only=False)
    options=legacy['training']['options'];options.pop('checkpoint_layer_group_size');options.pop('checkpoint_layer_group_size_history')
    for name in ('adopted','explicit','optout'):
        directory=tmp_path/name;directory.mkdir();torch.save(legacy,directory/'checkpoint.pt')
    assert trainer.main([*args,'--out',str(tmp_path/'adopted'),'--refresh-after-updates','1'])==0
    assert trainer.main([*args,'--out',str(tmp_path/'explicit'),'--checkpoint-layer-group-size','4','--refresh-after-updates','1'])==0
    assert trainer.main([*args,'--out',str(tmp_path/'optout'),'--checkpoint-layer-group-size','1','--refresh-after-updates','1'])==0
    actual=torch.load(tmp_path/'adopted/checkpoint.pt',weights_only=False)
    expected=torch.load(tmp_path/'explicit/checkpoint.pt',weights_only=False)
    optout=torch.load(tmp_path/'optout/checkpoint.pt',weights_only=False)
    assert actual['training']['options']['checkpoint_layer_group_size']==4
    history=actual['training']['options']['checkpoint_layer_group_size_history']
    assert history[-1]['from']==1 and history[-1]['to']==4 and history[-1]['weights_optimizer_rng_exposure']=='preserved'
    assert optout['training']['options']['checkpoint_layer_group_size']==1
    for key in ('model','channel'):
        if key in actual:_same(actual[key],expected[key])
    for key in ('optimizer','rng','global_step','exposure_counts','source_row_bindings','source_row_order'):
        _same(actual['training'][key],expected['training'][key])
        _same(actual['training'][key],optout['training'][key])


def test_coefficient_grouping_adopts_and_preserves_explicit_resume_choice():
    from natlang_neuralese.rl.native_coefficient_stage import _configure_coefficient_checkpoint_group
    port=SimpleNamespace();state={'step':7}
    assert _configure_coefficient_checkpoint_group(port,state,None)==4
    assert state['checkpoint_layer_group_size_history']==[{'schema':'natlang.checkpoint-layer-group-transition/1',
        'from':1,'to':4,'at_step':7,'weights_optimizer_rng_exposure':'preserved'}]
    assert _configure_coefficient_checkpoint_group(port,state,2)==2
    assert state['checkpoint_layer_group_size_history'][-1]['from']==4
    assert _configure_coefficient_checkpoint_group(port,state,None)==2
    assert len(state['checkpoint_layer_group_size_history'])==2


def test_capacity_verification_rejects_claimed_group_without_actual_execution():
    from natlang_neuralese.eval.native_memory import _require_checkpoint_group_execution
    model=SimpleNamespace(port=SimpleNamespace(checkpoint_layer_group_size=1))
    args=SimpleNamespace(checkpoint_layer_group_size=4)
    with pytest.raises(ValueError,match='differs from actual model'):
        _require_checkpoint_group_execution(model,args)
    model.port.checkpoint_layer_group_size=4
    _require_checkpoint_group_execution(model,args)


def test_oversize_capacity_reuse_binds_actual_group_size(monkeypatch):
    from natlang_neuralese.train import native_resident
    from natlang_neuralese.train.native_packed_capacity import oversize_execution_identity
    monkeypatch.setattr(native_resident,'code_inventory',lambda package:{})
    model=torch.nn.Linear(2,2);model.port=SimpleNamespace(checkpoint_layer_group_size=1)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    args=SimpleNamespace(checkpoint_layers=True,chunk_tokens=32,accumulate_batches=1,
                         cotangent_accumulator_backend='plain')
    first=oversize_execution_identity(model,optimizer,args=args,backend=None,objective_binding=None)
    model.port.checkpoint_layer_group_size=4
    grouped=oversize_execution_identity(model,optimizer,args=args,backend=None,objective_binding=None)
    assert first['sha256']!=grouped['sha256']
    model.port.ffn_chunk_tokens=4096
    chunked=oversize_execution_identity(model,optimizer,args=args,backend=None,objective_binding=None)
    assert grouped['sha256']!=chunked['sha256']
    from natlang_neuralese.common.checkpoint_offload import configure_checkpoint_input_offload
    configure_checkpoint_input_offload(model.port,enabled=True)
    offloaded=oversize_execution_identity(model,optimizer,args=args,backend=None,objective_binding=None)
    assert chunked['sha256']!=offloaded['sha256']


@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl','native_offline_preference','native_coefficient_rl'])
def test_group_override_is_a_recorded_resume_safe_stage_transition(tmp_path,kind):
    from natlang_neuralese.train.recipe import native_stage_parameter_handoffs
    before={'stages':[{'id':'grouped','kind':kind,'parameters':{'checkpoint_layer_group_size':1}}]}
    after={'stages':[{'id':'grouped','kind':kind,'parameters':{'checkpoint_layer_group_size':2}}]}
    changes,old,new=native_stage_parameter_handoffs(before,after,tmp_path)
    assert changes['grouped']['checkpoint_layer_group_size']==[1,2]
    assert old==new


def test_coefficient_cli_parses_group_and_rejects_invalid_before_loading(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_coefficient_stage
    monkeypatch.setattr(native_coefficient_stage,'_run_cli',lambda args:args.checkpoint_layer_group_size)
    args=['--coefficient-policy','policy','--graphs','graphs','--objective','objective','--continue-from','base',
          '--held-native-graphs','held','--out',str(tmp_path/'out'),'--coefficient-optimizer','{}',
          '--task-eval','{}','--task-eval-every','1']
    assert native_coefficient_stage.main([*args,'--checkpoint-layer-group-size','2'])==2
    with pytest.raises(ValueError,match='positive integer'):
        native_coefficient_stage.main([*args,'--checkpoint-layer-group-size','0'])
