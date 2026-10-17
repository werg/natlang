"""Declared norm dispatch preserves legacy arithmetic independently of storage."""
import copy
import pytest
import torch

from natlang_neuralese.model.native_parameter_precision import configure_norm_execution, validate_norm_execution
from natlang_neuralese.model.native_model import load_native_model, load_native_checkpoint, model_state, native_identity, CHECKPOINT_SCHEMA
from natlang_neuralese.model.shared_state import SharedStateModel, channel_checkpoint, restore_channel_checkpoint


@pytest.mark.parametrize('family', ['lfm','mellum'])
def test_execution_is_explicit_and_preserves_parameters_rng(family):
    from test_shared_state import tiny
    model = SharedStateModel(tiny(family), (60,61,62,63), rank=4)
    assert model.native_norm_execution == model.port.native_norm_execution == 'source-v1'
    assert 'native_norm_execution' not in channel_checkpoint(model)
    parameters = {name:(id(p),p.detach().clone()) for name,p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    configure_norm_execution(model, 'stable-v1')
    validate_norm_execution(model, {'native_norm_execution':'stable-v1'})
    assert torch.equal(rng, torch.get_rng_state())
    assert all(id(p)==parameters[name][0] and torch.equal(p,parameters[name][1]) for name,p in model.named_parameters())
    saved = channel_checkpoint(model)
    configure_norm_execution(model, None)
    with pytest.raises(ValueError, match='native_norm_execution'):
        restore_channel_checkpoint(model, saved)
    with pytest.raises(ValueError, match='declared profile'):
        validate_norm_execution(model, {'native_norm_execution':'stable-v1'})
    with pytest.raises(ValueError): configure_norm_execution(model, 'invented')


@pytest.mark.parametrize('family', ['lfm','mellum'])
@pytest.mark.parametrize('profile', [None,'source-v1','stable-v1'])
def test_full_restore_raw_identity_and_composite(tmp_path,family,profile):
    from test_native_full_updates import source
    from natlang_neuralese.model.native_composite import write_native_composite_checkpoint
    from natlang_neuralese.rl.native_raw_policy import prepare_raw_policy
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    dtype=source(tmp_path/'base',family)
    model,_,_,config=load_native_model(tmp_path/'base',dtype=dtype,rank=4,native_norm_execution=profile)
    assert ('native_norm_execution' in config)==(profile is not None)
    parent=tmp_path/'parent.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model)},parent)
    restored,_,_,actual,_=load_native_checkpoint(parent)
    assert actual==config
    assert restored.native_norm_execution==(profile or 'source-v1')
    identity=native_identity(restored,actual,checkpoint_sha256=sha256_file_hex_cached(parent))
    raw=prepare_raw_policy(parent,base=tmp_path/'base',tokenizer_out=tmp_path/'tokens',require_optimizer=False)
    assert raw['native_identity']==identity
    mismatched=copy.deepcopy(torch.load(parent,weights_only=False))
    mismatched['interface_state']['native_norm_execution']='source-v1' if profile=='stable-v1' else 'stable-v1'
    torch.save(mismatched,tmp_path/'mismatched.pt')
    with pytest.raises(ValueError,match='norm execution'):
        prepare_raw_policy(tmp_path/'mismatched.pt',base=tmp_path/'base',tokenizer_out=tmp_path/'bad-tokens',require_optimizer=False)
    bad=copy.deepcopy(torch.load(parent,weights_only=False))
    bad['model_config']['native_norm_execution']=None
    torch.save(bad,tmp_path/'invalid.pt')
    with pytest.raises(ValueError,match='invalid norm execution'):
        load_native_checkpoint(tmp_path/'invalid.pt')
    changed=copy.deepcopy(restored);new_config=copy.deepcopy(config)
    configure_norm_execution(changed,'stable-v1');new_config['native_norm_execution']='stable-v1'
    binding=write_native_composite_checkpoint(parent,tmp_path/'composite',changed,new_config,
        base_identity=identity,parent_model=restored,updated_parameter_names=list(dict(changed.named_parameters())))
    composite,_,_,round_config,_=load_native_checkpoint(binding['path'])
    assert round_config==new_config
    assert composite.native_norm_execution=='stable-v1'
    assert all(torch.equal(value,composite.state_dict()[name]) for name,value in changed.state_dict().items())
    configure_norm_execution(restored,'stable-v1')
    with pytest.raises(ValueError,match='declared profile'):
        native_identity(restored,{**config,'native_norm_execution':'source-v1'},checkpoint_sha256='1'*64)


def test_recipe_and_cli_accept_only_declared_modes():
    from natlang_neuralese.train.native_state_sft import build_parser
    from natlang_neuralese.train.recipe import validate_stage_parameters
    for profile in ('source-v1','stable-v1'):
        assert build_parser().parse_args(['--native-graphs','train','--held-native-graphs','held','--out','out','--native-norm-execution',profile]).native_norm_execution==profile
        validate_stage_parameters('native_state_sft', {'native_norm_execution':profile})
    with pytest.raises(ValueError):
        validate_stage_parameters('native_state_sft', {'native_norm_execution':'wrong'})


def test_completed_trainer_transition_preserves_optimizer_rng_and_credits(tmp_path):
    import json
    from test_native_full_updates import source
    from test_native_shared_checkpoint import equal_tree
    from natlang_neuralese.train import native_state_sft as trainer
    source(tmp_path/'base','lfm')
    _,_,_,config=load_native_model(tmp_path/'base',dtype=torch.bfloat16,rank=4)
    for split in ('train','test'):
        row={'schema':'natlang.neuralese-native-graph/1','id':split,'split':split,'source_groups':[split],
             'tokenizer_fingerprint':config['tokenizer_fingerprint'],
             'nodes':[{'name':'a','token_ids':[4,5,6],'target_weights':[0,1,1]}]}
        (tmp_path/(split+'.jsonl')).write_text(json.dumps(row)+'\n')
    out=tmp_path/'out'
    args=['--base',str(tmp_path/'base'),'--dtype','bfloat16','--rank','4','--device','cpu',
        '--native-graphs',str(tmp_path/'train.jsonl'),'--held-native-graphs',str(tmp_path/'test.jsonl'),
        '--epochs','1','--eval-every','1','--checkpoint-every','1','--seed','23','--out',str(out),'--stage-id','foundation']
    assert trainer.main(args+['--native-norm-execution','source-v1'])==0
    before=torch.load(out/'checkpoint.pt',weights_only=False)
    assert trainer.main(args+['--native-norm-execution','stable-v1'])==0
    after=torch.load(out/'checkpoint.pt',weights_only=False)
    assert after['model_config']['native_norm_execution']=='stable-v1'
    for key in ('optimizer','rng','step','global_step','exposure_counts','source_row_bindings','source_receipts'):
        equal_tree(before['training'][key],after['training'][key])
    equal_tree(before['backbone_state'],after['backbone_state'])
    assert after['training']['options']['native_norm_execution_history']==[
        {'from':'source-v1','to':'stable-v1','at_global_step':1,'weights_optimizer_rng_exposure':'preserved'}]
    assert trainer.main(args+['--epochs','2'])==0
    final=torch.load(out/'checkpoint.pt',weights_only=False)
    assert final['training']['step']==2
    assert final['model_config']['native_norm_execution']=='stable-v1'
    assert final['training']['options']['native_norm_execution_history']==after['training']['options']['native_norm_execution_history']


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('profile',['source-v1','stable-v1'])
def test_profiles_graph_cached_execution_and_adjoints(family,profile):
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import Invocation,compile_graph,native_step
    port=tiny(family)
    model=SharedStateModel(port,tuple(range(port.embedding_weight.shape[0]-4,port.embedding_weight.shape[0])),
        rank=4,native_norm_execution=profile)
    torch.nn.init.normal_(model.adapter.up.weight,std=.03)
    plan=compile_graph([Invocation('ordinary',(2,3,4,5,6))])
    flat=model(plan)['h_final']
    prefix=native_step(model,torch.tensor([[2,3]]))
    suffix=native_step(model,torch.tensor([[4,5,6]]),prefix['cache'])
    cached=torch.cat((prefix['h_final'],suffix['h_final']),dim=1)
    torch.testing.assert_close(flat,cached,atol=3e-6,rtol=3e-5)
    weights=torch.randn_like(flat)
    parameters=(model.input_rows,model.adapter.up.weight)
    a=torch.autograd.grad((flat*weights).sum(),parameters)
    b=torch.autograd.grad((cached*weights).sum(),parameters)
    for actual,expected in zip(a,b):torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA norm dispatch')
@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_transition_reference_gate_only_when_dispatch_changes(monkeypatch,family,dtype):
    from test_shared_state import tiny
    from natlang_neuralese.model.native_parameter_precision import norm_execution_changes_arithmetic
    from natlang_neuralese.train.native_state_sft import qualify_precision_transition
    model=SharedStateModel(tiny(family).to(device='cuda',dtype=dtype),(60,61,62,63),rank=4,
        native_norm_execution='stable-v1')
    monkeypatch.setenv('NATLANG_MAPLE_COMPILE','1')
    monkeypatch.setenv('NATLANG_MAPLE_FUSED_NORM','1')
    assert norm_execution_changes_arithmetic(model,'source-v1')==(family=='mellum' and dtype==torch.float32)
    args=dict(previous_precision='ieee',previous_norm_execution='source-v1',checkpoint_sha256='a'*64,
        forward_atol=None,gradient_atol=None,device='cuda')
    config={'native_linear_precision':'ieee','native_norm_execution':'stable-v1'}
    if family=='mellum' and dtype==torch.float32:
        with pytest.raises(ValueError,match='declared forward and gradient tolerances'):
            qualify_precision_transition(model,None,None,config,**args)
        from natlang_neuralese.eval import native_state
        calls=[]
        def qualified(*unused,**kwargs):
            calls.append(kwargs)
            torch.rand(1,device='cuda')
            return {'passed':True}
        monkeypatch.setattr(native_state,'qualify_existing',qualified)
        rng=torch.cuda.get_rng_state().clone()
        result=qualify_precision_transition(model,None,None,config,**{**args,'forward_atol':1e-4,'gradient_atol':1e-5})
        assert result['norm_execution']=={'from':'source-v1','to':'stable-v1'}
        assert len(calls)==1 and torch.equal(rng,torch.cuda.get_rng_state())
    else:
        assert qualify_precision_transition(model,None,None,config,**args) is None
    monkeypatch.setenv('NATLANG_MAPLE_FUSED_NORM','0')
    assert norm_execution_changes_arithmetic(model,'source-v1')==(family=='mellum')
    monkeypatch.setenv('NATLANG_MAPLE_COMPILE','0')
    assert not norm_execution_changes_arithmetic(model,'source-v1')
