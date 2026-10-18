"""Existing token-local FFN execution: full graph, full named backward, real resume."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.train import native_state_sft as trainer
from natlang_neuralese.model.shared_state import forward_graph
from test_shared_state import tiny,graph


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('chunk',[1,5,11,12,13])
@pytest.mark.parametrize('checkpointed',[False,True])
def test_whole_shared_graph_ffn_chunks_preserve_outputs_and_all_named_derivatives(family,chunk,checkpointed):
    port=tiny(family);port.checkpoint_layers=checkpointed;plan=graph()
    inputs=torch.randn(1,plan.token_count,port.embedding_weight.shape[1],requires_grad=True)
    cotangent=torch.randn_like(inputs);named=[('input',inputs),*port.named_parameters()]
    parameters=[p for _,p in named if p.requires_grad];names=[n for n,p in named if p.requires_grad]
    trainer.configure_ffn_chunk_tokens(port,0,{})
    expected=forward_graph(port,plan,embeddings=inputs)['h_final']
    reference=torch.autograd.grad((expected*cotangent).sum(),parameters,allow_unused=True)
    widths=[];hooks=[]
    for layer in port.layers:
        module=getattr(layer,'feed_forward',None)or getattr(layer,'mlp',None)
        hooks.append(module.register_forward_pre_hook(lambda module,args:widths.append(args[0].shape[1])))
    try:
        assert trainer.configure_ffn_chunk_tokens(port,chunk,{})==chunk
        actual=forward_graph(port,plan,embeddings=inputs)['h_final']
        derivatives=torch.autograd.grad((actual*cotangent).sum(),parameters,allow_unused=True)
    finally:
        for hook in hooks:hook.remove()
    assert widths and max(widths)<=min(chunk,plan.token_count)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    for name,actual,expected in zip(names,derivatives,reference,strict=True):
        assert (actual is None)==(expected is None),name
        if actual is not None:torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5,msg=name)


@pytest.mark.parametrize('requested,saved,expected',[(None,{},0),(None,{'ffn_chunk_tokens':7},7),(0,{'ffn_chunk_tokens':7},0),(2,{'ffn_chunk_tokens':7},2)])
def test_ffn_execution_inherits_exact_saved_setting_with_explicit_override(requested,saved,expected):
    port=SimpleNamespace()
    assert trainer.configure_ffn_chunk_tokens(port,requested,{'options':saved})==port.ffn_chunk_tokens==expected


@pytest.mark.parametrize('bad',[-1,1.5,True,'4',None])
def test_corrupt_saved_ffn_setting_rejects(bad):
    with pytest.raises(ValueError,match='nonnegative integer'):
        trainer.configure_ffn_chunk_tokens(SimpleNamespace(),None,{'options':{'ffn_chunk_tokens':bad}})


def test_negative_ffn_setting_rejects_before_any_model_allocation(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer,'load_native_model',lambda *a,**k:pytest.fail('allocated model'))
    with pytest.raises(ValueError,match='ffn-chunk-tokens'):
        trainer.main(['--native-graphs','unused','--held-native-graphs','unused','--out',str(tmp_path/'out'),'--ffn-chunk-tokens','-1'])


def test_capacity_proof_distinguishes_actual_ffn_execution(tmp_path):
    from test_native_policy_capacity import actual
    from natlang_neuralese.eval.native_memory import qualify_policy_objective
    engine,model,opt,objective,train,args,config,backend,batches=actual(tmp_path,'lfm')
    try:
        first=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,checkpoint_sha256='a'*64,pending_batches=batches)
        args.ffn_chunk_tokens=trainer.configure_ffn_chunk_tokens(model.port,2,{})
        second=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,checkpoint_sha256='a'*64,pending_batches=batches,prior_proof=first)
        assert second['path']!=first['path']
        proof=json.loads(Path(second['path']).read_text());assert proof['passed']and proof['identity']['ffn_chunk_tokens']==2
        assert all(row['gradients']for row in proof['capacity']['cases'])
    finally:engine.stop();engine.store.close()


def test_real_resume_keeps_ffn_execution_and_exact_optimizer_rng_history(tmp_path,monkeypatch):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.data.native_graph import SCHEMA
    from test_native_relative_quantum import _same
    base=tmp_path/'base';tiny_source(base);_,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    row={'schema':SCHEMA,'id':'train','split':'train','source_groups':['train'],'tokenizer_fingerprint':config['tokenizer_fingerprint'],
        'nodes':[{'name':'answer','token_ids':[1,2,3,4,5],'target_weights':[0,1,1,1,1]}]}
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl';train.write_text(json.dumps(row)+'\n');held.write_text(json.dumps({**row,'id':'held','split':'test','source_groups':['held']})+'\n')
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--native-graphs',str(train),'--held-native-graphs',str(held),'--epochs','3','--eval-every','1','--seed','1','--checkpoint-every','1']
    run=tmp_path/'run';assert trainer.main([*args,'--out',str(run),'--ffn-chunk-tokens','2','--refresh-after-updates','1'])==0
    first=torch.load(run/'checkpoint.pt',weights_only=False);assert first['training']['options']['ffn_chunk_tokens']==2
    seen=[];configure=trainer.configure_ffn_chunk_tokens
    def observe(port,requested,saved):
        value=configure(port,requested,saved);seen.append((requested,value));return value
    monkeypatch.setattr(trainer,'configure_ffn_chunk_tokens',observe)
    assert trainer.main([*args,'--out',str(run),'--refresh-after-updates','1'])==0
    resumed=torch.load(run/'checkpoint.pt',weights_only=False);assert seen==[(None,2)]and resumed['training']['options']['ffn_chunk_tokens']==2
    baseline=tmp_path/'baseline';baseline.mkdir();torch.save(first,baseline/'checkpoint.pt')
    assert trainer.main([*args,'--out',str(baseline),'--ffn-chunk-tokens','2','--refresh-after-updates','1'])==0
    expected=torch.load(baseline/'checkpoint.pt',weights_only=False)
    for key in('model','channel'):
        if key in resumed:_same(resumed[key],expected[key])
    for key in('optimizer','rng','global_step','exposure_counts','source_row_bindings','source_row_order'):
        _same(resumed['training'][key],expected['training'][key])

@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl','native_offline_preference'])
@pytest.mark.parametrize('value',[None,0,7])
def test_shared_recipe_accepts_exact_ffn_setting(kind,value):
    from natlang_neuralese.train.recipe import HANDLERS,validate_stage_parameters,stage_parameter_args
    assert 'ffn_chunk_tokens'in HANDLERS[kind]['parameters']
    validate_stage_parameters(kind,{'ffn_chunk_tokens':value})
    if value is not None:assert stage_parameter_args({'ffn_chunk_tokens':value})==['--ffn-chunk-tokens',str(value)]

@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl','native_offline_preference'])
@pytest.mark.parametrize('value',[-1,True,1.5,'7'])
def test_shared_recipe_rejects_invalid_ffn_setting(kind,value):
    from natlang_neuralese.train.recipe import validate_stage_parameters
    with pytest.raises(ValueError,match='ffn_chunk_tokens'):
        validate_stage_parameters(kind,{'ffn_chunk_tokens':value})
