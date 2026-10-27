"""Attention arithmetic is per model, checkpointed, and shared by graph and serving."""
import copy

import pytest
import torch

from natlang_neuralese.model.native_precision import set_native_attention_precision
from natlang_neuralese.model.shared_state import SharedStateModel


def test_attention_mode_preserves_legacy_identity_and_restores_explicit_checkpoint(tmp_path):
    from test_native_serve import _tiny_source as tiny_source
    from natlang_neuralese.model.native_model import (
        load_native_model, load_native_checkpoint, model_state, native_identity, CHECKPOINT_SCHEMA)
    base = tmp_path/'base';tiny_source(base)
    model, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    legacy = {'schema':CHECKPOINT_SCHEMA,'model_config':copy.deepcopy(config),**model_state(model)}
    assert 'native_attention_precision' not in config
    assert 'native_attention_precision' not in legacy['interface_state']
    old_identity = native_identity(model,config,checkpoint_sha256='a'*64)
    flags = (torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    parameters = dict(model.named_parameters())
    weights = {name:value.detach().clone() for name,value in parameters.items()}
    set_native_attention_precision(model,'tf32x3')
    assert all(dict(model.named_parameters())[name] is value for name,value in parameters.items())
    assert all(torch.equal(weights[name],value) for name,value in model.named_parameters())
    assert flags == (torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    with pytest.raises(ValueError,match='attention arithmetic'):
        native_identity(model,config,checkpoint_sha256='a'*64)
    config['native_attention_precision']='tf32x3'
    assert native_identity(model,config,checkpoint_sha256='a'*64) != old_identity
    path=tmp_path/'current.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model)},path)
    restored,_,_,restored_config,_=load_native_checkpoint(path)
    assert restored.native_attention_precision == restored.port.native_attention_precision == 'tf32x3'
    assert restored_config == config
    assert native_identity(restored,config,checkpoint_sha256='a'*64)==native_identity(model,config,checkpoint_sha256='a'*64)
    from natlang_neuralese.serve import load_native_engine
    engine=load_native_engine(str(path),device='cpu')
    try:
        assert engine.shared_model.native_attention_precision == engine.model_config['native_attention_precision'] == 'tf32x3'
    finally:
        engine.stop()
    legacy_path=tmp_path/'legacy.pt';torch.save(legacy,legacy_path)
    old,_,_,old_config,_=load_native_checkpoint(legacy_path)
    assert old_config == legacy['model_config']
    assert native_identity(old,old_config,checkpoint_sha256='a'*64)==old_identity
    bad={'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model)}
    bad['interface_state'].pop('native_attention_precision')
    torch.save(bad,path)
    with pytest.raises(ValueError,match='native_attention_precision'):
        load_native_checkpoint(path)


@pytest.mark.parametrize('kind',['native_state_sft','native_offline_preference','native_state_rl'])
def test_recipe_declares_and_validates_attention_arithmetic(kind,tmp_path):
    from natlang_neuralese.train.recipe import HANDLERS, validate_stage_parameters,native_stage_parameter_handoffs
    assert 'native_attention_precision' in HANDLERS[kind]['parameters']
    validate_stage_parameters(kind,{'native_attention_precision':'tf32x3'})
    validate_stage_parameters(kind,{'native_attention_precision':'tf32'})
    with pytest.raises(ValueError,match='attention precision'):
        validate_stage_parameters(kind,{'native_attention_precision':'bf16'})
    before={'stages':[{'id':'native','kind':kind,'parameters':{'native_attention_precision':'ieee'}}]}
    after=copy.deepcopy(before);after['stages'][0]['parameters']['native_attention_precision']='tf32x3'
    changes,_,_=native_stage_parameter_handoffs(before,after,tmp_path)
    assert changes['native']['native_attention_precision']==['ieee','tf32x3']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual native CUDA attention')
@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_real_both_port_graph_cached_adjoints_and_incremental_serving(family,dtype):
    from test_shared_state import tiny, graph
    from natlang_neuralese.model.shared_state import native_step,compile_graph,Invocation
    from natlang_neuralese.model.native_precision import set_native_linear_precision
    model=SharedStateModel(tiny(family).to(device='cuda',dtype=dtype),(60,61,62,63),rank=4)
    model.port.checkpoint_layers=True
    set_native_linear_precision(model,'bf16x6' if dtype==torch.float32 else 'ieee')
    set_native_attention_precision(model,'tf32x3')
    with torch.no_grad():model.adapter.up.weight.normal_(std=.01)
    plan=graph()
    flat,cached=model(plan),model(plan,reference=True)
    torch.testing.assert_close(flat['h_final'],cached['h_final'],atol=5e-6,rtol=2e-5)
    named=tuple((name,value) for name,value in model.named_parameters() if value.requires_grad)
    parameters=tuple(value for _,value in named)
    left=model.logits(flat['h_final'][:,-1]).float().square().mean()
    right=model.logits(cached['h_final'][:,-1]).float().square().mean()
    actual=torch.autograd.grad(left,parameters,allow_unused=True)
    reference=torch.autograd.grad(right,parameters,allow_unused=True)
    observed=0
    for (name,parameter),a,b in zip(named,actual,reference):
        assert (a is None)==(b is None)
        if a is not None:
            if parameter.dtype==torch.float32:
                torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4,msg=lambda message:name+': '+message)
            observed+=int(bool(a.abs().sum()))
    assert observed>=3
    adapter_index=next(i for i,value in enumerate(parameters) if value is model.adapter.up.weight)
    assert actual[adapter_index].norm()>0
    if dtype==torch.bfloat16:
        # This option must leave ALL existing BF16 math unchanged. Frozen tiny
        # Mellum trains its interface; BF16 LFM's backbone is a diagnostic path.
        # FP32 interface/cache adjoints retain the existing strict checks above.
        set_native_attention_precision(model,'ieee')
        baseline=model(plan)
        baseline_loss=model.logits(baseline['h_final'][:,-1]).float().square().mean()
        baseline_grad=torch.autograd.grad(baseline_loss,parameters,allow_unused=True)
        assert torch.equal(flat['h_final'],baseline['h_final'])
        for a,b in zip(actual,baseline_grad):
            assert (a is None)==(b is None)
            if a is not None:assert torch.equal(a,b)
        set_native_attention_precision(model,'tf32x3')
    ids=torch.tensor([[2,3,4,5,6,7,8,9,10,11,12,13,14]],device='cuda')
    single=compile_graph([Invocation('serving',tuple(ids[0].tolist()))])
    with torch.no_grad():
        expected=model(single)['h_final']
        cache=None;chunks=[]
        for start,stop in ((0,4),(4,9),(9,13)):
            result=native_step(model,ids[:,start:stop],cache)
            chunks.append(result['h_final']);cache=result['cache']
        torch.testing.assert_close(torch.cat(chunks,dim=1),expected,atol=5e-6,rtol=2e-5)


def test_attention_transition_rejects_without_tolerances_and_restores_failed_rng(monkeypatch):
    from natlang_neuralese.train.native_state_sft import qualify_precision_transition
    from natlang_neuralese.eval import native_attention_transition
    kwargs=dict(previous_precision='ieee',previous_attention_precision='ieee',
        checkpoint_sha256='a'*64,device='cpu')
    config={'native_linear_precision':'ieee','native_attention_precision':'tf32x3'}
    with pytest.raises(ValueError,match='declared forward and gradient tolerances'):
        qualify_precision_transition(None,None,None,config,forward_atol=None,gradient_atol=None,**kwargs)
    rng=torch.get_rng_state().clone()
    def failed(*args,**kwargs):
        torch.rand(19)
        return {'passed':False,'candidate_cached_reference':{'passed':False}}
    monkeypatch.setattr(native_attention_transition,'qualify_attention_transition',failed)
    with pytest.raises(ValueError,match='unchanged cached-reference'):
        qualify_precision_transition(None,None,None,config,forward_atol=1e-4,gradient_atol=1e-5,optimizer=object(),**kwargs)
    assert torch.equal(torch.get_rng_state(),rng)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual FP32 tensor-core attention')
@pytest.mark.parametrize('width,amplitude',[(64,1.),(128,3.)])
def test_actual_fp32_kernel_and_all_adjoints_match_fp64_oracle_and_cached_layout(monkeypatch,width,amplitude):
    from natlang_neuralese.model import shared_state as shared
    from natlang_neuralese.model.native_graph_plan import compile_graph,Invocation
    torch.manual_seed(160126+width)
    counts=(257,259);total=sum(counts)
    plan=compile_graph([Invocation(str(i),tuple(range(n))) for i,n in enumerate(counts)])
    q=(torch.randn(1,8,total,width,device='cuda')*amplitude).requires_grad_()
    k=torch.randn(1,2,total,width,device='cuda',requires_grad=True)
    v=torch.randn_like(k,requires_grad=True);upstream=torch.randn_like(q)*1e-3
    original=shared._compiled_flex;observed=[]
    def record(*args,**kwargs):
        assert kwargs['precision']=='tf32x3'
        compiled=original(*args,**kwargs)
        def execute(*args,**kwargs):
            options=kwargs['kernel_options']
            assert options['FLOAT32_PRECISION']=="'tf32x3'" and options['num_stages']==1
            observed.append(True)
            return compiled(*args,**kwargs)
        return execute
    monkeypatch.setattr(shared,'_compiled_flex',record)
    slots=plan.tensor(plan.slot_sources,'cuda')
    actual=shared.graph_attention(q,k[:,:,slots],v[:,:,slots],plan,None,width**-.5,precision='tf32x3')
    actual_grads=torch.autograd.grad((actual*upstream).sum(),(q,k,v))
    references=[];cached=[];offset=0
    dq,dk,dv=[value.detach().double().requires_grad_() for value in (q,k,v)]
    for n in counts:
        sq,sk,sv=[value[:,:,offset:offset+n] for value in (dq,dk,dv)]
        scores=sq@sk.repeat_interleave(4,dim=1).transpose(-1,-2)*width**-.5
        causal=torch.arange(n,device='cuda')[:,None]>=torch.arange(n,device='cuda')[None]
        references.append(scores.masked_fill(~causal,-torch.inf).softmax(-1)@sv.repeat_interleave(4,dim=1))
        cached.append(shared.graph_attention(q[:,:,offset:offset+n],k[:,:,offset:offset+n],v[:,:,offset:offset+n],
            shared.AttentionLayout(tuple(range(n)),n),None,width**-.5,precision='tf32x3'))
        offset+=n
    expected=torch.cat(references,dim=2);cached=torch.cat(cached,dim=2)
    reference_grads=torch.autograd.grad((expected*upstream.double()).sum(),(dq,dk,dv))
    cached_grads=torch.autograd.grad((cached*upstream).sum(),(q,k,v))
    assert len(observed)==3 and torch.equal(actual,cached)
    torch.testing.assert_close(actual.double(),expected,atol=6e-6,rtol=2e-5)
    for a,b,c in zip(actual_grads,reference_grads,cached_grads):
        assert torch.equal(a,c)
        assert float((a.double()-b).norm()/b.norm())<3e-6


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA optimizer continuation')
def test_actual_attention_transition_keeps_optimizer_rng_and_completed_exposure(tmp_path,monkeypatch):
    import json
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from test_native_shared_checkpoint import equal_tree
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    paths={}
    for split in ('train','test'):
        paths[split]=tmp_path/(split+'.jsonl')
        paths[split].write_text(json.dumps({**graph(split),'split':split,
            'tokenizer_fingerprint':config['tokenizer_fingerprint']})+'\n')
    out=tmp_path/'run'
    argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cuda',
        '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
        '--epochs','3','--eval-every','100','--out',str(out),
        '--forward-atol','1e-4','--gradient-atol','1e-5']
    def stop(policy,step,**kwargs):policy.stop.requested=True;return False
    monkeypatch.setattr(trainer.CheckpointPolicy,'due',stop)
    assert trainer.main(argv)==0
    before=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu')
    loads=[];rngs=[]
    make,restore=trainer.make_native_optimizer,trainer.restore_training_rng_state
    def observed_make(*args,**kwargs):
        optimizer,named=make(*args,**kwargs)
        original=optimizer.load_state_dict
        def observed_load(state,**options):
            loads.append(copy.deepcopy(state));return original(state,**options)
        optimizer.load_state_dict=observed_load
        return optimizer,named
    def observed_rng(state,*args):
        rngs.append(copy.deepcopy(state));return restore(state,*args)
    monkeypatch.setattr(trainer,'make_native_optimizer',observed_make)
    monkeypatch.setattr(trainer,'restore_training_rng_state',observed_rng)
    assert trainer.main(argv+['--native-attention-precision','tf32x3'])==0
    after=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu')
    assert len(loads)==1
    equal_tree(loads[0],before['training']['optimizer'])
    assert len(rngs)==2
    equal_tree(rngs[0],before['training']['rng']);equal_tree(rngs[1],rngs[0])
    assert before['training']['exposure_counts']=={'train':1}
    assert after['training']['exposure_counts']=={'train':2}
    assert after['training']['global_step']==2
    assert after['model_config']['native_attention_precision']=='tf32x3'
    assert after['interface_state']['native_attention_precision']=='tf32x3'
    event=after['training']['options']['native_attention_precision_history'][-1]
    assert event['from']=='ieee' and event['to']=='tf32x3' and event['at_global_step']==1
    assert event['reference_check']['passed']
    assert any(not torch.equal(value,after['backbone_state'][name])
               for name,value in before['backbone_state'].items())
