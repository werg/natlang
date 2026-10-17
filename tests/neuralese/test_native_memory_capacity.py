"""Actual native gradients release without SGD; target capacity needs target runs."""
import copy
import random

import pytest
import torch

from natlang_neuralese.eval.native_memory import qualify_capacity
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
from natlang_neuralese.common.checkpoint_offload import configure_checkpoint_input_offload
from test_shared_state import tiny
from test_native_optimizer_checkpointing import examples,assert_exact


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('offload',[False,True])
def test_actual_capacity_check_releases_gradients_and_preserves_full_state(family,offload):
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    native_training_update(model,optimizer,examples()[:1],checkpoint_layers=True,chunk_tokens=3)
    optimizer.zero_grad(set_to_none=True)
    configure_checkpoint_input_offload(model.port,enabled=offload)
    weights=copy.deepcopy(model.state_dict());port=copy.deepcopy(model.port.hf.state_dict())
    state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone();python_rng=random.getstate()
    hook_counts={name:len(value._post_accumulate_grad_hooks or {}) for name,value in model.named_parameters()}
    model.train();model.adapter.eval()
    modes={module:module.training for module in model.modules()}
    report=qualify_capacity(model,optimizer,examples()[:2],checkpoint_layers=True,chunk_tokens=3,
        checkpoint_sha256='a'*64,source_binding={'path':'actual-control-fixture','sha256':'b'*64})
    assert report['passed'] and report['optimizer_updates']==0
    assert report['gradients_released_per_parameter']
    assert all(row['gradients'] and row['kv_slots']>=row['forward_tokens'] for row in report['cases'])
    assert_exact(weights,model.state_dict());assert_exact(port,model.port.hf.state_dict())
    assert_exact(state,optimizer.state_dict())
    assert torch.equal(rng,torch.get_rng_state()) and python_rng==random.getstate()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(module.training==mode for module,mode in modes.items())
    assert hook_counts=={name:len(value._post_accumulate_grad_hooks or {}) for name,value in model.named_parameters()}


@pytest.mark.parametrize('failure',['oom','nonfinite','exception'])
def test_capacity_failure_restores_hooks_backend_rng_and_existing_state(monkeypatch,failure):
    from natlang_neuralese.train import native_state_sft
    from test_native_backend_qualification import Backend
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    backend=Backend();before=copy.deepcopy(backend.extra_state());counts=dict(backend.row_counts)
    weights=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone();python_rng=random.getstate()
    def bad(*args,**kwargs):
        assert backend.active
        torch.rand(3);random.random();backend.state['scale'][0]=99;backend.row_counts['calls']=100
        if failure=='oom':raise torch.cuda.OutOfMemoryError('injected target allocation failure')
        if failure=='nonfinite':return sum(p.sum() for p in model.parameters())*float('nan'),[]
        raise RuntimeError('injected kernel failure')
    monkeypatch.setattr(native_state_sft,'packed_graph_loss',bad)
    kwargs=dict(backend=backend,checkpoint_sha256='a'*64,source_binding={})
    if failure=='oom':
        result=qualify_capacity(model,optimizer,examples()[:1],**kwargs)
        assert not result['passed'] and result['cases'][0]['status']=='out_of_memory'
    else:
        with pytest.raises((RuntimeError,FloatingPointError)):
            qualify_capacity(model,optimizer,examples()[:1],**kwargs)
    assert_exact(weights,model.state_dict())
    assert backend.extra_state()==before and backend.row_counts==counts and backend.has_executed
    assert torch.equal(rng,torch.get_rng_state()) and python_rng==random.getstate()
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('checkpoint_layers',[True,False])
def test_offload_small_actual_weight_comparison_uses_bitwise_all_gradient_digests(family,checkpoint_layers):
    from natlang_neuralese.eval.native_memory import offload_numerical_parity
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    hook=configure_checkpoint_input_offload(model.port,enabled=True)
    report=offload_numerical_parity(model,optimizer,examples()[0],chunk_tokens=3,
        checkpoint_sha256='a'*64,source_binding={},checkpoint_layers=checkpoint_layers)
    assert report['candidate_checkpoint_layers'] is checkpoint_layers
    assert report['passed'] and report['gradient_parameters']>0
    assert not report['different_gradient_names'] and not report['missing_gradient_names']
    assert model.port.activation_checkpoint is hook


def prepared_fixture(tmp_path,monkeypatch,novel_split="train",checkpoint_layers=True):
    import json
    from types import SimpleNamespace
    from pathlib import Path
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from test_native_state_sft import graph
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    configure_checkpoint_input_offload(model.port,enabled=checkpoint_layers)
    sources=[]
    for name in ('active','novel'):
        path=tmp_path/(name+'.jsonl');row=graph(name);split=novel_split if name=='novel' else 'train';row['split']=split;path.write_text(json.dumps(row)+'\n')
        sources.append(NativeGraphSource(path,'tokenizer-sha',required_split=split,role_ids=(58,59,60,61)))
    active,novel=sources
    args=SimpleNamespace(out=tmp_path/'out',checkpoint_input_offload=checkpoint_layers,checkpoint_layers=checkpoint_layers,
        chunk_tokens=3,batch_tokens=32,batch_graphs=1,device='cpu',fp8_training='off',fp8_recipe='delayed',
        fp8_amax_history_len=16,resident_geometry_qualification=None)
    args.out.mkdir();config={'tokenizer_fingerprint':'tokenizer-sha'}
    def row(source):
        return {'index':0,'id':source.entries[0]['id'],'binding':source.entries[0]['binding']}
    request={'schema':'natlang.native-memory-qualification-request/1',
        'identity':memory.capacity_identity(config,args,None),
        'runtime_code_sha256':resident.digest(resident.code_inventory(Path(memory.__file__).parents[1])),
        'source':resident.binding(novel.path),'indexes':resident._source_witness(novel),
        'cases':[row(novel)],'reference':row(active),'split':novel_split}
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    args.memory_qualification=json.dumps(resident.binding(path))
    return model,optimizer,active,SimpleNamespace(ids=(58,59,60,61)),config,args


def test_prepared_actual_whole_graph_check_recovers_certificate_and_reuses_immutable_proof(tmp_path,monkeypatch):
    from natlang_neuralese.eval import native_memory as memory
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch)
    before=copy.deepcopy(optimizer.state_dict());weights=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    result=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert result['passed'] and result['capacity']['cases'][0]['id']=='novel'
    certificate=args.resident_geometry_qualification;proof=certificate.parent/'proof.json';original=proof.read_bytes()
    certificate.unlink();args.resident_geometry_qualification=None
    monkeypatch.setattr(memory,'qualify_capacity',lambda *a,**k:pytest.fail('must reuse proof without GPU work'))
    again=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert again==result and certificate.exists() and args.resident_geometry_qualification==certificate
    assert proof.read_bytes()==original
    assert_exact(before,optimizer.state_dict());assert_exact(weights,model.state_dict());assert torch.equal(rng,torch.get_rng_state())
    # No mutable overwriting of a content-addressed proof/certificate.
    certificate.write_text('{}')
    with pytest.raises(ValueError,match='immutable memory capacity certificate'):
        memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)


def test_failed_novel_capacity_keeps_existing_sgd_and_never_qualifies_new_source(tmp_path,monkeypatch):
    from natlang_neuralese.eval import native_memory as memory
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(memory,'offload_numerical_parity',lambda *a,**k:{'passed':True})
    monkeypatch.setattr(memory,'qualify_capacity',lambda *a,**k:(_ for _ in ()).throw(torch.cuda.OutOfMemoryError('actual target allocation')))
    result=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert not result['passed'] and args.resident_geometry_qualification is None
    assert not list(args.out.rglob('capacity.json'))
    # Existing useful source still makes a real optimizer update after the failed probe.
    model_before=copy.deepcopy(model.state_dict())
    native_training_update(model,optimizer,list(train.iter_indices([0])),checkpoint_layers=True,chunk_tokens=3)
    assert any(not torch.equal(value,model.state_dict()[name]) for name,value in model_before.items() if torch.is_tensor(value))


def test_backend_restore_failure_is_fatal_and_rng_is_still_restored(monkeypatch):
    from natlang_neuralese.eval.native_memory import CapacityStateChangedError
    from test_native_backend_qualification import Backend
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True);backend=Backend()
    before=torch.get_rng_state().clone()
    monkeypatch.setattr(backend,'restore_extra_state',lambda *a:(_ for _ in ()).throw(RuntimeError('cannot restore amax')))
    with pytest.raises(CapacityStateChangedError,match='restore backend/RNG'):
        qualify_capacity(model,optimizer,examples()[:1],backend=backend,chunk_tokens=3,
            checkpoint_sha256='a'*64,source_binding={})
    assert torch.equal(before,torch.get_rng_state())


def test_memory_request_is_strict_bound_recipe_metadata():
    from natlang_neuralese.train.recipe import validate_stage_parameters,stage_parameter_args
    good={'path':'/prepared/request.json','sha256':'a'*64}
    validate_stage_parameters('native_state_sft',{'memory_qualification':good})
    assert '--memory-qualification' in stage_parameter_args({'memory_qualification':good})
    for bad in ('unbound',{'path':'a','sha256':'bad'},{**good,'bytes':23}):
        with pytest.raises(ValueError,match='exact path/sha256'):
            validate_stage_parameters('native_state_sft',{'memory_qualification':bad})


def test_preference_capacity_descriptor_resolves_from_live_source_and_objective(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train import native_resident as resident

    manifest=tmp_path/'train-sources.json';manifest.write_text('{}\n')
    source=SimpleNamespace(path=manifest,entries=[{'id':'pair-id','binding':{'sha256':'b'*64}},
        {'id':'small-reference','binding':{'sha256':'e'*64}}])
    examples_by_index={0:SimpleNamespace(id='pair-id',split='train',plan=SimpleNamespace(token_count=94071,slot_sources=(1,)*94071)),
        1:SimpleNamespace(id='small-reference',split='train',plan=SimpleNamespace(token_count=12,slot_sources=(1,)*10))}
    source.iter_indices=lambda indices: iter([examples_by_index[index] for index in indices])
    monkeypatch.setattr(resident,'_source_witness',lambda current:{'source':resident.binding(manifest),'indexes':[]})

    objective=SimpleNamespace(kind='native-offline-pairwise',rows={'pair-id':{'graph_sha256':'b'*64},
        'small-reference':{'graph_sha256':'e'*64}},
        binding=lambda:{'kind':'native-offline-pairwise','sha256':'c'*64,'beta':0.1,'margin':0.0},
        forward_inputs=lambda context:{})
    descriptor={'schema':'natlang.native-preference-capacity-case/1',
        'source':resident.binding(manifest),'objective':objective.binding(),
        'case':{'id':'pair-id','graph_sha256':'b'*64,'split':'train',
                'geometry':{'forward_tokens':94071,'kv_slots':94071}},
        'numerical_reference':{'id':'small-reference','graph_sha256':'e'*64,'split':'train',
                'geometry':{'forward_tokens':12,'kv_slots':10}}}
    request_path=tmp_path/'requests'/'request.json'
    result=memory._materialize_preference_capacity_request(source,objective,descriptor,
        identity={'checkpoint':'actual'},runtime_code_sha256='d'*64,request_path=request_path)
    request=json.loads(request_path.read_text())
    assert result==resident.binding(request_path)
    assert request['identity']=={'checkpoint':'actual'}
    assert request['runtime_code_sha256']=='d'*64
    assert request['cases']==[{'index':0,'id':'pair-id','binding':source.entries[0]['binding']}]
    assert request['reference']=={'index':1,'id':'small-reference','binding':source.entries[1]['binding']}
    assert request['objective']==objective.binding()


@pytest.mark.parametrize('change',[
    {'source':{'path':'/elsewhere','sha256':'e'*64}},
    {'objective':{'kind':'native-offline-pairwise','sha256':'f'*64,'beta':0.1,'margin':0.0}},
    {'case':{'id':'missing-id','binding':{'sha256':'b'*64},'split':'train',
             'geometry':{'forward_tokens':94071,'kv_slots':94071}}},
    {'case':{'id':'pair-id','graph_sha256':'e'*64,'split':'train',
             'geometry':{'forward_tokens':94071,'kv_slots':94071}}},
    {'case':{'id':'pair-id','graph_sha256':'b'*64,'split':'train',
             'geometry':{'forward_tokens':94070,'kv_slots':94071}}},
    {'case':{'id':'pair-id','graph_sha256':'b'*64,'split':'test',
             'geometry':{'forward_tokens':94071,'kv_slots':94071}}},
    {'numerical_reference':{'id':'missing-id','graph_sha256':'b'*64,'split':'train',
             'geometry':{'forward_tokens':94071,'kv_slots':94071}}},
    {'numerical_reference':{'id':'pair-id','graph_sha256':'f'*64,'split':'train',
             'geometry':{'forward_tokens':94071,'kv_slots':94071}}},
])
def test_preference_capacity_descriptor_fails_closed_on_changed_bindings(tmp_path,monkeypatch,change):
    from types import SimpleNamespace
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train import native_resident as resident
    manifest=tmp_path/'train-sources.json';manifest.write_text('{}\n')
    source=SimpleNamespace(path=manifest,entries=[{'id':'pair-id','binding':{'sha256':'b'*64}}])
    example=SimpleNamespace(id='pair-id',split='train',plan=SimpleNamespace(token_count=94071,slot_sources=(1,)*94071))
    source.iter_indices=lambda indices: iter([example] if indices==[0] else [])
    monkeypatch.setattr(resident,'_source_witness',lambda current:{'indexes':[]})
    objective=SimpleNamespace(kind='native-offline-pairwise',rows={'pair-id':{'graph_sha256':'b'*64}},
        binding=lambda:{'kind':'native-offline-pairwise','sha256':'c'*64,'beta':0.1,'margin':0.0},
        forward_inputs=lambda context:{})
    descriptor={'schema':'natlang.native-preference-capacity-case/1','source':resident.binding(manifest),
        'objective':objective.binding(),'case':{'id':'pair-id','graph_sha256':'b'*64,
        'split':'train','geometry':{'forward_tokens':94071,'kv_slots':94071}},
        'numerical_reference':{'id':'pair-id','graph_sha256':'b'*64,
        'split':'train','geometry':{'forward_tokens':94071,'kv_slots':94071}}}
    descriptor.update(change)
    with pytest.raises(ValueError):
        memory._materialize_preference_capacity_request(source,objective,descriptor,
            identity={},runtime_code_sha256='d'*64,request_path=tmp_path/'request.json')
    assert not (tmp_path/'request.json').exists()


def test_opt_in_preference_descriptor_uses_standard_runner_then_restores_argument(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.eval import native_memory as memory
    descriptor={'schema':'natlang.native-preference-capacity-case/1','path':'descriptor'}
    resolved={'path':'/prepared/request.json','sha256':'a'*64}
    args=SimpleNamespace(memory_qualification=json.dumps(descriptor))
    seen=[]
    monkeypatch.setattr(memory,'_resolve_memory_qualification',lambda reference,**kwargs:resolved)
    def run(*unused,**kwargs):
        seen.append(json.loads(args.memory_qualification))
        return {'passed':True}
    monkeypatch.setattr(memory,'_run_one_prepared',run)
    result=memory.run_prepared(None,None,None,None,{},args,backend=None,checkpoint_sha256='b'*64)
    assert result=={'passed':True}
    assert seen==[resolved]
    assert json.loads(args.memory_qualification)==descriptor


def test_default_capacity_check_retains_qualified_nonprefix_packer_signature(monkeypatch):
    from natlang_neuralese.train import native_state_sft
    original=native_state_sft.packed_graph_loss
    def qualified(model,rows,*,checkpoint_layers=True,chunk_tokens=128):
        return original(model,rows,checkpoint_layers=checkpoint_layers,chunk_tokens=chunk_tokens)
    monkeypatch.setattr(native_state_sft,'packed_graph_loss',qualified)
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    result=qualify_capacity(model,optimizer,examples()[:1],chunk_tokens=3,
        checkpoint_sha256='a'*64,source_binding={})
    assert result['passed']


def test_later_novel_oom_retains_independently_qualified_cohort_capacity(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train.native_resident import binding
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch)
    first=json.loads(args.memory_qualification);second_path=tmp_path/'second-request.json'
    second_path.write_text(json.dumps({**json.loads(Path(first['path']).read_text()),
        'cohort':'second exact source cohort'}))
    second=binding(second_path)
    args.memory_qualification=json.dumps([first,second]);original=args.memory_qualification;seen=[]
    def cohort(*a,**kwargs):
        seen.append(json.loads(args.memory_qualification))
        if len(seen)==1:args.resident_geometry_qualification=tmp_path/'qualified-smaller-cohort.json'
        return {'passed':len(seen)==1,'request':seen[-1],'parent_checkpoint_sha256':'a'*64}
    monkeypatch.setattr(memory,'_run_one_prepared',cohort)
    result=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert not result['passed'] and result['checks'][0]['passed'] and not result['checks'][1]['passed']
    assert result['retained_capacity'].endswith('qualified-smaller-cohort.json')
    assert args.memory_qualification==original and seen==[first,second]
    from natlang_neuralese.train.recipe import validate_stage_parameters
    validate_stage_parameters('native_state_sft',{'memory_qualification':[first,second]})
    from natlang_neuralese.train.recipe import stage_parameter_args
    encoded=stage_parameter_args({'memory_qualification':[first,second]})
    assert encoded[0]=='--memory-qualification' and json.loads(encoded[1])==[first,second]


def test_held_capacity_check_is_explicit_read_only_and_preserves_all_training_state(tmp_path,monkeypatch):
    from natlang_neuralese.eval import native_memory as memory
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch,novel_split='test')
    before=copy.deepcopy(optimizer.state_dict());weights=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    result=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert result['passed'] and result['capacity']['cases'][0]['id']=='novel'
    assert result['optimizer_updates']==0
    assert_exact(before,optimizer.state_dict());assert_exact(weights,model.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_combined_successful_case_envelopes_preserve_both_dimensions_and_bind_parent(tmp_path):
    import json
    from natlang_neuralese.eval.native_memory import combine_capacity_proofs
    from natlang_neuralese.train.native_resident import binding
    identity={'actual':'runtime'};proofs=[]
    for i,(tokens,slots) in enumerate([(112008,124339),(96447,173589)]):
        path=tmp_path/f'proof-{i}.json'
        path.write_text(json.dumps({'passed':True,'identity':identity,'runtime_code_sha256':'c'*64,
            'parent_checkpoint_sha256':'a'*64,'optimizer_updates':0,'numerical_reference':{'passed':True},
            'capacity':{'passed':True,'cases':[{'status':'passed','forward_tokens':tokens,'kv_slots':slots}]}}))
        proofs.append(binding(path))
    certificate=combine_capacity_proofs(tmp_path,proofs,identity=identity,code_sha='c'*64,checkpoint_sha256='a'*64)
    value=json.loads(certificate.read_text());assert value['limits']=={'forward_tokens':112008,'kv_slots':173589}
    before=certificate.read_bytes()
    assert combine_capacity_proofs(tmp_path,proofs,identity=identity,code_sha='c'*64,checkpoint_sha256='a'*64)==certificate
    assert certificate.read_bytes()==before
    with pytest.raises(ValueError,match='unchanged-parent'):
        combine_capacity_proofs(tmp_path,proofs,identity=identity,code_sha='c'*64,checkpoint_sha256='b'*64)
    path=tmp_path/'proof-1.json';row=json.loads(path.read_text());row['passed']=False;path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='unchanged-parent'):
        combine_capacity_proofs(tmp_path,[proofs[0],binding(path)],identity=identity,code_sha='c'*64,checkpoint_sha256='a'*64)


@pytest.mark.parametrize('checkpoint_layers',[True,False])
def test_actual_two_capacity_cohorts_publish_one_bound_case_union_without_optimizer_updates(tmp_path,monkeypatch,checkpoint_layers):
    import json
    from pathlib import Path
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train.native_resident import binding
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch,checkpoint_layers=checkpoint_layers)
    first=json.loads(args.memory_qualification);second=tmp_path/'second.json'
    second.write_text(json.dumps({**json.loads(Path(first['path']).read_text()),'cohort':'second exact source cohort'}))
    args.memory_qualification=json.dumps([first,binding(second)])
    weights=copy.deepcopy(model.state_dict());moments=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    report=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert report['passed'] and len(report['checks'])==2
    capacity=json.loads(args.resident_geometry_qualification.read_text())
    proof=json.loads(Path(capacity['evidence']['path']).read_text())
    assert len(proof['proofs'])==2 and len(proof['capacity']['cases'])==2
    assert all(case['status']=='passed' for case in proof['capacity']['cases'])
    assert_exact(weights,model.state_dict());assert_exact(moments,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_noncheckpointed_numerical_failure_requires_explicit_exact_state_fallback(tmp_path,monkeypatch):
    from natlang_neuralese.eval import native_memory as memory
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch,checkpoint_layers=False)
    monkeypatch.setattr(memory,'offload_numerical_parity',lambda *a,**k:{'passed':False})
    with pytest.raises(ValueError,match='checkpointed fallback required'):
        memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    with pytest.raises(ValueError,match='checkpointed fallback required'):
        memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)


def test_capacity_identity_binds_exact_causal_prefix_packing_mode(monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.eval import native_memory as memory
    monkeypatch.setattr(memory,'memory_runtime',lambda device:{'device':device})
    args=SimpleNamespace(batch_graphs=1,batch_tokens=65536,checkpoint_layers=True,
        chunk_tokens=1024,checkpoint_input_offload=False,device='cpu',
        fp8_training='off',fp8_recipe='delayed',fp8_amax_history_len=16)
    legacy=memory.capacity_identity({},args,None)
    assert 'share_causal_prefixes' not in legacy
    args.share_causal_prefixes=False
    assert memory.capacity_identity({},args,None)==legacy
    args.share_causal_prefixes=True
    shared=memory.capacity_identity({},args,None)
    assert shared['share_causal_prefixes'] is True and shared!=legacy
    assert {k:v for k,v in shared.items() if k!='share_causal_prefixes'}=={k:v for k,v in legacy.items() if k!='share_causal_prefixes'}



@pytest.mark.parametrize('retained_smaller',[False,True])
def test_actual_sft_needs_capacity_but_keeps_smaller_pass_after_larger_failure(tmp_path,monkeypatch,retained_smaller):
    import json
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.eval import native_memory as memory
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl';fp=config['tokenizer_fingerprint']
    train.write_text(json.dumps(dict(graph('train'),tokenizer_fingerprint=fp))+'\n')
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    updates=[];observed={};original=trainer.native_training_update
    def update(*args,**kwargs):
        updates.append(True);return original(*args,**kwargs)
    def failed_frontier(model,optimizer,source,roles,config,args,**kwargs):
        observed['model']=model;observed['weights']=copy.deepcopy(model.state_dict())
        observed['optimizer']=optimizer;observed['moments']=copy.deepcopy(optimizer.state_dict())
        report={'passed':False,'request':json.loads(args.memory_qualification),
            'parent_checkpoint_sha256':'a'*64,'checks':[{'passed':retained_smaller},{'passed':False}]}
        (args.out/'memory-qualification.json').write_text(json.dumps(report))
        if retained_smaller:
            certificate=tmp_path/'qualified-smaller.json';certificate.write_text(json.dumps({'passed':True,'scope':'injected retained capacity result; caller regression only'}))
            args.resident_geometry_qualification=certificate
        return report
    monkeypatch.setattr(memory,'run_prepared',failed_frontier)
    monkeypatch.setattr(trainer,'native_training_update',update)
    out=tmp_path/'out';args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(train),'--held-native-graphs',str(held),'--epochs','1',
        '--eval-every','100','--out',str(out),'--memory-qualification',json.dumps({'path':'injected-frontiers','sha256':'b'*64})]
    if retained_smaller:
        assert trainer.main(args)==0 and len(updates)==1
        report=json.loads((out/'report.json').read_text())
        assert report['metrics'] and (out/'checkpoint.pt').is_file()
    else:
        with pytest.raises(ValueError,match='no usable physical capacity'):
            trainer.main(args)
        assert updates==[] and (out/'memory-qualification.json').is_file()
        assert_exact(observed['weights'],observed['model'].state_dict())
        assert_exact(observed['moments'],observed['optimizer'].state_dict())
