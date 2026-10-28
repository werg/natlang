"""Actual native gradients release without SGD; target capacity needs target runs."""
import copy
import json
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
    from natlang_neuralese.model.native_weight_carrier import active_cotangent_scope
    assert active_cotangent_scope() is None


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_capacity_observed_gradient_matches_production_shared_update(family,monkeypatch):
    from contextlib import nullcontext
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':
        model.port.hf.natlang_precision='bf16'
        for module in model.port.hf.modules():
            experts=getattr(module,'experts',None)
            if callable(getattr(experts,'make_latent',None)):experts.make_latent(quantize=False)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    batch=examples()[:1]
    observed={}
    def capacity_observer(name,gradient):observed[name]=gradient.detach().clone()
    before=copy.deepcopy(model.state_dict())
    report=qualify_capacity(model,optimizer,batch,checkpoint_layers=True,chunk_tokens=3,
        checkpoint_sha256='a'*64,source_binding={'path':'capacity-production-parity','sha256':'b'*64},
        gradient_observer=capacity_observer)
    assert report['passed'] and observed
    assert_exact(before,model.state_dict())

    # Keep the ordinary optimizer step after backward so this test can inspect
    # its exact production gradients before the optimizer consumes them. The
    # registered gated hooks remain inert outside their normal in-backward
    # context and do not alter the tested forward/backward arithmetic.
    production={}
    original_step=optimizer.step
    def capture_step(*args,**kwargs):
        production.update({name:parameter.grad.detach().clone()
            for name,parameter in optimizer._native_named_parameters if parameter.grad is not None})
        return original_step(*args,**kwargs)
    optimizer.step=capture_step
    monkeypatch.setattr(optimizer.latent,'in_backward',lambda:nullcontext())
    native_training_update(model,optimizer,batch,checkpoint_layers=True,chunk_tokens=3)
    assert observed.keys()==production.keys()
    for name in observed:torch.testing.assert_close(observed[name],production[name],rtol=0,atol=0,msg=name)


def test_capacity_rejects_unscoped_low_precision_gradient_and_closes_scope(monkeypatch):
    from natlang_neuralese.train import native_state_sft
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope,active_cotangent_scope
    model=SharedStateModel(tiny('mellum'),(58,59,60,61),rank=4)
    model.port.hf.natlang_precision='bf16'
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    parameter=next(parameter for _,parameter in optimizer._native_named_parameters if parameter.ndim>=2)
    parameter.data=parameter.detach().to(torch.bfloat16)
    scopes=[];original_factory=ParameterCotangentScope.for_model.__func__
    def capture_factory(cls,*args,**kwargs):
        scope=original_factory(cls,*args,**kwargs);scopes.append(scope);return scope
    monkeypatch.setattr(ParameterCotangentScope,'for_model',classmethod(capture_factory))
    def bypass(*unused,**kwargs):return parameter.float().square().mean(),[]
    monkeypatch.setattr(native_state_sft,'packed_graph_loss',bypass)
    with pytest.raises(RuntimeError,match='low-precision parameter gradient arrived without a cotangent carrier:'):
        qualify_capacity(model,optimizer,examples()[:1],checkpoint_sha256='a'*64,source_binding={})
    assert active_cotangent_scope() is None
    assert len(scopes)==1 and scopes[0]._closed
    assert not __import__('os').path.exists(scopes[0].directory)
    assert all(value.grad is None for value in model.parameters())


@pytest.mark.parametrize('failure',['oom','backward_oom','nonfinite','exception'])
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
        if failure=='backward_oom':
            parameter=next(parameter for parameter in model.parameters() if parameter.requires_grad)
            class OOMBackward(torch.autograd.Function):
                @staticmethod
                def forward(ctx,value):return value.float().sum()
                @staticmethod
                def backward(ctx,gradient):raise torch.cuda.OutOfMemoryError('injected backward allocation failure')
            return OOMBackward.apply(parameter),[]
        if failure=='nonfinite':return sum(p.sum() for p in model.parameters())*float('nan'),[]
        raise RuntimeError('injected kernel failure')
    monkeypatch.setattr(native_state_sft,'packed_graph_loss',bad)
    kwargs=dict(backend=backend,checkpoint_sha256='a'*64,source_binding={})
    if failure in ('oom','backward_oom'):
        result=qualify_capacity(model,optimizer,examples()[:1],**kwargs)
        assert not result['passed'] and result['cases'][0]['status']=='out_of_memory'
    else:
        with pytest.raises((RuntimeError,FloatingPointError)):
            qualify_capacity(model,optimizer,examples()[:1],**kwargs)
    assert_exact(weights,model.state_dict())
    assert backend.extra_state()==before and backend.row_counts==counts and backend.has_executed
    assert torch.equal(rng,torch.get_rng_state()) and python_rng==random.getstate()
    assert all(p.grad is None for p in model.parameters())
    from natlang_neuralese.model.native_weight_carrier import active_cotangent_scope
    assert active_cotangent_scope() is None


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


def prepared_fixture(tmp_path,monkeypatch,novel_split="train",checkpoint_layers=True,family='lfm'):
    import json
    from types import SimpleNamespace
    from pathlib import Path
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from test_native_state_sft import graph
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
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


def test_failure_repair_dpo_capacity_descriptor_uses_complete_pair_geometry(tmp_path):
    import json
    from natlang_neuralese.data.native_failure_repair import _write_admitted_rows
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train.native_preference import NativePreferenceObjective
    from test_native_failure_repair_dpo import record
    from natlang_neuralese.train.native_resident import binding
    out=tmp_path/'admission'
    _write_admitted_rows([record(),record('7'*64)],out,tokenizer_fingerprint='tok',protected_groups=set())
    objective=NativePreferenceObjective(out/'repair-objective.json')
    source=NativeGraphSource(out/'repair-sources.json','tok',required_split='train')
    first,second=list(source.iter_indices([0,1]))
    geom=lambda example:{'forward_tokens':example.plan.token_count,'kv_slots':len(example.plan.slot_sources)}
    descriptor={'schema':'natlang.native-preference-capacity-case/1','source':binding(source.path),
        'objective':objective.binding(),
        'case':{'id':first.id,'graph_sha256':objective.rows[first.id]['graph_sha256'],
            'split':'train','geometry':geom(first)},
        'numerical_reference':{'id':second.id,'graph_sha256':objective.rows[second.id]['graph_sha256'],
            'split':'train','geometry':geom(second)}}
    request_path=tmp_path/'requests'/'request.json'
    bound=memory._materialize_preference_capacity_request(source,objective,descriptor,
        identity={'checkpoint':'actual'},runtime_code_sha256='d'*64,request_path=request_path)
    request=json.loads(request_path.read_text())
    assert bound==binding(request_path)
    assert request['objective']['kind']=='native-failure-repair-dpo'
    assert request['cases'][0]['id']==first.id and request['reference']['id']==second.id


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


def test_default_cotangent_backend_preserves_legacy_capacity_identity(monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.eval import native_memory as memory
    monkeypatch.setattr(memory,'memory_runtime',lambda device:{'device':device})
    args=SimpleNamespace(batch_graphs=1,batch_tokens=65536,checkpoint_layers=True,
        chunk_tokens=1024,checkpoint_input_offload=False,device='cpu',
        fp8_training='off',fp8_recipe='delayed',fp8_amax_history_len=16)
    legacy=memory.capacity_identity({},args,None)
    assert 'cotangent_accumulator_backend' not in legacy
    args.cotangent_accumulator_backend='cpu_file'
    assert memory.capacity_identity({},args,None)==legacy
    args.cotangent_accumulator_backend='device_fp32'
    device_identity=memory.capacity_identity({},args,None)
    assert device_identity['cotangent_accumulator_backend']=='device_fp32'
    assert {key:value for key,value in device_identity.items()
            if key!='cotangent_accumulator_backend'}==legacy



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


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_identical_regular_request_aliases_measure_once_and_recheck_each_declaration(tmp_path,monkeypatch,family):
    import json
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train import native_resident as resident
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch,family=family)
    native_training_update(model,optimizer,list(train.iter_indices([0])),checkpoint_layers=True,chunk_tokens=3)
    optimizer.zero_grad(set_to_none=True)
    first=resident.bound(json.loads(args.memory_qualification));second=tmp_path/'pending-long-0.json'
    second.write_bytes(first.read_bytes())
    assert first.is_file() and second.is_file() and not first.is_symlink() and not second.is_symlink()
    requests=[resident.binding(first),resident.binding(second)];args.memory_qualification=json.dumps(requests)
    before=copy.deepcopy(model.state_dict()),copy.deepcopy(optimizer.state_dict()),torch.get_rng_state().clone(),random.getstate()
    resolved=[];resolve=memory._resolve_memory_qualification
    def resolve_each(reference,**kwargs):
        value=resolve(reference,**kwargs);resolved.append(value);return value
    monkeypatch.setattr(memory,'_resolve_memory_qualification',resolve_each)
    checks=[];run=memory._run_one_prepared
    def check_each(*arguments,**kwargs):
        checks.append(json.loads(args.memory_qualification));return run(*arguments,**kwargs)
    monkeypatch.setattr(memory,'_run_one_prepared',check_each)
    selected=json.loads(first.read_text())['source'];measure=memory.qualify_capacity;measured=[]
    def actual(*arguments,**kwargs):
        if kwargs.get('source_binding')==selected:
            assert len(resolved)==2
            measured.append(1)
        return measure(*arguments,**kwargs)
    monkeypatch.setattr(memory,'qualify_capacity',actual)
    result=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert result['passed'] and len(result['checks'])==2 and len(measured)==1
    assert checks==[requests[0],requests[0]]
    assert result['request']==requests
    assert [item['resolved_binding']for item in result['request_aliases']]==requests
    assert all(item['canonical_binding']==requests[0]for item in result['request_aliases'])
    assert all(check['request']==requests[0]for check in result['checks'])
    certificate=json.loads(args.resident_geometry_qualification.read_text())
    combined=json.loads(resident.bound(certificate['evidence']).read_text())
    assert len(combined['proofs'])==len(combined['capacity']['cases'])==1
    # Reordering aliases retains the immutable original proof, with no remeasurement.
    args.memory_qualification=json.dumps(list(reversed(requests)))
    again=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert again['passed'] and len(measured)==1 and len(checks)==4
    assert all(check['request']==requests[0]for check in again['checks'])
    assert first.read_bytes()==second.read_bytes() and not second.is_symlink()
    assert_exact(before[0],model.state_dict());assert_exact(before[1],optimizer.state_dict())
    assert torch.equal(before[2],torch.get_rng_state()) and before[3]==random.getstate()


def test_request_alias_changed_after_resolution_fails_before_cached_check(tmp_path,monkeypatch):
    import json
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train.native_resident import binding,bound
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch)
    first=bound(json.loads(args.memory_qualification));second=tmp_path/'alias.json';second.write_bytes(first.read_bytes())
    args.memory_qualification=json.dumps([binding(first),binding(second)])
    measure=memory.qualify_capacity;selected=json.loads(first.read_text())['source'];calls=[]
    def change_alias(*arguments,**kwargs):
        result=measure(*arguments,**kwargs)
        if kwargs.get('source_binding')==selected:
            calls.append(1);second.write_bytes(b'changed logical request after its source was resolved')
        return result
    monkeypatch.setattr(memory,'qualify_capacity',change_alias)
    with pytest.raises(ValueError,match='resident artifact changed'):
        memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert len(calls)==1


def test_capacity_oom_retains_failing_phase_stack_and_actual_offload_state(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_state_sft
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    hook=configure_checkpoint_input_offload(model.port,enabled=True,directory=tmp_path/'activation-files')
    before=copy.deepcopy(model.state_dict());history=copy.deepcopy(optimizer.state_dict())
    rng=torch.get_rng_state().clone()
    def failing_actual_forward(*args,**kwargs):
        raise torch.cuda.OutOfMemoryError('fixture allocation failure in forward')
    monkeypatch.setattr(native_state_sft,'packed_graph_loss',failing_actual_forward)
    report=qualify_capacity(model,optimizer,examples()[:1],checkpoint_layers=True,
        checkpoint_sha256='a'*64,source_binding={'path':'actual-fixture','sha256':'b'*64})
    assert not report['passed'] and report['optimizer_updates']==0
    case=report['cases'][0]
    assert case['failure_phase']=='forward' and case['status']=='out_of_memory'
    assert 'failing_actual_forward' in case['failure_traceback']
    assert 'fixture allocation failure in forward' in case['failure_traceback']
    assert case['checkpoint_input_offload']==hook.snapshot()
    assert_exact(before,model.state_dict());assert_exact(history,optimizer.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())


def test_sft_descriptor_qualifies_actual_loaded_source_without_updates(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train.native_packed_capacity import geometry_for_source
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch)
    _, geometry, _ = geometry_for_source(train, args.out, allow_build=True)
    descriptor = {'schema': 'natlang.native-sft-capacity-cases/1', 'source': resident.binding(train.path),
                  'geometry': geometry}
    path = tmp_path / 'sft-descriptor.json'; path.write_text(json.dumps(descriptor))
    args.memory_qualification = json.dumps(resident.binding(path))
    before=copy.deepcopy(optimizer.state_dict()); weights=copy.deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
    result=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert result['passed'] and result['optimizer_updates']==0
    assert result['capacity']['cases'][0]['id']==train.entries[0]['id']
    assert args.resident_geometry_qualification.is_file()
    assert_exact(before,optimizer.state_dict()); assert_exact(weights,model.state_dict()); assert torch.equal(rng,torch.get_rng_state())


def test_sft_descriptor_rejects_wrong_source_and_geometry(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train.native_packed_capacity import geometry_for_source
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch)
    _, geometry, _ = geometry_for_source(train, args.out, allow_build=True)
    descriptor={'schema':'natlang.native-sft-capacity-cases/1','source':resident.binding(train.path),'geometry':geometry}
    descriptor['source']={'path':str(tmp_path/'wrong'),'sha256':'0'*64}
    with pytest.raises(ValueError,match='descriptor source differs'):
        memory._materialize_sft_capacity_request(train, descriptor, identity={}, runtime_code_sha256='x', request_path=tmp_path/'r')
    descriptor['source']=resident.binding(train.path)
    damaged=json.loads(resident.bound(geometry).read_text()); damaged['kv_slots'][0] += 1
    bad=tmp_path/'damaged.json'; bad.write_text(json.dumps(damaged)); descriptor['geometry']=resident.binding(bad)
    with pytest.raises(ValueError,match='geometry'):
        memory._materialize_sft_capacity_request(train, descriptor, identity={}, runtime_code_sha256='x', request_path=tmp_path/'r')
