import contextlib,json
from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_packed_capacity import _rows_digest, oversize_baseline, qualify_oversize_batches


def test_oversize_baseline_authenticates_exact_prefix(tmp_path):
    sourcefile=tmp_path/'old.json';sourcefile.write_text('{}')
    entries=[{'id':'old','binding':'a'*64,'forward_tokens':7},{'id':'new','binding':'b'*64,'forward_tokens':50}]
    geometry={'identity':{'parser':'pinned'},'source':{'path':str(sourcefile),'sha256':sha256_file_hex(sourcefile)},'kv_slots':[9],
              'rows_sha256':_rows_digest(entries[:1])}
    p=tmp_path/'geometry.json';p.write_text(json.dumps(geometry))
    result=oversize_baseline(SimpleNamespace(entries=entries),{'path':str(p),'sha256':sha256_file_hex(p)},fallback_tokens=3)
    assert result['forward_tokens']==7 and result['kv_slots']==9 and result['prefix_rows']==1
    changed=tmp_path/'changed.json';changed.write_text(json.dumps({'identity':{'parser':'different'}}))
    with pytest.raises(ValueError,match='parser or tokenizer'):
        oversize_baseline(SimpleNamespace(entries=entries),{'path':str(p),'sha256':sha256_file_hex(p)},fallback_tokens=3,current_geometry={'path':str(changed),'sha256':sha256_file_hex(changed)})
    entries[0]['binding']='c'*64
    with pytest.raises(ValueError,match='exact source prefix'):
        oversize_baseline(SimpleNamespace(entries=entries),{'path':str(p),'sha256':sha256_file_hex(p)},fallback_tokens=3)


@pytest.mark.parametrize('with_objective', [False, True])
def test_oversize_gate_uses_actual_point_and_restores_prior_gradients(tmp_path, monkeypatch, with_objective):
    from natlang_neuralese.eval import native_memory
    calls=[]
    def fake(model,optimizer,cases,**kwargs):
        calls.append(kwargs)
        for _,parameter in optimizer._native_named_parameters:assert parameter.grad is None
        return {'passed':True,'optimizer_updates':0}
    monkeypatch.setattr(native_memory,'qualify_capacity',fake)
    parameter=torch.nn.Parameter(torch.ones(2));parameter.grad=torch.tensor([3.,4.]);gradient=parameter.grad
    model=torch.nn.Linear(2,2);optimizer=SimpleNamespace(_native_named_parameters=[('p',parameter)])
    example=SimpleNamespace(plan=SimpleNamespace(token_count=100,slot_sources=range(101)))
    source=tmp_path/'source.json';source.write_text('{}')
    args=SimpleNamespace(cotangent_accumulator_backend='plain',checkpoint_layers=True,chunk_tokens=16,accumulate_batches=16,out=tmp_path)
    precision=(object(),'fp8',0.06,1.5)
    objective=object() if with_objective else None
    objective_binding={'kind':'bound-policy','sha256':'b'*64} if with_objective else None
    ref=qualify_oversize_batches(model,optimizer,[[example]],args=args,backend=None,precision=precision,
        source_path=source,checkpoint_sha256='a'*64,global_step=62,baseline={'forward_tokens':10,'kv_slots':11},
        objective=objective,objective_binding=objective_binding)
    assert ref and calls[0]['cotangent_mode']=='plain' and calls[0]['precision'] is precision
    assert calls[0]['retained_gradients'] and parameter.grad is gradient
    assert calls[0]['objective'] is objective
    proof=json.loads(open(ref['path']).read())
    assert proof['optimizer_updates']==0 and proof['objective']==objective_binding
    assert qualify_oversize_batches(model,optimizer,[[example]],args=args,backend=None,precision=precision,
        source_path=source,checkpoint_sha256='a'*64,global_step=62,baseline={'forward_tokens':100,'kv_slots':101}) is None


def test_capacity_uses_selected_precision_context_and_plain_backend(monkeypatch):
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples,assert_exact
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.eval.native_memory import qualify_capacity
    from natlang_neuralese.train import native_state_sft
    import copy
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    before=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    flag={'active':False,'observed':False};original=native_state_sft.packed_graph_loss
    class Qat:
        @contextlib.contextmanager
        def context(self,point,mix):
            assert point=='fp8' and mix==0.2
            flag['active']=True
            try:yield
            finally:flag['active']=False
    def checked(*args,**kwargs):
        assert flag['active'] and kwargs['allow_communication_prefixes']
        flag['observed']=True
        return original(*args,**kwargs)
    monkeypatch.setattr(native_state_sft,'packed_graph_loss',checked)
    report=qualify_capacity(model,optimizer,examples()[:1],checkpoint_sha256='a'*64,source_binding={},
        cotangent_mode='plain',precision=(Qat(),'fp8',0.2,1.5),production_packing=True)
    assert report['passed'] and flag['observed'] and report['qat_point']['point']=='fp8'
    assert_exact(before,model.state_dict());assert_exact(state,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_capacity_with_real_inline_quantization_preserves_state():
    import copy
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples,assert_exact
    from test_quantization import component
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.train.quantization import Quantization
    from natlang_neuralese.eval.native_memory import qualify_capacity
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    qat=Quantization(component());assert qat.install(model,inline=True)
    before=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict());qat_state=copy.deepcopy(qat.state_dict());rng=torch.get_rng_state().clone()
    result=qualify_capacity(model,optimizer,examples()[:1],checkpoint_sha256='a'*64,source_binding={},
        cotangent_mode='plain',precision=(qat,'q4',0.5,1.5),production_packing=True)
    assert result['passed'] and result['qat_point']=={'point':'q4','mix':0.5,'loss_weight':1.5}
    assert_exact(before,model.state_dict());assert_exact(state,optimizer.state_dict());assert_exact(qat_state,qat.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_failed_oversize_gate_preserves_gradients_and_records_no_update(tmp_path, monkeypatch):
    from natlang_neuralese.eval import native_memory
    def fail(*args,**kwargs):return {'passed':False,'optimizer_updates':0,'cases':[{'status':'out_of_memory'}]}
    monkeypatch.setattr(native_memory,'qualify_capacity',fail)
    parameter=torch.nn.Parameter(torch.ones(2));parameter.grad=torch.tensor([3.,4.]);gradient=parameter.grad
    model=torch.nn.Linear(2,2);optimizer=SimpleNamespace(_native_named_parameters=[('p',parameter)])
    example=SimpleNamespace(plan=SimpleNamespace(token_count=100,slot_sources=range(101)))
    source=tmp_path/'source.json';source.write_text('{}')
    args=SimpleNamespace(cotangent_accumulator_backend='plain',checkpoint_layers=True,chunk_tokens=16,accumulate_batches=16,out=tmp_path)
    with pytest.raises(ValueError,match='failed live-model/QAT capacity'):
        qualify_oversize_batches(model,optimizer,[[example]],args=args,backend=None,precision=None,
            source_path=source,checkpoint_sha256='a'*64,global_step=62,baseline={'forward_tokens':10,'kv_slots':11})
    assert parameter.grad is gradient
    record=json.loads(next((tmp_path/'oversize-capacity').glob('*.json')).read_text())
    assert not record['passed'] and record['optimizer_updates']==record['earned_exposure']==0


def test_gate_failure_before_update_preserves_durable_checkpoint(tmp_path, monkeypatch):
    from test_native_source_fetch_recovery import source,equal
    from natlang_neuralese.train import native_state_sft as trainer,native_packed_capacity
    base=tmp_path/'base';paths=source(base,'lfm');out=tmp_path/'out'
    args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu',
          '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
          '--epochs','2','--batch-graphs','1','--accumulate-batches','1','--eval-every','1','--checkpoint-every','1','--checkpoint-seconds','0',
          '--seed','23','--lr-warmup','8','--refresh-after-updates','1','--batch-tokens','1','--out',str(out)]
    assert trainer.main(args)==0
    checkpoint=out/'checkpoint.pt';before=torch.load(checkpoint,map_location='cpu',weights_only=False)
    def fail(*args,**kwargs):raise ValueError('injected oversize capacity failure')
    monkeypatch.setattr(native_packed_capacity,'qualify_oversize_batches',fail)
    with pytest.raises(ValueError,match='injected oversize capacity'):
        trainer.main(args+['--continue-from',str(checkpoint)])
    after=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert after['training']['global_step']==before['training']['global_step']
    equal(before['training']['optimizer'],after['training']['optimizer'])
    equal(before['training']['rng'],after['training']['rng'])
    assert before['training']['exposure_counts']==after['training']['exposure_counts']


@pytest.mark.parametrize('mode', ['plain', 'vjp'])
def test_retained_capacity_keeps_accumulated_gradients_during_backward(mode):
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.eval.native_memory import qualify_capacity
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    params=[p for _,p in optimizer._native_named_parameters if p.requires_grad]
    counts=[]
    result=qualify_capacity(model,optimizer,examples()[:1],checkpoint_sha256='a'*64,source_binding={},
        cotangent_mode=mode,retained_gradients=True,
        gradient_observer=lambda name,gradient:counts.append(sum(p.grad is not None for p in params)))
    assert result['passed'] and not result['gradients_released_per_parameter']
    assert counts and min(counts)==len(params)
    assert all(p.grad is None for p in params)


def test_retained_gradient_allocation_oom_is_reported_without_update(monkeypatch):
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.eval.native_memory import qualify_capacity
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    def fail(*args,**kwargs):raise torch.cuda.OutOfMemoryError('injected retained buffer OOM')
    monkeypatch.setattr(torch,'zeros_like',fail)
    result=qualify_capacity(model,optimizer,examples()[:1],checkpoint_sha256='a'*64,source_binding={},
        cotangent_mode='plain',retained_gradients=True)
    assert not result['passed'] and result['optimizer_updates']==0
    assert result['cases'][0]['failure_phase']=='retained-gradient-allocation'
    assert all(p.grad is None for p in model.parameters())


def test_oversize_checkpoint_barrier_drains_before_forward_and_leaves_ordinary_async():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_packed_capacity import checkpoint_overlap_barrier
    class Policy:
        def __init__(self): self.drains = 0
        def drain_async(self): self.drains += 1
    def graph(tokens, slots):
        return SimpleNamespace(plan=SimpleNamespace(token_count=tokens, slot_sources=range(slots)))
    policy = Policy()
    baseline = {'forward_tokens': 100, 'kv_slots': 120}
    assert not checkpoint_overlap_barrier(policy, [[graph(100, 120)]], baseline)
    assert policy.drains == 0
    assert checkpoint_overlap_barrier(policy, [[graph(101, 120)]], baseline)
    assert policy.drains == 1
    assert checkpoint_overlap_barrier(policy, [[graph(90, 121)]], baseline)
    assert policy.drains == 2


def test_qualified_oversize_update_can_save_async_with_next_update_barrier():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_packed_capacity import checkpoint_overlap_barrier
    from natlang_neuralese.train.native_state_sft import _cadence_async_checkpoint_allowed

    class Policy:
        def __init__(self): self.drains = 0
        def drain_async(self): self.drains += 1

    def graph(tokens, slots):
        return SimpleNamespace(plan=SimpleNamespace(token_count=tokens, slot_sources=range(slots)))

    policy = Policy()
    baseline = {'forward_tokens': 100, 'kv_slots': 120}
    large = [[graph(101, 120)]]

    # Existing writes are drained before live capacity qualification and SGD.
    assert checkpoint_overlap_barrier(policy, large, baseline)
    assert policy.drains == 1
    assert _cadence_async_checkpoint_allowed(requested=True, oversized_update=True,
        oversize_proof={'path': 'bound-live-capacity-proof'}, before_final_update=True,
        stop_requested=False, refresh_due=False)
    # The next larger/unqualified batch drains that detached write before its gate.
    assert checkpoint_overlap_barrier(policy, large, baseline)
    assert policy.drains == 2

    assert not _cadence_async_checkpoint_allowed(requested=True, oversized_update=True,
        oversize_proof=None, before_final_update=True, stop_requested=False, refresh_due=False)
    assert _cadence_async_checkpoint_allowed(requested=True, oversized_update=False,
        oversize_proof=None, before_final_update=True, stop_requested=False, refresh_due=False)
    for change in ({'before_final_update': False}, {'stop_requested': True}, {'refresh_due': True}):
        kwargs = {'requested': True, 'oversized_update': True,
            'oversize_proof': {'path': 'bound-live-capacity-proof'}, 'before_final_update': True,
            'stop_requested': False, 'refresh_due': False}
        kwargs.update(change)
        assert not _cadence_async_checkpoint_allowed(**kwargs)


def test_capacity_envelope_uses_one_measured_group_not_crossed_maxima():
    from natlang_neuralese.train.native_packed_capacity import _one_group_dominates
    candidate={'graphs':[{'forward_tokens':80,'kv_slots':50}],
        'forward_tokens':80,'kv_slots':50,'graph_count':1}
    token_heavy={'graphs':[{'forward_tokens':100,'kv_slots':30}],
        'forward_tokens':100,'kv_slots':30,'graph_count':1}
    slot_heavy={'graphs':[{'forward_tokens':70,'kv_slots':60}],
        'forward_tokens':70,'kv_slots':60,'graph_count':1}
    jointly_large={'graphs':[{'forward_tokens':100,'kv_slots':60}],
        'forward_tokens':100,'kv_slots':60,'graph_count':1}
    assert not _one_group_dominates(token_heavy,candidate)
    assert not _one_group_dominates(slot_heavy,candidate)
    assert _one_group_dominates(jointly_large,candidate)


def test_persisted_capacity_envelope_reuses_only_exact_runtime_point_and_device_context(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_packed_capacity as capacity
    from natlang_neuralese.eval import native_memory
    model=torch.nn.Linear(2,2)
    optimizer=SimpleNamespace(_native_named_parameters=list(model.named_parameters()))
    source=tmp_path/'source.json';source.write_text('{}')
    from natlang_neuralese.train.native_resident import binding
    execution={'schema':'natlang.native-oversize-execution-identity/1','sha256':'a'*64}
    qat={'point':'fp8','mix':0.25,'loss_weight':1.5}
    group={'graphs':[{'id':'measured','forward_tokens':100,'kv_slots':120}],
        'forward_tokens':100,'kv_slots':120,'graph_count':1}
    proof={'schema':'natlang.native-oversize-live-capacity/1','global_step':61,'source':binding(source),
        'execution_identity':execution,
        'effective_layout':{'share_causal_prefixes':False,'allow_communication_prefixes':True},
        'memory_context':{'device_type':'cpu','free_bytes_at_probe':None,'total_bytes':None},
        'candidate_groups':[group], 'passed':True,'optimizer_updates':0,'earned_exposure':0,
        'model_boundary':'current-live-model-before-update',
        'capacity':{'passed':True,'optimizer_updates':0,'production_packing':True,
            'whole_graphs':True,'cpu_and_all_cuda_rng_restored':True,'task_quality_qualified':False,'qat_point':qat,
            'cases':[{'status':'passed','forward_tokens':100,'kv_slots':120,'id':'measured','example_ids':['measured']}]}}
    proof_path=tmp_path/'prior-proof.json';proof_path.write_text(json.dumps(proof))
    history=[binding(proof_path)]
    example=SimpleNamespace(id='candidate',plan=SimpleNamespace(token_count=80,slot_sources=range(90)))
    args=SimpleNamespace(checkpoint_layers=True,chunk_tokens=16,cotangent_accumulator_backend='plain',
        accumulate_batches=1,out=tmp_path)
    def fail(*args,**kwargs):raise AssertionError('a dominated shape should reuse its authenticated proof')
    monkeypatch.setattr(native_memory,'qualify_capacity',fail)
    ref=capacity.qualify_oversize_batches(model,optimizer,[[example]],args=args,backend=None,
        precision=(None,'fp8',0.99,1.5),source_path=source,checkpoint_sha256='b'*64,
        global_step=62,baseline={'forward_tokens':10,'kv_slots':10},prior_history=history,
        execution_identity=execution)
    reuse=json.loads(open(ref['path']).read())
    assert reuse['schema']=='natlang.native-oversize-capacity-envelope-reuse/1'
    assert reuse['measured_case_id']=='measured'
    assert reuse['precision']=={'point':'fp8','mix':0.99,'loss_weight':1.5}
    assert reuse['scope'].startswith('candidate is bounded componentwise by one complete measured group')


def test_capacity_envelope_rejects_different_execution_or_precision_point(tmp_path):
    from natlang_neuralese.train.native_packed_capacity import _find_oversize_envelope
    from natlang_neuralese.train.native_resident import binding
    source=tmp_path/'source.json';source.write_text('{}')
    proof={'schema':'natlang.native-oversize-live-capacity/1','global_step':4,'passed':True,
        'source':binding(source),'optimizer_updates':0,'earned_exposure':0,
        'model_boundary':'current-live-model-before-update',
        'execution_identity':{'sha256':'a'},
        'effective_layout':{'share_causal_prefixes':False,'allow_communication_prefixes':True},
        'memory_context':{'device_type':'cpu','free_bytes_at_probe':None,'total_bytes':None},
        'candidate_groups':[{'graphs':[{'id':'measured','forward_tokens':10,'kv_slots':12}],
            'forward_tokens':10,'kv_slots':12,'graph_count':1}],
        'capacity':{'passed':True,'optimizer_updates':0,'production_packing':True,
            'whole_graphs':True,'cpu_and_all_cuda_rng_restored':True,'task_quality_qualified':False,
            'qat_point':{'point':'bf16','mix':0.0,'loss_weight':1.0},
            'cases':[{'status':'passed','forward_tokens':10,'kv_slots':12,'example_ids':['measured']}]}}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof));history=[binding(path)]
    candidate=[{'graphs':[{'id':'candidate','forward_tokens':5,'kv_slots':6}],
        'forward_tokens':5,'kv_slots':6,'graph_count':1}]
    assert _find_oversize_envelope(history,candidate,execution_identity={'sha256':'b'},
        precision=(None,'bf16',0.0,1.0),effective_layout=proof['effective_layout'],
        current_memory=proof['memory_context']) is None
    assert _find_oversize_envelope(history,candidate,execution_identity={'sha256':'a'},
        precision=(None,'fp8',0.2,1.0),effective_layout=proof['effective_layout'],
        current_memory=proof['memory_context']) is None


def test_known_envelope_skips_async_drain_but_unknown_execution_drains(tmp_path):
    from natlang_neuralese.train.native_packed_capacity import preflight_oversize_update
    from natlang_neuralese.train.native_resident import binding
    model=torch.nn.Linear(2,2)
    source=tmp_path/'source.json';source.write_text('{}')
    execution={'schema':'natlang.native-oversize-execution-identity/1','sha256':'a'*64}
    layout={'share_causal_prefixes':False,'allow_communication_prefixes':True}
    group={'graphs':[{'id':'measured','forward_tokens':100,'kv_slots':120}],
        'forward_tokens':100,'kv_slots':120,'graph_count':1}
    proof={'schema':'natlang.native-oversize-live-capacity/1','global_step':61,'source':binding(source),
        'execution_identity':execution,'effective_layout':layout,
        'memory_context':{'device_type':'cpu','free_bytes_at_probe':None,'total_bytes':None},
        'candidate_groups':[group],'passed':True,'optimizer_updates':0,'earned_exposure':0,
        'model_boundary':'current-live-model-before-update',
        'capacity':{'passed':True,'optimizer_updates':0,'production_packing':True,
            'whole_graphs':True,'cpu_and_all_cuda_rng_restored':True,'task_quality_qualified':False,
            'qat_point':None,'cases':[{'status':'passed','forward_tokens':100,'kv_slots':120,
                'id':'measured','example_ids':['measured']}]}}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof));history=[binding(path)]
    example=SimpleNamespace(id='candidate',plan=SimpleNamespace(token_count=80,slot_sources=range(90)))
    class Policy:
        def __init__(self):self.drains=0
        def drain_async(self):self.drains+=1
    policy=Policy()
    oversized,match,groups,actual_layout,_=preflight_oversize_update(policy,model,[[example]],
        {'forward_tokens':10,'kv_slots':10},history=history,execution_identity=execution,
        precision=None,backend=None)
    assert oversized and match is not None and groups and actual_layout==layout and policy.drains==0
    oversized,match,_,_,_=preflight_oversize_update(policy,model,[[example]],
        {'forward_tokens':10,'kv_slots':10},history=history,execution_identity={'sha256':'different'},
        precision=None,backend=None)
    assert oversized and match is None and policy.drains==1
