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
    example=SimpleNamespace(id='oversize-case',plan=SimpleNamespace(token_count=100,slot_sources=range(101)))
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
    example=SimpleNamespace(id='oversize-failure-case',plan=SimpleNamespace(token_count=100,slot_sources=range(101)))
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
    assert trainer.main(args+['--initialize-only']) is None
    checkpoint=out/'checkpoint.pt';before=torch.load(checkpoint,map_location='cpu',weights_only=False)
    def fail(*args,**kwargs):raise ValueError('injected oversize capacity failure')
    monkeypatch.setattr(native_packed_capacity,'qualify_oversize_batches',fail)
    monkeypatch.setattr(native_packed_capacity,'oversize_baseline',
        lambda *args,**kwargs:{'forward_tokens':1,'kv_slots':1,'basis':'injected-test-boundary'})
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


def test_capacity_envelope_uses_incremental_peak_and_current_reusable_headroom():
    from natlang_neuralese.train.native_packed_capacity import _incremental_headroom_covers
    case={'cuda_incremental_peak_allocated_bytes':4<<30,'host_incremental_peak_bytes':0}
    # A 1 GiB checkpoint snapshot may reduce free unified memory, while this
    # process can still reuse cached allocator blocks. Require measured peak
    # plus 25% headroom (at least 1 GiB), not the old proof-time free watermark.
    assert _incremental_headroom_covers(case,{'device_type':'cuda','free_bytes_at_probe':2<<30,
        'reusable_cache_bytes':3<<30,'total_bytes':128<<30,'host_available_bytes':6<<30})
    assert not _incremental_headroom_covers(case,{'device_type':'cuda','free_bytes_at_probe':2<<30,
        'reusable_cache_bytes':2<<30,'total_bytes':128<<30,'host_available_bytes':6<<30})
    assert _incremental_headroom_covers({'host_incremental_peak_bytes':2<<30},
        {'device_type':'cpu','host_available_bytes':5<<30})
    assert not _incremental_headroom_covers({'host_incremental_peak_bytes':2<<30},
        {'device_type':'cpu','host_available_bytes':2<<30})


def test_capacity_host_headroom_uses_nested_process_cgroup_without_losing_host_fallback(tmp_path):
    from natlang_neuralese.train.trajectory_state import available_system_memory_bytes
    root=tmp_path/'cgroup';leaf=root/'user.slice'/'job.service'
    (leaf/'memory.current').parent.mkdir(parents=True)
    (leaf/'memory.current').write_text(str(3<<30))
    (leaf/'memory.max').write_text(str(8<<30))
    parent=leaf.parent
    (parent/'memory.current').write_text(str(5<<30))
    (parent/'memory.max').write_text(str(6<<30))
    meminfo=tmp_path/'meminfo';meminfo.write_text('MemTotal: 16 kB\nMemAvailable: 7000000 kB\n')
    cgroup=tmp_path/'cgroup.txt';cgroup.write_text('0::/user.slice/job.service\n')
    assert available_system_memory_bytes(meminfo_path=meminfo,cgroup_path=cgroup,cgroup_root=root,
        cgroup_v1_root=tmp_path/'v1-memory')==1<<30

    # Unlimited nested cgroups inherit any finite parent; an unlimited or
    # absent controller hierarchy leaves the host measurement available.
    (leaf/'memory.max').write_text('max')
    assert available_system_memory_bytes(meminfo_path=meminfo,cgroup_path=cgroup,cgroup_root=root,
        cgroup_v1_root=tmp_path/'v1-memory')==1<<30
    (parent/'memory.max').write_text('max')
    assert available_system_memory_bytes(meminfo_path=meminfo,cgroup_path=cgroup,cgroup_root=root,
        cgroup_v1_root=tmp_path/'v1-memory')==7000000*1024

    # A missing cgroup mount/file is not grounds to discard the valid host
    # MemAvailable measurement or to make all cached envelopes ineligible.
    cgroup.write_text('0::/missing/service\n')
    assert available_system_memory_bytes(meminfo_path=meminfo,cgroup_path=cgroup,cgroup_root=root,
        cgroup_v1_root=tmp_path/'v1-memory')==7000000*1024

    # If a finite bound is advertised but cannot be read, return unknown rather
    # than treating the larger host measurement as safe headroom.
    unreadable=root/'broken'/'service'
    unreadable.mkdir(parents=True)
    (unreadable/'memory.max').write_text(str(4<<30))
    cgroup.write_text('0::/broken/service\n')
    assert available_system_memory_bytes(meminfo_path=meminfo,cgroup_path=cgroup,cgroup_root=root,
        cgroup_v1_root=tmp_path/'v1-memory') is None
    (unreadable/'memory.current').mkdir()
    assert available_system_memory_bytes(meminfo_path=meminfo,cgroup_path=cgroup,cgroup_root=root,
        cgroup_v1_root=tmp_path/'v1-memory') is None


def test_capacity_peak_keeps_prior_large_case_for_later_small_case():
    from natlang_neuralese.eval.native_memory import _gate_cuda_peaks
    rows=[{'cuda_peak_allocated_bytes':12<<30,'cuda_peak_reserved_bytes':15<<30},
        {'cuda_peak_allocated_bytes':7<<30,'cuda_peak_reserved_bytes':9<<30}]
    allocated,reserved=_gate_cuda_peaks(rows,start_allocated=5<<30,start_reserved=6<<30,is_cuda=True)
    assert allocated==7<<30
    assert reserved==9<<30
    # A later small case cannot reset the global high-water used by async
    # checkpoint admission or shrink the cached proof's required headroom.
    rows.append({'cuda_peak_allocated_bytes':6<<30,'cuda_peak_reserved_bytes':7<<30})
    assert _gate_cuda_peaks(rows,start_allocated=5<<30,start_reserved=6<<30,is_cuda=True)==(allocated,reserved)


def test_persisted_capacity_envelope_reuses_only_exact_runtime_point_and_device_context(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_packed_capacity as capacity
    from natlang_neuralese.eval import native_memory
    model=torch.nn.Linear(2,2)
    optimizer=SimpleNamespace(_native_named_parameters=list(model.named_parameters()))
    source=tmp_path/'source.json';source.write_text('{}')
    monkeypatch.setattr(capacity,'_capacity_memory_context',lambda device:{'device_type':'cpu',
        'free_bytes_at_probe':None,'total_bytes':None,'host_available_bytes':8<<30})
    from natlang_neuralese.train.native_resident import binding
    execution={'schema':'natlang.native-oversize-execution-identity/1','sha256':'a'*64}
    qat={'point':'fp8','mix':0.25,'loss_weight':1.5}
    group={'graphs':[{'id':'measured','forward_tokens':100,'kv_slots':120}],
        'forward_tokens':100,'kv_slots':120,'graph_count':1}
    proof={'schema':'natlang.native-oversize-live-capacity/1','global_step':61,'source':binding(source),
        'execution_identity':execution,
        'effective_layout':{'share_causal_prefixes':False,'allow_communication_prefixes':True},
        'memory_context':{'device_type':'cpu','free_bytes_at_probe':None,'total_bytes':None,'host_available_bytes':8<<30},
        'candidate_groups':[group], 'passed':True,'optimizer_updates':0,'earned_exposure':0,
        'model_boundary':'current-live-model-before-update',
        'capacity':{'passed':True,'optimizer_updates':0,'production_packing':True,
            'whole_graphs':True,'cpu_and_all_cuda_rng_restored':True,'task_quality_qualified':False,'qat_point':qat,
            'cases':[{'status':'passed','forward_tokens':100,'kv_slots':120,'id':'measured','example_ids':['measured'],
                'cuda_incremental_peak_allocated_bytes':0,'cuda_incremental_peak_reserved_bytes':0,
                'host_incremental_peak_bytes':0}]}}
    proof_path=tmp_path/'prior-proof.json';proof_path.write_text(json.dumps(proof))
    history=[binding(proof_path)]
    example=SimpleNamespace(id='candidate',plan=SimpleNamespace(token_count=80,slot_sources=range(90)))
    args=SimpleNamespace(checkpoint_layers=True,chunk_tokens=16,cotangent_accumulator_backend='plain',
        accumulate_batches=1,out=tmp_path)
    def fail(*args,**kwargs):raise AssertionError('a dominated shape should reuse its authenticated proof')
    monkeypatch.setattr(native_memory,'qualify_capacity',fail)
    ref=capacity.qualify_oversize_batches(model,optimizer,[[example]],args=args,backend=None,
        precision=(None,'fp8',0.99,1.5),source_path=source,checkpoint_sha256='b'*64,
        global_step=62,baseline={'forward_tokens':10,'kv_slots':10},prior_frontier=capacity.load_oversize_capacity_frontier(
            [{'proof':history[0]}]),
        execution_identity=execution)
    reuse=json.loads(open(ref['path']).read())
    assert reuse['schema']=='natlang.native-oversize-capacity-envelope-reuse/1'
    assert reuse['measured_case_id']=='measured'
    assert reuse['precision']=={'point':'fp8','mix':0.99,'loss_weight':1.5}
    assert reuse['scope'].startswith('candidate is bounded componentwise by one complete measured group')


def test_capacity_envelope_rejects_different_execution_or_precision_point(tmp_path):
    from natlang_neuralese.train import native_packed_capacity as capacity
    from natlang_neuralese.train.native_packed_capacity import _find_oversize_envelope
    from natlang_neuralese.train.native_resident import binding
    source=tmp_path/'source.json';source.write_text('{}')
    proof={'schema':'natlang.native-oversize-live-capacity/1','global_step':4,'passed':True,
        'source':binding(source),'optimizer_updates':0,'earned_exposure':0,
        'model_boundary':'current-live-model-before-update',
        'execution_identity':{'sha256':'a'},
        'effective_layout':{'share_causal_prefixes':False,'allow_communication_prefixes':True},
        'memory_context':{'device_type':'cpu','free_bytes_at_probe':None,'total_bytes':None,'host_available_bytes':8<<30},
        'candidate_groups':[{'graphs':[{'id':'measured','forward_tokens':10,'kv_slots':12}],
            'forward_tokens':10,'kv_slots':12,'graph_count':1}],
        'capacity':{'passed':True,'optimizer_updates':0,'production_packing':True,
            'whole_graphs':True,'cpu_and_all_cuda_rng_restored':True,'task_quality_qualified':False,
            'qat_point':{'point':'bf16','mix':0.0,'loss_weight':1.0},
            'cases':[{'status':'passed','forward_tokens':10,'kv_slots':12,'example_ids':['measured'],
                'cuda_incremental_peak_allocated_bytes':0,'cuda_incremental_peak_reserved_bytes':0,
                'host_incremental_peak_bytes':0}]}}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof));history=capacity.load_oversize_capacity_frontier(
        [{'proof':binding(path)}])
    candidate=[{'graphs':[{'id':'candidate','forward_tokens':5,'kv_slots':6}],
        'forward_tokens':5,'kv_slots':6,'graph_count':1}]
    assert _find_oversize_envelope(history,candidate,execution_identity={'sha256':'b'},
        precision=(None,'bf16',0.0,1.0),effective_layout=proof['effective_layout'],
        current_memory=proof['memory_context']) is None
    assert _find_oversize_envelope(history,candidate,execution_identity={'sha256':'a'},
        precision=(None,'fp8',0.2,1.0),effective_layout=proof['effective_layout'],
        current_memory=proof['memory_context']) is None


def test_known_envelope_skips_async_drain_but_unknown_execution_drains(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_packed_capacity as capacity
    from natlang_neuralese.train.native_packed_capacity import preflight_oversize_update
    from natlang_neuralese.train.native_resident import binding
    model=torch.nn.Linear(2,2)
    monkeypatch.setattr(capacity,'_capacity_memory_context',lambda device:{'device_type':'cpu',
        'free_bytes_at_probe':None,'total_bytes':None,'host_available_bytes':8<<30})
    source=tmp_path/'source.json';source.write_text('{}')
    execution={'schema':'natlang.native-oversize-execution-identity/1','sha256':'a'*64}
    layout={'share_causal_prefixes':False,'allow_communication_prefixes':True}
    group={'graphs':[{'id':'measured','forward_tokens':100,'kv_slots':120}],
        'forward_tokens':100,'kv_slots':120,'graph_count':1}
    proof={'schema':'natlang.native-oversize-live-capacity/1','global_step':61,'source':binding(source),
        'execution_identity':execution,'effective_layout':layout,
        'memory_context':{'device_type':'cpu','free_bytes_at_probe':None,'total_bytes':None,'host_available_bytes':8<<30},
        'candidate_groups':[group],'passed':True,'optimizer_updates':0,'earned_exposure':0,
        'model_boundary':'current-live-model-before-update',
        'capacity':{'passed':True,'optimizer_updates':0,'production_packing':True,
            'whole_graphs':True,'cpu_and_all_cuda_rng_restored':True,'task_quality_qualified':False,
            'qat_point':None,'cases':[{'status':'passed','forward_tokens':100,'kv_slots':120,
                'id':'measured','example_ids':['measured'],'cuda_incremental_peak_allocated_bytes':0,
                'cuda_incremental_peak_reserved_bytes':0,'host_incremental_peak_bytes':0}]}}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof));history=capacity.load_oversize_capacity_frontier(
        [{'proof':binding(path)}])
    example=SimpleNamespace(id='candidate',plan=SimpleNamespace(token_count=80,slot_sources=range(90)))
    class Policy:
        def __init__(self):self.drains=0
        def drain_async(self):self.drains+=1
    policy=Policy()
    oversized,match,groups,actual_layout,_=preflight_oversize_update(policy,model,[[example]],
        {'forward_tokens':10,'kv_slots':10},frontier=history,execution_identity=execution,
        precision=None,backend=None)
    assert oversized and match is not None and groups and actual_layout==layout and policy.drains==0
    oversized,match,_,_,_=preflight_oversize_update(policy,model,[[example]],
        {'forward_tokens':10,'kv_slots':10},frontier=history,execution_identity={'sha256':'different'},
        precision=None,backend=None)
    assert oversized and match is None and policy.drains==1


def test_sequential_oversize_groups_each_reuse_own_complete_measured_case():
    from natlang_neuralese.train.native_packed_capacity import _find_oversize_envelope
    identity={'sha256':'a'}
    layout={'share_causal_prefixes':True,'allow_communication_prefixes':True}
    memory={'device_type':'cpu','total_bytes':None,'host_available_bytes':8 << 30}
    def group(spec):
        graphs=[{'id':str(i),'forward_tokens':t,'kv_slots':k} for i,(t,k) in enumerate(spec)]
        return {'graphs':graphs,'forward_tokens':sum(t for t,k in spec),
                'kv_slots':sum(k for t,k in spec),'graph_count':len(spec)}
    def entry(name,shape):
        return {'proof':{'path':name,'sha256':'x'},'record':{'execution_identity':identity,
            'effective_layout':layout,'memory_context':memory,'global_step':1,
            'capacity':{'qat_point':None}},'group':group(shape),
            'case':{'id':name,'host_incremental_peak_bytes':0}}
    frontier=[entry('single',[(300,310)]),entry('pair',[(60,70),(60,70)])]
    args=dict(execution_identity=identity,precision=None,effective_layout=layout,current_memory=memory)
    result=_find_oversize_envelope(frontier,[group([(200,210)]),group([(50,60),(50,60)])],**args)
    assert [entry[0]['path'] for entry in result]==['single','pair']
    # No individual measured pair covers these shapes: do not combine maxima
    # from different cases, and do not treat a single graph as two graphs.
    assert _find_oversize_envelope(frontier,[group([(75,80),(75,80)])],**args) is None
    assert _find_oversize_envelope(frontier,[group([(200,210)])],**{**args,'execution_identity':{'sha256':'other'}}) is None


def test_multi_envelope_receipt_binds_every_sequential_case(tmp_path):
    from natlang_neuralese.train.native_packed_capacity import record_oversize_envelope_reuse
    source=tmp_path/'source';source.write_text('source')
    groups=[{'graphs':[{'id':'a','forward_tokens':2,'kv_slots':2}],
             'forward_tokens':2,'kv_slots':2,'graph_count':1},
            {'graphs':[{'id':'b','forward_tokens':4,'kv_slots':4}],
             'forward_tokens':4,'kv_slots':4,'graph_count':1}]
    matches=[({'path':str(tmp_path/'p1'),'sha256':'1'}, {'global_step':4}, {'id':'case1'}, groups[0]),
             ({'path':str(tmp_path/'p2'),'sha256':'2'}, {'global_step':5}, {'id':'case2'}, groups[1])]
    ref=record_oversize_envelope_reuse(SimpleNamespace(out=tmp_path),match=matches,
        candidate_groups=groups,source_path=source,execution_identity={'sha256':'a'},
        precision=None,effective_layout={},current_memory={},checkpoint_sha256='b',global_step=6)
    record=json.loads(open(ref['path']).read())
    assert [x['measured_case_id'] for x in record['case_matches']]==['case1','case2']
    assert [x['envelope_source']['sha256'] for x in record['case_matches']]==['1','2']
    assert [x['candidate_group'] for x in record['case_matches']]==groups


def test_oversize_gate_selects_actual_nondominated_groups_without_inventing_maxima():
    from natlang_neuralese.train.native_packed_capacity import _nondominated_group_indices
    def group(spec):
        graphs=[{'id':str(i),'forward_tokens':t,'kv_slots':k} for i,(t,k) in enumerate(spec)]
        return {'graphs':graphs,'forward_tokens':sum(t for t,k in spec),
                'kv_slots':sum(k for t,k in spec),'graph_count':len(spec)}
    groups=[group([(41015,42289)]),group([(45772,45954)]),group([(63438,64500)]),
            group([(33403,33678)]),group([(56666,58730)])]
    selected,coverage=_nondominated_group_indices(groups)
    assert selected==[2] and coverage==[2]*5
    # Crossing per-graph geometry cannot be turned into a synthetic maximum.
    groups=[group([(100,110)]),group([(90,120)]),group([(40,40),(40,40)])]
    selected,coverage=_nondominated_group_indices(groups)
    assert selected==[0,1,2] and coverage==[0,1,2]
    selected,coverage=_nondominated_group_indices([groups[0],groups[0]])
    assert selected==[0] and coverage==[0,0]


def test_live_gate_measures_one_actual_dominator_and_records_all_requested(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_packed_capacity as capacity
    from natlang_neuralese.eval import native_memory
    model=torch.nn.Linear(2,2);optimizer=SimpleNamespace(_native_named_parameters=list(model.named_parameters()))
    source=tmp_path/'source.json';source.write_text('{}')
    monkeypatch.setattr(capacity,'_capacity_memory_context',lambda device:{'device_type':'cpu',
        'free_bytes_at_probe':None,'total_bytes':None,'host_available_bytes':8<<30})
    lengths=[(41,42),(63,64),(56,58)]
    batches=[[SimpleNamespace(id=str(i),plan=SimpleNamespace(token_count=t,slot_sources=range(k)))]
             for i,(t,k) in enumerate(lengths)]
    seen=[]
    def qualify(model,optimizer,selected,**kwargs):
        seen.extend(selected)
        return {'passed':True,'optimizer_updates':0,'production_packing':True,'whole_graphs':True,
                'cpu_and_all_cuda_rng_restored':True,'task_quality_qualified':False,'qat_point':None,
                'cases':[{'status':'passed','forward_tokens':63,'kv_slots':64,'example_ids':['1'],
                          'cuda_incremental_peak_allocated_bytes':0,'cuda_incremental_peak_reserved_bytes':0,
                          'host_incremental_peak_bytes':0}]}
    monkeypatch.setattr(native_memory,'qualify_capacity',qualify)
    args=SimpleNamespace(checkpoint_layers=True,chunk_tokens=16,cotangent_accumulator_backend='plain',
                         accumulate_batches=1,out=tmp_path)
    ref=capacity.qualify_oversize_batches(model,optimizer,batches,args=args,backend=None,precision=None,
        source_path=source,checkpoint_sha256='a'*64,global_step=59,
        baseline={'forward_tokens':10,'kv_slots':10},execution_identity={'sha256':'a'})
    record=json.loads(open(ref['path']).read())
    assert seen==[batches[1]]
    assert len(record['candidate_groups'])==1 and len(record['requested_candidate_groups'])==3
    assert [c['measured_index'] for c in record['requested_group_coverage']]==[0,0,0]
    assert len(capacity.load_oversize_capacity_frontier([{'proof':ref}]))==1


def test_capacity_dominance_requires_padded_attention_and_node_layout():
    from natlang_neuralese.train.native_packed_capacity import _one_group_dominates
    def group(tokens,slots,workspace,nodes,targets=1):
        graph={'id':'g','forward_tokens':tokens,'kv_slots':slots,
               'attention_workspace_rows':workspace,'node_shapes':nodes,
               'target_positions':targets,'decision_count':0}
        return {'graphs':[graph],'forward_tokens':tokens,'kv_slots':slots,'graph_count':1}
    measured=group(1000,1200,1024,[(1000,1200,1,1,0)],20)
    assert _one_group_dominates(measured,group(900,1000,1024,[(900,1000,1,1,0)],10))
    assert not _one_group_dominates(measured,group(900,1000,1152,[(900,1000,1,1,0)],10))
    assert not _one_group_dominates(measured,group(900,1000,1024,[(450,500,1,1,0),(450,500,1,1,0)],10))
    assert not _one_group_dominates(measured,group(900,1000,1024,[(900,1000,1,1,0)],21))
    legacy=group(1000,1200,1024,[(1000,1200,1,1,0)],20)
    del legacy['graphs'][0]['node_shapes']
    assert not _one_group_dominates(legacy,group(900,1000,1024,[(900,1000,1,1,0)],10))
