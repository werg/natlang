import contextlib
import copy
import random
from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.eval import native_attention_transition as transition
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from test_shared_state import tiny


class Tokenizer:
    all_special_ids=[]
    def __call__(self,*a,**k):return SimpleNamespace(input_ids=[2,3,4,5,6,7,8,9])
    def decode(self,*a,**k):return 'lexical source'


class Backend:
    def __init__(self):self.row_counts=dict(calls=7,rows=21);self.has_executed=True;self.state={'scale':[3]};self.active=False
    def extra_state(self):return self.state
    def restore_extra_state(self,state):self.state=state
    def collect_row_counts(self):return dict(self.row_counts)
    def configuration(self):return dict(backend='counting-state-contract-only')
    @contextlib.contextmanager
    def autocast(self):
        previous=self.active;self.active=True
        try:yield
        finally:self.active=previous


def fixture(family,device="cpu",inactive=False):
    model=SharedStateModel(tiny(family).to(device),(58,59,60,61),rank=4);model.port.checkpoint_layers=False
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    if inactive:
        expert=torch.nn.Module();expert.self_attn=torch.nn.Module();expert.self_attn.q_proj=torch.nn.Linear(3,3,bias=False,device=device)
        model.port.layers[0].add_module('inactive_expert',expert)
    with torch.no_grad():model.adapter.up.weight.normal_(std=.03)
    optimizer,named=make_native_optimizer(model,lr=3e-6,in_backward=False)
    for _,p in named:p.grad=torch.randn_like(p)*.001
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    model.train();model.adapter.eval()
    roles=SimpleNamespace(produce_open=58,produce_close=59,consume_open=60,consume_close=61)
    config=dict(architecture=model.port.config.model_type,tokenizer_fingerprint='b'*64)
    return model,optimizer,named,roles,config


def equal(left,right):
    if torch.is_tensor(left):assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):equal(a,b)
    else:assert left==right


def call(model,optimizer,roles,config,**kwargs):
    return transition.qualify_attention_transition(model,Tokenizer(),roles,config,optimizer=optimizer,
        previous_mode='ieee',candidate_mode='tf32x3',checkpoint_sha256='a'*64,
        forward_atol=1e-4,gradient_atol=1e-5,**kwargs)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_complete_both_port_derivatives_preserve_real_inherited_optimizer_and_original_parameters(family):
    model,optimizer,named,roles,config=fixture(family)
    original=dict(model.named_parameters());weights={name:p.detach().clone()for name,p in original.items()};versions={name:p._version for name,p in original.items()}
    named[0][1].grad=torch.randn_like(named[0][1]);grad=named[0][1].grad.clone();state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone();python_rng=random.getstate();modes={m:m.training for m in model.modules()}
    report=call(model,optimizer,roles,config)
    assert report['passed']and report['cross_mode_passed']and report['cached_reference_passed']and report['source_state_preserved']
    assert not report['readiness_qualified']and report['optimizer_updates']==0
    assert report['channel_copy_bytes']>0
    for execution in('graph','cached'):
        assert set(report['cross_mode'][execution]['comparisons'])=={'input_embeddings',*(name for name,_ in named)}
        assert report['cross_mode'][execution]['parameter_grads_preserved']
    assert all(report['roles'].values())
    assert all(dict(model.named_parameters())[name]is p and p._version==versions[name]for name,p in original.items())
    for name,p in original.items():assert torch.equal(p,weights[name])
    equal(optimizer.state_dict(),state);assert torch.equal(named[0][1].grad,grad)
    assert torch.equal(torch.get_rng_state(),rng)and random.getstate()==python_rng
    assert {m:m.training for m in model.modules()}==modes and model.port.checkpoint_layers is False
    assert model.native_attention_precision==model.port.native_attention_precision=='ieee'


def test_real_wrong_candidate_backbone_derivative_fails_without_waiving_cached_reference():
    model,opt,named,roles,config=fixture('lfm');p=next(p for name,p in named if name.startswith('backbone.'))
    hook=p.register_hook(lambda value:value*2 if getattr(model,'native_attention_precision','ieee')=='tf32x3'else value)
    try:report=call(model,opt,roles,config)
    finally:hook.remove()
    assert not report['passed']and not report['cross_mode_passed']
    assert any(not row['passed']for row in report['cross_mode']['graph']['comparisons'].values())
    assert report['source_state_preserved']


@pytest.mark.parametrize('execute',[True,False])
def test_actual_backend_scope_counters_restored_and_no_execution_cannot_pass(execute):
    model,opt,named,roles,config=fixture('lfm');backend=Backend()
    def count(module,args,result):
        assert backend.active
        if execute:backend.row_counts['calls']+=1;backend.row_counts['rows']+=result['h_final'].shape[1]
    hook=model.register_forward_hook(count)
    try:report=call(model,opt,roles,config,backend=backend)
    finally:hook.remove()
    assert report['passed']is execute
    assert backend.row_counts==dict(calls=7,rows=21)and backend.state=={'scale':[3]}and backend.has_executed and not backend.active
    for which in('baseline_cached_reference','candidate_cached_reference'):
        assert report[which]['training_backend_execution']['passed']is execute
    for which in('graph','cached'):
        assert all(row['passed']is execute for row in report['cross_mode'][which]['backend_execution'].values())


def test_exception_restores_channel_objects_modes_rng_and_backend(monkeypatch):
    model,opt,named,roles,config=fixture('lfm');backend=Backend();params=dict(model.named_parameters());versions={id(p):p._version for p in params.values()};rng=torch.get_rng_state().clone()
    def fail(*a,**k):backend.state['scale'][0]=99;torch.rand(4);raise RuntimeError('compiler failed')
    monkeypatch.setattr(transition,'compare_complete_gradients',fail)
    with pytest.raises(RuntimeError,match='compiler'):call(model,opt,roles,config,backend=backend)
    assert all(dict(model.named_parameters())[name]is p and p._version==versions[id(p)]for name,p in params.items())
    assert torch.equal(torch.get_rng_state(),rng)and backend.state=={'scale':[3]}and not backend.active
    assert model.training and not model.adapter.training and not model.port.checkpoint_layers


def test_original_backbone_write_is_not_excused_as_channel_roundtrip(monkeypatch):
    model,opt,named,roles,config=fixture('lfm');p=next(p for name,p in named if name.startswith('backbone.'));real=transition.qualify_existing
    def changed(*a,**k):
        result=real(*a,**k)
        with torch.no_grad():p.add_(1)
        return result
    monkeypatch.setattr(transition,'qualify_existing',changed)
    with pytest.raises(transition.CapacityStateChangedError,match='changed parameters'):call(model,opt,roles,config)


def test_missing_inventory_or_armed_optimizer_fails_before_graph(monkeypatch):
    model,opt,named,roles,config=fixture('lfm');opt._native_named_parameters=named[:-1]
    with pytest.raises(ValueError,match='omitted'):call(model,opt,roles,config)


def test_equal_mode_compares_distinct_old_and_candidate_executions():
    model,opt,named,roles,config=fixture('lfm')
    report=transition.qualify_attention_transition(model,Tokenizer(),roles,config,optimizer=opt,
        previous_mode='ieee',candidate_mode='ieee',checkpoint_sha256='a'*64,
        forward_atol=1e-4,gradient_atol=1e-5)
    assert report['passed']
    for row in report['cross_mode'].values():
        assert row['forward_max_abs']==row['consumer_ce_abs']==0
        assert all(value.get('max_abs',0)==0 for value in row['comparisons'].values())


def test_temporary_channel_value_write_is_rejected_and_original_owner_preserved(monkeypatch):
    model,opt,named,roles,config=fixture('lfm');original=model.input_rows;value=original.detach().clone()
    real=transition.qualify_existing
    def changed(*args,**kwargs):
        result=real(*args,**kwargs)
        with torch.no_grad():model.input_rows.add_(1)
        return result
    monkeypatch.setattr(transition,'qualify_existing',changed)
    with pytest.raises(transition.CapacityStateChangedError,match='temporary channel'):
        call(model,opt,roles,config)
    assert model.input_rows is original and torch.equal(original,value)


def test_actual_sft_caller_provisions_complete_optimizer_and_restores_candidate_mode():
    from natlang_neuralese.train.native_state_sft import qualify_precision_transition
    from natlang_neuralese.model.native_precision import set_native_attention_precision
    model,opt,named,roles,config=fixture('lfm')
    set_native_attention_precision(model,'tf32x3')
    config.update(native_attention_precision='tf32x3',native_linear_precision='ieee')
    result=qualify_precision_transition(model,Tokenizer(),roles,config,optimizer=opt,
        previous_precision='ieee',previous_attention_precision='ieee',
        checkpoint_sha256='a'*64,forward_atol=1e-4,gradient_atol=1e-5,device='cpu')
    assert result['attention_complete_comparison']['passed']
    assert result['attention_complete_comparison']['roles']['backbone']
    assert result['runtime_reference']['complete_gradient_qualification']['passed']
    assert model.native_attention_precision==model.port.native_attention_precision=='tf32x3'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA attention transition')
@pytest.mark.parametrize('family',['lfm','mellum'])
def test_cuda_real_complete_transition_preserves_inherited_optimizer(family):
    model,opt,named,roles,config=fixture(family,device='cuda')
    state=copy.deepcopy(opt.state_dict());rng=torch.cuda.get_rng_state().clone()
    versions={id(p):p._version for p in model.parameters()}
    report=call(model,opt,roles,config)
    assert report['passed']and report['cross_mode_passed']and report['cached_reference_passed']
    assert set(report['cross_mode']['graph']['comparisons'])=={'input_embeddings',*(name for name,_ in named)}
    assert all(p._version==versions[id(p)]for p in model.parameters())
    equal(opt.state_dict(),state);assert torch.equal(torch.cuda.get_rng_state(),rng)


def test_stateful_backend_old_vjp_precedes_candidate_forward_and_all_state_is_restored():
    model,opt,named,roles,config=fixture('lfm');backend=Backend();backwards=[]
    def forward(module,args,result):
        assert backend.active
        backend.row_counts['calls']+=1;backend.row_counts['rows']+=result['h_final'].shape[1]
        backend.state['scale'][0]+=1
        expected=backend.state['scale'][0]
        if result['h_final'].requires_grad:
            def backward(value):
                assert backend.state['scale'][0]==expected,'candidate forward changed old backward state'
                backwards.append(expected)
                return value
            result['h_final'].register_hook(backward)
    hook=model.register_forward_hook(forward)
    try:report=call(model,opt,roles,config,backend=backend)
    finally:hook.remove()
    assert report['passed']and len(backwards)>=8
    assert backend.state=={'scale':[3]}and backend.row_counts==dict(calls=7,rows=21)


@pytest.mark.parametrize('named',[None,[('parameter',None)]])
def test_backend_relative_gate_is_forwarded_with_or_without_complete_inventory(monkeypatch,named):
    from natlang_neuralese.eval import native_backend
    backend=Backend()
    def qualify(*args,**kwargs):
        assert kwargs['gradient_rtol']==2e-5
        assert kwargs.get('gradient_parameters')==named
        backend.row_counts.update(calls=8,rows=23)
        return {'passed':True}
    monkeypatch.setattr(native_backend,'qualify_existing',qualify)
    result=native_backend.qualify_training_backend(None,None,None,{},backend=backend,
        checkpoint_sha256='a'*64,forward_atol=1e-4,gradient_atol=1e-5,gradient_rtol=2e-5,
        gradient_parameters=named,device='cpu')
    assert result['passed']


def inherited_cached_gap(model):
    # A real graph/cached discrepancy shared by both arithmetic modes. It does
    # not alter parameters or numeric gates and is retained in both reports.
    def alter(module,args,kwargs,output):
        if kwargs.get('reference'):
            return {**output,'h_final':output['h_final']*1.01}
    return model.register_forward_hook(alter,with_kwargs=True)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_real_inherited_cached_gap_does_not_hide_cross_mode_equivalence(family):
    model,opt,named,roles,config=fixture(family);hook=inherited_cached_gap(model)
    try:report=call(model,opt,roles,config)
    finally:hook.remove()
    assert not report['passed']and not report['cached_reference_passed']and not report['readiness_qualified']
    assert not report['baseline_cached_reference']['passed']and not report['candidate_cached_reference']['passed']
    assert report['cross_mode_passed']and report['source_state_preserved']
    assert report['transition_equivalence']['passed']and report['transition_equivalence']['inherited_cached_reference_debt']
    assert not report['transition_equivalence']['readiness_qualified']


@pytest.mark.parametrize('change',['new-failure','worse-error','nonfinite','missing','cross-missing'])
def test_equivalence_denies_new_or_worse_named_cached_failures(change):
    model,opt,named,roles,config=fixture('lfm');hook=inherited_cached_gap(model)
    try:report=call(model,opt,roles,config)
    finally:hook.remove()
    assert report['transition_equivalence']['passed']
    new=report['candidate_cached_reference']['complete_gradient_qualification']['comparisons']
    before=report['baseline_cached_reference']['complete_gradient_qualification']['comparisons']
    name=next(k for k,v in new.items()if v.get('finite')is True)
    if change=='new-failure':before[name]['passed']=True;new[name]['passed']=False
    elif change=='worse-error':new[name]['max_abs']=before[name]['max_abs']+report['gradient_atol']*100
    elif change=='nonfinite':new[name]['relative_l2']=float('nan')
    elif change=='missing':new.pop(name)
    else:report['cross_mode']['cached']['comparisons'].pop(name)
    assert not transition.transition_equivalence(report)['passed']


def test_attention_only_resume_keeps_failed_runtime_readiness_proof():
    from natlang_neuralese.train.native_state_sft import qualify_precision_transition
    from natlang_neuralese.model.native_precision import set_native_attention_precision
    model,opt,named,roles,config=fixture('lfm');set_native_attention_precision(model,'tf32x3');config.update(native_attention_precision='tf32x3',native_linear_precision='ieee')
    hook=inherited_cached_gap(model)
    try:
        result=qualify_precision_transition(model,Tokenizer(),roles,config,previous_precision='ieee',
            previous_attention_precision='ieee',checkpoint_sha256='a'*64,forward_atol=1e-4,gradient_atol=1e-5,device='cpu',optimizer=opt)
    finally:hook.remove()
    assert result['attention_transition_equivalence']['passed']and result['inherited_cached_reference_debt']
    assert not result['runtime_reference']['passed']and not result['attention_complete_comparison']['passed']


@pytest.mark.parametrize('other_change',['linear','norm'])
def test_simultaneous_other_change_retains_strict_cached_gate(monkeypatch,other_change):
    from natlang_neuralese.train.native_state_sft import qualify_precision_transition
    from natlang_neuralese.model.native_precision import set_native_attention_precision
    from natlang_neuralese.model import native_parameter_precision
    model,opt,named,roles,config=fixture('lfm');hook=inherited_cached_gap(model)
    try:report=call(model,opt,roles,config)
    finally:hook.remove()
    assert report['transition_equivalence']['passed']and not report['passed']
    # Exercise the combination-control guard with a genuine computed report;
    # CPU fixtures do not claim CUDA-only bf16x6 arithmetic qualification.
    set_native_attention_precision(model,'tf32x3');config.update(native_attention_precision='tf32x3',native_linear_precision='bf16x6'if other_change=='linear'else'ieee')
    monkeypatch.setattr(transition,'qualify_attention_transition',lambda *a,**kw:report)
    if other_change=='norm':monkeypatch.setattr(native_parameter_precision,'norm_execution_changes_arithmetic',lambda *a:True)
    with pytest.raises(ValueError,match='cached-reference'):
        qualify_precision_transition(model,Tokenizer(),roles,config,previous_precision='ieee',previous_attention_precision='ieee',previous_norm_execution='stock'if other_change=='norm'else None,checkpoint_sha256='a'*64,forward_atol=1e-4,gradient_atol=1e-5,device='cpu',optimizer=opt)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_real_inactive_expert_derivative_remains_covered_and_equivalent(family):
    model,opt,named,roles,config=fixture(family,inactive=True);state=copy.deepcopy(opt.state_dict());hook=inherited_cached_gap(model)
    try:report=call(model,opt,roles,config)
    finally:hook.remove()
    inactive=next(name for name,p in named if 'inactive_expert'in name)
    for proof in[report['baseline_cached_reference']['complete_gradient_qualification'],report['candidate_cached_reference']['complete_gradient_qualification'],*report['cross_mode'].values()]:
        assert inactive in {r['name']for r in proof['inventory']}
        assert proof['comparisons'][inactive]=={'passed':True,'unused':True}
    assert report['transition_equivalence']['passed']and not report['passed']
    equal(opt.state_dict(),state)
    report['candidate_cached_reference']['complete_gradient_qualification']['comparisons'][inactive]={'passed':False,'reason':'flat-only or missing cached derivative'}
    assert not transition.transition_equivalence(report)['passed']
