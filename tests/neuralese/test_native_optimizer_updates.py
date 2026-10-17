"""Actual current optimizer recurrences, inherited history and nonmutation."""
import copy
import pytest
import torch
from natlang_neuralese.train.optim import PortMuonAdamW,MuonSR
from natlang_neuralese.eval.native_optimizer_updates import CompleteOptimizerComparison
from natlang_neuralese.eval.native_gradients import compare_complete_gradients
from test_native_shared_checkpoint import equal_tree


def fixture(dtype=torch.float32,device='cpu',matrix_shape=(16,12)):
    named=[('norm',torch.nn.Parameter(torch.ones(12,device=device))),
        ('rows',torch.nn.Parameter(torch.randn(4,12,device=device)*.01)),
        ('backbone',torch.nn.Parameter(torch.randn(*matrix_shape,dtype=dtype,device=device)*.02))]
    optimizer=PortMuonAdamW(named[:2],lr=3e-6,vocab_size=64,
        force_adamw_ids={id(p)for _,p in named[:2]},
        latent=[('backbone',named[-1][1],3e-6,1)],latent_optimizer='muonsr',
        latent_options={'weight_decay':.1,'ns':'quintic'})
    # Meaningful nonzero saved history and non-default per-group learning rates.
    for group in optimizer.auxiliary.param_groups:group['lr']=3e-4
    for _,p in named:p.grad=torch.randn_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return named,optimizer


def clone_fixture(named,optimizer):
    fresh_named,fresh_optimizer=fixture(named[-1][1].dtype,named[-1][1].device,tuple(named[-1][1].shape))
    with torch.no_grad():
        for (_,source),(_,target)in zip(named,fresh_named):target.copy_(source)
    fresh_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return fresh_named,fresh_optimizer


def whole_order(optimizer,named):
    names={id(p):name for name,p in named}
    return [names[id(p)]for child in(optimizer.muon,optimizer.auxiliary,optimizer.latent)if child is not None
        for group in child.param_groups for p in group['params']]


def _whole_control(dtype,device='cpu',matrix_shape=(16,12),split_rows=1):
    torch.manual_seed(4521);named,optimizer=fixture(dtype,device,matrix_shape)
    optimizer.latent.param_groups[0]['split_rows']=split_rows
    gradients={name:torch.randn_like(p)*.1 for name,p in named}
    if len(matrix_shape)>2:gradients['backbone'][-1].zero_()
    state=copy.deepcopy(optimizer.state_dict());parameters={name:p.detach().clone()for name,p in named}
    original_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device).clone()if str(device).startswith('cuda')else None
    observer=CompleteOptimizerComparison(optimizer,named)
    # Capture each actual whole-optimizer output for exact comparison with its
    # one-parameter counterpart, using the same real update and RNG sequence.
    shadow_named,shadow=clone_fixture(named,optimizer)
    torch.set_rng_state(original_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,device)
    for name,p in shadow_named:p.grad=gradients[name].clone()
    shadow.step();whole_rng=torch.get_rng_state().clone()
    whole_cuda_rng=torch.cuda.get_rng_state(device).clone()if cuda_rng is not None else None
    expected={name:p.detach().clone()for name,p in shadow_named}
    expected_states={shadow._parameter_names[id(p)]:copy.deepcopy(child.state[p])
        for child in(shadow.muon,shadow.auxiliary,shadow.latent)if child is not None
        for group in child.param_groups for p in group['params']}
    torch.set_rng_state(original_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,device)
    actual_shadow=observer._shadow;seen={};seen_states={}
    def inspect(name,gradient):
        p,child=actual_shadow(name,gradient);step=child.step
        def update():
            step();seen[name]=p.detach().clone();seen_states[name]=copy.deepcopy(child.state[p])
        child.step=update
        return p,child
    observer._shadow=inspect
    for name in whole_order(optimizer,named):observer(name,gradients[name],gradients[name])
    report=observer.report()
    assert report['passed'] and report['source_optimizer_updates']==0
    assert all(row['stored_weight_disagreement_coordinates']==0 for row in report['comparisons'].values())
    assert all(row['flat_changed_coordinates']>0 for row in report['comparisons'].values())
    for name in expected:
        assert torch.equal(seen[name],expected[name])
        equal_tree(seen_states[name],expected_states[name])
    assert torch.equal(observer.private_rng['cpu'],whole_rng)
    assert torch.equal(torch.get_rng_state(),original_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device),cuda_rng)
        assert torch.equal(observer.private_rng['cuda'][torch.device(device).index or 0],whole_cuda_rng)
    equal_tree(optimizer.state_dict(),state)
    for name,p in named:assert torch.equal(p,parameters[name])and p.grad is None


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_same_real_whole_optimizer_update_matches_parameter_composition_and_preserves_source(dtype):
    _whole_control(dtype)


def test_whole_optimizer_split_and_idle_experts_keep_actual_moments_and_decay():
    _whole_control(torch.float32,matrix_shape=(3,16,12),split_rows=2)


@pytest.mark.gpu_heavy
@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA fused MuonSR/Philox control')
def test_real_cuda_fused_muonsr_philox_matches_whole_optimizer_and_preserves_rng():
    _whole_control(torch.bfloat16,'cuda',matrix_shape=(512,384),split_rows=2)


def test_real_inherited_moments_are_used_and_full_backbone_perturbation_is_visible():
    named,optimizer=fixture();before=copy.deepcopy(optimizer.state_dict());observer=CompleteOptimizerComparison(optimizer,named)
    for name,p in named:
        gradient=torch.randn_like(p)*.1
        cached=gradient.clone()
        if name=='backbone':cached[0,0]+=.1
        observer(name,gradient,cached)
    report=observer.report();row=report['comparisons']['backbone']
    assert not report['passed'] and not row['gradient']['passed']
    assert row['stored_weight_disagreement_coordinates']>0 and row['flat_changed_coordinates']>0
    assert not row['stored_update']['passed']
    assert row['moments']['momentum']['mismatched_elements']>0
    assert all(item['passed']for name,item in report['comparisons'].items()if name!='backbone')
    equal_tree(optimizer.state_dict(),before)


def test_zero_stored_updates_do_not_hide_gradient_failure():
    named,optimizer=fixture()
    for group in optimizer.param_groups:group['lr']=0.
    observer=CompleteOptimizerComparison(optimizer,named)
    for name,p in named:
        observer(name,torch.ones_like(p),torch.ones_like(p)*2)
    report=observer.report()
    assert not report['passed']
    assert all(row['zero_updates_with_nonzero_gradient']for row in report['comparisons'].values())
    assert all(row['stored_weight_disagreement_coordinates']==0 for row in report['comparisons'].values())


def test_missing_unused_and_duplicate_parameter_coverage_rejects():
    named,optimizer=fixture();observer=CompleteOptimizerComparison(optimizer,named)
    observer('norm',None,None)
    with pytest.raises(ValueError,match='omitted'):observer.report()
    with pytest.raises(ValueError,match='repeated'):observer('norm',None,None)
    observer('rows',None,torch.zeros_like(named[1][1]))
    observer('backbone',None,None)
    report=observer.report();assert not report['passed'] and report['comparisons']['norm']['unused']
    with pytest.raises(ValueError,match='inventory differ'):CompleteOptimizerComparison(optimizer,named[:-1])


def test_observer_error_preserves_rng_parameters_and_optimizer():
    named,optimizer=fixture();observer=CompleteOptimizerComparison(optimizer,named)
    before=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone();actual=observer._shadow
    def broken(name,gradient):
        p,child=actual(name,gradient)
        def fail():torch.rand(8);raise RuntimeError('controlled observer failure')
        child.step=fail;return p,child
    observer._shadow=broken
    with pytest.raises(RuntimeError,match='controlled observer failure'):
        observer('backbone',torch.ones_like(named[-1][1]),torch.ones_like(named[-1][1]))
    assert torch.equal(torch.get_rng_state(),rng)
    equal_tree(optimizer.state_dict(),before);observer.assert_preserved()


def test_existing_complete_vjp_comparison_observes_all_parameters_without_extra_forward():
    named,optimizer=fixture();observer=CompleteOptimizerComparison(optimizer,named)
    inputs=torch.randn(2,12,requires_grad=True);cached_inputs=inputs.detach().clone().requires_grad_()
    calls=[]
    def loss(x):
        calls.append(1)
        return sum((p.float().sum()*x.square().mean())for _,p in named)
    flat=loss(inputs)
    proof=compare_complete_gradients(flat,None,named,inputs,None,atol=1e-5,rtol=1e-4,
        cached_factory=lambda:(loss(cached_inputs),cached_inputs),gradient_observer=observer)
    assert proof['passed'] and len(calls)==2
    assert observer.report()['passed']
    assert all(p.grad is None for _,p in named)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_tiny_both_port_complete_graph_reaches_every_optimizer_parameter(tmp_path,family):
    from test_native_full_updates import source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.eval.native_state import qualify_existing
    base=tmp_path/family;dtype=source(base,family)
    model,tokenizer,roles,config=load_native_model(base,dtype=dtype,rank=4)
    optimizer,named=make_native_optimizer(model,lr=3e-6,in_backward=False)
    # Both real ports use the actual declared trainable inventory; this is not
    # a readiness proof, and any strict gradient discrepancy stays reported.
    for _,p in named:p.grad=torch.randn_like(p)*.01
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    observer=CompleteOptimizerComparison(optimizer,named)
    before={name:p.detach().clone()for name,p in named};state=copy.deepcopy(optimizer.state_dict())
    model.port.checkpoint_layers=True
    result=qualify_existing(model,tokenizer,roles,config,checkpoint_sha256='a'*64,
        forward_atol=1e-4,gradient_atol=1e-5,gradient_parameters=named,gradient_observer=observer)
    report=observer.report()
    equal_tree(optimizer.state_dict(),state)
    for name,p in named:assert torch.equal(p,before[name])
    assert len(report['comparisons'])==len(named)
    assert set(report['comparisons'])=={name for name,_ in named}
    assert result['complete_gradient_qualification']['parameter_grads_preserved']
    assert report['diagnostic_only']and not report['readiness_qualification']
    assert any(row.get('flat_changed_coordinates',0)>0 for row in report['comparisons'].values())


def test_rng_changing_shadow_optimizers_run_only_after_complete_vjp_returns():
    named,optimizer=fixture(torch.bfloat16)
    observer=CompleteOptimizerComparison(optimizer,named)
    shadow=observer._shadow;steps=[]
    def checked(name,gradient):
        assert torch._C._current_autograd_node()is None
        steps.append(name);return shadow(name,gradient)
    observer._shadow=checked
    inputs=torch.randn(2,12,requires_grad=True);cached_inputs=inputs.detach().clone().requires_grad_()
    class StatefulBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx,value):return value.clone()
        @staticmethod
        def backward(ctx,gradient):return gradient*torch.rand((),device=gradient.device)
    def loss(x):return StatefulBackward.apply(sum(p.float().sum()*x.square().mean()for _,p in named))
    rng=torch.get_rng_state().clone()
    flat=loss(inputs)
    def cached():
        torch.set_rng_state(rng)
        return loss(cached_inputs),cached_inputs
    proof=compare_complete_gradients(flat,None,named,inputs,None,atol=1e-5,rtol=1e-4,
        cached_factory=cached,gradient_observer=observer)
    assert proof['passed']and observer.report()['passed']
    assert len(steps)==2*len(named)
    expected=torch.get_rng_state().clone()
    torch.set_rng_state(rng);torch.rand(())
    assert torch.equal(torch.get_rng_state(),expected)
    report=observer.report()
    assert report['retained_vjp_cpu_snapshot_bytes']==2*sum(p.numel()*p.element_size()for _,p in named)


def test_backward_observer_failure_drops_cpu_snapshots_without_source_mutation():
    named,optimizer=fixture();state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    parameters={name:p.detach().clone()for name,p in named}
    class Failing(CompleteOptimizerComparison):
        def __call__(self,name,flat,cached):
            super().__call__(name,flat,cached)
            if name!='input_embeddings':raise RuntimeError('controlled backward observer failure')
    observer=Failing(optimizer,named)
    x=torch.ones(2,12,requires_grad=True);other=x.detach().clone().requires_grad_()
    def loss(value):return sum(p.float().sum()*value.sum()for _,p in named)
    with pytest.raises(RuntimeError,match='controlled backward observer failure'):
        compare_complete_gradients(loss(x),None,named,x,None,atol=1e-5,rtol=1e-4,
            cached_factory=lambda:(loss(other),other),gradient_observer=observer)
    assert observer.buffer is None and observer.aborted
    with pytest.raises(ValueError,match='aborted'):observer.report()
    assert torch.equal(torch.get_rng_state(),rng)
    equal_tree(optimizer.state_dict(),state)
    for name,p in named:assert torch.equal(p,parameters[name])and p.grad is None
