"""Original occurrence VJPs while fixed native projections reuse primal rows."""
import copy
from contextlib import nullcontext
import pytest
import torch
from test_shared_state import tiny
from test_native_communication_prefix import episodes
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.model.native_gradient_partitions import primal_row_reuse
from natlang_neuralese.train.native_state_sft import pack_native_graphs
from natlang_neuralese.train.native_prefix_packing import communication_primal_rows


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_cuda_primal_reuse_retains_every_original_occurrence_gradient(family,dtype,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from natlang_neuralese.model.native_parameter_precision import configure_norm_precision
    from natlang_neuralese.model.native_precision import set_native_linear_precision
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope
    from natlang_neuralese.train.backbone_policy import configure_backbone_training
    from natlang_neuralese.maple import fused_moe
    model=SharedStateModel(tiny(family).to(device='cuda',dtype=dtype),(58,59,60,61),rank=4,
        native_norm_execution='stable-v1')
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    configure_norm_precision(model,'fp32-norms-v1')
    set_native_linear_precision(model,'bf16x6')
    configure_backbone_training(model.port,'latent')
    model.port.checkpoint_layers=True
    with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    donor=copy.deepcopy(model)
    examples=episodes(empty_relay=True)
    plan,_,_,_=pack_native_graphs(examples)
    aliases=communication_primal_rows(examples)
    assert len(set(aliases))<len(aliases)
    calls=[]
    original=fused_moe._invariant_linear_forward
    def measured(x,w,precision):
        calls.append(x.shape[0])
        return original(x,w,precision)
    monkeypatch.setattr(fused_moe,'_invariant_linear_forward',measured)
    results=[]
    for owner,reuse in [(donor,False),(model,True)]:
        calls.clear()
        with ParameterCotangentScope.for_model(owner,mode='vjp') as scope:
            with primal_row_reuse(plan,aliases) if reuse else nullcontext():
                result=owner(plan,retain_states=True)
                # Distinct per-occurrence signed cotangents exercise BF16 casts.
                weight=torch.arange(result['h_final'].numel(),device='cuda').reshape(result['h_final'].shape)%17-8
                loss=(result['h_final'].float()*weight/100).sum()
            loss.backward();scope.finish()
        results.append((result,tuple(calls)))
    if dtype==torch.bfloat16:
        assert results[1][1].count(len(set(aliases)))>results[0][1].count(len(set(aliases)))
    else:
        assert results[1][1]==results[0][1]
    torch.testing.assert_close(results[0][0]['h_final'],results[1][0]['h_final'],atol=0,rtol=0)
    for left,right in zip(results[0][0]['states'],results[1][0]['states']):
        for a,b in zip(left,right):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for (name,p),(other,q) in zip(model.named_parameters(),donor.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('kind',['communication','ordinary'])
def test_cuda_two_bf16_updates_preserve_all_optimizer_state_and_rng(family,kind):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from natlang_neuralese.model.native_parameter_precision import configure_norm_precision
    from natlang_neuralese.model.native_precision import set_native_linear_precision
    from natlang_neuralese.train.backbone_policy import configure_backbone_training
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
    model=SharedStateModel(tiny(family).to(device='cuda',dtype=torch.bfloat16),(58,59,60,61),rank=4,
        native_norm_execution='stable-v1')
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    configure_norm_precision(model,'fp32-norms-v1')
    set_native_linear_precision(model,'bf16x6')
    configure_backbone_training(model.port,'latent')
    with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    donor=copy.deepcopy(model)
    model.share_causal_prefixes=True
    optimizer,_=make_native_optimizer(model,lr=1e-5,in_backward=True)
    original,_=make_native_optimizer(donor,lr=1e-5,in_backward=True)
    def exact(a,b,path):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0,msg=path)
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for key in a:exact(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),path
            for index,(left,right) in enumerate(zip(a,b)):exact(left,right,path+'/'+str(index))
        else:assert a==b,path
    from test_native_prefix_packing import examples as ordinary_examples
    rows=episodes(empty_relay=True) if kind=='communication' else ordinary_examples()
    for step in range(2):
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        expected,details= native_training_update(donor,original,rows,chunk_tokens=3)
        after_cpu=torch.get_rng_state();after_gpu=torch.cuda.get_rng_state()
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        actual,current=native_training_update(model,optimizer,rows,chunk_tokens=3)
        exact(actual,expected,'loss')
        exact(model.state_dict(),donor.state_dict(),'weights')
        exact(optimizer.state_dict(),original.state_dict(),'optimizer')
        exact(torch.get_rng_state(),after_cpu,'cpu_rng')
        exact(torch.cuda.get_rng_state(),after_gpu,'cuda_rng')


def test_primal_binding_is_exact_plan_scoped_and_validates_representatives():
    from natlang_neuralese.model.native_gradient_partitions import bound_primal_rows
    plan,_,_,_=pack_native_graphs(episodes())
    aliases=communication_primal_rows(episodes())
    other=copy.copy(plan)
    assert bound_primal_rows(plan) is None
    with primal_row_reuse(plan,aliases):
        assert bound_primal_rows(plan).rows==aliases
        assert bound_primal_rows(other) is None
    assert bound_primal_rows(plan) is None
    for invalid in [(0,),tuple([plan.token_count]*plan.token_count)]:
        with pytest.raises(ValueError,match='representatives'):
            with primal_row_reuse(plan,invalid):pass
