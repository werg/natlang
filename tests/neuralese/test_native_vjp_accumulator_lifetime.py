"""Retire storage only after the genuine once-cast leaf audit has completed."""
import copy
import pytest
import torch
from torch import nn
from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual device accumulator requires CUDA')


def test_complete_vjp_gradients_release_each_leaf_and_reduce_owned_peak(tmp_path):
    torch.manual_seed(723)
    source=nn.Sequential(nn.Linear(128,128,bias=False),nn.SiLU(),nn.Linear(128,128,bias=False),nn.SiLU(),nn.Linear(128,128,bias=False)).to(device='cuda',dtype=torch.bfloat16)
    models=[copy.deepcopy(source),copy.deepcopy(source)]
    x=torch.randn(7,128,device='cuda',dtype=torch.bfloat16)
    outputs=[];reports=[]
    rng=torch.cuda.get_rng_state().clone()
    for model,backend in zip(models,('cpu_file','device_fp32')):
        with ParameterCotangentScope.for_model(model,mode='vjp',directory=tmp_path,accumulator_backend=backend)as scope:
            loss=model(x).float().square().sum()
            outputs.append(torch.autograd.grad(loss,tuple(model.parameters())))
            scope.finish();reports.append(scope.allocation_report())
            if backend=='device_fp32':
                assert all(e['vjp_retired']and e['anchor_backward']==1 and e['count']for e in scope.entries.values())
                assert all(e['accumulator']is None for e in scope.entries.values())
        assert torch.equal(torch.cuda.get_rng_state(),rng)
    assert all(torch.equal(a,b)for a,b in zip(*outputs))
    report=reports[1]
    assert report['allocated_accumulator_bytes']==3*128*128*4
    assert report['live_accumulator_bytes']==0
    assert report['released_accumulator_bytes']==report['allocated_accumulator_bytes']
    assert 0<report['peak_live_accumulator_bytes']<report['allocated_accumulator_bytes']
    assert reports[0]['released_accumulator_bytes']==0


def test_retained_vjp_graph_and_late_contribution_fail_closed_after_audit(tmp_path):
    layer=nn.Linear(7,5,bias=False,device='cuda',dtype=torch.bfloat16)
    x=torch.randn(3,7,device='cuda',dtype=torch.bfloat16)
    with ParameterCotangentScope.for_model(layer,mode='vjp',directory=tmp_path,accumulator_backend='device_fp32')as scope:
        loss=layer(x).float().sum()
        torch.autograd.grad(loss,(layer.weight,),retain_graph=True)
        with pytest.raises(RuntimeError,match='retired VJP'):
            scope.add(layer.weight,torch.zeros_like(layer.weight,dtype=torch.float32))
        with pytest.raises(RuntimeError,match='retired VJP'):
            torch.autograd.grad(loss,(layer.weight,))
        scope.finish()


def test_uncarried_derivative_is_rejected_before_retiring_storage(tmp_path):
    layer=nn.Linear(7,5,bias=False,device='cuda',dtype=torch.bfloat16)
    x=torch.ones(3,7,device='cuda',dtype=torch.bfloat16)
    with ParameterCotangentScope.for_model(layer,mode='vjp',directory=tmp_path,accumulator_backend='device_fp32')as scope:
        loss=(layer(x)+torch.nn.functional.linear(x,layer.weight)).float().sum()
        with pytest.raises(RuntimeError,match='uncarried path'):
            torch.autograd.grad(loss,(layer.weight,))
        entry=scope.entries[id(layer.weight)]
        assert not entry['vjp_retired']and entry['accumulator']is not None
        assert scope.allocation_report()['released_accumulator_bytes']==0
