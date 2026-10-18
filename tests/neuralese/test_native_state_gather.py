"""Wide fan-in is exact for shared histories without wide CUDA history buffers."""
import pytest
import torch
from natlang_neuralese.model.state_gather import gather_convolution_state


@pytest.mark.parametrize('shape,index',[
    ((2,7,13),[[0,1,1],[2,-1,0],[1,5,1]]),
    ((1,2,8),[]),
    ((3,5,9),[1,2,1,1,4,-1]),
])
def test_duplicate_and_negative_shared_sources_match_reference(shape,index):
    torch.manual_seed(31)
    source=torch.randn(shape,dtype=torch.float64,requires_grad=True)
    ids=torch.tensor(index,dtype=torch.int64)
    output=gather_convolution_state(source,ids)
    reference=source[:,ids].float()
    assert output.dtype==torch.float32 and torch.equal(output,reference)
    grad=torch.randn_like(output)
    actual=torch.autograd.grad(output,source,grad)[0]
    expected=torch.autograd.grad(reference,source,grad)[0]
    assert actual.dtype==torch.float64 and torch.equal(actual,expected)


def test_bad_index_and_geometry_are_not_silently_changed():
    x=torch.randn(1,3,8,dtype=torch.float64)
    for index in (torch.tensor([3]),torch.tensor([-4])):
        with pytest.raises(IndexError):gather_convolution_state(x,index)
    with pytest.raises(ValueError):gather_convolution_state(x.float(),torch.tensor([0]))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA state-gather kernel required')
@pytest.mark.parametrize('channels',[13,1024])
def test_actual_cuda_values_gradients_and_workspace(channels):
    torch.manual_seed(71)
    x=torch.randn(2,4099,channels,device='cuda',dtype=torch.float64,requires_grad=True)
    index=torch.stack([torch.arange(4099,device='cuda')%4013,
        torch.arange(4099,device='cuda').flip(0)%4013,
        torch.full((4099,),-1,device='cuda')],1)
    # Compile before measuring allocation. Only a FP32 history plus one FP64
    # source adjoint is needed, never an additional FP64 history expansion.
    gather_convolution_state(x,index).sum().backward();x.grad=None
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    before=torch.cuda.memory_allocated()
    result=gather_convolution_state(x,index)
    g=torch.randn_like(result)
    actual=torch.autograd.grad(result,x,g)[0]
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-before
    reference=x[:,index].float()
    expected=torch.autograd.grad(reference,x,g)[0]
    assert torch.equal(result,reference)
    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-12)
    allowed=result.numel()*4*2+x.numel()*8+index.numel()*8+8*1024**2
    assert peak<=allowed,(peak,allowed)
