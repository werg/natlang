import os,signal,subprocess,sys


def test_compiled_caller_preserves_chunk_boundaries_and_gradients():
    import torch
    from natlang_neuralese.model.native_gradient_partitions import GradientPartitions,gradient_partitions,token_local_chunks
    calls=[]
    def operation(x):
        def callback(part):
            calls.append(part.shape[1])
            return part.square()
        with gradient_partitions(GradientPartitions((3,2))):
            return token_local_chunks(x,2,callback)
    compiled=torch.compile(operation,backend='eager')
    x=torch.arange(5.,requires_grad=True).reshape(1,5,1);x.retain_grad()
    y=compiled(x);y.sum().backward()
    assert torch.equal(y,x.square())
    assert torch.equal(x.grad,2*x)
    assert calls==[2,1,2]


def test_compiled_caller_norm_wrapper_preserves_values_and_gradients():
    import torch
    from natlang_neuralese.model.native_parameter_precision import rowwise_norm_forward
    gain=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    def operation(x):
        return rowwise_norm_forward(lambda value:value.square()*gain,x)
    compiled=torch.compile(operation,backend='eager')
    x=torch.arange(15.,requires_grad=True).reshape(5,3);x.retain_grad()
    out=compiled(x);out.sum().backward()
    assert torch.equal(out,x.square()*gain)
    assert torch.equal(x.grad,2*x*gain)
    assert torch.equal(gain.grad,x.square().sum(0))
