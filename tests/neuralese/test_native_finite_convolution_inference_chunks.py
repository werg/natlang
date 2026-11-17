import pytest
import torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('bias_enabled', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_inference_query_chunks_preserve_exact_finite_history(dtype, bias_enabled, device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.model import native_weight_carrier as carrier
    monkeypatch.setattr(carrier, '_FINITE_CONV_INFERENCE_QUERY_CHUNK', 5)
    torch.manual_seed(418)
    # Same strided rolling history layout as native_step, including a partial
    # final chunk and nontrivial batch/channel dimensions.
    source=torch.randn(2,19,23,device=device,dtype=dtype)
    histories=source.unfold(1,7,1).permute(0,1,3,2)
    weight=torch.randn(23,1,7,device=device,dtype=dtype)
    bias=torch.randn(23,device=device,dtype=dtype) if bias_enabled else None
    gate=torch.randn(2,13,23,device=device,dtype=dtype)
    def reference(h,w,b,c):
        out=(h.contiguous().float()*w[:,0].T.float()[None,None]).sum(2)
        if b is not None:out=out+b.float()
        return c*out.to(c.dtype)
    with torch.inference_mode():
        actual=carrier.stored_finite_convolution(histories,weight,bias,gate)
        expected=reference(histories,weight,bias,gate)
    assert torch.equal(actual,expected)
    assert torch.equal(actual[:,[4,5,9,10,12]],expected[:,[4,5,9,10,12]])
    # Even over the threshold, differentiable execution keeps the original
    # arithmetic and parameter/input/gate adjoints exactly.
    h=histories.detach().requires_grad_();w=weight.detach().requires_grad_();c=gate.detach().requires_grad_()
    b=bias.detach().requires_grad_() if bias is not None else None
    args=(h,w,c) if b is None else (h,w,c,b)
    actual=carrier.stored_finite_convolution(h,w,b,c)
    expected=reference(h,w,b,c)
    g=torch.randn_like(actual)
    left=torch.autograd.grad(actual,args,g,retain_graph=True)
    right=torch.autograd.grad(expected,args,g)
    assert torch.equal(actual,expected)
    assert all(torch.equal(x,y) for x,y in zip(left,right))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_inference_graph_convolution_gathers_bounded_blocks_bitwise(dtype, device, monkeypatch):
    """No-grad graph convolution gathers the K-fold histories one row block at a time, bit for bit."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from types import SimpleNamespace
    from natlang_neuralese.model import native_weight_carrier as carrier
    from natlang_neuralese.model import shared_state
    torch.manual_seed(1116)
    lengths, width, channels = (6, 1, 13, 4), 7, 29
    rows = sum(lengths)
    plan = SimpleNamespace(token_count=rows, slot_stops=False,
                           invocations=[SimpleNamespace(token_ids=[0] * n) for n in lengths])
    layer = SimpleNamespace(conv=SimpleNamespace(conv=SimpleNamespace(
        weight=torch.randn(channels, 1, width, device=device, dtype=dtype),
        bias=torch.randn(channels, device=device, dtype=dtype))))
    gated = torch.randn(1, rows, channels, device=device, dtype=dtype)
    gate = torch.randn(1, rows, channels, device=device, dtype=dtype)
    # Histories reach back across invocations, repeat sources and pad with the zero row (-1).
    index = torch.randint(-1, rows, (rows, width), device=device)
    stops = torch.rand(rows, width, device=device) < 0.3
    noise = torch.randn(rows, width, channels, device=device, dtype=dtype)
    for kwargs in ({}, {'index': (index, stops)}, {'noise': noise}):
        kwargs.setdefault('index', index)
        with torch.inference_mode():
            monkeypatch.setattr(carrier, '_FINITE_CONV_INFERENCE_QUERY_CHUNK', 10**9)
            expected = shared_state._convolve_eager(layer, gated, gate, plan, range(rows), **kwargs)
            for block in (5, 7, rows - 1):
                monkeypatch.setattr(carrier, '_FINITE_CONV_INFERENCE_QUERY_CHUNK', block)
                actual = shared_state._convolve_eager(layer, gated, gate, plan, range(rows), **kwargs)
                assert torch.equal(actual, expected), (block, sorted(kwargs))
