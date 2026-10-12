"""The scatter-free grouped-MoE glue (maple/moe_glue.py) against the eager gather/scatter glue it replaces in
``dense_grouped_experts``: output, input gradient, routing-weight gradient and expert weight gradients."""
import pytest
import torch

if not torch.cuda.is_available() or not hasattr(torch, '_grouped_mm'):
    pytest.skip('the grouped MoE glue is CUDA only', allow_module_level=True)

from natlang_neuralese.maple.fused_moe import dense_grouped_experts


def _run(monkeypatch, glue, x, index, logits, gate_up, down):
    monkeypatch.setenv('NATLANG_MOE_GLUE', '1' if glue else '0')
    x, logits, gate_up, down = (t.detach().clone().requires_grad_(True) for t in (x, logits, gate_up, down))
    weights = torch.softmax(logits, -1)
    y = dense_grouped_experts(x, index, weights, gate_up, down, None)
    assert y.dtype == torch.float32
    g = torch.randn(y.shape, device=y.device, generator=torch.Generator(y.device).manual_seed(1))
    y.backward(g)
    return [y.detach(), x.grad.float(), logits.grad, gate_up.grad.float(), down.grad.float()]


@pytest.mark.parametrize('tokens', [1, 37, 900])
def test_glue_matches_the_eager_permutation(monkeypatch, tokens):
    torch.manual_seed(tokens)
    experts, top_k, width, ff = 8, 4, 256, 64
    x = torch.randn(tokens, width, device='cuda').bfloat16()
    logits = torch.randn(tokens, top_k, device='cuda')
    index = torch.stack([torch.randperm(experts, device='cuda')[:top_k] for _ in range(tokens)])
    gate_up = (torch.randn(experts, 2 * ff, width, device='cuda') * 0.05).bfloat16()
    down = (torch.randn(experts, width, ff, device='cuda') * 0.05).bfloat16()
    eager = _run(monkeypatch, False, x, index, logits, gate_up, down)
    glue = _run(monkeypatch, True, x, index, logits, gate_up, down)
    assert torch.equal(glue[0], eager[0])  # the forward: same products, same slot-order sum
    for name, a, b in zip(('grad_x', 'grad_weights', 'grad_gate_up', 'grad_down'), glue[1:], eager[1:]):
        assert float((a - b).norm() / b.norm().clamp_min(1e-30)) < 5e-3, name


def test_grouped_linear_grads_are_in_parameter_layout(monkeypatch):
    """maple/fp8_experts.GroupedLinear: equals torch._grouped_mm, with contiguous [E, N, K] weight grads."""
    from natlang_neuralese.maple.fp8_experts import GroupedLinear
    torch.manual_seed(0)
    experts, width, out_width = 4, 256, 128
    counts = torch.tensor([30, 0, 70, 28])
    offsets = counts.cumsum(0).to(torch.int32).cuda()
    rows = torch.randn(int(counts.sum()), width, device='cuda').bfloat16().requires_grad_(True)
    weight = (torch.randn(experts, out_width, width, device='cuda') * 0.05).bfloat16().requires_grad_(True)
    g = torch.randn(rows.shape[0], out_width, device='cuda').bfloat16()
    reference = torch._grouped_mm(rows, weight.transpose(-2, -1), offs=offsets)
    reference.backward(g)
    expected = [reference.detach().float(), rows.grad.float(), weight.grad.float()]
    for fp8, tolerance in ((False, 1e-2),):
        rows.grad = weight.grad = None
        y = GroupedLinear.apply(rows, weight, offsets, fp8)
        y.backward(g)
        assert weight.grad.is_contiguous() and weight.grad.shape == weight.shape
        for a, b in zip([y.detach().float(), rows.grad.float(), weight.grad.float()], expected):
            assert torch.isfinite(a).all() and float((a - b).norm() / b.norm()) < tolerance, fp8


@pytest.mark.parametrize('tokens', [5, 700])
def test_grouped_fp8_experts_track_bf16(monkeypatch, tokens):
    """maple/expert_gemm.fp8_dense_experts (DENSE_MOE["fp8"] = "grouped", the FP8 expert backend): finite, and within FP8 rounding of the BF16
    grouped experts in output and every gradient (an idle expert included)."""
    torch.manual_seed(tokens)
    experts, top_k, width, ff = 8, 4, 256, 128
    x = torch.randn(tokens, width, device='cuda').bfloat16()
    logits = torch.randn(tokens, top_k, device='cuda')
    index = torch.stack([torch.randperm(experts - 1, device='cuda')[:top_k] for _ in range(tokens)])  # expert 7 idle
    gate_up = (torch.randn(experts, 2 * ff, width, device='cuda') * 0.05).bfloat16()
    down = (torch.randn(experts, width, ff, device='cuda') * 0.05).bfloat16()
    from natlang_neuralese.maple.model import DENSE_MOE
    monkeypatch.setitem(DENSE_MOE, 'fp8', None)
    reference = _run(monkeypatch, True, x, index, logits, gate_up, down)
    monkeypatch.setitem(DENSE_MOE, 'fp8', 'grouped')
    fp8 = _run(monkeypatch, True, x, index, logits, gate_up, down)
    for name, a, b in zip(('out', 'grad_x', 'grad_weights', 'grad_gate_up', 'grad_down'), fp8, reference):
        assert torch.isfinite(a).all() and float((a - b).norm() / b.norm().clamp_min(1e-30)) < 0.1, name
    assert float(fp8[3][7].abs().max()) == 0.0  # the idle expert gets no gradient


def test_the_fp8_backend_selects_the_expert_path_only_inside_its_autocast():
    """train/fp8_backend.make_backend('triton-experts'): FP8 experts inside autocast (training), BF16 under
    native_modules (evaluation) and outside; a checkpoint envelope without scale state."""
    from types import SimpleNamespace
    from natlang_neuralese.maple.model import DENSE_MOE, DenseExperts
    from natlang_neuralese.train.fp8_backend import make_backend
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type='mellum')
    model.experts = DenseExperts(2, 16, 8)
    backend = make_backend('triton-experts', model, device='cuda')
    assert DENSE_MOE['fp8'] is None
    with backend.autocast():
        assert DENSE_MOE['fp8'] == 'grouped'
        with backend.native_modules():
            assert DENSE_MOE['fp8'] is None
        assert DENSE_MOE['fp8'] == 'grouped'
    assert DENSE_MOE['fp8'] is None and backend.has_executed
    backend.restore_extra_state(backend.extra_state())
    with pytest.raises(ValueError):
        make_backend('triton-experts', SimpleNamespace(config=SimpleNamespace(model_type='lfm2')), device='cuda')
