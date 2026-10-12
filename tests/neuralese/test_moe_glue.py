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
