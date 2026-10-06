from types import SimpleNamespace

import torch
from torch import nn

from natlang_neuralese.model.causal_feedback import CausalFeedbackProjection


class NativeNorm(nn.Module):
    """LFM's cast-before-gain convention, including BF16 rounding."""
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.7, 1.4, width).bfloat16())

    def forward(self, states):
        normalized = states.float() * torch.rsqrt(states.float().square().mean(-1, keepdim=True) + 1e-5)
        return self.weight * normalized.to(states.dtype)


def backbone():
    torch.manual_seed(6)
    table = torch.randn(32, 8).bfloat16()
    rows = torch.randn(2, 8)
    norm = NativeNorm(8)
    def embed(ids):
        value = table[ids].clone()
        for index, row in zip([2, 3], rows):
            value[ids == index] = row.to(value)
        return value
    def logits(states):
        value = norm(states)
        result = value @ table.t()
        result[..., 2] = value @ rows[0].to(value)
        result[..., 3] = value @ rows[1].to(value)
        return result
    return SimpleNamespace(embedding_weight=table, output_weight=table, norm_eps=1e-5,
                           controls=SimpleNamespace(open_id=2, close_id=3), control_rows=rows,
                           hf=SimpleNamespace(model=SimpleNamespace(embedding_norm=norm)),
                           logits=logits, embed=embed)


def test_full_output_initialization_is_exact_even_with_bf16_and_controls():
    base = backbone()
    projection = CausalFeedbackProjection(base)
    states = torch.randn(3, 7, 8).bfloat16()
    assert torch.equal(projection.logits(states), base.logits(states))
    assert torch.equal(projection(states), base.embed(base.logits(states).argmax(-1)))


def test_surrogate_changes_backward_only_and_roundtrips():
    base = backbone()
    projection = CausalFeedbackProjection(base)
    states = torch.randn(3, 7, 8).bfloat16()
    assert torch.equal(projection(states, straight_through=True), projection(states))
    projection(states, straight_through=True).float().square().mean().backward()
    assert projection.state_out.weight.grad is not None
    assert torch.isfinite(projection.state_out.weight.grad).all()
    restored = CausalFeedbackProjection(base)
    restored.load_state_dict(projection.state_dict())
    assert torch.equal(restored(states), projection(states))
    assert not projection.final_norm.weight.requires_grad


def test_long_projection_chunks_preserve_primal_and_all_adjoints():
    import copy
    projection = CausalFeedbackProjection(backbone()).float()
    with torch.no_grad():
        projection.state_out.weight.normal_(0, .03)
    reference = copy.deepcopy(projection)
    state = torch.randn(2, 257, 8, requires_grad=True)
    other = state.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(state)
    actual = projection(state, straight_through=True)
    expected = reference._project_tokens(other, straight_through=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(state.grad, other.grad, atol=2e-5, rtol=2e-5)
    for (name, parameter), (_, original) in zip(projection.named_parameters(), reference.named_parameters()):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, original.grad, atol=2e-4, rtol=2e-5, msg=name)
    with torch.no_grad():
        assert torch.equal(projection(state), reference._project_tokens(state, straight_through=False))
