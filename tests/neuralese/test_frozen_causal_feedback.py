from types import SimpleNamespace

import torch

from natlang_neuralese.model.causal_feedback import CausalFeedbackProjection


def test_frozen_zero_correction_preserves_primal_and_state_gradient():
    torch.manual_seed(5)
    table = torch.randn(19, 4)
    backbone = SimpleNamespace(embedding_weight=table, output_weight=table, tied=True,
                              norm_eps=1e-6, control_rows=table[-2:].clone(),
                              controls=SimpleNamespace(open_id=17, close_id=18),
                              hf=SimpleNamespace(model=SimpleNamespace(norm=torch.nn.Identity())))
    projection = CausalFeedbackProjection(backbone)
    for p in projection.parameters():
        p.requires_grad_(False)
    state = torch.randn(2, 3, 4, requires_grad=True)
    expected = projection(state)
    expected_grad = torch.autograd.grad(expected.square().sum(), state)[0]
    assert projection.configure_frozen_identity()
    actual = projection(state)
    actual_grad = torch.autograd.grad(actual.square().sum(), state)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    with torch.no_grad():
        projection.state_out.bias.fill_(.2)
    assert not torch.equal(projection.complete_state(state), state)
    assert not projection._frozen_identity
    with torch.no_grad():
        projection.state_out.bias.zero_()
    assert projection.configure_frozen_identity()
    projection.state_out.weight.requires_grad_(True)
    projection.complete_state(state).sum().backward()
    assert projection.state_out.weight.grad.abs().sum() > 0
    assert not projection._frozen_identity
    projection.state_out.weight.requires_grad_(False)
    projection.load_state_dict(projection.state_dict())
    assert not projection._frozen_identity
    # A full training checkpoint restore invalidates the cache; reconfigure
    # only after restoring values and freezing the reference again.
    restored_state = torch.randn(2, 3, 4)
    assert projection.configure_frozen_identity()
    assert torch.equal(projection.complete_state(restored_state), restored_state)
    with torch.no_grad():
        projection.state_out.bias.fill_(.1)
    assert not projection.configure_frozen_identity()
    with torch.no_grad():
        projection.state_out.bias.zero_()
    assert projection.configure_frozen_identity()
    projection.state_out.weight.requires_grad_(True)
    assert not projection.configure_frozen_identity()
    projection.state_out.weight.requires_grad_(False)
    with torch.no_grad():
        projection.state_out.bias.fill_(.2)
    assert not projection.configure_frozen_identity()
