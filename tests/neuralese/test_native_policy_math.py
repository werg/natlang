"""Native sampled-token objective uses the shared clipped policy loss."""
import torch
from natlang_neuralese.rl.policy_math import clipped_token_loss


def test_clipped_policy_loss_has_signed_gradients_and_finite_difference():
    old = torch.tensor([-2., -3.], dtype=torch.float64)
    current = torch.tensor([-2.05, -2.97], dtype=torch.float64, requires_grad=True)
    advantage = torch.tensor([.7, -.4])
    value = clipped_token_loss(current, old, advantage).sum()
    value.backward()
    for index in range(2):
        plus, minus = current.detach().clone(), current.detach().clone()
        plus[index] += .001
        minus[index] -= .001
        numeric = (clipped_token_loss(plus, old, advantage).sum() -
                   clipped_token_loss(minus, old, advantage).sum()) / .002
        assert torch.allclose(current.grad[index].float(), numeric, atol=1e-4)


def test_asymmetric_clips_zero_only_the_saturated_signed_direction():
    old = torch.tensor([-2.] * 4)
    now = (old + torch.tensor([0., .4, -.4, 0.])).requires_grad_()
    advantages = torch.tensor([1., 1., -1., -1.])
    clipped_token_loss(now, old, advantages).sum().backward()
    assert now.grad[0] < 0 and now.grad[3] > 0
    assert now.grad[1] == 0 and now.grad[2] == 0
    assert torch.isfinite(now.grad).all()
