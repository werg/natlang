"""Option B: exact readout values, TCP gradients, and no causal-history backbone leak."""
import copy

import torch
import pytest

from natlang_neuralese.model.neuralese_projection import TokenChoiceProjection, project_token_states
from natlang_neuralese.train.direct_sft import detached_target_mask


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_detached_output_preserves_readout_and_tcp_gradient_without_backbone_history_gradient(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    projection = TokenChoiceProjection(4, blocks=2, hidden=7, kernel=3).to(device)
    with torch.no_grad():
        for parameter in projection.parameters():
            parameter.uniform_(-0.3, 0.3)
    reference = copy.deepcopy(projection)
    hidden = torch.randn(1, 6, 4, requires_grad=True, device=device)
    segments = torch.tensor([[0, 0, 0, 0, 1, 1]], device=device)
    mask = torch.tensor([[False, False, False, True, False, False]], device=device)
    read = project_token_states(hidden, projection, segments, mask)
    expected = reference(hidden.detach(), segments)
    assert torch.equal(read, expected)
    read[mask].square().sum().backward()
    expected[mask].square().sum().backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))
    for actual, wanted in zip(projection.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=0, atol=0)
    assert any(p.grad.abs().sum() > 0 for p in projection.parameters())


def test_ordinary_outputs_keep_backbone_and_causal_history_gradients():
    projection = TokenChoiceProjection(3, blocks=1, hidden=5, kernel=3)
    with torch.no_grad():
        for parameter in projection.parameters():
            parameter.fill_(0.2)
    hidden = torch.randn(1, 5, 3, requires_grad=True)
    mask = torch.tensor([[False, False, False, True, False]])
    project_token_states(hidden, projection, None, mask)[0, 2].square().sum().backward()
    assert hidden.grad[0, :3].abs().sum() > 0
    assert hidden.grad[0, 3:].abs().sum() == 0


def test_detached_target_coordinates_follow_target_tokens_not_predictor_positions():
    window = {'start': 10, 'prefix': 3, 'ids': list(range(8))}
    assert detached_target_mask(window, [(12, 15), (17, 19)]) == [True, True, False, False, True]
