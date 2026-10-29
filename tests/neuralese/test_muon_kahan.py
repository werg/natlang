"""MuonSR Kahan rounding: BF16 matrices follow the FP32 trajectory instead of a stochastic-rounding random walk."""
import copy

import torch

from natlang_neuralese.train.optim import MuonSR


def run(dtype, rounding, steps=300, lr=2e-5):
    torch.manual_seed(5)
    weight = (torch.randn(64, 128) * .02).to(torch.bfloat16).float()  # one representable start for every run
    parameter = torch.nn.Parameter(weight.to(dtype))
    optimizer = MuonSR([{'params': [parameter], 'lr': lr, 'weight_decay': .1, 'name': 'w'}], lr=lr,
                       rounding=rounding)
    generator = torch.Generator().manual_seed(9)
    for _ in range(steps):
        # Identical BF16-representable gradients for every precision, so only the weight writes differ.
        parameter.grad = torch.randn(64, 128, generator=generator).to(torch.bfloat16).to(dtype)
        optimizer.step()
    return parameter.detach().float(), optimizer


def test_kahan_tracks_the_fp32_trajectory_where_stochastic_rounding_drifts():
    reference, _ = run(torch.float32, 'stochastic')
    kahan, optimizer = run(torch.bfloat16, 'kahan')
    stochastic, _ = run(torch.bfloat16, 'stochastic')
    # The represented value of a Kahan latent is its BF16 weight plus the carried residual.
    state = optimizer.state[optimizer.param_groups[0]['params'][0]]
    kahan_error = (kahan + state['compensation'].float() - reference).square().mean().sqrt()
    stochastic_error = (stochastic - reference).square().mean().sqrt()
    assert kahan_error < .1 * stochastic_error, (float(kahan_error), float(stochastic_error))
    # The BF16 weight itself stays within rounding of the reference instead of drifting.
    assert (kahan - reference).abs().max() < 2 ** -7 * reference.abs().max()
    assert state['compensation'].dtype == torch.bfloat16 and state['compensation'].abs().max() > 0


def test_kahan_residual_survives_a_state_round_trip_and_rounding_is_validated():
    _, optimizer = run(torch.bfloat16, 'kahan', steps=5)
    saved = copy.deepcopy(optimizer.state_dict())
    parameter = torch.nn.Parameter(optimizer.param_groups[0]['params'][0].detach().clone())
    restored = MuonSR([{'params': [parameter], 'lr': 2e-5, 'weight_decay': .1, 'name': 'w'}], lr=2e-5, rounding='kahan')
    restored.load_state_dict(saved)
    original = optimizer.state[optimizer.param_groups[0]['params'][0]]['compensation']
    assert torch.equal(restored.state[parameter]['compensation'], original)
    assert restored.state[parameter]['compensation'].dtype == torch.bfloat16
    try:
        MuonSR([{'params': [parameter], 'lr': 1., 'name': 'w'}], lr=1., rounding='floor')
    except ValueError as error:
        assert 'rounding' in str(error)
    else:
        raise AssertionError('unknown rounding accepted')
