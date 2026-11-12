"""MuonSR Kahan rounding: BF16 matrices follow the FP32 trajectory instead of a stochastic-rounding random walk."""
import copy

import torch

from natlang_neuralese.train.optim import MuonSR


def run(dtype, rounding, steps=300, lr=2e-5, compensation_dtype=None):
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
        if compensation_dtype is not None:  # the buffer as stored before 2026-11-09: rounded after every update
            state = optimizer.state[parameter]
            state['compensation'] = state['compensation'].to(compensation_dtype)
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
    assert state['compensation'].dtype == torch.float32 and state['compensation'].abs().max() > 0


def test_fp32_compensation_tracks_the_fp32_trajectory_better_than_a_bf16_buffer():
    """2026-11-09 sanity test: the BF16 buffer's own rounding was 6-11% of the tracked update at production rates."""
    reference, _ = run(torch.float32, 'stochastic', steps=400, lr=2e-6)
    errors = {}
    for label, dtype in (('fp32', None), ('bf16', torch.bfloat16)):
        weight, optimizer = run(torch.bfloat16, 'kahan', steps=400, lr=2e-6, compensation_dtype=dtype)
        state = optimizer.state[optimizer.param_groups[0]['params'][0]]
        errors[label] = float((weight + state['compensation'].float() - reference).square().mean().sqrt())
    assert errors['fp32'] < .05 * errors['bf16'], errors


def test_bf16_compensation_is_promoted_exactly_on_load_and_lazily_at_the_step():
    _, optimizer = run(torch.bfloat16, 'kahan', steps=5, compensation_dtype=torch.bfloat16)
    parameter = optimizer.param_groups[0]['params'][0]
    legacy = optimizer.state[parameter]['compensation']
    assert legacy.dtype == torch.bfloat16
    saved = copy.deepcopy(optimizer.state_dict())
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = MuonSR([{'params': [restored_parameter], 'lr': 2e-5, 'weight_decay': .1, 'name': 'w'}], lr=2e-5,
                      rounding='kahan')
    restored.load_state_dict(saved)
    promoted = restored.state[restored_parameter]['compensation']
    assert promoted.dtype == torch.float32 and torch.equal(promoted, legacy.float())
    assert restored.promoted_compensation_buffers == 1
    # State that reached the optimizer through torch's dtype-casting loader is promoted at its next step.
    restored.state[restored_parameter]['compensation'] = legacy.clone()
    restored_parameter.grad = torch.ones_like(restored_parameter)
    restored.step()
    assert restored.state[restored_parameter]['compensation'].dtype == torch.float32


def test_float32_gradient_is_consumed_without_being_modified_and_matches_the_bf16_gradient_step():
    gradient = torch.randn(64, 128, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    results = []
    for dtype in (torch.bfloat16, torch.float32):
        parameter = torch.nn.Parameter((torch.randn(64, 128, generator=torch.Generator().manual_seed(4)) * .02).bfloat16())
        parameter.grad_dtype = dtype
        optimizer = MuonSR([{'params': [parameter], 'lr': 1e-3, 'weight_decay': .1, 'name': 'w'}], lr=1e-3,
                           rounding='kahan')
        parameter.grad = gradient.to(dtype)
        optimizer.step()
        assert torch.equal(parameter.grad, gradient.to(dtype))  # the Nesterov mix did not write into .grad
        results.append((parameter.detach().clone(), optimizer.state[parameter]['compensation'].clone()))
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])


def test_kahan_residual_survives_a_state_round_trip_and_rounding_is_validated():
    _, optimizer = run(torch.bfloat16, 'kahan', steps=5)
    saved = copy.deepcopy(optimizer.state_dict())
    parameter = torch.nn.Parameter(optimizer.param_groups[0]['params'][0].detach().clone())
    restored = MuonSR([{'params': [parameter], 'lr': 2e-5, 'weight_decay': .1, 'name': 'w'}], lr=2e-5, rounding='kahan')
    restored.load_state_dict(saved)
    original = optimizer.state[optimizer.param_groups[0]['params'][0]]['compensation']
    assert torch.equal(restored.state[parameter]['compensation'], original)
    assert restored.state[parameter]['compensation'].dtype == torch.float32
    try:
        MuonSR([{'params': [parameter], 'lr': 1., 'name': 'w'}], lr=1., rounding='floor')
    except ValueError as error:
        assert 'rounding' in str(error)
    else:
        raise AssertionError('unknown rounding accepted')
