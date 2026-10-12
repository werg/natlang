"""MuonSR's fused elementwise step (train/muon_kernels.py) is bit-identical to the eager step with the eager noise
draws: weights, momentum and the running expert gradient norms over several updates, with idle experts, split rows,
weight decay on and off. The in-kernel Philox draws round without bias and differ per call."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('the fused MuonSR step is CUDA only', allow_module_level=True)

from natlang_neuralese.train.optim import MuonSR


def _run(fused, monkeypatch, weight_decay):
    monkeypatch.setenv('NATLANG_MUON_FUSED', '1' if fused else '0')
    monkeypatch.setenv('NATLANG_MUON_SR_NOISE', 'torch')  # the eager stochastic_round_ draws
    torch.manual_seed(0)
    stack = torch.nn.Parameter((torch.randn(6, 64, 48, device='cuda') * 0.02).bfloat16())
    matrix = torch.nn.Parameter((torch.randn(96, 40, device='cuda') * 0.02).bfloat16())
    optimizer = MuonSR([{'params': [stack], 'split_rows': 2}, {'params': [matrix]}], lr=1e-3,
                       weight_decay=weight_decay, ns='cubic5')
    for step in range(3):
        stack.grad = (torch.randn_like(stack, dtype=torch.float32) * 1e-2).bfloat16()
        stack.grad[step % 6] = 0  # an idle expert
        matrix.grad = (torch.randn_like(matrix, dtype=torch.float32) * 1e-2).bfloat16()
        torch.manual_seed(100 + step)  # the stochastic rounding draws
        optimizer.step()
    state = optimizer.state[stack]
    return [stack.detach().clone(), matrix.detach().clone(), state['momentum'].clone(), state['grad_norm'].clone(),
            optimizer.state[matrix]['momentum'].clone()]


@pytest.mark.parametrize('weight_decay', [0.0, 0.1])
def test_fused_muon_step_is_bit_identical(monkeypatch, weight_decay):
    eager = _run(False, monkeypatch, weight_decay)
    fused = _run(True, monkeypatch, weight_decay)
    for name, a, b in zip(('stack', 'matrix', 'momentum', 'grad_norm', 'matrix_momentum'), eager, fused):
        assert torch.equal(a, b), (name, (a.float() - b.float()).abs().max())


def test_philox_stochastic_rounding_is_unbiased(monkeypatch):
    """``apply_update`` with in-kernel noise: a step far below BF16 resolution survives in expectation, every
    result is one of the two neighbouring BF16 values, and two calls draw different noise."""
    from natlang_neuralese.train.muon_kernels import apply_update
    monkeypatch.setenv('NATLANG_MUON_SR_NOISE', 'philox')
    start = torch.full((1024, 1024), 1.0, device='cuda', dtype=torch.bfloat16)
    step = 2 ** -7 * 0.3  # 0.3 of the BF16 spacing at 1.0
    w1, w2 = start.clone(), start.clone()
    direction = torch.ones(1024, 1024, device='cuda', dtype=torch.bfloat16)
    apply_update(w1, direction, None, None, step)
    apply_update(w2, direction, None, None, step)
    assert set(w1.float().unique().tolist()) <= {1.0, 1.0 + 2 ** -7}
    assert abs(float(w1.float().mean()) - (1.0 + step)) < 2 ** -7 * 0.01
    assert not torch.equal(w1, w2)
