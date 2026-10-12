"""MuonSR's fused elementwise step (train/muon_kernels.py) is bit-identical to the eager step: weights, momentum and the
running expert gradient norms over several updates, with idle experts, split rows, weight decay on and off."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('the fused MuonSR step is CUDA only', allow_module_level=True)

from natlang_neuralese.train.optim import MuonSR


def _run(fused, monkeypatch, weight_decay):
    monkeypatch.setenv('NATLANG_MUON_FUSED', '1' if fused else '0')
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
