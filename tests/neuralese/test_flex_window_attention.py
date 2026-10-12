"""Sliding-window attention through FlexAttention (model/hf_port.py ``_flex_window``) against the banded SDPA blocks it
replaces: outputs and q/k/v gradients within BF16 attention tolerance at window-crossing lengths (not multiples of the
128 block), with GQA; several lengths through one compiled kernel."""
import pytest
import torch
from types import SimpleNamespace

if not torch.cuda.is_available():
    pytest.skip('FlexAttention runs on CUDA here', allow_module_level=True)

from natlang_neuralese.model.hf_port import _attend_window


def _run(monkeypatch, flex, q, k, v, g, window):
    monkeypatch.setenv('NATLANG_FLEX_WINDOW', '1' if flex else '0')
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = _attend_window(SimpleNamespace(scaling=q.shape[-1] ** -0.5), q, k, v, q.shape[2], window)
    out.backward(g)
    return out.detach().float(), [t.grad.float() for t in (q, k, v)]


@pytest.mark.parametrize('steps', [300, 777, 1100])
def test_flex_window_matches_the_banded_blocks(monkeypatch, steps):
    torch.manual_seed(steps)
    window = 256
    q = torch.randn(2, 8, steps, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 2, steps, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(2, 2, steps, 64, device='cuda', dtype=torch.bfloat16)
    g = torch.randn(2, 8, steps, 64, device='cuda', dtype=torch.bfloat16)
    reference, reference_grads = _run(monkeypatch, False, q, k, v, g, window)
    out, grads = _run(monkeypatch, True, q, k, v, g, window)
    assert (out - reference).abs().max() < 2e-2
    for name, a, b in zip('qkv', grads, reference_grads):
        assert float((a - b).norm() / b.norm()) < 1e-2, name
