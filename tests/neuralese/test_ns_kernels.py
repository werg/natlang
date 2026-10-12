"""MuonSR's Triton Newton-Schulz (train/ns_kernels.py) against the eager folded cubic5 step (train/optim.py
``newton_schulz`` with ``NATLANG_MUON_NS_TRITON=0``): the symmetric Gram equals ``baddbmm`` up to accumulation order,
and whole orthogonalizations of wide, tall, stacked and ragged (non-tile-multiple) latents agree to BF16 rounding."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('the Triton Newton-Schulz kernels are CUDA only', allow_module_level=True)

from natlang_neuralese.train.ns_kernels import gram, matmul
from natlang_neuralese.train.optim import newton_schulz


@pytest.mark.parametrize('transposed', [False, True])
def test_symmetric_gram_matches_baddbmm(transposed):
    torch.manual_seed(0)
    x = (torch.randn(3, 200, 520, device='cuda') * 0.05).bfloat16()
    stored = x.transpose(-2, -1).contiguous() if transposed else x
    got = gram(stored, 2.5, -1.4, transposed=transposed)
    eye = torch.eye(200, device='cuda', dtype=torch.bfloat16).expand(3, -1, -1)
    want = torch.baddbmm(eye, x, x.transpose(-2, -1), beta=2.5, alpha=-1.4)
    assert torch.equal(got, got.transpose(-2, -1))
    assert float((got.float() - want.float()).abs().max()) <= 2 * float(want.float().abs().max()) * 2 ** -8


def test_strided_matmul_matches_bmm():
    torch.manual_seed(1)
    a = torch.randn(2, 300, 170, device='cuda').bfloat16()
    b = torch.randn(2, 260, 170, device='cuda').bfloat16().transpose(-2, -1)  # column-major operand
    want = torch.bmm(a.float(), b.float())
    got = matmul(a, b).float()
    assert float((got - want).norm() / want.norm()) < 4e-3


@pytest.mark.parametrize('shape', [(2, 2, 96, 260), (2, 1, 260, 96), (300, 140), (140, 300)])
def test_orthogonalization_matches_eager(monkeypatch, shape):
    torch.manual_seed(2)
    update = torch.randn(*shape, device='cuda').bfloat16()
    monkeypatch.setenv('NATLANG_MUON_NS_TRITON', '0')
    want = newton_schulz(update, variant='cubic5').float()
    monkeypatch.setenv('NATLANG_MUON_NS_TRITON', '1')
    got = newton_schulz(update, variant='cubic5')
    assert got.shape == update.shape and got.dtype == torch.bfloat16 and got.is_contiguous()
    assert float((got.float() - want).norm() / want.norm()) < 2e-2
