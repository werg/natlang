"""The fused RMSNorm kernel (maple/norm_kernels.py) against the eager rule it replaces (maple/model.py ``_rms_norm``):
forward and both gradients, for the hidden width and the per-head q/k norm width, BF16 and FP32 gains, also under
non-reentrant activation checkpointing. Only the FP32 row reduction order differs, so outputs and input gradients agree
to within one BF16 rounding step on a small fraction of elements."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('the fused RMSNorm is a CUDA kernel', allow_module_level=True)

from natlang_neuralese.maple.model import RMSNorm, _rms_norm
from natlang_neuralese.maple.norm_kernels import fused_rms_norm


def _ulp_mismatch(a, b):
    """Fraction of elements that differ, and the largest difference relative to the BF16 spacing at that value."""
    a, b = a.float(), b.float()
    differ = (a != b)
    spacing = torch.maximum(a.abs(), b.abs()).clamp_min(1e-30) * 2 ** -7
    return float(differ.float().mean()), float(((a - b).abs() / spacing).max())


@pytest.mark.parametrize('columns,rows', [(2304, 3000), (128, 9000)])
@pytest.mark.parametrize('weight_dtype', [torch.bfloat16, torch.float32])
def test_fused_rms_norm_matches_the_eager_rule(columns, rows, weight_dtype):
    torch.manual_seed(columns + rows)
    x = (torch.randn(2, rows // 2, columns, device='cuda') * 3).bfloat16()
    w = (torch.rand(columns, device='cuda') + 0.5).to(weight_dtype)
    g = torch.randn(2, rows // 2, columns, device='cuda').bfloat16()
    outputs = []
    for fn in (_rms_norm, fused_rms_norm):
        xi = x.clone().requires_grad_(True)
        wi = w.clone().requires_grad_(True)
        y = fn(xi, wi, 1e-6)
        y.backward(g)
        outputs.append((y.detach(), xi.grad, wi.grad))
    (y0, dx0, dw0), (y1, dx1, dw1) = outputs
    assert y1.dtype == y0.dtype == torch.bfloat16 and dx1.dtype == dx0.dtype and dw1.dtype == dw0.dtype == weight_dtype
    fraction, steps = _ulp_mismatch(y0, y1)
    assert fraction < 0.01 and steps <= 2.01, (fraction, steps)  # one BF16 step (the spacing bound is 2x loose)
    # grad_x = grad_y * rstd + c * x: where the two terms cancel, FP32 order differences show up relative to the small
    # result, so measure against the row's gradient scale: a few BF16 steps of the row RMS at most.
    fraction, _ = _ulp_mismatch(dx0, dx1)
    rms = dx0.float().square().mean(-1, keepdim=True).sqrt()
    assert fraction < 0.01 and float(((dx0.float() - dx1.float()).abs() / rms).max()) < 0.05
    # The weight gradient is a sum over all rows: FP32 accumulation order differs, so relative tolerance.
    error = float((dw0.float() - dw1.float()).norm() / dw0.float().norm())
    assert error < (4e-3 if weight_dtype == torch.bfloat16 else 1e-4), error


def test_rms_norm_module_uses_the_kernel_and_checkpoint_recompute_is_exact(monkeypatch):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    norm = RMSNorm(256, 1e-6).cuda().bfloat16()
    x = torch.randn(4, 64, 256, device='cuda').bfloat16().requires_grad_(True)
    plain = norm(x)
    monkeypatch.setenv('NATLANG_MAPLE_FUSED_NORM', '0')
    monkeypatch.setenv('NATLANG_MAPLE_COMPILE', '0')
    eager = norm(x)
    assert _ulp_mismatch(plain.detach(), eager.detach())[1] <= 2.01
    monkeypatch.delenv('NATLANG_MAPLE_FUSED_NORM')
    direct = norm(x).float().square().sum()
    (dx_direct,) = torch.autograd.grad(direct, x)
    recomputed = checkpoint(lambda v: norm(v), x, use_reentrant=False).float().square().sum()
    (dx_checkpoint,) = torch.autograd.grad(recomputed, x)
    assert torch.equal(dx_direct, dx_checkpoint)  # deterministic: the recompute replays the same kernel
