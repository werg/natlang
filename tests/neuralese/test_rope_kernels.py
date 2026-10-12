"""maple/rope_kernels.qk_norm_rope against the eager composition it replaces in the Maple port (RMSNorm per head,
apply_partial_rope with full rotary, transpose to head-major): forward equal (the eager roundings replayed; only the RMS reduction order may differ by one BF16 step), input and gain gradients
within BF16 tolerance, for q (32 heads) and k (4 heads) widths."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('the fused q/k norm + rope is a CUDA kernel', allow_module_level=True)

from natlang_neuralese.maple.model import RMSNorm, apply_partial_rope
from natlang_neuralese.maple.rope_kernels import qk_norm_rope


@pytest.mark.parametrize('heads', [32, 4])
@pytest.mark.parametrize('weight_dtype', [torch.bfloat16, torch.float32])
def test_fused_qk_norm_rope_matches_the_eager_path(monkeypatch, heads, weight_dtype):
    monkeypatch.setenv('NATLANG_MAPLE_FUSED_NORM', '0')  # eager reference norm
    monkeypatch.setenv('NATLANG_MAPLE_COMPILE', '0')
    torch.manual_seed(heads)
    B, T, D = 2, 300, 128
    norm = RMSNorm(D, 1e-6).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.rand(D) + 0.5)
    norm.weight.data = norm.weight.data.to(weight_dtype)
    x = (torch.randn(B, T, heads * D, device='cuda') * 2).bfloat16()
    angles = torch.rand(B, T, D // 2, device='cuda') * 6.3
    cos = torch.cat([angles.cos(), angles.cos()], -1).bfloat16()
    sin = torch.cat([angles.sin(), angles.sin()], -1).bfloat16()
    g = torch.randn(B, heads, T, D, device='cuda').bfloat16()

    xr = x.clone().requires_grad_(True)
    q = norm(xr.view(B, T, heads, D)).transpose(1, 2)
    reference, _ = apply_partial_rope(q, q, cos, sin)
    reference.backward(g)
    ref_dx, ref_dw = xr.grad.float(), norm.weight.grad.float()
    norm.weight.grad = None

    xf = x.clone().requires_grad_(True)
    out = qk_norm_rope(xf, norm.gain(), cos, sin, heads, norm.eps)
    out.backward(g)
    assert out.shape == reference.shape and out.is_contiguous()
    differ = (out != reference)
    assert float(differ.float().mean()) < 1e-3
    spacing = reference.float().abs().clamp_min(1e-3) * 2 ** -7
    assert torch.equal(out, reference) or float(((out.float() - reference.float()).abs() / spacing).max()) <= 2.01
    rms = ref_dx.square().mean(-1, keepdim=True).sqrt()
    assert float(((xf.grad.float() - ref_dx).abs() / rms).max()) < 0.05
    assert float((norm.weight.grad.float() - ref_dw).norm() / ref_dw.norm()) < 5e-3
