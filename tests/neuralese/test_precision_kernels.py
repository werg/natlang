"""The fused precision ramp (maple/precision_kernels.py) is bit-identical to the eager rule (maple/ternary.py)."""
import pytest
import torch

from natlang_neuralese.common.paths import resolve
from natlang_neuralese.maple import ternary

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='the fused ramp is CUDA-only')
MELLUM = resolve('models', 'mellum21-12b-a2.5b-thinking')
POINTS = [('int4', {'group': 32}), ('int4', {'group': 64}), ('ternary', {})]
MIXES = (1.0, 0.47, 0.03)


def ramp(weight, fmt, options, mix, fused):
    old = ternary.PRECISION['fused']
    ternary.PRECISION['fused'] = fused
    try:
        with ternary.active_precision({'g': (fmt, mix, options)}, 'p'):
            return ternary.precision_value(weight, 'g')
    finally:
        ternary.PRECISION['fused'] = old


def assert_identical(weight):
    for fmt, options in POINTS:
        for mix in MIXES:
            eager, fused = ramp(weight, fmt, options, mix, False), ramp(weight, fmt, options, mix, True)
            assert fused.dtype == eager.dtype == torch.bfloat16
            differing = int((fused.view(torch.int16) != eager.view(torch.int16)).sum())
            assert differing == 0, (fmt, options, mix, differing, tuple(weight.shape))


def test_fused_ramp_is_bit_identical_on_hard_cases():
    torch.manual_seed(0)
    w = (torch.randn(6, 96, 256, device='cuda') * 0.02).to(torch.bfloat16)
    w[0, 0] = 0  # empty rows and blocks
    w[0, 1, :32] = 0.01  # all-equal block: ties at the max
    w[0, 2, :32] = torch.tensor([0.01, -0.01] * 16)  # opposite-sign ties: the first one is the max
    w[0, 3] = w[0, 3].abs().mean()  # every |w| at the row mean: threshold ties
    w[1] *= 1e-30  # tiny scales (subnormal FP16 d)
    w[2] *= 3e3  # large values
    assert_identical(w)
    assert_identical(w[0])  # 2-D (attention) shapes


def test_fused_ramp_is_straight_through():
    latent = torch.nn.Parameter((torch.randn(64, 128, device='cuda') * 0.02).to(torch.bfloat16))
    for fmt, options in POINTS:
        latent.grad = None
        with ternary.active_precision({'g': (fmt, 0.5, options)}, 'p'):
            ternary.precision_value(latent, 'g').float().sum().backward()
        assert torch.equal(latent.grad, torch.ones_like(latent))


def test_q4_scales_reproduce_the_reference_reduction():
    torch.manual_seed(1)
    w = (torch.randn(8, 512, device='cuda') * 0.02).to(torch.bfloat16)
    blocks = w.float().reshape(8, 16, 32)
    reference = blocks.gather(-1, blocks.abs().argmax(-1, keepdim=True)) / -8.0
    assert torch.equal(ternary.q4_scales(w, 32), reference)
    assert torch.equal(ternary.q4_scales(w.float(), 32), reference)


@pytest.mark.skipif(not (MELLUM / 'config.json').is_file(), reason='Mellum checkpoint not present')
def test_fused_ramp_is_bit_identical_on_real_mellum_layers():
    from natlang_neuralese.maple.model import load_maple

    model = load_maple(MELLUM, device='cuda', layers=2, ternary_attention=False, dense_experts=True)
    checked = 0
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if '.layers.' in name and tensor.ndim >= 2 and tensor.dtype == torch.bfloat16 and 'gate.weight' not in name:
            assert_identical(tensor.detach())
            checked += 1
    assert checked >= 2 * 6  # q, k, v, o, experts gate_up and down per layer
