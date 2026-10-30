"""Real NVFP4 forward GEMM of an inline QAT pass at λ = 1 (model/nvfp4_gemm.py) against the fake-quant forward."""
import pytest
import torch

from natlang_neuralese.model import nvfp4_gemm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not nvfp4_gemm.supported('cuda'),
                                reason='block-scaled FP4 GEMMs need a Blackwell GPU')


def _operands(tokens, k, n, seed=0):
    from natlang_neuralese.maple.ternary import _ramped_weight
    from natlang_neuralese.train import qformats as Q
    torch.manual_seed(seed)
    x = (torch.randn(tokens, k, device='cuda') * 2).to(torch.bfloat16).requires_grad_(True)
    latent = (torch.randn(n, k, device='cuda') * 0.05).to(torch.bfloat16).requires_grad_(True)
    # A calibrated range below this batch's max: saturated group scales are exercised too.
    input_scale = Q.nvfp4_global_scale(x.detach().abs().amax().float() * 0.8)
    global_amax = latent.detach().abs().amax()
    ramped = _ramped_weight(latent.detach(), 'nvfp4', {'group': 16, 'global_amax': global_amax},
                            torch.ones((), device='cuda'))
    return x, latent, input_scale, global_amax, ramped


def _fake(x, latent, ramped, input_scale):
    """The fake-quant pass at λ = 1 (maple/ternary._inline_operands): BF16 operands, straight-through."""
    from natlang_neuralese.maple.ternary import _OperandStraightThrough
    from natlang_neuralese.train import qformats as Q
    value = Q.nvfp4_activation_value(x.detach(), input_scale).to(x.dtype)
    return torch.nn.functional.linear(_OperandStraightThrough.apply(x, value),
                                      _OperandStraightThrough.apply(latent, ramped))


@pytest.mark.parametrize('shape', [(300, 256, 512), (1024, 2048, 7680), (777, 7680, 2048)])
def test_fp4_gemm_matches_the_deployed_product_and_the_fake_quant_gradients(shape):
    from natlang_neuralese.train import qformats as Q
    x, latent, input_scale, global_amax, ramped = _operands(*shape)
    deployed_x = Q.nvfp4_activation_value(x.detach(), input_scale)
    exact = deployed_x.double() @ Q.nvfp4_weight_value(latent.detach(), global_amax=global_amax).double().t()
    real = nvfp4_gemm.nvfp4_linear(x, latent, ramped, nvfp4_gemm.weight_operand(latent, global_amax), input_scale)
    fake = _fake(x, latent, ramped, input_scale)
    error = lambda out: float((out.detach().double() - exact).norm() / exact.norm())  # noqa: E731
    # The FP4 GEMM multiplies the deployed codes with FP32 accumulation; fake quant rounds each operand to BF16 first.
    assert torch.isfinite(real).all() and error(real) < 3e-3 and error(real) <= error(fake), (error(real), error(fake))
    grad = torch.randn_like(real)
    assert all(torch.equal(a, b) for a, b in zip(torch.autograd.grad(real, (x, latent), grad),
                                                   torch.autograd.grad(fake, (x, latent), grad)))


def test_compiled_layer_piece_traces_the_fp4_linear_without_a_graph_break():
    """Inside the compiled layer pieces the activation encode fuses and only the GEMM is opaque: a full-graph compile
    of an LFM3-shaped FFN (w1/w3 SiLU product into w2) runs and keeps the fake-quant straight-through gradients."""
    x, w1, s1, a1, r1 = _operands(512, 256, 768, seed=1)
    _, w2, _, a2, r2 = _operands(512, 768, 256, seed=2)
    from natlang_neuralese.train import qformats as Q
    with torch.no_grad():
        s2 = Q.nvfp4_global_scale((torch.nn.functional.silu(x @ r1.t()) * (x @ r1.t())).abs().amax().float())
    o1, o2 = nvfp4_gemm.weight_operand(w1, a1), nvfp4_gemm.weight_operand(w2, a2)

    def real(h):
        up = nvfp4_gemm.nvfp4_linear(h, w1, r1, o1, s1)
        return nvfp4_gemm.nvfp4_linear(torch.nn.functional.silu(up) * up, w2, r2, o2, s2)

    def fake(h):
        up = _fake(h, w1, r1, s1)
        return _fake(torch.nn.functional.silu(up) * up, w2, r2, s2)

    compiled = torch.compile(real, dynamic=True, fullgraph=True)(x)
    eager = real(x)
    grad = torch.randn_like(eager)
    gc = torch.autograd.grad(compiled, (x, w1, w2), grad)
    ge = torch.autograd.grad(eager, (x, w1, w2), grad)
    gf = torch.autograd.grad(fake(x), (x, w1, w2), grad)
    rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())  # noqa: E731
    # Inductor's activation codec divides approximately (as the fake pass's compiled codec), so a few E2M1 codes of
    # the second projection's input flip by one step (up to a third of a value): a few percent at the block output.
    assert torch.isfinite(compiled).all() and rel(compiled, eager) < 6e-2
    assert all(rel(a, b) < 6e-2 for a, b in zip(gc, ge)) and all(rel(a, b) < 6e-2 for a, b in zip(ge, gf))
