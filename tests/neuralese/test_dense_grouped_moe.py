"""Dense trainable experts: the grouped-GEMM path (maple/fused_moe.dense_grouped_experts) and the batch-invariant
fixed-tile path (maple/fused_moe.invariant_dense_experts) equal the per-expert loop of SparseMoE up to BF16
accumulation order, in values and in gradients, with and without a precision point; the invariant path gives a token
the same output and routing whatever the other tokens of the call."""
import pytest
import torch

from natlang_neuralese.maple import ternary
from natlang_neuralese.maple.model import DENSE_MOE, MapleConfig, SparseMoE

pytestmark = [pytest.mark.skipif(not (torch.cuda.is_available() and hasattr(torch, '_grouped_mm')),
                                 reason='grouped GEMM needs CUDA and torch._grouped_mm'),
              pytest.mark.gpu_heavy]  # up to ~6.4 GB of CUDA memory


def layer(experts, hidden, ff, seed=0):
    torch.manual_seed(seed)
    config = MapleConfig(hidden_size=hidden, num_experts=experts, moe_intermediate_size=ff, num_experts_per_tok=4,
                         dense_experts=True, swiglu_clamp=None)
    moe = SparseMoE(config).cuda()
    with torch.no_grad():
        moe.experts.gate_up.normal_(0, 0.05)
        moe.experts.down.normal_(0, 0.05)
        moe.gate.weight.normal_(0, 0.5)
    moe.active_experts = experts - 1  # the last expert gets no token: an empty group
    moe.experts.make_latent(quantize=False)
    moe.experts.precision_group = 'experts'
    return moe


def run(moe, x, kernel, monkeypatch, precision):
    monkeypatch.setitem(DENSE_MOE, 'kernel', kernel)
    for p in (moe.experts.gate_up, moe.experts.down):
        p.grad = None
    x = x.clone().requires_grad_(True)
    with ternary.active_precision(precision, 'p' if precision else None):
        y = moe(x)
        (y.float() * torch.linspace(-1, 1, y.shape[-1], device=y.device)).square().sum().backward()
    return y.detach().float(), x.grad.float(), moe.experts.gate_up.grad.float(), moe.experts.down.grad.float()


def relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@pytest.mark.parametrize('precision', [None, {'experts': ('int4', 0.5, {'group': 32})},
                                       {'experts': ('ternary', 1.0, {})}])
@pytest.mark.parametrize('shape', [(8, 128, 64, 37), (64, 2304, 896, 300)])
@pytest.mark.parametrize('kernel', ['grouped', 'invariant'])
def test_grouped_dense_experts_match_the_loop(shape, precision, kernel, monkeypatch):
    experts, hidden, ff, tokens = shape
    moe = layer(experts, hidden, ff)
    x = (torch.randn(1, tokens, hidden, device='cuda') * 0.5).to(torch.bfloat16)
    reference = run(moe, x, 'loop', monkeypatch, precision)
    grouped = run(moe, x, kernel, monkeypatch, precision)
    for name, a, b in zip(('y', 'dx', 'd_gate_up', 'd_down'), grouped, reference):
        assert relative(a, b) < 1e-2, (name, relative(a, b))
    assert float(grouped[2][-1].abs().max()) == 0.0  # the unrouted expert gets no gradient



@pytest.mark.parametrize('shape', [(16, 256, 128), (64, 2304, 896)])
def test_the_invariant_kernel_gives_a_token_the_same_output_whatever_the_other_tokens(shape, monkeypatch):
    """Mellum batch-row dragnet (NUMERICAL_PARITY.md): with cuBLAS kernels chosen by shape, identical rows of one batch
    came out 1e-6 apart after one MoE layer and flipped top-k routing after the next."""
    experts, hidden, ff = shape
    moe = layer(experts, hidden, ff)
    monkeypatch.setitem(DENSE_MOE, 'kernel', 'invariant')
    h = (torch.randn(1, 700, hidden, device='cuda') * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        one = moe(h)
        two = moe(h.expand(2, -1, -1).contiguous())
        many = moe(torch.cat([(torch.randn(3, 700, hidden, device='cuda') * 0.5).to(torch.bfloat16), h]))
        single = torch.cat([moe(h[:, i:i + 1]) for i in range(0, 700, 97)], 1)  # decode-sized calls
    assert torch.equal(two[0], two[1]) and torch.equal(two[0], one[0]) and torch.equal(many[-1], one[0])
    assert torch.equal(single[0], one[0, ::97])
