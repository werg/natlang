"""Golden tests for core training math: Lion/Muon/AdamW updates against hand-computed references, optimizer
state round-trips and routing, and the QAT ramp w + mix*(Q(w)-w) with its straight-through gradient. Tiny CPU tensors."""
import math

import pytest
import torch
from torch import nn

from natlang_neuralese.maple import ternary as T
from natlang_neuralese.train.optim import LionSR, PortMuonAdamW, muon_eligible

needs_muon = pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch Muon unavailable")


# ---------------------------------------------------------------- Lion

def lion_reference(p, g, m, lr, betas=(0.9, 0.99), wd=0.0):
    b1, b2 = betas
    update = (b1 * m + (1 - b1) * g).sign()
    return p * (1 - lr * wd) - lr * update, b2 * m + (1 - b2) * g


@pytest.mark.parametrize("wd", [0.0, 0.1])
def test_lion_two_steps_match_reference_with_decoupled_weight_decay(wd):
    p = nn.Parameter(torch.tensor([0.5, -0.25, 1.0, 0.0]))
    opt = LionSR([p], lr=0.01, weight_decay=wd)
    ref_p, ref_m = p.detach().clone(), torch.zeros(4)
    for g in (torch.tensor([0.3, -0.2, 0.0, 1e-3]), torch.tensor([-1.0, 0.1, 0.5, -1e-3])):
        p.grad = g.clone()
        opt.step()
        ref_p, ref_m = lion_reference(ref_p, g, ref_m, 0.01, wd=wd)
        assert torch.allclose(p.detach(), ref_p, atol=1e-7)
        # momentum is stored in BF16
        assert torch.allclose(opt.state[p]["momentum"].float(), ref_m, rtol=1e-2, atol=1e-6)
        assert opt.state[p]["momentum"].dtype == torch.bfloat16


def test_lion_sign_update_ignores_gradient_magnitude_and_zero_grad_does_not_move():
    p = nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([1e-6, -1e6, 0.0])
    LionSR([p], lr=0.5).step()
    assert p.tolist() == [-0.5, 0.5, 0.0]


def test_lion_weight_decay_is_decoupled_from_gradient():
    p = nn.Parameter(torch.tensor([2.0]))
    p.grad = torch.tensor([0.0])
    LionSR([p], lr=0.1, weight_decay=0.5).step()
    assert p.item() == pytest.approx(2.0 * (1 - 0.05))


def test_lion_state_dict_round_trip_gives_identical_next_step():
    torch.manual_seed(0)
    w0 = torch.randn(5, 3)
    a, b = nn.Parameter(w0.clone()), nn.Parameter(w0.clone())
    oa = LionSR([a], lr=0.02, weight_decay=0.1)
    for _ in range(3):
        a.grad = torch.randn(5, 3)
        oa.step()
    with torch.no_grad():
        b.copy_(a)
    ob = LionSR([b], lr=0.02, weight_decay=0.1)
    ob.load_state_dict(oa.state_dict())
    g = torch.randn(5, 3)
    a.grad, b.grad = g.clone(), g.clone()
    oa.step()
    ob.step()
    assert torch.equal(a, b)


def test_lion_chunking_does_not_change_the_update():
    torch.manual_seed(1)
    w0, g = torch.randn(7, 5), torch.randn(7, 5)
    a, b = nn.Parameter(w0.clone()), nn.Parameter(w0.clone())
    a.grad, b.grad = g.clone(), g.clone()
    LionSR([a], lr=0.01, chunk=1 << 26).step()
    LionSR([b], lr=0.01, chunk=4).step()
    assert torch.equal(a, b)


def test_lion_step_in_backward_two_layer_model_matches_full_gradient_reference():
    torch.manual_seed(0)
    l1, l2 = nn.Linear(6, 5, bias=False), nn.Linear(5, 3, bias=False)
    x = torch.randn(4, 6)
    ref1, ref2 = l1.weight.detach().clone(), l2.weight.detach().clone()
    m1, m2 = torch.zeros_like(ref1), torch.zeros_like(ref2)
    opt = LionSR([{"params": [l1.weight]}, {"params": [l2.weight]}], lr=0.01, weight_decay=0.05)
    opt.step_in_backward()
    for _ in range(3):
        # reference: full gradients of the same loss at the reference parameters, then the reference Lion step
        w1, w2 = ref1.clone().requires_grad_(), ref2.clone().requires_grad_()
        (torch.tanh(x @ w1.T) @ w2.T).square().sum().backward()
        ref1, m1 = lion_reference(ref1, w1.grad, m1, 0.01, wd=0.05)
        ref2, m2 = lion_reference(ref2, w2.grad, m2, 0.01, wd=0.05)
        (l2(torch.tanh(l1(x)))).square().sum().backward()
        assert l1.weight.grad is None and l2.weight.grad is None
        assert torch.allclose(l1.weight.detach(), ref1, atol=1e-6)
        assert torch.allclose(l2.weight.detach(), ref2, atol=1e-6)


# ---------------------------------------------------------------- AdamW / Muon references

def adamw_first_step(p, g, lr, eps=1e-8):
    return p - lr * g / (g.abs() + eps)  # bias-corrected m/(sqrt(v)+eps) at t=1, weight_decay 0


def newton_schulz_reference(update, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    a, b, c = coeffs
    x = update.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


def muon_first_step(p, g, lr, momentum=0.95):
    buf = (1 - momentum) * g                    # buf = lerp(0, g, 1-mu)
    nesterov = g + momentum * (buf - g)         # g.lerp(buf, mu)
    ortho = newton_schulz_reference(nesterov).float()
    adjust = 0.2 * math.sqrt(max(p.shape))      # match_rms_adamw
    return p - lr * adjust * ortho


def test_newton_schulz_output_is_roughly_orthogonal():
    torch.manual_seed(0)
    out = newton_schulz_reference(torch.randn(8, 5)).float()
    s = torch.linalg.svdvals(out)
    assert s.min() > 0.4 and s.max() < 1.5   # quintic NS maps singular values into ~[0.7, 1.2]


@needs_muon
def test_port_muon_adamw_first_step_matches_hand_reference():
    torch.manual_seed(0)
    hidden, bias = nn.Parameter(torch.randn(6, 4)), nn.Parameter(torch.randn(4))
    gh, gb = torch.randn(6, 4), torch.randn(4)
    h0, b0 = hidden.detach().clone(), bias.detach().clone()
    opt = PortMuonAdamW([("heads.hidden", hidden), ("heads.bias", bias)], lr=0.01, vocab_size=1000)
    hidden.grad, bias.grad = gh.clone(), gb.clone()
    opt.step()
    assert torch.allclose(hidden.detach(), muon_first_step(h0, gh, 0.01), atol=1e-3)
    assert torch.allclose(bias.detach(), adamw_first_step(b0, gb, 0.01), atol=1e-6)


@needs_muon
def test_port_muon_adamw_routing_as_documented():
    vocab = 50
    embedding = nn.Parameter(torch.randn(10, 4))
    params = {
        "backbone.control_rows": nn.Parameter(torch.randn(3, 4)),
        "heads.hidden.weight": nn.Parameter(torch.randn(6, 4)),
        "heads.hidden.bias": nn.Parameter(torch.randn(6)),
        "heads.readout.weight": nn.Parameter(torch.randn(vocab, 4)),   # vocabulary-sized
        "heads.wide.weight": nn.Parameter(torch.randn(4, vocab)),
        "heads.embed.weight": embedding,                                # embedding table
        "heads.single_row.weight": nn.Parameter(torch.randn(1, 4)),
        "heads.lora_A": nn.Parameter(torch.randn(2, 4)),
        "heads.mlp.gate.weight": nn.Parameter(torch.randn(8, 4)),       # MoE router
        "heads.experts.learned_scale": nn.Parameter(torch.randn(3, 4)),
        "heads.experts.gate_up_blocks": nn.Parameter(torch.randn(3, 4)),
    }
    opt = PortMuonAdamW(list(params.items()), lr=0.01, vocab_size=vocab, embedding_ids={id(embedding)})
    routed = {row["name"]: row["optimizer"] for row in opt.schema}
    assert routed.pop("heads.hidden.weight") == "muon"
    assert set(routed.values()) == {"adamw"}
    assert len(opt.muon.param_groups[0]["params"]) == 1
    assert not muon_eligible("x", torch.zeros(3), vocab, set())  # 1-D


@needs_muon
def test_port_muon_adamw_state_round_trip_and_added_groups_are_adamw():
    def build():
        torch.manual_seed(1)
        h, b = nn.Parameter(torch.randn(6, 4)), nn.Parameter(torch.randn(4))
        return h, b, PortMuonAdamW([("h", h), ("b", b)], lr=0.01, vocab_size=1000)
    ha, ba, oa = build()
    hb, bb, ob = build()
    torch.manual_seed(2)
    for _ in range(2):
        ha.grad, ba.grad = torch.randn(6, 4), torch.randn(4)
        oa.step()
    with torch.no_grad():
        hb.copy_(ha)
        bb.copy_(ba)
    ob.load_state_dict(oa.state_dict())
    gh, gb = torch.randn(6, 4), torch.randn(4)
    for h, b in ((ha, ba), (hb, bb)):
        h.grad, b.grad = gh.clone(), gb.clone()
    oa.step()
    ob.step()
    assert torch.equal(ha, hb) and torch.equal(ba, bb)
    # a mismatching schema is refused
    other = PortMuonAdamW([("h", nn.Parameter(torch.randn(6, 4))), ("c", nn.Parameter(torch.randn(4)))],
                          lr=0.01, vocab_size=1000)
    with pytest.raises(ValueError):
        other.load_state_dict(oa.state_dict())
    lora = nn.Parameter(torch.randn(2, 4))
    oa.add_param_group({"params": [lora], "lr": 0.02})
    assert oa.schema[-1]["optimizer"] == "adamw"
    assert any(lora is q for q in oa.auxiliary.param_groups[-1]["params"])


@needs_muon
def test_scripts_muon_with_adamw_first_step_and_routing():
    from scripts.training_optimizers import MuonWithAdamW

    torch.manual_seed(0)
    hidden, bias, head = nn.Parameter(torch.randn(6, 4)), nn.Parameter(torch.randn(4)), nn.Parameter(torch.randn(9, 4))
    g = [torch.randn_like(t) for t in (hidden, bias, head)]
    p0 = [t.detach().clone() for t in (hidden, bias, head)]
    opt = MuonWithAdamW([("layer.weight", hidden), ("layer.bias", bias), ("lm_head.weight", head)], lr=0.01)
    assert {r["name"]: r["optimizer"] for r in opt.schema} == {
        "layer.weight": "muon", "layer.bias": "adamw", "lm_head.weight": "adamw"}
    for t, gi in zip((hidden, bias, head), g):
        t.grad = gi.clone()
    opt.step()
    assert torch.allclose(hidden.detach(), muon_first_step(p0[0], g[0], 0.01), atol=1e-3)
    assert torch.allclose(bias.detach(), adamw_first_step(p0[1], g[1], 0.01), atol=1e-6)
    assert torch.allclose(head.detach(), adamw_first_step(p0[2], g[2], 0.01), atol=1e-6)


# ---------------------------------------------------------------- QAT ramp and STE

@pytest.fixture
def mix():
    old = T.QUANT_MIX["value"]
    yield lambda v: T.QUANT_MIX.__setitem__("value", v)
    T.QUANT_MIX["value"] = old


def _weight():
    torch.manual_seed(0)
    return torch.randn(4, 64).to(torch.bfloat16).float()


def test_ramp_endpoints_are_exact(mix):
    w = _weight()
    mix(0.0)
    assert torch.equal(T.ramped_ternarize_ste(w), w)
    mix(1.0)
    assert torch.equal(T.ramped_ternarize_ste(w), T.ternarize(w))
    assert torch.equal(T.ramped_ternarize_ste(w), T.ternarize_ste(w))


def test_ramp_interpolates_linearly_in_mix(mix):
    w = _weight()
    q = T.ternarize(w)
    for lam in (0.25, 0.5, 0.9):
        mix(lam)
        assert torch.allclose(T.ramped_ternarize_ste(w), w + lam * (q - w), atol=1e-6)


def test_ternary_values_are_minus_s_zero_plus_s_per_row():
    w = _weight()
    q = T.ternarize(w)
    for row in q:
        scale = row.abs().max().item()
        assert set(row.unique().tolist()) <= {-scale, 0.0, scale}
    codes, alpha = T.ternary_codes(w)
    assert torch.equal(q, codes.float() * alpha.float())


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_ramp_gradient_is_identity_at_the_endpoints(mix, lam):
    w = _weight().requires_grad_()
    mix(lam)
    up = torch.randn(4, 64)
    T.ramped_ternarize_ste(w).backward(up)
    assert torch.allclose(w.grad, up)


@pytest.mark.parametrize("lam", [0.25, 0.5, 0.9])
def test_ramp_gradient_at_intermediate_mix_is_straight_through_identity(mix, lam):
    # Q(w) and w are detached inside the interpolation term, so d out / d w = 1 for every mix.
    w = _weight().requires_grad_()
    mix(lam)
    up = torch.randn(4, 64)
    T.ramped_ternarize_ste(w).backward(up)
    assert torch.allclose(w.grad, up)


def test_ramp_schedule_formula_endpoints_and_monotonicity():
    # run_train's inline schedule (qat_convert.py, a closure so not importable; mirrored): min(1, at/ramp_steps).
    ramp = 1000
    mixes = [min(1.0, at / ramp) for at in range(0, 1501, 50)]
    assert mixes[0] == 0.0 and mixes[20] == 1.0 and mixes[-1] == 1.0
    assert all(a <= b for a, b in zip(mixes, mixes[1:]))
