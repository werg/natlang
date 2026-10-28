"""Stable softmax/CE/divergence numerics against FP64 references (Barron et al. 2026)."""

import math

import pytest
import torch
import torch.nn.functional as F

from natlang_neuralese.common import stable_softmax as S


def _ref_grad(fn, z):
    z = z.detach().double().requires_grad_(True)
    fn(z).sum().backward()
    return z.grad


def _grad(fn, z):
    z = z.detach().clone().requires_grad_(True)
    fn(z).sum().backward()
    return z.grad.double()


def _hard_ref(z, labels):
    """FP64 analytic CE value and gradient: L = log1p(sum_{j!=t} e^{z_j - z_t}); immune to the FP64 cliff."""
    z = z.double()
    zt = z.gather(-1, labels[:, None])
    others = torch.where(torch.arange(z.shape[-1])[None] == labels[:, None], torch.zeros_like(z), (z - zt).exp())
    loss = torch.log1p(others.sum(-1))
    grad = others / (1 + others.sum(-1, keepdim=True))
    grad = grad.scatter(-1, labels[:, None], -grad.sum(-1, keepdim=True))
    return loss, grad


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_hard_ce_values_and_margin_sweep(dtype):
    V = 64
    worst_fixed = 0.
    naive_dead = False
    for margin in (1., 4., 8., 12., 20., 30., 60.):
        z = torch.zeros(3, V)
        z[:, 5] = margin
        z[1, 7] = margin - 0.5
        z[2] = torch.randn(V) * 0.1
        z[2, 5] = margin
        z = z.to(dtype)
        labels = torch.full((3,), 5)
        ref_value, ref = _hard_ref(z, labels)
        fixed = _grad(lambda x: S.cross_entropy(x, labels), z)
        naive = _grad(lambda x: F.cross_entropy(x, labels, reduction='none'), z)
        worst_fixed = max(worst_fixed, max(_rel(fixed[i], ref[i]) for i in range(3)))
        # Zero-sum per row to rounding of the input dtype.
        assert float(fixed.sum(-1).abs().max()) <= 1e-2 * float(fixed.abs().max(-1).values.max()) + 1e-30
        if bool((naive[:, 5] == 0).any()) and bool((ref[:, 5] != 0).all()):
            naive_dead = True
        value = S.cross_entropy(z, labels)
        torch.testing.assert_close(value.double(), ref_value, rtol=1e-5 if dtype == torch.float32 else 1e-2,
                                   atol=1e-30)
    assert worst_fixed < (1e-5 if dtype == torch.float32 else 1e-2)
    assert naive_dead  # the collapse the fix removes


def test_hard_ce_target_gradient_beyond_cliff_fp32():
    z = torch.zeros(1, 10)
    z[0, 0] = 30.
    g = _grad(lambda x: S.cross_entropy(x, torch.tensor([0])), z)
    expected = -9 * math.exp(-30.)
    assert abs(float(g[0, 0]) - expected) / abs(expected) < 1e-5
    assert abs(float(g.sum())) < 1e-6 * abs(expected)


def test_selected_ids_repeat_weights_ignore_and_chunks():
    torch.manual_seed(0)
    z = torch.randn(5, 3, 50) * 4
    ids = torch.randint(0, 50, (5, 3, 4))
    ids[..., 1] = ids[..., 0]
    w = torch.randn(5, 3, 4)
    ref = _ref_grad(lambda x: torch.log_softmax(x, -1).gather(-1, ids) * w.double(), z)
    got = _grad(lambda x: S.selected_log_softmax(x, ids, chunk_elements=60) * w, z)
    assert _rel(got, ref) < 1e-5
    labels = torch.randint(0, 50, (5, 3))
    labels[0, 0] = -100
    weights = torch.rand(5, 3)
    torch.testing.assert_close(S.cross_entropy(z, labels, weights=weights, reduction='sum'),
                               (F.cross_entropy(z.flatten(0, 1), labels.flatten(), reduction='none',
                                                ignore_index=-100) * weights.flatten()).sum())
    torch.testing.assert_close(S.cross_entropy(z, labels, reduction='mean'),
                               F.cross_entropy(z.flatten(0, 1), labels.flatten()))


def test_log_softmax_matches_and_exact_zero_at_max():
    torch.manual_seed(1)
    z = torch.randn(4, 30) * 5
    torch.testing.assert_close(S.log_softmax(z), torch.log_softmax(z, -1))
    z[0, 3] = 40.
    out = S.log_softmax(z)
    assert float(out[0, 3]) < 0  # -softplus(-m), not rounded to 0
    ref = _ref_grad(lambda x: torch.log_softmax(x, -1)[:, 3], z)
    got = _grad(lambda x: S.log_softmax(x)[:, 3], z)
    assert _rel(got, ref) < 1e-5


KINDS = [('forward_kl', {}), ('soft_ce', {}), ('reverse_kl', {}), ('jsd', {'beta': 0.3})]


def _ref_divergence(zs, zt, kind, beta=0.5):
    with S.use_numerics_policy(S.LEGACY_POLICY):
        return S.divergence(zs, zt, kind, beta=beta)


@pytest.mark.parametrize('kind,kw', KINDS)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_equal_logits_zero_gradient(kind, kw, dtype):
    torch.manual_seed(2)
    zt = (torch.randn(6, 300) * 3).to(dtype)
    g = _grad(lambda x: S.divergence(x, zt, kind, **kw), zt)
    assert float(g.abs().max()) == 0.
    value = S.divergence(zt, zt, kind, **kw)
    if kind != 'soft_ce':
        assert float(value.abs().max()) == 0.


@pytest.mark.parametrize('kind,kw', KINDS)
def test_near_agreement_gradient_sweep_fp32(kind, kw):
    torch.manual_seed(3)
    zt = torch.randn(8, 500) * 3
    for eps in (1e-1, 1e-3, 1e-5, 1e-7):
        zs = zt + eps * torch.randn_like(zt)
        ref = _ref_grad(lambda x: _ref_divergence(x, zt.double(), kind, **kw), zs)
        fixed = _grad(lambda x: S.divergence(x, zt, kind, **kw), zs)
        assert _rel(fixed, ref) < 2e-4, (kind, eps, _rel(fixed, ref))
        v = S.divergence(zs, zt, kind, **kw).double()
        vref = _ref_divergence(zs.double(), zt.double(), kind, **kw)
        assert float((v - vref).abs().max()) <= 1e-3 * float(vref.abs().max()) + 1e-10


@pytest.mark.parametrize('kind,kw', KINDS)
def test_divergence_far_from_agreement_matches(kind, kw):
    torch.manual_seed(4)
    zt = torch.randn(7, 40) * 6
    zs = torch.randn(7, 40) * 6
    ref = _ref_grad(lambda x: _ref_divergence(x, zt.double(), kind, **kw), zs)
    fixed = _grad(lambda x: S.divergence(x, zt, kind, chunk_elements=50, **kw), zs)
    assert _rel(fixed, ref) < 1e-5
    torch.testing.assert_close(S.divergence(zs, zt, kind, **kw).double(),
                               _ref_divergence(zs.double(), zt.double(), kind, **kw), rtol=1e-5, atol=1e-6)


def test_topk_teacher_matches_old_definition():
    torch.manual_seed(5)
    zs = torch.randn(9, 80) * 3
    teacher = torch.log_softmax(torch.randn(9, 80) * 3, -1)
    t_logp, t_ids = teacher.topk(8, -1)

    def old(x):
        t = t_logp.double() - torch.logsumexp(t_logp.double(), -1, keepdim=True)
        s = torch.log_softmax(x, -1).gather(-1, t_ids)
        return (t.exp() * (t - s)).sum(-1)

    ref = _ref_grad(old, zs)
    got = _grad(lambda x: S.topk_forward_kl(x, t_ids, t_logp), zs)
    assert _rel(got, ref) < 1e-5
    torch.testing.assert_close(S.topk_forward_kl(zs, t_ids, t_logp).double(), old(zs.double()), rtol=1e-5, atol=1e-6)
    # At agreement on the support with no student tail the gradient is exactly zero.
    zs_eq = torch.full((9, 80), -float('inf')).scatter(-1, t_ids, t_logp)
    zs_eq = torch.where(torch.isinf(zs_eq), torch.full_like(zs_eq, -1e4), zs_eq)
    assert float(_grad(lambda x: S.topk_forward_kl(x, t_ids, t_logp), zs_eq).abs().max()) < 1e-12


def test_teacher_must_be_detached():
    z = torch.randn(2, 5, requires_grad=True)
    with pytest.raises(ValueError):
        S.forward_kl(z, z)


@pytest.mark.parametrize('importance', ['plain', 'tis', 'mis'])
@pytest.mark.parametrize('noise', [1e-3, 0.7])
def test_score_centering_coefficients(importance, noise):
    torch.manual_seed(6)
    z = torch.randn(4, 3, 200) * 3
    sampler_full = torch.log_softmax(z.double() + noise * torch.randn(4, 3, 200).double(), -1)
    logq, ids = sampler_full.topk(16, -1)
    logq = logq.float()
    info = S.score_centering_coefficients(z, ids, logq, importance=importance)
    p = torch.log_softmax(z.double(), -1)
    pH = p.gather(-1, ids).exp()
    qH = logq.double().exp()
    rho = (1 - qH.sum(-1)) / (1 - pH.sum(-1))
    ratio = pH / qH
    if importance == 'plain':
        w, alpha = torch.ones_like(pH), rho
    elif importance == 'tis':
        w, alpha = ratio.clamp(max=2.), rho * (1 / rho).clamp(max=2.)
    else:
        w = torch.where((ratio >= .5) & (ratio <= 5.), ratio, torch.zeros_like(ratio))
        alpha = torch.where(((1 / rho) >= .5) & ((1 / rho) <= 5.), torch.ones_like(rho), torch.zeros_like(rho))
    c = qH * w - alpha[..., None] * pH
    err = (info['coefficients'].double() - c).abs()
    assert float((err / (pH * 1e-4 + 1e-12)).max()) < 1, float(err.max())
    torch.testing.assert_close(info['log_rho'].double(), rho.log(), rtol=1e-4, atol=1e-5)


def test_score_centered_surrogate_gradient():
    torch.manual_seed(7)
    z = torch.randn(5, 100)
    logq, ids = torch.log_softmax(z, -1).topk(10, -1)
    y = torch.randint(0, 100, (5,))
    ylogq = torch.log_softmax(z, -1).gather(-1, y[:, None])[:, 0]
    surrogate, info = S.score_centered_surrogate(z.requires_grad_(True), y, ylogq, ids, logq)
    # At exact agreement q == p: rho == 1, every head coefficient vanishes.
    assert float(info['coefficients'].abs().max()) < 1e-6
    surrogate.sum().backward()
    ref = _ref_grad(lambda x: torch.log_softmax(x, -1).gather(-1, y[:, None])[:, 0], z.detach())
    assert _rel(z.grad.double(), ref) < 1e-5


def test_legacy_policy_and_records():
    z = torch.randn(3, 20)
    labels = torch.tensor([1, 2, 3])
    with S.use_numerics_policy(S.LEGACY_POLICY):
        assert S.active_policy() == S.LEGACY_POLICY
        torch.testing.assert_close(S.cross_entropy(z, labels), F.cross_entropy(z, labels, reduction='none'))
    assert S.active_policy() == S.POLICY
    assert S.numerics_record()['softmax_numerics'] == S.POLICY
    assert S.numerics_transition(None) == {'kind': 'softmax-numerics-transition-v1', 'from': S.LEGACY_POLICY,
                                           'to': S.POLICY, 'reference': S.REFERENCE}
    assert S.numerics_transition(S.POLICY) is None
    with pytest.raises(ValueError):
        S.set_numerics_policy('nope')


def test_bf16_input_grad_dtype_and_native_compute():
    z = (torch.randn(4, 64) * 4).to(torch.bfloat16).requires_grad_(True)
    S.cross_entropy(z, torch.tensor([0, 1, 2, 3])).sum().backward()
    assert z.grad.dtype == torch.bfloat16
    zt = (torch.randn(4, 64) * 4).to(torch.bfloat16)
    x = zt.clone().requires_grad_(True)
    S.forward_kl(x, zt, compute_dtype=torch.bfloat16).sum().backward()
    assert float(x.grad.float().abs().max()) == 0.


def test_double_backward_matches_reference():
    torch.manual_seed(8)
    z = torch.randn(3, 20, dtype=torch.float64)
    labels = torch.tensor([1, 5, 7])
    v = torch.randn(3, 20, dtype=torch.float64)
    zt = torch.randn(3, 20, dtype=torch.float64)

    def hvp(fn):
        x = z.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(fn(x).sum(), x, create_graph=True)
        (h,) = torch.autograd.grad((g * v).sum(), x)
        return h

    ref = hvp(lambda x: F.cross_entropy(x, labels, reduction='none'))
    torch.testing.assert_close(hvp(lambda x: S.cross_entropy(x, labels)), ref)
    with S.use_numerics_policy(S.LEGACY_POLICY):
        ref = hvp(lambda x: S.divergence(x, zt, 'jsd', beta=0.3))
    torch.testing.assert_close(hvp(lambda x: S.divergence(x, zt, 'jsd', beta=0.3)), ref)


def test_fused_helpers_match_autograd():
    torch.manual_seed(9)
    z = torch.randn(6, 40) * 5
    labels = torch.randint(0, 40, (6,))
    loss, grad = S.cross_entropy_value_and_grad(z, labels)
    ref_value, ref = _hard_ref(z, labels)
    assert _rel(grad.double(), ref) < 1e-5
    torch.testing.assert_close(loss.double(), ref_value, rtol=1e-5, atol=1e-7)
    zt = z + 1e-4 * torch.randn_like(z)
    value, grad = S.divergence_value_and_grad(z, zt, 'forward_kl')
    assert _rel(grad.double(), _ref_grad(lambda x: _ref_divergence(x, zt.double(), 'forward_kl'), z)) < 1e-3
