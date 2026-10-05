import torch

from natlang_neuralese.shared_space import Bijection, InvertibleLinear, SharedSpace


def test_invertible_linear_roundtrip_and_set_affine():
    torch.manual_seed(0)
    layer = InvertibleLinear(6)
    w, b = torch.randn(6, 6), torch.randn(6)
    layer.set_affine(w, b)
    x = torch.randn(4, 3, 6)
    assert torch.allclose(layer(x), x @ w.T + b, atol=1e-4)
    assert torch.allclose(layer.inverse(layer(x)), x, atol=1e-4)


def test_bijection_with_couplings_is_exact():
    torch.manual_seed(1)
    f = Bijection(8, couplings=3, hidden=16)
    for layer in f.couplings:  # move away from the identity start
        torch.nn.init.normal_(layer.net[-1].weight, std=0.3)
    f.linear.set_affine(torch.randn(8, 8), torch.randn(8))
    x = torch.randn(5, 8)
    assert torch.allclose(f.inverse(f(x)), x, atol=1e-4)


def _paired(n=4000, large=12, small=6, shared=4, seed=2):
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(n, shared, generator=g)
    h_large = torch.cat([latent, torch.randn(n, large - shared, generator=g)], 1) @ torch.randn(large, large, generator=g)
    h_small = torch.cat([latent, torch.randn(n, small - shared, generator=g)], 1) @ torch.randn(small, small, generator=g)
    return h_large + 3.0, h_small - 1.0


def test_cca_init_finds_shared_directions_and_self_roundtrip_is_exact():
    h_large, h_small = _paired()
    space = SharedSpace(12, 6)
    rho = space.init_from_pairs(h_large, h_small, ridge=1e-9)
    assert rho[:4].min() > 0.99 and rho[4:].max() < 0.15
    common_large, private = space.large_to_z(h_large)
    common_small = space.small_to_c(h_small)
    # whitened: unit variance; shared variates agree
    assert torch.allclose(common_large.var(0), torch.ones(6), atol=0.05)
    assert torch.allclose(common_large[:, :4], common_small[:, :4], atol=0.15)
    assert torch.allclose(space.z_to_large(common_large, private), h_large, atol=1e-3)
    assert torch.allclose(space.c_to_small(common_small), h_small, atol=1e-3)


def test_translation_predicts_the_other_model():
    h_large, h_small = _paired()
    space = SharedSpace(12, 6)
    space.init_from_pairs(h_large, h_small)
    predicted = space.large_to_small(h_large)
    residual = (predicted - h_small).pow(2).sum(-1).mean()
    baseline = (h_small - h_small.mean(0)).pow(2).sum(-1).mean()
    assert residual < 0.6 * baseline
    back = space.small_to_large(h_small)
    assert back.shape == h_large.shape


def test_prior_and_agreement_losses_train():
    h_large, h_small = _paired(n=512)
    space = SharedSpace(12, 6, couplings=1, hidden=8)
    space.init_from_pairs(h_large, h_small)
    loss = space.prior_nll(h_large) + space.common_agreement(h_large, h_small)
    loss.backward()
    assert space.prior_log_variance.grad is not None
    assert space.f_small.linear.lower.grad is not None
