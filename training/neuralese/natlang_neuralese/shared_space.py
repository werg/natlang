"""A Neuralese space shared by a large and a small model, with an exact bijection onto each.

``Z = C ⊕ P``: C is common (the small model's width), P is private to the large model. ``f_large`` maps the large
model's residual space onto all of Z and ``f_small`` maps the small model's onto C; both are exactly invertible, so a
model's own Neuralese round-trips unchanged. Translation drops P (large → small) or fills it from a learned prior
(small → large), and shrinks C by a learned per-coordinate factor that starts at the canonical correlations (the
best linear prediction of one model's coordinates from the other's). See plans/neuralese/MAPLE_QWEN_JOINT.md §4.
"""

from __future__ import annotations

import torch
from torch import nn


class InvertibleLinear(nn.Module):
    """``y = W x + b`` with ``W = P L (U + diag(sign * exp(log_scale)))``: invertible by construction."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("permutation", torch.eye(dim))
        self.register_buffer("sign", torch.ones(dim))
        self.lower = nn.Parameter(torch.zeros(dim, dim))
        self.upper = nn.Parameter(torch.zeros(dim, dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _factors(self):
        eye = torch.eye(self.dim, device=self.lower.device, dtype=self.lower.dtype)
        lower = torch.tril(self.lower, -1) + eye
        upper = torch.triu(self.upper, 1) + torch.diag(self.sign * torch.exp(self.log_scale))
        return lower, upper

    def weight(self) -> torch.Tensor:
        lower, upper = self._factors()
        return self.permutation @ lower @ upper

    @torch.no_grad()
    def set_affine(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        p, lower, upper = torch.linalg.lu(weight.double())
        diagonal = torch.diagonal(upper)
        self.permutation.copy_(p.to(self.permutation.dtype))
        self.lower.copy_(lower.to(self.lower.dtype))
        self.upper.copy_(upper.to(self.upper.dtype))
        self.sign.copy_(torch.sign(diagonal).to(self.sign.dtype))
        self.log_scale.copy_(diagonal.abs().log().to(self.log_scale.dtype))
        self.bias.copy_(bias.to(self.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight().T + self.bias

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        lower, upper = self._factors()
        rhs = (y - self.bias).reshape(-1, self.dim).T            # (dim, n)
        rhs = self.permutation.T @ rhs
        rhs = torch.linalg.solve_triangular(lower, rhs, upper=False, unitriangular=True)
        x = torch.linalg.solve_triangular(upper, rhs, upper=True)
        return x.T.reshape(y.shape)


class AffineCoupling(nn.Module):
    """RealNVP coupling on alternating halves; starts at the identity."""

    def __init__(self, dim: int, hidden: int, flip: bool):
        super().__init__()
        self.half = dim // 2
        self.flip = flip
        condition = self.half if not flip else dim - self.half
        transformed = dim - condition
        self.net = nn.Sequential(nn.Linear(condition, hidden), nn.SiLU(), nn.Linear(hidden, 2 * transformed))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _split(self, x):
        a, b = x[..., :self.half], x[..., self.half:]
        return (b, a) if self.flip else (a, b)

    def _join(self, condition, transformed):
        return torch.cat([transformed, condition] if self.flip else [condition, transformed], dim=-1)

    def forward(self, x):
        condition, transformed = self._split(x)
        log_scale, shift = self.net(condition).chunk(2, dim=-1)
        return self._join(condition, transformed * torch.exp(torch.tanh(log_scale)) + shift)

    def inverse(self, y):
        condition, transformed = self._split(y)
        log_scale, shift = self.net(condition).chunk(2, dim=-1)
        return self._join(condition, (transformed - shift) * torch.exp(-torch.tanh(log_scale)))


class Bijection(nn.Module):
    """An invertible linear layer followed by ``couplings`` affine coupling layers."""

    def __init__(self, dim: int, couplings: int = 0, hidden: int = 512):
        super().__init__()
        self.linear = InvertibleLinear(dim)
        self.couplings = nn.ModuleList(AffineCoupling(dim, hidden, flip=bool(i % 2)) for i in range(couplings))

    def forward(self, x):
        y = self.linear(x)
        for layer in self.couplings:
            y = layer(y)
        return y

    def inverse(self, y):
        for layer in reversed(self.couplings):
            y = layer.inverse(y)
        return self.linear.inverse(y)


class SharedSpace(nn.Module):
    def __init__(self, large_dim: int, small_dim: int, couplings: int = 0, hidden: int = 512):
        super().__init__()
        if small_dim > large_dim:
            raise ValueError("the common part cannot be wider than the large model")
        self.large_dim, self.small_dim = large_dim, small_dim
        self.f_large = Bijection(large_dim, couplings, hidden)
        self.f_small = Bijection(small_dim, couplings, hidden)
        private = large_dim - small_dim
        self.prior_mean = nn.Linear(small_dim, private) if private else None
        self.prior_log_variance = nn.Parameter(torch.zeros(private))
        self.log_shrink = nn.Parameter(torch.zeros(small_dim))
        if self.prior_mean is not None:
            nn.init.zeros_(self.prior_mean.weight)
            nn.init.zeros_(self.prior_mean.bias)

    # coordinates
    def large_to_z(self, h):
        z = self.f_large(h)
        return z[..., :self.small_dim], z[..., self.small_dim:]

    def z_to_large(self, common, private):
        return self.f_large.inverse(torch.cat([common, private], dim=-1))

    def small_to_c(self, h):
        return self.f_small(h)

    def c_to_small(self, common):
        return self.f_small.inverse(common)

    def prior(self, common):
        if self.prior_mean is None:
            return common[..., :0]
        return self.prior_mean(common)

    # translation between models
    def shrink(self, common):
        return common * torch.exp(self.log_shrink)

    def large_to_small(self, h):
        common, _ = self.large_to_z(h)
        return self.c_to_small(self.shrink(common))

    def small_to_large(self, h):
        common = self.shrink(self.small_to_c(h))
        return self.z_to_large(common, self.prior(common))

    # losses
    def prior_nll(self, h_large):
        """Gaussian negative log-likelihood of the large model's private part given its common part (per vector)."""
        common, private = self.large_to_z(h_large)
        error = private - self.prior(common)
        log_var = self.prior_log_variance
        return 0.5 * (error.pow(2) * torch.exp(-log_var) + log_var).sum(-1).mean()

    def common_agreement(self, h_large, h_small, detach_large: bool = True):
        """Mean squared distance in C between the large model's (shrunk) and the small model's coordinates."""
        common, _ = self.large_to_z(h_large)
        target = self.shrink(common)
        if detach_large:
            target = target.detach()
        return (self.small_to_c(h_small) - target).pow(2).sum(-1).mean()

    @torch.no_grad()
    def init_from_pairs(self, h_large: torch.Tensor, h_small: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
        """Initialise the linear layers by CCA on paired residuals (same tokens, same positions).

        C holds the canonical variates of each model (whitened, unit variance); P holds the large model's whitened
        directions orthogonal to its canonical ones; the shrink starts at the canonical correlations and the prior at
        zero (uncorrelated with C by construction). Returns the canonical correlations."""
        x = h_large.double().reshape(-1, self.large_dim)
        y = h_small.double().reshape(-1, self.small_dim)
        mean_x, mean_y = x.mean(0), y.mean(0)
        x, y = x - mean_x, y - mean_y
        n = x.shape[0]
        whiten_x = _inverse_sqrt(x.T @ x / (n - 1), ridge)
        whiten_y = _inverse_sqrt(y.T @ y / (n - 1), ridge)
        cross = whiten_x @ (x.T @ y / (n - 1)) @ whiten_y
        u, rho, vh = torch.linalg.svd(cross, full_matrices=True)  # u: large × large, vh: small × small
        # u's first small_dim columns are canonical, the rest span the complement (orthonormal in whitened space)
        large_matrix = u.T @ whiten_x
        small_matrix = vh @ whiten_y
        self.f_large.linear.set_affine(large_matrix, -(large_matrix @ mean_x))
        self.f_small.linear.set_affine(small_matrix, -(small_matrix @ mean_y))
        self.log_shrink.copy_(rho.clamp_min(1e-3).log().to(self.log_shrink.dtype))
        return rho.float()


def _inverse_sqrt(covariance: torch.Tensor, ridge: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(0) + ridge * values.mean().clamp_min(1e-12)
    return vectors @ torch.diag(values.rsqrt()) @ vectors.T
