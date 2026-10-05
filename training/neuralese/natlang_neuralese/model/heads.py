"""The new port modules of S3 §2: feedback projection, stop head, content projection (a payload
distribution with temperature-gated sampling), interface norm."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, gain: float = 1.0):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.full((dim,), float(gain)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def embedding_rms(embedding: torch.Tensor) -> float:
    return float(embedding.float().pow(2).mean(-1).sqrt().mean())


class InterfaceNorm(RMSNorm):
    """Applied to every vector entering the read port and to every sketch input.

    Gain starts at the mean RMS of the embedding rows, so vectors enter at token scale.
    """

    def __init__(self, embedding: torch.Tensor, eps: float = 1e-5):
        super().__init__(embedding.shape[1], eps=eps, gain=embedding_rms(embedding))


class FeedbackProjection(nn.Module):
    """Maps the shallow residual h_k[i] to the next sketch input s[i+1].

    Two branches: a vocabulary mixture E^T softmax(R_k(h)/tau) through a temporary
    readout R_k initialised from the output head (with the final norm), and a residual MLP
    that starts at zero. The sum passes the interface norm.
    """

    def __init__(self, embedding: torch.Tensor, final_norm_weight: torch.Tensor, interface: InterfaceNorm,
                 tau: float = 1.0, eps: float = 1e-5, head: torch.Tensor | None = None):
        super().__init__()
        vocab, dim = embedding.shape
        self.tau = tau
        self.readout_norm = RMSNorm(dim, eps=eps)
        with torch.no_grad():
            self.readout_norm.weight.copy_(final_norm_weight.float())
        self.readout = nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self.readout.weight.copy_((embedding if head is None else head).float())
        # The mixture table, in the backbone's dtype. The sketch step is bandwidth-bound on the
        # two vocabulary-sized tables, so they are never cast per step.
        self.register_buffer("embedding", embedding.detach().clone(), persistent=False)
        self._readout_cache: tuple | None = None
        self.mlp_norm = RMSNorm(dim, eps=eps)
        self.mlp_in = nn.Linear(dim, 2 * dim)
        self.mlp_out = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)
        self.gate = nn.Parameter(torch.ones(()))
        self.interface = interface

    def readout_logits(self, h: torch.Tensor) -> torch.Tensor:
        """R_k(h): the temporary vocabulary readout, distilled in phase B."""
        if not torch.is_grad_enabled() and self.embedding.dtype != self.readout.weight.dtype:
            # Inference: a copy of the readout in the table's dtype, refreshed when the weight changes.
            weight = self.readout.weight
            key = (weight._version, weight.data_ptr(), self.embedding.dtype)
            if self._readout_cache is None or self._readout_cache[0] != key:
                self._readout_cache = (key, weight.detach().to(self.embedding.dtype))
            normed = self.readout_norm(h).to(self.embedding.dtype)
            return normed @ self._readout_cache[1].t()
        return self.readout(self.readout_norm(h).to(self.readout.weight.dtype))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.readout_logits(h).float() / self.tau, dim=-1)
        mixture = (weights.to(self.embedding.dtype) @ self.embedding).float()
        mlp = self.mlp_out(F.gelu(self.mlp_in(self.mlp_norm(h).to(self.mlp_in.weight.dtype))))
        return self.interface(mixture.to(h.dtype) + self.gate.to(h.dtype) * mlp.to(h.dtype))


class StopHead(nn.Module):
    """P(stop | h[i], i). Starts rarely stopping (bias -3).

    `h` is the sketch state h_k (stop source "shallow") or the completed full-depth state h_D (stop source "final").
    Without `use_position` the count is not an input (one learned constant takes its place), so the decision has to
    come from content: with a count input and spans of one length, the pilot's head learned to stop at that count.
    """

    def __init__(self, dim: int, max_length: int, position_dim: int = 64, hidden: int = 256, eps: float = 1e-5,
                 use_position: bool = True):
        super().__init__()
        self.max_length = max_length
        self.use_position = use_position
        self.norm = RMSNorm(dim, eps=eps)
        self.position = nn.Embedding(max_length + 1, position_dim)
        nn.init.normal_(self.position.weight, std=0.02)
        self.mlp_in = nn.Linear(dim + position_dim, hidden)
        self.mlp_out = nn.Linear(hidden, 1)
        nn.init.normal_(self.mlp_out.weight, std=0.02)
        nn.init.constant_(self.mlp_out.bias, -3.0)

    def forward(self, h: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Stop logit after `count` vectors have been written (count = 0 is masked by the caller)."""
        position = self.position(count.clamp(max=self.max_length) if self.use_position else torch.zeros_like(count))
        features = torch.cat([self.norm(h).to(position.dtype), position], dim=-1)
        return self.mlp_out(F.gelu(self.mlp_in(features))).squeeze(-1)


class ContentProjection(nn.Module):
    """The payload distribution of a written block.

    Mean: mu[i] = s[i] + P(h_D[i]); P starts at zero so the first payload equals the sketch.
    Scale: a per-dimension log-sigma head on h_D[i], starting small (`init_sigma`). Noise
    lives in the *normalised* payload space u = mu / rms(mu), where the read port's interface
    norm puts every vector anyway. A delivered payload is

        z = rms(mu) * (u + tau * sigma * eps),  eps ~ N(0, I),

    so temperature tau = 0 gives the deterministic mean (the inference default) and tau > 0
    gives a sample with a log-likelihood (`payload_log_prob`) for policy-gradient objectives
    and a KL to N(0, I) (`payload_kl`) for VAE-style robustness.
    """

    def __init__(self, dim: int, eps: float = 1e-5, init_sigma: float = 0.05):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.log_sigma = nn.Linear(dim, dim)
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.constant_(self.log_sigma.bias, math.log(init_sigma))
        self.eps = eps

    def forward(self, sketch: torch.Tensor, h_final: torch.Tensor) -> torch.Tensor:
        """The mean payload mu."""
        return sketch + self.proj(self.norm(h_final).to(self.proj.weight.dtype)).to(sketch.dtype)

    def distribution(self, sketch: torch.Tensor, h_final: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.forward(sketch, h_final)
        log_sigma = self.log_sigma(self.norm(h_final).to(self.log_sigma.weight.dtype)).float().clamp(-9.0, 2.0)
        return mu, log_sigma


@dataclass
class PayloadSample:
    payload: torch.Tensor          # z, [..., L, d]
    mean: torch.Tensor             # mu
    log_sigma: torch.Tensor        # per dimension, float32
    noise: torch.Tensor | None     # eps (None at tau = 0)
    temperature: float


def rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()


def sample_payload(mu: torch.Tensor, log_sigma: torch.Tensor, temperature: float = 0.0,
                   generator: torch.Generator | None = None) -> PayloadSample:
    if temperature <= 0:
        return PayloadSample(mu, mu, log_sigma, None, 0.0)
    noise = torch.randn(mu.shape, generator=generator, dtype=torch.float32).to(mu.device)
    scale = rms(mu)
    z = scale * (mu.float() / scale + temperature * log_sigma.exp() * noise)
    return PayloadSample(z.to(mu.dtype), mu, log_sigma, noise, temperature)


def payload_log_prob(sample: PayloadSample, payload: torch.Tensor | None = None) -> torch.Tensor:
    """log N(u_z; u_mu, tau^2 sigma^2) in normalised space, summed over dimensions: [..., L].

    Differentiable in mu and sigma with the payload held fixed (pass `payload` to score a
    different, e.g. recorded, payload). Undefined at tau = 0.
    """
    if sample.temperature <= 0:
        raise ValueError("payload log-likelihood needs temperature > 0")
    z = (sample.payload if payload is None else payload).detach().float()
    scale = rms(sample.mean)
    std = sample.temperature * sample.log_sigma.exp()
    standardized = (z / scale - sample.mean.float() / scale) / std
    return (-0.5 * standardized.pow(2) - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)


def payload_kl(sample: PayloadSample) -> torch.Tensor:
    """KL(N(u_mu, sigma^2) || N(0, I)) in normalised space, averaged over dimensions: [..., L]."""
    u = sample.mean.float() / rms(sample.mean)
    var = (2 * sample.log_sigma).exp()
    return 0.5 * (var + u.pow(2) - 1 - 2 * sample.log_sigma).mean(-1)


class PortHeads(nn.Module):
    """All trainable port modules for one backbone and cutoff."""

    def __init__(self, backbone, cutoff: int, max_length: int = 128, tau: float = 1.0, stop_source: str = "shallow",
                 stop_position: bool | None = None):
        super().__init__()
        if stop_source not in ("shallow", "final"):
            raise ValueError(f"stop_source is shallow or final, got {stop_source!r}")
        # "final": the stop decision after i vectors reads the completed state h_D[i]. Completion is causal, so a
        # block completed past its end and truncated is exactly the block that stopped there (lookahead).
        self.stop_source = stop_source
        if not 0 < cutoff < backbone.num_layers:
            raise ValueError(f"cutoff must be inside the stack, got {cutoff}")
        embedding = backbone.embedding_weight.detach()
        eps = backbone.norm_eps
        self.cutoff = cutoff
        self.max_length = max_length
        self.interface = InterfaceNorm(embedding, eps=eps)
        # The feedback readout starts from the output head (the embedding itself when tied).
        head = backbone.output_weight.detach()
        self.feedback = FeedbackProjection(embedding, backbone.final_norm_weight.detach(), self.interface, tau=tau,
                                           eps=eps, head=None if head is embedding else head)
        self.stop = StopHead(embedding.shape[1], max_length, eps=eps,
                             use_position=(stop_source == "shallow") if stop_position is None else stop_position)
        self.content = ContentProjection(embedding.shape[1], eps=eps)

    def stop_states(self, shallow: torch.Tensor, final: torch.Tensor | None) -> torch.Tensor:
        """The states the stop head reads: sketch states, or completed states with the final source."""
        if self.stop_source == "final":
            if final is None:
                raise ValueError("the final stop source needs completed states")
            return final
        return shallow

    def port_config(self) -> dict:
        return {"stop_source": self.stop_source, "stop_position": self.stop.use_position}
