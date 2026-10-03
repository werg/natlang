"""The new port modules of S3 §2: feedback projection, stop head, content projection, interface norm."""

from __future__ import annotations

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
    readout R_k initialised from the tied head (with the final norm), and a residual MLP
    that starts at zero. The sum passes the interface norm.
    """

    def __init__(self, embedding: torch.Tensor, final_norm_weight: torch.Tensor, interface: InterfaceNorm,
                 tau: float = 1.0, eps: float = 1e-5):
        super().__init__()
        vocab, dim = embedding.shape
        self.tau = tau
        self.readout_norm = RMSNorm(dim, eps=eps)
        with torch.no_grad():
            self.readout_norm.weight.copy_(final_norm_weight.float())
        self.readout = nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self.readout.weight.copy_(embedding.float())
        self.register_buffer("embedding", embedding.detach().clone(), persistent=False)
        self.mlp_norm = RMSNorm(dim, eps=eps)
        self.mlp_in = nn.Linear(dim, 2 * dim)
        self.mlp_out = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)
        self.gate = nn.Parameter(torch.ones(()))
        self.interface = interface

    def readout_logits(self, h: torch.Tensor) -> torch.Tensor:
        """R_k(h): the temporary vocabulary readout, distilled in phase B."""
        return self.readout(self.readout_norm(h).to(self.readout.weight.dtype))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.readout_logits(h).float() / self.tau, dim=-1)
        mixture = weights @ self.embedding.float()
        mlp = self.mlp_out(F.gelu(self.mlp_in(self.mlp_norm(h).to(self.mlp_in.weight.dtype))))
        return self.interface(mixture.to(h.dtype) + self.gate.to(h.dtype) * mlp.to(h.dtype))


class StopHead(nn.Module):
    """P(stop | h_k[i], i). Starts rarely stopping (bias -3)."""

    def __init__(self, dim: int, max_length: int, position_dim: int = 64, hidden: int = 256, eps: float = 1e-5):
        super().__init__()
        self.max_length = max_length
        self.norm = RMSNorm(dim, eps=eps)
        self.position = nn.Embedding(max_length + 1, position_dim)
        nn.init.normal_(self.position.weight, std=0.02)
        self.mlp_in = nn.Linear(dim + position_dim, hidden)
        self.mlp_out = nn.Linear(hidden, 1)
        nn.init.normal_(self.mlp_out.weight, std=0.02)
        nn.init.constant_(self.mlp_out.bias, -3.0)

    def forward(self, h: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Stop logit after `count` vectors have been written (count = 0 is masked by the caller)."""
        position = self.position(count.clamp(max=self.max_length))
        features = torch.cat([self.norm(h).to(position.dtype), position], dim=-1)
        return self.mlp_out(F.gelu(self.mlp_in(features))).squeeze(-1)


class ContentProjection(nn.Module):
    """p[i] = s[i] + P(h_D[i]); P starts at zero so the first payload equals the sketch."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, sketch: torch.Tensor, h_final: torch.Tensor) -> torch.Tensor:
        return sketch + self.proj(self.norm(h_final).to(self.proj.weight.dtype)).to(sketch.dtype)


class PortHeads(nn.Module):
    """All trainable port modules for one backbone and cutoff."""

    def __init__(self, backbone, cutoff: int, max_length: int = 128, tau: float = 1.0):
        super().__init__()
        if not 0 < cutoff < backbone.num_layers:
            raise ValueError(f"cutoff must be inside the stack, got {cutoff}")
        embedding = backbone.embedding_weight.detach()
        self.cutoff = cutoff
        self.max_length = max_length
        self.interface = InterfaceNorm(embedding, eps=backbone.config.norm_eps)
        self.feedback = FeedbackProjection(embedding, backbone.hf.model.embedding_norm.weight.detach(),
                                           self.interface, tau=tau, eps=backbone.config.norm_eps)
        self.stop = StopHead(embedding.shape[1], max_length, eps=backbone.config.norm_eps)
        self.content = ContentProjection(embedding.shape[1], eps=backbone.config.norm_eps)
