"""Token-to-Neuralese input map: parallel training inputs for Neuralese positions without sketch passes.

Owner decision (2026-10-08): instead of sketch rollouts, Neuralese positions read a learned transform of the gold
token embeddings, f(tokens) = E[tok] + causal depthwise conv (local context) + low-rank residual. Both residuals
start at zero, so f starts exactly at the crisp history. The map is trained toward the model's own projected output
at the same slot (stop-gradient self-consistency), so it tracks the systematic Neuralese drift as the model trains;
the residual between that projection and f measures what Neuralese carries beyond the tokens. Training-time only:
at inference Neuralese inputs are the model's own projections.
"""
import torch
from torch import nn
from torch.nn import functional as F


class NeuraleseInputMap(nn.Module):
    def __init__(self, width, kernel=4, rank=64):
        super().__init__()
        if kernel < 1 or rank < 1:
            raise ValueError('positive kernel and rank required')
        self.kernel = kernel
        self.conv = nn.Conv1d(width, width, kernel, groups=width, bias=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, embeddings):
        """[B,T,d] token embeddings of the slots (each slot sees its own and earlier tokens) -> Neuralese inputs."""
        x = embeddings.float()
        if x.shape[1] == 0:
            # No slots, no inputs (a one-token span's mapped history in the second pass): the causal map of an empty
            # sequence is empty, and the padded conv would refuse an input shorter than its kernel.
            return embeddings.new_zeros(embeddings.shape)
        local = self.conv(F.pad(x.transpose(1, 2), (self.kernel - 1, 0))).transpose(1, 2)
        return (x + local + self.up(self.down(x))).to(embeddings.dtype)
