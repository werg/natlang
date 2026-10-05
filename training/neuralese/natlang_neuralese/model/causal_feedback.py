"""Raw next-token embedding bootstrap, anchored to the frozen full LM head.

At full depth the zero residual gives the ordinary greedy next-token embedding.
At a shallower cutoff only the state correction is distilled. No interface RMS,
payload compression, extra positions or embedding-mixture assumption is involved.
"""
import torch
import copy
from torch import nn
from torch.nn import functional as F

from .heads import RMSNorm


class CausalFeedbackProjection(nn.Module):
    def __init__(self, backbone, hidden=None):
        super().__init__()
        table = backbone.embedding_weight.detach().clone()
        output = backbone.output_weight.detach().clone()
        for index, row in zip((backbone.controls.open_id, backbone.controls.close_id), backbone.control_rows):
            table[index].copy_(row.to(table))
        self.register_buffer('embedding', table)
        # Tied tables share checkpoint storage. Untied architectures remain explicit.
        if torch.equal(table, output):
            output = table
        self.register_buffer('readout', output)
        self.register_buffer('control_rows', backbone.control_rows.detach().clone())
        self.control_ids = (backbone.controls.open_id, backbone.controls.close_id)
        width = table.shape[1]
        hidden = hidden or width * 2
        self.state_norm = RMSNorm(width, eps=backbone.norm_eps)
        self.state_in = nn.Linear(width, hidden)
        self.state_out = nn.Linear(hidden, width)
        nn.init.zeros_(self.state_out.weight)
        nn.init.zeros_(self.state_out.bias)
        # LFM multiplies its gain AFTER rounding normalized states to the input
        # dtype. Reimplementing it as float gain * float states then rounding
        # changes BF16 logits. Copy the real frozen module, not just its gain.
        self.final_norm = copy.deepcopy(backbone.hf.model.embedding_norm)
        for parameter in self.final_norm.parameters():
            parameter.requires_grad_(False)

    def complete_state(self, state):
        correction = self.state_out(F.gelu(self.state_in(self.state_norm(state).float())))
        return state + correction.to(state.dtype)

    def logits(self, state):
        final = self.complete_state(state)
        normed = self.final_norm(final).to(self.readout.dtype)
        logits = normed @ self.readout.t()
        logits = logits.clone()
        # Match the backbone's separate control-row GEMVs as well as its GEMM.
        rows = self.control_rows.to(normed.dtype)
        logits[..., self.control_ids[0]] = normed @ rows[0]
        logits[..., self.control_ids[1]] = normed @ rows[1]
        return logits

    def forward(self, state, *, straight_through=False):
        logits = self.logits(state)
        selected = F.embedding(logits.argmax(-1), self.embedding)
        if not straight_through:
            return selected
        # Forward is selected E exactly; backward uses a soft categorical surrogate.
        probabilities = logits.float().softmax(-1).to(self.embedding.dtype)
        relaxed = probabilities @ self.embedding
        return selected + (relaxed - relaxed.detach())
