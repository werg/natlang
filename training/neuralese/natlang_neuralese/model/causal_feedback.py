"""Raw next-token embedding bootstrap, anchored to the frozen full LM head.

At full depth the zero residual gives the ordinary greedy next-token embedding.
At a shallower cutoff only the state correction is distilled. No interface RMS,
payload compression, extra positions or embedding-mixture assumption is involved.
"""
import torch
import copy
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..maple.model import eager_rms_norm
from .heads import RMSNorm


class CausalFeedbackProjection(nn.Module):
    def __init__(self, backbone, hidden=None):
        super().__init__()
        table = backbone.embedding_weight.detach().clone()
        output = backbone.output_weight.detach().clone()
        # Snapshot reference rows without retaining the constructor's graph.
        # control_rows can still be trainable while a port is constructed.
        with torch.no_grad():
            for index, row in zip((backbone.controls.open_id, backbone.controls.close_id), backbone.control_rows):
                table[index].copy_(row.to(table))
        self.register_buffer('embedding', table, persistent=False)
        # Tied tables share checkpoint storage. Untied architectures remain explicit.
        if torch.equal(table, output):
            output = table
        self.register_buffer('readout', output, persistent=False)
        # Untied backbones have distinct output control rows.
        head_rows = backbone.control_rows if getattr(backbone, 'tied', True) else backbone.control_head_rows
        self.register_buffer('control_rows', head_rows.detach().clone(), persistent=False)

        self.control_ids = (backbone.controls.open_id, backbone.controls.close_id)
        width = table.shape[1]
        hidden = hidden or width * 2
        self.state_norm = RMSNorm(width, eps=backbone.norm_eps)
        self.state_in = nn.Linear(width, hidden)
        self.state_out = nn.Linear(hidden, width)
        nn.init.zeros_(self.state_out.weight)
        nn.init.zeros_(self.state_out.bias)
        self._frozen_identity = False

        # LFM multiplies its gain AFTER rounding normalized states to the input
        # dtype. Reimplementing it as float gain * float states then rounding
        # changes BF16 logits. Copy the real frozen module, not just its gain.
        # The backbone's own final norm module: LFM's embedding_norm, Qwen3/Maple's norm.
        model = backbone.hf.model
        self.final_norm = copy.deepcopy(model.embedding_norm if hasattr(model, 'embedding_norm') else model.norm)
        for parameter in self.final_norm.parameters():
            parameter.requires_grad_(False)

    def configure_frozen_identity(self):
        """Skip a certified zero correction only when it cannot be optimized."""
        self._frozen_identity = (not any(p.requires_grad for p in self.parameters()) and
                                 not bool(torch.count_nonzero(self.state_out.weight)) and
                                 not bool(torch.count_nonzero(self.state_out.bias)))
        self._identity_parameters = tuple(self.parameters())
        self._identity_signature = self._correction_signature()
        return self._frozen_identity

    def _correction_signature(self):
        return (id(self.state_out.weight), self.state_out.weight._version,
                id(self.state_out.bias), self.state_out.bias._version)

    def _load_from_state_dict(self, *args, **kwargs):
        self._frozen_identity = False
        return super()._load_from_state_dict(*args, **kwargs)

    def complete_state(self, state):
        if self._frozen_identity:
            if (self._identity_signature == self._correction_signature() and
                    not any(p.requires_grad for p in self._identity_parameters)):
                return state
            self._frozen_identity = False
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

    def readout_logits(self, state):
        return self.logits(state)

    def forward(self, state, *, straight_through=None):
        if straight_through is None:
            straight_through = torch.is_grad_enabled()
        # Vocabulary intermediates scale with tokens × vocabulary, while the
        # returned embeddings only scale with tokens × width. Recompute each
        # bounded slice on backward rather than retaining every softmax table.
        flat = state.reshape(-1, state.shape[-1])
        if flat.shape[0] > 256:
            pieces = []
            for chunk in flat.split(256):
                def project(value):
                    # The checkpoint recompute must replay the forward's exact graph: keep Maple's dynamic-shape
                    # compiled final norm out of it (an odd tail chunk recompiled differently, see eager_rms_norm).
                    with eager_rms_norm():
                        return self._project_tokens(value, straight_through=straight_through)
                if torch.is_grad_enabled() and (chunk.requires_grad or any(p.requires_grad for p in self.parameters())):
                    pieces.append(checkpoint(project, chunk, use_reentrant=False))
                else:
                    pieces.append(project(chunk))
            return torch.cat(pieces).reshape(state.shape)
        return self._project_tokens(state, straight_through=straight_through)

    def _project_tokens(self, state, *, straight_through):
        logits = self.logits(state)
        selected = F.embedding(logits.argmax(-1), self.embedding)
        if not straight_through:
            return selected
        # Forward is selected E exactly; backward uses a soft categorical surrogate.
        probabilities = logits.float().softmax(-1).to(self.embedding.dtype)
        relaxed = probabilities @ self.embedding
        return selected + (relaxed - relaxed.detach())


def load_projection_state(projection, state):
    """Explicit handoff: frozen tables come from the pinned teacher.

    Earlier experimental snapshots duplicated them. Validate those bytes before
    omitting redundant fields; never substitute a different teacher silently.
    """
    state = dict(state)
    for name in ['embedding', 'readout', 'control_rows']:
        if name in state:
            actual = getattr(projection, name)
            saved = state.pop(name).to(actual)
            if not torch.equal(saved, actual):
                raise ValueError('projection handoff teacher table differs: ' + name)
    projection.load_state_dict(state)
