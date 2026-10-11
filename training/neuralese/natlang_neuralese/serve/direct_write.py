"""Full-depth direct Neuralese writing, shared by serving and differentiable replay.

Each final state predicts the next ordinary token and projects its next input
embedding. The projected value is fed through the full backbone once. A call is
one causal-convolution segment; no projection state crosses a call boundary.
"""
from __future__ import annotations

import torch


class DirectWriter:
    def __init__(self, backbone, projection, opened, max_length, stop_ids, *, length=None):
        if opened.top is None:
            raise ValueError('direct writing needs the final causal prefix state')
        self.backbone, self.projection, self.opened = backbone, projection, opened
        self.cache = opened.cache
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError('direct writing needs positive capacity')
        self.length = None if length is None else max(1, min(int(length), self.max_length))
        self.stop_ids = tuple(sorted(set(stop_ids)))
        if not self.stop_ids:
            raise ValueError('direct writing needs the ordinary readout stop tokens')
        self.history = [opened.top]
        self.payloads, self.states, self.stop_tensors = [], [], []
        self.stop_logits = []
        self.count, self.done, self.truncated, self.passes = 0, False, False, 0
        config = projection.stack.config
        self.history_limit = 1 + config['blocks'] * (config['kernel'] - 1)

    def stop_logit(self):
        logits = self.backbone.logits(self.history[-1][:, None])[:, -1].float()
        logp = torch.log_softmax(logits, -1)
        stop_logp = torch.logsumexp(logp[:, self.stop_ids], -1)
        # Log odds are useful for replay scoring; greedy stopping itself uses argmax.
        logit = stop_logp - torch.log1p(-stop_logp.exp().clamp(max=1 - 1e-7))
        return logit, logits.argmax(-1)

    def decide(self):
        if self.done:
            return None
        stop_logit, token = self.stop_logit()
        if self.count:
            self.stop_tensors.append(stop_logit)
            self.stop_logits.append(float(stop_logit.detach()[0]))
        stop = self.count > 0 and int(token[0]) in self.stop_ids
        if self.length is not None:
            stop = self.count == self.length
        if stop or self.count == self.max_length:
            self.done = True
            self.truncated = not stop and self.length is None
            return None
        hidden = torch.stack(self.history[-self.history_limit:], 1)
        # Only the last position needs a vocabulary mixture. Causal conv still
        # receives its complete receptive field, identically to the sequence pass.
        logits = self.backbone.logits(hidden[:, -1:])
        mixture = torch.softmax(logits.float(), -1).to(self.backbone.embedding_weight.dtype) @ \
            self.backbone.embedding_weight
        segments = torch.zeros(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        value = self.projection(hidden, mixture, segments, last_only=True)
        return value[:, 0]

    def advance(self, value, final, cache):
        self.payloads.append(value)
        self.states.append(final)
        self.history.append(final)
        # Past states outside the convolution receptive field remain in the
        # backbone KV cache, but need not be retained a second time here.
        self.history = self.history[-self.history_limit:]
        self.cache = cache
        self.count += 1

    def step(self):
        value = self.decide()
        if value is None:
            return True
        final, cache = self.backbone.run_layers(value[:, None], range(self.backbone.num_layers), self.cache)
        self.advance(value, final[:, 0], cache)
        return False

    def complete(self, temperature=0.0, generator=None):
        if temperature:
            raise ValueError('direct Neuralese values have no trained stochastic payload distribution')
        payload = torch.stack(self.payloads, 1)
        # The common transport accepts mean/scale companions. This deterministic
        # writer records its identity, so no Gaussian density is scored for it.
        return payload, payload, torch.zeros_like(payload, dtype=torch.float32)


def write_direct(backbone, projection, opened, max_length, stop_ids, *, length=None):
    """Run the same incremental writer without a scheduler; retains gradient paths."""
    writer = DirectWriter(backbone, projection, opened, max_length, stop_ids, length=length)
    while not writer.step():
        pass
    return writer
