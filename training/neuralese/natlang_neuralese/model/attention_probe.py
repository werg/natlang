"""Max pre-softmax attention logit per attention layer, read from the real training forward (telemetry only).

The shared native graph forward calls ``observe`` with the exact post-QK-norm, post-RoPE queries and the key slots it
feeds to attention, so no extra model forward is needed. The probe recomputes ``scale * q . k`` over the visible
(row, key slot) pairs of the packed graph plan in bounded chunks (one extra QK product per observed layer, on logged
steps only; a layer subset bounds that cost further). It runs under ``no_grad`` on detached tensors, consumes no
RNG and never raises into the training step (a failure is recorded in its result).

Three maxima per layer: ``all`` visible pairs; ``imported_keys`` pairs whose key slot is spliced Neuralese state
(imported from a producer or external block; raw exact-prefix cache imports are not Neuralese); ``consumer_rows``
pairs of query rows that can see at least one imported Neuralese slot, i.e. positions that consume imports.
Activation-checkpoint recomputation re-enters the layer, but a layer is measured once per probe.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

_ACTIVE = None


def active_probe():
    return _ACTIVE


@contextmanager
def probing(probe):
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = probe
    try:
        yield probe
    finally:
        _ACTIVE = previous


def observe_attention(layer, q, k, plan, window, scale):
    """Call site helper: free when no probe is active."""
    probe = _ACTIVE
    if probe is not None:
        probe.observe(layer, q, k, plan, window, scale)


def _slot_classes(plan, device):
    """Per key slot: imported Neuralese state. Per query row: whether it can see one is computed from a prefix sum."""
    token_count = plan.token_count
    neural = torch.zeros(len(plan.slot_sources), dtype=torch.bool)
    for index, node in enumerate(plan.invocations):
        low = plan.offsets[index]
        high = low + len(node.token_ids)
        first = plan.slot_offsets[index]
        for position, source in enumerate(plan.sequences[index]):
            if low <= source < high:
                continue
            producer = plan.invocations[plan.node_ids[source]] if source < token_count else None
            if producer is None or not producer.cache_prefix:
                neural[first + position] = True
    return neural.to(device)


class AttentionLogitProbe:
    def __init__(self, layers=None, *, chunk_elements=1 << 26):
        self.layers = None if layers is None else set(layers)
        self.chunk_elements = int(chunk_elements)
        self._pending = {}
        self._plan_cache = None
        self.results = {}

    def observe(self, layer, q, k, plan, window, scale):
        if layer in self._pending or layer in self.results or (self.layers is not None and layer not in self.layers):
            return
        if not hasattr(plan, 'node_ids') or q.shape[2] != plan.token_count:
            return
        try:
            with torch.no_grad():
                self._pending[layer] = self._measure(q.detach(), k.detach(), plan, window, scale)
        except Exception as error:  # telemetry must never fail the training step
            self.results[layer] = {'error': f'{type(error).__name__}: {error}'[:300]}

    def _measure(self, q, k, plan, window, scale):
        device = q.device
        if self._plan_cache is None or self._plan_cache[0] is not plan or self._plan_cache[1] != device:
            neural = _slot_classes(plan, device)
            cumulative = torch.cat((torch.zeros(1, dtype=torch.long, device=device), neural.long().cumsum(0)))
            self._plan_cache = (plan, device, neural, cumulative)
        _, _, neural, cumulative = self._plan_cache
        starts, ends = plan.bounds(device, window)
        rows, slots = q.shape[2], k.shape[2]
        heads, kv_heads, width = q.shape[1], k.shape[1], q.shape[3]
        group = heads // kv_heads
        queries = q[0].reshape(kv_heads, group, rows, width)
        keys = k[0].float()
        index = torch.arange(slots, device=device)
        consumer = (cumulative[ends + 1] - cumulative[starts]) > 0
        low = torch.full((), -float('inf'), device=device)
        best_all, best_neural, best_consumer = low.clone(), low.clone(), low.clone()
        chunk = max(1, self.chunk_elements // max(1, heads * slots))
        for begin in range(0, rows, chunk):
            stop = min(rows, begin + chunk)
            scores = torch.einsum('kgcd,ksd->kgcs', queries[:, :, begin:stop].float(), keys) * scale
            visible = (index[None] >= starts[begin:stop, None]) & (index[None] <= ends[begin:stop, None])
            scores = scores.masked_fill(~visible[None, None], -float('inf'))
            best_all = torch.maximum(best_all, scores.amax())
            best_neural = torch.maximum(best_neural,
                scores.masked_fill(~neural[None, None, None], -float('inf')).amax())
            best_consumer = torch.maximum(best_consumer,
                scores.masked_fill(~consumer[begin:stop][None, None, :, None], -float('inf')).amax())
        return {'all': best_all, 'imported_keys': best_neural, 'consumer_rows': best_consumer,
                'imported_slots': int(neural.sum()), 'consumer_row_count': int(consumer.sum()),
                'rows': rows, 'key_slots': slots}

    def finalize(self):
        """Host values per layer (one synchronization); ``None`` when no pair of that kind exists."""
        for layer, measured in self._pending.items():
            entry = {}
            for key, value in measured.items():
                if torch.is_tensor(value):
                    number = float(value)
                    entry[key] = number if number != -float('inf') else None
                else:
                    entry[key] = value
            self.results[layer] = entry
        self._pending = {}
        return dict(sorted(self.results.items()))
