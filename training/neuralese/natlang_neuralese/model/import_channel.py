"""Gaussian import channel: a variational bottleneck on imported native state (plans/neuralese/NEURALESE_HINDSIGHT_HINTS.md).

A declared set of producer invocations (``sources``) is read by its consumers through a noisy channel. For every
imported row r of a channelled source, every layer l and field f (attention ``k`` before rotary encoding and ``v``;
finite convolution ``gated``), with a frozen reference scale c = ``scales[(l, f)]``:

    z = mu + sigma * c * eps,   eps ~ N(0, I) fixed per (seed, source, layer, field, row)   (reparameterized)
    KL = sum_d [ (mu^2 / c^2 + sigma^2) / (2 sigma_p^2) - 1/2 + log(sigma_p / sigma) ]      = KL(N(mu, sigma^2 c^2) || N(0, sigma_p^2 c^2))

One noise sample per imported row: every consumer slot that reads a row sees the same realization (the written
state is transmitted once), and that row's KL counts once. The producer's own rows (its own attention and
convolution windows) are never perturbed, so the writer's sampling law is unchanged. The packed graph forward
(``shared_state._forward_graph``) applies the channel; it returns ``result['import_channel']`` with the KL summed
per source (nats, FP32, differentiable). Mechanism only: which edges are channelled and the constants are the
caller's declaration.
"""
from __future__ import annotations

import contextvars
import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from ..common.hashing import canonical_json_sha256_hex

_CHANNEL = contextvars.ContextVar('natlang_import_channel', default=None)
FIELDS = ('k', 'v', 'gated')


@dataclass(frozen=True)
class ImportChannel:
    sources: tuple            # producer invocation names whose imported rows pass the channel
    sigma: float
    sigma_prior: float
    scales: dict = field(compare=False)   # {(layer, field): reference RMS}
    seed: int = 0

    def __post_init__(self):
        if not self.sources or len(set(self.sources)) != len(self.sources):
            raise ValueError('an import channel names distinct source invocations')
        for name in ('sigma', 'sigma_prior'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'import channel {name} must be a positive finite number')
        if any(not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0)
               for value in self.scales.values()):
            raise ValueError('import channel reference scales must be positive and finite')

    def declaration(self):
        return {'schema': 'natlang.import-channel/1', 'sigma': float(self.sigma), 'sigma_prior': float(self.sigma_prior),
                'seed': self.seed, 'scales': sorted([layer, name, float(value)]
                                                    for (layer, name), value in self.scales.items())}

    def scale(self, layer, name):
        try:
            return float(self.scales[(layer, name)])
        except KeyError:
            raise ValueError(f'import channel has no reference scale for layer {layer} field {name}') from None

    def noise(self, source, layer, name, rows, shape, device):
        """eps for one source's rows 0..rows-1 (CPU generator, so identical on every device and recompute)."""
        key = canonical_json_sha256_hex([self.seed, source, layer, name])
        generator = torch.Generator().manual_seed(int(key[:15], 16))
        return torch.randn((rows, *shape), generator=generator, dtype=torch.float32).to(device)

    def kl_constant(self):
        return math.log(self.sigma_prior / self.sigma) + self.sigma ** 2 / (2 * self.sigma_prior ** 2) - .5


@contextmanager
def import_channel(channel):
    token = _CHANNEL.set(channel)
    try:
        yield channel
    finally:
        _CHANNEL.reset(token)


def active_channel():
    return _CHANNEL.get()


class PlannedChannel:
    """The active channel resolved against one compiled plan (slots, rows and per-layer noise)."""

    def __init__(self, channel, plan, device):
        names = [node.name for node in plan.invocations]
        unknown = set(channel.sources) - set(names)
        if unknown:
            raise ValueError('import channel names no invocation of this graph: ' + ', '.join(sorted(unknown)))
        self.channel, self.plan, self.device = channel, plan, device
        index = {name: number for number, name in enumerate(names)}
        self.sources = tuple(channel.sources)
        self.ranges = {}  # source -> (number, physical offset, own token count)
        for number, name in enumerate(self.sources):
            node = index[name]
            self.ranges[name] = (number, plan.offsets[node], len(plan.invocations[node].token_ids))
        source_of_row = {}
        for name, (number, offset, count) in self.ranges.items():
            for row in range(offset, offset + count):
                source_of_row[row] = (number, row - offset)
        # Channelled slots: positions of a consumer's sequence holding a channelled source's row (never the source's
        # own sequence: its own rows are its own computation).
        slots, keys = [], []
        imported = set()
        for node, (sequence, base) in enumerate(zip(plan.sequences, plan.slot_offsets)):
            own = names[node]
            for position, row in enumerate(sequence):
                found = source_of_row.get(row)
                if found is None or self.sources[found[0]] == own:
                    continue
                slots.append(base + position)
                keys.append(found)
                imported.add(found)
        self.imported = sorted(imported)
        global_index = {key: number for number, key in enumerate(self.imported)}
        self.slot_index = torch.tensor(slots, dtype=torch.long, device=device)
        self.slot_rows = torch.tensor([global_index[key] for key in keys], dtype=torch.long, device=device)
        self.kl_rows = torch.tensor([self.ranges[self.sources[number]][1] + row for number, row in self.imported],
                                    dtype=torch.long, device=device)
        self.kl_sources = torch.tensor([number for number, _ in self.imported], dtype=torch.long, device=device)
        self.slot_global = dict(zip(slots, (global_index[key] for key in keys)))
        self._conv = {}

    @property
    def empty(self):
        return not self.imported

    def table(self, layer, name, shape):
        """sigma * c * eps for every imported row (global row order), FP32."""
        by_source = {}
        for number, row in self.imported:
            by_source.setdefault(number, []).append(row)
        rows = torch.empty((len(self.imported), *shape), dtype=torch.float32, device=self.device)
        position = {key: i for i, key in enumerate(self.imported)}
        for number, wanted in by_source.items():
            source = self.sources[number]
            eps = self.channel.noise(source, layer, name, self.ranges[source][2], shape, self.device)
            targets = torch.tensor([position[(number, row)] for row in wanted], device=self.device)
            rows[targets] = eps[torch.tensor(wanted, device=self.device)]
        return rows * (self.channel.sigma * self.channel.scale(layer, name))

    def kl(self, layer, name, mu):
        """Per-source KL of the clean imported rows ``mu`` [G, dims...] (FP32 sum over dims)."""
        c = self.channel.scale(layer, name)
        per_row = (mu.float().pow(2).flatten(1).sum(1) / (c * c)) / (2 * self.channel.sigma_prior ** 2)
        per_row = per_row + mu[0].numel() * self.channel.kl_constant()
        return per_row.new_zeros(len(self.sources)).index_add(0, self.kl_sources, per_row)

    def attention(self, layer, k, v, key_slots, value_slots):
        """Noisy key/value slots and this layer's per-source KL. ``k``/``v``: [1, H, rows, D] own rows (pre-rotary)."""
        heads, dim = k.shape[1], k.shape[3]
        noise_k = self.table(layer, 'k', (heads, dim))[self.slot_rows]   # [n, H, D]
        noise_v = self.table(layer, 'v', (heads, dim))[self.slot_rows]
        key_slots = key_slots.index_add(2, self.slot_index, noise_k.permute(1, 0, 2)[None].to(key_slots.dtype))
        value_slots = value_slots.index_add(2, self.slot_index, noise_v.permute(1, 0, 2)[None].to(value_slots.dtype))
        mu_k = k[0][:, self.kl_rows].permute(1, 0, 2)
        mu_v = v[0][:, self.kl_rows].permute(1, 0, 2)
        return key_slots, value_slots, self.kl(layer, 'k', mu_k) + self.kl(layer, 'v', mu_v)

    def convolution_ids(self, width):
        """[rows, width] global imported-row id per convolution history entry (-1: none), as ``_convolution_index``."""
        if width not in self._conv:
            plan, result = self.plan, []
            for row in range(plan.token_count):
                node = plan.node_ids[row]
                end = plan.own_positions[row] + 1
                start = max(0, end - width)
                base = plan.slot_offsets[node]
                ids = [self.slot_global.get(base + position, -1) for position in range(start, end)]
                result.append([-1] * (width - len(ids)) + ids)
            self._conv[width] = torch.tensor(result, dtype=torch.long, device=self.device)
        return self._conv[width]

    def convolution(self, layer, gated, width):
        """(noise [rows, width, C] for the gathered histories, this layer's per-source KL)."""
        channels = gated.shape[-1]
        table = torch.cat([self.table(layer, 'gated', (channels,)), gated.new_zeros(1, channels, dtype=torch.float32)])
        ids = self.convolution_ids(width)
        noise = table[torch.where(ids < 0, len(self.imported), ids)]
        return noise, self.kl(layer, 'gated', gated[0][self.kl_rows])


def measure_scales(model, plan):
    """Reference RMS of k, v (attention) and gated (convolution) over a plan's own rows, per layer."""
    with torch.no_grad():
        result = model(plan, retain_states=True)
    scales = {}
    rows = plan.token_count
    for layer, (kind, state) in enumerate(zip(result['layer_kinds'], result['states'])):
        if kind == 'attention':
            scales[(layer, 'k')] = float(state[0][:, :, :rows].float().pow(2).mean().sqrt())
            scales[(layer, 'v')] = float(state[1][:, :, :rows].float().pow(2).mean().sqrt())
        elif kind == 'finite_convolution':
            scales[(layer, 'gated')] = float(state[0][:, :rows].float().pow(2).mean().sqrt())
        else:
            raise ValueError('the import channel supports attention and finite-convolution layers')
    return scales
