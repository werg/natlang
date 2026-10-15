"""Shared per-token semantic posterior and model-specific native-state deltas.

This is the post-native-training adapter. Its latent is never substituted for
the native KV channel during the initial model-training phase.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .state_block import NativeStateBlock
from ..common.hashing import sha256_hex


@dataclass(frozen=True)
class StateLayout:
    embedding_width: int
    fields: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_block(cls, block):
        block.validate()
        fields = tuple((f'{index}.{name}', tuple(layer[name].shape[1:]))
            for index, layer in enumerate(block.layers)
            for name in (('raw_k', 'v') if layer['kind'] == 'attention' else ('gated_inputs',)))
        return cls(block.embeddings.shape[1], fields)

    @property
    def state_width(self):
        return sum(math.prod(shape) for _, shape in self.fields)

    def state(self, block):
        if StateLayout.from_block(block) != self:
            raise ValueError('semantic VAE state layout differs from its trained model layout')
        return torch.cat([block.layers[int(name.split('.')[0])][name.split('.')[1]].reshape(len(block.token_ids), -1)
                          for name, _ in self.fields], dim=-1).float()

    def features(self, block):
        return torch.cat([block.embeddings.float(), self.state(block)], dim=-1)

    def replace_state(self, baseline, values):
        if values.shape != (len(baseline.token_ids), self.state_width):
            raise ValueError('decoded semantic state dimensions disagree with the target token layout')
        layers = [{'kind': layer['kind']} for layer in baseline.layers]
        start = 0
        for name, shape in self.fields:
            index, key = name.split('.')
            count = math.prod(shape)
            reference = baseline.layers[int(index)][key]
            layers[int(index)][key] = values[:, start:start+count].reshape(-1, *shape).to(reference)
            start += count
        return NativeStateBlock(text=baseline.text, token_ids=baseline.token_ids,
            tokenizer_fingerprint=baseline.tokenizer_fingerprint, checkpoint_sha256=baseline.checkpoint_sha256,
            embedding_weight_sha256=baseline.embedding_weight_sha256, embeddings=baseline.embeddings,
            positions=baseline.positions, layers=tuple(layers), architecture=baseline.architecture).validate()


def alignment_weights(source_offsets, target_offsets, *, device='cpu', allow_nearest=False):
    """Sparse overlap matching for tokenizers over the same text coordinates.

    Exact overlaps are normalized per target token. A nearest-token fallback is
    opt-in, recorded, and never represented as exact tokenizer equivalence.
    """
    if not source_offsets or not target_offsets:
        raise ValueError('semantic alignment requires source and target token spans')
    for offsets in (source_offsets, target_offsets):
        if any(len(pair) != 2 or any(type(x) is not int for x in pair) or pair[0] < 0 or pair[1] < pair[0]
               for pair in offsets) or any(a[0] > b[0] for a, b in zip(offsets, offsets[1:])):
            raise ValueError('semantic token offsets must be ordered nonnegative half-open intervals')
    rows, columns, weights, approximate = [], [], [], []
    first = 0
    for target, (begin, end) in enumerate(target_offsets):
        while first < len(source_offsets) and source_offsets[first][1] <= begin:
            first += 1
        matches = []
        for source in range(first, len(source_offsets)):
            lo, hi = source_offsets[source]
            if lo >= end:
                break
            overlap = max(0, min(end, hi) - max(begin, lo))
            if overlap:
                matches.append((source, overlap))
        if not matches:
            if not allow_nearest:
                raise ValueError(f'target token {target} has no source overlap; explicit alignment repair required')
            source = min(range(len(source_offsets)),
                key=lambda i: abs(sum(source_offsets[i]) - begin - end))
            matches = [(source, 1)]
            approximate.append(target)
        total = sum(weight for _, weight in matches)
        for source, weight in matches:
            rows.append(target); columns.append(source); weights.append(weight / total)
    matrix = torch.sparse_coo_tensor(torch.tensor([rows, columns], dtype=torch.int64, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
        (len(target_offsets), len(source_offsets)), check_invariants=True).coalesce()
    return matrix, {'source_tokens': len(source_offsets), 'target_tokens': len(target_offsets),
                    'nearest_fallback_target_tokens': approximate, 'best_effort': bool(approximate)}


class SemanticStateVAE(nn.Module):
    """Model adapters share the same semantic encoder/decoder and latent axes."""
    def __init__(self, layouts, *, latent_width=128, hidden_width=512):
        super().__init__()
        if len(layouts) < 2 or latent_width <= 0 or hidden_width <= 0:
            raise ValueError('shared semantic VAE requires at least two models and positive widths')
        self.layouts = dict(layouts)
        self.model_names = tuple(sorted(layouts))
        self.keys = {name: 'model_' + sha256_hex(name.encode()) for name in self.model_names}
        self.input_adapters = nn.ModuleDict({self.keys[name]: nn.Sequential(
            nn.LayerNorm(layout.embedding_width + layout.state_width),
            nn.Linear(layout.embedding_width + layout.state_width, hidden_width), nn.GELU())
            for name, layout in layouts.items()})
        self.posterior = nn.Sequential(nn.Linear(hidden_width, hidden_width), nn.GELU(),
                                       nn.Linear(hidden_width, latent_width * 2))
        self.shared_decoder = nn.Sequential(nn.Linear(latent_width, hidden_width), nn.GELU(),
                                            nn.Linear(hidden_width, hidden_width), nn.GELU())
        self.output_adapters = nn.ModuleDict({self.keys[name]: nn.Linear(hidden_width, layout.state_width)
                                             for name, layout in layouts.items()})
        self.latent_width, self.hidden_width = latent_width, hidden_width

    def add_model(self, name, layout):
        """Append a newly ready model without renaming existing optimizer slots."""
        if name in self.layouts:
            if self.layouts[name] != layout:
                raise ValueError('an existing semantic model identity cannot change state layout')
            return []
        key = 'model_' + sha256_hex(name.encode())
        reference = next(self.parameters())
        encoder = nn.Sequential(nn.LayerNorm(layout.embedding_width + layout.state_width),
            nn.Linear(layout.embedding_width + layout.state_width, self.hidden_width), nn.GELU()).to(reference)
        decoder = nn.Linear(self.hidden_width, layout.state_width).to(reference)
        self.layouts[name], self.keys[name] = layout, key
        self.model_names = tuple(sorted(self.layouts))
        self.input_adapters[key], self.output_adapters[key] = encoder, decoder
        return [(f'input_adapters.{key}.{n}', p) for n, p in encoder.named_parameters()] + [
               (f'output_adapters.{key}.{n}', p) for n, p in decoder.named_parameters()]

    def specification(self):
        return {'schema': 'natlang.semantic-state-vae/1', 'latent_width': self.latent_width,
                'hidden_width': self.hidden_width, 'models': {name: {
                    'embedding_width': layout.embedding_width,
                    'fields': [[key, list(shape)] for key, shape in layout.fields]}
                    for name, layout in sorted(self.layouts.items())}}

    @classmethod
    def from_specification(cls, spec):
        if spec.get('schema') != 'natlang.semantic-state-vae/1':
            raise ValueError('unsupported semantic VAE specification')
        layouts = {name: StateLayout(value['embedding_width'],
                    tuple((key, tuple(shape)) for key, shape in value['fields']))
                    for name, value in spec['models'].items()}
        return cls(layouts, latent_width=spec['latent_width'], hidden_width=spec['hidden_width'])

    def encode(self, model_name, block):
        features = self.layouts[model_name].features(block)
        mu, unconstrained = self.posterior(self.input_adapters[self.keys[model_name]](features)).chunk(2, dim=-1)
        # Stable positive standard deviation; KL is computed from this actual
        # posterior rather than a clipped, inconsistently reported variance.
        std = F.softplus(unconstrained) + torch.finfo(unconstrained.dtype).eps
        return mu, std

    @staticmethod
    def sample(mu, std, *, generator=None):
        return mu + std * torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)

    @staticmethod
    def kl(mu, std):
        return (mu.square() + std.square() - 1 - 2 * std.log()).mean() * 0.5

    def decode(self, model_name, code, baseline, *, alignment=None):
        if alignment is not None:
            code = torch.sparse.mm(alignment.to(device=code.device, dtype=code.dtype), code)
        if code.shape != (len(baseline.token_ids), self.latent_width):
            raise ValueError('semantic code needs explicit alignment to the target tokenizer')
        delta = self.output_adapters[self.keys[model_name]](self.shared_decoder(code))
        layout = self.layouts[model_name]
        return layout.replace_state(baseline, layout.state(baseline) + delta)

    def reconstruction(self, model_name, predicted, target):
        layout = self.layouts[model_name]
        actual, expected = layout.state(predicted), layout.state(target)
        if actual.shape != expected.shape:
            raise ValueError('semantic reconstruction tokenization differs from its paired target')
        terms, start = [], 0
        for _, shape in layout.fields:
            end = start + math.prod(shape)
            truth = expected[:, start:end]
            terms.append((actual[:, start:end] - truth).square().mean() /
                         truth.detach().square().mean().clamp_min(1e-6))
            start = end
        return torch.stack(terms).mean()

    def pair_loss(self, source_name, source, source_baseline, target_name, target, target_baseline,
                  *, source_to_target, target_to_source, kl_weight=0.001, alignment_weight=0.1,
                  cross_weight=1., generator=None):
        if source_name == target_name:
            raise ValueError('cross-model VAE pair requires distinct model identities')
        for reference, baseline in ((source, source_baseline), (target, target_baseline)):
            baseline.require_native(**{name: getattr(reference, name) for name in (
                'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')})
            if baseline.text != reference.text or not torch.equal(baseline.token_ids, reference.token_ids):
                raise ValueError('semantic baseline and paired target must cover identical text/token IDs')
        if source.text != target.text:
            raise ValueError('semantic cross-model pair must describe the same source text')
        if any(not math.isfinite(value) or value < 0 for value in (kl_weight, alignment_weight, cross_weight)):
            raise ValueError('semantic VAE loss weights must be finite and nonnegative')
        source_mu, source_std = self.encode(source_name, source)
        target_mu, target_std = self.encode(target_name, target)
        source_code, target_code = (self.sample(source_mu, source_std, generator=generator),
                                    self.sample(target_mu, target_std, generator=generator))
        source_self = self.decode(source_name, source_code, source_baseline)
        target_self = self.decode(target_name, target_code, target_baseline)
        target_cross = self.decode(target_name, source_code, target_baseline, alignment=source_to_target)
        source_cross = self.decode(source_name, target_code, source_baseline, alignment=target_to_source)
        self_loss = (self.reconstruction(source_name, source_self, source) +
                     self.reconstruction(target_name, target_self, target)) * 0.5
        cross_loss = (self.reconstruction(source_name, source_cross, source) +
                      self.reconstruction(target_name, target_cross, target)) * 0.5
        kl = (self.kl(source_mu, source_std) + self.kl(target_mu, target_std)) * 0.5
        aligned_mu = torch.sparse.mm(source_to_target.to(source_mu), source_mu)
        alignment = F.mse_loss(aligned_mu, target_mu)
        loss = self_loss + cross_weight * cross_loss + kl_weight * kl + alignment_weight * alignment
        return loss, {'self_reconstruction': self_loss, 'cross_reconstruction': cross_loss,
                      'kl': kl, 'semantic_alignment': alignment,
                      'target_block': target_cross, 'source_block': source_cross}
