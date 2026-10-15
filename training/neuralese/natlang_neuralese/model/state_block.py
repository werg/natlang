"""Exact shared-state block contract; no projected communication latent.

Tokens are the first dimension of every stored state tensor. Attention keys are
pre-RoPE, so the consuming model can place them at its own logical positions.
The model graph executor owns state capture/splicing; this module owns exact
shape, provenance and compatibility checks used by storage and serving.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import torch

SCHEMA = 'neuralese.native-state/1'
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@dataclass
class NativeStateBlock:
    text: str
    token_ids: torch.Tensor
    tokenizer_fingerprint: str
    checkpoint_sha256: str
    embedding_weight_sha256: str
    embeddings: torch.Tensor
    positions: torch.Tensor
    layers: tuple[dict[str, Any], ...]
    architecture: str

    def _validate_token_fields(self):
        if not isinstance(self.text, str) or not isinstance(self.architecture, str) or not self.architecture:
            raise ValueError('native state needs text and an architecture identity')
        for name in ('tokenizer_fingerprint', 'checkpoint_sha256', 'embedding_weight_sha256'):
            if not isinstance(getattr(self, name), str) or not re.fullmatch('[0-9a-f]{64}', getattr(self, name)):
                raise ValueError(f'native state {name} must be a SHA-256 digest')
        if (not isinstance(self.token_ids, torch.Tensor) or self.token_ids.dtype != torch.int64
                or self.token_ids.ndim != 1):
            raise ValueError('native state token_ids must be int64 [tokens]')
        count = len(self.token_ids)
        if bool((self.token_ids < 0).any()):
            raise ValueError('native state token IDs must be nonnegative')
        if (not isinstance(self.positions, torch.Tensor) or self.positions.dtype != torch.int64
                or self.positions.shape != (count,)
                or bool((self.positions < 0).any())):
            raise ValueError('native state positions must be nonnegative int64 [tokens]')
        if count > 1 and not bool((self.positions[1:] == self.positions[:-1] + 1).all()):
            raise ValueError('native state positions must form a contiguous span')
        self._float_tensor('embeddings', self.embeddings, count, 2)
        return count

    def validate(self) -> NativeStateBlock:
        count = self._validate_token_fields()
        if not self.layers:
            raise ValueError('native state must contain every backbone layer state')
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, dict):
                raise ValueError(f'native state layer {index} must be an object')
            kind = layer.get('kind')
            if kind == 'attention':
                if set(layer) != {'kind', 'raw_k', 'v'}:
                    raise ValueError('attention state contains exactly kind, raw_k and v')
                for name in ('raw_k', 'v'):
                    self._float_tensor(f'layer {index} {name}', layer[name], count, 3)
                if layer['raw_k'].shape != layer['v'].shape or layer['raw_k'].dtype != layer['v'].dtype:
                    raise ValueError('attention raw keys and values must have identical shape and dtype')
            elif kind == 'finite_convolution':
                if set(layer) != {'kind', 'gated_inputs'}:
                    raise ValueError('finite convolution contains exactly kind and gated_inputs')
                self._float_tensor(f'layer {index} gated_inputs', layer['gated_inputs'], count, 2)
            else:
                raise ValueError(f'unsupported native shared layer state: {kind!r}')
        return self

    @staticmethod
    def _float_tensor(name, value, count, dimensions):
        if (not isinstance(value, torch.Tensor) or value.ndim != dimensions or value.shape[0] != count
                or any(size <= 0 for size in value.shape[1:]) or value.dtype not in FLOAT_DTYPES):
            raise ValueError(f'native state {name} must have token-first floating dimensions')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'native state {name} contains non-finite values')

    def require_native(self, *, checkpoint_sha256, tokenizer_fingerprint, embedding_weight_sha256,
                       architecture, layer_kinds=None) -> None:
        """A foreign block requires explicit adaptation/reencoding, never raw KV reuse."""
        self.validate()
        for name, expected in (('checkpoint_sha256', checkpoint_sha256),
                               ('tokenizer_fingerprint', tokenizer_fingerprint),
                               ('embedding_weight_sha256', embedding_weight_sha256),
                               ('architecture', architecture)):
            if getattr(self, name) != expected:
                raise ValueError(f'foreign native state {name}: adapt or reencode its text explicitly')
        if layer_kinds is not None and tuple(layer['kind'] for layer in self.layers) != tuple(layer_kinds):
            raise ValueError('native state layer layout differs from the consuming backbone')

    def metadata(self) -> dict[str, Any]:
        return {'schema': SCHEMA, 'text': self.text, 'architecture': self.architecture,
                'tokenizer_fingerprint': self.tokenizer_fingerprint,
                'checkpoint_sha256': self.checkpoint_sha256,
                'embedding_weight_sha256': self.embedding_weight_sha256,
                'layer_kinds': [layer['kind'] for layer in self.layers]}

    def tensors(self) -> dict[str, torch.Tensor]:
        return {'token_ids': self.token_ids, 'positions': self.positions, 'embeddings': self.embeddings,
                **{f'layers.{index}.{name}': value for index, layer in enumerate(self.layers)
                   for name, value in layer.items() if name != 'kind'}}

    @classmethod
    def from_tensors(cls, metadata, tensors):
        if metadata.get('schema') == 'neuralese.semantic-state/1':
            from .semantic_message import SemanticStateMessage
            return SemanticStateMessage.from_tensors(metadata, tensors)
        if metadata.get('schema') != SCHEMA:
            raise ValueError('unsupported native state schema')
        kinds = metadata.get('layer_kinds')
        if not isinstance(kinds, list) or not kinds:
            raise ValueError('native state needs a layer layout')
        layers = []
        expected = {'token_ids', 'positions', 'embeddings'}
        for index, kind in enumerate(kinds):
            names = ('raw_k', 'v') if kind == 'attention' else ('gated_inputs',) if kind == 'finite_convolution' else ()
            if not names:
                raise ValueError(f'unsupported native shared layer state: {kind!r}')
            keys = {name: f'layers.{index}.{name}' for name in names}
            expected.update(keys.values())
            layers.append({'kind': kind, **{name: tensors[key] for name, key in keys.items()}})
        if set(tensors) != expected:
            raise ValueError('native state tensor inventory differs from its layer layout')
        block = cls(**{name: metadata[name] for name in ('text', 'architecture', 'tokenizer_fingerprint',
                        'checkpoint_sha256', 'embedding_weight_sha256')},
                    **{name: tensors[name] for name in ('token_ids', 'positions', 'embeddings')},
                    layers=tuple(layers))
        block.validate()
        return block
