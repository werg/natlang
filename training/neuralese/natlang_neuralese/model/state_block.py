"""Exact shared-state block contract; no projected communication latent.

Tokens are the first dimension of every stored state tensor. Attention keys are
pre-RoPE, so the consuming model can place them at its own logical positions.
The model graph executor owns state capture/splicing; this module owns exact
shape, provenance and compatibility checks used by storage and serving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import torch

from .native_gated_delta import PAYLOAD_FIELDS

SCHEMA = 'neuralese.native-state/1'
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def native_state_dependencies(block):
    """One immutable input inventory for storage, transport and RL archives."""
    from .native_derivative_contract import derivation_dependencies
    binding=block.adapter_binding
    dependencies=derivation_dependencies(block.derivation)
    if binding:dependencies.extend(entry['id'] for entry in binding['adapters'])
    return sorted(set(dependencies))


def layer_fields(kind) -> tuple[str, ...]:
    """The state tensors one layer kind carries (`layers.<index>.<field>`)."""
    if kind == 'attention':
        return ('raw_k', 'v')
    if kind == 'finite_convolution':
        return ('gated_inputs',)
    if kind == 'gated_delta':
        return tuple(PAYLOAD_FIELDS)
    raise ValueError(f'unsupported native shared layer state: {kind!r}')


# How a state's tensors came about, when they are the plain forward of its own tokens (`producer_kind`, optional, part of
# the content id): 'text' = `native.text-write` (the text forced as a produce span from position 0, nothing before it);
# 'span' = a produce span in its producer's call (`native.capture`, the grammar capture, a produce span sampled in
# generation). Every engine reads such a state, when it is exact and carries no adapter binding or derivation, as its
# token ids inline in the reader's own context (`in-context-text`): a 1.7B reader could not read the KV of a context it
# never saw (DECISIONS 2026-11-23 "Neuralese read-back"; ENGINE_PARITY). Other states (contextual bodies, optimized,
# derived and soft states) carry no kind and splice as stored. States written before the field existed keep their ids.
PRODUCER_KINDS = ('text', 'span')


def native_state_metadata(*, text, architecture, tokenizer_fingerprint, checkpoint_sha256, embedding_weight_sha256,
                          layer_kinds, adapter_binding=None, derivation=None, producer_kind=None) -> dict[str, Any]:
    """The canonical metadata of a native state (hashed into its id, stored in its file header)."""
    return {'schema': SCHEMA, 'text': text, 'architecture': architecture,
            'tokenizer_fingerprint': tokenizer_fingerprint, 'checkpoint_sha256': checkpoint_sha256,
            'embedding_weight_sha256': embedding_weight_sha256, 'layer_kinds': list(layer_kinds),
            **({'adapter_binding': adapter_binding} if adapter_binding is not None else {}),
            **({'derivation': derivation} if derivation is not None else {}),
            **({'producer_kind': producer_kind} if producer_kind is not None else {})}


def validate_native_layout(metadata, descriptors) -> int:
    """Check a native state's metadata and tensor inventory, dtypes and shapes without reading a tensor value: the one
    layout check of in-memory blocks (`NativeStateBlock.validate`) and native-state files (`serve.store`). Returns the
    token count. `descriptors`: name -> (torch dtype, shape)."""
    from .native_derivative_contract import validate_derivation
    if not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA:
        raise ValueError('unsupported native state schema')
    validate_derivation(metadata.get('derivation'))
    if 'producer_kind' in metadata and metadata['producer_kind'] not in PRODUCER_KINDS:
        raise ValueError(f"native state producer_kind must be one of {PRODUCER_KINDS}")
    if 'producer_kind' in metadata and metadata.get('derivation') is not None:
        raise ValueError('a derived native state has no producer kind: its tensors are not its tokens\' forward')
    if not isinstance(metadata.get('text'), str) or not isinstance(metadata.get('architecture'), str) or not metadata['architecture']:
        raise ValueError('native state needs text and an architecture identity')
    for name in ('tokenizer_fingerprint', 'checkpoint_sha256', 'embedding_weight_sha256'):
        if not isinstance(metadata.get(name), str) or not re.fullmatch('[0-9a-f]{64}', metadata[name]):
            raise ValueError(f'native state {name} must be a SHA-256 digest')
    kinds = metadata.get('layer_kinds')
    if not isinstance(kinds, list) or not kinds:
        raise ValueError('native state must contain every backbone layer state')
    expected = {'token_ids', 'positions', 'embeddings'}
    for index, kind in enumerate(kinds):
        expected.update(f'layers.{index}.{field}' for field in layer_fields(kind))
    if set(descriptors) != expected:
        raise ValueError('native state tensor inventory differs from its layer layout')
    dtype, shape = descriptors['token_ids']
    if dtype != torch.int64 or len(shape) != 1:
        raise ValueError('native state token_ids must be int64 [tokens]')
    count = shape[0]
    if descriptors['positions'] != (torch.int64, (count,)):
        raise ValueError('native state positions must be nonnegative int64 [tokens]')

    def floating(name, dimensions):
        dtype, shape = descriptors[name]
        if (len(shape) != dimensions or shape[0] != count or any(size <= 0 for size in shape[1:])
                or dtype not in FLOAT_DTYPES):
            raise ValueError(f'native state {name} must have token-first floating dimensions')

    floating('embeddings', 2)
    for index, kind in enumerate(kinds):
        for field in layer_fields(kind):
            floating(f'layers.{index}.{field}', 3 if kind == 'attention' else 2)
        if kind == 'attention' and descriptors[f'layers.{index}.raw_k'] != descriptors[f'layers.{index}.v']:
            raise ValueError('attention raw keys and values must have identical shape and dtype')
        if kind == 'gated_delta' and descriptors[f'layers.{index}.b'][1] != descriptors[f'layers.{index}.a'][1]:
            raise ValueError('gated delta beta and decay projections must have identical shapes')
    return count


def check_token_values(token_ids, positions) -> None:
    """Value checks of the two small index tensors: nonnegative ids, one contiguous span of nonnegative positions."""
    if bool((token_ids < 0).any()):
        raise ValueError('native state token IDs must be nonnegative')
    if bool((positions < 0).any()):
        raise ValueError('native state positions must be nonnegative int64 [tokens]')
    if len(positions) > 1 and not bool((positions[1:] == positions[:-1] + 1).all()):
        raise ValueError('native state positions must form a contiguous span')


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
    adapter_binding: dict | None = field(default=None, kw_only=True)
    derivation: dict | None = field(default=None,kw_only=True)
    producer_kind: str | None = field(default=None, kw_only=True)

    def descriptors(self) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
        """name -> (dtype, shape) of every tensor, as a native-state file header declares them."""
        for name, value in self.tensors().items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f'native state {name} must be a tensor')
        return {name: (value.dtype, tuple(value.shape)) for name, value in self.tensors().items()}

    def validate(self) -> NativeStateBlock:
        """The layout (`validate_native_layout`, shared with file headers) and the values: nonnegative token ids, one
        contiguous span of nonnegative positions, finite floating state."""
        if not isinstance(self.layers, tuple) or not all(isinstance(layer, dict) for layer in self.layers):
            raise ValueError('native state layers must be objects')
        validate_native_layout(self.metadata(), self.descriptors())
        if self.adapter_binding is not None:
            from .native_adapters import validate_state_binding
            validate_state_binding(self.adapter_binding, self)
        check_token_values(self.token_ids, self.positions)
        for name, value in self.tensors().items():
            if value.dtype in FLOAT_DTYPES and not bool(torch.isfinite(value).all()):
                raise ValueError(f'native state {name} contains non-finite values')
        return self

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
        return native_state_metadata(text=self.text, architecture=self.architecture,
                                     tokenizer_fingerprint=self.tokenizer_fingerprint,
                                     checkpoint_sha256=self.checkpoint_sha256,
                                     embedding_weight_sha256=self.embedding_weight_sha256,
                                     layer_kinds=[layer.get('kind') for layer in self.layers],
                                     adapter_binding=self.adapter_binding, derivation=self.derivation,
                                     producer_kind=self.producer_kind)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {'token_ids': self.token_ids, 'positions': self.positions, 'embeddings': self.embeddings,
                **{f'layers.{index}.{name}': value for index, layer in enumerate(self.layers)
                   for name, value in layer.items() if name != 'kind'}}

    @classmethod
    def from_tensors(cls, metadata, tensors):
        if metadata.get('schema') != SCHEMA:
            raise ValueError('unsupported native state schema')
        kinds = metadata.get('layer_kinds')
        if not isinstance(kinds, list) or not kinds:
            raise ValueError('native state needs a layer layout')
        layers = []
        expected = {'token_ids', 'positions', 'embeddings'}
        for index, kind in enumerate(kinds):
            names = layer_fields(kind)
            keys = {name: f'layers.{index}.{name}' for name in names}
            expected.update(keys.values())
            layers.append({'kind': kind, **{name: tensors[key] for name, key in keys.items()}})
        if set(tensors) != expected:
            raise ValueError('native state tensor inventory differs from its layer layout')
        block = cls(**{name: metadata[name] for name in ('text', 'architecture', 'tokenizer_fingerprint',
                        'checkpoint_sha256', 'embedding_weight_sha256')},
                    **{name: tensors[name] for name in ('token_ids', 'positions', 'embeddings')},
                    layers=tuple(layers), adapter_binding=metadata.get("adapter_binding"),derivation=metadata.get("derivation"),
                    producer_kind=metadata.get("producer_kind"))
        block.validate()
        return block
