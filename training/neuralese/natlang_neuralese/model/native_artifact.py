"""Typed named tensors for native parameter sets, gradients and updates.

These artifacts preserve tensor names and provenance. They are intentionally
not compatible with the old flattened ``nz1`` value/vector domain.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
from dataclasses import dataclass

import torch

SCHEMA = 'natlang.neuralese-native-tensor-artifact/1'
DOMAIN = 'natlang.neuralese-native-tensor-artifact/1'
_DTYPES = {torch.float16: ('F16', 2), torch.bfloat16: ('BF16', 2),
           torch.float32: ('F32', 4), torch.float64: ('F64', 8)}
_FROM_SAFE = {name: (dtype, size) for dtype, (name, size) in _DTYPES.items()}
_KINDS = {'parameter-set', 'parameter-gradient', 'parameter-update', 'state-gradient', 'optimizer-state', 'adapter-basis', 'adapter-coefficients', 'adapter-gradient', 'derivative-program'}


def _raw(tensor):
    value = tensor.detach().contiguous().cpu()
    return (value.view(torch.int16).numpy().tobytes() if value.dtype == torch.bfloat16
            else value.numpy().tobytes())


@dataclass
class NativeTensorArtifact:
    kind: str
    native_identity: dict
    tensors: dict[str, torch.Tensor]
    source_artifact_id: str | None = None
    provenance: dict | None = None
    # Wire metadata bytes are identity-bearing. Preserve numeric spellings
    # across Python/JS (1.0/1, 1e-09/1e-9) without changing existing IDs.
    canonical_metadata: str | None = None

    def validate(self):
        if self.kind not in _KINDS:
            raise ValueError(f'unsupported native tensor artifact kind {self.kind!r}')
        identity = self.native_identity
        if not isinstance(identity, dict) or set(identity) != {
                'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'}:
            raise ValueError('native artifact needs the exact four-field model identity')
        for name in ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256'):
            if not isinstance(identity.get(name), str) or not re.fullmatch(r'[0-9a-f]{64}', identity[name]):
                raise ValueError(f'native artifact {name} must be a SHA-256 digest')
        if not isinstance(identity.get('architecture'), str) or not identity['architecture']:
            raise ValueError('native artifact architecture is required')
        if self.source_artifact_id is not None and not re.fullmatch(r'nzs1_[a-z2-7]+|nnt1_[a-z2-7]+', self.source_artifact_id):
            raise ValueError('native artifact source must identify a native state or tensor artifact')
        if not isinstance(self.tensors, dict) or (not self.tensors and self.kind!='derivative-program'):
            raise ValueError('native tensor artifact needs named tensors')
        if self.kind=='derivative-program':
            from .native_derivative_contract import validate_program_artifact
            validate_program_artifact(self.provenance)
            if self.tensors:raise ValueError('native derivative program stores exact metadata, never fake parameter tensors')
        for name, tensor in self.tensors.items():
            if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
                raise ValueError('native tensors require nonempty names and tensor values')
            if tensor.dtype not in _DTYPES or tensor.ndim < 1 or any(size <= 0 for size in tensor.shape):
                raise ValueError(f'native tensor {name!r} must have a supported floating dtype and positive dimensions')
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'native tensor {name!r} contains non-finite values')
        if self.provenance is not None and not isinstance(self.provenance, dict):
            raise ValueError('native artifact provenance must be an object')
        binding = (self.provenance or {}).get('adapter_binding') if self.kind == 'adapter-gradient' else None
        if binding is not None:
            from types import SimpleNamespace
            from .native_adapters import validate_state_binding
            validate_state_binding(binding, SimpleNamespace(**self.provenance.get('effective_native_identity', {})))
            if binding['base_native_identity'] != self.native_identity:
                raise ValueError('adapter gradient actor binding changed its exact base identity')
        from .native_derivative_contract import validate_derivative_target
        derivative_target=validate_derivative_target(self)
        from .native_derivative_contract import validate_functional_actor
        validate_functional_actor(self)
        from .native_actor_graph import validate_invocation_behavior_provenance
        validate_invocation_behavior_provenance(self.provenance,self.native_identity)
        actors=(self.provenance or {}).get('invocation_actors')
        if actors is not None:
            from .native_actor_graph import validate_invocation_actor_provenance
            validate_invocation_actor_provenance(self.provenance,self.native_identity)
            if self.kind=='adapter-gradient' and derivative_target is None and self.source_artifact_id not in {
                    entry['id'] for actor in actors['invocations'].values()
                    for entry in (actor['adapter_binding'] or {}).get('adapters',[])}:
                raise ValueError('native invocation gradient target is not one of its exact actors')
        return self

    def metadata(self):
        self.validate()
        tensors = [{'name': name, 'dtype': _DTYPES[value.dtype][0], 'shape': list(value.shape)}
                   for name, value in sorted(self.tensors.items())]
        return {'schema': SCHEMA, 'kind': self.kind, 'native_identity': self.native_identity,
                'source_artifact_id': self.source_artifact_id, 'provenance': self.provenance or {},
                'tensors': tensors}


def _metadata_text(artifact):
    metadata = artifact.metadata()
    if artifact.canonical_metadata is not None:
        if json.loads(artifact.canonical_metadata) != metadata:
            raise ValueError('native artifact preserved metadata bytes differ from its actual fields')
        return artifact.canonical_metadata
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def native_tensor_artifact_id(artifact):
    metadata = _metadata_text(artifact).encode()
    digest = hashlib.sha256()
    digest.update(DOMAIN.encode() + b'\0' + struct.pack('<Q', len(metadata)) + metadata)
    for name, tensor in sorted(artifact.tensors.items()):
        encoded_name = name.encode()
        raw = _raw(tensor)
        descriptor = json.dumps({'dtype': _DTYPES[tensor.dtype][0], 'shape': list(tensor.shape)},
                                separators=(',', ':')).encode()
        digest.update(struct.pack('<I', len(encoded_name)) + encoded_name)
        digest.update(struct.pack('<I', len(descriptor)) + descriptor)
        digest.update(struct.pack('<Q', len(raw)) + raw)
    return 'nnt1_' + base64.b32encode(digest.digest()).decode().rstrip('=').lower()


def encode_native_tensor_artifact(artifact):
    artifact.validate()
    entries, payloads, offset = {}, [], 0
    for name, tensor in sorted(artifact.tensors.items()):
        raw = _raw(tensor)
        entries[name] = {'dtype': _DTYPES[tensor.dtype][0], 'shape': list(tensor.shape),
                         'data_offsets': [offset, offset + len(raw)]}
        payloads.append(raw)
        offset += len(raw)
    artifact_id = native_tensor_artifact_id(artifact)
    # id sorts first in the existing metadata schema; insert it without
    # reserializing any identity-bearing numeric token.
    entries['__metadata__'] = {'natlang.native_tensor_artifact':
        '{"id":' + json.dumps(artifact_id) + ',' + _metadata_text(artifact)[1:]}
    header = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    header += b' ' * ((8 - len(header) % 8) % 8)
    return struct.pack('<Q', len(header)) + header + b''.join(payloads)


def decode_native_tensor_artifact(body):
    try:
        if len(body) < 8:
            raise ValueError('not a safetensors body')
        (header_size,) = struct.unpack('<Q', body[:8])
        if header_size > len(body) - 8:
            raise ValueError('header exceeds body')
        header = json.loads(body[8:8 + header_size])
        if not isinstance(header, dict):
            raise ValueError('native tensor header must be an object')
        metadata_text = header.get('__metadata__', {}).get('natlang.native_tensor_artifact', '{}')
        metadata = json.loads(metadata_text)
        if not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA:
            raise ValueError('unsupported native tensor artifact schema')
        tensors, spans = {}, []
        payload = body[8 + header_size:]
        for name, entry in header.items():
            if name == '__metadata__':
                continue
            dtype, itemsize = _FROM_SAFE[entry['dtype']]
            shape, offsets = entry['shape'], entry['data_offsets']
            if (not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2 or any(type(i) is not int for i in offsets)):
                raise ValueError(f'invalid tensor descriptor {name!r}')
            count = 1
            for size in shape:
                count *= size
            start, end = offsets
            if start < 0 or end < start or end > len(payload) or end - start != count * itemsize:
                raise ValueError(f'invalid tensor byte range {name!r}')
            spans.append((start, end, name))
            raw = payload[start:end]
            tensor = (torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)
                      if dtype == torch.bfloat16 else torch.frombuffer(bytearray(raw), dtype=dtype))
            tensors[name] = tensor.reshape(shape).clone()
        spans.sort()
        if (not spans and (metadata.get('kind')!='derivative-program' or len(payload)!=0)
                or spans and (spans[0][0] != 0 or spans[-1][1] != len(payload) or any(a[1] != b[0] for a, b in zip(spans, spans[1:])))):
            raise ValueError('native tensor payload has gaps or overlaps')
        expected = {row['name']: (row['dtype'], row['shape']) for row in metadata.get('tensors', [])}
        actual = {name: (_DTYPES[value.dtype][0], list(value.shape)) for name, value in tensors.items()}
        if expected != actual:
            raise ValueError('native tensor descriptors differ from metadata inventory')
        prefix = '{"id":' + json.dumps(metadata.get('id')) + ','
        canonical_metadata = '{' + metadata_text[len(prefix):] if metadata_text.startswith(prefix) else None
        artifact = NativeTensorArtifact(metadata['kind'], metadata['native_identity'], tensors,
            metadata.get('source_artifact_id'), metadata.get('provenance'), canonical_metadata)
        artifact.validate()
        actual_id = native_tensor_artifact_id(artifact)
        if metadata.get('id') != actual_id:
            raise ValueError(f'native tensor artifact ID does not match content ({actual_id})')
        return artifact, actual_id
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError(f'malformed native tensor artifact: {error}') from error


def native_tensor_dependencies(artifact):
    """Required immutable inputs for interpreting/updating a named artifact."""
    dependencies = []
    if artifact.kind=='derivative-program':
        from .native_derivative_contract import validate_program_artifact
        validate_program_artifact(artifact.provenance)
        dependencies.extend(artifact.provenance['dependencies'])
    execution=(artifact.provenance or {}).get('derivative_execution_id')
    if execution is not None:
        from .native_derivative_contract import native_id
        if not native_id(execution,'nnt1_'):raise ValueError('native derivative gradient needs its exact execution artifact')
        dependencies.append(execution)
    program=(artifact.provenance or {}).get('derivative_program_id')
    if program is not None:
        from .native_derivative_contract import native_id
        if not native_id(program,'nnt1_'):raise ValueError('native derivative gradient needs its exact program artifact')
        dependencies.append(program)
    if artifact.kind=='state-gradient' and artifact.source_artifact_id:
        dependencies.append(artifact.source_artifact_id)
    derivation=(artifact.provenance or {})
    if derivation.get('scope') in ('natlang.native-state-derivation/1','natlang.native-derivative-program/1') or derivation.get('schema')=='natlang.native-state-derivation/1':
        for key in ('gradient_id','parent_state_id','optimizer_state_parent_id','parent_value_id'):
            if derivation.get(key):dependencies.append(derivation[key])
        if artifact.source_artifact_id:dependencies.append(artifact.source_artifact_id)
    if (artifact.kind in ('adapter-coefficients', 'adapter-gradient') or
            artifact.kind == 'optimizer-state' and
            (artifact.provenance or {}).get('scope') == 'natlang.native-adapter/1'):
        if artifact.source_artifact_id:dependencies.append(artifact.source_artifact_id)
    actor=(artifact.provenance or {}).get('functional_actor')
    if actor is not None:
        dependencies.extend(entry['id'] for entry in actor['parameter_values'])
        for entries in [actor['uniform_adapters'],*(actor['invocation_adapters'] or {}).values()]:dependencies.extend(entry['id'] for entry in entries)
    for value in ((artifact.provenance or {}).get('evaluation_inputs') or {}).values():
        dependencies.append(value['value_id'])
    if program is not None:
        for key in ('parent_value_id','parameter_value_id','gradient_id','optimizer_state_parent_id'):
            if (artifact.provenance or {}).get(key):dependencies.append(artifact.provenance[key])
        if artifact.source_artifact_id:dependencies.append(artifact.source_artifact_id)
    binding = (artifact.provenance or {}).get('adapter_binding') if artifact.kind == 'adapter-gradient' else None
    if binding:
        dependencies.extend(entry['id'] for entry in binding['adapters'])
    actors=(artifact.provenance or {}).get('invocation_actors')
    if actors is not None:
        from .native_actor_graph import validate_invocation_actor_provenance
        validate_invocation_actor_provenance(artifact.provenance,artifact.native_identity)
        for actor in actors['invocations'].values():
            dependencies.extend(entry['id'] for entry in (actor['adapter_binding'] or {}).get('adapters',[]))
    from .native_actor_graph import validate_invocation_behavior_provenance
    behavior=validate_invocation_behavior_provenance(artifact.provenance,artifact.native_identity)
    if behavior is not None:
        for receipt in behavior['episodes'].values():
            for actor in receipt['invocations'].values():
                dependencies.extend(entry['id'] for entry in (actor['adapter_binding'] or {}).get('adapters',[]))
    for contract in ((artifact.provenance or {}).get('external_adapter_state_contracts') or {}).values():
        for key in ('source_adapter_binding', 'target_adapter_binding'):
            binding = contract.get(key)
            if binding:
                dependencies.extend(entry['id'] for entry in binding['adapters'])
    return list(dict.fromkeys(dependencies))
