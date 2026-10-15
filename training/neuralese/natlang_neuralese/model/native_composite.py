"""Restorable native checkpoints with immutable named-tensor updates.

The manifest hash is the effective checkpoint identity. Each manifest points
to an immutable full native parent and a content-addressed named interface
patch, so small updates do not rewrite a multi-gigabyte backbone checkpoint.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import canonical_json_bytes
from .native_artifact import NativeTensorArtifact, decode_native_tensor_artifact, encode_native_tensor_artifact

SCHEMA = 'natlang.neuralese-native-composite-checkpoint/1'


def _durable_immutable(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'immutable native artifact path already contains different bytes: {path}')
        return
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.pending', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f'immutable native artifact path collision: {path}')
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def _channel_tensors(model):
    from .shared_state import channel_checkpoint
    channel = channel_checkpoint(model)
    tensors = {'input_rows': channel.pop('input_rows')}
    output = channel.pop('output_rows')
    if output is not None:
        tensors['output_rows'] = output
    tensors.update({'adapter.' + name: value for name, value in channel.pop('adapter').items()})
    return channel, tensors, output is not None


def write_native_composite_checkpoint(base_checkpoint, output_dir, model, model_config, *,
                                      base_identity, source_gradient_id=None, optimizer=None,
                                      optimizer_tensors=None, update_provenance=None, participant=None):
    """Write a durable immutable composite checkpoint over `base_checkpoint`.

    Only the actual shared embedding adapter and four role rows are patched;
    no projected-value/head tensors are accepted. Future full-backbone updates
    require a declared named trainable set and its measured storage policy.
    """
    base = Path(base_checkpoint).resolve(strict=True)
    base_sha = sha256_file_hex_cached(base)
    if base_identity.get('checkpoint_sha256') != base_sha:
        raise ValueError('native composite base identity does not match the parent checkpoint bytes')
    binding = getattr(model, '_native_backbone_binding', None)
    current_parameters = tuple((name, id(parameter), parameter._version)
                               for name, parameter in model.port.hf.named_parameters())
    if (not isinstance(binding, dict) or binding.get('checkpoint_sha256') != base_sha
            or binding.get('parameters') != current_parameters):
        raise ValueError('interface-only composite update requires the unchanged backbone loaded from this exact parent checkpoint')
    if base_identity.get('architecture') != model_config.get('architecture'):
        raise ValueError('native composite base architecture differs from the live model')
    channel, tensors, has_output_rows = _channel_tensors(model)
    expected_names = {'input_rows', 'adapter.down.weight', 'adapter.up.weight'}
    if has_output_rows:
        expected_names.add('output_rows')
    if set(tensors) != expected_names:
        raise ValueError('native composite parameter inventory differs from the model channel contract')
    artifact = NativeTensorArtifact('parameter-update', dict(base_identity), tensors,
        source_gradient_id, {'base_checkpoint_sha256': base_sha,
                             'operation': 'replace-native-interface-parameters',
                             'optimizer': optimizer or {}, **(update_provenance or {})}).validate()
    artifact_bytes = encode_native_tensor_artifact(artifact)
    from .native_artifact import native_tensor_artifact_id
    artifact_id = native_tensor_artifact_id(artifact)
    directory = Path(output_dir).resolve()
    tensor_path = directory / f'{artifact_id}.safetensors'
    _durable_immutable(tensor_path, artifact_bytes)
    optimizer_binding = None
    if optimizer_tensors:
        optimizer_artifact = NativeTensorArtifact('optimizer-state', dict(base_identity),
            {name: value for name, value in optimizer_tensors.items()}, artifact_id,
            {'for_parameter_artifact_id': artifact_id}).validate()
        from .native_artifact import native_tensor_artifact_id
        optimizer_id = native_tensor_artifact_id(optimizer_artifact)
        optimizer_path = directory / f'{optimizer_id}.safetensors'
        _durable_immutable(optimizer_path, encode_native_tensor_artifact(optimizer_artifact))
        optimizer_binding = {'id': optimizer_id, 'path': optimizer_path.name,
                             'sha256': sha256_file_hex_cached(optimizer_path)}
    from .native_model import native_identity
    output_embed_sha = native_identity(model, model_config, checkpoint_sha256=base_sha)['embedding_weight_sha256']
    manifest = {'schema': SCHEMA,
        'base_checkpoint': {'path': str(base), 'sha256': base_sha, 'participant': participant},
        'parameter_artifact': {'path': tensor_path.name, 'id': artifact_id,
                               'sha256': sha256_file_hex_cached(tensor_path)},
        'optimizer_state_artifact': optimizer_binding,
        'model_config': {key: model_config[key] for key in ('architecture', 'rank', 'marker_ids',
            'tokenizer_fingerprint', 'base_inventory_sha256', 'precision', 'dtype')},
        'output_embedding_weight_sha256': output_embed_sha,
        'optimizer': optimizer or {}, 'update_provenance': update_provenance or {}}
    payload = canonical_json_bytes(manifest) + b'\n'
    import hashlib
    manifest_sha = hashlib.sha256(payload).hexdigest()
    manifest_path = directory / f'native-composite-{manifest_sha}.json'
    _durable_immutable(manifest_path, payload)
    return {'path': str(manifest_path), 'sha256': manifest_sha, 'schema': SCHEMA,
            'native_identity': {'checkpoint_sha256': manifest_sha,
                'tokenizer_fingerprint': base_identity['tokenizer_fingerprint'],
                'embedding_weight_sha256': output_embed_sha,
                'architecture': base_identity['architecture']},
            'parameter_artifact_id': artifact_id}


def load_native_composite_checkpoint(path, *, base=None, device='cpu', dtype=None):
    """Restore the verified base, then apply its exact named native interface patch."""
    manifest_path = Path(path).resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError('native composite checkpoint manifest is malformed') from error
    if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported native composite checkpoint schema')
    parent = manifest['base_checkpoint']
    # `base` is the HF source directory override understood by the shared
    # loader. The immutable parent checkpoint remains the manifest's file.
    base_path = Path(parent['path']).resolve(strict=True)
    if sha256_file_hex_cached(base_path) != parent['sha256']:
        raise ValueError('native composite parent checkpoint bytes changed')
    from .native_model import load_native_checkpoint, native_identity
    kwargs = {'base': base, 'device': device, 'dtype': dtype}
    if parent.get('participant') is not None:
        kwargs['participant'] = parent['participant']
    model, tokenizer, roles, config, parent_state = load_native_checkpoint(base_path, **kwargs)
    parent_identity = native_identity(model, config, checkpoint_sha256=parent['sha256'])
    artifact_binding = manifest['parameter_artifact']
    artifact_path = (manifest_path.parent / artifact_binding['path']).resolve(strict=True)
    if artifact_path.parent != manifest_path.parent:
        raise ValueError('native composite tensor patch must stay beside its immutable manifest')
    if sha256_file_hex_cached(artifact_path) != artifact_binding['sha256']:
        raise ValueError('native composite tensor patch bytes changed')
    artifact, artifact_id = decode_native_tensor_artifact(artifact_path.read_bytes())
    if artifact_id != artifact_binding['id'] or artifact.kind != 'parameter-update':
        raise ValueError('native composite patch artifact binding or kind is invalid')
    if artifact.native_identity != parent_identity:
        raise ValueError('native composite patch belongs to another exact parent model')
    from .shared_state import channel_checkpoint, restore_channel_checkpoint
    channel = channel_checkpoint(model)
    required = {'input_rows', 'adapter.down.weight', 'adapter.up.weight'}
    if channel['output_rows'] is not None:
        required.add('output_rows')
    if set(artifact.tensors) != required:
        raise ValueError('native composite patch tensor names differ from the native interface')
    channel['input_rows'] = artifact.tensors['input_rows']
    channel['output_rows'] = artifact.tensors.get('output_rows')
    channel['adapter'] = {name.removeprefix('adapter.'): value for name, value in artifact.tensors.items()
                          if name.startswith('adapter.')}
    restore_channel_checkpoint(model, channel)
    optimizer_binding = manifest.get('optimizer_state_artifact')
    optimizer_artifact = None
    if optimizer_binding is not None:
        optimizer_path = (manifest_path.parent / optimizer_binding['path']).resolve(strict=True)
        if optimizer_path.parent != manifest_path.parent or sha256_file_hex_cached(optimizer_path) != optimizer_binding['sha256']:
            raise ValueError('native composite optimizer-state artifact changed or escaped its manifest directory')
        optimizer_artifact, optimizer_id = decode_native_tensor_artifact(optimizer_path.read_bytes())
        if (optimizer_id != optimizer_binding['id'] or optimizer_artifact.kind != 'optimizer-state'
                or optimizer_artifact.native_identity != parent_identity
                or optimizer_artifact.source_artifact_id != artifact_id):
            raise ValueError('native composite optimizer-state artifact is not bound to its parameter update')
    output_identity = native_identity(model, config,
        checkpoint_sha256=__import__('hashlib').sha256(manifest_bytes).hexdigest())
    if (output_identity['embedding_weight_sha256'] != manifest['output_embedding_weight_sha256']
            or manifest['model_config']['architecture'] != config['architecture']):
        raise ValueError('native composite effective model identity differs from its manifest')
    state = {'schema': SCHEMA, 'model_config': config, 'base_state': parent_state,
             'parameter_artifact_id': artifact_id, 'optimizer': manifest.get('optimizer') or {},
             'optimizer_tensors': optimizer_artifact.tensors if optimizer_artifact is not None else {},
             'update_provenance': manifest.get('update_provenance') or {}}
    from .native_model import _bind_backbone_revision
    _bind_backbone_revision(model, __import__('hashlib').sha256(manifest_bytes).hexdigest())
    return model, tokenizer, roles, config, state
