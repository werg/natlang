"""Restorable native checkpoints with immutable named-tensor updates.

The manifest hash is the effective checkpoint identity. Each manifest points
to an immutable full native parent and a content-addressed named parameter
patch, so updates need not rewrite unchanged multi-gigabyte backbone tensors.
"""
from __future__ import annotations

import json
import io
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
    if channel.get('token_override_ids'):
        tensors['extra_input_rows'] = channel.pop('extra_input_rows')
        extra_output = channel.pop('extra_output_rows')
        if extra_output is not None:tensors['extra_output_rows'] = extra_output
    tensors.update({'adapter.' + name: value for name, value in channel.pop('adapter').items()})
    return channel, tensors, output is not None


def write_native_composite_checkpoint(base_checkpoint, output_dir, model, model_config, *,
                                      base_identity, source_gradient_id=None, optimizer=None,
                                      optimizer_tensors=None, update_provenance=None, participant=None,
                                      updated_parameter_names=None, parent_model=None, training_state=None):
    """Write a durable immutable composite checkpoint over `base_checkpoint`.

    Old interface-only callers preserve their channel contract. Named full
    updates require a detached selected-parameter copy and its unchanged parent;
    no projected-value/head tensors are accepted.
    """
    base = Path(base_checkpoint).resolve(strict=True)
    base_sha = sha256_file_hex_cached(base)
    if base_identity.get('checkpoint_sha256') != base_sha:
        raise ValueError('native composite base identity does not match the parent checkpoint bytes')
    if base_identity.get('architecture') != model_config.get('architecture'):
        raise ValueError('native composite base architecture differs from the live model')
    if updated_parameter_names is None:
        binding = getattr(model, '_native_backbone_binding', None)
        current_parameters = tuple((name, id(parameter), parameter._version)
                                   for name, parameter in model.port.hf.named_parameters())
        if (not isinstance(binding, dict) or binding.get('checkpoint_sha256') != base_sha
                or binding.get('parameters') != current_parameters):
            raise ValueError('interface-only composite update requires the unchanged backbone loaded from this exact parent checkpoint')
        channel, tensors, has_output_rows = _channel_tensors(model)
        expected_names = {'input_rows', 'adapter.down.weight', 'adapter.up.weight'}
        if has_output_rows:
            expected_names.add('output_rows')
        if model.token_override_ids:
            expected_names.add('extra_input_rows')
            if not model.tied:expected_names.add('extra_output_rows')
        if set(tensors) != expected_names:
            raise ValueError('native composite parameter inventory differs from the model channel contract')
        scope = 'native-interface'
    else:
        if parent_model is None or parent_model is model:
            raise ValueError('full native parameter publication needs its unchanged live parent')
        binding = getattr(parent_model, '_native_parameter_binding', None)
        parent_parameters = dict(parent_model.named_parameters())
        current = tuple((name,id(value),value._version) for name,value in parent_parameters.items())
        if (not binding or binding['checkpoint_sha256'] != base_sha or binding['parameters'] != current):
            raise ValueError('native parameter update parent changed since exact checkpoint restoration')
        parameters = dict(model.named_parameters())
        names = set(updated_parameter_names)
        added_names = parameters.keys() - parent_parameters.keys()
        explicit_upgrade = getattr(model, '_native_token_row_upgrade', None)
        allowed_added = {'extra_input_rows'} | ({'extra_output_rows'} if not model.tied else set())
        added_ok = (added_names == allowed_added and not parent_model.token_override_ids
                    and explicit_upgrade and explicit_upgrade.get('token_ids') == list(model.token_override_ids)
                    and added_names <= names)
        if (not names or len(names) != len(updated_parameter_names) or not names <= parameters.keys()
                or not parent_parameters.keys() <= parameters.keys() or added_names and not added_ok):
            raise ValueError('native parameter patch must declare its exact model parameter inventory')
        for name,value in parameters.items():
            if name in added_names:continue  # exact declared sparse-row addition, verified again at restore
            parent = parent_parameters[name]
            if name in names:
                promotion = (explicit_upgrade or {}).get('promoted_parameters', {}).get(name)
                promoted_ok = (name in {'input_rows','output_rows','extra_input_rows','extra_output_rows'}
                    and promotion == {'from':str(parent.dtype),'to':'torch.float32'}
                    and parent.dtype in (torch.bfloat16,torch.float16) and value.dtype == torch.float32)
                if value is parent or value.shape != parent.shape or value.dtype != parent.dtype and not promoted_ok:
                    raise ValueError('native updated parameters must be detached matching parent copies')
            elif value is not parent:
                raise ValueError('native patch changed an undeclared model parameter: '+name)
        tensors = {name: parameters[name].detach().cpu().clone() for name in sorted(names)}
        scope = 'model-parameters'
    artifact = NativeTensorArtifact('parameter-update', dict(base_identity), tensors,
        source_gradient_id, {'base_checkpoint_sha256': base_sha,
                             'operation': 'replace-native-interface-parameters' if scope == 'native-interface' else 'replace-native-model-parameters',
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
            {'for_parameter_artifact_id': artifact_id,
             **({'optimizer': optimizer['name'], 'step': optimizer['step']}
                if isinstance(optimizer, dict) and 'name' in optimizer and 'step' in optimizer else {})}).validate()
        from .native_artifact import native_tensor_artifact_id
        optimizer_id = native_tensor_artifact_id(optimizer_artifact)
        optimizer_path = directory / f'{optimizer_id}.safetensors'
        _durable_immutable(optimizer_path, encode_native_tensor_artifact(optimizer_artifact))
        optimizer_binding = {'id': optimizer_id, 'path': optimizer_path.name,
                             'sha256': sha256_file_hex_cached(optimizer_path)}
    training_binding = None
    if training_state is not None:
        buffer = io.BytesIO()
        torch.save(training_state, buffer)
        payload = buffer.getvalue()
        import hashlib
        digest = hashlib.sha256(payload).hexdigest()
        path = directory / f'native-training-{digest}.pt'
        _durable_immutable(path, payload)
        training_binding = {'path':path.name,'sha256':digest}
    from .native_model import native_identity
    output_embed_sha = native_identity(model, model_config, checkpoint_sha256=base_sha)['embedding_weight_sha256']
    manifest = {'schema': SCHEMA, 'parameter_scope': scope,
        'updated_parameter_names': sorted(tensors), 'training_state': training_binding,
        'base_checkpoint': {'path': str(base), 'sha256': base_sha, 'participant': participant},
        'parameter_artifact': {'path': tensor_path.name, 'id': artifact_id,
                               'sha256': sha256_file_hex_cached(tensor_path)},
        'optimizer_state_artifact': optimizer_binding,
        'model_config': {key: model_config[key] for key in ('architecture', 'rank', 'marker_ids',
            'tokenizer_fingerprint', 'base_inventory_sha256', 'precision', 'dtype')},
        'output_embedding_weight_sha256': output_embed_sha,
        'optimizer': optimizer or {}, 'update_provenance': update_provenance or {}}
    manifest['model_config']['native_linear_precision'] = model_config.get('native_linear_precision','ieee')
    if 'native_attention_precision' in model_config:
        manifest['model_config']['native_attention_precision'] = model_config['native_attention_precision']
    if 'sparse_row_dtype' in model_config:
        manifest['model_config']['sparse_row_dtype'] = model_config['sparse_row_dtype']
    if model.token_override_ids:
        manifest['model_config']['token_override_ids'] = list(model.token_override_ids)
        if 'token_override_dtype' in model_config:
            manifest['model_config']['token_override_dtype'] = model_config['token_override_dtype']
    bound_tokens = tuple((getattr(model, '_native_backbone_binding', None) or {}).get('token_override_ids', ()))
    bound_dtype = (getattr(model, '_native_backbone_binding', None) or {}).get('sparse_row_dtype', str(model.input_rows.dtype))
    actual_extra_dtype = str(model.extra_input_rows.dtype) if model.token_override_ids else None
    bound_extra_dtype = (getattr(model, '_native_backbone_binding', None) or {}).get('token_override_dtype', actual_extra_dtype)
    if bound_tokens != model.token_override_ids or bound_dtype != str(model.input_rows.dtype) or bound_extra_dtype != actual_extra_dtype:
        upgrade = getattr(model, '_native_token_row_upgrade', None)
        if (bound_tokens != model.token_override_ids and bound_tokens or not upgrade
                or upgrade.get('token_ids') != list(model.token_override_ids)):
            raise ValueError('native composite sparse-row addition requires its explicit parent upgrade')
        manifest['token_row_upgrade'] = upgrade
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


def resolve_composite_artifact(binding, manifest_path, *, sibling=False):
    """Resolve a pinned role through the shared exact artifact resolver."""
    from ..common.artifact_paths import ArtifactResolver
    manifest_path = Path(manifest_path).resolve(strict=True)
    value = Path(binding['path'])
    if sibling and (value.is_absolute() or len(value.parts) != 1):
        raise ValueError('native composite tensor/training artifact must stay beside its manifest')
    directory = manifest_path.parent
    resolved = value if value.is_absolute() else directory/value
    objects = directory if directory.name == 'objects' else directory/'objects'
    resolver = ArtifactResolver({}, {'native-composite-artifact': {
        'logical': str(value), 'resolved': str(resolved), 'sha256':binding['sha256']}},content_root=objects)
    return Path(resolver.path('native-composite-artifact',required=True)).resolve(strict=True)


def native_update_directory(checkpoint):
    """Use one stable directory across arbitrarily many immutable updates."""
    current = Path(checkpoint).resolve(strict=True)
    seen = set()
    while current not in seen:
        seen.add(current)
        with current.open('rb') as stream:
            if stream.read(1) != b'{':
                return current.parent/'native-updates'
        manifest = json.loads(current.read_text())
        if manifest.get('schema') != SCHEMA:
            return current.parent/'native-updates'
        parent = manifest['base_checkpoint']
        current = resolve_composite_artifact(parent,current)
        if sha256_file_hex_cached(current) != parent['sha256']:
            raise ValueError('native composite update ancestor bytes changed')
    raise ValueError('native composite checkpoint ancestry contains a cycle')


def load_native_composite_checkpoint(path, *, base=None, device='cpu', dtype=None):
    """Restore full parent once, then apply immutable patches oldest to newest."""
    from .native_model import load_native_checkpoint
    current = Path(path).resolve(strict=True)
    chain, seen = [], set()
    participant = None
    while True:
        if current in seen:
            raise ValueError('native composite checkpoint ancestry contains a cycle')
        seen.add(current)
        with current.open('rb') as stream:
            prefix = stream.read(1)
        if prefix != b'{':
            break
        manifest = json.loads(current.read_text())
        if manifest.get('schema') != SCHEMA:
            break
        chain.append(current)
        parent = manifest['base_checkpoint']
        current = resolve_composite_artifact(parent,current)
        if sha256_file_hex_cached(current) != parent['sha256']:
            raise ValueError('native composite parent checkpoint bytes changed')
        participant = parent.get('participant')
    if not chain:
        raise ValueError('unsupported native composite checkpoint schema')
    result = load_native_checkpoint(current,base=base,device=device,dtype=dtype,participant=participant)
    for manifest_path in reversed(chain):
        result = _apply_native_composite_checkpoint(manifest_path,result)
    return result


def _apply_native_composite_checkpoint(path, restored):
    """Apply one verified patch to the already restored immediate parent."""
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
    base_path = resolve_composite_artifact(parent,manifest_path)
    if sha256_file_hex_cached(base_path) != parent['sha256']:
        raise ValueError('native composite parent checkpoint bytes changed')
    from .native_model import native_identity
    model, tokenizer, roles, config, parent_state = restored
    parent_identity = native_identity(model, config, checkpoint_sha256=parent['sha256'])
    artifact_binding = manifest['parameter_artifact']
    artifact_path = resolve_composite_artifact(artifact_binding,manifest_path,sibling=True)
    if sha256_file_hex_cached(artifact_path) != artifact_binding['sha256']:
        raise ValueError('native composite tensor patch bytes changed')
    artifact, artifact_id = decode_native_tensor_artifact(artifact_path.read_bytes())
    if artifact_id != artifact_binding['id'] or artifact.kind != 'parameter-update':
        raise ValueError('native composite patch artifact binding or kind is invalid')
    if artifact.native_identity != parent_identity:
        raise ValueError('native composite patch belongs to another exact parent model')
    if manifest.get('token_row_upgrade'):
        from .native_model import upgrade_reasoning_tokens
        declared = manifest['token_row_upgrade']
        if (declared.get('schema') != 'natlang.native-token-row-upgrade/1'
                or declared.get('added') is True and model.token_override_ids
                or declared.get('added') is not True and not declared.get('promoted_parameters')):
            raise ValueError('native composite cannot replace an existing sparse-row contract')
        actual = upgrade_reasoning_tokens(model, tokenizer, config,
            dtype={'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16,
                   'torch.float16':torch.float16}[manifest['model_config'].get('token_override_dtype',config['dtype'])],
            sparse_row_dtype={'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16,
                   'torch.float16':torch.float16}[manifest['model_config'].get('sparse_row_dtype',config['dtype'])])
        if 'token_override_dtype' not in manifest['model_config']:
            config.pop('token_override_dtype', None)
        if 'sparse_row_dtype' not in manifest['model_config']:config.pop('sparse_row_dtype', None)
        if actual != declared or config.get('token_override_ids') != manifest['model_config'].get('token_override_ids'):
            raise ValueError('native composite reasoning-row upgrade differs from actual pinned tokenizer')
    scope = manifest.get('parameter_scope','native-interface')
    if scope == 'model-parameters':
        parameters = dict(model.named_parameters())
        names = manifest['updated_parameter_names']
        if set(artifact.tensors) != set(names) or len(names) != len(set(names)) or not set(names) <= parameters.keys():
            raise ValueError('native composite full parameter inventory differs from the declared patch')
        with torch.no_grad():
            for name,value in artifact.tensors.items():
                parameter = parameters[name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError('native composite full parameter shape/dtype differs: '+name)
                parameter.copy_(value.to(parameter.device))
    elif scope == 'native-interface':
        from .shared_state import channel_checkpoint, restore_channel_checkpoint
        channel = channel_checkpoint(model)
        required = {'input_rows', 'adapter.down.weight', 'adapter.up.weight'}
        if channel['output_rows'] is not None:
            required.add('output_rows')
        if model.token_override_ids:
            required.add('extra_input_rows')
            if not model.tied:required.add('extra_output_rows')
        if set(artifact.tensors) != required:
            raise ValueError('native composite patch tensor names differ from the native interface')
        channel['input_rows'] = artifact.tensors['input_rows']
        channel['output_rows'] = artifact.tensors.get('output_rows')
        if model.token_override_ids:
            channel['extra_input_rows'] = artifact.tensors['extra_input_rows']
            channel['extra_output_rows'] = artifact.tensors.get('extra_output_rows')
        channel['adapter'] = {name.removeprefix('adapter.'): value for name, value in artifact.tensors.items()
                              if name.startswith('adapter.')}
        restore_channel_checkpoint(model, channel)
    else:
        raise ValueError('unsupported native composite parameter patch scope')
    optimizer_binding = manifest.get('optimizer_state_artifact')
    optimizer_artifact = None
    if optimizer_binding is not None:
        optimizer_path = resolve_composite_artifact(optimizer_binding,manifest_path,sibling=True)
        if sha256_file_hex_cached(optimizer_path) != optimizer_binding['sha256']:
            raise ValueError('native composite optimizer-state artifact changed or escaped its manifest directory')
        optimizer_artifact, optimizer_id = decode_native_tensor_artifact(optimizer_path.read_bytes())
        if (optimizer_id != optimizer_binding['id'] or optimizer_artifact.kind != 'optimizer-state'
                or optimizer_artifact.native_identity != parent_identity
                or optimizer_artifact.source_artifact_id != artifact_id):
            raise ValueError('native composite optimizer-state artifact is not bound to its parameter update')
    if any(config.get(key) != value for key,value in manifest['model_config'].items()):
        raise ValueError('native composite model configuration differs from the restored parent')
    output_identity = native_identity(model, config,
        checkpoint_sha256=__import__('hashlib').sha256(manifest_bytes).hexdigest())
    if (output_identity['embedding_weight_sha256'] != manifest['output_embedding_weight_sha256']
            or manifest['model_config']['architecture'] != config['architecture']):
        raise ValueError('native composite effective model identity differs from its manifest')
    training = {}
    if manifest.get('training_state') is not None:
        binding = manifest['training_state']
        path = resolve_composite_artifact(binding,manifest_path,sibling=True)
        if sha256_file_hex_cached(path) != binding['sha256']:
            raise ValueError('native composite training state changed or escaped its manifest directory')
        from ..train.checkpoint_policy import CheckpointPolicy
        training = CheckpointPolicy.load(path)
        if not isinstance(training,dict):
            raise ValueError('native composite training state must be a dictionary')
    # Preserve the original full trainer parent once; retaining every ancestor's
    # optimizer tensors would make reload memory grow with update count.
    root_state = parent_state.get('base_state',parent_state) if parent_state.get('schema') == SCHEMA else parent_state
    state = {'schema': SCHEMA, 'training': training, 'model_config': config, 'base_state': root_state,
             'parent_checkpoint': parent,
             'parameter_artifact_id': artifact_id, 'optimizer': manifest.get('optimizer') or {},
             'optimizer_tensors': optimizer_artifact.tensors if optimizer_artifact is not None else {},
             'update_provenance': manifest.get('update_provenance') or {}}
    from .native_model import _bind_backbone_revision
    _bind_backbone_revision(model, __import__('hashlib').sha256(manifest_bytes).hexdigest())
    return model, tokenizer, roles, config, state
