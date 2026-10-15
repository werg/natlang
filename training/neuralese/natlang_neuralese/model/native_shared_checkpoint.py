"""One shared checkpoint identity with independently consumable immutable payloads.

The global manifest commits last through CheckpointPolicy. Participant payloads
retain the existing model/optimizer/FP8/CE contract; this changes storage only.
"""
from __future__ import annotations

import os
import contextlib
import fcntl
import json
import re
from pathlib import Path
import uuid

from ..common.hashing import sha256_file_hex_cached, canonical_json_sha256_hex

SHARDED_SCHEMA = 'natlang.neuralese-native-shared-checkpoint-shards/1'
SHARED_SCHEMA = 'natlang.neuralese-native-shared-checkpoint/1'
PARTICIPANT_SCHEMA = 'natlang.native-shared-participant-payload/1'
SEMANTIC_SCHEMA = 'natlang.native-shared-semantic-payload/1'
TRAINING_SCHEMA = 'natlang.native-shared-training-payload/1'


def _load_raw(path, map_location='cpu'):
    from ..train.checkpoint_policy import CheckpointPolicy
    return CheckpointPolicy.load(path, map_location, expand_shared=False)


def shared_manifest(path):
    """Read metadata without expanding any participant payloads."""
    with Path(path).open('rb') as stream:
        if stream.read(2) != b'PK':
            return None
    value = _load_raw(path)
    if not isinstance(value, dict) or value.get('schema') != SHARDED_SCHEMA:
        return None
    if (not isinstance(value.get('participants'), dict) or not value['participants']
            or set(value) != {'schema', 'participants', 'semantic', 'training'}):
        raise ValueError('shared checkpoint manifest has an invalid payload closure')
    return value


def shared_dependencies(manifest, path, *, participant=None, semantic_only=False):
    """Resolve only explicitly consumed roles through the existing artifact resolver."""
    from .native_composite import resolve_composite_artifact
    if participant is not None and participant not in manifest['participants']:
        raise ValueError('shared checkpoint does not declare the requested participant')
    chosen = {} if semantic_only else (manifest['participants'] if participant is None
                                      else {participant: manifest['participants'][participant]})
    roles = [('participant:' + name, binding) for name, binding in chosen.items()]
    roles.extend((role, manifest[role]) for role in ('semantic', 'training'))
    result = {}
    for role, binding in roles:
        if set(binding) != {'path', 'sha256', 'bytes'} or type(binding['bytes']) is not int or binding['bytes'] <= 0:
            raise ValueError('shared checkpoint payload requires exact path/SHA/size')
        resolved = resolve_composite_artifact(binding, path)
        if resolved.stat().st_size != binding['bytes']:
            raise ValueError('shared checkpoint payload size differs from its global manifest')
        result[role] = resolved
    return result


def expand_shared(manifest, path, *, participant=None, semantic_only=False, map_location='cpu'):
    """Restore the original shared-state dictionary, optionally selecting one model."""
    dependencies = shared_dependencies(manifest, path, participant=participant, semantic_only=semantic_only)
    result = {'schema': SHARED_SCHEMA, 'participants': {}}
    for role, source in dependencies.items():
        payload = _load_raw(source, map_location)
        if role.startswith('participant:'):
            name = role.removeprefix('participant:')
            if payload.get('schema') != PARTICIPANT_SCHEMA or payload.get('participant') != name:
                raise ValueError('shared participant payload belongs to another declared model')
            result['participants'][name] = {key: value for key, value in payload.items()
                                          if key not in ('schema', 'participant')}
        elif role == 'semantic':
            if payload.get('schema') != SEMANTIC_SCHEMA:
                raise ValueError('shared semantic payload schema differs')
            result.update({key: payload[key] for key in ('vae_specification', 'vae_state', 'vae_optimizer')})
        else:
            if payload.get('schema') != TRAINING_SCHEMA:
                raise ValueError('shared training payload schema differs')
            result['training'] = payload['training']
    return result


def _save_shared_checkpoint(policy, state):
    """Seal payloads first and commit the one global manifest through the shared writer."""
    from ..train.trajectory_state import _atomic_checkpoint
    from ..common.io_quiet import io_quiet
    if (set(state) != {'schema', 'participants', 'vae_specification', 'vae_state', 'vae_optimizer', 'training'}
            or state.get('schema') != SHARED_SCHEMA or not state.get('participants')):
        raise ValueError('only complete shared native training states may be sharded')
    out = Path(policy.out).resolve(); objects = out/'objects'; objects.mkdir(parents=True, exist_ok=True)
    created, manifest = [], None

    def write(payload):
        temporary = objects/('.shared-' + uuid.uuid4().hex + '.pt')
        try:
            _atomic_checkpoint(temporary, payload)
            digest = sha256_file_hex_cached(temporary); target = objects/digest
            try:
                os.link(temporary, target)
                created.append(target)
            except FileExistsError:
                if sha256_file_hex_cached(target) != digest:
                    raise ValueError('shared checkpoint content object collision')
            descriptor = os.open(objects, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
            return {'path': str(target), 'sha256': digest, 'bytes': target.stat().st_size}
        finally:
            temporary.unlink(missing_ok=True)

    try:
        with io_quiet('shared checkpoint payload commit'):
            participants = {}
            for name, entry in state['participants'].items():
                if not isinstance(name, str) or {'schema', 'participant'} & entry.keys():
                    raise ValueError('shared participant metadata overwrites a payload role')
                participants[name] = write({'schema': PARTICIPANT_SCHEMA, 'participant': name, **entry})
            semantic = write({'schema': SEMANTIC_SCHEMA, **{key: state[key]
                for key in ('vae_specification', 'vae_state', 'vae_optimizer')}})
            training = write({'schema': TRAINING_SCHEMA, 'training': state['training']})
            manifest = {'schema': SHARDED_SCHEMA, 'participants': participants,
                        'semantic': semantic, 'training': training}
            return policy.save(manifest)
    except BaseException:
        # A failure after the global atomic rename may be in policy metadata;
        # never remove payloads of an already committed recoverable manifest.
        current = shared_manifest(policy.slot) if policy.slot.exists() else None
        if manifest is None or current != manifest:
            for path in created: path.unlink(missing_ok=True)
            descriptor = os.open(objects, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        raise


@contextlib.contextmanager
def shared_checkpoint_lock(out):
    """Serialize the commit, full-closure pin and retention operations."""
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    with (root/'.shared-checkpoint.lock').open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def pin_shared_checkpoint(checkpoint, *, participant=None):
    """Pin the committed global inode AND every consumed immutable payload."""
    from .native_bundle import verify_bundle
    from ..common.jsonio import write_json_durable
    checkpoint = Path(checkpoint).resolve()
    with shared_checkpoint_lock(checkpoint.parent):
        manifest = shared_manifest(checkpoint)
        if manifest is None:
            raise ValueError('a shared closure pin requires a sharded global checkpoint')
        digest = sha256_file_hex_cached(checkpoint)
        target = checkpoint.parent/'pins'/digest/('full' if participant is None else canonical_json_sha256_hex(participant))
        target.mkdir(parents=True, exist_ok=True)
        dependencies = shared_dependencies(manifest, checkpoint, participant=participant)
        files = []
        def pin(source, destination):
            try: os.link(source, destination)
            except FileExistsError:
                if sha256_file_hex_cached(source) != sha256_file_hex_cached(destination):
                    raise ValueError('shared closure pin contains different immutable bytes')
        pin(checkpoint, target/'checkpoint.pt')
        (target/'objects').mkdir(exist_ok=True)
        for source in dependencies.values():
            sha = sha256_file_hex_cached(source); destination = target/'objects'/sha
            pin(source, destination)
            files.append({'path': 'objects/'+sha, 'sha256': sha, 'bytes': source.stat().st_size})
        descriptor = os.open(target/'objects', os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        # The same verifier and resolver as portable bundles prove the closure.
        receipt = {'schema': 'natlang.native-checkpoint-bundle/1', 'participant': participant,
                   'checkpoint': {'path': 'checkpoint.pt', 'sha256': digest}, 'files': files,
                   'semantic_adapter': None, 'semantic_files': [], 'identity_rewritten': False}
        write_json_durable(target/'bundle.json', receipt)
        return verify_bundle(target)


def prune_shared_objects(out):
    """Drop superseded payload paths; explicit live pins retain full closures."""
    root = Path(out)
    with shared_checkpoint_lock(root):
        keep = set()
        roots = list(root.glob('*.pt'))
        roots.extend(root.glob('pins/*/*/checkpoint.pt'))
        for checkpoint in roots:
            if not checkpoint.exists(): continue
            manifest = shared_manifest(checkpoint)
            if manifest is None: continue
            # A selected pin may deliberately omit a peer. Its bundle verifier
            # already binds its own closure; hardlinks retain those payloads.
            if root/'pins' in checkpoint.parents: continue
            keep.update(binding['sha256'] for binding in [*manifest['participants'].values(),
                                                         manifest['semantic'], manifest['training']])
        removed, freed = [], 0
        for path in (root/'objects').glob('*'):
            if re.fullmatch('[0-9a-f]{64}', path.name) and path.name not in keep:
                freed += path.stat().st_size; path.unlink(); removed.append(path.name)
        return {'removed': removed, 'unlinked_bytes': freed}


def save_shared_checkpoint(policy, state):
    with shared_checkpoint_lock(policy.out):
        event = _save_shared_checkpoint(policy, state)
    prune_shared_objects(policy.out)
    return event
