"""Exact consumed-input closure for the external provider file bridge.

This enumerates already-declared inputs, never chooses a training source. Public
base directories and built task runtimes are verified in place; payload files
stay in the unchanged native-production namespace for the existing ferry.
"""
from __future__ import annotations
import json
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from natlang_neuralese.model.native_bundle import composite_dependencies, _semantic_dependencies
from natlang_neuralese.train.recipe import native_source_dependencies


def dependency_closure(request, *, namespace='/work/native-production'):
    files, assets, active_configs = {}, {}, set()
    root = Path(namespace).absolute()

    def add(path, expected=None):
        # Preserve the declared lexical namespace of valid offload symlinks.
        path = Path(path).absolute()
        if '..' in path.parts or root not in path.parents:
            raise ValueError('provider consumed file escaped its declared namespace: ' + str(path))
        digest = sha256_file_hex_cached(path)
        if expected is not None and digest != expected:
            raise ValueError('provider consumed dependency hash changed: ' + str(path))
        item = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
        if str(path) in files and files[str(path)] != item:
            raise ValueError('provider consumed dependency changed while enumerating')
        files[str(path)] = item
        return path

    def bound(value):
        path = add(value['path'], value['sha256'])
        if 'bytes' in value and path.stat().st_size != value['bytes']:
            raise ValueError('provider consumed dependency size changed')
        return path

    def checkpoint(value):
        path = bound(value) if isinstance(value, dict) else add(value)
        for digest, child in composite_dependencies(path).items():
            # Bundle resolver may resolve an offload symlink outside the root.
            # Such a dependency needs an explicit portable bundle before use.
            add(child, digest)
            with child.open('rb') as stream:
                binary = stream.read(2) == b'PK'
            if binary:
                # Trusted hash-bound native Torch checkpoints use mmap; tensor
                # payloads stay on disk and are never restored into a model.
                import torch
                state = torch.load(child, map_location='cpu', weights_only=False, mmap=True)
                configs = [state.get('model_config')]
                configs.extend(item.get('model_config') for item in state.get('participants', {}).values())
                for declaration in configs:
                    if declaration is not None:
                        asset(declaration['base'], base=True, expected=declaration['base_inventory'])
                del state

    def graphs(path):
        path = add(path)
        for item in native_source_dependencies(path):
            add(item['path'], item['sha256'])

    def asset(path, *, base=False, expected=None):
        path = Path(path).absolute()
        if base:
            from natlang_neuralese.model.native_model import base_inventory
            inventory = base_inventory(path)
        else:
            # Only the task runtime is an asset here; exclude package caches.
            # Its entire built dist is content-bound.
            required = path/'dist'
            if not (required/'teacher'/'cli.js').is_file():
                raise ValueError('provider task runtime has not been built')
            inventory = {str(p.relative_to(path)): sha256_file_hex_cached(p)
                         for p in sorted(required.rglob('*')) if p.is_file()}
        if expected is not None and inventory != expected:
            raise ValueError('provider checkpoint pinned base inventory changed')
        assets[str(path)] = {'path': str(path), 'kind': 'base' if base else 'task-runtime',
                             'inventory': inventory,
                             'sha256': canonical_json_sha256_hex(inventory)}

    def config_value(value, marker):
        if marker in active_configs:
            raise ValueError("provider consumed configuration contains a cycle")
        active_configs.add(marker)
        # Exact fields consumed by production.run/native semantic refresh.
        for key in ('seed_native_graphs', 'held_native_graphs'):
            if value.get(key): graphs(value[key])
        for key in ('cases', 'production_recipe'):
            if value.get(key):
                consumed = add(value[key])
                if key == 'production_recipe' and json.loads(consumed.read_text()).get('extends'):
                    raise ValueError('provider production_recipe must be its exact resolved portable file')
        if value.get('semantic_refresh_config'): config(add(value['semantic_refresh_config']))
        if value.get('base'): asset(value['base'], base=True)
        if value.get('runtime'): asset(value['runtime'])
        for key in ('pairs', 'held_pairs'):
            item = value.get(key)
            if item:
                bound(item) if isinstance(item, dict) else add(item)
        for binding in value.get('external_bundles', {}).values():
            manifest_path = bound(binding)
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('schema') != 'natlang.rl-block-bundle/1':
                raise ValueError('provider semantic external input is not a native block bundle')
            for entry in manifest['blocks'].values():
                child = (manifest_path.parent/entry['path']).resolve(strict=True)
                if child.parent != manifest_path.parent.resolve():
                    raise ValueError('provider semantic block escaped its bound bundle')
                bound({**entry, 'path': str(child)})
        for declaration in value.get('models', {}).values():
            if declaration.get('checkpoint'): checkpoint(declaration['checkpoint'])
        active_configs.remove(marker)
        return value

    def config(path):
        return config_value(json.loads(path.read_text()), str(path))

    for key in ('parent_checkpoint',):
        if request.get(key): checkpoint(request[key])
    for value in request.get('updates', {}).values(): checkpoint(value)
    for key in ('config', 'semantic_config'):
        if request.get(key): config(bound(request[key]))
    if request.get('state'):
        state = json.loads(bound(request['state']).read_text())
        for key in ('checkpoint', 'pending_native_checkpoint'):
            if state.get(key): checkpoint(state[key])
    if request.get('semantic_adapter'):
        path = bound(request['semantic_adapter'])
        value = json.loads(path.read_text())
        curriculum = value.get('qualified_curriculum')
        if curriculum is not None:
            if canonical_json_sha256_hex(curriculum['config']) != curriculum['sha256']:
                raise ValueError('provider qualified semantic curriculum hash changed')
            config_value(curriculum['config'], 'qualified-curriculum:'+curriculum['sha256'])
        checkpoint(value['checkpoint'])
        for child, digest in _semantic_dependencies(path, value['checkpoint']['sha256']).values():
            add(child, digest)
    return {'files': list(files.values()), 'assets': list(assets.values())}


def verify_assets(assets):
    """Verify pre-staged external assets; this never accepts path existence alone."""
    for item in assets:
        path = Path(item['path'])
        if item['kind'] == 'base':
            from natlang_neuralese.model.native_model import base_inventory
            actual = base_inventory(path)
        elif item['kind'] == 'task-runtime':
            actual = {str(p.relative_to(path)): sha256_file_hex_cached(p)
                      for p in sorted((path/'dist').rglob('*')) if p.is_file()}
        else:
            raise ValueError('unknown provider pre-staged asset kind')
        if actual != item['inventory'] or canonical_json_sha256_hex(actual) != item['sha256']:
            raise ValueError('provider pre-staged asset content differs: ' + str(path))
