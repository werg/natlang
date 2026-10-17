"""Exact consumed-input closure for the external provider file bridge.

This enumerates already-declared inputs, never chooses a training source. Public
base directories and built task runtimes are verified in place; payload files
stay in the unchanged native-production namespace for the existing ferry.
"""
from __future__ import annotations
import json
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from natlang_neuralese.common.runtime_inventory import task_runtime_inventory
from natlang_neuralese.model.native_bundle import composite_dependencies, _semantic_dependencies
from natlang_neuralese.train.recipe import native_source_dependencies


def dependency_closure(request, *, namespace='/work/native-production'):
    files, assets, active_configs, relocations = {}, {}, set(), {}
    consumed_adapters = set()
    consumed_checkpoints=set()
    consumed_payloads=set()
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

    def training_inputs(training):
        if not isinstance(training, dict):
            return
        policy_sources(training.get('policy_source_history', []))
        if not training.get('sft_progress') and training.get('schema') != 'natlang.neuralese-native-sft-training/1':
            return
        from natlang_neuralese.train.native_continuation import collect_sft_progress, sft_frame_complete
        progress = collect_sft_progress(training, borrow_frames=True)
        for name, frame in progress.items():
            if sft_frame_complete(frame):
                continue  # Completed historical source files are not read by mandatory SFT.
            identity = frame.get('identity') or {}
            if identity.get('objective_kind') == 'native-offline-pairwise':
                objective = identity.get('objective')
                if not isinstance(objective, dict) or not all(objective.get(key) for key in ('path', 'sha256')):
                    raise ValueError('incomplete offline preference frame needs its exact objective binding')
                bound(objective)
            for declaration in (frame.get('pending_source_declarations') or {}).values():
                from natlang_neuralese.train.native_pending_sources import pending_source_dependencies
                path = bound(declaration['binding'])
                for reference in pending_source_dependencies(path):
                    bound(reference)
            for source in frame.get('source_receipts', []):
                graphs(bound(source))

    def checkpoint(value, *, semantic_only=False):
        path = bound(value) if isinstance(value, dict) else add(value)
        selected=request.get('native_participant') if request.get('kind')=='round' else None
        if selected is None and isinstance(value,dict):selected=value.get('participant')
        context=(selected,semantic_only,tuple(sorted((type(key).__name__,str(key),str(value)) for key,value in relocations.items())))
        marker=(str(path.resolve()),files[str(path)]['sha256'],context)
        if marker in consumed_checkpoints:return
        consumed_checkpoints.add(marker)
        decoded={}
        def load_once(child):
            from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
            key=str(Path(child).resolve())
            if key not in decoded:decoded[key]=CheckpointPolicy.load(child,expand_shared=False)
            return decoded[key]
        for digest, child in composite_dependencies(path, participant=selected,
                semantic_only=semantic_only,checkpoint_loader=load_once).items():
            # Bundle resolver may resolve an offload symlink outside the root.
            # Such a dependency needs an explicit portable bundle before use.
            add(child, digest)
            with child.open('rb') as stream:
                binary = stream.read(2) == b'PK'
            payload=(str(child.resolve()),digest,context)
            if binary and payload not in consumed_payloads:
                consumed_payloads.add(payload)
                # Trusted hash-bound native Torch checkpoints use mmap; tensor
                # payloads stay on disk and are never restored into a model.
                from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
                state = load_once(child)
                training_inputs(state)
                training_inputs(state.get('training'))
                training_inputs(state.get('shared_training'))
                for participant in state.get('participants', {}).values():
                    training_inputs(participant)
                configs = [(state.get('participant',selected),state.get('model_config'))]
                configs.extend((name,item.get('model_config')) for name,item in state.get('participants', {}).items()
                               if selected is None or name==selected)
                for name,declaration in configs:
                    if declaration is not None:
                        asset(relocations.get(name,declaration['base']), base=True, expected=declaration['base_inventory'])
                del state
            decoded.pop(str(child.resolve()),None)

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
            inventory = task_runtime_inventory(path)
        if expected is not None and inventory != expected:
            raise ValueError('provider checkpoint pinned base inventory changed')
        if base and root in path.parents:
            for name, digest in inventory.items():
                add(path/name, digest)
        assets[str(path)] = {'path': str(path), 'kind': 'base' if base else 'task-runtime',
                             'inventory': inventory,
                             'sha256': canonical_json_sha256_hex(inventory)}

    def bundle(reference, *, policy_id=None):
        manifest_path = bound(reference)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema') != 'natlang.rl-block-bundle/1':
            raise ValueError('provider semantic external input is not a native block bundle')
        if policy_id is not None and manifest.get('policy_id') != policy_id:
            raise ValueError('provider policy bundle differs from its behavior adapter checkpoint')
        from natlang_neuralese.rl.graph_replay import verified_bundle_path
        for ident, entry in manifest['blocks'].items():
            child = (manifest_path.parent/entry['path']).resolve(strict=True)
            if child.parent != manifest_path.parent.resolve():
                raise ValueError('provider semantic block escaped its bound bundle')
            bound({**entry, 'path': str(child)})
            verified_bundle_path(manifest_path, ident, policy_id=manifest.get('policy_id'), manifest=manifest)

    def semantic_adapter(reference):
        path = bound(reference)
        value = json.loads(path.read_text())
        if str(path) in consumed_adapters:
            return value
        consumed_adapters.add(str(path))
        curriculum = value.get('qualified_curriculum')
        if curriculum is not None:
            if canonical_json_sha256_hex(curriculum['config']) != curriculum['sha256']:
                raise ValueError('provider qualified semantic curriculum hash changed')
            config_value(curriculum['config'], 'qualified-curriculum:'+curriculum['sha256'])
        checkpoint(value['checkpoint'], semantic_only=True)
        for child, digest in _semantic_dependencies(path, value['checkpoint']['sha256']).values():
            add(child, digest)
        policy_sources((value.get('policy_training_progress') or {}).get('policy_source_history', []))
        return value

    def policy_sources(sources):
        for source in sources:
            bound(source['rollouts'])
            adapter = semantic_adapter(source['adapter'])
            bundle(source['block_bundle'], policy_id=adapter['checkpoint']['sha256'])

    def config_value(value, marker):
        if marker in active_configs:
            raise ValueError("provider consumed configuration contains a cycle")
        active_configs.add(marker)
        if value.get('base'):
            relocations.setdefault(value.get('native_participant'),value['base'])
        for name,declaration in value.get('models',{}).items():
            if declaration.get('base'):
                relocations.setdefault(name,declaration['base'])
        policy_sources((value.get('policy_training') or {}).get('sources', []))
        # Exact fields consumed by production.run/native semantic refresh.
        if value.get('pending_native_sources'):
            from natlang_neuralese.train.native_pending_sources import pending_source_dependencies
            reference = value['pending_native_sources']
            path = bound(reference) if isinstance(reference, dict) else add(reference)
            for item in pending_source_dependencies(path):
                bound(item)
        for key in ('seed_native_graphs', 'held_native_graphs', 'foundation_native_graphs'):
            if value.get(key):
                graphs(bound(value[key]) if isinstance(value[key], dict) else value[key])
        if value.get('withdrawn_native_graphs'):
            item=value['withdrawn_native_graphs']
            proof=bound(item) if isinstance(item,dict) else add(item)
            declaration=json.loads(proof.read_text())
            if declaration.get('schema')!='natlang.native-graph-withdrawal/1':
                raise ValueError('provider withdrawal declaration schema differs')
            evidence=declaration['evidence'];path=Path(evidence['path'])
            if not path.is_absolute():path=proof.parent/path
            path=path.resolve(strict=True)
            bound({**evidence,'path':str(path)})
        from natlang_neuralese.eval.task_cadence import parse_task_eval
        specifications = [value.get('task_eval'), (value.get('trainer_parameters') or {}).get('task_eval'),
                          (value.get('seed_trainer_parameters') or {}).get('task_eval')]
        for command_name in ('seed_command','learner_command'):
            command=value.get(command_name) or []
            for index,argument in enumerate(command):
                if argument=='--task-eval':
                    if index+1>=len(command):raise ValueError('provider task-eval command omitted its value')
                    specification=command[index+1]
                elif argument.startswith('--task-eval='):specification=argument.split('=',1)[1]
                else:continue
                if specification.startswith('{') and specification.endswith('}') and specification[1:-1].isidentifier():
                    name=specification[1:-1]
                    if name not in value:raise ValueError('provider task-eval command has an unbound placeholder')
                    specification=value[name]
                specifications.append(specification)
        for specification in specifications:
            specification=parse_task_eval(specification)
            if specification:
                bound(specification['packet'])
                if specification.get('runtime'):
                    asset(specification['runtime'])
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
        for reference in value.get('external_bundles', {}).values():
            bundle(reference)
        for declaration in value.get('models', {}).values():
            if declaration.get('base'):asset(declaration['base'],base=True)
            if declaration.get('checkpoint'): checkpoint(declaration['checkpoint'])
        active_configs.remove(marker)
        return value

    def config(path):
        return config_value(json.loads(path.read_text()), str(path))

    # Resolve explicitly consumed relocations before checking saved checkpoint
    # inventories. Saved paths may name a different model on the coordinator.
    for key in ('config', 'semantic_config'):
        if request.get(key): config(bound(request[key]))
    for key in ('parent_checkpoint', 'partial_checkpoint'):
        if request.get(key): checkpoint(request[key])
    for value in request.get('updates', {}).values():
        checkpoint(value)
        update_path = Path(value['path'])
        receipt_path = add(update_path.parent/'native-training-receipt.json')
        receipt = json.loads(receipt_path.read_text())
        bound(receipt['training_receipt'])
    if request.get('state'):
        state = json.loads(bound(request['state']).read_text())
        for key in ('checkpoint', 'pending_native_checkpoint', 'partial_checkpoint'):
            if state.get(key): checkpoint(state[key])
        progress=state.get('last_learner_progress') or {}
        if progress.get('full_report'):bound(progress['full_report'])
        if state.get('pending_semantic_source'):
            policy_sources([state['pending_semantic_source']])
    if request.get('semantic_adapter'):
        semantic_adapter(request['semantic_adapter'])
    return {'files': list(files.values()), 'assets': list(assets.values())}


def verify_assets(assets):
    """Verify pre-staged external assets; this never accepts path existence alone."""
    for item in assets:
        path = Path(item['path'])
        if item['kind'] == 'base':
            from natlang_neuralese.model.native_model import base_inventory
            actual = base_inventory(path)
        elif item['kind'] == 'task-runtime':
            actual = task_runtime_inventory(path)
        else:
            raise ValueError('unknown provider pre-staged asset kind')
        if actual != item['inventory'] or canonical_json_sha256_hex(actual) != item['sha256']:
            raise ValueError('provider pre-staged asset content differs: ' + str(path))
