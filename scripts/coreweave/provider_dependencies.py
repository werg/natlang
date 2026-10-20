"""Exact consumed-input closure for the external provider file bridge.

This enumerates already-declared inputs, never chooses a training source. Public
base directories, built task runtimes and explicitly declared read-only inputs
are verified in place; ferry files stay in the native-production namespace.
"""
from __future__ import annotations
import json
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from natlang_neuralese.common.runtime_inventory import task_runtime_inventory
from natlang_neuralese.model.native_bundle import composite_dependencies, _semantic_dependencies
from natlang_neuralese.train.recipe import native_source_dependencies


def _readonly_roots(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError('provider readonly input roots must be a list')
    roots = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError('invalid provider readonly input root')
        path = Path(value)
        if (not path.is_absolute() or '..' in path.parts or str(path) != value
                or path == Path('/') or path in roots or path.resolve() != path):
            raise ValueError('invalid provider readonly input root')
        roots.append(path)
    return tuple(roots)


def _readonly_file(path, roots, *, namespace=None):
    path = Path(path)
    resolved = path.resolve()
    targets = (*roots, *((Path(namespace),) if namespace is not None else ()))
    allowed = (any(root in path.parents and root in resolved.parents for root in roots)
        if namespace is None else
        any(root in path.parents for root in roots) and any(root in resolved.parents for root in targets))
    if (not path.is_absolute() or '..' in path.parts or
            not allowed):
        raise ValueError('provider input file escaped its declared readonly roots: ' + str(path))
    if not path.is_file():
        raise ValueError('provider readonly input file is missing: ' + str(path))
    return path


def dependency_closure(request, *, namespace='/work/native-production', readonly_input_roots=(),
                       _checkpoint_payload=None):
    files, assets, active_configs, relocations = {}, {}, set(), {}
    consumed_adapters = set()
    consumed_checkpoints=set()
    consumed_payloads=set()
    checkpoint_controls={}
    root = Path(namespace).absolute()
    readonly_roots = _readonly_roots(readonly_input_roots)

    def add(path, expected=None):
        # Preserve the declared lexical namespace of valid offload symlinks.
        path = Path(path).absolute()
        external = root not in path.parents
        if '..' in path.parts or (external and not readonly_roots):
            raise ValueError('provider consumed file escaped its declared namespace: ' + str(path))
        if external:
            _readonly_file(path, readonly_roots, namespace=root)
        digest = sha256_file_hex_cached(path)
        if expected is not None and digest != expected:
            raise ValueError('provider consumed dependency hash changed: ' + str(path))
        item = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
        destination = assets if external else files
        if external:
            item['kind'] = 'input-file'
        if str(path) in destination and destination[str(path)] != item:
            raise ValueError('provider consumed dependency changed while enumerating')
        destination[str(path)] = item
        return path

    def bound(value):
        path = add(value['path'], value['sha256'])
        if 'bytes' in value and path.stat().st_size != value['bytes']:
            raise ValueError('provider consumed dependency size changed')
        return path

    def ce_policy_input(reference):
        from natlang_neuralese.train.native_supervision import ce_policy_dependencies
        for item in ce_policy_dependencies(reference):
            bound(item)

    def classification_input(reference):
        from natlang_neuralese.train.native_classification_continuation import classification_proof_dependencies
        path = bound(reference) if isinstance(reference, dict) else add(reference)
        for item in classification_proof_dependencies({'path':str(path),'sha256':sha256_file_hex_cached(path)}):
            bound(item)

    def training_inputs(training):
        if not isinstance(training, dict):
            return
        policy_sources(training.get('policy_source_history', []))
        history = training.get('config_history') or []
        supervision = (history[-1].get('config') or {}).get('ce_supervision_policy') if history else None
        if supervision is not None:
            ce_policy_input(supervision)
        # A shared participant carries the selected next-SFT policy outside
        # historical per-stage frames, alongside its exact optimizer/RNG.
        supervision = (training.get('training_options') or training.get('options') or {}).get('ce_supervision_policy_binding')
        if supervision is not None:
            ce_policy_input(supervision)
        if not training.get('sft_progress') and training.get('schema') != 'natlang.neuralese-native-sft-training/1':
            return
        from natlang_neuralese.train.native_continuation import collect_sft_progress, sft_frame_complete
        progress = collect_sft_progress(training, borrow_frames=True)
        for name, frame in progress.items():
            supervision = (frame.get('options') or {}).get('ce_supervision_policy_binding')
            if supervision is not None:
                ce_policy_input(supervision)
            if sft_frame_complete(frame):
                continue  # Completed historical source files are not read by mandatory SFT.
            selected_classification = (frame.get('options') or {}).get('source_classification_migration')
            if selected_classification is not None:classification_input(selected_classification)
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
        declared = Path(value['path'] if isinstance(value, dict) else value).absolute()
        if '..' in declared.parts or root not in declared.parents:
            raise ValueError('provider checkpoint escaped its declared namespace')
        path = bound(value) if isinstance(value, dict) else add(value)
        selected=request.get('native_participant') if request.get('kind')=='round' else None
        if selected is None and isinstance(value,dict):selected=value.get('participant')
        context=(selected,semantic_only,tuple(sorted((type(key).__name__,str(key),str(value)) for key,value in relocations.items())))
        marker=(str(path.resolve()),files[str(path)]['sha256'],context)
        if marker in consumed_checkpoints:return
        consumed_checkpoints.add(marker)
        control = checkpoint_controls.get(files[str(path)]['sha256'])
        if control is not None and selected is None and not semantic_only:
            from natlang_neuralese.model.native_checkpoint_control import read_checkpoint_control, checkpoint_control_dependencies
            metadata = read_checkpoint_control(path, control)
            # Different immutable verifier revisions conservatively take the
            # unchanged full path. Corrupt explicitly bound artifacts reject.
            if (metadata['dependency_producer']['sha256'] == sha256_file_hex_cached(__file__)
                    and metadata['dependency_context'] == {'namespace': str(root),
                        'readonly_input_roots': sorted(str(item) for item in readonly_roots)}):
                for reference in checkpoint_control_dependencies(control, checkpoint=path):
                    bound(reference)
                for reference in metadata['dependencies']['files']:
                    bound(reference)
                for item in metadata['dependencies']['assets']:
                    if item['kind'] == 'input-file':
                        bound(item)
                    elif item['kind'] == 'base':
                        target = relocations.get(None, item['path'])
                        asset(target, base=True, expected=item['inventory'])
                    elif item['kind'] == 'task-runtime':
                        asset(item['path'], expected=item['inventory'])
                    else:
                        raise ValueError('checkpoint control has an unknown consumed asset kind')
                return
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
            if '..' in child.parts or root not in child.absolute().parents:
                raise ValueError('provider checkpoint dependency escaped its declared namespace')
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
                checkpoint_payload(state, selected)
                del state
            decoded.pop(str(child.resolve()),None)

    def checkpoint_payload(state, selected=None):
        # This exact traversal is shared by ordinary full decoding and the
        # opt-in derivative producer's already-validated in-process object.
        training_inputs(state)
        training_inputs(state.get('training'))
        training_inputs(state.get('shared_training'))
        for participant in state.get('participants', {}).values():
            training_inputs(participant)
        configs = [(state.get('participant', selected), state.get('model_config'))]
        configs.extend((name, item.get('model_config')) for name, item in state.get('participants', {}).items()
                       if selected is None or name == selected)
        for name, declaration in configs:
            if declaration is not None:
                asset(relocations.get(name, declaration['base']), base=True, expected=declaration['base_inventory'])

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
        adaptation=value.get('completed_supervised_adaptation')
        if adaptation is not None:
            if request.get('kind') == 'round' and value.get('mandatory_sft_only') is True:
                # production.run dispatches this directly to supervised.execute.
                # Its inherited controller adaptation is provenance, not an
                # instruction to replay that old invocation under current defaults.
                bound(adaptation)
            else:
                from natlang_neuralese.rl.native_completed_supervised import verify,dependencies as completed_dependencies
                proof=verify(adaptation)
                for reference in completed_dependencies(adaptation).values():bound(reference)
                config_value(json.loads(bound(proof['future_config']).read_text()),
                             'completed-future:'+adaptation['sha256'])
        controls=value.get('native_checkpoint_controls', {})
        if not isinstance(controls,dict):
            raise ValueError('native checkpoint controls must map full source SHA256 to explicit bindings')
        import re
        for digest, reference in controls.items():
            if not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None:
                raise ValueError('checkpoint control selector needs an exact full source SHA256')
            if digest in checkpoint_controls and checkpoint_controls[digest]!=reference:
                raise ValueError('checkpoint control selector has conflicting bound artifacts')
            checkpoint_controls[digest]=reference
        if value.get('base'):
            relocations.setdefault(value.get('native_participant'),value['base'])
        for name,declaration in value.get('models',{}).items():
            if declaration.get('base'):
                relocations.setdefault(name,declaration['base'])
        policy_sources((value.get('policy_training') or {}).get('sources', []))
        # Exact fields consumed by production.run/native semantic refresh.
        for reference in value.get('operational_inputs', []):
            bound(reference)
        classification = value.get('source_classification_migration')
        if classification is not None:classification_input(classification)
        for parameters in (value.get('trainer_parameters') or {}, value.get('seed_trainer_parameters') or {}):
            selected = parameters.get('source_classification_migration')
            if selected is not None:classification_input(selected)
        for command_name in ('seed_command','learner_command'):
            command=value.get(command_name) or []
            for index,argument in enumerate(command):
                if argument=='--source-classification-migration':
                    if index+1>=len(command):raise ValueError('provider classification command omitted its value')
                    classification_input(command[index+1])
                elif argument.startswith('--source-classification-migration='):
                    classification_input(argument.split('=',1)[1])
        qualification=value.get('foundation_qualification_request')
        supervision=value.get('ce_supervision_policy')
        if supervision is not None:
            ce_policy_input(supervision)
        for parameters in (value.get('trainer_parameters') or {}, value.get('seed_trainer_parameters') or {}):
            selected = parameters.get('ce_supervision_policy')
            if selected is not None and (supervision is None or selected != supervision['path']):
                raise ValueError('provider CE policy needs the exact bound supervision declaration')
        if qualification is not None:
            if not isinstance(qualification,dict):
                raise ValueError('provider foundation qualification request needs its exact binding')
            declared=json.loads(bound(qualification).read_text())
            if declared.get('schema')!='natlang.native-foundation-qualification-request/1':
                raise ValueError('provider foundation qualification request schema differs')
            for parameters in (value.get('trainer_parameters') or {},value.get('seed_trainer_parameters') or {}):
                if parameters.get('foundation_qualification_request') not in (None,qualification['path']):
                    raise ValueError('provider trainer qualification request differs from its bound input')
        elif any(parameters.get('foundation_qualification_request') for parameters in
                (value.get('trainer_parameters') or {},value.get('seed_trainer_parameters') or {})):
            raise ValueError('provider trainer qualification request lacks its exact bound input')
        adoption=value.get('snapshot_adoption')
        if adoption is not None:
            if not isinstance(adoption,dict) or adoption.get('schema')!='natlang.native-snapshot-adoption/1':
                raise ValueError('provider snapshot adoption schema differs')
            plan=json.loads(bound(adoption['plan']).read_text())
            from natlang_neuralese.rl.native_role_release import ownership,SCHEMA as role_schema
            _,exchange=ownership(plan)
            if exchange is None:raise ValueError('provider snapshot adoption needs its source-host exchange')
            for reference in adoption['releases']:
                proof=json.loads(bound(reference).read_text());role=proof.get('role')
                if (proof.get('schema')!=role_schema or role not in exchange['roles']
                        or proof.get('kind')!='released' or proof.get('optimizer_released') is not True):
                    raise ValueError('provider snapshot adoption requires an actual source-host release')
                current=json.loads(add(Path(exchange['directory'])/(role+'-status.json')).read_text())
                if current['sha256']!=reference['sha256']:
                    raise ValueError('provider snapshot adoption refers to a retired role release')
                bound(current)
        if value.get('native_actor_view'):
            from natlang_neuralese.model.native_bundle import actor_view_dependencies
            for reference in actor_view_dependencies(value['native_actor_view']):
                bound(reference)
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
    for reference in request.get('semantic_participants',[]):
        checkpoint(reference,semantic_only=True)
    if request.get('semantic_adapter'):
        semantic_adapter(request['semantic_adapter'])
    if _checkpoint_payload is not None:
        if request:
            raise ValueError('in-process checkpoint payload is a producer-only dependency request')
        checkpoint_payload(_checkpoint_payload)
    return {'files': list(files.values()), 'assets': list(assets.values())}


def verify_assets(assets, *, readonly_input_roots=(), namespace='/work/native-production'):
    """Verify pre-staged external assets; this never accepts path existence alone."""
    roots = _readonly_roots(readonly_input_roots)
    for item in assets:
        path = Path(item['path'])
        if item['kind'] == 'input-file':
            _readonly_file(path, roots, namespace=Path(namespace).absolute())
            if (type(item.get('bytes')) is not int or item['bytes'] < 0
                    or path.stat().st_size != item['bytes']
                    or sha256_file_hex_cached(path) != item['sha256']):
                raise ValueError('provider pre-staged input file content differs: ' + str(path))
            continue
        if item['kind'] == 'base':
            from natlang_neuralese.model.native_model import base_inventory
            actual = base_inventory(path)
        elif item['kind'] == 'task-runtime':
            actual = task_runtime_inventory(path)
        else:
            raise ValueError('unknown provider pre-staged asset kind')
        if actual != item['inventory'] or canonical_json_sha256_hex(actual) != item['sha256']:
            raise ValueError('provider pre-staged asset content differs: ' + str(path))
