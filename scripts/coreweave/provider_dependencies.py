"""Exact consumed-input closure for the external provider file bridge.

This enumerates already-declared inputs, never chooses a training source. Public
base directories, built task runtimes and explicitly declared read-only inputs
are verified in place; ferry files stay in the native-production namespace.
"""
from __future__ import annotations
import json
import os
import hashlib
import tarfile
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from natlang_neuralese.common.runtime_inventory import task_runtime_inventory
from natlang_neuralese.model.native_bundle import composite_dependencies
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


def source_task_baseline_dependencies(training):
    """Verify and enumerate the source-weight TaskEval closure in a shared state."""
    rows=training.get('source_task_baselines') if isinstance(training,dict) else None
    if rows is None:return []
    if not isinstance(rows,dict):raise ValueError('shared source task baselines must map participant IDs to proofs')
    from natlang_neuralese.eval.native_state import validate_task_execution_receipt
    references=[]
    for participant,baseline in rows.items():
        if not isinstance(participant,str)or not isinstance(baseline,dict):
            raise ValueError('shared source task baseline entry is malformed')
        identity=baseline.get('native_identity');task_ref=baseline.get('task_execution_receipt')
        if not isinstance(identity,dict)or not isinstance(task_ref,dict):
            raise ValueError('shared source task baseline omitted its exact source receipt')
        task_path=Path(task_ref.get('path','')).resolve(strict=True)
        if sha256_file_hex_cached(task_path)!=task_ref.get('sha256'):
            raise ValueError('shared source task baseline receipt bytes changed')
        task=json.loads(task_path.read_text())
        if (not validate_task_execution_receipt(task,identity)
                or task.get('source_baseline') is not True
                or task.get('baseline_comparison_floor')!=0
                or task.get('native_identity')!=identity
                or task.get('task_successes')!=baseline.get('task_successes')
                or task.get('task_eval_policy_sha256')!=baseline.get('task_eval_policy_sha256')):
            raise ValueError('shared source task baseline receipt no longer matches its participant')
        score=baseline.get('score')
        if (not isinstance(score,dict)or task.get('score_path')!=score.get('path')
                or task.get('score_sha256')!=score.get('sha256')):
            raise ValueError('shared source task baseline score binding changed')
        runtime=baseline.get('task_runtime_identity')
        from natlang_neuralese.common.runtime_inventory import task_runtime_identity_files
        if not isinstance(runtime,dict) or task.get('task_runtime_identity')!=runtime:
            raise ValueError('shared source task baseline omitted its evaluator runtime')
        references.extend((task_ref,score))
        references.extend(task_runtime_identity_files(runtime))
    unique={}
    for reference in references:
        key=(reference.get('path'),reference.get('sha256'))
        if not all(isinstance(value,str)and value for value in key):
            raise ValueError('shared source task dependency lacks its exact binding')
        if reference.get('bytes') is not None and type(reference['bytes'])is not int:
            raise ValueError('shared source task dependency has an invalid byte count')
        unique[key]=reference
    return list(unique.values())


def dependency_closure(request, *, namespace='/work/native-production', readonly_input_roots=(),
                       _checkpoint_payload=None):
    files, assets, active_configs, relocations = {}, {}, set(), {}
    consumed_checkpoints=set()
    consumed_payloads=set()
    checkpoint_controls={}
    root = Path(namespace).absolute()
    readonly_roots = _readonly_roots(readonly_input_roots)

    def add(path, expected=None):
        # Preserve the declared lexical namespace of valid offload symlinks.
        path = Path(path).absolute()
        if '..' in path.parts:
            normalized = Path(os.path.normpath(path))
            if root not in normalized.parents and not readonly_roots:
                raise ValueError('provider consumed file escaped its declared namespace: ' + str(path))
            if path.resolve(strict=True) != normalized.resolve(strict=True):
                raise ValueError('provider dependency traversal changes its symlink target: ' + str(path))
            path = normalized
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

    def external_source_provenance(options):
        """Bind compact historical evidence without demanding offbox raw files."""
        from natlang_neuralese.train.recipe import external_source_provenances
        for provenance in external_source_provenances(options.get('recipe_binding')):
            if provenance.get('schema') != 'natlang.external-source-publication-provenance/1':
                raise ValueError('external source publication provenance schema differs')
            closure = json.loads(bound(provenance['closure']).read_text())
            schema = closure.get('schema')
            if closure != provenance.get('custody') or schema not in ('natlang.big2-tranche-custody-publication-closure/1', 'natlang.big2-multi-tranche-custody-closure/1'):
                raise ValueError('external source custody differs from its exact closure')
            def historical_identity(item):
                return (isinstance(item, dict) and isinstance(item.get('path'), str) and item['path'].startswith('/')
                        and isinstance(item.get('sha256'), str) and len(item['sha256']) == 64
                        and all(c in '0123456789abcdef' for c in item['sha256']))
            if not historical_identity(closure.get('archive')):
                raise ValueError('external source historical identity is malformed')
            if schema == 'natlang.big2-tranche-custody-publication-closure/1':
                if not historical_identity(closure.get('recovered_raw_records')):
                    raise ValueError('external source historical identity is malformed')
            else:
                proofs = [json.loads(bound(ref).read_text()) for ref in closure['historical_proofs']]
                subset = next((p for p in proofs if p.get('schema') == 'natlang.big2-multi-tranche-archive-subset-proof/1'), None)
                if not subset or subset.get('archive') != closure['archive'] or not closure.get('tranches'):
                    raise ValueError('multi-tranche archive subset proof differs')
                for name, tranche in closure['tranches'].items():
                    raw = tranche.get('raw_subset')
                    if raw != subset.get('tranches', {}).get(name) or not isinstance(raw.get('records_sha256'), str) or len(raw['records_sha256']) != 64 or any(c not in '0123456789abcdef' for c in raw['records_sha256']):
                        raise ValueError('multi-tranche raw subset identity differs')
                    if type(raw.get('records_bytes')) is not int or raw['records_bytes'] <= 0 or type(raw.get('records_rows')) is not int or raw['records_rows'] <= 0:
                        raise ValueError('multi-tranche raw subset size differs')
                    bound(raw['extract_receipt'])
            archive = bound(provenance['mechanisms_archive'])
            bound(provenance['content_audit']); bound(provenance['clean_closure_audit'])
            members = {'verify.py': closure['verifier']['sha256'],
                       'student_tokenizers.py': closure['tokenizer_helper']['sha256'],
                       'runtime.tar': closure['runtime_snapshot']['sha256'],
                       'closure.json': provenance['closure']['sha256']}
            if schema == 'natlang.big2-tranche-custody-publication-closure/1':
                members['runtime-receipt.json'] = closure['runtime_snapshot_receipt']['sha256']
            members.update({f'proofs/{index}.json': proof['sha256']
                            for index, proof in enumerate(closure['historical_proofs'])})
            with tarfile.open(archive, 'r:*') as tar:
                entries = tar.getmembers()
                if any(not item.isfile() for item in entries) or len(entries) != len(members) or {item.name for item in entries} != set(members):
                    raise ValueError('external source mechanism archive inventory differs')
                for item in entries:
                    if hashlib.file_digest(tar.extractfile(item), 'sha256').hexdigest() != members[item.name]:
                        raise ValueError('external source mechanism archive member differs: ' + item.name)

    def classification_input(reference):
        from natlang_neuralese.train.native_classification_continuation import classification_proof_dependencies
        path = bound(reference) if isinstance(reference, dict) else add(reference)
        for item in classification_proof_dependencies({'path':str(path),'sha256':sha256_file_hex_cached(path)}):
            bound(item)

    def packed_capacity_inputs(options):
        if not isinstance(options, dict) or options.get('sft_packed_capacity') is None:
            return
        receipt = options['sft_packed_capacity']
        if not isinstance(receipt, dict): raise ValueError('packed capacity receipt is malformed')
        geometry = bound(receipt['geometry'])
        proof = bound(receipt['proof'])
        geom = json.loads(geometry.read_text()); evidence = json.loads(proof.read_text())
        if (geom.get('schema') != 'natlang.native-packed-geometry/1'
                or evidence.get('schema') != 'natlang.native-sft-packed-capacity/1'
                or evidence.get('geometry') != receipt['geometry'] or evidence.get('passed') is not True
                or not isinstance(evidence.get('capacity_identity'), dict)
                or not isinstance(evidence.get('runtime_code_sha256'), str) or len(evidence['runtime_code_sha256']) != 64
                or evidence.get('optimizer_updates') != 0 or evidence.get('earned_exposure') != 0):
            raise ValueError('packed capacity proof/geometry bindings differ')
        bound(geom['source']); bound(evidence['source'])

    def packed_geometry_input(reference):
        if isinstance(reference, str):
            try: reference = json.loads(reference)
            except ValueError as error: raise ValueError('packed geometry requires a JSON file binding') from error
        if not isinstance(reference, dict): raise ValueError('packed geometry requires an exact file binding')
        geometry = json.loads(bound(reference).read_text())
        if geometry.get('schema') != 'natlang.native-packed-geometry/1':
            raise ValueError('packed geometry source schema differs')
        bound(geometry['source'])

    def foundation_inputs(options):
        if not isinstance(options, dict):
            return
        header = options.get('foundation_native_graphs_binding')
        recovery = options.get('foundation_declaration_recovery_binding')
        active = bool(options.get('communication_foundation'))
        selected_header = options.get('foundation_native_graphs')
        selected_recovery = options.get('foundation_declaration_recovery')
        if active and not isinstance(header, dict):
            raise ValueError('active communication foundation lacks its exact source-header binding')
        if active and not selected_header:
            raise ValueError('active communication foundation lacks its selected source-header path')
        if selected_header and not isinstance(header, dict):
            raise ValueError('selected foundation source lacks its exact header binding')
        if selected_recovery and not isinstance(recovery, dict):
            raise ValueError('selected foundation recovery lacks its exact proof binding')
        if selected_header:
            if Path(selected_header).resolve() != Path(header['path']).resolve():
                raise ValueError('selected foundation source and header binding paths differ')
            from natlang_neuralese.train.native_foundation_recovery import native_graph_bindings
            actual = native_graph_bindings(header)
            expected = header.get('dependencies')
            if not isinstance(expected, list):
                raise ValueError('foundation source header lacks its exact dependency inventory')
            actual_dependencies = [item for item in actual if item['path'] != str(Path(header['path']).resolve())]
            if actual_dependencies != expected:
                raise ValueError('foundation source dependency inventory changed')
            for reference in actual:
                bound(reference)
        if selected_recovery:
            if Path(selected_recovery).resolve() != Path(recovery['path']).resolve():
                raise ValueError('selected foundation recovery and proof binding paths differ')
            from natlang_neuralese.train.native_foundation_recovery import dependency_bindings
            for reference in dependency_bindings(recovery):
                bound(reference)

    def retired_source_predecessor_inputs(options):
        if not isinstance(options, dict):
            return
        from natlang_neuralese.train.native_predecessor_retirement import load_retirement
        for item in options.get('source_predecessor_retirement_history') or []:
            reference = {'path': item.get('proof_path'), 'sha256': item.get('proof_sha256')}
            path = bound(reference)
            proof = load_retirement(path)
            if any(item.get(key) != value for key, value in proof.items()):
                raise ValueError('retired source predecessor proof history differs')

    def source_correction_inputs(options):
        if not isinstance(options, dict):
            return
        history = options.get('source_correction_history') or []
        selected = options.get('source_correction_binding')
        if selected is not None and (not isinstance(selected, dict)
                or not isinstance(selected.get('path'), str) or not selected['path']
                or not isinstance(selected.get('sha256'), str) or not selected['sha256']):
            raise ValueError('selected native source correction lacks its exact proof binding')
        if selected is not None and not any(
                item.get('proof_sha256') == selected.get('sha256')
                and Path(item.get('proof_path', '')).resolve() == Path(selected.get('path', '')).resolve()
                for item in history):
            raise ValueError('selected native source correction is absent from its history')
        from natlang_neuralese.train.native_source_correction import source_correction_dependencies
        for item in history:
            ref = {'path': item.get('proof_path'), 'sha256': item.get('proof_sha256')}
            for dependency in source_correction_dependencies(ref):
                bound(dependency)

    def held_correction_inputs(options):
        if not isinstance(options, dict):
            return
        history = options.get('held_correction_history') or []
        selected = options.get('held_correction_binding')
        if not isinstance(history, list):
            raise ValueError('held correction history is malformed')
        if selected is not None and (not isinstance(selected, dict)
                or not isinstance(selected.get('path'), str)
                or not isinstance(selected.get('sha256'), str)):
            raise ValueError('selected held correction lacks its exact proof binding')
        if selected is not None and not any(
                item.get('proof_sha256') == selected.get('sha256')
                and Path(item.get('proof_path', '')).resolve() == Path(selected.get('path', '')).resolve()
                for item in history if isinstance(item, dict)):
            raise ValueError('selected held correction is absent from its history')
        if history and selected is None:
            raise ValueError('held correction history has no selected proof binding')
        from natlang_neuralese.train.native_source_correction import source_correction_dependencies
        for item in history:
            ref = {'path': item.get('proof_path'), 'sha256': item.get('proof_sha256')}
            for dependency in source_correction_dependencies(ref):
                bound(dependency)

    def focus_inputs(value):
        from natlang_neuralese.train.native_pending_focus import dependencies as focus_dependencies
        references = ([value['pending_focus']] if value.get('pending_focus') else [])
        references += [record['request'] for record in value.get('pending_focus_history', [])]
        for reference in references:
            if isinstance(reference, dict):
                path = bound(reference)
                reference = {'path': str(path), 'sha256': reference['sha256']}
            else:
                path = add(reference)
                reference = {'path': str(path), 'sha256': sha256_file_hex_cached(path)}
            for item in focus_dependencies(reference)[1]:
                bound(item)

    def training_inputs(training):
        if not isinstance(training, dict):
            return
        for options in (training.get('options') or {}, training.get('training_options') or {}):
            external_source_provenance(options)
            focus_inputs(options)
            transition = options.get('new_outer_reanchor')
            if transition is not None:
                if transition.get('schema') != 'natlang.diloco-new-outer-transition/1':
                    raise ValueError('invalid new outer recovery transition')
                first = transition.get('origin') == 'first'
                if first and transition.get('historical_outer_checkpoint') is not None:
                    raise ValueError('first outer has unexpected historical outer custody')
                for name in (('declaration', 'parent', 'seed') if first else ('declaration', 'parent', 'historical_outer_checkpoint', 'seed')):
                    reference = transition.get(name)
                    if not isinstance(reference, dict):
                        raise ValueError('new outer recovery lacks dependency ' + name)
                    bound(reference)
                if transition['declaration']['sha256'] != transition.get('declaration_sha256') or \
                        transition['seed']['sha256'] != transition.get('seed_sha256'):
                    raise ValueError('new outer recovery dependency binding differs')
                manifest = json.loads(Path(transition['declaration']['path']).read_text())
                if manifest.get('schema') != 'natlang.diloco-new-outer-manifest/1' or \
                        len(manifest.get('peers', [])) != manifest.get('workers'):
                    raise ValueError('invalid common new outer manifest')
                for index, reference in enumerate(manifest['peers']):
                    peer_path=bound(reference)
                    if first:
                        peer=json.loads(peer_path.read_text())
                        if (peer.get('schema')!='natlang.diloco-first-outer/1' or peer.get('worker')!=index
                                or peer.get('seed_sha256')!=transition['seed_sha256']
                                or peer.get('historical_outer_checkpoint') is not None):
                            raise ValueError('first outer declaration origin or seed differs')
                        if index==transition.get('worker') and peer.get('parent')!=transition['parent']:
                            raise ValueError('first outer parent differs from committed transition')
        foundation_inputs(training.get('options') or {})
        foundation_inputs(training.get('training_options') or {})
        packed_capacity_inputs(training.get('options') or {})
        packed_capacity_inputs(training.get('training_options') or {})
        retired_source_predecessor_inputs(training.get('options') or {})
        retired_source_predecessor_inputs(training.get('training_options') or {})
        source_correction_inputs(training.get('options') or {})
        source_correction_inputs(training.get('training_options') or {})
        held_correction_inputs(training.get('options') or {})
        held_correction_inputs(training.get('training_options') or {})
        for reference in source_task_baseline_dependencies(training):
            bound(reference)
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
            # Foundation declarations are checkpoint dependencies even for a
            # completed carried frame; inspect before the source-frame skip.
            foundation_inputs(frame.get('options') or {})
            packed_capacity_inputs(frame.get('options') or {})
            retired_source_predecessor_inputs(frame.get('options') or {})
            source_correction_inputs(frame.get('options') or {})
            held_correction_inputs(frame.get('options') or {})
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

    def checkpoint(value):
        declared = Path(value['path'] if isinstance(value, dict) else value).absolute()
        if '..' in declared.parts or root not in declared.parents:
            raise ValueError('provider checkpoint escaped its declared namespace')
        path = bound(value) if isinstance(value, dict) else add(value)
        selected=request.get('native_participant') if request.get('kind')=='round' else None
        if selected is None and isinstance(value,dict):selected=value.get('participant')
        context=(selected,tuple(sorted((type(key).__name__,str(key),str(value)) for key,value in relocations.items())))
        marker=(str(path.resolve()),files[str(path)]['sha256'],context)
        if marker in consumed_checkpoints:return
        consumed_checkpoints.add(marker)
        control = checkpoint_controls.get(files[str(path)]['sha256'])
        if control is not None and selected is None:
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
                checkpoint_loader=load_once).items():
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
            raise ValueError('provider external input is not a native block bundle')
        if policy_id is not None and manifest.get('policy_id') != policy_id:
            raise ValueError('provider policy bundle differs from its behavior adapter checkpoint')
        from natlang_neuralese.rl.graph_replay import verified_bundle_path
        for ident, entry in manifest['blocks'].items():
            child = (manifest_path.parent/entry['path']).resolve(strict=True)
            if child.parent != manifest_path.parent.resolve():
                raise ValueError('provider bundle block escaped its bound bundle')
            bound({**entry, 'path': str(child)})
            verified_bundle_path(manifest_path, ident, policy_id=manifest.get('policy_id'), manifest=manifest)

    def config_value(value, marker, *, inherited_completed_provenance=False):
        if marker in active_configs:
            raise ValueError("provider consumed configuration contains a cycle")
        active_configs.add(marker)
        adaptation=value.get('completed_supervised_adaptation')
        if adaptation is not None:
            if inherited_completed_provenance or (request.get('kind') == 'round' and value.get('mandatory_sft_only') is True):
                # A verified adaptation's future configuration carries prior
                # controller provenance. The actual current adaptation was
                # verified by its caller; this old marker is not another active
                # invocation under today's defaults. Mandatory SFT likewise
                # dispatches directly to supervised.execute.
                bound(adaptation)
            else:
                from natlang_neuralese.rl import native_completed_supervised as completed
                paired=getattr(completed,'verified_dependencies',None)
                if callable(paired):proof,completed_inputs=paired(adaptation)
                else:
                    # Immutable older runtime closures retain their original API.
                    proof=completed.verify(adaptation)
                    completed_inputs=completed.dependencies(adaptation)
                for reference in completed_inputs.values():bound(reference)
                config_value(json.loads(bound(proof['future_config']).read_text()),
                             'completed-future:'+adaptation['sha256'],
                             inherited_completed_provenance=True)
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
        # Exact fields consumed by production.run.
        for reference in value.get('operational_inputs', []):
            bound(reference)
        classification = value.get('source_classification_migration')
        if classification is not None:classification_input(classification)
        for parameters in (value.get('trainer_parameters') or {}, value.get('seed_trainer_parameters') or {}):
            selected = parameters.get('source_classification_migration')
            if selected is not None:classification_input(selected)
            selected_geometry = parameters.get('sft_packed_geometry')
            if selected_geometry is not None: packed_geometry_input(selected_geometry)
        for command_name in ('seed_command','learner_command'):
            command=value.get(command_name) or []
            for index,argument in enumerate(command):
                if argument=='--source-classification-migration':
                    if index+1>=len(command):raise ValueError('provider classification command omitted its value')
                    classification_input(command[index+1])
                elif argument.startswith('--source-classification-migration='):
                    classification_input(argument.split('=',1)[1])
                elif argument=='--sft-packed-geometry':
                    if index+1>=len(command):raise ValueError('provider packed geometry command omitted its value')
                    packed_geometry_input(command[index+1])
                elif argument.startswith('--sft-packed-geometry='):
                    packed_geometry_input(argument.split('=',1)[1])
        supervision=value.get('ce_supervision_policy')
        if supervision is not None:
            ce_policy_input(supervision)
        for parameters in (value.get('trainer_parameters') or {}, value.get('seed_trainer_parameters') or {}):
            selected = parameters.get('ce_supervision_policy')
            if selected is not None and (supervision is None or selected != supervision['path']):
                raise ValueError('provider CE policy needs the exact bound supervision declaration')
        if value.get('snapshot_adoption') is not None:
            raise ValueError('provider snapshot adoption belonged to the removed cross-model role exchange')
        if value.get('native_actor_view'):
            from natlang_neuralese.model.native_bundle import actor_view_dependencies
            for reference in actor_view_dependencies(value['native_actor_view']):
                bound(reference)
        focus_inputs(value)
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
                if key == 'production_recipe':
                    recipe = json.loads(consumed.read_text())
                    for stage in recipe.get('stages', []):
                        selected = (stage.get('parameters') or {}).get('sft_packed_geometry')
                        if selected is not None: packed_geometry_input(selected)
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
    if request.get('config'): config(bound(request['config']))
    for key in ('parent_checkpoint', 'partial_checkpoint'):
        if request.get(key): checkpoint(request[key])
    if request.get('state'):
        state = json.loads(bound(request['state']).read_text())
        for key in ('checkpoint', 'pending_native_checkpoint', 'partial_checkpoint'):
            if state.get(key): checkpoint(state[key])
        progress=state.get('last_learner_progress') or {}
        if progress.get('full_report'):bound(progress['full_report'])
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
        if item['kind'] == 'task-source-file':
            if (type(item.get('bytes')) is not int or item['bytes'] < 0
                    or not path.is_absolute() or '..' in path.parts):
                raise ValueError('provider TaskEval source file has no exact safe binding')
            cursor=path
            while cursor!=cursor.parent:
                if cursor.is_symlink():raise ValueError('provider TaskEval source traverses a symlink')
                cursor=cursor.parent
            if (not path.is_file() or path.stat().st_size!=item['bytes']
                    or sha256_file_hex_cached(path)!=item['sha256']):
                raise ValueError('provider TaskEval source file differs from its exact runtime binding: '+str(path))
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
