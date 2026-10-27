#!/usr/bin/env python3
"""Prepare a privately verified teacher-source append for an existing native SFT stage.

This writes a future configuration and exact source/CE/capacity provenance. It
does not update the active config, start a trainer, or grant row exposure.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time


DESCRIPTOR_SCHEMA = 'natlang.private-teacher-append-descriptor/1'
APPEND_SCHEMA = 'natlang.private-teacher-append/1'
CENSUS_SCHEMA = 'natlang.private-ready-source-census/1'
ROW_SELECTION_SCHEMA = 'natlang.exact-native-row-selection/1'
SOURCE_EQUALITY_SCHEMA = 'natlang.exact-native-source-equality/1'
SUPPORTED_MODEL_ROLES = frozenset({'lfm25', 'mellum21'})
# New, independently verified teacher rows enter the existing urgent/debt
# scheduler; this is scheduling metadata, not an exposure credit.
NEW_TEACHER_TRAINING_PRIORITY = 1


def _file_binding(path: Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    path_after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, field) != getattr(after, field)
           or getattr(after, field) != getattr(path_after, field) for field in fields):
        raise ValueError(f'file changed while calculating exact binding: {path}')
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest(), 'bytes': after.st_size}


def _json_binding(path: Path) -> dict:
    value = _file_binding(path)
    return {'path': value['path'], 'sha256': value['sha256']}


def _copy_binding(binding: dict) -> dict:
    return {'path': binding['path'], 'sha256': binding['sha256']}


def _seed_source_bindings(resolved: dict, *, train: dict, held: dict, ce: dict) -> dict:
    """Retain every resolved optional source role while rebinding only train and CE."""
    result = copy.deepcopy(resolved)
    result.update({'native_graphs': train, 'held_native_graphs': held,
                   'ce_supervision_policy': ce})
    return result


def _path_in(path: Path, root: Path, label: str) -> Path:
    path = Path(path).resolve(strict=True)
    root = Path(root).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'{label} must remain inside its declared root: {path}') from exc
    return path


def _descriptor(value: dict) -> dict:
    if not isinstance(value, dict) or value.get('schema') != DESCRIPTOR_SCHEMA:
        raise ValueError(f'descriptor schema must be {DESCRIPTOR_SCHEMA}')
    required = {'schema', 'runtime_root', 'artifact_repo', 'supervision',
                'ready_export_root', 'model_role', 'output_root', 'stage_id', 'append_key'}
    allowed = required | {'private_receipt_custody'}
    if set(value) not in (required, allowed):
        raise ValueError('descriptor fields must be required fields plus optional private_receipt_custody')
    if 'private_receipt_custody' not in value:
        value['private_receipt_custody'] = None
    custody = value['private_receipt_custody']
    if custody is not None:
        if not isinstance(custody, dict) or set(custody) != {'source_map', 'transfer_manifest'}:
            raise ValueError('private_receipt_custody must bind staged-source-map and transfer manifest')
        for name, ref in custody.items():
            if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
                    or not isinstance(ref['path'], str)
                    or not re.fullmatch(r'[0-9a-f]{64}', str(ref['sha256']))):
                raise ValueError(f'private_receipt_custody.{name} must be an exact path/sha256 binding')
    if not isinstance(value['artifact_repo'], dict) or set(value['artifact_repo']) != {
            'root', 'registry_path', 'manifests_root'}:
        raise ValueError('artifact_repo must bind root, registry_path and manifests_root')
    if not isinstance(value['supervision'], dict) or set(value['supervision']) != {
            'root', 'config_path', 'plan_path'}:
        raise ValueError('supervision must bind root, config_path and plan_path')
    if value['model_role'] not in SUPPORTED_MODEL_ROLES:
        raise ValueError('model_role must be lfm25 or mellum21')
    for key in ('stage_id', 'append_key'):
        if not isinstance(value[key], str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', value[key]):
            raise ValueError(f'{key} must be a safe lowercase identifier')
    return value


def _load_runtime(runtime_root: Path):
    """Import production readers only from the descriptor's frozen source tree."""
    runtime_root = Path(runtime_root).resolve(strict=True)
    package_root = runtime_root / 'training' / 'neuralese'
    coreweave_root = runtime_root / 'scripts' / 'coreweave'
    if not (package_root / 'natlang_neuralese').is_dir() or not (coreweave_root / 'private_export_guard.py').is_file():
        raise ValueError('runtime_root lacks frozen Neuralese modules or private export verifier')
    for path in (runtime_root, package_root, coreweave_root):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

    required_modules = (
        'natlang_neuralese.data.native_sources',
        'natlang_neuralese.common.jsonio',
        'natlang_neuralese.train.native_resident',
        'natlang_neuralese.train.native_supervision',
        'natlang_neuralese.train.recipe',
        'natlang_neuralese.serve.chat',
        'natlang_neuralese.rl.native_supervised',
        'scripts.neuralese_data.build_native_ce_supervision_policy',
        'private_export_guard',
    )
    modules = {name: importlib.import_module(name) for name in required_modules}
    for name, module in modules.items():
        module_path = Path(module.__file__).resolve()
        if name == 'private_export_guard':
            expected_root = coreweave_root.resolve()
        else:
            expected_root = runtime_root
        try:
            module_path.relative_to(expected_root)
        except ValueError as exc:
            raise RuntimeError(f'{name} was not loaded from frozen runtime_root: {module_path}') from exc
    return modules


def _load_json(path: Path):
    return json.loads(Path(path).read_text())


def _manifest_file(manifest: dict, name: str) -> dict:
    matches = [item for item in manifest.get('files', []) if item.get('path') == name]
    if len(matches) != 1:
        raise ValueError(f'manifest must have exactly one {name!r} file binding')
    return matches[0]


def _registered_package_root(entry_path: str, artifact_root: Path) -> Path:
    """Resolve a relative registry alias while allowing its target storage outside the repo.

    The registry path itself must stay inside the artifact namespace and cannot
    traverse with ``..``. Its final symlink target defines the authenticated
    package root; every role member is then constrained to that resolved root.
    """
    if not isinstance(entry_path, str) or not entry_path:
        raise ValueError('registered source package path must be a nontraversing artifact-relative path')
    relative = Path(entry_path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('registered source package path must be a nontraversing artifact-relative path')
    lexical = Path(os.path.abspath(Path(artifact_root) / relative))
    root = Path(artifact_root).resolve(strict=True)
    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError('registered source package alias escapes the artifact namespace') from exc
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError('registered source package target is not a directory')
    return resolved


def _verify_source_package(*, ready_path: Path, ready: dict, registry: dict,
                           artifact_root: Path, registry_path: Path, manifests_root: Path,
                           role: str, expected_tokenizer: str, expected_history: str | None,
                           expected_failure_policy: str | None, runtime: dict,
                           ready_root: Path, receipt_relocations: dict,
                           relocation_map_binding: dict | None):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from private_export_guard import validate_receipt

    corpus_id = ready.get('corpus_id')
    if not isinstance(corpus_id, str) or not corpus_id:
        raise ValueError(f'{ready_path}: missing corpus_id')
    entries = [entry for entry in registry.get('corpora', []) if entry.get('id') == corpus_id]
    if len(entries) != 1:
        raise ValueError(f'{corpus_id}: registry must contain exactly one corpus entry')
    entry = entries[0]
    if entry.get('training_admission') is not True:
        raise ValueError(f'{corpus_id}: corpus is not training-admitted')
    if canonical_json_sha256_hex(entry) != ready.get('entry_sha256'):
        raise ValueError(f'{corpus_id}: ready receipt registry digest differs from current admitted entry')

    manifest_path = Path(manifests_root) / f'{corpus_id}.json'
    manifest_binding = _file_binding(manifest_path)
    if manifest_binding['sha256'] != ready.get('manifest_sha256'):
        raise ValueError(f'{corpus_id}: current manifest digest differs from ready receipt')
    manifest = _load_json(manifest_path)
    package_root = _registered_package_root(entry.get('path'), artifact_root)
    source_rel = f'{role}/train/native-graphs.jsonl.zst'
    source_path = _path_in(package_root / source_rel, package_root, 'registered graph leaf')
    source_member = _manifest_file(manifest, source_rel)
    source_binding = _file_binding(source_path)
    if any(source_member.get(key) != source_binding.get(key) for key in ('sha256', 'bytes')):
        raise ValueError(f'{corpus_id}: graph leaf does not match the manifest')

    role_receipt_rel = f'{role}/receipt.json'
    role_receipt_path = _path_in(package_root / role_receipt_rel, package_root, 'role receipt')
    role_receipt_member = _manifest_file(manifest, role_receipt_rel)
    role_receipt_binding = _file_binding(role_receipt_path)
    if any(role_receipt_member.get(key) != role_receipt_binding.get(key) for key in ('sha256', 'bytes')):
        raise ValueError(f'{corpus_id}: role receipt does not match the manifest')
    role_receipt = _load_json(role_receipt_path)
    if role_receipt.get('model_key') != role:
        raise ValueError(f'{corpus_id}: role receipt model_key differs from selected stage role {role}')
    if role_receipt.get('tokenizer', {}).get('expanded_fingerprint') != expected_tokenizer:
        raise ValueError(f'{corpus_id}: source tokenizer differs from selected model tokenizer')
    tokenizer_history = role_receipt.get('tokenizer', {}).get('history_reasoning')
    explicit_history = role_receipt.get('explicit_history_reasoning_policy')
    if role == 'lfm25' and explicit_history != 'keep':
        raise ValueError(f'{corpus_id}: LFM role requires explicit history_reasoning_policy=keep')
    recorded_history = explicit_history if isinstance(explicit_history, str) and explicit_history else None
    history_probe = None
    if isinstance(tokenizer_history, dict):
        nested_policy = tokenizer_history.get('policy')
        history_probe = tokenizer_history.get('probe')
        if isinstance(nested_policy, str) and nested_policy and recorded_history is not None and nested_policy != recorded_history:
            raise ValueError(f'{corpus_id}: explicit and tokenizer history policies contradict')
        if recorded_history is None:
            recorded_history = nested_policy if isinstance(nested_policy, str) else None
    if recorded_history is None:
        tokenizer_info = role_receipt.get('tokenizer', {})
        model_type = tokenizer_info.get('backbone') or tokenizer_info.get('model_type')
        if not isinstance(model_type, str) or not model_type:
            model_type = 'mellum' if role == 'mellum21' else 'lfm2'
        recorded_history = runtime['natlang_neuralese.serve.chat'].declared_history_reasoning(model_type)
    if recorded_history != expected_history:
        raise ValueError(f'{corpus_id}: history policy {recorded_history!r} differs from selected model/stage {expected_history!r}')
    if history_probe is not None:
        expected_probe = runtime['natlang_neuralese.serve.chat']._PROBE_EXPECTED.get(recorded_history)
        if expected_probe is None or history_probe != expected_probe:
            raise ValueError(f'{corpus_id}: tokenizer history probe contradicts effective history policy {recorded_history!r}')
    failure_policy = role_receipt.get('failed_assistant_history_policy')
    if failure_policy not in ('preserve', 'mask-runtime-failures'):
        raise ValueError(f'{corpus_id}: unsupported failed assistant history policy')
    if role == 'lfm25' and failure_policy != 'mask-runtime-failures':
        raise ValueError(f'{corpus_id}: LFM role requires mask-runtime-failures')
    if expected_failure_policy is not None and failure_policy != expected_failure_policy:
        raise ValueError(f'{corpus_id}: failed assistant history policy changed across this append')

    declared_private_path = ready.get('private_receipt')
    if not isinstance(declared_private_path, str) or not declared_private_path:
        raise ValueError(f'{corpus_id}: ready record lacks its original private receipt path')
    declared_path = Path(declared_private_path)
    if not declared_path.is_absolute():
        declared_path = Path(ready_path).parent / declared_path
    relocation = receipt_relocations.get(corpus_id)
    if relocation is not None:
        relocation_declared = Path(relocation['declared_path'])
        if not relocation_declared.is_absolute():
            relocation_declared = Path(ready_path).parent / relocation_declared
        if declared_path.resolve() != relocation_declared.resolve():
            raise ValueError(f'{corpus_id}: relocation does not name the original ready.json private receipt path')
        staged_receipt = Path(relocation['staged_path'])
        if not staged_receipt.is_absolute():
            staged_receipt = Path(ready_path).parent / staged_receipt
        private_path = _path_in(staged_receipt, ready_root, 'relocated private receipt')
        private_binding = _file_binding(private_path)
        if (relocation['original_sha256'] != relocation['staged_sha256']
                or private_binding['sha256'] != relocation['staged_sha256']
                or private_binding['sha256'] != relocation['original_sha256']
                or private_binding['bytes'] != relocation['bytes']):
            raise ValueError(f'{corpus_id}: relocated private receipt differs from its declared original custody binding')
    else:
        if relocation_map_binding is not None:
            raise ValueError(f'{corpus_id}: explicit receipt custody map does not cover its declared private receipt')
        private_path = declared_path
        if not private_path.is_file():
            raise ValueError(f'{corpus_id}: original private receipt is unavailable and no exact relocation map was declared')
        private_binding = _file_binding(private_path)
    private_receipt = _load_json(private_path)
    validate_receipt(private_receipt, ready.get('identity'))
    return {
        'corpus_id': corpus_id,
        'entry': entry,
        'manifest': manifest,
        'manifest_binding': _json_binding(manifest_path),
        'source_path': source_path,
        'source_binding': source_binding,
        'role_receipt': role_receipt,
        'role_receipt_binding': role_receipt_binding,
        'private_receipt_binding': private_binding,
        'private_receipt_custody': {'declared_path': declared_private_path,
            'actual_path': str(private_path.resolve()), 'actual_binding': private_binding,
            'relocation': copy.deepcopy(relocation),
            'relocation_map': copy.deepcopy(relocation_map_binding)},
        'ready_path': Path(ready_path).resolve(),
        'ready_binding': _file_binding(ready_path),
        'private_receipt_path': private_path.resolve(),
    }


def _write_selection_proof(runtime, path: Path, value: dict):
    runtime['natlang_neuralese.common.jsonio'].write_json_durable(path, value)
    return _json_binding(path)


def _read_memory_requests(parameters: dict, runtime: dict, train_binding: dict, role: str):
    value = parameters.get('memory_qualification')
    refs = value if isinstance(value, list) else [value]
    if not refs or any(not isinstance(ref, dict) or set(ref) != {'path', 'sha256'} for ref in refs):
        raise ValueError('selected native SFT stage needs exact memory_qualification request bindings')
    requests = []
    tokenizer = marker_ids = identity_config = None
    for ref in refs:
        request_path = runtime['natlang_neuralese.train.native_resident'].bound(ref)
        request = _load_json(request_path)
        source_ref = request.get('source')
        if not isinstance(source_ref, dict) or _copy_binding(source_ref) != _copy_binding(train_binding):
            raise ValueError('memory_qualification source differs from the selected stage train binding')
        identity = request.get('identity')
        model_config = identity.get('model_config') if isinstance(identity, dict) else None
        if not isinstance(model_config, dict):
            raise ValueError('memory_qualification lacks its actual model configuration identity')
        current_tokenizer = model_config.get('tokenizer_fingerprint')
        current_roles = model_config.get('marker_ids')
        if not isinstance(current_tokenizer, str) or not current_tokenizer:
            raise ValueError('memory_qualification lacks the model tokenizer fingerprint')
        if not isinstance(current_roles, list) or any(type(item) is not int for item in current_roles):
            raise ValueError('memory_qualification lacks exact model marker IDs')
        if tokenizer is None:
            tokenizer, marker_ids, identity_config = current_tokenizer, tuple(current_roles), copy.deepcopy(model_config)
        elif (tokenizer != current_tokenizer or marker_ids != tuple(current_roles)
              or identity_config != model_config):
            raise ValueError('selected stage capacity requests disagree on model/tokenizer identity')
        cases = request.get('cases')
        if not isinstance(cases, list) or not cases:
            raise ValueError('memory_qualification must retain nonempty actual capacity cases')
        limits = []
        for case in cases:
            tokens, kv = case.get('forward_tokens'), case.get('kv_slots')
            if type(tokens) is not int or tokens <= 0 or type(kv) is not int or kv < 0:
                raise ValueError('memory_qualification case lacks exact forward/KV limits')
            limits.append((tokens, kv))
        requests.append((ref, request, max(x[0] for x in limits), max(x[1] for x in limits)))
    expected_history = parameters.get('history_reasoning_policy')
    if not isinstance(expected_history, str) or not expected_history:
        expected_history = identity_config.get('history_reasoning_policy')
    if not isinstance(expected_history, str) or not expected_history:
        model_type = identity_config.get('backbone') or identity_config.get('model_type')
        if not isinstance(model_type, str) or not model_type:
            model_type = 'mellum' if role == 'mellum21' else 'lfm2'
        expected_history = runtime['natlang_neuralese.serve.chat'].declared_history_reasoning(model_type)
    if not isinstance(expected_history, str) or not expected_history:
        raise ValueError('selected model has no effective frozen history-reasoning policy')
    return requests, tokenizer, marker_ids, expected_history


def _descriptor_path_fields(descriptor: dict):
    runtime_root = Path(descriptor['runtime_root']).resolve(strict=True)
    artifact = descriptor['artifact_repo']
    artifact_root = Path(artifact['root']).resolve(strict=True)
    registry_path = _path_in(Path(artifact['registry_path']), artifact_root, 'corpus registry')
    manifests_root = _path_in(Path(artifact['manifests_root']), artifact_root, 'corpus manifest root')
    supervision = descriptor['supervision']
    supervision_root = Path(supervision['root']).resolve(strict=True)
    config_path = _path_in(Path(supervision['config_path']), supervision_root, 'supervision config')
    plan_path = _path_in(Path(supervision['plan_path']), supervision_root, 'supervision plan')
    ready_root = Path(descriptor['ready_export_root']).resolve(strict=True)
    custody_refs = descriptor.get('private_receipt_custody')
    relocation_map_binding = None
    receipt_relocations = {}
    if custody_refs is not None:
        source_map_path = Path(custody_refs['source_map']['path']).resolve(strict=True)
        transfer_path = Path(custody_refs['transfer_manifest']['path']).resolve(strict=True)
        source_map_binding = _file_binding(source_map_path)
        transfer_binding = _file_binding(transfer_path)
        if source_map_binding['sha256'] != custody_refs['source_map']['sha256']:
            raise ValueError('staged source map differs from its descriptor binding')
        if transfer_binding['sha256'] != custody_refs['transfer_manifest']['sha256']:
            raise ValueError('source transfer manifest differs from its descriptor binding')
        raw_map, transfer = _load_json(source_map_path), _load_json(transfer_path)
        if raw_map.get('schema') != 'natlang.staged-source-map/1' or not isinstance(raw_map.get('paths'), dict):
            raise ValueError('unsupported staged source-map schema')
        if (transfer.get('model_role') != descriptor['model_role']
                or not isinstance(transfer.get('files'), list)):
            raise ValueError('source transfer manifest model role differs from descriptor or file bindings are absent')
        transfer_by_original = {}
        for item in transfer['files']:
            original = item.get('original')
            if not isinstance(original, str):
                continue
            transfer_by_original.setdefault(original, []).append(item)
        for declared, staged in raw_map['paths'].items():
            if not isinstance(declared, str) or not isinstance(staged, str) or not declared.endswith('/private-receipt.json'):
                continue
            entries = transfer_by_original.get(declared, [])
            if len(entries) != 1:
                raise ValueError(f'source map private receipt lacks exactly one transfer binding: {declared}')
            entry = entries[0]
            staged_path = Path(staged)
            if not staged_path.is_absolute():
                raise ValueError(f'source map staged path must be absolute: {declared}')
            transfer_path_value = Path(entry['path'])
            if not transfer_path_value.is_absolute():
                # Transfer paths are relative to the extracted transport root,
                # which is the parent of the ready-export directory in the
                # standardized staged layout.
                transfer_path_value = ready_root.parent / transfer_path_value
            if transfer_path_value.resolve() != staged_path.resolve():
                raise ValueError(f'source map and transfer receipt disagree on staged private receipt: {declared}')
            digest = entry.get('sha256')
            byte_count = entry.get('bytes')
            if (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
                    or type(byte_count) is not int or byte_count <= 0):
                raise ValueError(f'source transfer manifest has an invalid private receipt binding: {declared}')
            corpus_name = Path(declared).parent.name
            if corpus_name in receipt_relocations:
                raise ValueError(f'duplicate staged private receipt mapping for {corpus_name}')
            receipt_relocations[corpus_name] = {'declared_path': declared,
                'staged_path': str(staged_path), 'original_sha256': digest,
                'staged_sha256': digest, 'bytes': byte_count}
        relocation_map_binding = {'source_map': _json_binding(source_map_path),
                                  'transfer_manifest': _json_binding(transfer_path)}
    output_root = Path(descriptor['output_root']).resolve()
    if output_root.exists():
        raise FileExistsError(f'output_root already exists; preserve existing evidence and choose a new path: {output_root}')
    return (runtime_root, artifact_root, registry_path, manifests_root, supervision_root,
            config_path, plan_path, ready_root, output_root,
            relocation_map_binding, receipt_relocations)


def prepare(descriptor: dict) -> dict:
    descriptor = _descriptor(descriptor)
    (runtime_root, artifact_root, registry_path, manifests_root, supervision_root,
     config_path, plan_path, ready_root, output_root, relocation_map_binding,
     receipt_relocations) = _descriptor_path_fields(descriptor)
    role = descriptor['model_role']
    runtime = _load_runtime(runtime_root)
    resident = runtime['natlang_neuralese.train.native_resident']
    recipe_api = runtime['natlang_neuralese.train.recipe']
    source_api = runtime['natlang_neuralese.data.native_sources']
    jsonio = runtime['natlang_neuralese.common.jsonio']
    supervisor = runtime['natlang_neuralese.rl.native_supervised']
    ce_builder = runtime['scripts.neuralese_data.build_native_ce_supervision_policy']
    write_json = jsonio.write_json_durable

    prior_config_binding = resident.binding(config_path)
    prior_plan_binding = resident.binding(plan_path)
    active_config = _load_json(resident.bound(prior_config_binding))
    config_recipe_path = Path(active_config['production_recipe']).resolve(strict=True)
    prior_recipe_binding = resident.binding(config_recipe_path)
    recipe = recipe_api.load_recipe(config_recipe_path)
    if resident.binding(config_recipe_path)['sha256'] != prior_recipe_binding['sha256']:
        raise ValueError('active recipe changed while it was being loaded')
    try:
        stage = next(item for item in recipe['stages'] if item.get('id') == descriptor['stage_id'])
    except StopIteration as exc:
        raise ValueError(f'selected recipe has no stage {descriptor["stage_id"]!r}') from exc
    if stage.get('kind') != 'native_state_sft':
        raise ValueError('private teacher append currently requires a native_state_sft stage')
    role_names = stage.get('inputs')
    if not isinstance(role_names, dict) or not {'native_graphs', 'held_native_graphs', 'ce_supervision_policy'} <= set(role_names):
        raise ValueError('selected stage must name train, protected-held and CE-policy input roles')
    resolved = recipe_api.resolve_stage_inputs(recipe, stage, {})
    train_binding = resolved['native_graphs']
    held_binding = resolved['held_native_graphs']
    ce_binding = resolved['ce_supervision_policy']
    prior_ce_path = resident.bound(ce_binding)
    prior_ce_binding = resident.binding(prior_ce_path)
    prior_ce = _load_json(prior_ce_path)
    prior_train_path = resident.bound(train_binding)
    old_prefix_binding = resident.binding(prior_train_path)
    held_path = resident.bound(held_binding)
    held_source_binding = resident.binding(held_path)
    old_tree = _load_json(prior_train_path)
    if old_tree.get('schema') != 'natlang.native-graph-sources/1' or not old_tree.get('sources'):
        raise ValueError('selected train binding must resolve to a nonempty native source tree')

    parameters = recipe_api.effective_stage_parameters(stage, [], recipe)
    memory_requests, tokenizer, role_ids, expected_history = _read_memory_requests(parameters, runtime, train_binding, role)
    memory_request_bindings = [resident.binding(resident.bound(item[0])) for item in memory_requests]
    expected_failure_policy = None
    if role == 'lfm25':
        if expected_history != 'keep':
            raise ValueError('selected LFM stage/model identity must explicitly declare history_reasoning_policy=keep')
        expected_failure_policy = 'mask-runtime-failures'

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    relocation_map_source_binding = relocation_map_binding
    if relocation_map_binding is not None:
        custody_dir = output_root / 'receipt-custody'
        custody_dir.mkdir()
        staged_custody = {}
        for name in ('source_map', 'transfer_manifest'):
            source_binding = relocation_map_source_binding[name]
            original = Path(source_binding['path'])
            staged = custody_dir / ('.natlang-source-map.json' if name == 'source_map' else 'transfer.json')
            shutil.copyfile(original, staged)
            staged_binding = _json_binding(staged)
            if staged_binding['sha256'] != source_binding['sha256']:
                raise ValueError(f'copied {name} changed bytes')
            staged_custody[name] = {'source': source_binding, 'staged': staged_binding}
        relocation_map_binding = staged_custody
    source_map_path = prior_train_path.parent / '.natlang-source-map.json'
    if not source_map_path.is_file():
        raise ValueError('selected train tree has no existing .natlang-source-map.json custody file')
    source_map_source_binding = _json_binding(source_map_path)
    source_map_value = _load_json(source_map_path)
    if source_map_value.get('schema') != 'natlang.staged-source-map/1' or not isinstance(source_map_value.get('paths'), dict):
        raise ValueError('selected train tree source map has an unsupported schema')
    output_source_map_path = output_root / source_map_path.name
    shutil.copyfile(source_map_path, output_source_map_path)
    source_map_staged_binding = _json_binding(output_source_map_path)
    if source_map_staged_binding['sha256'] != source_map_source_binding['sha256']:
        raise ValueError('copied prior source map changed bytes')

    old_train = source_api.NativeGraphSource(prior_train_path, tokenizer,
        required_split='train', role_ids=role_ids)
    prefix = [(entry['id'], entry['binding']) for entry in old_train.entries]
    old_bindings = dict(old_train.bindings)
    old_groups = set(old_train.groups)
    old_count = len(old_train)
    held_source = source_api.NativeGraphSource(held_path, tokenizer,
        required_split='test', role_ids=role_ids)
    held_groups = set(held_source.groups)
    del old_train
    gc.collect()

    registry_binding = _file_binding(registry_path)
    registry = _load_json(registry_path)
    if _file_binding(registry_path)['sha256'] != registry_binding['sha256']:
        raise ValueError('admission registry changed while it was being loaded')
    ready_snapshot_at = time.time()
    ready_snapshot = sorted(ready_root.glob('*/ready.json'))
    if not ready_snapshot:
        raise ValueError('ready_export_root has no private-ready source census')
    ready_records = []
    seen_corpus_ids = set()
    for ready_path in ready_snapshot:
        ready_path = _path_in(ready_path, ready_root, 'ready export record')
        ready = _load_json(ready_path)
        corpus_id = ready.get('corpus_id')
        if corpus_id in seen_corpus_ids:
            raise ValueError(f'duplicate ready corpus identity in snapshot: {corpus_id}')
        seen_corpus_ids.add(corpus_id)
        if ready.get('passed') is not True or ready.get('training_exposure') is not False:
            raise ValueError(f'{corpus_id}: source must be privately ready and have no learner exposure')
        ready_records.append(_verify_source_package(ready_path=ready_path, ready=ready,
            registry=registry, artifact_root=artifact_root, registry_path=registry_path,
            manifests_root=manifests_root, role=role, expected_tokenizer=tokenizer,
            expected_history=expected_history, expected_failure_policy=expected_failure_policy,
            runtime=runtime, ready_root=ready_root, receipt_relocations=receipt_relocations,
            relocation_map_binding=relocation_map_binding))
    unknown_relocations = set(receipt_relocations) - seen_corpus_ids
    if unknown_relocations:
        raise ValueError('private receipt relocation map contains records outside ready snapshot: '
                         + ', '.join(sorted(unknown_relocations)[:8]))
    failure_values = {entry['role_receipt']['failed_assistant_history_policy'] for entry in ready_records}
    if len(failure_values) != 1:
        raise ValueError('selected role receipts disagree on failed assistant history behavior')
    append_failure_policy = next(iter(failure_values))

    old_binding_by_hash = {}
    for item in old_tree['sources']:
        old_binding_by_hash.setdefault(item['sha256'], item)
    old_binding_by_path = {str(Path(item['path']).resolve()): item for item in old_tree['sources']}
    prepared_sources, duplicate_sources, already_active, proof_refs = [], [], [], []
    appended_descriptors, appended_ids, appended_bindings = [], set(), {}
    index_additions = []
    appended_groups = set()
    selected_ready_refs = []
    memory_envelopes = [(item[0], item[2], item[3]) for item in memory_requests]

    for package in ready_records:
        corpus_id = package['corpus_id']
        ready_path = package['ready_path']
        ready_binding = package['ready_binding']
        source_path = package['source_path']
        source_binding = package['source_binding']
        manifest_binding = package['manifest_binding']
        role_binding = package['role_receipt_binding']
        private_binding = package['private_receipt_binding']
        selected_ready_refs.append(ready_binding)

        same_old_path = old_binding_by_path.get(str(source_path.resolve()))
        same_old_hash = old_binding_by_hash.get(source_binding['sha256'])
        if same_old_path is not None or same_old_hash is not None:
            existing = same_old_path or same_old_hash
            if (existing.get('sha256') != source_binding['sha256']
                    or existing.get('bytes', source_binding['bytes']) != source_binding['bytes']):
                raise ValueError(f'{corpus_id}: old source path/hash equality conflicts with registered bytes')
            proof_path = output_root / 'selection-proofs' / f'{corpus_id}.json'
            proof_path.parent.mkdir(parents=True, exist_ok=True)
            proof = {'schema': SOURCE_EQUALITY_SCHEMA, 'corpus_id': corpus_id,
                'ready': ready_binding, 'manifest': manifest_binding,
                'role_receipt': role_binding, 'private_receipt': private_binding,
                'private_receipt_custody': package['private_receipt_custody'],
                'original_source': source_binding, 'existing_source': copy.deepcopy(existing),
                'existing_source_tree': _copy_binding(train_binding), 'selected_source': None,
                'whole_source_sha256_identical': True, 'new_rows': 0, 'new_admission': False,
                'reason': 'Complete source SHA/size equals an authenticated source leaf in the retained train tree; no duplicate graph rows or credit added.'}
            proof_ref = _write_selection_proof(runtime, proof_path, proof)
            proof_refs.append(proof_ref)
            already_active.append({'corpus_id': corpus_id, 'ready': ready_binding,
                'source': source_binding, 'manifest': manifest_binding,
                'role_receipt': role_binding, 'private_receipt': private_binding,
                'private_receipt_custody': package['private_receipt_custody'],
                'existing_source': copy.deepcopy(existing), 'equality_proof': proof_ref})
            continue

        graph_source = source_api.NativeGraphSource(source_path, tokenizer,
            required_split='train', role_ids=role_ids)
        if not len(graph_source):
            raise ValueError(f'{corpus_id}: admitted source has no complete graph rows')
        original_source_binding = _file_binding(source_path)
        if original_source_binding['sha256'] != source_binding['sha256'] or original_source_binding['bytes'] != source_binding['bytes']:
            raise ValueError(f'{corpus_id}: source changed after manifest validation')
        keep, equal_rows = {}, []
        for row_id, row_binding in graph_source.bindings.items():
            old_value = old_bindings.get(row_id)
            selected_value = appended_bindings.get(row_id)
            existing = old_value if old_value is not None else selected_value
            if existing is not None:
                if existing != row_binding:
                    raise ValueError(f'{corpus_id}: conflicting admitted graph identity {row_id!r}; refusing to drop or replace it')
                equal_rows.append({'id': row_id, 'binding': row_binding,
                    'already_in': 'retained-prefix' if old_value is not None else 'selected-new-source'})
            else:
                keep[row_id] = row_binding
        original_rows = len(graph_source)
        derivation = None
        if not keep:
            proof_path = output_root / 'selection-proofs' / f'{corpus_id}.json'
            proof_path.parent.mkdir(parents=True, exist_ok=True)
            proof = {'schema': ROW_SELECTION_SCHEMA, 'corpus_id': corpus_id,
                'ready': ready_binding, 'manifest': manifest_binding,
                'role_receipt': role_binding, 'private_receipt': private_binding,
                'private_receipt_custody': package['private_receipt_custody'],
                'original_source': original_source_binding, 'selected_source': None,
                'existing_source_tree': _copy_binding(train_binding), 'kept_rows': [],
                'omitted_equal_rows': equal_rows, 'whole_graphs_unchanged': True,
                'new_admission': False,
                'reason': 'Every complete admitted graph already has the exact same ID/binding in the retained prefix or an earlier selected new source.'}
            derivation = _write_selection_proof(runtime, proof_path, proof)
            proof_refs.append(derivation)
            duplicate_sources.append({'corpus_id': corpus_id, 'ready': ready_binding,
                'source': original_source_binding, 'manifest': manifest_binding,
                'role_receipt': role_binding, 'private_receipt': private_binding,
                'private_receipt_custody': package['private_receipt_custody'],
                'rows': original_rows, 'duplicate_rows': equal_rows, 'new_rows': 0,
                'derivation': derivation})
            del graph_source
            gc.collect()
            continue

        selected_path = source_path
        selected_source = graph_source
        if equal_rows:
            selected_path = output_root / 'deduplicated' / corpus_id / 'native-graphs.jsonl.zst'
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            written_ids = []
            with jsonio.open_jsonl_text(source_path) as reader, jsonio.write_jsonl_text(selected_path) as writer:
                for line in reader:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row['id'] in keep:
                        writer.write(line if line.endswith('\n') else line + '\n')
                        written_ids.append(row['id'])
            if len(written_ids) != len(keep) or set(written_ids) != set(keep):
                raise ValueError(f'{corpus_id}: filtered whole-graph rows do not match selected graph identities')
            del selected_source, graph_source
            gc.collect()
            selected_source = source_api.NativeGraphSource(selected_path, tokenizer,
                required_split='train', role_ids=role_ids)
            if selected_source.bindings != keep:
                raise ValueError(f'{corpus_id}: filtered graph bytes changed a retained complete graph binding')
            selected_binding = _file_binding(selected_path)
            proof_path = selected_path.with_name('selection.json')
            proof = {'schema': ROW_SELECTION_SCHEMA, 'corpus_id': corpus_id,
                'ready': ready_binding, 'manifest': manifest_binding,
                'role_receipt': role_binding, 'private_receipt': private_binding,
                'private_receipt_custody': package['private_receipt_custody'],
                'original_source': original_source_binding,
                'selected_source': selected_binding,
                'kept_rows': [{'id': row_id, 'binding': keep[row_id]} for row_id in written_ids],
                'omitted_equal_rows': equal_rows, 'whole_graphs_unchanged': True,
                'new_admission': False,
                'reason': 'Retain each original admitted graph exactly once; existing prefix wins identical identities. Conflicting bindings fail.'}
            derivation = _write_selection_proof(runtime, proof_path, proof)
            proof_refs.append(derivation)

        for graph in selected_source.iter_indices(range(len(selected_source))):
            tokens, kv = graph.plan.token_count, len(graph.plan.slot_sources)
            exceeded = [(ref, max_tokens, max_kv) for ref, max_tokens, max_kv in memory_envelopes
                        if tokens > max_tokens or kv > max_kv]
            if exceeded:
                details = '; '.join(f'{ref["path"]}: tokens<={max_tokens}, kv_slots<={max_kv}'
                                    for ref, max_tokens, max_kv in exceeded)
                raise ValueError(f'{corpus_id}/{graph.id}: complete retained graph geometry tokens={tokens}, kv_slots={kv} exceeds at least one existing capacity request: {details}; keep the source and obtain a real larger-case qualification')
        overlap_ids = set(selected_source.bindings) & set(held_source.bindings)
        if overlap_ids:
            raise ValueError(f'{corpus_id}: selected graph IDs overlap protected held rows: '
                             + ', '.join(sorted(overlap_ids)[:8]))
        for row_id, row_binding in selected_source.bindings.items():
            if row_id in appended_bindings:
                raise ValueError(f'{corpus_id}: selected source repeats graph identity {row_id!r}')
            appended_bindings[row_id] = row_binding
            appended_ids.add(row_id)
        appended_groups.update(selected_source.groups)
        index_additions.extend(resident._source_witness(selected_source, runtime_root / 'training' / 'neuralese' / 'natlang_neuralese'))
        selected_binding = _file_binding(selected_path)
        appended_descriptors.append({'path': selected_binding['path'], 'sha256': selected_binding['sha256'],
            'kind': 'native-graphs', 'epochs': 1, 'weight': 1.0,
            'training_priority': NEW_TEACHER_TRAINING_PRIORITY})
        prepared_sources.append({'corpus_id': corpus_id, 'ready': ready_binding,
            'source': selected_binding, 'original_source': original_source_binding,
            'manifest': manifest_binding, 'role_receipt': role_binding,
            'private_receipt': private_binding,
            'private_receipt_custody': package['private_receipt_custody'],
            'rows': len(selected_source),
            'original_rows': original_rows, 'communication_rows': sum(bool(entry.get('communication')) for entry in selected_source.entries),
            'derivation': derivation})
        del selected_source
        gc.collect()

    if not prepared_sources:
        raise ValueError('private-ready census contains no new complete graph rows to append')
    overlap = appended_groups & held_groups
    if overlap:
        raise ValueError('new admitted train graphs overlap protected held source groups: ' + ', '.join(sorted(overlap)[:8]))

    tree = copy.deepcopy(old_tree)
    tree['sources'].extend(appended_descriptors)
    new_tree_path = output_root / 'native-training-sources.json'
    write_json(new_tree_path, tree)
    new_tree_binding = _file_binding(new_tree_path)
    combined = source_api.NativeGraphSource(new_tree_path, tokenizer, required_split='train', role_ids=role_ids)
    if [(entry['id'], entry['binding']) for entry in combined.entries[:old_count]] != prefix:
        raise ValueError('append changed the exact old native graph prefix or order')
    if len(combined) != old_count + len(appended_ids):
        raise ValueError('append row count does not equal exact old prefix plus unique complete new rows')
    active_rows = len(combined)
    del combined, held_source
    gc.collect()

    prior_contract = prior_ce.get('source_contract', {})
    context_weight = prior_contract.get('default_raw_context_weight')
    if type(context_weight) not in (int, float):
        raise ValueError('prior CE policy has no exact inherited default raw-context coefficient')
    inventory = [new_tree_binding, *selected_ready_refs,
                 *[item['manifest'] for item in prepared_sources],
                 *[item['private_receipt'] for item in prepared_sources],
                 *[item['role_receipt'] for item in prepared_sources], *proof_refs]
    # Include every package's actual receipts, including sources omitted by exact
    # equality, then bind a finite complete census below.
    package_records = []
    for package in ready_records:
        package_records.append({'corpus_id': package['corpus_id'],
            'ready': package['ready_binding'], 'manifest': package['manifest_binding'],
            'source': package['source_binding'], 'role_receipt': package['role_receipt_binding'],
            'private_receipt': package['private_receipt_binding'],
            'private_receipt_custody': package['private_receipt_custody']})
    existing_ids = {item['corpus_id'] for item in prepared_sources + duplicate_sources + already_active}
    if existing_ids != seen_corpus_ids:
        raise ValueError('source census does not account for every privately ready package')
    dispositions = {item['corpus_id']: 'append-unique-whole-graphs' for item in prepared_sources}
    dispositions.update({item['corpus_id']: 'omit-identical-graph-bindings' for item in duplicate_sources})
    dispositions.update({item['corpus_id']: 'already-active-identical-source' for item in already_active})
    census = {'schema': CENSUS_SCHEMA, 'ready_directory_snapshot_at': _datetime.datetime.fromtimestamp(
            ready_snapshot_at, _datetime.timezone.utc).isoformat(), 'packages': len(package_records),
        'records': [{'corpus_id': item['corpus_id'], 'disposition': dispositions[item['corpus_id']],
                     **item} for item in package_records],
        'new_rows': len(appended_ids),
        'scope': 'Finite list returned by one ready-directory enumeration. New arrivals after enumeration remain eligible for a later append; no numeric batch cutoff or novel omission.'}
    census_path = output_root / 'ready-source-census.json'
    write_json(census_path, census)
    census_binding = _json_binding(census_path)
    prior_ce_inventory = prior_contract.get('inventory', [])
    if not isinstance(prior_ce_inventory, list):
        raise ValueError('prior CE source_contract.inventory is malformed')
    new_inventory = [*inventory, census_binding, source_map_source_binding,
                     source_map_staged_binding]
    if relocation_map_binding is not None:
        for name in ('source_map', 'transfer_manifest'):
            new_inventory.extend((relocation_map_binding[name]['source'],
                                  relocation_map_binding[name]['staged']))
    new_ce = ce_builder.complete_source_contract(prior_ce,
        [{'path': item['path'], 'sha256': item['sha256']} for item in tree['sources']],
        context_weight=context_weight, inventory=new_inventory)
    inherited_inventory = []
    seen_inventory = set()
    for item in [prior_ce_binding, *prior_ce_inventory, *new_ce['source_contract']['inventory']]:
        key = (item.get('path'), item.get('sha256'))
        if key not in seen_inventory:
            inherited_inventory.append({'path': item['path'], 'sha256': item['sha256']})
            seen_inventory.add(key)
    new_ce['source_contract']['inventory'] = inherited_inventory
    ce_path = output_root / 'ce-supervision-policy.json'
    write_json(ce_path, new_ce)
    ce_binding = _file_binding(ce_path)
    runtime['natlang_neuralese.train.native_supervision'].load_policy(_json_binding(ce_path))

    # Update every bound capacity request without dropping cases or old indexes.
    output_memory_refs = []
    memory_bindings = parameters.get('memory_qualification')
    is_memory_list = isinstance(memory_bindings, list)
    for index, (old_ref, request, _max_tokens, _max_kv) in enumerate(memory_requests):
        prior_indexes = copy.deepcopy(request['indexes'])
        request['indexes'].extend(index_additions)
        resident._check_indexes(request['indexes'])
        request['source'] = _json_binding(new_tree_path)
        request['scope'] = (str(request.get('scope', '')).rstrip() +
            ' Private teacher append: complete selected graphs fit the existing declared capacity cases; '
            'all prior cases, references and index positions remain intact. The real entry must repeat its normal capacity checks before updates.')
        request_path = output_root / ('memory-request.json' if len(memory_requests) == 1 else f'memory-request-{index:02d}.json')
        write_json(request_path, request)
        new_ref = _json_binding(request_path)
        output_memory_refs.append(new_ref)
        if len(request['cases']) != len(_load_json(resident.bound(old_ref))['cases']) or request['indexes'][:len(prior_indexes)] != prior_indexes:
            raise ValueError('capacity request lost existing cases or index order')
    next_recipe = copy.deepcopy(recipe)
    next_recipe['input_bindings'][role_names['native_graphs']] = _json_binding(new_tree_path)
    next_recipe['input_bindings'][role_names['ce_supervision_policy']] = _json_binding(ce_path)
    # The held role remains its original exact named binding and bytes.
    next_stage = next(item for item in next_recipe['stages'] if item['id'] == descriptor['stage_id'])
    next_stage['parameters']['memory_qualification'] = output_memory_refs if is_memory_list else output_memory_refs[0]
    if 'urgent_cohort_fairness' not in recipe_api.HANDLERS[next_stage['kind']]['parameters']:
        raise ValueError('teacher-source priority requires the existing urgent cohort fairness trainer')
    next_stage['parameters']['urgent_cohort_fairness'] = True
    remaining_recipe = next_recipe.setdefault('remaining_production_recipe', {})
    history = remaining_recipe.setdefault('private_teacher_append_history', [])
    prior_appends = [(key, value) for key, value in remaining_recipe.items()
                     if key.startswith('private_teacher_append_') and isinstance(value, dict)]
    for key, prior_append in sorted(prior_appends):
        history.append({'key': key, 'recipe': prior_recipe_binding,
                        'ce_supervision_policy': prior_ce_binding,
                        'append': copy.deepcopy(prior_append)})
    append_record = {'stage_id': descriptor['stage_id'], 'model_role': role,
        'sources': prepared_sources, 'duplicate_sources': duplicate_sources,
        'already_active_sources': already_active, 'selection_proofs': proof_refs,
        'source_census': census_binding,
        'source_map': {'source': source_map_source_binding, 'staged': source_map_staged_binding},
        'old_rows': old_count, 'new_rows': len(appended_ids), 'active_rows': active_rows,
        'existing_rows_unchanged': True, 'held_unchanged': True,
        'training_exposure': False, 'new_optimizer_updates': 0,
        'history_policy': expected_history, 'failed_assistant_history_policy': append_failure_policy}
    next_recipe['remaining_production_recipe'][descriptor['append_key']] = append_record
    next_recipe_path = output_root / 'recipe.json'
    write_json(next_recipe_path, next_recipe)
    recipe_api.load_recipe(next_recipe_path)

    # Validate that the complete loader still resolves exactly the stage roles we
    # just rebound, with no unreferenced catalog inputs or hidden source roles.
    loaded_recipe = recipe_api.load_recipe(next_recipe_path)
    rebound = recipe_api.resolve_stage_inputs(loaded_recipe, next_stage, {})
    if (rebound['native_graphs']['sha256'] != new_tree_binding['sha256']
            or rebound['ce_supervision_policy']['sha256'] != ce_binding['sha256']
            or rebound['held_native_graphs']['sha256'] != held_binding['sha256']):
        raise ValueError('complete frozen recipe loader resolved different train/CE/held stage bindings')
    for source_role, binding in resolved.items():
        if source_role in {'native_graphs', 'ce_supervision_policy', 'held_native_graphs'}:
            continue
        # Roles may deliberately share one named input (the foundation often
        # uses the entire training source). Such aliases follow that binding;
        # separately named predecessor/held inputs remain exact and unchanged.
        name = role_names[source_role]
        expected = (next_recipe['input_bindings'][name]
            if name in {role_names['native_graphs'], role_names['ce_supervision_policy']}
            else binding)
        if source_role not in rebound or rebound[source_role]['sha256'] != expected['sha256']:
            raise ValueError(f'optional stage source role {source_role!r} changed during append preparation')

    # Preserve the active seed's operational values, including Mellum's inherited
    # checkpoint_every=50 override, while supplying newly bound stage data roles.
    next_config = copy.deepcopy(active_config)
    next_config['production_recipe'] = str(next_recipe_path.resolve())
    lineage = active_config.get('recipe_lineage_sha256')
    if not isinstance(lineage, str) or not re.fullmatch(r'[0-9a-f]{64}', lineage):
        raise ValueError('active config has no exact preserved recipe lineage')
    next_config['recipe_lineage_sha256'] = lineage
    recipe_binding = recipe_api.native_recipe_binding(next_recipe_path, lineage)
    next_config['trainer_parameters'] = {**next_config.get('trainer_parameters', {}), 'recipe_binding': recipe_binding,
        'urgent_cohort_fairness': True}
    seed_parameters = dict(parameters)
    seed_parameters.update(next_config.get('seed_trainer_parameters', {}))
    seed_parameters['stage_id'] = descriptor['stage_id']
    seed_parameters['recipe_binding'] = recipe_binding
    seed_parameters['memory_qualification'] = next_stage['parameters']['memory_qualification']
    seed_parameters['urgent_cohort_fairness'] = True
    next_config['seed_trainer_parameters'] = seed_parameters
    seed_sources = _seed_source_bindings(rebound, train=_json_binding(new_tree_path),
        held=held_binding, ce=_json_binding(ce_path))
    next_config = supervisor.bind_seed_sources(next_config, seed_sources)
    supervisor.bind_seed_parameters(next_config, seed_parameters)
    next_config_path = output_root / 'next-config.json'
    write_json(next_config_path, next_config)
    next_plan_path = output_root / 'next-plan.json'
    shutil.copyfile(plan_path, next_plan_path)

    # Snapshot stability and compare-and-swap validation are the final gate.
    for package in ready_records:
        for key in ('ready_binding', 'source_binding', 'manifest_binding', 'role_receipt_binding', 'private_receipt_binding'):
            expected = package[key]
            if _file_binding(expected['path'])['sha256'] != expected['sha256']:
                raise ValueError(f'{package["corpus_id"]}: source custody changed during preparation ({key})')
    if resident.binding(config_path)['sha256'] != prior_config_binding['sha256']:
        raise ValueError('active config changed during preparation; staged future config is stale')
    if resident.binding(plan_path)['sha256'] != prior_plan_binding['sha256']:
        raise ValueError('active plan changed during preparation; staged future plan is stale')
    if resident.binding(config_recipe_path)['sha256'] != prior_recipe_binding['sha256']:
        raise ValueError('active recipe changed during preparation; staged future recipe is stale')
    if resident.binding(prior_ce_path)['sha256'] != prior_ce_binding['sha256']:
        raise ValueError('active CE policy changed during preparation; staged future CE policy is stale')
    if resident.binding(prior_train_path)['sha256'] != old_prefix_binding['sha256']:
        raise ValueError('old active source changed during append preparation')
    if resident.binding(held_path)['sha256'] != held_source_binding['sha256']:
        raise ValueError('protected held source changed during append preparation')
    if _file_binding(registry_path)['sha256'] != registry_binding['sha256']:
        raise ValueError('admission registry changed during append preparation')
    if _file_binding(source_map_path)['sha256'] != source_map_source_binding['sha256']:
        raise ValueError('prior source map changed during append preparation')
    if _file_binding(output_source_map_path)['sha256'] != source_map_staged_binding['sha256']:
        raise ValueError('staged source map changed during append preparation')
    for ref, expected in zip(memory_requests, memory_request_bindings, strict=True):
        if resident.binding(resident.bound(ref[0]))['sha256'] != expected['sha256']:
            raise ValueError('capacity request changed during append preparation')
    if relocation_map_binding is not None:
        for name in ('source_map', 'transfer_manifest'):
            for key in ('source', 'staged'):
                binding = relocation_map_binding[name][key]
                if _file_binding(binding['path'])['sha256'] != binding['sha256']:
                    raise ValueError(f'{name} custody binding changed during append preparation ({key})')

    prepared = {'schema': APPEND_SCHEMA, 'at': time.time(), 'passed': True, 'deployed': False,
        'prior_config': prior_config_binding, 'prior_plan': prior_plan_binding,
        'next_config': _json_binding(next_config_path), 'next_plan': _file_binding(next_plan_path),
        'recipe': _json_binding(next_recipe_path), 'ce_supervision_policy': _json_binding(ce_path),
        'native_training_sources': _json_binding(new_tree_path),
        'source_census': census_binding, 'ready_packages': len(ready_records),
        'selected_sources': len(prepared_sources), 'duplicate_only_sources': len(duplicate_sources),
        'already_active_sources': len(already_active), 'selection_proofs': len(proof_refs),
        'private_receipt_custody': relocation_map_binding,
        'old_rows': old_count, 'new_rows': len(appended_ids), 'active_rows': active_rows,
        'new_optimizer_updates': 0, 'existing_prefix_unchanged': True,
        'source_held_overlap': False, 'capacity_checks_preserved': True,
        'source_selection_policy': 'All privately ready packages in one finite ready-directory snapshot; no numeric batch cutoff. Complete source SHA equality and exact row-ID/binding equality only; conflicts fail.',
        'model_role': role, 'stage_id': descriptor['stage_id'],
        'review_required': 'Independent operator review before future-only config/plan CAS adoption.'}
    prepared_path = output_root / 'prepared.json'
    write_json(prepared_path, prepared)
    return {**prepared, 'prepared': _file_binding(prepared_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True,
                        help='explicit model/runtime/artifact/supervision/output descriptor JSON')
    args = parser.parse_args(argv)
    result = prepare(_load_json(args.descriptor))
    print(json.dumps({'passed': result['passed'], 'deployed': result['deployed'],
        'prepared': result['prepared'], 'ready_packages': result['ready_packages'],
        'selected_sources': result['selected_sources'], 'new_rows': result['new_rows']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
