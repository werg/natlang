"""Exact source-curated full-session cohort verifier and registry proposal."""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import os
from pathlib import Path

from natlang_neuralese.common.hashing import (
    canonical_json_sha256_hex, sha256_file_hex, sha256_file_hex_cached)
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease
from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.model.native_model import tokenizer_inventory
from scripts.neuralese_data.build_acquired_full_session_graphs import (
    sessions, source_target_ids, completed_shards, session_exclusion,
    prevalidate_session_tool_arguments, prove_render_pending_runtime_continuation)
from natlang_neuralese.serve.chat import RequestError
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.admission_resume import (
    artifact_bindings, load_model_checkpoint, load_or_compute_model_checkpoint,
    model_checkpoint_identity, save_model_checkpoint,
)

ROOT = Path(os.environ.get('NATLANG_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()


def bound(path, binding):
    path = Path(path)
    if (sha256_file_hex_cached(path, trust_archive_receipt=False) != binding['sha256']
            or ('bytes' in binding and path.stat().st_size != binding['bytes'])):
        raise ValueError('immutable acquired artifact changed: ' + str(path))
    return path


def _closure_artifact(path, binding, *, use_cache=True):
    """Content-hash a closure input through the race-checked shared cache."""
    path = Path(path).resolve(strict=True)
    digest = (sha256_file_hex_cached(path, trust_archive_receipt=False)
              if use_cache else sha256_file_hex(path))
    size = path.stat().st_size
    if digest != binding['sha256'] or ('bytes' in binding and size != binding['bytes']):
        raise ValueError('immutable acquired closure artifact changed: ' + str(path))
    return path, digest, size


def decision_contract(bindings):
    """Checkout location is observed metadata; exact hold filenames/bytes bind policy."""
    return sorted((Path(item['path']).name, item['sha256'], item['bytes'], item['kind']) for item in bindings)


def source_occurrence(value):
    return 'text:' + value['trajectory']['id'] + ':source-occurrence:' + canonical_json_sha256_hex(
        [value['source_file']['sha256'], value['source_row']])


def acquired_source_closure(path, protection, *, use_cache=True):
    """Reconstruct all source components, including not-yet-converted siblings."""
    manifest = json.loads(Path(path).read_text()); bindings = manifest.get('normalized_export_bindings')
    if not isinstance(bindings, list) or not bindings:
        raise ValueError('acquired admission needs the declared complete cross-source held closure')
    held = set(); counts = Counter(); seen = set(); group_rows = []
    inherited = manifest.get('inherited_source_components')
    if inherited:
        parent = Path(path).parent
        component_path, _, _ = _closure_artifact(parent / inherited['components']['path'], inherited['components'], use_cache=use_cache)
        projection_path, _, _ = _closure_artifact(parent / inherited['projection']['path'], inherited['projection'], use_cache=use_cache)
        projection = json.loads(projection_path.read_text())
        proof = projection.get('acquired_source_group_closure') or {}
        original = next((item for item in projection['files']
            if item['path'] == projection.get('source_group_components')), None)
        inherited_rows = [row['source_groups'] for row in sessions(component_path)]
        if (proof.get('passed') is not True or original is None
                or original['sha256'] != inherited['components']['sha256']
                or canonical_json_sha256_hex(inherited_rows) !=
                   (proof.get('accounting') or {}).get('source_group_components_sha256')):
            raise ValueError('inherited acquired components differ from their qualified parent projection')
        group_rows.extend(inherited_rows)
    for binding in bindings:
        export_path, _, _ = _closure_artifact(binding['path'], binding, use_cache=use_cache)
        if export_path in seen: raise ValueError('global source closure repeats an export receipt')
        seen.add(export_path); export = json.loads(export_path.read_text())
        source, _, _ = _closure_artifact(export_path.parent / 'full-trajectories.jsonl.zst',
            {'sha256': binding['source_sha256']}, use_cache=use_cache)
        if export['outputs'][source.name] != binding['source_sha256']:
            raise ValueError('global source export and closure disagree on immutable session bytes')
        rows = 0
        for value in sessions(source):
            trajectory = value['trajectory']; rows += 1
            if trajectory['split'] not in {'train', 'test', 'validation'}:
                raise ValueError('global source closure contains an undeclared split')
            if trajectory['split'] != 'train': held.update(trajectory['split_groups'])
            group_rows.append(trajectory['split_groups'])
        # Reject a source mutation that raced the streaming parse, even when it
        # left the number of decoded sessions unchanged.
        _closure_artifact(source, {'sha256': binding['source_sha256']}, use_cache=use_cache)
        if rows != export['complete_sessions']: raise ValueError('global source closure lost session accounting')
        counts[export_path.parent.parent.name] += rows
    declared = set()
    for directory in {export_path.parent.parent for export_path in seen}:
        complete_path = directory / 'complete.json'
        complete = json.loads(complete_path.read_text())
        # Complete inventories may retain a logical mount alias while a
        # verified projection binds the canonical target path. Compare the
        # actual files, not their spelling, after resolving every declared
        # receipt; the receipt bytes remain checked by the bindings above.
        declared.update(Path(item['receipt']).resolve(strict=True) for item in complete['files']
                        if item['status'] in {'exported', 'reused'})
    if declared != seen:
        raise ValueError('global held closure omits a declared normalized source-file inventory')
    from scripts.neuralese_data.splits import close_group_sets
    closed, components = close_group_sets(group_rows,
        {**{key: protection[key] for key in ('held', 'quality', 'pending')},
         'held': protection['held'] | held})
    accounting = {'normalized_sessions': sum(counts.values()), 'by_source': dict(counts),
        'export_receipts': len(bindings), 'declared_held_groups': len(held),
        'declared_held_groups_sha256': canonical_json_sha256_hex(sorted(held)),
        'normalized_export_bindings_sha256': canonical_json_sha256_hex(bindings),
        'source_group_components_sha256': canonical_json_sha256_hex(components),
        'closed_group_counts': {key: len(value) for key, value in closed.items()}}
    return accounting, closed, components


SOURCE_CLOSURE_RECEIPT_SCHEMA = 'natlang.acquired-source-closure-receipt/1'


def _source_closure_implementation_bindings():
    root = Path(__file__).resolve().parents[2]
    paths = {
        'closure_verifier': Path(__file__),
        'session_stream_parser': root / 'scripts/neuralese_data/build_acquired_full_session_graphs.py',
        'connected_group_closure': root / 'scripts/neuralese_data/splits.py',
        'current_protection': root / 'scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py',
        'source_review_policy': root / 'scripts/source_review.py',
        'receipt_lease': root / 'scripts/neuralese_data/conversion_output_lease.py',
        'jsonl_io': root / 'training/neuralese/natlang_neuralese/common/jsonio.py',
        'hashing': root / 'training/neuralese/natlang_neuralese/common/hashing.py',
    }
    return {name: {'path': path.resolve(strict=True).relative_to(root).as_posix(), 'sha256': sha256_file_hex(path),
                   'bytes': path.stat().st_size} for name, path in sorted(paths.items())}


def _source_closure_artifacts(path, manifest, *, use_cache=True):
    """Return the complete exact input closure, including per-directory inventories."""
    path = Path(path).resolve(strict=True)
    artifacts = {}

    def checked_cached(candidate, binding=None):
        candidate = Path(candidate).resolve(strict=True)
        digest = (sha256_file_hex_cached(candidate, trust_archive_receipt=False)
                  if use_cache else sha256_file_hex(candidate))
        size = candidate.stat().st_size
        if binding is not None and (digest != binding['sha256']
                                    or ('bytes' in binding and size != binding['bytes'])):
            raise ValueError('acquired closure artifact changed: ' + str(candidate))
        return candidate, digest, size

    def add(candidate, *, digest=None, size=None, binding=None):
        if digest is None or size is None:
            candidate, digest, size = checked_cached(candidate, binding)
        else:
            candidate = Path(candidate).resolve(strict=True)
        artifacts[str(candidate)] = {'path': str(candidate), 'sha256': digest, 'bytes': size}

    for item in manifest.get('files', []):
        add(path.parent / item['path'], binding=item)
    component_name = manifest.get('source_group_components')
    if component_name is not None:
        component_binding = next((item for item in manifest.get('files', [])
                                  if item.get('path') == component_name), None)
        if component_binding is None:
            raise ValueError('source-group component inventory lacks an explicit manifest binding')
        add(path.parent / component_name, binding=component_binding)
    inherited = manifest.get('inherited_source_components') or {}
    for item in inherited.values():
        add(path.parent / item['path'], binding=item)
    bindings = manifest.get('normalized_export_bindings')
    if not isinstance(bindings, list) or not bindings:
        raise ValueError('acquired closure receipt needs the complete normalized export set')
    directories = set()
    for binding in bindings:
        export_path, export_digest, export_size = checked_cached(binding['path'], binding)
        export = json.loads(export_path.read_text())
        source, source_digest, source_size = checked_cached(
            export_path.parent / 'full-trajectories.jsonl.zst', {'sha256': binding['source_sha256']})
        if export.get('outputs', {}).get(source.name) != binding['source_sha256']:
            raise ValueError('closure receipt source/export binding differs')
        add(export_path, digest=export_digest, size=export_size)
        add(source, digest=source_digest, size=source_size)
        directories.add(export_path.parent.parent.resolve(strict=True))
    declared = set()
    for directory in directories:
        complete_path = directory / 'complete.json'
        complete = json.loads(complete_path.read_text())
        add(complete_path)
        for item in complete.get('files', []):
            if item.get('status') not in {'exported', 'reused'}:
                continue
            receipt = Path(item['receipt'])
            if not receipt.is_absolute():
                receipt = directory / receipt
            declared.add(receipt.resolve(strict=True))
    expected = {Path(item['path']).resolve(strict=True) for item in bindings}
    if declared != expected:
        raise ValueError('closure receipt does not cover each complete normalized export inventory')
    return [artifacts[key] for key in sorted(artifacts)]


def verify_global_source_closure(path, protection, *, receipt_path=None, write_receipt_path=None,
                                force_full_validation=False):
    """Reject direct-key protection that misses a transitive source sibling."""
    if force_full_validation:
        receipt_path = None
    if receipt_path is not None and not Path(receipt_path).exists():
        receipt_path = Path(receipt_path)
        with acquire_output_lease(receipt_path.parent):
            if receipt_path.exists():
                return verify_global_source_closure(path, protection, receipt_path=receipt_path,
                                                    force_full_validation=force_full_validation)
            return verify_global_source_closure(path, protection, write_receipt_path=receipt_path,
                                                force_full_validation=force_full_validation)
    manifest_path = Path(path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if receipt_path is not None and Path(receipt_path).exists():
        receipt_path = Path(receipt_path).resolve(strict=True)
        receipt = json.loads(receipt_path.read_text())
        current_protection = {key: sorted(protection[key]) for key in ('held', 'quality', 'pending')}
        if (receipt.get('schema') != SOURCE_CLOSURE_RECEIPT_SCHEMA
                or receipt.get('manifest') != {'path': str(manifest_path),
                    'sha256': sha256_file_hex(manifest_path), 'bytes': manifest_path.stat().st_size}
                or receipt.get('protection') != current_protection
                or receipt.get('protection_sha256') != canonical_json_sha256_hex(current_protection)
                or receipt.get('verifier_sha256') != sha256_file_hex(__file__)):
            raise ValueError('durable acquired-source closure receipt is stale for current protection or verifier')
        if receipt.get('implementation_files') != _source_closure_implementation_bindings():
            raise ValueError('durable acquired-source closure implementation dependency changed')
        artifacts = _source_closure_artifacts(manifest_path, manifest)
        if artifacts != receipt.get('input_artifacts'):
            raise ValueError('durable acquired-source closure input inventory changed')
        accounting = receipt.get('accounting')
        if (receipt.get('training_admission') is not False
                or not isinstance(accounting, dict)
                or accounting.get('normalized_export_bindings_sha256') != canonical_json_sha256_hex(manifest['normalized_export_bindings'])
                or accounting.get('export_receipts') != len(manifest['normalized_export_bindings'])
                or accounting.get('closed_group_counts') != {key: len(protection[key]) for key in ('held', 'quality', 'pending')}
                or type(accounting.get('normalized_sessions')) is not int
                or accounting['normalized_sessions'] < 1
                or not isinstance(accounting.get('by_source'), dict)
                or sum(accounting['by_source'].values()) != accounting['normalized_sessions']):
            raise ValueError('durable acquired-source closure accounting is incomplete')
        return accounting
    accounting, closed, components = acquired_source_closure(
        path, protection, use_cache=not force_full_validation)
    if any(closed[key] != protection[key] for key in closed):
        raise ValueError('global acquired source protection lacks complete connected-group closure')
    if manifest.get('source_group_components'):
        stored = [row['source_groups'] for row in sessions(Path(path).parent / manifest['source_group_components'])]
        if stored != components:
            raise ValueError('protected source component inventory differs from complete normalized sources')
    if write_receipt_path is not None:
        if Path(write_receipt_path).exists():
            raise ValueError('source-closure receipts are immutable; choose a new output path')
        protection_binding = {key: sorted(protection[key]) for key in ('held', 'quality', 'pending')}
        receipt = {'schema': SOURCE_CLOSURE_RECEIPT_SCHEMA,
            'manifest': {'path': str(manifest_path), 'sha256': sha256_file_hex(manifest_path),
                         'bytes': manifest_path.stat().st_size},
            'protection': protection_binding,
            'protection_sha256': canonical_json_sha256_hex(protection_binding),
            'input_artifacts': _source_closure_artifacts(
                manifest_path, manifest, use_cache=not force_full_validation),
            'implementation_files': _source_closure_implementation_bindings(),
            'accounting': accounting, 'verifier_sha256': sha256_file_hex(__file__),
            'training_admission': False}
        write_json_durable(write_receipt_path, receipt)
    return accounting


def source_member_scope(projection, selected_bindings):
    """Describe one independently complete source member within a full closure.

    Conversion/admission can publish a complete file member before sibling
    files finish rendering, while the projection still proves protection over
    the entire declared source set. This helper makes the unfinished members
    explicit and refuses a member not present in that global projection.
    """
    declared = projection.get('normalized_export_bindings')
    if not isinstance(declared, list) or not declared:
        raise ValueError('incremental member needs a complete projected source closure')
    by_content = {}
    for item in declared:
        by_content.setdefault((item.get('sha256'), item.get('source_sha256')), []).append(item)
    selected_content = {(item.get('sha256'), item.get('source_sha256')) for item in selected_bindings}
    if len(selected_bindings) != 1 or len(selected_content) != 1 or len(by_content.get(next(iter(selected_content)), [])) != 1:
        raise ValueError('incremental member must be one exact export bound by the global projection')
    selected = next(iter(selected_content))
    return {'kind': 'complete-normalized-source-member', 'selected_exports': [
            {'path': item['path'], 'sha256': item['sha256'], 'source_sha256': item['source_sha256']}
            for item in by_content[selected]],
        'global_export_bindings_sha256': canonical_json_sha256_hex(declared),
        'pending_exports': [
            {'path': item['path'], 'sha256': item['sha256'], 'source_sha256': item['source_sha256']}
            for item in declared if (item.get('sha256'), item.get('source_sha256')) != selected],
        'pending_export_count': len(declared) - 1}


def incremental_member_source_kind(export):
    """Return the declared complete-member contract admitted by the incremental path."""
    contract = export.get('contract', {})
    source = contract.get('source', {})
    if isinstance(source.get('path'), str) and source['path'].startswith('jsonl_sft/'):
        return 'jsonl_sft'
    if (isinstance(source.get('path'), str) and not source['path'].startswith('dpo/')
            and contract.get('trajectory_representation') == 'existing complete Trajectory dataclass'
            and type(contract.get('source_rows')) is int and contract['source_rows'] > 0
            and type(export.get('complete_sessions')) is int and export['complete_sessions'] > 0):
        # Preserve the historical one-row contract's precise meaning. A
        # normalized export can also be one complete member of the global
        # source projection while containing many sessions from one pinned
        # source file (for example a parquet member); it is not a single row.
        if contract['source_rows'] == 1:
            return 'complete-trajectory-single-source-row'
        return 'complete-trajectory-source-export'
    return None


def verify_graph_source(graph, value, *, protection, held_sources, fingerprint,
                        explicit_history_reasoning_policy=None):
    """Authenticate an actual graph against its original complete source occurrence."""
    if value.get('positive_candidate') is not True or session_exclusion(value, protection, held_sources):
        raise ValueError('native graph targets an excluded or nonpositive source session')
    trajectory = value['trajectory']; provenance = graph.get('conversion', {}).get('acquired_session', {})
    expected = {'normalized_session_sha256': canonical_json_sha256_hex(value),
        'original_source_row': value['source_row'], 'original_source_file': value['source_file'],
        'source_revision': value['source_revision'], 'original_target_ids': source_target_ids(value),
        'original_messages': len(trajectory['messages']),
        'original_assistant_turns': sum(m['role'] == 'assistant' for m in trajectory['messages']),
        'source_lineage': trajectory['lineage'], 'source_license': trajectory['license'],
        'admission_class': 'source-curated-third-party-SFT', 'oracle_verified': False,
        'native_execution_verified': False, 'observed_actor_state_exchange': False, 'truncated': False}
    if any(provenance.get(key) != item for key, item in expected.items()):
        raise ValueError('native graph lost its exact original session/target provenance')
    if explicit_history_reasoning_policy is not None:
        profile = explicit_history_reasoning_policy
        provenance_profile = (provenance.get('history_reasoning_policy')
            or provenance.get('history_reasoning', {}).get('policy'))
        if profile not in ('keep', 'last_turn_only') or provenance_profile != profile:
            raise ValueError('native graph history rendering profile differs from its conversion receipt')
    if graph.get('source_groups') != sorted(set(trajectory['split_groups'])):
        raise ValueError('native graph changed original source groups')
    groups = set(trajectory['split_groups'])
    split = 'test' if groups & protection['held'] else 'train'
    if trajectory['split'] not in {'train', 'test', 'validation'} or (
            trajectory['split'] != 'train' and split != 'test') or graph.get('split') != split:
        raise ValueError('native graph violates global source-declared held closure')
    base = source_occurrence(value)
    if explicit_history_reasoning_policy is not None:
        base += ':history-reasoning:' + explicit_history_reasoning_policy
    view = provenance.get('view')
    auxiliary = provenance.get('auxiliary_reasoning_assistant_ordinals')
    assistants = [m for m in trajectory['messages'] if m['role'] == 'assistant']
    coverage = provenance.get('reasoning_coverage', {})
    reasoning = [i for i, m in enumerate(assistants) if m.get('reasoning')]
    expected_auxiliary = [i for i in reasoning if not coverage.get(str(i), {}).get('present_in_whole_view')]
    if (not isinstance(auxiliary, list) or auxiliary != expected_auxiliary
            or provenance.get('whole_view_reasoning_assistant_ordinals') != [i for i in reasoning if i not in auxiliary]
            or set(coverage) != {str(i) for i in reasoning}):
        raise ValueError('native graph reasoning-view coverage inventory differs from original assistants')
    expected_ids = {base, *(base + ':reasoning-turn:' + str(i) for i in auxiliary)}
    if view == 'whole-trajectory':
        expected_id = base
    elif view == 'full-prefix-reasoning-target' and provenance.get('assistant_turn') in auxiliary:
        expected_id = base + ':reasoning-turn:' + str(provenance['assistant_turn'])
    else:
        raise ValueError('native graph claims an unsupported or unnecessary session view')
    if (graph.get('id') != expected_id or graph['conversion'].get('source_row_id') != expected_id.removeprefix('text:')
            or graph['conversion'].get('source_record_ids') != [trajectory['id']]
            or graph['conversion'].get('source_twin_id') != canonical_json_sha256_hex([trajectory['id']])):
        raise ValueError('native graph occurrence/view identity differs from original source')
    example = parse_graph(graph, tokenizer_fingerprint=fingerprint)
    if (len(example.plan.invocations) != 1 or example.external_states
            or example.plan.invocations[0].imports or not any(w > 0 for w in example.target_weights[1:])):
        raise ValueError('complete-session SFT needs a nonempty ordinary native target graph')
    return expected_ids, len(example.target_weights), split


def _prepared_template_backbone(receipt):
    """Read the actual template selector from the conversion's tokenizer binding."""
    try:
        backbone = receipt['prepared_tokenizer']['rendering_configuration']['history_reasoning']['backbone']
    except (KeyError, TypeError) as error:
        raise ValueError('native conversion lacks its prepared tokenizer template binding') from error
    if backbone not in ('lfm2', 'mellum'):
        raise ValueError('native conversion has an unsupported prepared tokenizer template')
    return backbone


def _prevalidate_source_for_conversion(value, receipt):
    """Re-run the converter's exact model-bound source preflight."""
    return prevalidate_session_tool_arguments(
        value, template_backbone=_prepared_template_backbone(receipt))


def verify_model_source_parity(model_views):
    """Require paired source coverage, allowing only the declared LFM2 scalar case."""
    models = list(model_views)
    if len(models) < 2:
        raise ValueError('acquired session admission needs both model views')
    for view in model_views.values():
        if set(view['accepted']) & set(view['pending']):
            raise ValueError('one model source occurrence is both accepted and render-pending')
    all_keys = set().union(*(set(view['accepted']) | set(view['pending']) for view in model_views.values()))
    for key in all_keys:
        states = {name: ('pending' if key in view['pending'] else
                         'accepted' if key in view['accepted'] else 'excluded')
                  for name, view in model_views.items()}
        state_set = set(states.values())
        if state_set == {'excluded'}:
            continue
        if state_set == {'accepted'}:
            identities = [view['accepted'][key] for view in model_views.values()]
            if any(identity != identities[0] for identity in identities[1:]):
                raise ValueError('model views differ in a jointly accepted complete source session')
        elif state_set == {'pending'}:
            continue
        elif state_set == {'accepted', 'pending'}:
            pending_models = [name for name, state in states.items() if state == 'pending']
            accepted_models = [name for name, state in states.items() if state == 'accepted']
            if (len(pending_models) != 1 or len(accepted_models) != 1
                    or model_views[pending_models[0]]['backbone'] != 'lfm2'
                    or model_views[accepted_models[0]]['backbone'] != 'mellum'):
                raise ValueError('cross-model source outcome differs outside the LFM2 template-pending case')
        else:
            raise ValueError('cross-model source occurrence is missing from a complete or explicit pending view')


def whole_trajectory_source_view(proofs):
    """Project render proofs to the historical twin-comparison scope."""
    whole = {}
    for proof in proofs:
        if proof['view'] != 'whole-trajectory':
            continue
        key = (proof['normalized_session_sha256'], proof['source_row'])
        whole.setdefault(key, set()).add((proof['id'], proof['normalized_session_sha256'],
            proof['split'], 'whole-trajectory'))
    return whole


def native_conversion_directory(native_root, export_path, export, model):
    """Resolve native output by its original source SHA or legacy export identity."""
    root = Path(native_root)
    source_sha = export.get('contract', {}).get('source', {}).get('sha256')
    keys = [export_path.parent.name]
    if isinstance(source_sha, str) and len(source_sha) == 64 and all(c in '0123456789abcdef' for c in source_sha):
        keys.append(source_sha)
    candidates = []
    for key in dict.fromkeys(keys):
        candidate = root / key / model
        if (candidate / 'receipt.json').is_file():
            resolved = candidate.resolve(strict=True)
            if resolved not in candidates:
                candidates.append(resolved)
    if len(candidates) > 1:
        raise ValueError('native conversion lookup is ambiguous between export identity and source SHA')
    if candidates:
        return candidates[0]
    return root / keys[0] / model


def _admission_verifier_files():
    """Exact Python modules whose code can change the cached conversion verdict."""
    functions = (sessions, source_target_ids, completed_shards, session_exclusion,
                 prevalidate_session_tool_arguments, prove_render_pending_runtime_continuation,
                 current_protection, load_decision_holds, parse_graph, tokenizer_inventory,
                 write_json_durable, sha256_file_hex_cached, RequestError,
                 decision_contract, verify_graph_source, _prevalidate_source_for_conversion,
                 load_model_checkpoint, load_or_compute_model_checkpoint,
                 model_checkpoint_identity, save_model_checkpoint)
    paths = {Path(__file__).resolve()}
    for function in functions:
        source = inspect.getsourcefile(function)
        if source:
            paths.add(Path(source).resolve())
    from scripts.neuralese_data.continue_acquired_protection import verify_protection_sets, verify_session_transition
    for function in (verify_protection_sets, verify_session_transition):
        source = inspect.getsourcefile(function)
        if source:
            paths.add(Path(source).resolve())
    return sorted(paths)


def _model_resume_identity(*, export_path, source_path, original, rows, model, base,
                           tokenizer, protection, conversion_protection,
                           decision_bindings, global_accounting, directory,
                           member_scope, source_closure_receipt=None,
                           historical_runtime_mapping=None):
    """Bind cached model verification to the exact source, conversion and current policy."""
    directory = Path(directory).resolve(strict=True)
    export = json.loads(Path(export_path).read_text(encoding='utf-8'))
    artifacts = [Path(export_path), Path(source_path)]
    if source_closure_receipt is not None and Path(source_closure_receipt).is_file():
        artifacts.append(Path(source_closure_receipt))
    artifacts.extend(path for path in directory.rglob('*') if path.is_file())
    source_identity = {
        'export_path': str(Path(export_path).resolve(strict=True)),
        'normalized_source_path': str(Path(source_path).resolve(strict=True)),
        'normalized_sessions': len(rows),
        'original_source': original,
        'export_complete_sessions': export['complete_sessions'],
        'source_output_sha256': export['outputs'][Path(source_path).name],
        'global_accounting': global_accounting,
        'member_scope': member_scope,
    }
    policy = {
        'eligibility_protection': protection['proof'],
        'conversion_protection': conversion_protection['proof'] if conversion_protection else None,
        'decision_holds': decision_contract(decision_bindings),
    }
    if historical_runtime_mapping is not None:
        policy['historical_runtime_mapping'] = historical_runtime_mapping
    return model_checkpoint_identity(
        source=source_identity,
        model={'name': model, 'base': str(Path(base).resolve(strict=True)),
               'tokenizer_inventory': tokenizer},
        policy=policy,
        artifacts=artifacts,
        verifier_files=_admission_verifier_files(),
    )


def _verify_conversion_shard(directory, shard, source_slice, *, expected_sources,
                            expected_omissions, expected_counts, protection, held_sources, fingerprint,
                            explicit_history_reasoning_policy=None):
    """Fully verify one durable source shard, including its complete graph views."""
    shard_path = Path(shard['path']).parent
    receipt = json.loads((shard_path / 'receipt.json').read_text())
    seen_omissions = set()
    for row in sessions(shard_path / 'omissions.jsonl'):
        key = canonical_json_sha256_hex([row['id'], row['source_row']])
        expected = expected_omissions.get(key)
        if key in seen_omissions or expected is None or any(row.get(name) != value for name, value in expected.items()):
            raise ValueError('native shard omission ledger differs from exact source omissions')
        seen_omissions.add(key)
    if seen_omissions != set(expected_omissions):
        raise ValueError('native shard omission ledger lost source omissions')

    source_by_hash = {canonical_json_sha256_hex(value): value for value in expected_sources}
    if len(source_by_hash) != len(expected_sources):
        raise ValueError('native shard source range repeats an exact positive session')
    graph_ids = set(); owed = set(); whole_sources = set()
    groups = {'train': set(), 'test': set()}; counts = Counter(); proofs = []; outputs = []
    for output in receipt['outputs']:
        path = shard_path / output['path']
        split = Path(output['path']).parent.name
        if split not in groups or path.resolve() != (shard_path / split / 'native-graphs.jsonl.zst').resolve():
            raise ValueError('native shard output escapes its split aggregate')
        rows = 0; tokens = 0; maximum = 0
        for graph in sessions(path):
            provenance = graph.get('conversion', {}).get('acquired_session', {})
            value = source_by_hash.get(provenance.get('normalized_session_sha256'))
            if value is None:
                raise ValueError('native shard graph has no exact eligible source from its bound range')
            expected, width, actual_split = verify_graph_source(
                graph, value, protection=protection, held_sources=held_sources, fingerprint=fingerprint,
                explicit_history_reasoning_policy=explicit_history_reasoning_policy)
            if actual_split != split or graph['id'] in graph_ids:
                raise ValueError('native shard duplicated a graph or stored it under the wrong split')
            graph_ids.add(graph['id']); owed.update(expected)
            if provenance['view'] == 'whole-trajectory':
                whole_sources.add(provenance['normalized_session_sha256'])
            rows += 1; tokens += width; maximum = max(maximum, width)
            groups[split].update(graph['source_groups']); counts['graphs:' + split] += 1
            if provenance['view'] != 'whole-trajectory':
                counts['auxiliary_reasoning_graphs:' + split] += 1
            proofs.append({'id': graph['id'], 'graph_sha256': canonical_json_sha256_hex(graph),
                'normalized_session_sha256': canonical_json_sha256_hex(value), 'source_row': value['source_row'],
                'view': provenance['view'], 'split': split, 'source_groups': graph['source_groups'],
                'tokens': width, 'source_outcome': value['trajectory']['outcome'], 'oracle_verified': False})
        if rows != output['rows']:
            raise ValueError('native shard graph count differs from its receipt')
        outputs.append({**output, 'tokens': tokens, 'max_row_tokens': maximum, 'split': split})
    if graph_ids != owed or whole_sources != set(source_by_hash):
        raise ValueError('native shard lacks a complete whole-session or required reasoning view')
    if +(Counter(expected_counts) + Counter(counts)) != +Counter(receipt['counts']):
        raise ValueError('native shard source and graph accounting differs from its receipt')
    return {'proofs': proofs, 'groups': {key: sorted(value) for key, value in groups.items()},
            'counts': dict(counts), 'outputs': outputs, 'graph_ids': sorted(graph_ids),
            'whole_sources': sorted(whole_sources), 'owed_ids': sorted(owed)}


def _sha256_concat(paths):
    import hashlib
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _verify_conversion_runtime(runtime, *, historical_observed_root=None,
                               historical_runtime_root=None):
    """Verify every converter dependency at its recorded or explicitly mapped path.

    A historical mapping is all-or-nothing: every declared dependency must be
    present under the supplied frozen root with the exact byte SHA already
    recorded in the immutable conversion receipt. The full receipt fingerprint
    is recomputed before any path is accepted. No dependency is waived or
    substituted individually.
    """
    files = runtime.get('files')
    observed = runtime.get('observed_paths')
    if (not isinstance(files, dict) or not files or not isinstance(observed, dict)
            or set(files) != set(observed)
            or runtime.get('fingerprint') != canonical_json_sha256_hex(files)):
        raise ValueError('native conversion runtime inventory or fingerprint is invalid')
    if historical_observed_root is None and historical_runtime_root is None:
        for name, binding in files.items():
            bound(observed[name], binding)
        return None
    if historical_observed_root is None or historical_runtime_root is None:
        raise ValueError('historical runtime mapping needs both recorded and frozen roots')
    observed_root = Path(historical_observed_root).resolve(strict=True)
    frozen_root = Path(historical_runtime_root).resolve(strict=True)
    mapped = {}
    for name, binding in files.items():
        recorded = Path(observed[name])
        if not recorded.is_absolute() or '..' in recorded.parts:
            raise ValueError('recorded converter dependency path is not a normalized absolute path: ' + name)
        try:
            relative = recorded.relative_to(observed_root)
        except ValueError:
            # Dependencies outside the explicitly mapped checkout keep their
            # original observed path and must still match the receipt exactly.
            original = recorded.resolve(strict=True)
            bound(original, binding)
            mapped[name] = {'path': str(original), **binding, 'location': 'original-outside-mapped-root'}
            continue
        candidate = (frozen_root / relative).resolve(strict=True)
        try:
            candidate.relative_to(frozen_root)
        except ValueError as exc:
            raise ValueError('historical converter dependency escapes the frozen root: ' + name) from exc
        bound(candidate, binding)
        mapped[name] = {'path': str(candidate), **binding, 'location': 'frozen-mapped-root'}
    return {
        'schema': 'natlang.acquired-conversion-historical-runtime-map/1',
        'recorded_root': str(observed_root), 'frozen_root': str(frozen_root),
        'runtime_fingerprint': runtime['fingerprint'],
        'files_sha256': canonical_json_sha256_hex(files),
        'mapped_files': mapped,
    }


def verify_conversion(directory, export_path, *, protection, held_sources, decision_bindings, source_rows,
                      conversion_protection=None, shard_checkpoint_root=None, checkpoint_context=None,
                      historical_observed_root=None, historical_runtime_root=None):
    directory = Path(directory).resolve(strict=True); receipt_path = directory / 'receipt.json'
    if checkpoint_context is None:
        checkpoint_context = {}
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 'natlang.acquired-full-session-graphs/1':
        raise ValueError('cohort member needs a completed native conversion receipt')
    _prepared_template_backbone(receipt)  # fail closed before validating any source rows
    bound(export_path, receipt['export_receipt'])
    source = bound(export_path.parent / 'full-trajectories.jsonl.zst', receipt['source'])
    numeric_protection = conversion_protection or protection
    if conversion_protection is not None:
        from scripts.neuralese_data.continue_acquired_protection import verify_protection_sets
        verify_protection_sets(numeric_protection, protection)
    if receipt['protection'] != numeric_protection['proof'] or decision_contract(receipt['decision_holds']) != decision_contract(decision_bindings):
        raise ValueError('native cohort conversion uses stale global protection or decision holds')
    state_path = bound(directory / 'resume-state.json', {'sha256': receipt['resume_state_sha256']})
    state = json.loads(state_path.read_text())
    if (receipt['source_record_count'] != len(source_rows)
            or state['completed_source_record_frontier'] != len(source_rows)
            or state['binding_sha256'] != canonical_json_sha256_hex(state['binding'])):
        raise ValueError('native cohort conversion lacks its complete durable source frontier')
    for key in ('export_receipt', 'protection', 'decision_holds', 'source_tokenizer_fingerprint',
                'expanded_tokenizer_fingerprint', 'prepared_tokenizer', 'held_group_closure_sha256'):
        if state['binding'].get(key) != receipt.get(key):
            raise ValueError('conversion receipt differs from durable shard policy: ' + key)
    if (state['binding']['source'] != {**receipt['source'], 'sessions': len(source_rows)}
            or state['binding']['runtime'] != {key: receipt['runtime'][key] for key in ('files', 'fingerprint')}
            or receipt['runtime']['fingerprint'] != canonical_json_sha256_hex(receipt['runtime']['files'])):
        raise ValueError('conversion source/runtime binding differs from its durable shard proof')
    _verify_conversion_runtime(receipt['runtime'],
        historical_observed_root=historical_observed_root,
        historical_runtime_root=historical_runtime_root)
    compatible_bindings = {}
    runtime_continuation = state.get('runtime_continuation')
    if runtime_continuation is not None:
        parent_binding = state.get('parent_binding')
        if parent_binding is None or runtime_continuation != prove_render_pending_runtime_continuation(
                parent_binding, state['binding'], runtime_continuation.get('parent_frontier')):
            raise ValueError('native cohort runtime continuation witness is invalid')
        if receipt.get('runtime_continuation') != runtime_continuation:
            raise ValueError('native cohort receipt differs from its runtime continuation witness')
        compatible_bindings[canonical_json_sha256_hex(parent_binding)] = runtime_continuation['parent_frontier']
    elif receipt.get('runtime_continuation') is not None or 'parent_binding' in state:
        raise ValueError('native cohort has an unbound runtime continuation')
    frontier, shards = completed_shards(directory / 'session-shards', state['binding'], len(source_rows),
        compatible_bindings=compatible_bindings)
    if frontier != len(source_rows) or shards != receipt['shards']:
        raise ValueError('native cohort conversion shard completion proof differs')
    if receipt['held_group_closure_sha256'] != canonical_json_sha256_hex(sorted(numeric_protection['held'])):
        raise ValueError('native cohort was not converted with the complete global held closure')
    bound(directory / 'omissions.jsonl', {'sha256': receipt['omissions_sha256']})
    expected_counts = Counter(); expected_sources = {}
    expected_omissions = {}
    for value in source_rows:
        if conversion_protection is not None:
            from scripts.neuralese_data.continue_acquired_protection import verify_session_transition
            verify_session_transition(value, numeric_protection, protection, held_sources)
        reason = session_exclusion(value, protection, held_sources)
        if reason:
            expected_counts[reason] += 1
            key = canonical_json_sha256_hex([value['trajectory']['id'], value['source_row']])
            if key in expected_omissions: raise ValueError('normalized source repeats an omitted occurrence')
            expected_omissions[key] = {'reason': reason, 'original_target_ids': source_target_ids(value)}
        else:
            try:
                _prevalidate_source_for_conversion(value, receipt)
            except RequestError as error:
                if error.code != 'tool-arguments':
                    raise
                reason = 'source-render-pending:tool-arguments'
                expected_counts[reason] += 1
                key = canonical_json_sha256_hex([value['trajectory']['id'], value['source_row']])
                if key in expected_omissions: raise ValueError('normalized source repeats an omitted occurrence')
                expected_omissions[key] = {'reason': reason,
                    'pending_status': 'source-faithful-review-pending',
                    'normalized_source_sha256': canonical_json_sha256_hex(value),
                    'original_source_file': value['source_file'],
                    'original_target_ids': source_target_ids(value), 'error_code': error.code}
                continue
            expected_counts['positive_sessions'] += 1
            expected_sources[canonical_json_sha256_hex(value)] = value
    if len(expected_sources) != expected_counts['positive_sessions']:
        raise ValueError('normalized source repeats exact positive session occurrences')
    # The aggregate omission ledger is a published artifact too. Per-shard
    # checks below prove each shard's omissions, but do not establish that the
    # aggregate file retained those same source occurrences. Recheck its
    # semantic contents against the independently derived full-source ledger.
    seen_aggregate_omissions = set()
    for row in sessions(directory / 'omissions.jsonl'):
        key = canonical_json_sha256_hex([row['id'], row['source_row']])
        expected = expected_omissions.get(key)
        if (key in seen_aggregate_omissions or expected is None
                or any(row.get(name) != value for name, value in expected.items())):
            raise ValueError('native aggregate omission ledger differs from exact source omissions')
        seen_aggregate_omissions.add(key)
    if seen_aggregate_omissions != set(expected_omissions):
        raise ValueError('native aggregate omission ledger lost source omissions')
    # Validate every committed shard independently and durably checkpoint its
    # complete graph/source proof. A preemption therefore repeats only the
    # unfinished shard, while the cohort-level duplicate, accounting, split,
    # parity and current-policy checks still run before publication.
    seen_occurrences = set()
    for value in source_rows:
        occurrence = canonical_json_sha256_hex([value['trajectory']['id'], value['source_row']])
        if occurrence in seen_occurrences:
            raise ValueError('normalized source repeats an occurrence across admission rows')
        seen_occurrences.add(occurrence)
    if len(receipt['outputs']) != 2 or {Path(item['path']).parent.name for item in receipt['outputs']} != {'train', 'test'}:
        raise ValueError('native cohort must bind exactly train and test aggregate outputs')
    aggregate_paths = {}
    for output in receipt['outputs']:
        path = bound(output['path'], output); split = path.parent.name
        if path.resolve() != (directory / split / 'native-graphs.jsonl.zst').resolve() or split not in {'train', 'test'}:
            raise ValueError('native cohort output escapes its completed aggregate')
        aggregate_paths[split] = (path, output)
    if set(aggregate_paths) != {'train', 'test'}:
        raise ValueError('native cohort aggregate split inventory is incomplete')

    shard_results = []
    for shard in shards:
        start, stop = shard['start'], shard['stop']
        source_slice = source_rows[start:stop]
        if len(source_slice) != stop - start:
            raise ValueError('native shard range escapes the normalized source')
        positive = [value for value in source_slice if canonical_json_sha256_hex(value) in expected_sources]
        slice_omissions = {}
        expected_shard_counts = Counter()
        for value in source_slice:
            occurrence = canonical_json_sha256_hex([value['trajectory']['id'], value['source_row']])
            omission_key = canonical_json_sha256_hex([value['trajectory']['id'], value['source_row']])
            if omission_key in expected_omissions:
                slice_omissions[omission_key] = expected_omissions[omission_key]
                expected_shard_counts[expected_omissions[omission_key]['reason']] += 1
            elif canonical_json_sha256_hex(value) in expected_sources:
                expected_shard_counts['positive_sessions'] += 1
            else:
                raise ValueError('source occurrence is neither eligible nor explicitly omitted')
        shard_path = Path(shard['path']).parent
        shard_receipt = json.loads((shard_path / 'receipt.json').read_text())
        shard_artifacts = [shard_path / 'receipt.json', shard_path / 'omissions.jsonl',
                           *(shard_path / item['path'] for item in shard_receipt['outputs'])]
        bound_artifacts = [export_path, source, receipt_path, state_path, *shard_artifacts]
        if checkpoint_context.get('source_closure_receipt'):
            bound_artifacts.append(checkpoint_context['source_closure_receipt'])
        shard_model_identity = dict(checkpoint_context)
        shard_model_identity['artifacts'] = artifact_bindings(bound_artifacts)
        shard_identity = {
            'schema': 'natlang.acquired-admission-shard-checkpoint-identity/1',
            'model_identity': shard_model_identity,
            'range': {'start': start, 'stop': stop,
                      'source_rows_sha256': canonical_json_sha256_hex(
                          [canonical_json_sha256_hex(value) for value in source_slice])},
            'artifacts': artifact_bindings(bound_artifacts),
        }
        def validate_shard(shard=shard, source_slice=source_slice, positive=positive,
                           slice_omissions=slice_omissions, expected_shard_counts=expected_shard_counts):
            return _verify_conversion_shard(
                directory, shard, source_slice, expected_sources=positive,
                expected_omissions=slice_omissions, expected_counts=expected_shard_counts,
                protection=protection, held_sources=held_sources,
                fingerprint=receipt['expanded_tokenizer_fingerprint'],
                explicit_history_reasoning_policy=receipt.get('explicit_history_reasoning_policy'))
        if shard_checkpoint_root is None:
            result = validate_shard()
        else:
            result, _, _ = load_or_compute_model_checkpoint(
                Path(shard_checkpoint_root) / f"{start:08d}-{stop:08d}", shard_identity, validate_shard)
        shard_results.append((shard, result))

    seen = set(); owed = set(); whole_sources = set()
    groups = {'train': set(), 'test': set()}; counts = Counter(); proofs = []
    output_stats = {split: {'rows': 0, 'tokens': 0, 'max_row_tokens': 0, 'shard_paths': []}
                    for split in ('train', 'test')}
    for shard, result in shard_results:
        shard_path = Path(shard['path']).parent
        owed.update(result['owed_ids'])
        for proof in result['proofs']:
            if proof['id'] in seen:
                raise ValueError('native cohort duplicated a graph across completed shards')
            seen.add(proof['id']); proofs.append(proof)
            if proof['view'] == 'whole-trajectory':
                whole_sources.add(proof['normalized_session_sha256'])
            if proof['normalized_session_sha256'] not in expected_sources:
                raise ValueError('cached native shard proof has no current eligible source')
        for split in ('train', 'test'):
            groups[split].update(result['groups'][split])
            output = next(item for item in result['outputs'] if item['split'] == split)
            stats = output_stats[split]
            stats['rows'] += output['rows']; stats['tokens'] += output['tokens']
            stats['max_row_tokens'] = max(stats['max_row_tokens'], output['max_row_tokens'])
            stats['shard_paths'].append(shard_path / output['path'])
        counts.update(result['counts'])
    for split, (aggregate_path, output) in aggregate_paths.items():
        stats = output_stats[split]
        if stats['rows'] != output['rows']:
            raise ValueError('aggregate graph count differs from completed source shards')
        if _sha256_concat(stats['shard_paths']) != output['sha256']:
            raise ValueError('aggregate output is not the exact ordered concatenation of verified shard outputs')
    if seen != owed or whole_sources != set(expected_sources):
        raise ValueError('native cohort lacks complete whole-session or required reasoning views')
    actual = Counter({**expected_counts, **counts})
    if +actual != +Counter(receipt['counts']) or groups['train'] & groups['test']:
        raise ValueError('native graph source accounting or train/held closure differs')
    outputs = [{**output, 'tokens': output_stats[Path(output['path']).parent.name]['tokens'],
                'max_row_tokens': output_stats[Path(output['path']).parent.name]['max_row_tokens'],
                'split': Path(output['path']).parent.name} for output in receipt['outputs']]
    # Preserve the historical aggregate proof order (train stream, then test
    # stream) even though validation and durable checkpoints are shard-local.
    proofs = [proof for split in ('train', 'test') for _, result in shard_results
              for proof in result['proofs'] if proof['split'] == split]
    return receipt, outputs, proofs, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True, help='one complete declared source cohort, e.g. D12')
    parser.add_argument('--native-root', type=Path, required=True, help='same cohort conversion directory')
    parser.add_argument('--protection', type=Path, required=True)
    parser.add_argument('--conversion-protection', type=Path, help='exact prior projection bound by unchanged numeric conversion receipts; eligibility uses --protection')
    parser.add_argument('--source-closure-receipt', type=Path,
                        help='prior full global source-closure receipt; all exact inputs are rechecked through the common hash cache')
    parser.add_argument('--force-full-source-closure', action='store_true',
                        help='ignore a closure receipt and stream-verify the complete normalized global source set')
    parser.add_argument('--id', required=True)
    parser.add_argument('--registry-path', required=True)
    parser.add_argument('--incremental-member', action='store_true',
                        help='admit one complete source export while retaining the full global protection closure')
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--historical-runtime-observed-root', type=Path,
                        help='root recorded in the existing conversion receipt observed_paths')
    parser.add_argument('--historical-runtime-root', type=Path,
                        help='frozen runtime tree whose complete dependency inventory must match the receipt')
    parser.add_argument('--model', action='append', required=True, help='NAME=exact local tokenizer base directory')
    args = parser.parse_args(argv)
    if (args.historical_runtime_observed_root is None) != (args.historical_runtime_root is None):
        raise ValueError('historical runtime verification requires both recorded and frozen roots')
    if args.historical_runtime_root is not None:
        args.historical_runtime_observed_root = args.historical_runtime_observed_root.resolve(strict=True)
        args.historical_runtime_root = args.historical_runtime_root.resolve(strict=True)
    # Registry paths are logical aliases; durable converter receipts name the
    # physical output directory. Compare the same exact files through either
    # entrypoint without treating symlink spelling as changed shard content.
    args.native_root = args.native_root.resolve(strict=True)
    complete = json.loads((args.exports / 'complete.json').read_text())
    paths = sorted(args.exports.glob('*/export.json'))
    if {str(path) for path in paths} != {row['receipt'] for row in complete['files'] if row['status'] in {'exported', 'reused'}}:
        raise ValueError('declared cohort must cover its complete normalized source file inventory')
    protection = current_protection(args.protection)
    conversion_protection = None
    if args.conversion_protection is None:
        global_accounting = verify_global_source_closure(args.protection, protection,
            receipt_path=args.source_closure_receipt,
            force_full_validation=args.force_full_source_closure)
    else:
        from scripts.neuralese_data.continue_acquired_protection import verify_protection_transition
        conversion_protection, current, global_accounting = verify_protection_transition(
            args.conversion_protection, args.protection,
            closure_receipt_path=args.source_closure_receipt,
            force_full_validation=args.force_full_source_closure)
        if current != protection:
            raise ValueError('current eligibility protection changed before cohort audit')
    member_scope = None
    if args.incremental_member:
        if len(paths) != 1:
            raise ValueError('incremental member admission accepts exactly one complete normalized source export')
        export_path = paths[0]
        export = json.loads(export_path.read_text())
        source_path = export_path.parent / 'full-trajectories.jsonl.zst'
        projection = json.loads(args.protection.read_text())
        member_scope = source_member_scope(projection, [{'path': str(export_path.resolve()),
            'sha256': sha256_file_hex(export_path), 'source_sha256': sha256_file_hex(source_path)}])
        if incremental_member_source_kind(export) is None:
            raise ValueError('incremental member needs an exact released SFT or single-row complete-trajectory contract')
    held_ids, decision_bindings = load_decision_holds(sorted((ROOT / 'training/decision-holds').glob('*.jsonl')))
    held_sources = held_ids | {value.split(':decision:', 1)[0] for value in held_ids}
    out = args.native_root / ('admission-' + args.id)
    if out.exists(): raise ValueError('admission output is immutable; choose a new cohort identity')
    model_items = [item.split('=', 1) for item in args.model]
    if any(len(item) != 2 or not item[0] or not item[1] for item in model_items):
        raise ValueError('each model needs NAME=exact local tokenizer base directory')
    models = dict(model_items)
    if len(models) != len(args.model) or len(models) < 2:
        raise ValueError('complete native cohort requires distinct explicit model/tokenizer views')
    for base in models.values():
        if not (Path(base).resolve(strict=True) / 'config.json').is_file():
            raise ValueError('native cohort needs an existing exact model/tokenizer base')
    inventories = {name: tokenizer_inventory(base) for name, base in models.items()}
    manifests = {model: {'train': [], 'test': []} for model in models}
    by_model = {}; proof_rows = []; include = []; bindings = []; normalized_count = 0
    validation_checkpoints = []
    checkpoint_rechecks = []
    model_source_views = {}
    all_groups = {'train': set(), 'test': set()}
    for export_path in paths:
        export = json.loads(export_path.read_text()); source = export_path.parent / 'full-trajectories.jsonl.zst'
        bound(source, {'sha256': export['outputs'][source.name]})
        original = export['contract']['source']; bound(args.original_root / original['path'], original)
        rows = list(sessions(source)); normalized_count += len(rows)
        if len(rows) != export['complete_sessions']: raise ValueError('normalized cohort source count differs')
        if not rows: continue
        for row in rows:
            if row['source_file'] != original or row['source_revision'] != export['contract']['revision']:
                raise ValueError('normalized session changed its pinned original source identity')
        model_source_views = {}
        for model in models:
            directory = native_conversion_directory(args.native_root, export_path, export, model)
            conversion_receipt = json.loads((directory / 'receipt.json').read_text())
            runtime_authority = _verify_conversion_runtime(conversion_receipt['runtime'],
                historical_observed_root=args.historical_runtime_observed_root,
                historical_runtime_root=args.historical_runtime_root)
            identity = _model_resume_identity(
                export_path=export_path, source_path=source, original=original, rows=rows,
                model=model, base=models[model], tokenizer=inventories[model],
                protection=protection, conversion_protection=conversion_protection,
                decision_bindings=decision_bindings, global_accounting=global_accounting,
                directory=directory, member_scope=member_scope,
                source_closure_receipt=args.source_closure_receipt,
                historical_runtime_mapping=runtime_authority)
            checkpoint_root = args.native_root / '.admission-checkpoints' / args.id / export_path.parent.name / model
            def validate_model():
                receipt, outputs, proofs, groups = verify_conversion(directory, export_path, protection=protection,
                    held_sources=held_sources, decision_bindings=decision_bindings, source_rows=rows,
                    conversion_protection=conversion_protection,
                    shard_checkpoint_root=checkpoint_root / 'shards', checkpoint_context=identity,
                    historical_observed_root=args.historical_runtime_observed_root,
                    historical_runtime_root=args.historical_runtime_root)
                runtime_authority = _verify_conversion_runtime(receipt['runtime'],
                    historical_observed_root=args.historical_runtime_observed_root,
                    historical_runtime_root=args.historical_runtime_root)
                return {'receipt': receipt, 'outputs': outputs, 'proofs': proofs,
                        'groups': {split: sorted(values) for split, values in groups.items()},
                        **({'runtime_authority': runtime_authority} if runtime_authority is not None else {})}
            result, checkpoint, resumed = load_or_compute_model_checkpoint(
                checkpoint_root, identity, validate_model)
            receipt, outputs, proofs = result['receipt'], result['outputs'], result['proofs']
            runtime_authority = result.get('runtime_authority')
            groups = {split: set(values) for split, values in result['groups'].items()}
            validation_checkpoints.append({
                'model': model, 'source_export': str(export_path.resolve()),
                'path': str(checkpoint.resolve()), 'sha256': sha256_file_hex_cached(
                    checkpoint, trust_archive_receipt=False), 'reused': resumed,
                'identity_sha256': canonical_json_sha256_hex(identity),
            })
            checkpoint_rechecks.append({
                'identity': identity, 'export_path': export_path, 'source_path': source,
                'original': original, 'rows': rows, 'model': model, 'base': models[model],
                'tokenizer': inventories[model], 'directory': directory,
                'runtime_authority': runtime_authority,
            })
            if receipt['prepared_tokenizer']['base_files'] != inventories[model]:
                raise ValueError('native cohort model selector differs from its exact tokenizer base inventory')
            whole = whole_trajectory_source_view(proofs)
            pending = set()
            for omission in sessions(directory / 'omissions.jsonl'):
                if omission.get('reason') == 'source-render-pending:tool-arguments':
                    pending.add((omission['normalized_source_sha256'], omission['source_row']))
            model_source_views[model] = {'backbone': _prepared_template_backbone(receipt),
                'accepted': whole, 'pending': pending}
            summary = by_model.setdefault(model, Counter()); summary.update(receipt['counts'])
            for split in groups: all_groups[split].update(groups[split])
            for output in outputs:
                relative = os.path.relpath(output['path'], out)
                if output['rows']:
                    manifests[model][output['split']].append({'kind': 'native-graphs', 'path': relative,
                        'sha256': output['sha256'], 'weight': 1., 'epochs': 1})
                summary['tokens:' + output['split']] += output['tokens']
                summary['max_row_tokens'] = max(summary['max_row_tokens'], output['max_row_tokens'])
                include.append(str(Path(output['path']).relative_to(args.native_root)))
            for name in ('receipt.json', 'omissions.jsonl', 'resume-state.json'):
                include.append(str((directory / name).relative_to(args.native_root)))
            bindings.append({'model': model, 'receipt': {'path': str(directory / 'receipt.json'),
                'sha256': sha256_file_hex(directory / 'receipt.json')}, 'outputs': outputs,
                **({'historical_runtime_authority': runtime_authority} if runtime_authority is not None else {}),
                **({'conversion_protection': receipt['protection']} if conversion_protection is not None else {})})
            proof_rows.extend({**proof, 'model': model, 'original_source': original} for proof in proofs)
        verify_model_source_parity(model_source_views)
    if all_groups['train'] & all_groups['test']: raise ValueError('cohort global train/held group intersection')
    if (conversion_protection is not None and current_protection(args.conversion_protection) != conversion_protection):
        raise ValueError('prior numeric conversion protection changed during cohort verification')
    if current_protection(args.protection) != protection or load_decision_holds(
            sorted((ROOT / 'training/decision-holds').glob('*.jsonl')))[1] != decision_bindings:
        raise ValueError('current protections changed during cohort verification')
    if args.source_closure_receipt is not None and not args.force_full_source_closure:
        if verify_global_source_closure(args.protection, protection,
                receipt_path=args.source_closure_receipt) != global_accounting:
            raise ValueError('full acquired-source closure changed during cohort verification')
    for check in checkpoint_rechecks:
        current_identity = _model_resume_identity(
            export_path=check['export_path'], source_path=check['source_path'],
            original=check['original'], rows=check['rows'], model=check['model'],
            base=check['base'], tokenizer=check['tokenizer'], protection=protection,
            conversion_protection=conversion_protection, decision_bindings=decision_bindings,
            global_accounting=global_accounting, directory=check['directory'],
            member_scope=member_scope, source_closure_receipt=args.source_closure_receipt,
            historical_runtime_mapping=_verify_conversion_runtime(
                json.loads((check['directory'] / 'receipt.json').read_text())['runtime'],
                historical_observed_root=args.historical_runtime_observed_root,
                historical_runtime_root=args.historical_runtime_root))
        if current_identity != check['identity']:
            raise ValueError('model conversion artifacts changed before admission publication')
    out.mkdir()
    with (out / 'row-proof.jsonl').open('w') as stream:
        for proof in proof_rows: stream.write(json.dumps(proof, ensure_ascii=False, separators=(',', ':')) + '\n')
    for model, splits in manifests.items():
        for split, sources in splits.items():
            if sources: write_json_durable(out / f'{model}-{split}-sources.json', {'schema': 'natlang.native-graph-sources/1', 'sources': sources})
    audit = {'schema': 'natlang.acquired-session-admission-audit/1', 'status': 'passed',
        'verifier_sha256': sha256_file_hex(__file__),
        'admission_class': 'source-curated-third-party-SFT', 'oracle_verified': False,
        'native_execution_verified': False, 'training_admission': False,
        'complete_cohort': {'path': str(args.exports / 'complete.json'), 'sha256': sha256_file_hex(args.exports / 'complete.json')},
        'normalized_sessions': normalized_count, 'by_model': by_model, 'conversions': bindings,
        'complete_acquired_source_accounting': global_accounting,
        'protection': protection['proof'], 'decision_holds': decision_bindings,
        'pending_source_groups_sha256': canonical_json_sha256_hex(sorted(protection['pending'])),
        'completed_model_validation_checkpoints': validation_checkpoints,
        'model_tokenizer_inventories': inventories,
        'train_held_group_intersections': 0, 'row_proof_sha256': sha256_file_hex(out / 'row-proof.jsonl'),
        'training_sources': 'aggregate outputs only; durability shards are not additional sources'}
    if member_scope is not None:
        audit['source_scope'] = member_scope
    if conversion_protection is not None:
        from scripts.neuralese_data.continue_acquired_protection import projection_evidence
        bundles = {role: projection_evidence(path, out) for role, path in (
            ('conversion', args.conversion_protection), ('eligibility', args.protection))}
        for bundle in bundles.values():
            include.extend(str((out / item['path']).relative_to(args.native_root))
                           for item in [bundle['manifest'], *bundle['files']])
        audit['conversion_protection_transition'] = {
            'schema': 'natlang.acquired-conversion-protection-transition/1',
            'prior_conversion_protection': conversion_protection['proof'],
            'portable_projections': bundles,
            'eligibility_protection': protection['proof'], 'numeric_rerendered': False,
            'all_original_source_records_checked': normalized_count,
            'all_original_source_eligibility_and_splits_unchanged': True,
            'original_conversion_receipts_unchanged': True,
            'new_global_source_closure': global_accounting}
    write_json_durable(out / 'audit.json', audit)
    include.extend(str(path.relative_to(args.native_root)) for path in out.iterdir() if path.is_file())
    reason = ('One complete normalized source member is source-curated, not oracle or native-execution verified. '
        'Both pinned tokenizer views preserve the full global normalized-source held/quality/pending closure and exact decision holds. '
        'Other normalized source members remain explicitly pending.') if member_scope is not None else (
        'Source-curated third-party SFT, not oracle or native-execution verified. Exact complete source sessions, '
        'full reasoning views and both pinned tokenizer variants preserve current global held/quality/pending '
        'and exact decision holds. Eligibility is not training exposure.')
    entry = {'id': args.id, 'owner': 'dgx', 'path': args.registry_path, 'include': sorted(include),
        'source_class': 'source-curated-complete-acquired-agent-native-session-graphs',
        'training_admission': True, 'training_admission_reason': reason,
        'split': 'original source split and complete cross-source global held-group closure; no truncation',
        'license': 'exact source license facts retained in each graph acquired_session provenance',
        'build': {'audit': str((out / 'audit.json').relative_to(args.native_root)),
            'audit_sha256': sha256_file_hex(out / 'audit.json')}, 'documents': by_model}
    write_json_durable(out / 'registry-proposal.json', entry)
    print(json.dumps({'output': str(out), 'by_model': by_model, 'normalized_sessions': normalized_count,
                      'registry_written': False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    main()
