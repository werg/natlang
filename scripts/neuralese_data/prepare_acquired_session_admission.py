"""Exact source-curated full-session cohort verifier and registry proposal."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease
from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.model.native_model import tokenizer_inventory
from scripts.neuralese_data.build_acquired_full_session_graphs import (
    sessions, source_target_ids, completed_shards, session_exclusion)
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds

ROOT = Path(os.environ.get('NATLANG_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()


def bound(path, binding):
    path = Path(path)
    if sha256_file_hex(path) != binding['sha256'] or ('bytes' in binding and path.stat().st_size != binding['bytes']):
        raise ValueError('immutable acquired artifact changed: ' + str(path))
    return path


def decision_contract(bindings):
    """Checkout location is observed metadata; exact hold filenames/bytes bind policy."""
    return sorted((Path(item['path']).name, item['sha256'], item['bytes'], item['kind']) for item in bindings)


def source_occurrence(value):
    return 'text:' + value['trajectory']['id'] + ':source-occurrence:' + canonical_json_sha256_hex(
        [value['source_file']['sha256'], value['source_row']])


def acquired_source_closure(path, protection):
    """Reconstruct all source components, including not-yet-converted siblings."""
    manifest = json.loads(Path(path).read_text()); bindings = manifest.get('normalized_export_bindings')
    if not isinstance(bindings, list) or not bindings:
        raise ValueError('acquired admission needs the declared complete cross-source held closure')
    held = set(); counts = Counter(); seen = set(); group_rows = []
    inherited = manifest.get('inherited_source_components')
    if inherited:
        parent = Path(path).parent
        component_path = bound(parent / inherited['components']['path'], inherited['components'])
        projection_path = bound(parent / inherited['projection']['path'], inherited['projection'])
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
        export_path = bound(binding['path'], binding)
        if export_path in seen: raise ValueError('global source closure repeats an export receipt')
        seen.add(export_path); export = json.loads(export_path.read_text())
        source = bound(export_path.parent / 'full-trajectories.jsonl.zst',
            {'sha256': binding['source_sha256']})
        if export['outputs'][source.name] != binding['source_sha256']:
            raise ValueError('global source export and closure disagree on immutable session bytes')
        rows = 0
        for value in sessions(source):
            trajectory = value['trajectory']; rows += 1
            if trajectory['split'] not in {'train', 'test', 'validation'}:
                raise ValueError('global source closure contains an undeclared split')
            if trajectory['split'] != 'train': held.update(trajectory['split_groups'])
            group_rows.append(trajectory['split_groups'])
        if rows != export['complete_sessions']: raise ValueError('global source closure lost session accounting')
        counts[export_path.parent.parent.name] += rows
    declared = set()
    for directory in {export_path.parent.parent for export_path in seen}:
        complete_path = directory / 'complete.json'
        complete = json.loads(complete_path.read_text())
        declared.update(Path(item['receipt']) for item in complete['files'] if item['status'] in {'exported', 'reused'})
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


def _source_closure_artifacts(path, manifest):
    """Return the complete exact input closure, including per-directory inventories."""
    path = Path(path).resolve(strict=True)
    artifacts = {}

    def add(candidate):
        candidate = Path(candidate).resolve(strict=True)
        artifacts[str(candidate)] = {'path': str(candidate), 'sha256': sha256_file_hex(candidate),
                                     'bytes': candidate.stat().st_size}

    for item in manifest.get('files', []):
        add(bound(path.parent / item['path'], item))
    component_name = manifest.get('source_group_components')
    if component_name is not None:
        component_binding = next((item for item in manifest.get('files', [])
                                  if item.get('path') == component_name), None)
        if component_binding is None:
            raise ValueError('source-group component inventory lacks an explicit manifest binding')
        add(bound(path.parent / component_name, component_binding))
    inherited = manifest.get('inherited_source_components') or {}
    for item in inherited.values():
        add(bound(path.parent / item['path'], item))
    bindings = manifest.get('normalized_export_bindings')
    if not isinstance(bindings, list) or not bindings:
        raise ValueError('acquired closure receipt needs the complete normalized export set')
    directories = set()
    for binding in bindings:
        export_path = bound(binding['path'], binding)
        export = json.loads(export_path.read_text())
        source = bound(export_path.parent / 'full-trajectories.jsonl.zst',
                       {'sha256': binding['source_sha256']})
        if export.get('outputs', {}).get(source.name) != binding['source_sha256']:
            raise ValueError('closure receipt source/export binding differs')
        add(export_path); add(source)
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


def verify_global_source_closure(path, protection, *, receipt_path=None, write_receipt_path=None):
    """Reject direct-key protection that misses a transitive source sibling."""
    if receipt_path is not None and not Path(receipt_path).exists():
        receipt_path = Path(receipt_path)
        with acquire_output_lease(receipt_path.parent):
            if receipt_path.exists():
                return verify_global_source_closure(path, protection, receipt_path=receipt_path)
            return verify_global_source_closure(path, protection, write_receipt_path=receipt_path)
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
    accounting, closed, components = acquired_source_closure(path, protection)
    if any(closed[key] != protection[key] for key in closed):
        raise ValueError('global acquired source protection lacks complete connected-group closure')
    if manifest.get('source_group_components'):
        stored = [row['source_groups'] for row in sessions(Path(path).parent / manifest['source_group_components'])]
        if stored != components:
            raise ValueError('protected source component inventory differs from complete normalized sources')
    if write_receipt_path is not None:
        protection_binding = {key: sorted(protection[key]) for key in ('held', 'quality', 'pending')}
        receipt = {'schema': SOURCE_CLOSURE_RECEIPT_SCHEMA,
            'manifest': {'path': str(manifest_path), 'sha256': sha256_file_hex(manifest_path),
                         'bytes': manifest_path.stat().st_size},
            'protection': protection_binding,
            'protection_sha256': canonical_json_sha256_hex(protection_binding),
            'input_artifacts': _source_closure_artifacts(manifest_path, manifest),
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


def verify_graph_source(graph, value, *, protection, held_sources, fingerprint):
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
    if graph.get('source_groups') != sorted(set(trajectory['split_groups'])):
        raise ValueError('native graph changed original source groups')
    groups = set(trajectory['split_groups'])
    split = 'test' if groups & protection['held'] else 'train'
    if trajectory['split'] not in {'train', 'test', 'validation'} or (
            trajectory['split'] != 'train' and split != 'test') or graph.get('split') != split:
        raise ValueError('native graph violates global source-declared held closure')
    base = source_occurrence(value); view = provenance.get('view')
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


def verify_conversion(directory, export_path, *, protection, held_sources, decision_bindings, source_rows, conversion_protection=None):
    directory = Path(directory).resolve(strict=True); receipt_path = directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 'natlang.acquired-full-session-graphs/1':
        raise ValueError('cohort member needs a completed native conversion receipt')
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
    for name, binding in receipt['runtime']['files'].items():
        bound(receipt['runtime']['observed_paths'][name], binding)
    frontier, shards = completed_shards(directory / 'session-shards', state['binding'], len(source_rows))
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
            expected_omissions[key] = (reason, source_target_ids(value))
        else:
            expected_counts['positive_sessions'] += 1
            expected_sources[canonical_json_sha256_hex(value)] = value
    if len(expected_sources) != expected_counts['positive_sessions']:
        raise ValueError('normalized source repeats exact positive session occurrences')
    seen_omissions = set()
    for row in sessions(directory / 'omissions.jsonl'):
        key = canonical_json_sha256_hex([row['id'], row['source_row']])
        if key in seen_omissions or expected_omissions.get(key) != (row['reason'], row['original_target_ids']):
            raise ValueError('native omission ledger differs from exact original excluded occurrences')
        seen_omissions.add(key)
    if seen_omissions != set(expected_omissions):
        raise ValueError('native omission ledger lost excluded source sessions')
    seen = set(); owed = set(); groups = {'train': set(), 'test': set()}; counts = Counter(); proofs = []; outputs = []
    if len(receipt['outputs']) != 2 or {Path(item['path']).parent.name for item in receipt['outputs']} != {'train', 'test'}:
        raise ValueError('native cohort must bind exactly train and test aggregate outputs')
    for output in receipt['outputs']:
        path = bound(output['path'], output); split = path.parent.name
        if path.resolve() != (directory / split / 'native-graphs.jsonl.zst').resolve() or split not in groups:
            raise ValueError('native cohort output escapes its completed aggregate')
        rows = 0; tokens = 0; maximum = 0
        for graph in sessions(path):
            provenance = graph.get('conversion', {}).get('acquired_session', {})
            value = expected_sources.get(provenance.get('normalized_session_sha256'))
            if value is None: raise ValueError('native graph has no exact eligible normalized source')
            expected, width, actual_split = verify_graph_source(graph, value, protection=protection,
                held_sources=held_sources, fingerprint=receipt['expanded_tokenizer_fingerprint'])
            if actual_split != split or graph['id'] in seen:
                raise ValueError('native graph duplicated or stored under a wrong split')
            owed.update(expected); seen.add(graph['id']); rows += 1; tokens += width; maximum = max(maximum, width)
            groups[split].update(graph['source_groups']); counts['graphs:' + split] += 1
            if provenance['view'] != 'whole-trajectory': counts['auxiliary_reasoning_graphs:' + split] += 1
            proofs.append({'id': graph['id'], 'graph_sha256': canonical_json_sha256_hex(graph),
                'normalized_session_sha256': canonical_json_sha256_hex(value), 'source_row': value['source_row'],
                'view': provenance['view'], 'split': split, 'source_groups': graph['source_groups'],
                'tokens': width, 'source_outcome': value['trajectory']['outcome'], 'oracle_verified': False})
        if rows != output['rows']: raise ValueError('aggregate graph count differs from conversion receipt')
        outputs.append({**output, 'tokens': tokens, 'max_row_tokens': maximum, 'split': split})
    whole_sources = {proof['normalized_session_sha256'] for proof in proofs if proof['view'] == 'whole-trajectory'}
    if seen != owed or whole_sources != set(expected_sources):
        raise ValueError('native cohort lacks complete whole-session or required reasoning views')
    actual = Counter({**expected_counts, **counts})
    if +actual != +Counter(receipt['counts']) or groups['train'] & groups['test']:
        raise ValueError('native graph source accounting or train/held closure differs')
    return receipt, outputs, proofs, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True, help='one complete declared source cohort, e.g. D12')
    parser.add_argument('--native-root', type=Path, required=True, help='same cohort conversion directory')
    parser.add_argument('--protection', type=Path, required=True)
    parser.add_argument('--conversion-protection', type=Path, help='exact prior projection bound by unchanged numeric conversion receipts; eligibility uses --protection')
    parser.add_argument('--id', required=True)
    parser.add_argument('--registry-path', required=True)
    parser.add_argument('--incremental-member', action='store_true',
                        help='admit one complete source export while retaining the full global protection closure')
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True, help='NAME=exact local tokenizer base directory')
    args = parser.parse_args(argv)
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
        global_accounting = verify_global_source_closure(args.protection, protection)
    else:
        from scripts.neuralese_data.continue_acquired_protection import verify_protection_transition
        conversion_protection, current, global_accounting = verify_protection_transition(
            args.conversion_protection, args.protection)
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
        source_kind = export.get('contract', {}).get('source', {}).get('path', '')
        if not source_kind.startswith('jsonl_sft/'):
            raise ValueError('incremental SFT member admission cannot consume a preference or non-SFT source')
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
        twin = None
        for model in models:
            directory = args.native_root / export_path.parent.name / model
            receipt, outputs, proofs, groups = verify_conversion(directory, export_path, protection=protection,
                held_sources=held_sources, decision_bindings=decision_bindings, source_rows=rows,
                conversion_protection=conversion_protection)
            if receipt['prepared_tokenizer']['base_files'] != inventories[model]:
                raise ValueError('native cohort model selector differs from its exact tokenizer base inventory')
            identities = {(p['id'], p['normalized_session_sha256'], p['split'], p['view']) for p in proofs if p['view'] == 'whole-trajectory'}
            if twin is not None and identities != twin: raise ValueError('model views differ in complete source session membership')
            twin = identities
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
                **({'conversion_protection': receipt['protection']} if conversion_protection is not None else {})})
            proof_rows.extend({**proof, 'model': model, 'original_source': original} for proof in proofs)
    if all_groups['train'] & all_groups['test']: raise ValueError('cohort global train/held group intersection')
    if (conversion_protection is not None and current_protection(args.conversion_protection) != conversion_protection):
        raise ValueError('prior numeric conversion protection changed during cohort verification')
    if current_protection(args.protection) != protection or load_decision_holds(
            sorted((ROOT / 'training/decision-holds').glob('*.jsonl')))[1] != decision_bindings:
        raise ValueError('current protections changed during cohort verification')
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
    reason = ('One complete normalized SFT source member is source-curated, not oracle or native-execution verified. '
        'Both pinned tokenizer views preserve the full global 15-export held/quality/pending closure and exact decision holds. '
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
