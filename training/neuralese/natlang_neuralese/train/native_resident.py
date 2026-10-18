"""Prepared, data-only adoption for the existing native SFT runner.

current.json is the sole commit record. Other files are recoverable projections;
checkpoint exposure remains authoritative for weights and completed updates.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex_cached as sha, sha256_file_hex
from ..common.jsonio import write_json_durable as write_json

SCHEMA = 'natlang.native-resident-adoption/1'


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def bound(value):
    path = Path(value['path']).resolve(strict=True)
    if sha(path) != value['sha256']:
        raise ValueError('resident artifact changed: ' + str(path))
    return path


def code_inventory(package):
    package = Path(package)
    return {str(p.relative_to(package)): sha256_file_hex(p) for p in sorted(package.rglob('*.py'))}


@contextmanager
def locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def current(directory):
    path = Path(directory) / 'current.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('schema') != SCHEMA or digest(value['payload']) != value['generation']:
        raise ValueError('resident commit is corrupt')
    return value


def repair(directory, state):
    """Idempotently project the already committed generation, never invent one."""
    directory = Path(directory)
    payload = state['payload']
    write_json(directory.parent / 'recipe-plan.json', payload['plan'])
    write_json(directory / 'active-binding.json', {'generation': state['generation'],
        'recipe': payload['recipe'], 'stage_id': payload['stage_id'],
        'train': payload['train'], 'held': payload['held'], 'adopted_at_step': state['at_global_step'],
        'checkpoint_required_for_weights': True})


def recover(directory):
    with locked(directory):
        state = current(directory)
        if state:
            # Verify the recipe before repairing anything a runner will consume.
            bound({'path': state['payload']['recipe']['path'],
                   'sha256': state['payload']['recipe']['file_sha256']})
            from .recipe import recipe_identity_sha256, load_recipe
            if (recipe_identity_sha256(state['payload']['recipe']['path']) != state['payload']['recipe']['sha256']
                    or load_recipe(Path(state['payload']['recipe']['path'])) != state['payload']['plan']['recipe']):
                raise ValueError('resident committed recipe policy/inheritance changed')
            repair(directory, state)
        return state


def _source_witness(source, package=None):
    """Bind the existing validation-cache entries; no second graph parser."""
    from ..common import row_index
    if source.kind == 'manifest':
        return [w for child in source.children for w in _source_witness(child, package)]
    package = Path(package or Path(__file__).resolve().parents[1])
    names = ('data/native_sources.py', 'data/native_graph_schema.py', 'model/native_graph_plan.py',
             'data/native_pair_schema.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')
    key = {'source_sha256': source.file_sha256, 'tokenizer': source.tokenizer_fingerprint,
           'split': source.required_split, 'policy_sha256': digest({n: sha256_file_hex(package/n) for n in names}),
           'kind': source.kind, 'pair_model': source.pair_model, 'role_ids': list(source.role_ids)}
    path = row_index.entry_path('native-graphs', digest(key))
    value = row_index.load(path)
    if not value or value.get('key') != key or value.get('metadata_sha256') != digest(source.entries):
        raise ValueError('resident preparation requires the exact completed source index: ' + str(source.path))
    return [{'cache': binding(path), 'key': key, 'metadata_sha256': value['metadata_sha256']}]


def _check_indexes(witnesses):
    from ..common import row_index
    for witness in witnesses:
        cache = bound(witness['cache'])
        if cache != row_index.entry_path('native-graphs', digest(witness['key'])).resolve():
            raise ValueError('resident cache must be staged in the learner row-index cache')
        value = row_index.load(cache)
        if (not value or value.get('key') != witness['key']
                or value.get('metadata_sha256') != witness['metadata_sha256']
                or digest(value.get('entries')) != witness['metadata_sha256']):
            raise ValueError('resident warmed source index changed')


def _contract(plan, proposed, stage_id):
    """One stage may append active data and/or mandatory pending declarations."""
    from .recipe import _data_continuation, _pending_source_continuation
    old_inputs, new_inputs = plan['stage_inputs'], proposed['stage_inputs']
    if plan['code'] != proposed['code'] or plan.get('device') != proposed.get('device'):
        raise ValueError('resident append cannot change runtime or device')
    if set(old_inputs) != set(new_inputs):
        raise ValueError('resident append cannot change stage membership')
    for sid in old_inputs:
        if sid != stage_id and old_inputs[sid] != new_inputs[sid]:
            raise ValueError('resident append cannot change another stage')
    old, new = old_inputs[stage_id], new_inputs[stage_id]
    if old.get('foundation_native_graphs') != new.get('foundation_native_graphs'):
        raise ValueError('resident append cannot change fixed foundation')
    if old.get('withdrawn_native_graphs') != new.get('withdrawn_native_graphs'):
        raise ValueError('resident append cannot withdraw graphs')
    pending_role = 'pending_native_sources'
    pending_changed = old.get(pending_role) != new.get(pending_role)
    if pending_changed and (not isinstance(new.get(pending_role), dict) or
            not _pending_source_continuation(old, {**old, pending_role: new[pending_role]})):
        raise ValueError('resident pending declarations must preserve all prior rows, quotas and dependencies')
    without_pending = lambda values: {k:v for k,v in values.items() if k != pending_role}
    active_changed = without_pending(old) != without_pending(new)
    if active_changed and not _data_continuation(without_pending(old), without_pending(new)):
        raise ValueError('resident append requires the exact training-source predecessor')
    if not active_changed and not pending_changed:
        raise ValueError('resident request must add active data or mandatory pending work')
    recipes = []
    for recipe in (plan['recipe'], proposed['recipe']):
        value = copy.deepcopy(recipe)
        value.pop('input_bindings', None)
        for stage in value['stages']:
            if stage['id'] == stage_id:
                if stage['kind'] != 'native_state_sft':
                    raise ValueError('resident append supports supervised native graphs only')
                for key in ('native_graphs', 'continued_native_graphs', 'held_native_graphs', 'continued_held_native_graphs', pending_role):
                    stage.get('inputs', {}).pop(key, None)
        recipes.append(value)
    if recipes[0] != recipes[1]:
        raise ValueError('resident append cannot change parameters, runtime, objective or policy')


def pending_declarations(plan, stage_id, tokenizer_fingerprint):
    """Authenticate the same declaration consumed by checkpoint progress/readiness."""
    from .native_pending_sources import load_pending_sources
    reference = plan['stage_inputs'][stage_id].get('pending_native_sources')
    return ([load_pending_sources(bound(reference), tokenizer_fingerprint=tokenizer_fingerprint)]
            if reference else [])


def validate_source_obligations(plan, stage_id, train, declarations, withdrawal_history=()):
    """Reject withdrawn active/pending targets before the sole commit point."""
    from .native_continuation import _retain_pending_sources, reject_withdrawn_targets
    history = copy.deepcopy(list(withdrawal_history))
    reference = plan['stage_inputs'][stage_id].get('withdrawn_native_graphs')
    if reference:
        proof = json.loads(bound(reference).read_text())
        if proof.get('schema') != 'natlang.native-graph-withdrawal/1':
            raise ValueError('resident withdrawal proof schema differs')
        history.append({'rows': proof['rows']})
    frame = {'source_row_bindings':train.bindings, 'source_row_order':train.ordered_ids(),
        'exposure_counts':{}, 'options':{'withdrawal_history':history,
        '_pending_native_source_declarations':declarations}}
    reject_withdrawn_targets(frame, train.bindings)
    return _retain_pending_sources(frame, {})


def same_geometry_identity(previous, current):
    """Compare memory execution, preserving precision/packing and declared profiles.

    KEEP changes history rendering, not tensor shapes or kernels. Its sole
    optional config flag is omitted for this memory comparison; actual recipe,
    saved profile history and actor attestation remain separately authenticated.
    """
    for value in (previous,current):
        model=value.get('model_config') if isinstance(value,dict) else None
        if (isinstance(model,dict) and 'history_reasoning_policy' in model
                and model['history_reasoning_policy'] != 'keep'):
            return False
    def explicit(value):
        if not isinstance(value,dict):return value
        result=dict(value)
        if isinstance(value.get('model_config'),dict):
            result['model_config']=dict(value['model_config'])
            result['model_config'].setdefault('native_attention_precision','ieee')
            result['model_config'].pop('history_reasoning_policy',None)
        return result
    return explicit(previous)==explicit(current)


def geometry_capacity(value, *, identity, code):
    certificate = json.loads(bound(value).read_text())
    if (certificate.get('schema') != 'natlang.native-resident-geometry-qualification/1'
            or certificate.get('passed') is not True or not same_geometry_identity(certificate.get('identity'), identity)
            or certificate.get('runtime_code_sha256') != digest(code)):
        raise ValueError('resident geometry qualification differs from actual model/runtime/packing')
    bound(certificate['evidence'])
    limits = certificate.get('limits') or {}
    if set(limits) != {'forward_tokens', 'kv_slots'} or any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError('resident geometry qualification needs positive measured forward/KV-slot limits')
    return limits


def new_geometry(source, first):
    maximum = {'forward_tokens': 0, 'kv_slots': 0}
    for example in source.iter_indices(range(first, len(source))):
        maximum['forward_tokens'] = max(maximum['forward_tokens'], example.plan.token_count)
        maximum['kv_slots'] = max(maximum['kv_slots'], len(example.plan.slot_sources))
    return maximum



def geometry_from_evidence(source, first, evidence, *, package, source_preparation=None):
    """Reuse a bound completed CPU parse, conservatively covering a new suffix.

    Legacy tiered prewarm receipts parsed TRAIN only. Their held indexes are
    admission evidence, never held geometry. A prepared resident request covers
    both splits, but its union maximum is deliberately retained for each split.
    """
    report = json.loads(bound(evidence).read_text())
    split = source.required_split
    whole_delta=report.get('schema')=='natlang.native-whole-delta-geometry/1'
    if report.get('schema') == SCHEMA:
        payload = report['payload']
        if digest(payload) != report.get('generation'):
            raise ValueError('geometry evidence resident request is corrupt')
        field, prior = ('train', 'prior_train') if split == 'train' else ('held', 'prior_held')
        source_binding, prior_binding = payload[field], payload[prior]
        witnesses, maximum = payload['indexes'], payload['geometry']
        expected_rows = payload['rows' if split == 'train' else 'held_rows']
        if payload.get('delta_validation') is not None:
            raise ValueError('delta preparation is not a full-source geometry evidence receipt')
        expected_entries = payload['entries_sha256' if split == 'train' else 'held_entries_sha256']
        start = None
    else:
        if report.get('passed') is not True or split != 'train':
            raise ValueError('legacy geometry evidence covers completed training geometry only')
        source_binding, prior_binding = report['source'], report['prior_train']
        witnesses = report['indexes']
        start = report['old_train_rows']
        expected_rows = start + report['new_train_rows']
        expected_entries = None
        maximum = {key: report['maxima'][key][key] for key in ('forward_tokens', 'kv_slots')}
        if whole_delta:
            if (first!=0 or prior_binding is not None or start!=0
                    or report.get('tokenizer_fingerprint')!=source.tokenizer_fingerprint
                    or report.get('role_ids')!=list(source.role_ids)
                    or not report.get('provenance')):
                raise ValueError('whole-delta geometry needs exact complete train scope and source contract')
            for reference in report['provenance']:bound(reference)
            expected_entries=report['entries_sha256']
    if source_preparation is None:
        from .native_source_preparation import NativeSourcePreparation
        source_preparation = NativeSourcePreparation(source.tokenizer_fingerprint, source.role_ids)
        source_preparation._remember(source)
    covered = source_preparation.load(source_binding, split=split)
    prior_source = source_preparation.load(prior_binding, split=split)if prior_binding is not None else None
    if prior_source is None and not whole_delta:
        raise ValueError('legacy geometry evidence needs its exact prior source')
    prior_entries=prior_source.entries if prior_source is not None else []
    if start is None:
        start = len(prior_source)
    if (type(start) is not int or start != len(prior_entries) or len(covered) != expected_rows
            or covered.entries[:start] != prior_entries
            or expected_entries is not None and digest(covered.entries) != expected_entries):
        raise ValueError('geometry evidence source prefix/count/metadata differs')
    _check_indexes(witnesses)
    recorded = {digest(w) for w in witnesses}
    if any(digest(w) not in recorded for w in _source_witness(covered, package)):
        raise ValueError('geometry evidence lacks exact source/tokenizer/split/policy indexes')
    measured = {entry['id']: entry for entry in covered.entries[start:]}
    requested = source.entries[first:]
    if any(measured.get(entry['id']) != entry for entry in requested):
        raise ValueError('geometry evidence does not cover every exact appended row')
    if set(maximum) != {'forward_tokens', 'kv_slots'} or any(type(v) is not int or v < 1 for v in maximum.values()):
        raise ValueError('geometry evidence needs positive physical and total-slot maxima')
    if maximum['forward_tokens'] < max((e['forward_tokens'] for e in covered.entries[start:]), default=0):
        raise ValueError('geometry evidence understates indexed physical geometry')
    if report.get('schema') != SCHEMA:
        descriptors = report['maxima']
        indices = sorted({v['index'] for v in descriptors.values()})
        if any(type(index) is not int or not 0 <= index < len(covered) for index in indices):
            raise ValueError('geometry maximum must name an exact covered-source row')
        actual = {}
        for index, example in zip(indices, covered.iter_indices(indices)):
            entry = covered.entries[index]
            actual[index] = {'index': index, 'id': example.id, 'binding': entry['binding'],
                'forward_tokens': example.plan.token_count, 'kv_slots': len(example.plan.slot_sources)}
        if any(actual.get(value['index']) != value for value in descriptors.values()):
            raise ValueError('geometry evidence maximum row identity/physical/total slots differ')
    return maximum


_SOURCE03_BRIDGE_PART = 'natlang.d32-source03-aux-bridge-geometry-part/1'


def _source03_bridge_geometry_part(part, *, preparation, package):
    """Verify one source03 bridge using its staged source and old-fit custody."""
    import hashlib
    import pickle
    from ..common.jsonio import open_jsonl_text
    from ..common.hashing import canonical_json_sha256_hex

    if part.get('schema') != _SOURCE03_BRIDGE_PART or part.get('model') != 'lfm25':
        raise ValueError('unsupported source03 bridge geometry part')
    required = ('source', 'corpus_manifest', 'supplement', 'old_join_provenance', 'limits')
    if any(not isinstance(part.get(name), dict) for name in required):
        raise ValueError('source03 bridge part lacks an exact bound input')
    source_manifest = bound(part['source'])
    corpus_manifest_path = bound(part['corpus_manifest'])
    supplement_path = bound(part['supplement'])
    old_provenance_path = bound(part['old_join_provenance'])
    corpus_manifest = json.loads(corpus_manifest_path.read_text())
    supplement = json.loads(supplement_path.read_text())
    provenance = json.loads(old_provenance_path.read_text())
    if (corpus_manifest.get('schema') != 'natlang.corpus-snapshot/1'
            or corpus_manifest.get('id') != 'd32-source03-history-keep-aux-bridge-v1'
            or supplement.get('schema') != 'natlang.d32-source03-aux-bridge-geometry-supplement/1'
            or provenance.get('schema') != 'natlang.d32-source03-aux-bridge-old-join-provenance/1'
            or corpus_manifest.get('training_admission') is True
            or provenance.get('training_admission') is not False
            or provenance.get('training_exposure') != 0):
        raise ValueError('source03 bridge custody/schema/exposure contract differs')

    def verify_local_file(path, reference):
        if (not isinstance(reference, dict) or type(reference.get('bytes')) is not int
                or path.stat().st_size != reference['bytes'] or sha(path) != reference.get('sha256')):
            raise ValueError('source03 bridge staged artifact differs from its bound byte identity: ' + str(path))
        return path

    files = {row.get('path'): row for row in corpus_manifest.get('files', [])}
    if len(files) != len(corpus_manifest.get('files', [])):
        raise ValueError('source03 bridge corpus manifest repeats a file')
    expected_files = {'lfm25/00-fit.jsonl.zst', 'lfm25/bridge-evidence-final.json',
        'lfm25/bridge-row-bindings.jsonl', 'lfm25/fit-sources.json', 'lfm25/registration-audit.json'}
    if set(files) != expected_files:
        raise ValueError('source03 bridge corpus member inventory differs')
    corpus_root = source_manifest.parent.parent.resolve()
    for relative, row in files.items():
        member_path = Path(relative)
        if member_path.is_absolute() or '..' in member_path.parts or type(row.get('bytes')) is not int:
            raise ValueError('source03 bridge corpus member path or size is malformed')
        member_path = (corpus_root / member_path).resolve(strict=True)
        try:
            member_path.relative_to(corpus_root)
        except ValueError:
            raise ValueError('source03 bridge corpus member escaped its staged root') from None
        if member_path.stat().st_size != row['bytes'] or sha(member_path) != row.get('sha256'):
            raise ValueError('source03 bridge corpus member bytes differ: ' + relative)
    source_doc = json.loads(source_manifest.read_text())
    if source_doc.get('schema') != 'natlang.native-graph-sources/1' or len(source_doc.get('sources', [])) != 1:
        raise ValueError('source03 bridge must bind its single graph source manifest')
    source_member = source_doc['sources'][0]
    if source_member.get('kind') != 'native-graphs' or '..' in Path(source_member.get('path', '')).parts:
        raise ValueError('source03 bridge graph source manifest is not a confined native-graph leaf')
    graph_path = (source_manifest.parent / source_member['path']).resolve(strict=True)
    try:
        graph_relative = graph_path.relative_to(source_manifest.parent.resolve())
    except ValueError:
        raise ValueError('source03 bridge graph escaped its declared source root') from None
    graph_binding = {'path': str(graph_path), 'sha256': source_member.get('sha256'),
                     'bytes': graph_path.stat().st_size}
    if (source_member.get('sha256') != supplement.get('source', {}).get('sha256')
            or graph_binding['bytes'] != supplement.get('source', {}).get('bytes')
            or files.get('lfm25/' + str(graph_relative), {}).get('sha256') != graph_binding['sha256']
            or files.get('lfm25/' + str(graph_relative), {}).get('bytes') != graph_binding['bytes']
            or files.get('lfm25/fit-sources.json', {}).get('sha256') != sha(source_manifest)
            or files.get('lfm25/fit-sources.json', {}).get('bytes') != source_manifest.stat().st_size):
        raise ValueError('source03 bridge graph leaf differs from its immutable corpus manifest')
    bridge_source = preparation.load(part['source'], split='train')
    if len(getattr(bridge_source, 'children', [])) != 1:
        raise ValueError('source03 bridge source manifest lost its single exact leaf')
    bridge_leaf_source = bridge_source.children[0]

    model_root = source_manifest.parent
    def local(reference, relative, inventory_path=None):
        path = (model_root / relative).resolve(strict=True)
        if path != model_root.resolve() / relative:
            try:
                path.relative_to(model_root.resolve())
            except ValueError:
                raise ValueError('source03 bridge artifact escaped its corpus root') from None
        verify_local_file(path, reference)
        if inventory_path is not None:
            entry = files.get(inventory_path)
            if not entry or entry.get('sha256') != reference.get('sha256') or entry.get('bytes') != reference.get('bytes'):
                raise ValueError('source03 bridge artifact differs from its corpus manifest')
        return path

    sidecar_path = local(supplement['source_row_binding_sidecar'], 'bridge-row-bindings.jsonl',
                         'lfm25/bridge-row-bindings.jsonl')
    final_evidence_path = local({'path': str(model_root / 'bridge-evidence-final.json'),
        'sha256': files.get('lfm25/bridge-evidence-final.json', {}).get('sha256'),
        'bytes': files.get('lfm25/bridge-evidence-final.json', {}).get('bytes')},
        'bridge-evidence-final.json', 'lfm25/bridge-evidence-final.json')
    final_evidence = json.loads(final_evidence_path.read_text())
    if final_evidence.get('schema') != 'natlang.d32-history-keep-aux-bridge-final/1':
        raise ValueError('source03 bridge final source proof is missing')
    final_source = final_evidence.get('source03', {})
    if (final_source.get('rows') != supplement.get('rows')
            or final_source.get('unique_keep_parents') != supplement.get('unique_keep_parents')
            or final_source.get('training_admission') is not False
            or final_source.get('training_exposure') != 0
            or final_source.get('old_fit_source', {}).get('sha256') != supplement.get('old_fit_source', {}).get('sha256')
            or final_source.get('old_coverage', {}).get('sha256') != supplement.get('old_fit_coverage', {}).get('sha256')
            or final_source.get('old_row_index', {}).get('sha256') != supplement.get('old_source_row_index', {}).get('sha256')
            or final_source.get('bridge_source', {}).get('sha256') != supplement.get('source', {}).get('sha256')
            or final_source.get('row_binding_sidecar', {}).get('sha256')
                != supplement.get('source_row_binding_sidecar', {}).get('sha256')):
        raise ValueError('source03 bridge final source evidence differs from its staged artifacts')

    geometry_root = supplement_path.parent.resolve()
    coverage_path = (geometry_root / 'coverage.jsonl').resolve(strict=True)
    index_rel = Path(supplement['source_row_index']['path']).name
    index_path = (geometry_root / 'row-index' / index_rel).resolve(strict=True)
    if not coverage_path.is_relative_to(geometry_root) or not index_path.is_relative_to(geometry_root):
        raise ValueError('source03 bridge geometry artifact escaped its supplement package')
    if (coverage_path.name != 'coverage.jsonl' or index_path.name != index_rel
            or verify_local_file(coverage_path, supplement['coverage']) != coverage_path
            or verify_local_file(index_path, supplement['source_row_index']) != index_path):
        raise ValueError('source03 bridge geometry files differ from their supplement bindings')

    old_root = old_provenance_path.parent.resolve()
    old_files = {row.get('path'): row for row in provenance.get('files', [])}
    if len(old_files) != len(provenance.get('files', [])):
        raise ValueError('source03 bridge old-fit provenance repeats a file')
    expected_old_names = {'old-fit/00-fit.jsonl.zst', 'old-fit/coverage.jsonl',
                 'old-fit/row-index/native-graphs-436a086fe6a15ca3669c145776184b1089fdfc41.pkl',
                 'join-report.json'}
    for name in expected_old_names:
        if name not in old_files:
            raise ValueError('source03 bridge old-fit provenance omits ' + name)
    if set(old_files) != expected_old_names:
        raise ValueError('source03 bridge old-fit provenance has unexpected members')
    old_fit_path = (old_root / 'old-fit/00-fit.jsonl.zst').resolve(strict=True)
    old_coverage_path = (old_root / 'old-fit/coverage.jsonl').resolve(strict=True)
    old_index_path = (old_root / 'old-fit/row-index/native-graphs-436a086fe6a15ca3669c145776184b1089fdfc41.pkl').resolve(strict=True)
    join_path = (old_root / 'join-report.json').resolve(strict=True)
    supplement_old_bindings = {
        'old-fit/00-fit.jsonl.zst': supplement.get('old_fit_source'),
        'old-fit/coverage.jsonl': supplement.get('old_fit_coverage'),
        'old-fit/row-index/native-graphs-436a086fe6a15ca3669c145776184b1089fdfc41.pkl': supplement.get('old_source_row_index'),
        'join-report.json': supplement.get('join_report'),
    }
    for name, reference in supplement_old_bindings.items():
        if (not isinstance(reference, dict) or reference.get('sha256') != old_files[name].get('sha256')
                or reference.get('bytes') != old_files[name].get('bytes')):
            raise ValueError('source03 bridge supplement and old-fit provenance disagree: ' + name)
    if (supplement.get('source', {}).get('sha256') != source_member.get('sha256')
            or supplement.get('source', {}).get('bytes') != graph_binding['bytes']
            or supplement.get('source_row_binding_sidecar', {}).get('sha256')
                != files.get('lfm25/bridge-row-bindings.jsonl', {}).get('sha256')):
        raise ValueError('source03 bridge supplement and registered source disagree')
    def old_member(path, name):
        expected = old_files[name]
        if sha(path) != expected.get('sha256') or path.stat().st_size != expected.get('bytes'):
            raise ValueError('source03 bridge old-fit staged member changed: ' + name)
        return path
    old_fit_path = old_member(old_fit_path, 'old-fit/00-fit.jsonl.zst')
    old_coverage_path = old_member(old_coverage_path, 'old-fit/coverage.jsonl')
    old_index_path = old_member(old_index_path, 'old-fit/row-index/native-graphs-436a086fe6a15ca3669c145776184b1089fdfc41.pkl')
    join_path = old_member(join_path, 'join-report.json')
    join_report = json.loads(join_path.read_text())
    if (join_report.get('schema') != 'natlang.d32-source03-aux-bridge-old-geometry-join/1'
            or join_report.get('rows') != provenance.get('rows')
            or join_report.get('mapping_sha256') != provenance.get('join_mapping_sha256')
            or join_report.get('old_fit_match_errors') or join_report.get('keep_parent_errors')
            or join_report.get('training_admission') is not False or join_report.get('training_exposure') != 0):
        raise ValueError('source03 bridge old-fit join report is incomplete')
    join_artifacts = join_report.get('artifacts', {})
    for report_name, reference in (
            ('old_fit_source', supplement.get('old_fit_source')),
            ('old_coverage', supplement.get('old_fit_coverage')),
            ('old_row_index', supplement.get('old_source_row_index')),
            ('bridge_source', supplement.get('source')),
            ('row_binding_sidecar', supplement.get('source_row_binding_sidecar'))):
        actual = join_artifacts.get(report_name, {})
        if actual.get('sha256') != reference.get('sha256') or actual.get('bytes') != reference.get('bytes'):
            raise ValueError('source03 bridge join report artifact differs: ' + report_name)
    join_keep_coverage = join_artifacts.get('keep_coverage', {})
    final_keep_coverage = final_source.get('keep_coverage', {})
    # The registered bridge evidence binds the current KEEP coverage by SHA,
    # while the separately staged old-join report also records its byte count.
    # Older final-evidence receipts legitimately omit that redundant count.
    if (join_keep_coverage.get('sha256') != final_keep_coverage.get('sha256')
            or ('bytes' in final_keep_coverage
                and join_keep_coverage.get('bytes') != final_keep_coverage.get('bytes'))):
        raise ValueError('source03 bridge current KEEP coverage binding differs')
    if (provenance.get('geometry_forward_tokens_sum') != supplement.get('forward_tokens_sum')
            or provenance.get('geometry_kv_slots_sum') != supplement.get('kv_slots_sum')
            or provenance.get('join_mapping_sha256') != supplement.get('join_mapping_sha256')
            or provenance.get('matched_old_fit_graph_sha') != join_report.get('matched_old_fit_graph_sha')
            or provenance.get('source_row_index_key') != supplement.get('row_index_key')
            or provenance.get('rows') != supplement.get('rows')
            or provenance.get('unique_keep_parents') != supplement.get('unique_keep_parents')):
        raise ValueError('source03 bridge old-fit provenance summary differs')

    with old_index_path.open('rb') as stream:
        old_index = pickle.load(stream)
    old_entries = old_index.get('entries')
    old_key = old_index.get('key')
    policy_transition = supplement.get('parser_policy_transition', {})
    old_components = policy_transition.get('old_components', {})
    new_components = policy_transition.get('new_components', {})
    component_names = set(old_components)
    expected_component_names = {
        'data/native_sources.py', 'data/native_graph_schema.py', 'model/native_graph_plan.py',
        'data/native_pair_schema.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py',
    }
    from ..common.hashing import sha256_file_hex
    package = Path(package).resolve(strict=True)
    actual_new_components = {name: sha256_file_hex(package / name) for name in expected_component_names}
    if (old_index.get('schema') != 'natlang.row-index/1' or not isinstance(old_entries, list)
            or digest(old_entries) != old_index.get('metadata_sha256')
            or old_key.get('source_sha256') != old_files['old-fit/00-fit.jsonl.zst']['sha256']
            or old_key.get('tokenizer') != bridge_source.tokenizer_fingerprint
            or old_key.get('role_ids') != list(bridge_source.role_ids)
            or old_key.get('kind') != 'native-graphs' or old_key.get('split') != 'train'
            or old_key.get('pair_model') is not None
            or old_key.get('policy_sha256') != policy_transition.get('old_policy_sha256')
            or policy_transition.get('new_policy_sha256') != supplement.get('row_index_key', {}).get('policy_sha256')
            or policy_transition.get('changed_components') != ['common/hashing.py']
            or component_names != expected_component_names
            or component_names != set(new_components)
            or new_components != actual_new_components
            or set(policy_transition.get('unchanged_parser_components', [])) != component_names - {'common/hashing.py'}
            or any(old_components[name] != new_components[name] for name in component_names - {'common/hashing.py'})
            or old_components.get('common/hashing.py') == new_components.get('common/hashing.py')
            or digest(old_components) != policy_transition.get('old_policy_sha256')
            or digest(new_components) != policy_transition.get('new_policy_sha256')):
        raise ValueError('source03 bridge old row index contract differs')
    with index_path.open('rb') as stream:
        bridge_index = pickle.load(stream)
    bridge_entries = bridge_index.get('entries')
    if (bridge_index.get('schema') != 'natlang.row-index/1' or bridge_index.get('key') != supplement.get('row_index_key')
            or not isinstance(bridge_entries, list) or digest(bridge_entries) != supplement.get('row_index_entries_sha256')
            or bridge_index.get('metadata_sha256') != supplement.get('row_index_entries_sha256')
            or bridge_entries != bridge_leaf_source.entries or len(bridge_entries) != supplement.get('rows')):
        raise ValueError('source03 bridge row-index differs from its exact current parser source')

    old_rows = {}
    with open_jsonl_text(old_coverage_path) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('model') == 'lfm25' and row.get('split') == 'train' and row.get('disposition') == 'fit':
                    old_rows.setdefault(row.get('id'), []).append(row)
    side_rows = []
    with open_jsonl_text(sidecar_path) as stream:
        side_rows = [json.loads(line) for line in stream if line.strip()]
    with open_jsonl_text(coverage_path) as stream:
        coverage_rows = [json.loads(line) for line in stream if line.strip()]
    row_count = supplement.get('rows')
    if (type(row_count) is not int or row_count < 1 or len(side_rows) != row_count
            or len(coverage_rows) != row_count or len(bridge_source.entries) != row_count
            or provenance.get('rows') != row_count or join_report.get('rows') != row_count
            or provenance.get('matched_old_fit_graph_sha') != row_count
            or provenance.get('unique_keep_parents') != supplement.get('unique_keep_parents')):
        raise ValueError('source03 bridge source/coverage/provenance counts differ')
    limits = part['limits']
    if (set(limits) != {'forward_tokens', 'kv_slots'}
            or any(type(value) is not int or value < 1 for value in limits.values())):
        raise ValueError('source03 bridge part needs the exact current positive geometry limits')

    # Hash the old raw source once and inspect only the referenced rows. The
    # byte-bound row index supplies their exact positions; this does not compile
    # or tokenize the 1.29B-position geometry.
    requested_old_rows = {row.get('old_fit_source_row') for row in side_rows}
    if (len(requested_old_rows) != row_count or any(type(i) is not int or i < 0 or i >= len(old_entries)
            for i in requested_old_rows)):
        raise ValueError('source03 bridge old-source row references are not unique and bounded')
    old_raw_by_index = {}
    from ..common.jsonio import open_jsonl_text as open_rows
    with open_rows(old_fit_path) as stream:
        for index, line in enumerate(stream):
            if index in requested_old_rows:
                old_raw_by_index[index] = (hashlib.sha256(line.encode('utf-8')).hexdigest(),
                                           canonical_json_sha256_hex(json.loads(line)))
    if len(old_raw_by_index) != row_count:
        raise ValueError('source03 bridge old source lacks a referenced row')

    maxima = {}
    parents = set()
    forward_sum = kv_sum = 0
    raw_digest = hashlib.sha256()
    rows_seen = 0
    actual_limits = limits
    with open_jsonl_text(graph_path) as stream:
        for index, (line, side, covered, entry, indexed) in enumerate(zip(
                stream, side_rows, coverage_rows, bridge_source.entries, bridge_entries, strict=True)):
            rows_seen += 1
            raw_digest.update(line.encode('utf-8'))
            graph = json.loads(line)
            raw_sha = hashlib.sha256(line.encode('utf-8')).hexdigest()
            old_index_at = side.get('old_fit_source_row')
            prior = old_entries[old_index_at]
            old_coverage_matches = old_rows.get(side.get('id'), [])
            old_coverage_matches = [row for row in old_coverage_matches
                if row.get('source_row') == old_index_at
                and row.get('source_graph_sha256') == side.get('old_source_graph_sha256')
                and row.get('source_row_bytes_sha256') == side.get('source_row_bytes_sha256')]
            geometry = covered.get('geometry')
            if (side.get('id') != entry.get('id') or side.get('id') != indexed.get('id')
                    or side.get('binding') != entry.get('binding') or side.get('binding') != indexed.get('binding')
                    or side.get('source_sha256') != old_key.get('source_sha256')
                    or raw_sha != side.get('source_row_bytes_sha256')
                    or canonical_json_sha256_hex(graph) != side.get('old_source_graph_sha256')
                    or old_raw_by_index[old_index_at] != (side.get('source_row_bytes_sha256'), side.get('old_source_graph_sha256'))
                    or prior.get('id') != side.get('id') or prior.get('binding') != side.get('binding')
                    or len(old_coverage_matches) != 1
                    or covered.get('schema') != 'natlang.d32-source03-aux-bridge-geometry-row/1'
                    or covered.get('bridge_row_index') != index or covered.get('id') != side.get('id')
                    or covered.get('binding') != side.get('binding')
                    or covered.get('source_graph_sha256') != side.get('old_source_graph_sha256')
                    or covered.get('source_row_bytes_sha256') != side.get('source_row_bytes_sha256')
                    or covered.get('old_row_index_binding') != prior.get('binding')
                    or covered.get('old_fit_coverage_row', {}).get('old_source_row_index') != old_index_at
                    or covered.get('old_fit_coverage_row', {}).get('source_sha256') != old_key.get('source_sha256')
                    or covered.get('old_fit_coverage_row', {}).get('coverage_sha256') != old_files['old-fit/coverage.jsonl']['sha256']
                    or old_coverage_matches[0].get('geometry') != geometry
                    or prior.get('forward_tokens') != old_coverage_matches[0].get('geometry', {}).get('forward_tokens')
                    or indexed.get('forward_tokens') != entry.get('forward_tokens')
                    or set(geometry or {}) != {'forward_tokens', 'kv_slots'}
                    or any(type(geometry[key]) is not int or geometry[key] <= 0 for key in geometry)
                    or geometry['forward_tokens'] != entry.get('forward_tokens')
                    or geometry['kv_slots'] != geometry['forward_tokens']
                    or any(geometry[key] > actual_limits[key] for key in geometry)):
                raise ValueError('source03 bridge source/old-fit/coverage/geometry join differs at row ' + str(index))
            parents.add(side.get('keep_parent_id'))
            forward_sum += geometry['forward_tokens']
            kv_sum += geometry['kv_slots']
            descriptor = {'index': index, 'id': entry['id'], 'binding': entry['binding'], **geometry}
            for key in geometry:
                if key not in maxima or descriptor[key] > maxima[key][key]:
                    maxima[key] = descriptor
    if rows_seen != row_count:
        raise ValueError('source03 bridge graph line count differs')
    if len(parents) != supplement.get('unique_keep_parents'):
        raise ValueError('source03 bridge unique KEEP-parent coverage count differs')
    if (forward_sum != supplement.get('forward_tokens_sum')
            or kv_sum != supplement.get('kv_slots_sum')
            or forward_sum != provenance.get('geometry_forward_tokens_sum')
            or kv_sum != provenance.get('geometry_kv_slots_sum')):
        raise ValueError('source03 bridge geometry totals differ from bound summaries')
    if (raw_digest.hexdigest()
            != final_source.get('bridge_source', {}).get('raw_selected_rows_sha256')
            or hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
            != final_source.get('row_binding_sidecar', {}).get('canonical_rows_sha256')
            or digest(sorted(parents)) != final_source.get('keep_pending_parent_ids_sha256')):
        raise ValueError('source03 bridge final ordered-row/source custody digest differs')
    return bridge_source, maxima, [part['source'], part['corpus_manifest'], part['supplement'],
        part['old_join_provenance']], row_count


def prepare_whole_delta_geometry(source, parts, output, *, package):
    """Verify completed ordinary partition coverage, then reuse its actual KV geometry.

    Receipts are supplied as exact admitted/staged bindings by the caller. This
    verifies every covered row; arbitrary provenance files alone prove nothing.
    """
    from .native_source_preparation import NativeSourcePreparation
    preparation=NativeSourcePreparation(source.tokenizer_fingerprint,source.role_ids)
    preparation._remember(source);offset=0;maxima={};provenance=[]
    if not isinstance(parts, list) or not parts:
        raise ValueError('delta geometry needs an ordered nonempty source-part list')
    bridge_parts = [part for part in parts if part.get('schema') == _SOURCE03_BRIDGE_PART]
    if bridge_parts:
        if len(bridge_parts) != 1 or parts[0] is not bridge_parts[0]:
            raise ValueError('source03 bridge must be the single explicit prefix of the delta source')
        bridge, bridge_maxima, bridge_provenance, bridge_count = _source03_bridge_geometry_part(
            parts[0], preparation=preparation, package=package)
        if bridge_count > len(source):
            raise ValueError('source03 bridge prefix exceeds the complete delta source')
        for index, entry in enumerate(bridge.entries):
            actual = source.entries[index]
            if ({key:value for key,value in actual.items() if key!='training_priority'}
                    != {key:value for key,value in entry.items() if key!='training_priority'}):
                raise ValueError('source03 bridge is not the exact ordered delta-source prefix')
        for key, descriptor in bridge_maxima.items():
            descriptor = {**descriptor, 'index': descriptor['index'] + offset}
            if key not in maxima or descriptor[key] > maxima[key][key]:
                maxima[key] = descriptor
        offset += bridge_count
        provenance.extend(bridge_provenance)
    ordinary_parts = parts[len(bridge_parts):] if bridge_parts else parts
    for part in ordinary_parts:
        if part.get('schema') == _SOURCE03_BRIDGE_PART:
            raise ValueError('source03 bridge may appear only once at the delta prefix')
        path=bound(part['receipt']);receipt=json.loads(path.read_text());model=part['model']
        if receipt.get('source_eligibility_unchanged')is not True or receipt.get('exact_original_row_bytes')is not True:
            raise ValueError('delta geometry needs exact original admitted row coverage')
        kind=receipt.get('schema');members=receipt['files'];declared_root=None
        if any(Path(row['path']).is_absolute()for row in members):
            # Completed teacher receipts retain their original absolute namespace
            # after the exact immutable bundle is staged on another source host.
            fit_declaration=Path((receipt.get('fit_sources',{}).get(model)or{}).get('path',''))
            if (kind!='natlang.admitted-ordinary-source-capacity-preparation/1'
                    or not fit_declaration.is_absolute()or '..'in fit_declaration.parts
                    or fit_declaration.name!='fit-sources.json'or fit_declaration.parent.name!=model):
                raise ValueError('absolute delta members lack their exact declared teacher root')
            declared_root=fit_declaration.parent.parent
        normalized=[]
        for row in members:
            declared=Path(row['path'])
            if '..'in declared.parts:
                raise ValueError('delta receipt member contains path traversal')
            if declared_root is not None:
                if not declared.is_absolute():raise ValueError('delta receipt mixes absolute and relative members')
                try:relative=declared.relative_to(declared_root)
                except ValueError:raise ValueError('delta teacher member escaped its declared original root')from None
            else:
                if declared.is_absolute():raise ValueError('delta receipt needs relative members')
                relative=declared
            if str(relative)in('','.'):raise ValueError('delta receipt member is not a file')
            normalized.append({**row,'path':str(relative)})
        if len({row['path']for row in normalized})!=len(normalized):
            raise ValueError('delta receipt repeated a normalized member')
        def member(reference):
            actual=bound(reference)
            try:relative=str(actual.relative_to(path.parent))
            except ValueError:raise ValueError('delta coverage escaped its receipt namespace')from None
            matches=[row for row in normalized if row['path']==relative]
            if len(matches)!=1 or matches[0]['sha256']!=reference['sha256']or matches[0]['bytes']!=actual.stat().st_size:
                raise ValueError('delta geometry member differs from its completed partition receipt')
            return actual
        coverage=member(part['coverage']);fit=member(part['source']);kind=receipt.get('schema')
        if kind=='natlang.acquired-native-capacity-partition/1':
            if coverage!=path.parent/receipt['coverage'] or fit!=path.parent/receipt['manifests'][model+':train:fit']:
                raise ValueError('delta geometry lost its exact fit/coverage disposition')
            expected=receipt['counts'][model]['train:fit']['rows'];limits=receipt['binding']['limits'][model]
            accepted=lambda row:row.get('model')==model and row.get('split')=='train'and row.get('partition')=='fit'
        elif kind=='natlang.admitted-ordinary-source-capacity-preparation/1':
            if (receipt.get('passed')is not True or receipt['fit_sources'][model]['sha256']!=part['source']['sha256']
                    or declared_root is not None and fit!=path.parent/model/'fit-sources.json'):
                raise ValueError('delta teacher geometry lacks its completed fit source')
            expected=sum(row['fit']for row in receipt['counts'][model].values());limits=receipt['limits'][model]
            accepted=lambda row:row.get('model')==model and row.get('split')=='train'and row.get('disposition')=='fit'
        else:raise ValueError('unsupported completed ordinary geometry coverage')
        parsed=preparation.load(part['source'],split='train')
        with coverage.open()as stream:rows=[json.loads(line)for line in stream if line.strip()]
        rows=[row for row in rows if accepted(row)]
        if (len(parsed)!=expected or len(rows)!=expected or len({row['id']for row in rows})!=expected
                or offset+expected>len(source)):
            raise ValueError('delta geometry coverage lost or repeated a fitted row')
        if kind=='natlang.acquired-native-capacity-partition/1':
            rows.sort(key=lambda row:row['partition_row_index'])
            if [row['partition_row_index']for row in rows]!=list(range(expected)):
                raise ValueError('delta geometry coverage order changed')
        for index,(row,entry)in enumerate(zip(rows,parsed.entries,strict=True)):
            actual=source.entries[offset+index];geometry=row['geometry']
            without_priority=lambda value:{key:item for key,item in value.items()if key!='training_priority'}
            if (row['id']!=entry['id']or without_priority(actual)!=without_priority(entry)
                    or entry.get('communication')or set(geometry)!={'forward_tokens','kv_slots'}
                    or any(type(value)is not int or value<1 for value in geometry.values())
                    or geometry['forward_tokens']!=entry['forward_tokens']
                    or geometry['kv_slots']!=geometry['forward_tokens']
                    or any(geometry[key]>limits[key]for key in geometry)):
                raise ValueError('delta ordinary coverage row/physical/KV geometry changed')
            descriptor={'index':offset+index,'id':entry['id'],'binding':entry['binding'],**geometry}
            for key in geometry:
                if key not in maxima or descriptor[key]>maxima[key][key]:maxima[key]=descriptor
        offset+=expected;provenance.extend([part['receipt'],part['coverage']])
    if not parts or offset!=len(source):raise ValueError('delta geometry coverage omitted source rows')
    report={'schema':'natlang.native-whole-delta-geometry/1','passed':True,'source':binding(source.path),
        'prior_train':None,'old_train_rows':0,'new_train_rows':len(source),'entries_sha256':digest(source.entries),
        'tokenizer_fingerprint':source.tokenizer_fingerprint,'role_ids':list(source.role_ids),
        'indexes':_source_witness(source,package),'provenance':provenance,'maxima':maxima}
    from uuid import uuid4
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name('.'+output.name+'.preparing-'+uuid4().hex)
    try:
        write_json(temporary,report)
        geometry_from_evidence(source,0,binding(temporary),package=package,source_preparation=preparation)
        with(output.parent/(output.name+'.lock')).open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if output.exists():
                if json.loads(output.read_text())!=report:raise ValueError('existing delta geometry report differs')
            else:write_json(output,report)
        return binding(output)
    finally:temporary.unlink(missing_ok=True)


def process_owner():
    return {'pid': os.getpid(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks': Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]}


def owner_alive(owner):
    try:
        return (owner['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and owner['start_ticks'] == Path('/proc', str(owner['pid']), 'stat').read_text().split(') ', 1)[1].split()[19])
    except (KeyError, OSError):
        return False


def retire(directory, checkpoint, terminal, *, handoff=None, verified_handoff=None):
    """After the optimizer owner drains, allow an ordinary runtime handoff.

    A normal checkpoint must already contain this adoption. No special save is
    requested here; stale checkpoints cannot retire the recovery authority.
    """
    import torch
    if (handoff is None) != (verified_handoff is None):
        raise ValueError('resident retirement reuse requires its bound handoff and verified decode together')
    directory = Path(directory)
    with locked(directory):
        state = current(directory)
        if state is None:
            return
        initial = json.loads((directory/'initial.json').read_text())
        if owner_alive(initial.get('owner') or {}):
            raise ValueError('resident retirement requires the actual optimizer process to exit')
        exit_record = json.loads(bound(terminal).read_text())
        if exit_record.get('returncode') != 0 or exit_record.get('at', 0) < state['at']:
            raise ValueError('resident retirement requires the drained optimizer owner')
        if verified_handoff is None:
            saved = torch.load(bound(checkpoint), map_location='cpu', weights_only=False, mmap=True)
        else:
            from ..rl.native_runtime_handoff import _loaded_handoff
            record, saved = _loaded_handoff(handoff, verified_handoff)
            adopted = record.get('resident') or {}
            if (record['checkpoint'] != checkpoint
                    or terminal not in (record.get('source_terminal'), record.get('terminal'))
                    or Path(adopted.get('directory', '')).resolve() != directory.resolve()
                    or not adopted.get('current') or not adopted.get('initial')
                    or json.loads(bound(adopted['current']).read_text()) != state
                    or json.loads(bound(adopted['initial']).read_text()) != initial):
                raise ValueError('resident retirement reuse differs from its exact drain and resident snapshots')
        from ..model.native_model import CHECKPOINT_SCHEMA
        from .native_state_sft import TRAINING_SCHEMA
        training = saved.get('training') or {}
        if (saved.get('schema') != CHECKPOINT_SCHEMA or training.get('schema') != TRAINING_SCHEMA
                or not training.get('optimizer') or not training.get('rng')
                or training.get('global_step', -1) < state['at_global_step']):
            raise ValueError('resident retirement requires actual full optimizer/RNG checkpoint state')
        if ((training.get('options', {}).get('resident_adoption') or {}).get('generation') != state['generation']
                or (training.get('source_receipts') or [{}])[0].get('sha256') != state['payload']['train']['sha256']):
            raise ValueError('resident retirement requires a checkpoint of the adopted generation')
        # The published request remains after adoption. Clear only this exact
        # retired generation under the same writer lock; preserve a newer request.
        pointer = directory/'pending.json'
        clear_pending = False
        if pointer.exists():
            request = json.loads(pointer.read_text())
            pending = json.loads(bound(request).read_text())
            if pending.get('schema') != SCHEMA or digest(pending.get('payload')) != pending.get('generation'):
                raise ValueError('resident pending request is corrupt before retirement')
            clear_pending = (pending['generation'] == state['generation']
                             and pending['payload'] == state['payload'])
        history = directory/'retired'; history.mkdir(exist_ok=True)
        write_json(history/(state['generation']+'.json'), {**state, 'checkpoint': checkpoint, 'terminal': terminal})
        if clear_pending:
            pointer.unlink()
        (directory/'current.json').unlink()
        descriptor = os.open(directory, os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def _delta_payload(directory, previous, plan, proposed, stage_id, recipe_path,
                   tokenizer_fingerprint, role_ids, package, delta_sources, geometry_evidence=None):
    """Validate only explicit new leaves; the learner validates the full union."""
    from .recipe import native_recipe_binding
    if not isinstance(delta_sources, dict) or set(delta_sources) != {'train'}:
        raise ValueError('resident delta preparation requires an explicit train delta')
    prior = plan['stage_inputs'][stage_id]
    actual = proposed['stage_inputs'][stage_id]
    if actual['held_native_graphs'] != prior['held_native_graphs']:
        raise ValueError('resident delta preparation requires unchanged held source')
    train_ref = binding(bound(actual['native_graphs']))
    prior_ref = binding(bound(prior['native_graphs']))
    held_ref = binding(bound(prior['held_native_graphs']))
    delta_ref = binding(bound(delta_sources['train']))
    wrapper = json.loads(bound(train_ref).read_text())
    expected = [{'kind': 'native-graphs', **prior_ref}, {'kind': 'native-graphs', **delta_ref}]
    if wrapper != {'schema': 'natlang.native-graph-sources/1', 'sources': expected}:
        raise ValueError('resident delta source must be exact prior source followed by declared delta')
    initial = json.loads((directory/'initial.json').read_text())
    prior_counts = previous['payload'] if previous else initial
    rows, held_rows = prior_counts.get('rows'), prior_counts.get('held_rows')
    if type(rows) is not int or rows < 1 or type(held_rows) is not int or held_rows < 1:
        raise ValueError('resident delta preparation lacks authenticated prior row counts')
    from .native_source_preparation import NativeSourcePreparation
    preparation = NativeSourcePreparation(tokenizer_fingerprint, role_ids)
    delta = preparation.load(delta_ref, split='train')
    capacity = initial.get('geometry_qualification')
    if not capacity:
        raise ValueError('resident delta append needs explicit geometry qualification')
    limits = geometry_capacity(capacity, identity=initial['geometry_identity'], code=plan['code'])
    if geometry_evidence and set(geometry_evidence)!={'train'}:
        raise ValueError('resident delta geometry evidence must cover training only')
    geometry = (geometry_from_evidence(delta,0,geometry_evidence['train'],package=package,
        source_preparation=preparation)if geometry_evidence else new_geometry(delta,0))
    if any(geometry[key] > limits[key] for key in limits):
        raise ValueError('resident delta exceeds qualified whole-graph geometry')
    pending_declarations(proposed, stage_id, tokenizer_fingerprint)
    payload={'predecessor': previous['generation'] if previous else digest(plan),
        'stage_id': stage_id, 'plan': proposed,
        'recipe': native_recipe_binding(recipe_path, plan['recipe_sha256']),
        'train': train_ref, 'prior_train': prior_ref, 'held': held_ref, 'prior_held': held_ref,
        'prior_recipe': previous['payload']['prior_recipe'] if previous else initial['recipe'],
        'tokenizer_fingerprint': tokenizer_fingerprint, 'role_ids': list(role_ids),
        'indexes': _source_witness(delta, package), 'rows': rows + len(delta), 'held_rows': held_rows,
        'geometry_qualification': capacity, 'geometry_identity': initial['geometry_identity'],
        'geometry': geometry,
        'delta_validation': {'schema': 'natlang.native-resident-delta-validation/1',
            'source': delta_ref, 'rows': len(delta), 'entries_sha256': digest(
                [{**entry, 'training_priority':entry.get('training_priority',0)} for entry in delta.entries]),
            'prior_rows': rows, 'prior_held_rows': held_rows, 'prior_plan_sha256': digest(plan)}}
    if geometry_evidence:payload['geometry_evidence']=geometry_evidence
    return payload


def prepare(directory, recipe_path, stage_id, *, tokenizer_fingerprint, role_ids,
            runtime_package=None, geometry_evidence=None, publish=True, delta_sources=None):
    """CPU preparation; callers decide admission and provide an immutable recipe."""
    if type(publish) is not bool:
        raise ValueError('resident preparation publish must be an explicit boolean')
    from .recipe import load_recipe, resolve_stage_inputs, _binding_files, native_recipe_binding
    from .native_state_sft import append_only_rows
    directory = Path(directory).resolve()
    with locked(directory):
        previous = current(directory)
        plan = (previous['payload']['plan'] if previous else
                json.loads((directory.parent / 'recipe-plan.json').read_text()))
        predecessor = previous['generation'] if previous else digest(plan)
    proposed = copy.deepcopy(plan)
    proposed['recipe'] = load_recipe(Path(recipe_path))
    proposed['stage_inputs'] = {stage['id']: ({} if stage.get('admitted') is False else
        resolve_stage_inputs(proposed['recipe'], stage,
                             {'records': None, 'pieces': None, 'text_data': None}))
        for stage in proposed['recipe']['stages']}
    _contract(plan, proposed, stage_id)
    package = Path(runtime_package or Path(__file__).resolve().parents[1])
    if code_inventory(package) != plan['code']:
        raise ValueError('resident preparation runtime differs from the running recipe')
    inputs = {}
    for roles in proposed['stage_inputs'].values():
        for value in roles.values():
            for dep in _binding_files(value):
                inputs[dep['path']] = dep['sha256']
    # Preserve non-role inputs, and remove superseded role dependencies.
    old_paths = {dep['path'] for roles in plan['stage_inputs'].values() for value in roles.values()
                 for dep in _binding_files(value)}
    proposed['inputs'] = {**{p: h for p, h in plan['inputs'].items() if p not in old_paths}, **inputs}
    if delta_sources is not None:
        payload = _delta_payload(directory, previous, plan, proposed, stage_id, recipe_path,
            tokenizer_fingerprint, role_ids, package, delta_sources,geometry_evidence)
    else:
        from .native_source_preparation import NativeSourcePreparation
        source_preparation = NativeSourcePreparation(tokenizer_fingerprint, role_ids)
        # New manifests expose already validated old subtrees; load those roots once.
        train = source_preparation.load(proposed['stage_inputs'][stage_id]['native_graphs'], split='train')
        held = source_preparation.load(proposed['stage_inputs'][stage_id]['held_native_graphs'], split='test')
        old = source_preparation.load(plan['stage_inputs'][stage_id]['native_graphs'], split='train')
        old_held = source_preparation.load(plan['stage_inputs'][stage_id]['held_native_graphs'], split='test')
        append_only_rows(old.bindings, train.bindings, label='resident training-source')
        append_only_rows(old_held.bindings, held.bindings, label='resident held-source')
        if (train.entries[:len(old.entries)] != old.entries or held.entries[:len(old_held.entries)] != old_held.entries
                or len(train) < len(old) or len(held) < len(old_held)
                or (len(train) == len(old) and len(held) == len(old_held)
                    and plan['stage_inputs'][stage_id].get('pending_native_sources') ==
                        proposed['stage_inputs'][stage_id].get('pending_native_sources'))):
            raise ValueError('resident append must preserve every prior row, order and scheduling priority')
        if train.groups & held.groups:
            raise ValueError('resident train/held groups overlap')
        declarations = pending_declarations(proposed, stage_id, tokenizer_fingerprint)
        # Reject conflicts before publishing, with no earned visits or model action.
        initial = json.loads((directory/'initial.json').read_text())
        validate_source_obligations(proposed, stage_id, train, declarations,
            initial.get('withdrawal_history') or ())
        capacity_binding = initial.get('geometry_qualification')
        if not capacity_binding:
            raise ValueError('resident append needs an explicit current geometry qualification')
        limits = geometry_capacity(capacity_binding, identity=initial['geometry_identity'], code=plan['code'])
        geometry_evidence = geometry_evidence or {}
        if set(geometry_evidence) - {'train', 'test'}:
            raise ValueError('geometry evidence must declare train/test scope explicitly')
        def measured(source, first):
            if len(source) == first:
                return {'forward_tokens': 0, 'kv_slots': 0}
            evidence = geometry_evidence.get(source.required_split)
            return (geometry_from_evidence(source, first, evidence, package=package, source_preparation=source_preparation) if evidence
                    else new_geometry(source, first))
        train_geometry = measured(train, len(old))
        held_geometry = measured(held, len(old_held))
        geometry = {key: max(train_geometry[key], held_geometry[key]) for key in train_geometry}
        if any(geometry[key] > limits[key] for key in limits):
            raise ValueError('resident append exceeds qualified whole-graph geometry; retain pending until memory handling is qualified')
        payload = {'predecessor': predecessor, 'stage_id': stage_id, 'plan': proposed,
            'recipe': native_recipe_binding(recipe_path, plan['recipe_sha256']),
            'train': binding(train.path), 'prior_train': binding(old.path),
            'held': binding(held.path), 'prior_held': binding(old_held.path),
            'prior_recipe': (previous['payload']['prior_recipe'] if previous else
                json.loads((directory/'initial.json').read_text())['recipe']),
            'tokenizer_fingerprint': tokenizer_fingerprint, 'role_ids': list(role_ids),
            'indexes': _source_witness(train, package) + _source_witness(held, package), 'rows': len(train),
            'held_rows': len(held), 'held_entries_sha256': digest(held.entries),
            'entries_sha256': digest(train.entries), 'geometry_qualification': capacity_binding,
            'geometry': geometry, 'geometry_identity': initial['geometry_identity']}
        if geometry_evidence:
            payload['geometry_evidence'] = geometry_evidence
    proposed.setdefault('resident_adoptions', []).append({'predecessor': predecessor,
        'recipe': payload['recipe'], 'train': payload['train'], 'held': payload['held'], 'stage_id': stage_id})
    request_id = digest(payload)
    with locked(directory):
        actual = current(directory)
        actual_predecessor = actual['generation'] if actual else digest(json.loads(
            (directory.parent/'recipe-plan.json').read_text()))
        if actual_predecessor != predecessor:
            raise ValueError('resident generation advanced during CPU preparation; prepare against current data')
        requests = directory / 'requests'
        requests.mkdir(exist_ok=True)
        path = requests / (request_id + '.json')
        if not path.exists():
            write_json(path, {'schema': SCHEMA, 'payload': payload, 'generation': request_id})
        if publish:
            write_json(directory / 'pending.json', binding(path))
        return binding(path)


class ResidentAdoption:
    def __init__(self, directory, *, stage_id, recipe_binding, train, held, package=None, geometry_qualification=None, geometry_identity=None, withdrawal_history=()):
        self.directory = Path(directory)
        self.stage_id, self.recipe_binding = stage_id, recipe_binding
        self.train, self.held = train, held
        self.withdrawal_history = copy.deepcopy(list(withdrawal_history))
        self.package = Path(package or Path(__file__).resolve().parents[1])
        self.state = current(directory)
        self.plan = self.state['payload']['plan'] if self.state else json.loads(
            (self.directory.parent/'recipe-plan.json').read_text())
        self.generation = self.state['generation'] if self.state else digest(self.plan)
        self.rejected = set()
        with locked(self.directory):
            initial = self.directory/'initial.json'
            prior = json.loads(initial.read_text()) if initial.exists() else None
            if prior and prior.get('owner') != process_owner() and owner_alive(prior.get('owner') or {}):
                raise ValueError('resident source state already has a live optimizer owner')
            if prior is None or (self.state is None and prior.get('plan_sha256') != digest(self.plan)) or prior.get('stage_id') != stage_id:
                write_json(initial, {'recipe': recipe_binding, 'stage_id': stage_id,
                    'geometry_qualification': binding(geometry_qualification) if geometry_qualification else None,
                    'geometry_identity': geometry_identity, 'plan_sha256': digest(self.plan),
                    'owner': process_owner(), 'withdrawal_history':self.withdrawal_history,
                    'rows':len(train), 'held_rows':len(held)})
            else:
                if any(item not in self.withdrawal_history for item in prior.get('withdrawal_history') or []):
                    raise ValueError('resident restart lost actual carried withdrawal history')
                write_json(initial, {**prior, 'owner': process_owner(), 'withdrawal_history':self.withdrawal_history})
        self.initial = json.loads((self.directory/'initial.json').read_text())
        if self.initial.get('geometry_identity') != geometry_identity:
            raise ValueError('resident geometry identity changed; use drained runtime handoff')
        actual = self.plan['stage_inputs'][stage_id]
        if (actual['native_graphs']['sha256'] != train.file_sha256
                or actual['held_native_graphs']['sha256'] != held.file_sha256):
            raise ValueError('resident plan differs from actual loaded sources')
        self.pending_declarations = pending_declarations(self.plan, stage_id, train.tokenizer_fingerprint)

    def poll(self, global_step):
        """Return validated new data or None. Rejections do not stop SGD."""
        pointer = self.directory / 'pending.json'
        if not pointer.exists():
            return None
        request = None
        try:
            request = json.loads(pointer.read_text())
            request_id = request['sha256']
            if request_id in self.rejected:
                return None
            value = json.loads(bound(request).read_text())
            payload = value['payload']
            if value.get('schema') != SCHEMA or digest(payload) != value['generation']:
                raise ValueError('resident request is corrupt')
            if value['generation'] == self.generation:
                return None
            if payload['predecessor'] != self.generation or payload['stage_id'] != self.stage_id:
                raise ValueError('resident request targets a different current generation/stage')
            _contract(self.plan, payload['plan'], self.stage_id)
            if binding(payload['train']['path']) != {k: payload['plan']['stage_inputs'][self.stage_id]['native_graphs'][k] for k in ('path','sha256')}:
                raise ValueError('resident payload source differs from its recipe plan')
            if binding(payload['held']['path']) != {k: payload['plan']['stage_inputs'][self.stage_id]['held_native_graphs'][k] for k in ('path','sha256')}:
                raise ValueError('resident held source differs from its recipe plan')
            if code_inventory(self.package) != self.plan['code']:
                raise ValueError('resident running runtime changed')
            if bound(payload['prior_held']) != self.held.path.resolve():
                raise ValueError('resident predecessor differs from loaded held source')
            if bound(payload['prior_train']) != self.train.path.resolve():
                raise ValueError('resident predecessor differs from loaded training source')
            recipe = payload['recipe']
            bound({'path': recipe['path'], 'sha256': recipe['file_sha256']})
            from .recipe import recipe_identity_sha256, load_recipe, resolve_stage_inputs
            if load_recipe(Path(recipe['path'])) != payload['plan']['recipe']:
                raise ValueError('resident recipe differs from prepared plan')
            resolved = {stage['id']: ({} if stage.get('admitted') is False else
                resolve_stage_inputs(payload['plan']['recipe'], stage,
                    {'records': None, 'pieces': None, 'text_data': None}))
                for stage in payload['plan']['recipe']['stages']}
            if resolved != payload['plan']['stage_inputs']:
                raise ValueError('resident resolved recipe inputs changed')
            if recipe_identity_sha256(recipe['path']) != recipe['sha256']:
                raise ValueError('resident recipe inheritance changed')
            if (payload.get('geometry_qualification') != self.initial.get('geometry_qualification')
                    or payload.get('geometry_identity') != self.initial.get('geometry_identity')):
                raise ValueError('resident geometry qualification changed')
            limits = geometry_capacity(payload['geometry_qualification'],
                identity=self.initial['geometry_identity'], code=self.plan['code'])
            if (set(payload['geometry']) != set(limits) or any(type(v) is not int or v < 0 for v in payload['geometry'].values())
                    or any(payload['geometry'][key] > limits[key] for key in limits)):
                raise ValueError('resident append exceeds qualified geometry')
            _check_indexes(payload['indexes'])
            from .native_source_preparation import NativeSourcePreparation
            source_preparation = NativeSourcePreparation(payload['tokenizer_fingerprint'], payload['role_ids'])
            source_preparation.retain(self.train)
            source_preparation.retain(self.held)
            train = source_preparation.load(payload['train'], split='train')
            held = source_preparation.load(payload['held'], split='test')
            full_validation = {'entries_sha256': digest(train.entries),
                'held_entries_sha256': digest(held.entries)}
            delta = payload.get('delta_validation')
            if delta is not None:
                if (delta.get('schema') != 'natlang.native-resident-delta-validation/1'
                        or delta['prior_plan_sha256'] != digest(self.plan)
                        or delta['prior_rows'] != len(self.train)
                        or delta['prior_held_rows'] != len(self.held)
                        or payload['held'] != payload['prior_held']):
                    raise ValueError('resident delta predecessor contract changed')
                suffix = source_preparation.load(delta['source'], split='train')
                suffix_entries = [{**entry, 'training_priority':entry.get('training_priority',0)} for entry in suffix.entries]
                if (len(suffix) != delta['rows'] or digest(suffix_entries) != delta['entries_sha256']
                        or train.entries[len(self.train):] != suffix_entries):
                    raise ValueError('resident delta suffix differs from prepared metadata')
            if (held.entries[:len(self.held.entries)] != self.held.entries or len(held) < len(self.held)
                    or len(held) != payload['held_rows'] or (delta is None and full_validation['held_entries_sha256'] != payload['held_entries_sha256'])):
                raise ValueError('resident held rows are not an exact class-preserving append')
            if (train.tokenizer_fingerprint != self.train.tokenizer_fingerprint
                    or train.role_ids != self.train.role_ids
                    or train.entries[:len(self.train.entries)] != self.train.entries
                    or len(train) < len(self.train) or train.groups & held.groups
                    or (len(train) == len(self.train) and len(held) == len(self.held)
                        and self.plan['stage_inputs'][self.stage_id].get('pending_native_sources') ==
                            payload['plan']['stage_inputs'][self.stage_id].get('pending_native_sources'))
                    or len(train) != payload['rows'] or (delta is None and full_validation['entries_sha256'] != payload['entries_sha256'])):
                raise ValueError('resident rows are not an exact protected append')
            declarations = pending_declarations(payload['plan'], self.stage_id, train.tokenizer_fingerprint)
            validate_source_obligations(payload['plan'], self.stage_id, train, declarations,
                self.withdrawal_history)
            with locked(self.directory):
                actual = current(self.directory)
                generation = actual['generation'] if actual else digest(json.loads(
                    (self.directory.parent/'recipe-plan.json').read_text()))
                if generation != self.generation:
                    raise ValueError('resident generation changed concurrently')
                committed = {**value, 'at_global_step': global_step, 'at': time.time(),
                    'validation': full_validation,
                    'history': [*((actual or {}).get('history') or []),
                                {'generation': value['generation'], 'at_global_step': global_step,
                                 'recipe': recipe, 'train': payload['train'], 'held': payload['held']} ]}
                write_json(self.directory/'current.json', committed)  # sole commit point
                # A crash after this point is recovered from current.json, not rejected.
                repair(self.directory, committed)
                self.state, self.plan, self.generation = committed, payload['plan'], value['generation']
                self.train, self.held, self.recipe_binding = train, held, recipe
                self.pending_declarations = declarations
                return train, held
        except (ValueError, KeyError, TypeError, OSError) as error:
            # A post-commit projection error is fatal/recoverable, never continue
            # SGD under stale source metadata after publishing a new generation.
            committed = current(self.directory)
            if committed and committed['generation'] != self.generation:
                raise RuntimeError('resident commit needs recovery before training continues') from error
            identity = (request or {}).get('sha256', digest(str(error)))
            self.rejected.add(identity)
            errors = self.directory/'rejections'
            errors.mkdir(parents=True, exist_ok=True)
            write_json(errors/(identity+'.json'), {'error': str(error), 'at': time.time(),
                'at_global_step': global_step, 'generation': self.generation, 'request': request})
            return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--retire-checkpoint', type=Path)
    parser.add_argument('--trainer-exit', type=Path)
    parser.add_argument('--stage-id')
    parser.add_argument('--tokenizer-fingerprint')
    parser.add_argument('--role-ids', nargs=4, type=int)
    parser.add_argument('--delta-train-source', type=Path,
                        help='validate only this exact appended source; learner verifies complete prefix and union')
    parser.add_argument('--no-publish', action='store_true',
                        help='validate and write an immutable request without publishing it to the learner')
    args = parser.parse_args(argv)
    if args.retire_checkpoint:
        if not args.trainer_exit:
            parser.error('retirement requires the exact trainer exit receipt')
        retire(args.directory, binding(args.retire_checkpoint), binding(args.trainer_exit))
        return
    if not all((args.recipe, args.stage_id, args.tokenizer_fingerprint, args.role_ids)):
        parser.error('preparation requires recipe, stage, tokenizer fingerprint and four role IDs')
    print(json.dumps(prepare(args.directory, args.recipe, args.stage_id,
                            tokenizer_fingerprint=args.tokenizer_fingerprint, role_ids=args.role_ids,
                            publish=not args.no_publish,
                            delta_sources={'train':binding(args.delta_train_source)} if args.delta_train_source else None)))


if __name__ == '__main__':
    main()
