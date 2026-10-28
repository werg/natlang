"""Resolve an integrity-bound RL task source for the shared bootstrap factory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex
from ..common.jsonio import iter_jsonl_rows


_QUARANTINE_SCHEMA = 'natlang.rl-task-source-correction/1'
_PORT_CONTINUATION_PREFIX = 'Continue this session with its next assistant message.'
# v2: one continuation item makes a task ill-posed (v1 required every item to be one).
PORT_CONTINUATION_PREDICATE = 'port-record-next-assistant-turn-unprovided-v2'


def _ill_posed_port_record_continuation(row):
    """Exact source predicate for the unauthenticated port-record continuation tasks.

    These tasks ask for a historical next assistant turn while grading a result
    assembled from the record packet. The packet never supplies that turn as
    evidence. Keep this predicate deliberately narrow and data-derived: it
    does not accept task IDs, caller-provided exclusion lists, or family-only
    matches. One continuation item makes the whole task ill-posed (a task that
    mixes continue and consume items still grades an invisible reply), and
    every task source applies it: correction manifests exclude such rows, plain
    manifests refuse them.
    """
    if not isinstance(row, dict) or row.get('family') != 'curriculum_port_record_calls':
        return False
    semantics = row.get('semantics')
    files = semantics.get('folder_files') if isinstance(semantics, dict) else None
    if not isinstance(files, dict):
        return False
    return any(request.lstrip().startswith(_PORT_CONTINUATION_PREFIX) for _, request in _port_item_requests(files))


def _port_item_requests(files):
    """(path, request) of every well-formed item; a malformed item never hides its well-formed siblings."""
    result = []
    for name, payload in sorted(files.items()):
        if not (isinstance(name, str) and name.startswith('items/') and name.endswith('.json')
                and isinstance(payload, str)):
            continue
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get('request'), str):
            result.append((name, item['request']))
    return result


def _canonical_row_sha256(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def correction_exclusion(line_no, row):
    """The exact proof record of one excluded row (shared by the builder and the resolver)."""
    return {
        'line': line_no, 'id': row['id'], 'row_sha256': _canonical_row_sha256(row),
        'predicate': PORT_CONTINUATION_PREDICATE,
        'reason': 'asks for historical next assistant message without supplying that message as evidence',
        'items': [{'path': name, 'request_sha256': hashlib.sha256(request.encode()).hexdigest()}
                  for name, request in _port_item_requests(row['semantics']['folder_files'])],
    }


def _no_protection_intersections(value):
    """Empty intersections in the shared custody form (protected_worlds) or the earlier three-projection form."""
    return (isinstance(value, dict) and all(item == [] for item in value.values())
            and set(value) in ({'protected_worlds'}, {'decision_ids', 'held_groups', 'quality_groups'}))


def _resolve_corrected_task_source(manifest_path, manifest):
    """Validate a successor packet as the exact predicate complement of its parent.

    The parent is the registered source packet itself, bound by its manifest
    SHA-256 (which transitively binds its cases and proofs); it may live in a
    sibling directory, so a correction never duplicates its parent's bytes.
    """
    def bound(name, digest_name, *, member=True):
        value = manifest.get(name)
        if not isinstance(value, str):
            raise ValueError(f'RL correction {name} path is missing')
        path = (manifest_path.parent / value).resolve(strict=True)
        if member and not path.is_relative_to(manifest_path.parent):
            raise ValueError('RL task source correction member escapes its immutable directory')
        digest = sha256_file_hex(path)
        if digest != manifest.get(digest_name):
            raise ValueError(f'RL task source correction {name} changed from its manifest binding')
        return path

    parent_manifest = bound('parent_manifest', 'parent_manifest_sha256', member=False)
    parent = resolve_task_source(parent_manifest, as_correction_parent=True)
    cases = bound('cases', 'cases_sha256')
    proof_path = bound('correction_proof', 'correction_proof_sha256')
    proof = json.loads(proof_path.read_text())
    parent_raw_lines = parent['cases'].read_bytes().splitlines(keepends=True)
    source_rows = []
    for line_no, raw_line in enumerate(parent_raw_lines, 1):
        if raw_line.strip():
            source_rows.append((line_no, json.loads(raw_line)))
    expected_excluded = []
    expected_retained = []
    for line_no, row in source_rows:
        if _ill_posed_port_record_continuation(row):
            expected_excluded.append(correction_exclusion(line_no, row))
        else:
            expected_retained.append((line_no, row))
    if (proof.get('schema') != 'natlang.rl-task-source-correction-proof/1'
            or proof.get('predicate') != PORT_CONTINUATION_PREDICATE
            or proof.get('parent_manifest_sha256') != parent['manifest_sha256']
            or proof.get('parent_cases_sha256') != parent['cases_sha256']
            or proof.get('parent_rows') != parent['rows']
            or proof.get('excluded') != expected_excluded
            or proof.get('retained_rows') != len(expected_retained)
            or proof.get('excluded_rows') != len(expected_excluded)
            or proof.get('cases_sha256') != manifest.get('cases_sha256')
            or proof.get('cases_rows') != manifest.get('rows')):
        raise ValueError('RL correction proof does not attest the exact predicate complement')
    expected_bytes = b''.join(parent_raw_lines[line_no - 1] for line_no, _ in expected_retained)
    if cases.read_bytes() != expected_bytes:
        raise ValueError('RL corrected cases are not the byte-preserving retained source rows in original order')
    actual_rows = [row for _, row in iter_jsonl_rows(cases)]
    parent_declaration = json.loads(parent_manifest.read_text())
    if (manifest.get('schema') != _QUARANTINE_SCHEMA or manifest.get('split') != parent_declaration.get('split')
            or manifest.get('rows') != len(expected_retained)
            or sha256_file_hex(cases) != proof.get('cases_sha256')):
        raise ValueError('RL corrected task source manifest differs from its verified retained packet')
    ids = [row.get('id') for row in actual_rows]
    if any(not isinstance(task_id, str) or not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError('RL corrected task source has invalid or duplicate task identities')
    return {'manifest': manifest_path, 'manifest_sha256': sha256_file_hex(manifest_path),
            'cases': cases, 'cases_sha256': manifest['cases_sha256'], 'rows': len(actual_rows),
            'audit': parent['audit'], 'audit_sha256': parent['audit_sha256'],
            'selection_proof': parent['selection_proof'],
            'selection_proof_sha256': parent['selection_proof_sha256'],
            'additional_sources': parent['additional_sources'],
            'source_correction': proof_path,
            'source_correction_sha256': manifest['correction_proof_sha256'],
            'source_correction_excluded_rows': len(expected_excluded)}


def resolve_task_source(manifest_path, *, as_correction_parent=False):
    """Verify and return one exact train-only task packet and its provenance.

    A plain packet holding ill-posed continuation tasks resolves only as the
    parent of a correction (builder or correction resolver), never for RL use.
    """
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') == _QUARANTINE_SCHEMA:
        return _resolve_corrected_task_source(manifest_path, manifest)
    if manifest.get('schema') != 'natlang.rl-task-source/1':
        raise ValueError('RL task source manifest has an unsupported schema')

    def bound(name, digest_name):
        path = (manifest_path.parent / manifest[name]).resolve(strict=True)
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError('RL task source manifest member escapes its immutable directory')
        digest = sha256_file_hex(path)
        if digest != manifest.get(digest_name):
            raise ValueError(f'RL task source {name} changed from its manifest binding')
        return path

    cases = bound('cases', 'cases_sha256')
    audit = bound('source_audit', 'source_audit_sha256')
    proof = bound('broad_selection_proof', 'broad_selection_proof_sha256')
    additional_sources = manifest.get('additional_admitted_train_sources', [])
    if not isinstance(additional_sources, list):
        raise ValueError('RL task source additional sources must be a list')
    additional_rows = 0
    additional_ids = set()
    additional_evidence = []
    for source in additional_sources:
        if not isinstance(source, dict):
            raise ValueError('RL task source additional source record is malformed')
        source_cases_name = source.get('cases')
        source_cases_sha = source.get('cases_sha256')
        source_rows = source.get('rows')
        evidence_name = source.get('qualification_proof')
        evidence_sha = source.get('qualification_proof_sha256')
        if (not isinstance(source_cases_name, str) or not isinstance(source_cases_sha, str)
                or type(source_rows) is not int or source_rows <= 0
                or not isinstance(evidence_name, str) or not isinstance(evidence_sha, str)):
            raise ValueError('RL task source additional source binding is incomplete')
        source_cases = (manifest_path.parent / source_cases_name).resolve(strict=True)
        evidence_path = (manifest_path.parent / evidence_name).resolve(strict=True)
        if not source_cases.is_relative_to(manifest_path.parent) or not evidence_path.is_relative_to(manifest_path.parent):
            raise ValueError('RL task source additional source escapes its immutable directory')
        if sha256_file_hex(source_cases) != source_cases_sha or sha256_file_hex(evidence_path) != evidence_sha:
            raise ValueError('RL task source additional source bytes differ from their manifest bindings')
        evidence = json.loads(evidence_path.read_text())
        if (evidence.get('schema') != 'natlang.rl-additional-task-source-qualification/1'
                or evidence.get('source_id') != source.get('id')
                or evidence.get('source_cases_sha256') != source_cases_sha
                or evidence.get('qualified_task_rows') != source_rows
                or evidence.get('all_train') is not True
                or evidence.get('all_source_bound') is not True
                or not _no_protection_intersections(evidence.get('current_protection_intersections'))):
            raise ValueError('RL task source additional qualification does not attest its exact protected train rows')
        source_count = 0
        source_ids = set()
        for _, row in iter_jsonl_rows(source_cases):
            source_count += 1
            task_id = row.get('id')
            if row.get('split') != manifest.get('split') or not isinstance(task_id, str) or not task_id or task_id in source_ids:
                raise ValueError('RL task source additional rows are not unique train tasks')
            source_ids.add(task_id)
        expected_ids = evidence.get('task_ids')
        if source_count != source_rows or not isinstance(expected_ids, list) or source_ids != set(expected_ids):
            raise ValueError('RL task source additional rows differ from qualification task IDs')
        if additional_ids & source_ids:
            raise ValueError('RL task source additional sources duplicate task identities')
        additional_ids.update(source_ids)
        additional_rows += source_rows
        additional_evidence.append(evidence)
    audit_value = json.loads(audit.read_text())
    if (audit_value.get('schema') != 'natlang.rl-task-breadth-union-audit/1'
            or audit_value.get('conflicting_ids') != []
            or audit_value.get('union', {}).get('sha256') != manifest['cases_sha256']
            or audit_value.get('union', {}).get('exact_same_id_duplicates') != 0
            or audit_value.get('union', {}).get('same_id_different_payload') != 0
            or audit_value.get('union', {}).get('rows') != manifest['rows']
            or audit_value.get('base', {}).get('rows') != manifest['preserved_base']['rows']
            or audit_value.get('base', {}).get('sha256') != manifest['preserved_base']['sha256']
            or audit_value.get('broad', {}).get('selection_proof_sha256') != manifest['broad_selection_proof_sha256']
            or audit_value.get('source_registry_revalidation', {}).get('all_sources_still_registered_admitted') is not True
            or audit_value.get('source_registry_revalidation', {}).get('all_source_files_sha_match_selection_proof') is not True
            or audit_value.get('current_protection', {}).get('union_rows_intersecting_current_decision_holds') != 0
            or audit_value.get('current_protection', {}).get('union_source_groups_intersecting_held') != 0
            or audit_value.get('current_protection', {}).get('union_source_groups_intersecting_quality') != 0):
        raise ValueError('RL task source audit does not attest this exact protected train union')
    proof_value = json.loads(proof.read_text())
    if (proof_value.get('cases_sha256') != audit_value['broad']['sha256']
            or proof_value.get('cases') != manifest['added_admitted_train_rows']
            or audit_value.get('selection_protection', {}).get('cases') != manifest['added_admitted_train_rows']
            or audit_value.get('selection_protection', {}).get('omissions', {}).get('nontrain')
               != manifest.get('broad_nontrain_omissions')):
        raise ValueError('RL broad selection proof does not attest the admitted added rows')
    count = 0
    ids = set()
    row_sha256_by_id = {}
    for _, row in iter_jsonl_rows(cases):
        count += 1
        if row.get('split') != manifest.get('split'):
            raise ValueError('RL task source includes a row outside its declared split')
        if not as_correction_parent and _ill_posed_port_record_continuation(row):
            raise ValueError('RL task source includes a port-record continuation task that grades an unseen '
                             'historical reply; publish a natlang.rl-task-source-correction/1 successor')
        task_id = row.get('id')
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError('RL task source has a missing or duplicate task identity')
        ids.add(task_id)
        row_sha256_by_id[task_id] = _canonical_row_sha256(row)
    if count != manifest.get('rows') or count != (manifest['preserved_base']['rows']
            + manifest['added_admitted_train_rows'] + additional_rows):
        raise ValueError('RL task source row count differs from its manifest')
    if not additional_ids.issubset(ids):
        raise ValueError('RL task source union omits a qualified additional task')
    for evidence in additional_evidence:
        expected_rows = evidence.get('union_task_row_sha256')
        if (not isinstance(expected_rows, dict)
                or set(expected_rows) != set(evidence.get('task_ids', []))
                or any(
                    row_sha256_by_id.get(task_id) != digest for task_id, digest in expected_rows.items())):
            raise ValueError('RL task source qualified task payload differs from its bound union rows')
    return {'manifest': manifest_path, 'manifest_sha256': sha256_file_hex(manifest_path),
            'cases': cases, 'cases_sha256': manifest['cases_sha256'], 'rows': count,
            'audit': audit, 'audit_sha256': manifest['source_audit_sha256'],
            'selection_proof': proof, 'selection_proof_sha256': manifest['broad_selection_proof_sha256'],
            'additional_sources': additional_evidence}
