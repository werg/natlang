"""Resolve an integrity-bound RL task source for the shared bootstrap factory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex
from ..common.jsonio import iter_jsonl_rows


def resolve_task_source(manifest_path):
    """Verify and return one exact train-only task packet and its provenance."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
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
                or evidence.get('current_protection_intersections') != {
                    'decision_ids': [], 'held_groups': [], 'quality_groups': [],
                }):
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
        task_id = row.get('id')
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError('RL task source has a missing or duplicate task identity')
        ids.add(task_id)
        row_sha256_by_id[task_id] = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
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
