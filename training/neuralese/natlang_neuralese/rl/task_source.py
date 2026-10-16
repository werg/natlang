"""Resolve an integrity-bound RL task source for the shared bootstrap factory."""
from __future__ import annotations

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
    for _, row in iter_jsonl_rows(cases):
        count += 1
        if row.get('split') != manifest.get('split'):
            raise ValueError('RL task source includes a row outside its declared split')
        task_id = row.get('id')
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise ValueError('RL task source has a missing or duplicate task identity')
        ids.add(task_id)
    if count != manifest.get('rows') or count != manifest['preserved_base']['rows'] + manifest['added_admitted_train_rows']:
        raise ValueError('RL task source row count differs from its manifest')
    return {'manifest': manifest_path, 'manifest_sha256': sha256_file_hex(manifest_path),
            'cases': cases, 'cases_sha256': manifest['cases_sha256'], 'rows': count,
            'audit': audit, 'audit_sha256': manifest['source_audit_sha256'],
            'selection_proof': proof, 'selection_proof_sha256': manifest['broad_selection_proof_sha256']}
