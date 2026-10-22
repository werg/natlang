"""Authenticated geometry-report continuation across classification-only indexes."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from . import native_resident as resident
from . import native_source_classification as classification
from ..common.jsonio import write_json_durable


def continue_geometry(evidence, proof_reference, source, output):
    """Preserve every original geometry fact; rebind only classified metadata.

    The caller still runs geometry_from_evidence and actual backward capacity.
    Existing classification parts are selected verbatim, never regenerated.
    """
    original_path = classification._checked(evidence)
    proof_path = classification._checked(proof_reference)
    report = json.loads(original_path.read_text())
    proof = json.loads(proof_path.read_text())
    if (report.get('schema') != 'natlang.native-whole-delta-geometry/1'
            or report.get('passed') is not True or report.get('old_train_rows') != 0
            or report.get('prior_train') is not None
            or resident.binding(source.path) != report.get('source')
            or report.get('new_train_rows') != len(source)
            or report.get('tokenizer_fingerprint') != source.tokenizer_fingerprint
            or report.get('role_ids') != list(source.role_ids)
            or proof.get('schema') != classification.SCHEMA):
        raise ValueError('classified geometry needs exact completed whole-source evidence')
    witnesses = resident._source_witness(source)
    parts = {}
    for part in proof.get('indexes', []):
        key = part['source']['sha256']
        if key in parts:
            raise ValueError('classified geometry migration has ambiguous source parts')
        parts[key] = part
    selected, old_by_id = [], {}
    def witness_identity(witness):
        return (resident.digest(witness['key']), witness['metadata_sha256'],
                witness['cache']['sha256'])
    recorded = {witness_identity(w): w for w in report['indexes']}
    if len(recorded) != len(report['indexes']):
        raise ValueError('classified geometry original witness identity is ambiguous')
    used = set()
    for witness in witnesses:
        part = parts.get(witness['key']['source_sha256'])
        if part is None:
            raise ValueError('classified geometry lost an original migration part')
        old = classification._index(part['old_index'])
        new = classification._index(part['new_index'])
        prior = {'cache': {k: part['old_index'][k] for k in ('path', 'sha256')},
                 'key': old['key'], 'metadata_sha256': old['metadata_sha256']}
        identity = witness_identity(prior)
        if (identity not in recorded
                or witness['key'] != new['key']
                or witness['metadata_sha256'] != new['metadata_sha256']
                or witness['cache']['sha256'] != part['new_index']['sha256']):
            raise ValueError('classified geometry index differs from original evidence/migration')
        # The original CPU helper may have used another cache directory.
        # Authenticate BOTH physical references; equal strong SHA proves an
        # immutable byte-for-byte relocation, never a policy/row exception.
        classification._checked(recorded[identity]['cache'])
        used.add(identity)
        selected.append(part)
        for entry in old['entries']:
            if entry['id'] in old_by_id:
                raise ValueError('classified geometry prior row identity is ambiguous')
            old_by_id[entry['id']] = entry
    if used != set(recorded):
        raise ValueError('classified geometry omitted original evidence indexes')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset = {**proof, 'train_sha256': resident.binding(source.path)['sha256'],
              'indexes': selected, 'held_indexes': [],
              'geometry_projection_of': dict(proof_reference)}
    subset_path = output.with_name(output.name + '.migration.json')
    write_json_durable(subset_path, subset)
    # Authenticates policy, producer, row order, every nonclassification field,
    # current classes, and ordinary/pair topology law using the shared verifier.
    from .native_classification_reader import migration_classes
    migration_classes(subset_path, source, directory=output.parent)
    prior_entries = []
    for current in source.entries:
        old = old_by_id.pop(current['id'], None)
        if old is None:
            raise ValueError('classified geometry has an unproved current row')
        prior_entries.append({**current,
            **{name: old[name] for name in classification.CLASS_FIELDS if name in old}})
        for name in classification.CLASS_FIELDS:
            if name not in old:
                prior_entries[-1].pop(name, None)
    if old_by_id or resident.entries_digest(prior_entries) != report['entries_sha256']:
        raise ValueError('classified geometry original ordered metadata digest differs')
    result = copy.deepcopy(report)
    result['entries_sha256'] = resident.entries_digest(source.entries)
    result['indexes'] = witnesses
    result['provenance'] = [*report['provenance'], dict(evidence), dict(proof_reference),
                            classification._binding(subset_path)]
    result['classification_continuation'] = {
        'original_evidence': dict(evidence), 'migration_proof': dict(proof_reference),
        'only_entry_classification_and_index_policy_changed': True,
        'optimizer_updates': 0, 'training_credit': 0}
    write_json_durable(output, result)
    return classification._binding(output)
