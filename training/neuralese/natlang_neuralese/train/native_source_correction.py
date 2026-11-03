"""Exact row-binding transition for repaired native graph targets.

This mechanism changes only source/exposure metadata. It never edits model,
optimizer, RNG, or numeric state. Old target exposure is retained as history;
corrected communication rows begin with fresh exposure debt.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached

SCHEMA = 'natlang.native-source-correction/1'


def _proof_file(proof_path, reference):
    """Resolve a dependency relative to its proof so the proof bundle can move."""
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = Path(proof_path).resolve().parent / candidate
    return candidate.resolve(strict=True)


def _binding(ref):
    if not isinstance(ref, dict) or not all(isinstance(ref.get(k), str) and ref[k]
                                             for k in ('path', 'sha256')):
        raise ValueError('source correction requires exact source references')
    return {'path': str(Path(ref['path']).resolve()), 'sha256': ref['sha256']}


def _pending_communication_class(training, pending_item, row_id, prior_receipt):
    """Authenticate a pending row's original class from its saved correction lineage."""
    if not isinstance(pending_item, dict) or type(pending_item.get('communication')) is not bool:
        raise ValueError('pending source row lacks its authenticated original class: ' + row_id)
    origin = pending_item.get('classification_provenance')
    if (not isinstance(origin, dict)
            or origin.get('schema') != 'natlang.pending-row-classification-provenance/1'
            or origin.get('row_id') != row_id
            or origin.get('classification_source') != 'saved_checkpoint_communication_ids'):
        raise ValueError('pending source row lacks exact classification provenance: ' + row_id)
    origin_ref = {'path': origin.get('proof_path'), 'sha256': origin.get('proof_sha256')}
    if (not isinstance(origin_ref['path'], str) or not origin_ref['path']
            or not isinstance(origin_ref['sha256'], str) or not origin_ref['sha256']):
        raise ValueError('pending source row classification origin lacks an exact proof binding: ' + row_id)
    history = (training.get('options') or {}).get('source_correction_history') or []
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError('pending source correction history is malformed')
    matching = [(index, item) for index, item in enumerate(history)
                if item.get('proof_sha256') == origin_ref['sha256']
                and Path(str(item.get('proof_path', ''))).resolve() == Path(str(origin_ref['path'])).resolve()]
    if len(matching) != 1:
        raise ValueError('pending source row classification proof is absent or ambiguous in saved history: ' + row_id)
    _, origin_record = matching[0]
    if row_id not in set(origin_record.get('deferred_rows') or []):
        raise ValueError('pending source row is not deferred by its authenticated correction history: ' + row_id)
    pinned_classes = origin_record.get('deferred_row_classes')
    if (not isinstance(pinned_classes, dict)
            or type(pinned_classes.get(row_id)) is not bool
            or pinned_classes[row_id] is not pending_item['communication']):
        raise ValueError('pending source row class differs from its saved deferral record: ' + row_id)
    if (_binding(origin.get('source_receipt')) != _binding(origin_record.get('prior_source'))
            or origin.get('model_id') != origin_record.get('model_id')
            or origin.get('stage_id') != origin_record.get('stage_id')):
        raise ValueError('pending source row classification is detached from its source lineage: ' + row_id)
    dependencies = source_correction_dependencies(origin_ref)
    proof_path = Path(origin_ref['path']).resolve(strict=True)
    proof = json.loads(proof_path.read_text())
    if (_binding(proof.get('current_source')) != _binding(origin_record.get('current_source'))
            or _binding(proof.get('prior_source')) != _binding(origin_record.get('prior_source'))
            or proof.get('stage_id') != origin.get('stage_id')
            or proof.get('model_id') != origin.get('model_id')):
        raise ValueError('pending source row origin proof differs from its saved lineage: ' + row_id)
    return pending_item['communication'], {'proof_path':str(proof_path),
        'proof_sha256':origin_ref['sha256'], 'dependencies':dependencies}


def _receipt_matches_with_staged_relocation(expected, actual, proof, proof_path):
    """Allow only a source-map-authenticated path relocation of identical bytes."""
    if _binding(expected) == _binding(actual):
        return True
    if (not isinstance(expected, dict) or not isinstance(actual, dict)
            or expected.get('sha256') != actual.get('sha256')):
        return False
    relocation = proof.get('prior_held_relocation')
    if (not isinstance(relocation, dict)
            or relocation.get('schema') != 'natlang.staged-receipt-relocation/1'
            or relocation.get('sha256') != expected.get('sha256')
            or {Path(relocation.get('logical_path', '')).resolve(),
                Path(relocation.get('staged_path', '')).resolve()}
               != {Path(expected['path']).resolve(), Path(actual['path']).resolve()}):
        return False
    source_map_ref = proof.get('source_map')
    if not isinstance(source_map_ref, dict):
        return False
    source_map_path = _proof_file(proof_path, source_map_ref.get('path', ''))
    if sha256_file_hex_cached(source_map_path) != source_map_ref.get('sha256'):
        raise ValueError('source correction relocation map changed')
    if (Path(relocation.get('source_map_path', '')).resolve() != source_map_path
            or relocation.get('source_map_sha256') != source_map_ref.get('sha256')):
        raise ValueError('source correction receipt relocation metadata differs from its map')
    source_map = json.loads(source_map_path.read_text())
    if (source_map.get('schema') != 'natlang.staged-source-map/1'
            or not isinstance(source_map.get('paths'), dict)):
        raise ValueError('source correction relocation map schema differs')
    mappings = source_map['paths']
    mapped = mappings.get(actual['path'])
    if isinstance(mapped, str) and Path(mapped).resolve() == Path(expected['path']).resolve():
        relocated_path = Path(mapped).resolve(strict=True)
    elif (mappings.get(expected['path']) is not None
            and Path(mappings[expected['path']]).resolve() == Path(actual['path']).resolve()):
        relocated_path = Path(mappings[expected['path']]).resolve(strict=True)
    else:
        return False
    return relocated_path.is_file() and sha256_file_hex_cached(relocated_path) == expected['sha256']


def validate_held_quality_hold(reference, *, held_receipt, held_entries):
    """Authenticate the exact protected rows excluded from held scoring.

    The hold is scoped to the loaded held receipt and each original row's
    binding, group membership, and source leaf. It cannot be used to exclude
    arbitrary held examples or to silently follow a changed held source.
    """
    path = Path(reference['path']).resolve(strict=True)
    digest = sha256_file_hex_cached(path)
    if digest != reference.get('sha256'):
        raise ValueError('held quality hold manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'natlang.native-held-quality-hold/1':
        raise ValueError('held quality hold schema differs')
    if _binding(manifest.get('held_source')) != _binding(held_receipt):
        raise ValueError('held quality hold names a different protected source')
    rows_path = _proof_file(path, manifest.get('row_bindings_path', ''))
    rows_sha = sha256_file_hex_cached(rows_path)
    if rows_sha != manifest.get('row_bindings_sha256'):
        raise ValueError('held quality hold row bindings changed')
    expected_sources = {}
    for source in manifest.get('pending_sources', []):
        source_sha = source.get('sha256')
        count = source.get('rows')
        reason = source.get('reason')
        if (not isinstance(source_sha, str) or len(source_sha) != 64
                or type(count) is not int or count < 1
                or not isinstance(reason, str) or not reason):
            raise ValueError('held quality hold source inventory is malformed')
        if source_sha in expected_sources:
            raise ValueError('held quality hold repeats a pending source')
        expected_sources[source_sha] = (count, reason)
    by_id = {entry['id']: entry for entry in held_entries}
    if len(by_id) != len(held_entries):
        raise ValueError('loaded held source repeats row identities')
    origins = _origins(held_receipt, [entry['id'] for entry in held_entries])
    excluded = {}
    from ..common.jsonio import open_jsonl_text
    with open_jsonl_text(rows_path) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get('id')
            if (not isinstance(row_id, str) or row_id in excluded
                    or row_id not in by_id):
                raise ValueError('held quality hold repeats or adds a row outside the held source')
            entry = by_id[row_id]
            source_sha = row.get('source_sha256')
            if (source_sha not in expected_sources
                    or row.get('reason') != expected_sources[source_sha][1]
                    or origins.get(row_id) != source_sha
                    or row.get('binding') != entry.get('binding')
                    or tuple(row.get('groups') or ()) != tuple(entry.get('groups') or ())):
                raise ValueError('held quality hold row does not match its exact source/binding/groups')
            excluded[row_id] = source_sha
    counts = {source_sha: sum(value == source_sha for value in excluded.values())
              for source_sha in expected_sources}
    if counts != {source_sha: value[0] for source_sha, value in expected_sources.items()}:
        raise ValueError('held quality hold rows do not cover the exact pending source counts')
    if (len(excluded) != manifest.get('excluded_rows')
            or len(excluded) != sum(value[0] for value in expected_sources.values())):
        raise ValueError('held quality hold total differs from its exact source inventory')
    return {'manifest': {'path': str(path), 'sha256': digest},
            'row_bindings': {'path': str(rows_path), 'sha256': rows_sha},
            'excluded_ids': set(excluded), 'excluded_rows': len(excluded),
            'source_sha256': held_receipt['sha256']}


def _origins(receipt, order):
    result = {}
    offset = 0

    def visit(item):
        nonlocal offset
        children = item.get('sources')
        if children:
            for child in children:
                visit(child)
            return
        count = item.get('rows')
        digest = item.get('sha256')
        if type(count) is not int or count < 1 or not isinstance(digest, str):
            raise ValueError('source correction predecessor has incomplete leaf receipts')
        ids = order[offset:offset + count]
        if len(ids) != count:
            raise ValueError('source correction receipt row counts exceed saved order')
        for row_id in ids:
            if row_id in result:
                raise ValueError('source correction source receipt repeats a row id')
            result[row_id] = digest
        offset += count

    visit(receipt)
    if offset != len(order):
        raise ValueError('source correction receipt rows differ from saved source order')
    return result


QUOTA_REVISION_SCHEMA = 'natlang.native-quota-revision/1'
FAMILY_REASSIGNMENT_SCHEMA = 'natlang.native-family-reassignment/1'
_LEDGER_FIELDS = {QUOTA_REVISION_SCHEMA: {'id', 'binding', 'prior_required_visits', 'required_visits'},
                  FAMILY_REASSIGNMENT_SCHEMA: {'id', 'binding', 'old_family', 'new_family'}}


def declared_row_ledger(proof_path, proof, key):
    """Load an optional proof-bound per-row ledger (quota revision or family reassignment).

    A quota revision lowers the visit debt of exact unchanged rows to a declared
    smaller quota; earned credit is never reduced (the applied quota is the larger
    of the declared quota and the row's earned visits). A family reassignment moves
    exact unchanged rows between declared CE families. Both are explicit, reviewed
    transitions with a nonempty reason, never inferred from a changed map or plan.
    """
    declaration = proof.get(key)
    if declaration is None:
        return None, None
    schema = {'quota_revision': QUOTA_REVISION_SCHEMA, 'family_reassignment': FAMILY_REASSIGNMENT_SCHEMA}[key]
    if (not isinstance(declaration, dict) or declaration.get('schema') != schema
            or not isinstance(declaration.get('reason'), str) or not declaration['reason'].strip()):
        raise ValueError(f'source correction {key} needs its schema and an explicit reason')
    path = _proof_file(proof_path, declaration.get('row_bindings_path', ''))
    digest = sha256_file_hex_cached(path)
    if digest != declaration.get('row_bindings_sha256'):
        raise ValueError(f'source correction {key} ledger changed')
    from ..common.jsonio import open_jsonl_text
    rows = {}
    with open_jsonl_text(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or set(row) != _LEDGER_FIELDS[schema]
                    or not isinstance(row['id'], str) or row['id'] in rows):
                raise ValueError(f'source correction {key} ledger row is malformed or repeated')
            if schema == QUOTA_REVISION_SCHEMA:
                prior, required = row['prior_required_visits'], row['required_visits']
                if type(prior) is not int or type(required) is not int or not 1 <= required < prior:
                    raise ValueError('quota revision may only lower a positive quota to a positive quota')
            elif (not all(isinstance(row[name], str) and row[name] for name in ('old_family', 'new_family'))
                    or row['old_family'] == row['new_family']):
                raise ValueError('family reassignment must name two different families')
            rows[row['id']] = row
    if not rows:
        raise ValueError(f'source correction {key} ledger is empty')
    return rows, {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def source_correction_dependencies(reference):
    """Validate and return the immutable proof file used by provider closure."""
    path = Path(reference['path']).resolve(strict=True)
    digest = sha256_file_hex_cached(path)
    if digest != reference.get('sha256'):
        raise ValueError('native source correction proof changed')
    proof = json.loads(path.read_text())
    if proof.get('schema') != SCHEMA:
        raise ValueError('native source correction schema differs')
    rows_path = _proof_file(path, proof.get('row_bindings_path', ''))
    rows_sha = sha256_file_hex_cached(rows_path)
    if rows_sha != proof.get('row_bindings_sha256'):
        raise ValueError('native source correction row ledger changed')
    dependencies = [
        {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size},
        {'path': str(rows_path), 'sha256': rows_sha, 'bytes': rows_path.stat().st_size},
    ]
    target_mask = proof.get('target_mask_correction')
    if target_mask is not None:
        dependencies.extend(_validate_target_mask_correction(path, proof, target_mask))
    static_context = proof.get('static_native_context_correction')
    if static_context is not None:
        from .native_static_context_correction import verify_static_context_correction
        _, static_dependencies = verify_static_context_correction(path, static_context)
        dependencies.extend(static_dependencies)
    source_map_ref = proof.get('source_map')
    if source_map_ref is not None:
        source_map_path = _proof_file(path, source_map_ref.get('path', ''))
        source_map_sha = sha256_file_hex_cached(source_map_path)
        if source_map_sha != source_map_ref.get('sha256'):
            raise ValueError('source correction staged-source map changed')
        dependencies.append({'path':str(source_map_path),'sha256':source_map_sha,
                             'bytes':source_map_path.stat().st_size})
    evidence = proof.get('evidence') or []
    if not isinstance(evidence, list):
        raise ValueError('native source correction evidence must be an exact file inventory')
    seen={str(path),str(rows_path)}
    for reference in evidence:
        if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
            raise ValueError('native source correction evidence lacks an exact file reference')
        evidence_path=_proof_file(path, reference['path'])
        evidence_sha=sha256_file_hex_cached(evidence_path)
        if evidence_sha != reference.get('sha256'):
            raise ValueError('native source correction provenance evidence changed')
        if str(evidence_path) not in seen:
            dependencies.append({'path':str(evidence_path),'sha256':evidence_sha,
                                 'bytes':evidence_path.stat().st_size})
            seen.add(str(evidence_path))
    for key in ('quota_revision', 'family_reassignment'):
        _, ledger = declared_row_ledger(path, proof, key)
        if ledger is not None and ledger['path'] not in seen:
            dependencies.append(ledger)
            seen.add(ledger['path'])
    protected = proof.get('protected_held_correction')
    if protected is not None:
        if protected.get('schema') != 'natlang.native-protected-held-correction/1':
            raise ValueError('native protected held correction schema differs')
        held_rows = _proof_file(path, protected.get('row_bindings_path', ''))
        held_sha = sha256_file_hex_cached(held_rows)
        if held_sha != protected.get('row_bindings_sha256'):
            raise ValueError('native protected held correction ledger changed')
        dependencies.append({'path': str(held_rows), 'sha256': held_sha,
                             'bytes': held_rows.stat().st_size})
    quality = proof.get('held_quality_hold')
    if quality is not None:
        quality_path = _proof_file(path, quality.get('path', ''))
        quality_sha = sha256_file_hex_cached(quality_path)
        if quality_sha != quality.get('sha256'):
            raise ValueError('native held quality hold manifest changed')
        quality_manifest = json.loads(quality_path.read_text())
        quality_rows = _proof_file(quality_path, quality_manifest.get('row_bindings_path', ''))
        quality_rows_sha = sha256_file_hex_cached(quality_rows)
        if quality_rows_sha != quality_manifest.get('row_bindings_sha256'):
            raise ValueError('native held quality hold row ledger changed')
        for dependency in ((quality_path, quality_sha), (quality_rows, quality_rows_sha)):
            if str(dependency[0]) not in seen:
                dependencies.append({'path': str(dependency[0]), 'sha256': dependency[1],
                                     'bytes': dependency[0].stat().st_size})
                seen.add(str(dependency[0]))
    return dependencies


def _validate_target_mask_correction(proof_path, proof, declaration):
    """Validate a same-ID ordinary graph correction against its exact mask build.

    This transition permits only a reviewed zeroing of target weights in one
    existing train row. It verifies that the source leaf is otherwise
    unchanged, preserving row identity and forward tokens. Exposure is handled
    separately by ``apply_source_correction`` and starts fresh for the changed
    row while its previous visits remain in the correction history.
    """
    from ..common.jsonio import open_jsonl_text
    from ..data.native_graph_quality import mask_exact_target_spans

    if declaration.get('schema') != 'natlang.native-target-mask-correction/1':
        raise ValueError('native target-mask correction schema differs')
    treatment_ref = declaration.get('source_treatment') or {}
    treatment_path = _proof_file(proof_path, treatment_ref.get('path', ''))
    treatment_sha = sha256_file_hex_cached(treatment_path)
    if (treatment_sha != treatment_ref.get('sha256')
            or ('bytes' in treatment_ref and treatment_ref['bytes'] != treatment_path.stat().st_size)):
        raise ValueError('native target-mask source-treatment receipt changed')
    treatment = json.loads(treatment_path.read_text())
    if treatment.get('schema') != 'natlang.native-target-mask-source-treatment/1':
        raise ValueError('native target-mask source-treatment schema differs')

    def checked_ref(value, label):
        if not isinstance(value, dict) or not isinstance(value.get('path'), str):
            raise ValueError(f'native target-mask {label} reference is malformed')
        path = _proof_file(proof_path, value['path'])
        digest = sha256_file_hex_cached(path)
        if digest != value.get('sha256') or ('bytes' in value and path.stat().st_size != value['bytes']):
            raise ValueError(f'native target-mask {label} changed')
        return path, digest

    review_path, review_sha = checked_ref(declaration.get('review'), 'review')
    mask_path, mask_sha = checked_ref(declaration.get('mask_output'), 'masked graph')
    review = json.loads(review_path.read_text())
    mask_receipt = review.get('mask_receipt') or {}
    graph_id = declaration.get('graph_id')
    old_binding = declaration.get('binding_before')
    new_binding = declaration.get('binding_after')
    old_leaf = declaration.get('source_leaf_before') or {}
    new_leaf = declaration.get('source_leaf_after') or {}
    old_leaf_path = Path(old_leaf.get('path', '')).resolve(strict=True)
    new_leaf_path = Path(new_leaf.get('path', '')).resolve(strict=True)
    if (sha256_file_hex_cached(old_leaf_path) != old_leaf.get('sha256')
            or sha256_file_hex_cached(new_leaf_path) != new_leaf.get('sha256')
            or ('bytes' in old_leaf and old_leaf_path.stat().st_size != old_leaf['bytes'])
            or ('bytes' in new_leaf and new_leaf_path.stat().st_size != new_leaf['bytes'])):
        raise ValueError('native target-mask source leaf bytes changed')
    treatment_old = treatment.get('source_leaf_before') or {}
    treatment_new = treatment.get('source_leaf_after') or {}
    if (not isinstance(graph_id, str) or not graph_id
            or mask_receipt.get('id') != graph_id
            or mask_receipt.get('binding_before') != old_binding
            or mask_receipt.get('binding_after') != new_binding
            or treatment.get('graph_id') != graph_id
            or treatment.get('binding_before') != old_binding
            or treatment.get('binding_after') != new_binding
            or _binding(old_leaf) != _binding(treatment_old)
            or _binding(new_leaf) != _binding(treatment_new)
            or treatment.get('mask_review', {}).get('sha256') != review_sha
            or treatment.get('mask_output', {}).get('sha256') != mask_sha
            or _binding(treatment.get('original_manifest')) != _binding(proof.get('prior_source'))
            or _binding(treatment.get('successor_manifest')) != _binding(proof.get('current_source'))
            or review.get('source', {}).get('graph_file_sha256') != old_leaf.get('sha256')
            or Path(review.get('source', {}).get('graph_file', '')).resolve() != old_leaf_path
            or treatment.get('rows_before') != treatment.get('rows_after')
            or treatment.get('changed_rows') != 1):
        raise ValueError('native target-mask correction differs from its exact review or leaf transformation')

    masked_rows = []
    with open_jsonl_text(mask_path) as stream:
        for line in stream:
            if line.strip():
                masked_rows.append(json.loads(line))
    if len(masked_rows) != 1 or masked_rows[0].get('id') != graph_id:
        raise ValueError('native target-mask artifact must contain exactly the reviewed graph')
    before_rows = {}
    with open_jsonl_text(old_leaf['path']) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('id') in before_rows:
                    raise ValueError('native target-mask source leaf repeats a graph ID')
                before_rows[row['id']] = row
    after_rows = {}
    with open_jsonl_text(new_leaf['path']) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('id') in after_rows:
                    raise ValueError('native target-mask corrected leaf repeats a graph ID')
                after_rows[row['id']] = row
    if (len(before_rows) != treatment.get('rows_before')
            or len(after_rows) != treatment.get('rows_after')
            or before_rows.keys() != after_rows.keys()
            or graph_id not in before_rows
            or after_rows[graph_id] != masked_rows[0]
            or sum(before_rows[key] != after_rows[key] for key in before_rows) != 1
            or treatment.get('unchanged_rows') != len(before_rows) - 1):
        raise ValueError('native target-mask source leaf contains changes outside the exact reviewed row')
    target = before_rows[graph_id]
    row_sha = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (target.get('split') != 'train' or target.get('tokenizer_fingerprint') != masked_rows[0].get('tokenizer_fingerprint')
            or row_sha(target) != mask_receipt.get('row_sha256_before')
            or row_sha(masked_rows[0]) != mask_receipt.get('row_sha256_after')):
        raise ValueError('native target-mask source rows differ from the reviewed exact train row')
    reviewed_spans = [{**span, 'expected_token_ids_sha256': span.get('token_ids_sha256')}
                      for span in (mask_receipt.get('spans') or [])]
    derived, derived_receipt = mask_exact_target_spans(target,
        tokenizer_fingerprint=target['tokenizer_fingerprint'], expected_id=graph_id,
        expected_binding=old_binding, spans=reviewed_spans)
    if (derived != masked_rows[0] or derived_receipt.get('binding_after') != new_binding
            or derived_receipt.get('row_sha256_before') != mask_receipt.get('row_sha256_before')
            or derived_receipt.get('row_sha256_after') != mask_receipt.get('row_sha256_after')):
        raise ValueError('native target-mask source transformation cannot be replayed exactly')
    return [
        {'path': str(treatment_path), 'sha256': treatment_sha, 'bytes': treatment_path.stat().st_size},
        {'path': str(review_path), 'sha256': review_sha, 'bytes': review_path.stat().st_size},
        {'path': str(mask_path), 'sha256': mask_sha, 'bytes': mask_path.stat().st_size},
        {'path': str(Path(old_leaf['path']).resolve()), 'sha256': old_leaf['sha256'],
         'bytes': Path(old_leaf['path']).stat().st_size},
        {'path': str(Path(new_leaf['path']).resolve()), 'sha256': new_leaf['sha256'],
         'bytes': Path(new_leaf['path']).stat().st_size},
    ]


def validate_held_correction_history(options, *, held_receipt):
    """Validate proof custody for held-source transitions independently of train rows."""
    if not isinstance(options, dict):
        return [], None
    history = options.get('held_correction_history') or []
    selected = options.get('held_correction_binding')
    if not isinstance(history, list):
        raise ValueError('held correction history is malformed')
    if not history:
        if selected is not None:
            raise ValueError('selected held correction is absent from its history')
        return [], None
    if (not isinstance(selected, dict) or not isinstance(selected.get('path'), str)
            or not isinstance(selected.get('sha256'), str)):
        raise ValueError('held correction history lacks its selected proof binding')
    prior_current = None
    model_id = None
    normalized = []
    for record in history:
        if not isinstance(record, dict):
            raise ValueError('held correction history record is malformed')
        reference = {'path': record.get('proof_path'), 'sha256': record.get('proof_sha256')}
        dependencies = source_correction_dependencies(reference)
        proof_path = Path(reference['path']).resolve(strict=True)
        proof = json.loads(proof_path.read_text())
        protected = proof.get('protected_held_correction')
        if (protected is None
                or _binding(record.get('prior_source')) != _binding(protected.get('prior_source'))
                or _binding(record.get('current_source')) != _binding(protected.get('current_source'))
                or record.get('origin_stage_id') != proof.get('stage_id')
                or record.get('model_id') != proof.get('model_id')):
            raise ValueError('held correction record differs from its authenticated full proof')
        expected_rows = _proof_file(proof_path, protected.get('row_bindings_path', ''))
        if (Path(str(record.get('row_bindings_path', ''))).resolve() != expected_rows
                or record.get('row_bindings_sha256') != protected.get('row_bindings_sha256')):
            raise ValueError('held correction record differs from its authenticated row ledger')
        if model_id is not None and proof.get('model_id') != model_id:
            raise ValueError('held correction history changes model identity')
        model_id = proof.get('model_id')
        if prior_current is not None and _binding(record.get('prior_source')) != _binding(prior_current):
            raise ValueError('held correction history is not a contiguous source chain')
        prior_current = record.get('current_source')
        normalized.append((record, proof, dependencies))
    last = normalized[-1][0]
    if (selected.get('sha256') != last.get('proof_sha256')
            or Path(selected.get('path')).resolve() != Path(last.get('proof_path')).resolve()):
        raise ValueError('selected held correction differs from its latest history record')
    last_proof = normalized[-1][1]
    if not _receipt_matches_with_staged_relocation(
            last.get('current_source'), held_receipt, last_proof,
            Path(normalized[-1][0]['proof_path']).resolve(strict=True)):
        raise ValueError('held correction history does not end at the loaded held source')
    return [item[0] for item in normalized], selected


def _fresh_quota(epochs, index, row_count, *, communication):
    """Match pending_exposure_batches' integer and fractional epoch schedule."""
    value = float(epochs)
    if not math.isfinite(value) or value < 0:
        raise ValueError('source correction requires a finite nonnegative epoch quota')
    whole = int(value)
    fractional = math.ceil((value - whole) * row_count)
    return max(int(bool(communication)), whole + int(index < fractional))


def _apply_protected_held_correction(result, training, proof, held, proof_path):
    """Apply exact same-membership held target replacements without train credit."""
    protected = proof.get('protected_held_correction')
    if protected is None:
        return 0
    if held is None:
        raise ValueError('protected held correction requires exact prior and current held sources')
    from ..common.jsonio import open_jsonl_text

    old_receipts = training.get('source_receipts') or []
    if len(old_receipts) < 2:
        raise ValueError('protected held correction requires the earned held source receipt')
    if (not _receipt_matches_with_staged_relocation(protected.get('prior_source'), old_receipts[1],
                proof, proof_path)
            or not _receipt_matches_with_staged_relocation(protected.get('prior_source'),
                held['prior_receipt'], proof, proof_path)
            or _binding(held['current_receipt']) != _binding(protected.get('current_source'))):
        raise ValueError('protected held correction source identity differs from the earned frame')
    prior_bindings = training.get('held_row_bindings') or {}
    current_bindings = held['current_bindings']
    prior_order, current_order = held['prior_order'], held['current_order']
    if (prior_bindings != held['prior_bindings'] or set(prior_bindings) != set(current_bindings)
            or list(prior_order) != list(current_order) or set(prior_order) != set(prior_bindings)):
        raise ValueError('protected held correction changes held membership or order')
    if set(held['prior_groups']) != set(held['current_groups']):
        raise ValueError('protected held correction changes held source groups')
    if held.get('prior_entries') is not None and held.get('current_entries') is not None:
        prior_classes={entry['id']:bool(entry.get('communication')) for entry in held['prior_entries']}
        current_classes={entry['id']:bool(entry.get('communication')) for entry in held['current_entries']}
        if prior_classes != current_classes:
            raise ValueError('protected held correction changes ordinary/communication class membership')
    prior_origins = _origins(held['prior_receipt'], prior_order)
    current_origins = _origins(held['current_receipt'], current_order)
    rows_path = _proof_file(proof_path, protected.get('row_bindings_path', ''))
    if sha256_file_hex_cached(rows_path) != protected.get('row_bindings_sha256'):
        raise ValueError('protected held correction row ledger digest differs')
    rows = {}
    with open_jsonl_text(rows_path) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                row_id = row.get('id')
                if not isinstance(row_id, str) or row_id in rows:
                    raise ValueError('protected held correction repeats or omits a row id')
                rows[row_id] = row
    changed = {row_id for row_id, binding in prior_bindings.items()
               if current_bindings.get(row_id) != binding}
    if set(rows) != changed:
        raise ValueError('protected held correction ledger does not exactly cover changed held rows')
    old_group_by_id, new_group_by_id = held['prior_groups_by_id'], held['current_groups_by_id']
    for row_id, row in rows.items():
        old_groups = tuple(old_group_by_id.get(row_id) or ())
        new_groups = tuple(new_group_by_id.get(row_id) or ())
        if (row.get('disposition') != 'corrected'
                or row.get('old_binding') != prior_bindings.get(row_id)
                or row.get('new_binding') != current_bindings.get(row_id)
                or row.get('old_source_sha256') != prior_origins.get(row_id)
                or row.get('new_source_sha256') != current_origins.get(row_id)
                or row.get('old_binding') == row.get('new_binding')
                or tuple(row.get('old_groups') or ()) != old_groups
                or tuple(row.get('new_groups') or ()) != new_groups or old_groups != new_groups):
            raise ValueError('protected held replacement changed identity, binding, or group: ' + row_id)
    result['held_row_bindings'] = dict(current_bindings)
    receipts = list(result.get('source_receipts') or [])
    if len(receipts) < 2:
        raise ValueError('protected held correction lost source receipt pair')
    receipts[1] = held['current_receipt']
    result['source_receipts'] = receipts
    return len(changed)


def _propagate_held_correction_to_frames(progress, *, held, proof, proof_path,
        quality_binding=None, excluded_ids=(), at_global_step=0):
    """Rebind every carried frame that consumed this exact protected held source.

    Train receipts and train exposure are intentionally not consulted or
    changed here: a DPO frame can share the held set while training on a
    different pair source. Held observations are retired only for rows whose
    target binding changed, then the exact quality exclusions are applied.
    """
    from .native_held import (bind_held_quality_exclusions,
                              rebind_held_coverage_after_exact_correction)

    proof_ref = {'path': str(Path(proof_path).resolve()),
                 'sha256': sha256_file_hex_cached(proof_path)}
    protected = proof.get('protected_held_correction') or {}
    current_receipt = held['current_receipt']
    binding_path = _proof_file(Path(proof_path), protected.get('row_bindings_path', ''))
    history_record = {
        'proof_path': proof_ref['path'], 'proof_sha256': proof_ref['sha256'],
        'model_id': proof.get('model_id'),
        'origin_stage_id': proof.get('stage_id'),
        'prior_source': protected.get('prior_source'),
        'current_source': protected.get('current_source'),
        'row_bindings_path': str(binding_path),
        'row_bindings_sha256': protected.get('row_bindings_sha256'),
        'weights_optimizer_rng_exposure': 'preserved',
        'held_observation_credit_from_correction': 0,
        'at_global_step': int(at_global_step),
    }
    if quality_binding is not None:
        history_record['held_quality_hold_binding'] = dict(quality_binding)
    transitioned = []
    for frame_id, original in list(progress.items()):
        receipts = list(original.get('source_receipts') or [])
        if len(receipts) < 2 or not _receipt_matches_with_staged_relocation(
                protected.get('prior_source'), receipts[1], proof, proof_path):
            continue
        frame = dict(original)
        if frame.get('held_row_bindings') != held['prior_bindings']:
            raise ValueError('carried held source has different exact row bindings: ' + str(frame_id))
        frame['held_row_bindings'] = dict(held['current_bindings'])
        receipts[1] = current_receipt
        frame['source_receipts'] = receipts

        metrics = dict(frame.get('metrics') or {})
        # The weight-EMA side evaluation keeps its own rotating ledger; it is re-bound the same way.
        for coverage_key in ('held_coverage', 'held_ema_coverage'):
            coverage = metrics.get(coverage_key)
            if coverage is None:
                continue
            coverage = rebind_held_coverage_after_exact_correction(
                held['prior_entries'], held['current_entries'], coverage,
                source_sha256=current_receipt['sha256'])
            if quality_binding is not None:
                coverage = bind_held_quality_exclusions(
                    held['current_entries'], coverage, excluded_ids,
                    source_sha256=current_receipt['sha256'], allow_transition=True)
                coverage['quality_hold_binding'] = dict(quality_binding)
            metrics[coverage_key] = coverage
        # A loss or accuracy summary measured the old held targets. Keep the
        # re-bound per-row observation ledger above, but require fresh scores.
        metrics.pop('held', None)
        metrics.pop('held_ema', None)
        frame['metrics'] = metrics

        options = dict(frame.get('options') or {})
        prior_quality = options.get('held_quality_hold_binding')
        if (prior_quality is not None and quality_binding is None
                and prior_quality.get('source_sha256') != current_receipt.get('sha256')):
            raise ValueError('held correction omits the carried frame quality hold for its new source')
        history = list(options.get('held_correction_history') or [])
        if not any(item.get('proof_sha256') == proof_ref['sha256']
                   and Path(str(item.get('proof_path', ''))).resolve() == Path(proof_ref['path']).resolve()
                   for item in history if isinstance(item, dict)):
            history.append(dict(history_record))
        options['held_correction_history'] = history
        options['held_correction_binding'] = dict(proof_ref)
        if quality_binding is not None:
            options['held_quality_hold_binding'] = dict(quality_binding)
        frame['options'] = options
        progress[frame_id] = frame
        transitioned.append({'stage_id': frame_id,
            'held_prior_source': _binding(protected.get('prior_source')),
            'held_current_source': _binding(protected.get('current_source')),
            'changed_held_rows': len({key for key, value in held['prior_bindings'].items()
                                      if held['current_bindings'].get(key) != value}),
            'held_visits_reset_for_changed_rows': True,
            'train_source_and_exposure': 'preserved'})
    return transitioned


def _propagate_source_correction_to_carried_frames(progress, *, stage_id, prior_receipt,
        current_receipt, prior_bindings, current_bindings, current_order,
        current_communication_ids, corrected, eligibility_decreased, deferred, added, pending,
        protected_held=None, proof=None, proof_path=None):
    """Rebind other saved SFT cursors that consumed the exact prior source.

    A checkpoint can carry several independently resumable supervised-stage
    cursors. Leaving one on the old target after correcting the shared source
    makes the checkpoint internally contradictory and can either reject a
    later resume or credit the retired target. Exact receipt equality scopes
    this transition; unrelated frames are not rewritten.
    """
    prior_ref = _binding(prior_receipt)
    current_ref = _binding(current_receipt)
    current_ids = set(current_bindings)
    prior_ids = set(prior_bindings)
    communication_ids = set(current_communication_ids)
    transitions = []
    for frame_id, original in progress.items():
        if frame_id == stage_id:
            continue
        receipts = list(original.get('source_receipts') or [])
        if not receipts or _binding(receipts[0]) != prior_ref:
            continue
        frame = dict(original)
        bindings = dict(frame.get('source_row_bindings') or {})
        order = list(frame.get('source_row_order') or [])
        if bindings != prior_bindings or len(order) != len(prior_ids) or set(order) != prior_ids:
            raise ValueError('carried frame bound to corrected source has a different row inventory: '+frame_id)
        counts = dict(frame.get('exposure_counts') or {})
        quotas = dict(frame.get('required_visits') or {})
        if set(counts) != prior_ids or not set(quotas).issubset(prior_ids):
            raise ValueError('carried frame exposure inventory differs from corrected source: '+frame_id)
        options = dict(frame.get('options') or {})
        epochs = float(options.get('epochs', 1))
        prior_frame_exposure = {key: int(counts.get(key, 0))
                                for key in corrected | eligibility_decreased | deferred}

        # Current source order is authoritative. Preserve earned counts for
        # unchanged IDs, start corrected/new IDs at zero, and remove only the
        # explicitly deferred rows into the exact pending-source ledger.
        next_counts = {key: int(counts.get(key, 0)) for key in current_ids}
        next_quotas = {key: int(quotas[key]) for key in current_ids if key in quotas}
        positions = {key: index for index, key in enumerate(current_order)}
        for key in added | corrected:
            next_counts[key] = 0 if key in corrected else int(counts.get(key, 0))
            next_quotas[key] = _fresh_quota(epochs, positions[key], len(current_order),
                                              communication=key in communication_ids)
        for key in corrected:
            next_counts[key] = 0
            next_quotas[key] = _fresh_quota(epochs, positions[key], len(current_order),
                                              communication=key in communication_ids)
        for key in deferred:
            next_counts.pop(key, None)
            next_quotas.pop(key, None)
        frame['source_row_bindings'] = dict(current_bindings)
        frame['source_row_order'] = list(current_order)
        receipts[0] = current_receipt
        frame['source_receipts'] = receipts
        frame['exposure_counts'] = next_counts
        frame['required_visits'] = next_quotas
        frame_pending = {**(frame.get('pending_source_rows') or {}), **pending}
        for key in corrected:
            frame_pending.pop(key, None)
        frame['pending_source_rows'] = frame_pending
        for option_key in ('_communication_ids', '_producer_consumer_ids'):
            if option_key in options:
                options[option_key] = [key for key in current_order if key in communication_ids]
        frame['options'] = options

        if corrected or eligibility_decreased or added or deferred:
            frame['completed_foundations'] = []
        progress[frame_id] = frame
        transitions.append({'stage_id': frame_id,
            'prior_exposure': prior_frame_exposure,
            'weights_optimizer_rng': 'preserved',
            'corrected_rows_reexposed': sorted(corrected),
            'eligibility_decreased_rows_exposure_preserved': sorted(eligibility_decreased),
            'deferred_rows': sorted(deferred),
            'new_rows_unexposed': sorted(added)})
    return transitions


def apply_source_correction(training, proof_path, *, model_id, stage_id,
                            current_source_receipt, current_bindings, current_order,
                            current_communication_ids, protected_held=None):
    """Apply a proof to an exact earned frame, preserving weights/optimizer/RNG.

    The proof's JSONL row ledger has records with ``id``, ``old_binding``,
    ``new_binding``, ``old_source_sha256``, ``new_source_sha256`` and
    ``disposition`` (``corrected``, ``eligibility-decreased``, ``deferred``,
    or ``added``). Corrected communication IDs start fresh exposure. A reviewed
    same-ID ordinary eligibility decrease preserves row position, exposure,
    and required visits. Deferred rows keep their old bindings and visit debt
    in the pending ledger until a later correction proof. Any unrelated
    appended rows are explicitly bound as ``added``.
    """
    from ..common.jsonio import open_jsonl_text

    path = Path(proof_path).resolve(strict=True)
    proof_sha = sha256_file_hex_cached(path)
    proof = json.loads(path.read_text())
    if proof.get('schema') != SCHEMA or proof.get('model_id') != model_id or proof.get('stage_id') != stage_id:
        raise ValueError('native source correction model/stage identity differs')
    ref = {'path': str(path), 'sha256': proof_sha}
    source_correction_dependencies(ref)
    history = (training.get('options') or {}).get('source_correction_history') or []
    active_ref = (training.get('options') or {}).get('source_correction_binding')
    if any(item.get('proof_sha256') == proof_sha for item in history):
        if active_ref != ref:
            raise ValueError('source correction history exists but its active proof differs')
        record = next(item for item in history if item.get('proof_sha256') == proof_sha)
        if (_binding(record.get('prior_source')) != _binding(proof.get('prior_source'))
                or _binding(record.get('current_source')) != _binding(proof.get('current_source'))
                or _binding(current_source_receipt) != _binding(proof.get('current_source'))
                or _binding((training.get('source_receipts') or [{}])[0]) != _binding(proof.get('current_source'))
                or current_bindings != training.get('source_row_bindings')
                or list(current_order) != training.get('source_row_order')):
            raise ValueError('source correction replay differs from its committed exact current source')
        protected = proof.get('protected_held_correction')
        if protected is not None and (protected_held is None
                or _binding(protected_held['current_receipt']) != _binding(protected.get('current_source'))
                or protected_held['current_bindings'] != training.get('held_row_bindings')):
            raise ValueError('protected held source correction replay differs from its committed source')
        return training, {'changed': False, 'proof_sha256': proof_sha}

    prior_receipts = training.get('source_receipts') or []
    if not prior_receipts:
        raise ValueError('source correction requires the earned source receipt')
    prior_ref = _binding(proof.get('prior_source'))
    prior_receipt = prior_receipts[0]
    if not _receipt_matches_with_staged_relocation(proof.get('prior_source'), prior_receipt,
            proof, path):
        raise ValueError('source correction predecessor differs from earned source')
    current_ref = _binding(proof.get('current_source'))
    if _binding(current_source_receipt) != current_ref:
        raise ValueError('source correction target differs from loaded native source')

    old_bindings = training.get('source_row_bindings') or {}
    old_order = training.get('source_row_order') or []
    if len(old_order) != len(old_bindings) or set(old_order) != set(old_bindings):
        raise ValueError('source correction needs complete earned row bindings and order')
    old_origins = _origins(prior_receipt, old_order)
    new_origins = _origins(current_source_receipt, current_order)
    progress_before = training.get('sft_progress') or {}
    source_frame = progress_before.get(stage_id) or {}
    top_pending = dict(training.get('pending_source_rows') or {})
    frame_pending = dict(source_frame.get('pending_source_rows') or {})
    if frame_pending and not _receipt_matches_with_staged_relocation(
            proof.get('prior_source'), (source_frame.get('source_receipts') or [{}])[0], proof, path):
        raise ValueError('pending source rows belong to a different saved stage source')
    for row_id, item in top_pending.items():
        if row_id in frame_pending and frame_pending[row_id] != item:
            raise ValueError('top-level and stage-frame pending source rows differ: ' + row_id)
    pending = {**frame_pending, **top_pending}
    row_path = _proof_file(path, proof.get('row_bindings_path', ''))
    if sha256_file_hex_cached(row_path) != proof.get('row_bindings_sha256'):
        raise ValueError('source correction row ledger digest differs')
    rows = {}
    with open_jsonl_text(row_path) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                row_id = row.get('id')
                if not isinstance(row_id, str) or row_id in rows:
                    raise ValueError('source correction row ledger repeats or omits an id')
                rows[row_id] = row
    corrected, eligibility_decreased, deferred, added = set(), set(), set(), set()
    static_context = proof.get('static_native_context_correction')
    static_rows = {}
    if static_context is not None:
        from .native_static_context_correction import verify_static_context_correction
        static_rows, _ = verify_static_context_correction(path, static_context)
    static_used = set()
    changes = []
    counts = dict(training.get('exposure_counts') or {})
    quotas = dict(training.get('required_visits') or {})
    for row_id in pending:
        if row_id not in counts and row_id in (source_frame.get('exposure_counts') or {}):
            counts[row_id] = source_frame['exposure_counts'][row_id]
        if row_id not in quotas and row_id in (source_frame.get('required_visits') or {}):
            quotas[row_id] = source_frame['required_visits'][row_id]
    options = training.get('options') or {}
    if static_rows:
        from .own_output_supervision import identity, policy_objective
        policy_ref=static_context['ce_policy']
        policy_path=_proof_file(path,policy_ref['path'])
        static_identity=identity(policy_objective(json.loads(policy_path.read_text())))
        if (options.get('supervision_objective') or {}).get('sha256')!=static_identity['sha256']:
            raise ValueError('static context correction objective differs from the earned supervision policy')
    prior_communication = set(options.get('_communication_ids') or [])
    current_communication = set(current_communication_ids)
    epochs = float(options.get('epochs', 1))
    current_positions = {row_id: index for index, row_id in enumerate(current_order)}
    target_mask_correction = proof.get('target_mask_correction')
    in_place_mask_ids = ({target_mask_correction['graph_id']}
                         if isinstance(target_mask_correction, dict) else set())
    if len(in_place_mask_ids) > 1:
        raise ValueError('one source correction proof may mask only one ordinary graph in place')
    for row_id, row in rows.items():
        disposition = row.get('disposition')
        old = row.get('old_binding')
        old_sha = row.get('old_source_sha256')
        active_old = old_bindings.get(row_id)
        pending_item = pending.get(row_id) or {}
        pending_old = pending_item.get('binding')
        actual_old = active_old if active_old is not None else pending_old
        actual_origin = old_origins.get(row_id) if active_old is not None else pending_item.get('source_sha256')
        if disposition == 'added':
            new = row.get('new_binding')
            new_sha = row.get('new_source_sha256')
            if (active_old is not None or pending_old is not None
                    or old is not None or old_sha is not None
                    or row_id not in current_bindings or current_bindings[row_id] != new
                    or new_origins.get(row_id) != new_sha):
                raise ValueError('source correction addition is not an exact new source row: ' + row_id)
            added.add(row_id)
            continue
        if old != actual_old or actual_origin != old_sha or not actual_origin:
            raise ValueError('source correction old row/source binding is not earned: ' + row_id)
        if disposition == 'corrected':
            new = row.get('new_binding')
            new_sha = row.get('new_source_sha256')
            if active_old is not None:
                old_communication = row_id in prior_communication
                if 'old_communication' in row and row.get('old_communication') is not old_communication:
                    raise ValueError('source correction changed the authenticated old row class: ' + row_id)
            else:
                old_communication, _ = _pending_communication_class(
                    training, pending_item, row_id, prior_receipt)
                if ('old_communication' not in row
                        or type(row.get('old_communication')) is not bool
                        or row['old_communication'] is not old_communication):
                    raise ValueError('pending replacement lacks its exact pinned old row class: ' + row_id)
            static = static_rows.get(row_id)
            ordinary_static = (static is not None and not old_communication and static.get('old_binding')==old
                               and static.get('new_binding')==new and counts.get(row_id,0)==0)
            if (row_id not in current_bindings or current_bindings[row_id] != new
                    or new_origins.get(row_id) != new_sha or not (old_communication or ordinary_static)
                    or row_id not in current_communication
                    or old == new):
                raise ValueError('source correction replacement is not an exact changed communication row: ' + row_id)
            if ordinary_static:static_used.add(row_id)
            corrected.add(row_id)
            if row_id in pending:
                pending.pop(row_id)
            changes.append({
                'id': row_id, 'old_binding': old, 'new_binding': new,
                'old_source_sha256': old_sha, 'new_source_sha256': new_sha,
                'prior_exposure': int(counts.get(row_id, 0)),
                'prior_required_visits': int(quotas.get(row_id, 0)),
            })
        elif disposition == 'eligibility-decreased':
            new = row.get('new_binding')
            new_sha = row.get('new_source_sha256')
            declaration = target_mask_correction
            if (row_id not in in_place_mask_ids or declaration is None
                    or active_old is None
                    or row_id not in current_bindings or current_bindings[row_id] != new
                    or new_origins.get(row_id) != new_sha
                    or row_id in prior_communication or row_id in current_communication
                    or declaration.get('binding_before') != old
                    or declaration.get('binding_after') != new
                    or declaration.get('source_leaf_before', {}).get('sha256') != old_sha
                    or declaration.get('source_leaf_after', {}).get('sha256') != new_sha
                    or old == new):
                raise ValueError('eligibility decrease is not the exact reviewed ordinary target-mask row: ' + row_id)
            eligibility_decreased.add(row_id)
            changes.append({
                'id': row_id, 'old_binding': old, 'new_binding': new,
                'old_source_sha256': old_sha, 'new_source_sha256': new_sha,
                'prior_exposure': int(counts.get(row_id, 0)),
                'prior_required_visits': int(quotas.get(row_id, 0)),
                'exposure_preserved': True,
            })
        elif disposition == 'deferred':
            if row_id in current_bindings or row.get('new_binding') is not None:
                raise ValueError('deferred source correction row remains active: ' + row_id)
            deferred.add(row_id)
            prior = pending.get(row_id)
            required = max(1, int(quotas.get(row_id,
                (prior or {}).get('required_visits', math.ceil(epochs)))))
            if active_old is not None:
                communication = row_id in prior_communication
                if 'old_communication' in row and row.get('old_communication') is not communication:
                    raise ValueError('source correction deferred row class differs from saved source: ' + row_id)
                classification = {'schema':'natlang.pending-row-classification-provenance/1',
                    'row_id':row_id, 'model_id':model_id, 'stage_id':stage_id,
                    'classification_source':'saved_checkpoint_communication_ids',
                    'proof_path':str(path), 'proof_sha256':proof_sha,
                    'source_receipt':prior_ref}
            else:
                communication, _ = _pending_communication_class(training, prior, row_id, prior_receipt)
                if ('old_communication' not in row or type(row.get('old_communication')) is not bool
                        or row['old_communication'] is not communication):
                    raise ValueError('re-deferred row lacks its exact pinned old class: ' + row_id)
                classification = dict(prior['classification_provenance'])
            retired_exposure = (dict(prior['retired_exposure']) if prior is not None else {
                'visits':int(counts.get(row_id, 0)),
                'required_visits':int(quotas.get(row_id, required)),
                'source_receipt':prior_ref})
            item = {'binding': old, 'source_sha256': old_sha,
                    'required_visits': required, 'reason': row.get('reason'),
                    'communication': communication,
                    'classification_provenance': classification,
                    'retired_exposure': retired_exposure}
            if prior is not None and prior != item:
                raise ValueError('source correction changed a prior pending row: ' + row_id)
            pending[row_id] = item
        else:
            raise ValueError('source correction row disposition is unsupported')

    if static_used!=set(static_rows):
        raise ValueError('static context correction evidence does not exactly cover ordinary replacements')
    changed_active = {key for key, value in old_bindings.items()
                      if current_bindings.get(key) not in (None, value)}
    removed_active = set(old_bindings) - set(current_bindings)
    corrected_active = (corrected | eligibility_decreased) & set(old_bindings)
    deferred_active = deferred & set(old_bindings)
    actual_added = set(current_bindings) - set(old_bindings) - corrected - eligibility_decreased
    if (changed_active != corrected_active or removed_active != deferred_active
            or actual_added != added):
        raise ValueError('source correction ledger does not exactly cover changed/removed rows')
    if len(current_order) != len(current_bindings) or set(current_order) != set(current_bindings):
        raise ValueError('source correction current order is not a one-to-one row inventory')
    # Explicitly corrected rows start a fresh exposure schedule, so their
    # replacement can be appended after the unchanged old-order prefix. The
    # row ledger binds those identities and prior exposure below; unchanged
    # earned rows still keep their exact positional prefix.
    retained_order = [key for key in old_order
                      if key not in deferred and key not in ((corrected | eligibility_decreased) - in_place_mask_ids)]
    if current_order[:len(retained_order)] != retained_order:
        raise ValueError('source correction reordered earned rows or inserted before their prefix')
    if not set(current_bindings).issuperset(set(old_bindings) - deferred):
        raise ValueError('source correction removed an undeclared original row')
    if any(current_bindings[key] != old_bindings[key] for key in old_bindings
           if key not in corrected and key not in eligibility_decreased and key not in deferred):
        raise ValueError('source correction changed an undeclared row binding')
    if not corrected and not eligibility_decreased and not deferred:
        raise ValueError('source correction has neither a replacement nor an explicit deferred obligation')

    result = dict(training)
    result['source_row_bindings'] = dict(current_bindings)
    result['source_row_order'] = list(current_order)
    result['source_receipts'] = [current_source_receipt, *(prior_receipts[1:])]
    result['pending_source_rows'] = pending
    new_counts = dict(counts)
    new_quotas = dict(quotas)
    for row_id in set(current_bindings) - set(old_bindings) - set(corrected) - set(eligibility_decreased):
        new_counts.setdefault(row_id, 0)
        new_quotas.setdefault(row_id, _fresh_quota(epochs, current_positions[row_id], len(current_order),
                                                    communication=row_id in current_communication))
    for row_id in corrected:
        new_counts[row_id] = 0
        new_quotas[row_id] = _fresh_quota(epochs, current_positions[row_id], len(current_order),
                                          communication=row_id in current_communication)
        if row_id in static_used:
            policy=options.get('communication_quota_policy') or {}
            minimum=policy.get('visits',1)
            if type(minimum) is not int or minimum<1:
                raise ValueError('static context correction has an invalid communication minimum')
            new_quotas[row_id]=max(new_quotas[row_id],minimum)
    for row_id in deferred:
        new_counts.setdefault(row_id, int(counts.get(row_id, 0)))
    revision, revision_ledger = declared_row_ledger(path, proof, 'quota_revision')
    revised = {}
    for row_id, item in (revision or {}).items():
        if (row_id in corrected or row_id in eligibility_decreased or row_id in deferred or row_id in added
                or old_bindings.get(row_id) != item['binding'] or current_bindings.get(row_id) != item['binding']):
            raise ValueError('quota revision names a row that is not an exact unchanged earned row: ' + row_id)
        if int(quotas.get(row_id, -1)) != item['prior_required_visits']:
            raise ValueError('quota revision prior quota differs from the earned saved quota: ' + row_id)
        # Earned credit is history: a lowered quota never falls below actual visits.
        revised[row_id] = max(item['required_visits'], int(counts.get(row_id, 0)))
        new_quotas[row_id] = revised[row_id]
    result['exposure_counts'] = new_counts
    result['required_visits'] = new_quotas
    result['options'] = dict(options)
    if revised:
        objective_quotas = dict(result['options'].get('_objective_required_visits') or {})
        for row_id, value in revised.items():
            if row_id in objective_quotas:
                objective_quotas[row_id] = value
        result['options']['_objective_required_visits'] = objective_quotas
    if static_used:
        objective_quotas=dict(result['options'].get('_objective_required_visits') or {})
        for row_id in static_used:objective_quotas[row_id]=new_quotas[row_id]
        result['options']['_objective_required_visits']=objective_quotas
    for option_key in ('_communication_ids', '_producer_consumer_ids'):
        if option_key in result['options']:
            result['options'][option_key] = [key for key in current_order if key in current_communication]
    new_history = list(history)
    record = {
        'proof_path': str(path), 'proof_sha256': proof_sha,
        'model_id': model_id,
        'prior_source': prior_ref, 'current_source': current_ref,
        'stage_id': stage_id,
        'corrected_rows': [item for item in changes if not item.get('exposure_preserved')],
        'eligibility_decreased_rows': [item for item in changes if item.get('exposure_preserved')],
        'deferred_rows': sorted(deferred), 'added_rows': sorted(added),
        'deferred_row_classes': {row_id:(pending.get(row_id) or {}).get('communication')
            for row_id in sorted(deferred)},
        'old_target_exposure_retired': bool(corrected),
        'eligibility_decrease_preserves_exposure': sorted(eligibility_decreased),
        'optimizer_rng_weights': 'preserved',
        'at_global_step': int(training.get('global_step', training.get('step', 0))),
        'new_corrected_row_exposure_added': 0,
        **({'static_native_context_rows': sorted(static_used),
            'static_native_context_required_visits': {row_id:new_quotas[row_id] for row_id in sorted(static_used)}}
           if static_used else {}),
        **({'quota_revision': {'ledger': revision_ledger, 'rows': len(revised),
            'reason': proof['quota_revision']['reason'],
            'required_visits_removed': sum(revision[k]['prior_required_visits'] - v for k, v in revised.items()),
            'earned_credit_retained': True}} if revised else {}),
    }
    new_history.append(record)
    result['options']['source_correction_history'] = new_history
    result['options']['source_correction_binding'] = ref

    # Any previously completed foundation depended on the old communication
    # row bindings. Keep its record as historical evidence, but make it
    # unavailable for the corrected source until fresh qualification.
    progress = dict(training.get('sft_progress') or {})
    frame = dict(progress.get(stage_id) or {})
    frame_completed = list(frame.get('completed_foundations') or [])
    completed = sorted(set(training.get('completed_foundations') or []) | set(frame_completed))
    qualification_history = list(result['options'].get('foundation_qualification_history') or [])
    if completed:
        qualification_history.append({
            'invalidated_by_source_correction': proof_sha,
            'prior_completed_foundations': completed,
            'at_global_step': record['at_global_step'],
            'no_new_qualification_claim': True,
        })
        result['completed_foundations'] = []
    result['options']['foundation_qualification_history'] = qualification_history
    if frame:
        frame['source_row_bindings'] = dict(current_bindings)
        frame['source_row_order'] = list(current_order)
        frame['source_receipts'] = [current_source_receipt, *(frame.get('source_receipts') or [])[1:]]
        frame_counts = dict(frame.get('exposure_counts') or {})
        frame_quotas = dict(frame.get('required_visits') or {})
        for row_id in set(current_bindings) - set(old_bindings) - set(corrected) - set(eligibility_decreased):
            frame_counts.setdefault(row_id, 0)
            frame_quotas.setdefault(row_id, _fresh_quota(epochs, current_positions[row_id], len(current_order),
                                                          communication=row_id in current_communication))
        for row_id in corrected:
            frame_counts[row_id] = 0
            frame_quotas[row_id] = _fresh_quota(epochs, current_positions[row_id], len(current_order),
                                                communication=row_id in current_communication)
            if row_id in static_used:frame_quotas[row_id]=new_quotas[row_id]
        for row_id, value in revised.items():
            frame_quotas[row_id] = value
        frame['exposure_counts'] = frame_counts
        frame['required_visits'] = frame_quotas
        frame['pending_source_rows'] = dict(pending)
        if frame_completed:
            frame['completed_foundations'] = []
        frame_options = dict(frame.get('options') or {})
        frame_options['source_correction_history'] = list(new_history)
        frame_options['source_correction_binding'] = ref
        frame_options['foundation_qualification_history'] = list(qualification_history)
        if static_used:
            frame_objective_quotas=dict(frame_options.get('_objective_required_visits') or {})
            for row_id in static_used:frame_objective_quotas[row_id]=new_quotas[row_id]
            frame_options['_objective_required_visits']=frame_objective_quotas
        if revised and frame_options.get('_objective_required_visits'):
            frame_objective_quotas=dict(frame_options['_objective_required_visits'])
            for row_id, value in revised.items():
                if row_id in frame_objective_quotas:frame_objective_quotas[row_id]=value
            frame_options['_objective_required_visits']=frame_objective_quotas
        for option_key in ('_communication_ids', '_producer_consumer_ids'):
            if option_key in frame_options:
                frame_options[option_key] = [key for key in current_order if key in current_communication]
        frame['options'] = frame_options
        progress[stage_id] = frame
    result['sft_progress'] = progress
    protected_held_rows = _apply_protected_held_correction(
        result, training, proof, protected_held, path)
    quality_ref = proof.get('held_quality_hold')
    quality_binding = None
    quality_excluded_ids = set()
    prior_quality_binding = (training.get('options') or {}).get('held_quality_hold_binding')
    if prior_quality_binding is not None and quality_ref is None:
        raise ValueError('source correction cannot silently discard an existing held quality hold')
    if quality_ref is not None:
        if protected_held is None:
            raise ValueError('held quality hold requires the exact loaded protected held source')
        quality = validate_held_quality_hold({**quality_ref,
            'path':str(_proof_file(path, quality_ref.get('path', '')))},
            held_receipt=protected_held['current_receipt'],
            held_entries=protected_held['current_entries'])
        if prior_quality_binding is not None:
            prior_receipt = (training.get('source_receipts') or [{}, {}])[1]
            prior_quality = validate_held_quality_hold(prior_quality_binding,
                held_receipt=prior_receipt, held_entries=protected_held['prior_entries'])
            held_correction = proof.get('protected_held_correction') or {}
            held_rows_path = _proof_file(path, held_correction.get('row_bindings_path', ''))
            from ..common.jsonio import open_jsonl_text
            corrected_held_ids = set()
            with open_jsonl_text(held_rows_path) as stream:
                corrected_held_ids = {row['id'] for line in stream if line.strip()
                    for row in [json.loads(line)] if row.get('disposition') == 'corrected'}
            no_longer_held = prior_quality['excluded_ids'] - quality['excluded_ids']
            if no_longer_held - corrected_held_ids:
                raise ValueError('source correction drops held exclusions without exact protected target replacements')
        pending_held = (proof.get('pending_held_correction') or {}).get('remaining_original_sources')
        if not isinstance(pending_held, list):
            raise ValueError('held quality hold lacks its exact pending correction inventory')
        expected = {}
        for item in pending_held:
            source = item.get('source') or {}
            digest, rows_count, reason = source.get('sha256'), item.get('rows'), item.get('reason')
            if (not isinstance(digest, str) or type(rows_count) is not int
                    or rows_count < 1 or not isinstance(reason, str) or not reason
                    or digest in expected):
                raise ValueError('pending held correction inventory is malformed')
            expected[digest] = (rows_count, reason)
        manifest = json.loads(_proof_file(path, quality_ref['path']).read_text())
        actual = {item['sha256']: (item['rows'], item['reason'])
                  for item in manifest.get('pending_sources', [])}
        if actual != expected:
            raise ValueError('held quality hold differs from unresolved protected correction debt')
        quality_binding = {
            **quality['manifest'], 'row_bindings_path': quality['row_bindings']['path'],
            'row_bindings_sha256': quality['row_bindings']['sha256'],
            'source_sha256': quality['source_sha256'],
            'excluded_rows': quality['excluded_rows']}
        result['options']['held_quality_hold_binding'] = quality_binding
        quality_excluded_ids = quality['excluded_ids']
        active_frame = (result.get('sft_progress') or {}).get(stage_id)
        if active_frame is not None:
            frame_options = dict(active_frame.get('options') or {})
            frame_options['held_quality_hold_binding'] = dict(quality_binding)
            active_frame['options'] = frame_options
        record['held_quality_hold'] = {
            'manifest_sha256': quality['manifest']['sha256'],
            'row_bindings_sha256': quality['row_bindings']['sha256'],
            'excluded_rows': quality['excluded_rows'],
            'full_held_qualification': False}
    progress = dict(result.get('sft_progress') or {})
    carried_transitions = _propagate_source_correction_to_carried_frames(
        progress, stage_id=stage_id, prior_receipt=prior_receipt,
        current_receipt=current_source_receipt, prior_bindings=old_bindings,
        current_bindings=current_bindings, current_order=current_order,
        current_communication_ids=current_communication_ids, corrected=corrected,
        eligibility_decreased=eligibility_decreased,
        deferred=deferred, added=added, pending=pending,
        protected_held=protected_held, proof=proof, proof_path=path)
    held_transitions = []
    if protected_held is not None:
        progress = dict(result.get('sft_progress') or {})
        held_transitions = _propagate_held_correction_to_frames(
            progress, held=protected_held, proof=proof, proof_path=path,
            quality_binding=quality_binding, excluded_ids=quality_excluded_ids,
            at_global_step=int(training.get('global_step', training.get('step', 0))))
        result['sft_progress'] = progress
        active_frame = progress.get(stage_id)
        if active_frame is not None:
            result['metrics'] = dict(active_frame.get('metrics') or {})
            active_options = dict(active_frame.get('options') or {})
            for key in ('held_correction_history', 'held_correction_binding',
                        'held_quality_hold_binding'):
                if key in active_options:
                    result['options'][key] = active_options[key]
        if held_transitions:
            result['options']['held_correction_history'] = list(
                (result.get('options') or {}).get('held_correction_history') or [])
            result['options']['held_correction_binding'] = dict(
                progress[stage_id].get('options', {}).get('held_correction_binding')
                if stage_id in progress else {'path':str(path),'sha256':proof_sha})
    result['sft_progress'] = progress
    if carried_transitions:
        result['options']['source_correction_history'][-1]['carried_frame_transitions'] = carried_transitions
        transitioned_ids = {item['stage_id'] for item in carried_transitions}
        for frame_id in transitioned_ids:
            frame = progress[frame_id]
            frame_options = dict(frame.get('options') or {})
            frame_options['source_correction_history'] = list(result['options']['source_correction_history'])
            frame_options['source_correction_binding'] = ref
            frame_options['foundation_qualification_history'] = list(qualification_history)
            if quality_ref is not None:
                frame_options['held_quality_hold_binding'] = dict(result['options']['held_quality_hold_binding'])
            frame['options'] = frame_options
    if held_transitions:
        result['options']['source_correction_history'][-1]['held_frame_transitions'] = held_transitions
    if protected_held_rows:
        record['protected_held_rows'] = protected_held_rows
    summary = {'changed': True, 'proof_sha256': proof_sha,
               'corrected_rows': len(corrected), 'deferred_rows': len(deferred),
               'added_rows': len(added), 'protected_held_rows': protected_held_rows}
    if eligibility_decreased:
        summary['eligibility_decreased_rows'] = len(eligibility_decreased)
    if revised:
        summary['quota_revised'] = dict(revised)
    return result, summary
