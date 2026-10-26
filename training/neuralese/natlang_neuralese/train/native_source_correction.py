"""Exact row-binding transition for repaired native graph targets.

This mechanism changes only source/exposure metadata. It never edits model,
optimizer, RNG, or numeric state. Old target exposure is retained as history;
corrected communication rows begin with fresh exposure debt.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached

SCHEMA = 'natlang.native-source-correction/1'


def _binding(ref):
    if not isinstance(ref, dict) or not all(isinstance(ref.get(k), str) and ref[k]
                                             for k in ('path', 'sha256')):
        raise ValueError('source correction requires exact source references')
    return {'path': str(Path(ref['path']).resolve()), 'sha256': ref['sha256']}


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


def source_correction_dependencies(reference):
    """Validate and return the immutable proof file used by provider closure."""
    path = Path(reference['path']).resolve(strict=True)
    digest = sha256_file_hex_cached(path)
    if digest != reference.get('sha256'):
        raise ValueError('native source correction proof changed')
    proof = json.loads(path.read_text())
    if proof.get('schema') != SCHEMA:
        raise ValueError('native source correction schema differs')
    rows_path = Path(proof.get('row_bindings_path', '')).resolve(strict=True)
    rows_sha = sha256_file_hex_cached(rows_path)
    if rows_sha != proof.get('row_bindings_sha256'):
        raise ValueError('native source correction row ledger changed')
    return [
        {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size},
        {'path': str(rows_path), 'sha256': rows_sha, 'bytes': rows_path.stat().st_size},
    ]


def apply_source_correction(training, proof_path, *, model_id, stage_id,
                            current_source_receipt, current_bindings, current_order,
                            current_communication_ids):
    """Apply a proof to an exact earned frame, preserving weights/optimizer/RNG.

    The proof's JSONL row ledger has records with ``id``, ``old_binding``,
    ``new_binding``, ``old_source_sha256``, ``new_source_sha256`` and
    ``disposition`` (``corrected``, ``deferred``, or ``added``). Corrected IDs
    are stable; only their row bindings change. Deferred rows keep their old
    bindings and visit debt in the pending ledger until a later correction
    proof. Any unrelated appended rows are explicitly bound as ``added``.
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
        return training, {'changed': False, 'proof_sha256': proof_sha}

    prior_receipts = training.get('source_receipts') or []
    if not prior_receipts:
        raise ValueError('source correction requires the earned source receipt')
    prior_ref = _binding(proof.get('prior_source'))
    prior_receipt = prior_receipts[0]
    if _binding(prior_receipt) != prior_ref:
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
    pending = dict(training.get('pending_source_rows') or {})
    row_path = Path(proof.get('row_bindings_path', '')).resolve(strict=True)
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
    corrected, deferred, added = set(), set(), set()
    changes = []
    counts = training.get('exposure_counts') or {}
    quotas = dict(training.get('required_visits') or {})
    options = training.get('options') or {}
    prior_communication = set(options.get('_communication_ids') or [])
    current_communication = set(current_communication_ids)
    epoch_quota = max(1, int(float(options.get('epochs', 1))))
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
            if (row_id not in current_bindings or current_bindings[row_id] != new
                    or new_origins.get(row_id) != new_sha or row_id not in prior_communication
                    or row_id not in current_communication
                    or old == new):
                raise ValueError('source correction replacement is not an exact changed communication row: ' + row_id)
            corrected.add(row_id)
            if row_id in pending:
                pending.pop(row_id)
            changes.append({
                'id': row_id, 'old_binding': old, 'new_binding': new,
                'old_source_sha256': old_sha, 'new_source_sha256': new_sha,
                'prior_exposure': int(counts.get(row_id, 0)),
                'prior_required_visits': int(quotas.get(row_id, 0)),
            })
        elif disposition == 'deferred':
            if row_id in current_bindings or row.get('new_binding') is not None:
                raise ValueError('deferred source correction row remains active: ' + row_id)
            deferred.add(row_id)
            required = max(1, int(quotas.get(row_id, epoch_quota)))
            prior = pending.get(row_id)
            item = {'binding': old, 'source_sha256': old_sha,
                    'required_visits': required,
                    'reason': row.get('reason')}
            if prior is not None and prior != item:
                raise ValueError('source correction changed a prior pending row: ' + row_id)
            pending[row_id] = item
        else:
            raise ValueError('source correction row disposition is unsupported')

    changed_active = {key for key, value in old_bindings.items()
                      if current_bindings.get(key) not in (None, value)}
    removed_active = set(old_bindings) - set(current_bindings)
    corrected_active = corrected & set(old_bindings)
    deferred_active = deferred & set(old_bindings)
    actual_added = set(current_bindings) - set(old_bindings) - corrected
    if (changed_active != corrected_active or removed_active != deferred_active
            or actual_added != added):
        raise ValueError('source correction ledger does not exactly cover changed/removed rows')
    if len(current_order) != len(current_bindings) or set(current_order) != set(current_bindings):
        raise ValueError('source correction current order is not a one-to-one row inventory')
    retained_order = [key for key in old_order if key not in deferred]
    if current_order[:len(retained_order)] != retained_order:
        raise ValueError('source correction reordered earned rows or inserted before their prefix')
    if not set(current_bindings).issuperset(set(old_bindings) - deferred):
        raise ValueError('source correction removed an undeclared original row')
    if any(current_bindings[key] != old_bindings[key] for key in old_bindings
           if key not in corrected and key not in deferred):
        raise ValueError('source correction changed an undeclared row binding')
    if not corrected and not deferred:
        raise ValueError('source correction has neither a replacement nor an explicit deferred obligation')

    result = dict(training)
    result['source_row_bindings'] = dict(current_bindings)
    result['source_row_order'] = list(current_order)
    result['source_receipts'] = [current_source_receipt, *(prior_receipts[1:])]
    result['pending_source_rows'] = pending
    new_counts = dict(counts)
    new_quotas = dict(quotas)
    for row_id in set(current_bindings) - set(old_bindings) - set(corrected):
        new_counts.setdefault(row_id, 0)
        new_quotas.setdefault(row_id, epoch_quota)
    for row_id in corrected:
        new_counts[row_id] = 0
        new_quotas[row_id] = epoch_quota
    for row_id in deferred:
        new_counts.setdefault(row_id, int(counts.get(row_id, 0)))
    result['exposure_counts'] = new_counts
    result['required_visits'] = new_quotas
    result['options'] = dict(options)
    new_history = list(history)
    record = {
        'proof_path': str(path), 'proof_sha256': proof_sha,
        'prior_source': prior_ref, 'current_source': current_ref,
        'stage_id': stage_id, 'corrected_rows': changes,
        'deferred_rows': sorted(deferred), 'added_rows': sorted(added),
        'old_target_exposure_retired': True,
        'optimizer_rng_weights': 'preserved',
        'at_global_step': int(training.get('global_step', training.get('step', 0))),
        'new_corrected_row_exposure_added': 0,
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
        for row_id in set(current_bindings) - set(old_bindings) - set(corrected):
            frame_counts.setdefault(row_id, 0)
            frame_quotas.setdefault(row_id, epoch_quota)
        for row_id in corrected:
            frame_counts[row_id] = 0
            frame_quotas[row_id] = epoch_quota
        frame['exposure_counts'] = frame_counts
        frame['required_visits'] = frame_quotas
        frame['pending_source_rows'] = dict(pending)
        if frame_completed:
            frame['completed_foundations'] = []
        frame_options = dict(frame.get('options') or {})
        frame_options['source_correction_history'] = list(new_history)
        frame_options['source_correction_binding'] = ref
        frame_options['foundation_qualification_history'] = list(qualification_history)
        frame['options'] = frame_options
        progress[stage_id] = frame
    result['sft_progress'] = progress
    return result, {'changed': True, 'proof_sha256': proof_sha,
                    'corrected_rows': len(corrected), 'deferred_rows': len(deferred),
                    'added_rows': len(added)}
