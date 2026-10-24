"""Continue recovered CE sources without forgiving temporarily missing data.

Policy is an explicit recipe-bound availability declaration. This mechanism
checks it against the actual earned checkpoint's source tree, moves only those
leaf rows to the existing pending ledger, and preserves every earned visit.
No absent payload is parsed, admitted anew, or represented by synthetic data.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.copying import deepcopy_metadata
from ..common.hashing import sha256_file_hex_cached
from .native_continuation import collect_sft_progress, required_visits, stage_key

SCHEMA = 'natlang.native-source-availability/1'


def defer_unavailable_sources(training, proof_path, *, current_bindings, model_id,
                              current_source_sha256s):
    path = Path(proof_path)
    proof = json.loads(path.read_text())
    if (proof.get('schema') != SCHEMA or proof.get('model_id') != model_id
            or proof.get('stage_id') != stage_key(training)
            or proof.get('preserve_obligations') is not True
            or not proof.get('reason')):
        raise ValueError('native source availability requires exact model/stage and retained obligations')
    if (training.get('identity') or {}).get('objective') is not None:
        raise ValueError('native source availability is an ordinary CE continuation only')
    sources = proof.get('unavailable_sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('native source availability requires enumerated unavailable leaves')
    selected = {}
    for source in sources:
        key = (source.get('path'), source.get('sha256'))
        if (not all(isinstance(value, str) and value for value in key)
                or key in selected or type(source.get('rows')) is not int or source['rows'] < 1
                or source['sha256'] in current_source_sha256s):
            raise ValueError('unavailable source is duplicated, active, or lacks its exact leaf binding')
        selected[key] = source
    proof_sha = sha256_file_hex_cached(path)
    history = (training.get('options') or {}).get('source_availability_history') or []
    existing = next((item for item in history if item['proof_sha256'] == proof_sha), None)
    if existing is not None:
        # Normal continuation reads the same immutable declaration again after
        # its first publication. The original source root can now be absent.
        pending = (collect_sft_progress(training, borrow_frames=True)
                   .get(stage_key(training), {}).get('pending_source_rows') or {})
        for key, row in existing['deferred_rows'].items():
            if key in current_bindings or pending.get(key) != row:
                raise ValueError('source availability declaration no longer matches pending rows')
        return training
    receipts = training.get('source_receipts') or []
    reference = proof.get('prior_train_source') or {}
    if not receipts or any(reference.get(key) != receipts[0].get(key) for key in ('path', 'sha256')):
        raise ValueError('source availability predecessor differs from the earned checkpoint')
    order = training.get('source_row_order') or []
    bindings = training.get('source_row_bindings') or {}
    if len(order) != len(bindings) or set(order) != set(bindings):
        raise ValueError('source availability needs the complete original row order and bindings')
    offset = 0
    removed = []
    witnessed = set()
    def visit(receipt):
        nonlocal offset
        if receipt.get('sources'):
            for child in receipt['sources']:
                visit(child)
            return
        rows = receipt.get('rows')
        if type(rows) is not int or rows < 1:
            raise ValueError('source availability checkpoint leaf lacks an exact row count')
        ids = order[offset:offset + rows]
        if len(ids) != rows:
            raise ValueError('source availability receipt exceeds saved row order')
        offset += rows
        key = (receipt.get('path'), receipt.get('sha256'))
        if key in selected:
            if key in witnessed or selected[key]['rows'] != rows:
                raise ValueError('source availability leaf row count differs or repeats')
            witnessed.add(key)
            removed.extend(ids)
    visit(receipts[0])
    if offset != len(order) or witnessed != set(selected):
        raise ValueError('source availability leaves do not match the complete saved source tree')
    removed = set(removed)
    if removed.intersection(current_bindings):
        raise ValueError('unavailable source rows are still active')
    kept = {key: value for key, value in bindings.items() if key not in removed}
    if any(current_bindings.get(key) != value for key, value in kept.items()):
        raise ValueError('source availability lost or changed a recovered original row')
    result = dict(training)  # Actual optimizer/RNG/FP8 objects stay untouched.
    result['options'] = deepcopy_metadata(training.get('options') or {})
    result['source_row_bindings'] = kept
    result['source_row_order'] = [key for key in order if key not in removed]
    result['exposure_counts'] = dict(training.get('exposure_counts') or {})
    quotas = dict(training.get('required_visits') or required_visits(
        order, float(result['options'].get('epochs', 1))))
    progress = collect_sft_progress(training)
    frame = progress[stage_key(training)]
    quotas = {**quotas, **(frame.get('required_visits') or {})}
    pending = deepcopy_metadata(frame.get('pending_source_rows') or {})
    deferred = {key: {'binding': bindings[key], 'required_visits': quotas[key]}
                for key in removed}
    if any(key in pending and pending[key] != row for key, row in deferred.items()):
        raise ValueError('source availability conflicts with an existing pending obligation')
    pending.update(deferred)
    record = {'proof_sha256': proof_sha, 'proof_path': str(path),
              'prior_train_source': reference, 'unavailable_sources': sources,
              'deferred_rows': deferred, 'at_global_step': training.get('global_step', training.get('step', 0)),
              'earned_visits_added': 0, 'earned_visits_discarded': 0,
              'weights_optimizer_rng': 'preserved'}
    result['options']['source_availability_history'] = [*history, record]
    result['required_visits'] = quotas
    result['pending_source_rows'] = pending
    # The normal save/collect path merges pending obligations from its carried
    # frame, including counts for rows absent from the current active source.
    frame.update(source_row_bindings=kept, source_row_order=result['source_row_order'],
                 required_visits=quotas, pending_source_rows=pending,
                 options=deepcopy_metadata(result['options']))
    frame.setdefault('pending_source_declarations', {})[proof_sha] = {
        'binding': {'path': str(path), 'sha256': proof_sha},
        'reason': proof['reason'], 'unavailable_sources': sources}
    result['sft_progress'] = progress
    return result
