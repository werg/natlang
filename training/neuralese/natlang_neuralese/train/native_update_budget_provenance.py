"""Authenticate a captured SFT reference when an old source was retired."""
from __future__ import annotations

from ..common.hashing import sha256_file_hex_cached


def verify_captured_source(reference, training, *, train, args, model_id):
    from .native_resident import bound
    source=reference.get('reference_train_source') or {}
    if source.get('sha256')!=reference.get('reference_train_sha256'):
        raise ValueError('captured SFT update reference source differs')
    try:
        bound(source)
        return
    except FileNotFoundError:
        pass
    # An actual unavailable-source continuation proves the historical root and
    # every absent leaf against the earned frame; its normal application later
    # still performs the same verification before any optimizer update.
    if args.unavailable_native_sources:
        from .native_availability import defer_unavailable_sources
        receipts=training.get('source_receipts') or []
        matches_current=bool(receipts and all(receipts[0].get(k)==source.get(k) for k in ('path','sha256')))
        digest=sha256_file_hex_cached(args.unavailable_native_sources)
        history=(training.get('options') or {}).get('source_availability_history') or []
        matches_history=any(item.get('proof_sha256')==digest and all(
            (item.get('prior_train_source') or {}).get(k)==source.get(k) for k in ('path','sha256'))
            for item in history)
        if matches_current or matches_history:
            leaves=set()
            def visit(item):
                if item.kind=='manifest':
                    for child in item.children:visit(child)
                else:leaves.add(item.file_sha256)
            visit(train)
            defer_unavailable_sources(training,args.unavailable_native_sources,
                current_bindings=train.bindings,model_id=model_id,current_source_sha256s=leaves)
            return
    # A consumed withdrawal can deliberately retire its original leaf. Verify
    # the committed proof and its evidence, and require this very source SHA in
    # the recorded removed rows. A missing unrelated source remains fatal.
    for path in (args.withdrawn_native_graphs,args.retired_native_withdrawal):
        if not path:continue
        digest=sha256_file_hex_cached(path)
        history=(training.get('options') or {}).get('withdrawal_history') or []
        record=next((item for item in history if item.get('proof_sha256')==digest),None)
        if record and any(row.get('source_sha256')==source['sha256'] for row in record.get('rows',())):
            from .native_continuation import verify_retired_withdrawal
            verify_retired_withdrawal(training,path,current_bindings=train.bindings,model_id=model_id)
            return
    raise FileNotFoundError('captured SFT reference source is missing without authenticated availability or withdrawal: '+str(source.get('path')))
