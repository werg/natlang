"""Two-phase, fail-closed paired retention; provider operations are explicit."""
from __future__ import annotations
import fcntl
from pathlib import Path
from .retention_snapshot import snapshot,refresh_references
from .retention_evidence import external_references
from .retention_eligibility import eligible_pair
from .retention_presence import current_presence
from .retention_transaction import retire_pair
from .retention_leases import leased_roles


def guarded_pair(*, candidate, roots, scan, reference_scope, read_live, api, lock_path, state_path, remove, present, write_durable, apply=False):
    """Scan outside the lock, then revalidate and refresh live facts under it.

    read_live(candidate, phase) returns current eligibility facts and authenticated
    receipts. It must include all owner parents, pending GET/publication roles and
    active service instances, including retries. The caller owns provider access.
    Dry output is advisory and does not authorize later apply.
    """
    before=snapshot(roots)
    refs=scan([candidate['sha256'],candidate['control_sha256']],roots)
    before,refs=refresh_references(roots,before,refs,[candidate['sha256'],candidate['control_sha256']],scan,allow_busy=True)
    for digest in (candidate['sha256'],candidate['control_sha256']):
        if external_references(digest,refs[digest],**reference_scope):
            raise ValueError('external dependency protects candidate')
    def inspect(value,phase):
        if phase=='prepare':
            _,updated=refresh_references(roots,before,refs,[value['sha256'],value['control_sha256']],scan,allow_busy=True)
            for digest in (value['sha256'],value['control_sha256']):
                if external_references(digest,updated[digest],**reference_scope):raise ValueError('new external dependency protects candidate')
        live=read_live(value,phase)
        eligible_pair(value,**live['eligibility'])
        receipts=live['receipts'];by_role={r.get('role'):r for r in receipts}
        if len(by_role)!=len(receipts):raise ValueError('duplicate receipt role')
        required={live['eligibility']['successor_role'],live['eligibility']['fallback_role']}
        if phase in ('prepare','dry'):required|={value['role'],value['control_role']}
        if not required<=set(by_role):raise ValueError('missing exact recovery receipt')
        for role in required:
            receipt=by_role[role];identity=receipt.get('identity') or {};verification=receipt.get('independent_verification') or {}
            if verification.get('sha256')!=identity.get('sha256') or verification.get('bytes')!=identity.get('bytes'):
                raise ValueError('receipt does not bind verified bytes')
        fallback=by_role[live['eligibility']['fallback_role']]['independent_verification']
        if fallback.get('readback_location')!='owner':raise ValueError('fallback lacks independent owner GET')
        for role,key in ((value['role'],'sha256'),(value['control_role'],'control_sha256')):
            if role in by_role and by_role[role]['identity']['sha256']!=value[key]:raise ValueError('pair receipt identity changed')
        current_presence(api,[by_role[role] for role in sorted(required)])
    if not apply:
        with Path(lock_path).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if {candidate['role'],candidate['control_role']} & leased_roles(Path(state_path).parent/'standalone-retention-consumer-leases.json'):raise ValueError('active consumer lease protects retirement pair')
            inspect(candidate,'prepare')
        return {'mode':'dry','role':candidate['role'],'control_role':candidate['control_role'],'deletion_authorized':False,'scope':'point-in-time JSON scan; managed consumer leases and owner state require the shared commit lock'}
    return retire_pair(lock_path=lock_path,state_path=state_path,candidate=candidate,inspect=inspect,remove=remove,present=present,write_durable=write_durable)


def guarded_full(*, candidate, roots, scan, reference_scope, read_live, api, lock_path, state_path, remove, present, write_durable, apply=False):
    """Guard a historical full-only journal entry without inventing a control."""
    from .retention_eligibility import eligible_full
    from .retention_transaction import retire_full
    if candidate.get('control_required') or candidate.get('control'):raise ValueError('control-bearing record needs paired retirement')
    before=snapshot(roots);refs=scan([candidate['sha256']],roots);before,refs=refresh_references(roots,before,refs,[candidate['sha256']],scan,allow_busy=True)
    blocking=external_references(candidate['sha256'],refs[candidate['sha256']],**reference_scope)
    if blocking:raise ValueError('external dependency protects candidate: '+repr(sorted(blocking)))
    def inspect(value,phase):
        if phase=='prepare':
            _,updated=refresh_references(roots,before,refs,[value['sha256']],scan,allow_busy=True)
            if external_references(value['sha256'],updated[value['sha256']],**reference_scope):raise ValueError('new external dependency protects candidate')
        live=read_live(value,phase);eligible_full(value,**live['eligibility'])
        receipts=live['receipts'];by_role={r.get('role'):r for r in receipts}
        if len(by_role)!=len(receipts):raise ValueError('duplicate receipt role')
        required={live['eligibility']['successor_role'],live['eligibility']['fallback_role']}
        if phase in ('prepare','dry'):required.add(value['role'])
        if not required<=set(by_role):raise ValueError('missing exact recovery receipt')
        for role in required:
            r=by_role[role];identity=r.get('identity') or {};verification=r.get('independent_verification') or {}
            if verification.get('sha256')!=identity.get('sha256') or verification.get('bytes')!=identity.get('bytes'):raise ValueError('receipt does not bind verified bytes')
        if by_role[live['eligibility']['fallback_role']]['independent_verification'].get('readback_location')!='owner':raise ValueError('fallback lacks independent owner GET')
        if value['role'] in by_role and by_role[value['role']]['identity']['sha256']!=value['sha256']:raise ValueError('full receipt identity changed')
        current_presence(api,[by_role[role] for role in sorted(required)])
    if not apply:
        with Path(lock_path).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if candidate['role'] in leased_roles(Path(state_path).parent/'standalone-retention-consumer-leases.json'):raise ValueError('active consumer lease protects checkpoint')
            inspect(candidate,'prepare')
        return {'mode':'dry','role':candidate['role'],'deletion_authorized':False,'scope':'point-in-time JSON scan; managed consumer leases and owner state require the shared commit lock'}
    return retire_full(lock_path=lock_path,state_path=state_path,candidate=candidate,inspect=inspect,remove=remove,present=present,write_durable=write_durable)
