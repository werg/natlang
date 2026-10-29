"""Build retention facts from actual journal, owner and receipt records."""
from __future__ import annotations
from .retention_evidence import owner_for_intent


def live_facts(*, candidate, exports, intents, owners, current_owner, owner_state, receipts, successor_role, fallback_role, active_units, transition=None):
    """Validate the selected journal entry and derive current protection facts.

    Inputs must be freshly read while holding the retention lock; all owner
    records must have been loaded with authenticated_owner_epochs.
    """
    matches=[e for e in exports if e.get('role')==candidate['role']]
    if len(matches)!=1:raise ValueError('candidate journal entry is not unique')
    entry=matches[0];intent=intents[candidate['role']]
    if intent.get('sha256')!=candidate['sha256'] or intent.get('step')!=entry.get('step') or intent.get('worker')!=entry.get('worker'):
        raise ValueError('candidate intent differs from journal')
    candidate_owner=owner_for_intent(intent,owners)
    successors=[e for e in exports if e.get('role')==successor_role]
    if len(successors)!=1 or successor_role not in intents:raise ValueError('successor journal entry is not unique')
    successor=successors[0];successor_intent=intents[successor_role]
    if candidate.get('successor_role')!=successor_role or candidate.get('successor_sha256')!=successor_intent.get('sha256') or candidate.get('fallback_role')!=fallback_role:raise ValueError('durable successor or fallback binding differs')
    if successor.get('worker')!=entry.get('worker') or successor_intent.get('worker')!=intent.get('worker') or successor_intent.get('step')!=successor.get('step') or successor['step']<=entry['step']:
        raise ValueError('successor worker or step differs')
    successor_owner=owner_for_intent(successor_intent,owners)
    if successor_owner!=candidate_owner:raise ValueError('successor is not from the same authenticated owner epoch')
    if current_owner!=candidate_owner:
        t=transition or {};accepted=candidate_owner['accepted']
        if not (t.get('full_state_verified') is True and t.get('immutable_pin_verified') is True and t.get('optimizer_verified') is True and t.get('rng_verified') is True and t.get('sha256')==current_owner.get('parent_sha256') and t.get('step')==current_owner.get('parent_step') and t.get('step',-1)>=successor['step'] and all(t.get(k)==candidate_owner.get(k) for k in ('sandbox_id','plan_sha256','runtime_commit')) and t.get('boot_id')==accepted.get('boot_id') and t.get('pid')==accepted.get('pid') and str(t.get('start_ticks'))==str(accepted.get('process_start_ticks'))):
            raise ValueError('current owner lacks authenticated continuation from successor epoch')
    if entry.get('unit')!=candidate['full_unit']:raise ValueError('candidate full unit differs')
    if bool(entry.get('control_required')) != bool(candidate.get('control_role')):raise ValueError('candidate control requirement differs')
    if entry.get('control_required'):
        control=entry.get('control') or {};archive=control.get('archive') or {}
        if control.get('role')!=candidate['control_role'] or archive.get('sha256')!=candidate.get('control_sha256') or control.get('unit')!=candidate.get('control_unit'):
            raise ValueError('candidate control identity differs')
        if control.get('status')!='independently_verified':raise ValueError('control not independently verified')
    elif entry.get('control'):raise ValueError('unexpected control on full-only entry')
    if not exports:raise ValueError('empty source journal')
    latest=max(exports,key=lambda e:e['step'])['role']
    verified=set(owner_state.get('verified_roles') or []);owner_exports=[e for e in exports if e.get('role') in verified]
    latest_owner=max(owner_exports,key=lambda e:e['step'])['role'] if owner_exports else None
    pending=owner_state.get('pending') or {};pending_roles=[pending['role']] if pending.get('role') else []
    for e in exports:
        if e.get('full_launch_pending') or (e.get('control_required') and (e.get('control') or {}).get('status') not in ('independently_verified','retired')):
            pending_roles.append(e['role']);pending_roles.extend([((e.get('control') or {}).get('role'))] if (e.get('control') or {}).get('role') else [])
    protected={o['parent_sha256'] for o in owners}|{current_owner['parent_sha256']}
    protected.update(v['historical_parent_sha256'] for v in intents.values() if v.get('historical_parent_sha256'))
    by_role={r['role']:r for r in receipts}
    if len(by_role)!=len(receipts):raise ValueError('duplicate receipts')
    for role in (candidate['role'],successor_role,fallback_role):
        if role not in by_role:raise ValueError('required receipt absent')
    if candidate.get('control_role') and candidate['control_role'] not in by_role:raise ValueError('control receipt absent')
    for role,v in intents.items():
        if role in by_role and by_role[role].get('identity')!={'sha256':v['sha256'],'bytes':v['bytes']}:raise ValueError('full receipt identity mismatch')
    if current_owner not in owners:raise ValueError('current owner is not authenticated')
    verified_roles=[]
    for role,r in by_role.items():
        proof=r.get('independent_verification') or {};identity=r.get('identity') or {}
        if r.get('private') and proof.get('sha256')==identity.get('sha256') and proof.get('bytes')==identity.get('bytes'):verified_roles.append(role)
    return {'eligibility':{'latest_role':latest,'latest_owner_role':latest_owner,'pending_roles':pending_roles,'protected_sha256s':protected,'active_units':active_units,'verified_roles':verified_roles,'successor_role':successor_role,'fallback_role':fallback_role},'receipts':receipts}
