"""Select individually bound full or paired candidates from a role journal."""
from __future__ import annotations
from .retention_live import live_facts
from .retention_eligibility import eligible_full,eligible_pair


def candidates(*, exports, intents, owners, current_owner, owner_state, receipts, fallback_role, active_units, transitions, protected_sha256s=()):
    """Return only candidates whose exact successor and current owner are bound.

    Provider presence, consumer leases and namespace references are checked by
    the caller through guarded_full/guarded_pair before any retirement.
    """
    by_role={r['role']:r for r in receipts}
    if fallback_role not in by_role:raise ValueError('fallback receipt missing')
    fallback_sha=by_role[fallback_role]['identity']['sha256'];out=[]
    for e in sorted(exports,key=lambda x:x['step']):
        role=e['role'];intent=intents[role]
        if role not in by_role:continue
        matching=[x for x in exports if x['step']>e['step'] and x['role'] in by_role and x.get('worker')==e.get('worker')]
        for successor in sorted(matching,key=lambda x:x['step'],reverse=True):
            v=intents[successor['role']];candidate={'role':role,'sha256':intent['sha256'],'full_unit':e['unit'],'successor_role':successor['role'],'successor_sha256':v['sha256'],'fallback_role':fallback_role,'fallback_sha256':fallback_sha,'control_required':bool(e.get('control_required'))}
            if candidate['control_required']:
                control=e.get('control') or {};archive=control.get('archive') or {}
                if not all((control.get('role'),control.get('unit'),archive.get('sha256'))):continue
                candidate.update(control_role=control['role'],control_unit=control['unit'],control_sha256=archive['sha256'])
            elif e.get('control'):continue
            try:
                live=live_facts(candidate=candidate,exports=exports,intents=intents,owners=owners,current_owner=current_owner,owner_state=owner_state,receipts=receipts,successor_role=successor['role'],fallback_role=fallback_role,active_units=active_units,transition=transitions.get(current_owner['parent_sha256']))
                live['eligibility']['protected_sha256s'].update(protected_sha256s)
                (eligible_pair if candidate['control_required'] else eligible_full)(candidate,**live['eligibility'])
            except (ValueError,KeyError):continue
            out.append(candidate);break
    return out
