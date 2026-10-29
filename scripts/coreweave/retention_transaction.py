"""Crash-resumable paired checkpoint/control retirement under one durable fence."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from .retention_leases import leased_roles


def retire_pair(*, lock_path, state_path, candidate, inspect, remove, present, write_durable):
    """Retire one fully inspected pair; callbacks must use authenticated identities.

    inspect(candidate, phase) must recheck current owner, pending GETs, live dependencies,
    replacement coverage and fresh presence, and raise on any uncertainty. The
    same lock is shared with publication retries. No network or path assumption is
    hidden here: the caller supplies its real provider operations.
    """
    required=('role','sha256','control_role','control_sha256','successor_role','successor_sha256','fallback_role','fallback_sha256')
    if any(not candidate.get(k) for k in required):raise ValueError('incomplete paired candidate')
    if candidate['role']==candidate['control_role']:raise ValueError('full and control roles must differ')
    lock_path=Path(lock_path);state_path=Path(state_path)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if {candidate['role'],candidate['control_role']} & leased_roles(state_path.parent/'standalone-retention-consumer-leases.json'):
            raise ValueError('active consumer lease protects retirement pair')
        state=json.loads(state_path.read_text()) if state_path.exists() else {'retired_roles':[],'operations':[]}
        prior=[x for x in state['operations'] if x.get('role')==candidate['role']]
        if len(prior)>1:raise ValueError('duplicate retirement operation')
        if prior:
            op=prior[0]
            if any(op.get(k)!=candidate[k] for k in required):raise ValueError('retirement identity changed')
            if op.get('phase')=='deleted':return op
        else:
            inspect(candidate,'prepare')
            op={**candidate,'phase':'intent'};state['operations'].append(op);write_durable(state_path,state)
        if op['phase']=='intent':
            inspect(candidate,'intent')
            if present(candidate['control_role'],candidate['control_sha256']):remove(candidate['control_role'],candidate['control_sha256'])
            if present(candidate['control_role'],candidate['control_sha256']):raise RuntimeError('control remains present')
            op['phase']='control_deleted';write_durable(state_path,state)
        if op['phase']=='control_deleted':
            inspect(candidate,'control_deleted')
            if present(candidate['role'],candidate['sha256']):remove(candidate['role'],candidate['sha256'])
            if present(candidate['role'],candidate['sha256']):raise RuntimeError('checkpoint remains present')
            op['phase']='deleted';op['absence_scope']='fresh provider listing absent or zero-byte listing confirmed HTTP 404';state['retired_roles']=list(dict.fromkeys([*state['retired_roles'],candidate['role'],candidate['control_role']]));write_durable(state_path,state)
        if op['phase']!='deleted':raise ValueError('unknown retirement phase')
        return op


def retire_full(*, lock_path, state_path, candidate, inspect, remove, present, write_durable):
    """Retire an authenticated historical full-only record with durable intent."""
    if candidate.get('control_required') or candidate.get('control'):raise ValueError('control-bearing record needs paired retirement')
    if not candidate.get('role') or not candidate.get('sha256'):raise ValueError('incomplete full candidate')
    with Path(lock_path).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state_path=Path(state_path)
        if candidate['role'] in leased_roles(state_path.parent/'standalone-retention-consumer-leases.json'):raise ValueError('active consumer lease protects checkpoint')
        state=json.loads(state_path.read_text()) if state_path.exists() else {'retired_roles':[],'operations':[]}
        previous=[x for x in state['operations'] if x.get('role')==candidate['role']]
        if len(previous)>1:raise ValueError('duplicate retirement operation')
        if previous:
            op=previous[0]
            if any(op.get(k)!=candidate.get(k) for k in ('sha256','successor_role','successor_sha256','fallback_role','fallback_sha256')) or op.get('control_role'):raise ValueError('retirement identity changed')
            if op.get('phase')=='deleted':return op
        else:
            inspect(candidate,'prepare');op={**candidate,'phase':'intent'};state['operations'].append(op);write_durable(state_path,state)
        if op.get('phase')!='intent':raise ValueError('unknown retirement phase')
        inspect(candidate,'intent')
        if present(candidate['role'],candidate['sha256']):remove(candidate['role'],candidate['sha256'])
        if present(candidate['role'],candidate['sha256']):raise RuntimeError('checkpoint remains present')
        op['phase']='deleted';op['absence_scope']='fresh provider listing absent or zero-byte listing confirmed HTTP 404';state['retired_roles']=list(dict.fromkeys([*state['retired_roles'],candidate['role']]));write_durable(state_path,state)
        return op
