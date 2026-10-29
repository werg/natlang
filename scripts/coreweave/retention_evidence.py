"""Scope immutable checkpoint/control bookkeeping without hiding other consumers."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path


def _contains(value, digest):
    return digest in json.dumps(value, sort_keys=True, ensure_ascii=False)


def _load(path):
    path=Path(path);before=path.stat();value=json.loads(path.read_text());after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('retention reference changed during inspection')
    return value


def external_references(digest, references, *, allowed_paths, journal_path, role, intent_path, control=None, control_receipt=None, control_phase=None):
    """Ignore only this exact entry and its authenticated control bookkeeping.

    Other journal entries, fields outside the matched entry, consumer files and
    unexpected digest-bearing receipt fields remain protective references.
    """
    allowed={str(Path(x)) for x in allowed_paths};journal_path=str(Path(journal_path));intent_path=str(Path(intent_path));result=set()
    for path in map(str,references):
        if path in allowed:continue
        if path==journal_path:
            value=_load(path);exports=value.get('exports')
            if not isinstance(exports,list):raise ValueError('retention journal has no export list')
            matches=[i for i,e in enumerate(exports) if e.get('role')==role and e.get('intent')==intent_path]
            if len(matches)!=1:raise ValueError('retention journal candidate is not unique')
            index=matches[0];copy=dict(value);copy['exports']=exports[:index]+exports[index+1:]
            if _contains(copy,digest):result.add(path)
            continue
        if control and control_receipt and path==str(Path(control_receipt)):
            value=_load(path);archive=control.get('archive') or {};identity=value.get('identity') or {}
            if value.get('role')!=control.get('role') or value.get('source')!=archive.get('path') or identity.get('sha256')!=archive.get('sha256') or identity.get('bytes')!=archive.get('bytes'):
                raise ValueError('control receipt does not match the exact candidate journal entry')
            basename=Path(archive['path']).name;expected_name=control['role']+'/'+archive['sha256']+'/'+basename
            if value.get('name')!=expected_name:raise ValueError('control receipt object name differs')
            copy=dict(value);copy.pop('source');copy.pop('name')
            if _contains(copy,digest):result.add(path)
            continue
        if control and control_phase and path==str(Path(control_phase)):
            value=_load(path);archive=control.get('archive') or {}
            if value.get('role')!=control.get('role') or value.get('source')!=archive.get('path'):
                raise ValueError('control phase does not match the exact candidate journal entry')
            copy=dict(value);copy.pop('source')
            if _contains(copy,digest):result.add(path)
            continue
        result.add(path)
    return result


def control_retirement_blocked(state, role):
    """A durable intent is enough to prevent retry/republication after a crash."""
    if role in set(state.get('retired_roles') or []):return True
    return any(op.get('control_role')==role and op.get('phase') in ('intent','control_deleted','deleted') for op in state.get('operations') or [])


def authenticated_owner_epochs(bindings):
    """Load explicitly named and digest-bound owner epochs, never infer ownership."""
    owners=[]
    for binding in bindings:
        path=Path(binding['path']);before=path.stat();raw=path.read_bytes();after=path.stat()
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if signature(before)!=signature(after):raise ValueError('owner epoch changed during inspection')
        if hashlib.sha256(raw).hexdigest()!=binding['sha256']:
            raise ValueError('owner epoch digest changed')
        value=json.loads(raw)
        accepted=value.get('accepted') or {}
        if accepted.get('accepted') is not True or not all(value.get(k) for k in ('sandbox_id','plan_sha256','runtime_commit','parent_sha256')) or type(value.get('parent_step')) is not int:
            raise ValueError('owner epoch is incomplete')
        owners.append(value)
    if not owners:raise ValueError('owner epoch list is empty')
    return owners


def owner_for_intent(intent, owners):
    keys=('owner_id','generation','boot_id','pid','process_start_ticks')
    selected=intent.get('owner') or {}
    if any(selected.get(k) is None for k in keys):raise ValueError('intent lacks exact owner process binding')
    matches=[o for o in owners if all(intent.get(k)==o.get(k) for k in ('sandbox_id','plan_sha256','runtime_commit')) and all(selected[k]==o['accepted'].get(k) for k in keys) and intent.get('step',-1)>o['parent_step']]
    if len(matches)!=1:raise ValueError('intent does not identify exactly one authenticated owner epoch')
    if intent.get('outer_credit') is not False:raise ValueError('unexpected outer credit')
    return matches[0]
