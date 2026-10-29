"""Validate narrowly scoped self-bookkeeping before excluding it from references."""
from __future__ import annotations
import json
from pathlib import Path


def _contains(value,digest):return digest in json.dumps(value,sort_keys=True)


def own_bookkeeping(*, role, digest, receipt, receipt_path=None, phase=None, cache=None, reviews=(), retirement_state=None, watched_digests=()):
    """Return exact document paths and contents that are only own bookkeeping.

    Other entries and fields remain protective even in the same JSON document.
    Caller must re-read and compare returned contents under the commit lock.
    """
    out={};identity=receipt['identity'];watched=set(watched_digests)|{digest}
    if receipt.get('role')!=role or identity.get('sha256')!=digest:raise ValueError('receipt identity differs')
    if receipt_path is not None:
        p=Path(receipt_path);v=json.loads(p.read_text())
        if v!=receipt:raise ValueError('receipt bytes changed')
        extra={k:x for k,x in v.items() if k not in ('identity','independent_verification','upload','name','source')}
        if any(_contains(extra,d) for d in watched):raise ValueError('receipt contains another reference')
        out[str(p)]=v
    if phase is not None:
        p=Path(phase);v=json.loads(p.read_text())
        if any(v.get(k)!=receipt.get(k) for k in ('role','run','name','identity')):raise ValueError('upload phase identity differs')
        upload=v.get('upload') or {}
        if upload.get('sha256')!=digest or upload.get('bytes')!=identity.get('bytes'):raise ValueError('upload phase bytes differ')
        extra={k:x for k,x in v.items() if k not in ('role','run','name','identity','upload','at')}
        if any(_contains(extra,d) for d in watched):raise ValueError('upload phase contains another reference')
        out[str(p)]=v
    if cache is not None:
        p=Path(cache);v=json.loads(p.read_text())
        if v.get('schema')!='natlang.private-export-cache/1' or v.get('identity')!=identity or v.get('receipt')!=receipt:raise ValueError('export cache identity differs')
        extra={k:x for k,x in v.items() if k not in ('schema','identity','receipt')}
        if any(_contains(extra,d) for d in watched):raise ValueError('export cache contains another reference')
        out[str(p)]=v
    for path in reviews:
        p=Path(path);v=json.loads(p.read_text());items=v.get('candidates')
        if v.get('schema')!='natlang.standalone-retention-guard-review/1' or not isinstance(items,list):raise ValueError('unexpected retention review')
        rest=dict(v);rest['candidates']=[x for x in items if not ((x.get('role')==role and x.get('sha256')==digest) or (x.get('newer_role')==role and x.get('newer_sha256')==digest))]
        if _contains(rest,digest):raise ValueError('retention review contains another reference')
        out[str(p)]=v
    if retirement_state is not None:
        p=Path(retirement_state);v=json.loads(p.read_text());items=v.get('operations')
        if not isinstance(items,list):raise ValueError('invalid retention state')
        rest=dict(v);rest['operations']=[x for x in items if not ((x.get('phase')=='deleted' and x.get('newer_role')==role and x.get('newer_sha256')==digest) or (x.get('phase') in ('intent','control_deleted') and x.get('role')==role and x.get('sha256')==digest))]
        if _contains(rest,digest):raise ValueError('retirement state contains another reference')
        out[str(p)]=v
    return out


def recheck_own_bookkeeping(documents, *, role, digest):
    for path,expected in documents.items():
        value=json.loads(Path(path).read_text())
        if 'operations' in expected:
            # A fresh operation may be appended while locked; an operation
            # already captured in expected must remain byte-for-byte intact.
            existing={(x.get('role'),x.get('sha256')) for x in expected['operations']}
            value=dict(value);value['operations']=[x for x in value['operations'] if not (x.get('phase') in ('intent','control_deleted') and x.get('role')==role and x.get('sha256')==digest and (role,digest) not in existing)]
            expected_op=next((x for x in expected['operations'] if x.get('role')==role and x.get('sha256')==digest),None)
            if expected_op and expected_op.get('phase')=='intent':
                value['operations']=[dict(x,phase='intent') if x.get('role')==role and x.get('sha256')==digest and x.get('phase')=='control_deleted' else x for x in value['operations']]
        if value!=expected:raise ValueError('own bookkeeping changed before commit: '+path)


def own_exporter_metadata(paths, *, role, source, program, watched_digests):
    """Recognize exact local exporter invocation history, not other consumers."""
    out={}
    for path in paths:
        p=Path(path)
        if p.name!='wandb-metadata.json':continue
        value=json.loads(p.read_text())
        if value.get('args')!=[role,source] or value.get('program')!=str(program):continue
        extra={k:v for k,v in value.items() if k!='args'}
        if any(_contains(extra,d) for d in watched_digests):raise ValueError('exporter metadata contains another reference')
        out[str(p)]=value
    return out


def own_pin_maintenance(paths, *, role, digest):
    """Recognize an exact completed local-pin dry record, not a consumer."""
    out={}
    for path in paths:
        p=Path(path)
        if not p.name.startswith('local-pin-maintenance-'):continue
        value=json.loads(p.read_text())
        if value.get('schema')!='natlang.local-pin-maintenance-dry/1' or value.get('role')!=role or value.get('no_unlink_performed') is not True or (value.get('candidate') or {}).get('sha256')!=digest:continue
        rest=dict(value);rest.pop('candidate')
        if _contains(rest,digest):raise ValueError('local pin dry contains another reference')
        out[str(p)]=value
    return out


def own_completed_pin_intents(paths, *, role, digest):
    """Recognize only a completed exact local-unlink intent and its result."""
    out={}
    for path in paths:
        p=Path(path)
        if not p.name.startswith('local-pin-maintenance-') or not p.name.endswith('-intent.json'):continue
        intent=json.loads(p.read_text());candidate=intent.get('candidate') or {}
        if intent.get('schema')!='natlang.local-pin-exclusive-maintenance/1' or intent.get('role')!=role or intent.get('approved_sha256')!=digest or candidate.get('sha256')!=digest:continue
        result_path=p.with_name(p.name.removesuffix('-intent.json')+'-result.json')
        if not result_path.exists():continue
        result=json.loads(result_path.read_text());remote=result.get('remote') or {}
        if result.get('schema')!='natlang.local-pin-exclusive-maintenance-result/1' or result.get('intent')!=str(p) or result.get('status')!='unlinked' or remote.get('unlinked') is not True or remote.get('path')!=candidate.get('path') or remote.get('inode')!=candidate.get('inode'):continue
        remainder=dict(intent);remainder.pop('candidate');remainder.pop('approved_sha256')
        if _contains(remainder,digest) or _contains(result,digest):raise ValueError('completed local pin maintenance contains another reference')
        out[str(p)]=intent;out[str(result_path)]=result
    return out
