"""Validate narrowly scoped self-bookkeeping before excluding it from references."""
from __future__ import annotations
import json
from pathlib import Path


def _contains(value,digest):return digest in json.dumps(value,sort_keys=True)


def own_bookkeeping(*, role, digest, receipt, phase=None, cache=None, reviews=(), retirement_state=None):
    """Return exact document paths and contents that are only own bookkeeping.

    Other entries and fields remain protective even in the same JSON document.
    Caller must re-read and compare returned contents under the commit lock.
    """
    out={};identity=receipt['identity']
    if receipt.get('role')!=role or identity.get('sha256')!=digest:raise ValueError('receipt identity differs')
    if phase is not None:
        p=Path(phase);v=json.loads(p.read_text())
        if any(v.get(k)!=receipt.get(k) for k in ('role','run','name','identity')):raise ValueError('upload phase identity differs')
        upload=v.get('upload') or {}
        if upload.get('sha256')!=digest or upload.get('bytes')!=identity.get('bytes'):raise ValueError('upload phase bytes differ')
        extra={k:x for k,x in v.items() if k not in ('role','run','name','identity','upload','at')}
        if _contains(extra,digest):raise ValueError('upload phase contains another reference')
        out[str(p)]=v
    if cache is not None:
        p=Path(cache);v=json.loads(p.read_text())
        if v.get('schema')!='natlang.private-export-cache/1' or v.get('identity')!=identity or v.get('receipt')!=receipt:raise ValueError('export cache identity differs')
        extra={k:x for k,x in v.items() if k not in ('schema','identity','receipt')}
        if _contains(extra,digest):raise ValueError('export cache contains another reference')
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
        rest=dict(v);rest['operations']=[x for x in items if not (x.get('phase')=='deleted' and x.get('newer_role')==role and x.get('newer_sha256')==digest)]
        if _contains(rest,digest):raise ValueError('retirement state contains another reference')
        out[str(p)]=v
    return out


def recheck_own_bookkeeping(documents, *, role, digest):
    for path,expected in documents.items():
        value=json.loads(Path(path).read_text())
        if 'operations' in expected:
            value=dict(value);value['operations']=[x for x in value['operations'] if not (x.get('phase')=='intent' and x.get('role')==role and x.get('sha256')==digest)]
        if value!=expected:raise ValueError('own bookkeeping changed before commit: '+path)
