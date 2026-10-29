"""Authenticated, one-time promotion of already packed pending SFT groups."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

SCHEMA='natlang.native-pending-focus/1'


def dependencies(reference):
    """Verify the immutable request and its source/actor/oracle evidence."""
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('focus requires an exact request binding')
    path=Path(reference['path']).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:raise ValueError('focus request SHA differs')
    doc=json.loads(path.read_text())
    if doc.get('schema')!=SCHEMA or not isinstance(doc.get('rows'),list) or not doc['rows']:raise ValueError('invalid pending focus request')
    result=[{'path':str(path),'sha256':reference['sha256']}];seen=set();verified={}
    for item in doc['rows']:
        if not isinstance(item,dict) or set(item)!={'id','binding','source_sha256','split','evidence'} or item['split']!='train':raise ValueError('focus row must be a bound train row')
        if not isinstance(item['id'],str) or not item['id'] or item['id'] in seen:raise ValueError('duplicate or empty focus row')
        seen.add(item['id'])
        if any(not isinstance(item[k],str) or len(item[k])!=64 or any(c not in '0123456789abcdef' for c in item[k]) for k in ('binding','source_sha256')):raise ValueError('invalid focus semantic or source binding')
        evidence=item['evidence']
        if not isinstance(evidence,list) or not evidence:raise ValueError('focus row lacks provenance evidence')
        for ref in evidence:
            if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('invalid focus evidence binding')
            p=Path(ref['path']).resolve(strict=True)
            if p not in verified:
                with p.open('rb') as stream:verified[p]=hashlib.file_digest(stream,'sha256').hexdigest()
            actual=verified[p]
            if actual!=ref['sha256']:raise ValueError('focus provenance SHA differs')
            value={'path':str(p),'sha256':actual}
            if value not in result:result.append(value)
    return doc,result


def resolve(doc,entries,bindings,cohort_ids,withdrawal_history=()):
    """Reject unbound requests; a previously proven withdrawal retires its row."""
    if len(entries)!=len(cohort_ids):raise ValueError('focus cohort inventory differs')
    positions={entry['id']:i for i,entry in enumerate(entries)}
    if len(positions)!=len(entries):raise ValueError('focus source repeats row IDs')
    withdrawn={(row['id'],row['binding']) for item in withdrawal_history for row in item.get('rows',[])}
    rewritten_sources={row.get('source_sha256') for item in withdrawal_history for row in item.get('rows',[])}
    active=set();retired=[]
    for row in doc['rows']:
        ident=row['id'];index=positions.get(ident)
        if index is None:
            if (ident,row['binding']) not in withdrawn:raise ValueError('focus row absent without a matching withdrawal')
            retired.append(ident);continue
        if bindings.get(ident)!=row['binding']:raise ValueError('focus row semantic binding differs')
        if cohort_ids[index]!=row['source_sha256'] and row['source_sha256'] not in rewritten_sources:raise ValueError('focus row source differs without an authenticated withdrawal')
        active.add(ident)
    return active,retired


def promote(batches,entries,focus_ids):
    """Stable-partition existing complete groups; retain their exact credits."""
    focused=[];ordinary=[];seen=set()
    for batch in batches:
        present={entries[i]['id'] for i in batch[1]} & focus_ids
        if present-seen:focused.append(batch);seen.update(present)
        else:ordinary.append(batch)
    combined=focused+ordinary
    ends={visit:i for i,(visit,_,_) in enumerate(combined)}
    return [(visit,indexes,ends[visit]==i) for i,(visit,indexes,_) in enumerate(combined)]
def active_for_operation(*, policy_operation, explicit_request):
    """SFT focus remains pending across a distinct policy-objective source."""
    if policy_operation and explicit_request is not None:
        raise ValueError('pending SFT focus cannot target a policy-objective source')
    return not policy_operation
