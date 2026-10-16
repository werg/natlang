"""Safe pinned archive documents and complete observed Cooper execution views."""
import hashlib
import json
from pathlib import PurePosixPath
import tarfile

from .common import Reject


def iter_archive_rows(path):
    """Stable original tar ordinals; no member is extracted or executed."""
    with tarfile.open(path,'r:gz') as archive:
        seen=set()
        for ordinal,member in enumerate(archive):
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.name in seen:
                raise Reject('unsafe or duplicate original archive member')
            seen.add(member.name)
            if member.isdir():continue
            if not member.isfile():raise Reject('non-regular original archive member')
            data=archive.extractfile(member).read()
            binding={'path':member.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            if member.name.endswith('.json'):
                try:value=json.loads(data);kind='json'
                except (ValueError,UnicodeDecodeError):
                    value=None;kind='invalid-json-source-artifact'
            else:
                try:value=data.decode('utf-8');kind='text'
                except UnicodeDecodeError:value=None;kind='binary-source-artifact'
            yield ordinal,{'schema':'natlang.acquired-archive-document/1','member':binding,'kind':kind,'value':value}


def _archive_sources(limit=None):
    from .acquired_agents import acquired_rows
    current=None;documents={};receipt=None;seen=0
    def convert(receipt,documents):
        for member,(rid,doc) in documents.items():
            name=PurePosixPath(member);row=doc['value']
            if not (doc['kind']=='json' and (name.name.startswith('agent') or name.name.startswith('solo'))
                    and member.endswith('_traj.json') and isinstance(row,dict)):
                reason='malformed original archive JSON; exact member retained in pinned source' if doc['kind']=='invalid-json-source-artifact' else 'non-session archive member; original complete trajectory is the consumer'
                yield receipt,rid,{'_archive_omission':reason};continue
            full=member.removesuffix('_traj.json')+'_full_traj.json'
            if not member.endswith('_full_traj.json') and full in documents:
                yield receipt,rid,{};continue
            neighbors={};bindings={}
            for n in ('eval.json','result.json','metadata.json','conversation.json'):
                found=documents.get(str(name.parent/n))
                if found is not None:neighbors[n]=found[1]['value'];bindings[n]=found[1]['member']
            value={**row,'_cooperation_member_path':member,'_cooperation_archive_selected':True,
                   '_cooperation_neighbors':neighbors,'_cooperation_neighbor_bindings':bindings}
            provenance={'member':doc['member'],'view':'original-complete-context'}
            messages=row.get('messages') or []
            # Codex releases can retain only the final response in *_traj.json.
            # Reconstruct its observed input from the exact raw stream suffix,
            # without guessing internal user/tool/assistant delimiters.
            if len(messages)==1 and messages[0].get('role')=='assistant':
                actor=row.get('agent_id') or name.name.removesuffix('_traj.json')
                stream=documents.get(str(name.parent/(str(actor)+'_stream.log')))
                target=messages[0].get('content')
                if stream is None or not isinstance(target,str) or not target:
                    yield receipt,rid,{'_archive_omission':'minimal source response lacks complete original stream'};continue
                text=stream[1]['value'];body=target.rstrip('\n');end=text.rstrip('\n')
                if not body or not end.endswith(body):
                    yield receipt,rid,{'_archive_omission':'source response does not bind exact original stream suffix'};continue
                prefix=end[:-len(body)]
                if not prefix or not prefix.endswith('\n'):
                    yield receipt,rid,{'_archive_omission':'source response lacks an exact full-stream boundary'};continue
                value['messages']=[{'role':'user','content':f'Continue the original coding execution for {row.get("repo") or name.parent.parts[-3]}:{row.get("task_id") or name.parent.parts[-2]} after the complete observed transcript.'},
                                   {'role':'tool','content':prefix},messages[0]]
                provenance.update(view='complete-observed-stream-to-exact-final-response',stream=stream[1]['member'],
                                  source_target_sha256=hashlib.sha256(target.encode()).hexdigest(),
                                  observed_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                                  original_context_roles_reconstructed=False)
            value['_cooperation_archive_context']=provenance
            yield receipt,rid,value
    for r,rid,document in acquired_rows('D12'):
        source=rid.rsplit(':',1)[0]
        if source!=current and documents:
            for item in convert(receipt,documents):
                if limit is not None and seen>=limit:return
                seen+=1;yield item
            documents={}
        current=source;receipt=r
        if document.get('schema')!='natlang.acquired-archive-document/1':
            yield r,rid,{};continue
        documents[document['member']['path']]=(rid,document)
    if documents:
        for item in convert(receipt,documents):
            if limit is not None and seen>=limit:return
            seen+=1;yield item


def cooper_archive_acquired(limit=None):
    from .acquired_agents import _cooperation_acquired
    yield from _cooperation_acquired('D12',rows=_archive_sources(limit))
