"""Durable, identity-bound closed chunks for verified source projections.

A caller supplies its exact source/actor/code/protection identity and a verifier
which returns a graph only after checking a source row.  A marker is published
only after the complete chunk is verified and its graph bytes are durable.
"""
from __future__ import annotations
import fcntl,hashlib,json,os
from pathlib import Path
from natlang_neuralese.common.jsonio import canonical_json_str,write_json_durable


def digest(value):
    return hashlib.sha256(canonical_json_str(value).encode()).hexdigest()


def _file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def verify_chunks(rows, *, out, identity, verify, validate_identity, chunk_rows=128):
    """Verify and publish contiguous chunks; resume only the exact proven prefix.

    ``validate_identity`` must check current source, actor, verifier code and
    protection against ``identity``.  Failed verification never advances the
    journal. Each chunk marker lists the exact input row hashes and graph hashes.
    Existing markers are validated against both disk bytes and replayed inputs.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'journal.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _verify_chunks(rows,out=out,identity=identity,verify=verify,
                              validate_identity=validate_identity,chunk_rows=chunk_rows)


def _verify_chunks(rows, *, out, identity, verify, validate_identity, chunk_rows):
    if not isinstance(identity,dict) or not identity or chunk_rows < 1:
        raise ValueError('exact nonempty identity and positive chunk size required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);ident=digest(identity)
    validate_identity(identity)
    markers=sorted(out.glob('chunk-*.receipt.json'))
    input_count=0;ids=set();existing=[]
    for number,path in enumerate(markers):
        if path.name!=f'chunk-{number:08d}.receipt.json':raise ValueError('chunk sequence has a gap')
        item=json.loads(path.read_text()); artifact=out/f'chunk-{number:08d}.jsonl'
        if (item.get('schema')!='natlang.verified-source-chunk/1' or item.get('identity_sha256')!=ident
                or item.get('identity')!=identity or item.get('start')!=input_count or item.get('rows')!=len(item.get('input',[]))
                or not item.get('rows') or item.get('graph_path')!=str(artifact.resolve())
                or item.get('graph_sha256')!=_file_hash(artifact)):
            raise ValueError('chunk differs from its exact identity or artifact binding')
        for row in item['input']:
            if row['id'] in ids:raise ValueError('duplicate verified input id')
            ids.add(row['id'])
        input_count+=item['rows'];existing.extend(item['input'])
    pending=[];seen=0;receipts=[]
    def close():
        nonlocal input_count
        if not pending:return
        validate_identity(identity)
        number=len(markers)+len(receipts); artifact=out/f'chunk-{number:08d}.jsonl'; marker=out/f'chunk-{number:08d}.receipt.json'
        payload=''.join(canonical_json_str(graph)+'\n' for _,graph in pending).encode()
        temporary=artifact.with_name(artifact.name+'.tmp')
        if marker.exists():raise ValueError('unexpected committed chunk marker')
        if artifact.exists():
            if _file_hash(artifact)!=hashlib.sha256(payload).hexdigest():
                raise ValueError('uncommitted chunk differs from freshly verified bytes')
        else:
            temporary.unlink(missing_ok=True)
            with temporary.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
            os.replace(temporary,artifact)
        validate_identity(identity)
        record={'schema':'natlang.verified-source-chunk/1','identity':identity,'identity_sha256':ident,'start':input_count,
                'rows':len(pending),'input':[entry for entry,_ in pending],'graph_path':str(artifact.resolve()),
                'graph_sha256':hashlib.sha256(payload).hexdigest(),'training_exposure':0,'optimizer_updates':0}
        write_json_durable(marker,record);receipts.append(record);input_count+=len(pending);pending.clear()
    for row in rows:
        rid=row.get('id')
        if not isinstance(rid,str) or not rid:raise ValueError('input lacks exact row identity')
        witness={'id':rid,'sha256':digest(row)}
        if seen < len(existing):
            if witness!=existing[seen]:raise ValueError('resumed input differs from verified prefix')
        else:
            if rid in ids:raise ValueError('duplicate source row id')
            graph=verify(row)
            if not isinstance(graph,dict):raise ValueError('verifier did not return an admitted graph')
            ids.add(rid);pending.append((witness,graph))
            if len(pending)==chunk_rows:close()
        seen+=1
    if seen<len(existing):raise ValueError('input is shorter than verified prefix')
    close();validate_identity(identity)
    return {'rows':seen,'reused_rows':len(existing),'new_chunks':len(receipts),'identity_sha256':ident}
