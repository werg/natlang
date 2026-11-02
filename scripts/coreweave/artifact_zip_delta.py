#!/usr/bin/env python3
"""Exact, bounded ZIP-payload deltas for independently bound immutable artifacts.

The receipt is an artifact: distribute it with its own independently bound SHA.
Copy ranges are selected only after hashing both actual compressed entry payloads.
Headers, descriptors, central directory, pickle and other changed bytes are inline.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, struct, tempfile, zipfile, zlib
from pathlib import Path
try:
    from .artifact_delta import CHUNK, digest_file, identity, publish
except ImportError:
    from artifact_delta import CHUNK, digest_file, identity, publish

SCHEMA='natlang.artifact-zip-delta/1'
HEX=re.compile(r'[0-9a-f]{64}\Z')
FLOOR=40*1024**3

def valid_hash(value):
    return isinstance(value,str) and HEX.fullmatch(value) is not None

def require_free(path, minimum):
    if type(minimum) is not int or minimum < 0:raise ValueError('invalid free-space floor')
    stat=os.statvfs(path)
    if stat.f_bavail*stat.f_frsize<minimum:raise ValueError('insufficient delta free-space floor')


def ranges(path):
    result={}
    with zipfile.ZipFile(path) as archive, open(path,'rb') as f:
        for entry in archive.infolist():
            if entry.filename in result: raise ValueError('duplicate ZIP entry name')
            f.seek(entry.header_offset); header=f.read(30)
            if len(header)!=30 or header[:4]!=b'PK\x03\x04': raise ValueError('invalid ZIP local header')
            n,e=struct.unpack_from('<HH',header,26)
            offset=entry.header_offset+30+n+e
            if offset+entry.compress_size > os.fstat(f.fileno()).st_size: raise ValueError('ZIP payload outside artifact')
            result[entry.filename]=(offset,entry.compress_size,entry.compress_type)
    return result


def stream_range(f, offset, size):
    f.seek(offset)
    while size:
        data=f.read(min(size,CHUNK))
        if not data: raise ValueError('short artifact range')
        size-=len(data);yield data


def range_hash(f,offset,size):
    h=hashlib.sha256()
    for data in stream_range(f,offset,size):h.update(data)
    return h.hexdigest()


def build(base,target,out,*,min_free_bytes=FLOOR):
    base,target=Path(base).resolve(strict=True),Path(target).resolve(strict=True)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    require_free(out,min_free_bytes)
    payload,receipt=out/'payload.zlib',out/'receipt.json'
    if payload.exists() or receipt.exists():raise ValueError('delta publication already exists')
    before=[identity(base),identity(target)]
    old,new=ranges(base),ranges(target)
    base_sha,target_sha=digest_file(base),digest_file(target)
    fd,pending=tempfile.mkstemp(prefix='.zip-delta-',dir=out)
    try:
        ops=[]; copied=0
        with os.fdopen(fd,'wb') as writer,base.open('rb') as a,target.open('rb') as b:
            def inline(start,end):
                if start==end:return
                begin=writer.tell();compressor=zlib.compressobj(1)
                for data in stream_range(b,start,end-start):
                    require_free(out,min_free_bytes+len(data))
                    writer.write(compressor.compress(data))
                writer.write(compressor.flush())
                ops.append({'kind':'inline','offset':begin,'bytes':writer.tell()-begin,'output_bytes':end-start})
            cursor=0
            for name,(offset,size,method) in sorted(new.items(),key=lambda item:item[1][0]):
                if offset<cursor:raise ValueError('overlapping ZIP entry payloads')
                prior=old.get(name)
                # data.pkl is metadata and may change; compare every candidate afresh.
                if not size or prior is None or prior[1:]!=(size,method):continue
                actual=range_hash(b,offset,size)
                if actual!=range_hash(a,prior[0],size):continue
                inline(cursor,offset)
                ops.append({'kind':'copy','offset':prior[0],'bytes':size,'sha256':actual})
                copied+=size;cursor=offset+size
            inline(cursor,before[1][2]);writer.flush();os.fsync(writer.fileno())
        if [identity(base),identity(target)]!=before:raise ValueError('delta input changed during encoding')
        value={'schema':SCHEMA,'base':{'sha256':base_sha,'bytes':before[0][2]},
               'target':{'sha256':target_sha,'bytes':before[1][2]},
               'payload':{'path':payload.name,'sha256':digest_file(pending),'bytes':Path(pending).stat().st_size},
               'copied_bytes':copied,'operations':ops}
        publish(pending,payload)
        fd,meta=tempfile.mkstemp(prefix='.zip-receipt-',dir=out)
        try:
            with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
            publish(meta,receipt)
        finally:Path(meta).unlink(missing_ok=True)
        return {'path':str(receipt),'sha256':digest_file(receipt),'target':value['target'],
                'payload':value['payload'],'copied_bytes':copied,'operations':len(ops)}
    finally:Path(pending).unlink(missing_ok=True)


def verify(base,manifest,*,expected_sha256,expected_bytes,expected_manifest_sha256,out=None,min_free_bytes=FLOOR):
    if not valid_hash(expected_sha256) or not valid_hash(expected_manifest_sha256) or type(expected_bytes)!=int or expected_bytes<=0:
        raise ValueError('independent target and receipt bindings required')
    if type(min_free_bytes) is not int or min_free_bytes < 0:raise ValueError('invalid free-space floor')
    base,manifest=Path(base).resolve(strict=True),Path(manifest).resolve(strict=True)
    before=[identity(base),identity(manifest)]
    if before[1][2]>64*1024**2:raise ValueError('receipt exceeds bounded parser size')
    raw=manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_manifest_sha256:raise ValueError('receipt hash differs')
    value=json.loads(raw)
    if value.get('schema')!=SCHEMA or value.get('target')!={'sha256':expected_sha256,'bytes':expected_bytes}:
        raise ValueError('target binding differs')
    ref=value['payload'];name=ref['path']
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):raise ValueError('invalid payload path')
    payload=manifest.parent/name;before.append(identity(payload))
    if before[0][2]!=value['base']['bytes'] or digest_file(base)!=value['base']['sha256']:raise ValueError('base binding differs')
    if before[2][2]!=ref['bytes'] or digest_file(payload)!=ref['sha256']:raise ValueError('payload binding differs')
    out=Path(out).absolute() if out is not None else None
    if out is not None:
        if out.exists() or out.is_symlink():raise ValueError('refusing to replace artifact')
        out.parent.mkdir(parents=True,exist_ok=True)
        require_free(out.parent,min_free_bytes+expected_bytes)
        fd,pending=tempfile.mkstemp(prefix='.zip-reconstruct-',dir=out.parent);writer=os.fdopen(fd,'wb')
    else:pending=None;writer=None
    h=hashlib.sha256();count=0;inline_cursor=0
    try:
        def emit(data):
            nonlocal count
            count+=len(data)
            if count>expected_bytes:raise ValueError('operations exceed target size')
            h.update(data)
            if writer:writer.write(data)
        with base.open('rb') as a,payload.open('rb') as p:
            for op in value['operations']:
                offset,size=op.get('offset'),op.get('bytes')
                if type(offset)!=int or type(size)!=int or offset<0 or size<=0:raise ValueError('invalid delta range')
                if op.get('kind')=='copy':
                    if offset+size>before[0][2] or not valid_hash(op.get('sha256')):raise ValueError('invalid base range')
                    rh=hashlib.sha256()
                    for data in stream_range(a,offset,size):rh.update(data);emit(data)
                    if rh.hexdigest()!=op['sha256']:raise ValueError('base range hash differs')
                elif op.get('kind')=='inline':
                    expected=op.get('output_bytes')
                    if offset!=inline_cursor or offset+size>before[2][2] or type(expected)!=int or expected<=0:raise ValueError('invalid inline range')
                    inline_cursor+=size;decoder=zlib.decompressobj();n=0
                    for block in stream_range(p,offset,size):
                        while block:
                            data=decoder.decompress(block,min(CHUNK,expected-n+1));block=decoder.unconsumed_tail
                            n+=len(data);emit(data)
                            if n>expected or decoder.unused_data:raise ValueError('invalid inline compressed data')
                    if not decoder.eof or n!=expected:raise ValueError('incomplete inline data')
                else:raise ValueError('unknown delta operation')
        if inline_cursor!=before[2][2] or count!=expected_bytes or h.hexdigest()!=expected_sha256:raise ValueError('reconstructed target differs')
        if [identity(base),identity(manifest),identity(payload)]!=before:raise ValueError('delta input changed during reconstruction')
        if writer:
            writer.flush();os.fsync(writer.fileno());writer.close();writer=None
            if digest_file(pending)!=expected_sha256:raise ValueError('reconstructed readback differs')
            publish(pending,out)
        return {'sha256':h.hexdigest(),'bytes':count,'receipt_sha256':expected_manifest_sha256,'path':str(out) if out else None}
    finally:
        if writer:writer.close()
        if pending:Path(pending).unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--base',required=True);b.add_argument('--target',required=True);b.add_argument('--out',required=True);b.add_argument('--min-free-bytes',type=int,default=FLOOR)
    v=sub.add_parser('verify');v.add_argument('--base',required=True);v.add_argument('--manifest',required=True);v.add_argument('--expected-sha256',required=True);v.add_argument('--expected-bytes',required=True,type=int);v.add_argument('--expected-manifest-sha256',required=True);v.add_argument('--out');v.add_argument('--min-free-bytes',type=int,default=FLOOR)
    v.add_argument('--result-out')
    args=vars(p.parse_args());command=args.pop('command');result_out=args.pop('result_out',None)
    result=build(**args) if command=='build' else verify(**args)
    if result_out:
        path=Path(result_out).absolute();path.parent.mkdir(parents=True,exist_ok=True)
        fd,pending=tempfile.mkstemp(prefix='.zip-verify-',dir=path.parent)
        try:
            with os.fdopen(fd,'w') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
            publish(pending,path)
        finally:Path(pending).unlink(missing_ok=True)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
