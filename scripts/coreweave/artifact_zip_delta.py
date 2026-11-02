#!/usr/bin/env python3
"""Exact, bounded ZIP-payload deltas for independently bound immutable artifacts.

The receipt is an artifact: distribute it with its own independently bound SHA.
Entry payloads use exact copies, compressed XOR, or compressed inline bytes.
Headers, descriptors and the central directory remain inline.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, struct, tempfile, zipfile, zlib
from pathlib import Path
try:
    from .artifact_delta import CHUNK, digest_file, identity, publish, xor_bytes
except ImportError:
    from artifact_delta import CHUNK, digest_file, identity, publish, xor_bytes

SCHEMA_V1='natlang.artifact-zip-delta/1'
SCHEMA='natlang.artifact-zip-delta/2'
SUPPORTED_SCHEMAS=(SCHEMA_V1,SCHEMA)
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


def entry_choices(base, target, prior, offset, size):
    """Measure both encodings in one bounded pass, without scratch files."""
    base_offset,base_size=prior[:2]
    base.seek(base_offset);target.seek(offset)
    source_hash,target_hash=hashlib.sha256(),hashlib.sha256()
    raw,zxor=zlib.compressobj(1),zlib.compressobj(1)
    raw_bytes=xor_size=0
    remaining=size;base_remaining=base_size
    while remaining:
        block=target.read(min(CHUNK,remaining))
        if not block:raise ValueError('short artifact range')
        take=min(len(block),base_remaining)
        old=base.read(take)
        if len(old)!=take:raise ValueError('short base range')
        source_hash.update(old);target_hash.update(block)
        raw_bytes+=len(raw.compress(block))
        xor_size+=len(zxor.compress(xor_bytes(block,old)))
        remaining-=len(block);base_remaining-=take
    while base_remaining:
        old=base.read(min(CHUNK,base_remaining))
        if not old:raise ValueError('short base range')
        source_hash.update(old);base_remaining-=len(old)
    raw_bytes+=len(raw.flush());xor_size+=len(zxor.flush())
    return raw_bytes,xor_size,source_hash.hexdigest(),target_hash.hexdigest()


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
        ops=[]; copied=0; xored=0
        with os.fdopen(fd,'wb') as writer,base.open('rb') as a,target.open('rb') as b:
            def write(data):
                if data:
                    require_free(out,min_free_bytes+len(data))
                    writer.write(data)
            def inline(start,end):
                if start==end:return
                begin=writer.tell();compressor=zlib.compressobj(1)
                for data in stream_range(b,start,end-start):
                    write(compressor.compress(data))
                write(compressor.flush())
                ops.append({'kind':'inline','offset':begin,'bytes':writer.tell()-begin,'output_bytes':end-start})
            cursor=0
            for name,(offset,size,method) in sorted(new.items(),key=lambda item:item[1][0]):
                if offset<cursor:raise ValueError('overlapping ZIP entry payloads')
                prior=old.get(name)
                if not size or prior is None:continue
                if prior[1]==size:
                    actual=range_hash(b,offset,size)
                    if actual==range_hash(a,prior[0],size):
                        inline(cursor,offset)
                        ops.append({'kind':'copy','offset':prior[0],'bytes':size,'sha256':actual})
                        copied+=size;cursor=offset+size
                        continue
                inline_size,xor_size,old_hash,actual=entry_choices(a,b,prior,offset,size)
                inline(cursor,offset)
                if xor_size < inline_size:
                    begin=writer.tell();compressor=zlib.compressobj(1)
                    a.seek(prior[0]);remaining=prior[1]
                    for block in stream_range(b,offset,size):
                        take=min(len(block),remaining);base_block=a.read(take)
                        if len(base_block)!=take:raise ValueError('short base range')
                        remaining-=take;write(compressor.compress(xor_bytes(block,base_block)))
                    write(compressor.flush())
                    if writer.tell()-begin!=xor_size:raise ValueError('XOR encoding changed during construction')
                    ops.append({'kind':'xor','offset':begin,'bytes':xor_size,'output_bytes':size,
                        'base_offset':prior[0],'base_bytes':prior[1],
                        'base_sha256':old_hash,'sha256':actual})
                    xored+=size
                else:
                    begin=writer.tell();inline(offset,offset+size)
                    if writer.tell()-begin!=inline_size:raise ValueError('inline encoding changed during construction')
                cursor=offset+size
            inline(cursor,before[1][2]);writer.flush();os.fsync(writer.fileno())
        if [identity(base),identity(target)]!=before:raise ValueError('delta input changed during encoding')
        value={'schema':SCHEMA,'base':{'sha256':base_sha,'bytes':before[0][2]},
               'target':{'sha256':target_sha,'bytes':before[1][2]},
               'payload':{'path':payload.name,'sha256':digest_file(pending),'bytes':Path(pending).stat().st_size},
               'copied_bytes':copied,'xored_bytes':xored,'operations':ops}
        publish(pending,payload)
        fd,meta=tempfile.mkstemp(prefix='.zip-receipt-',dir=out)
        try:
            with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
            publish(meta,receipt)
        finally:Path(meta).unlink(missing_ok=True)
        return {'path':str(receipt),'sha256':digest_file(receipt),'target':value['target'],
                'payload':value['payload'],'copied_bytes':copied,'xored_bytes':xored,'operations':len(ops)}
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
    schema=value.get('schema')
    if schema not in SUPPORTED_SCHEMAS or value.get('target')!={'sha256':expected_sha256,'bytes':expected_bytes}:
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
                elif op.get('kind') in ('inline','xor'):
                    is_xor=op.get('kind')=='xor'
                    expected=op.get('output_bytes')
                    if offset!=inline_cursor or offset+size>before[2][2] or type(expected)!=int or expected<=0:raise ValueError('invalid inline range')
                    if is_xor:
                        base_offset,base_size=op.get('base_offset'),op.get('base_bytes')
                        if (schema!=SCHEMA or type(base_offset)!=int or base_offset<0 or type(base_size)!=int
                                or base_size<0 or base_offset+base_size>before[0][2]
                                or not valid_hash(op.get('base_sha256')) or not valid_hash(op.get('sha256'))):
                            raise ValueError('invalid XOR base range')
                        if range_hash(a,base_offset,base_size)!=op['base_sha256']:
                            raise ValueError('XOR base range hash differs')
                        a.seek(base_offset);base_remaining=base_size;output_hash=hashlib.sha256()
                    inline_cursor+=size;decoder=zlib.decompressobj();n=0
                    for block in stream_range(p,offset,size):
                        while block:
                            data=decoder.decompress(block,min(CHUNK,expected-n+1));block=decoder.unconsumed_tail
                            n+=len(data)
                            if n>expected or decoder.unused_data:raise ValueError('invalid inline compressed data')
                            if is_xor:
                                take=min(len(data),base_remaining);old=a.read(take)
                                if len(old)!=take:raise ValueError('short XOR base range')
                                base_remaining-=take;data=xor_bytes(data,old);output_hash.update(data)
                            emit(data)
                    if not decoder.eof or n!=expected:raise ValueError('incomplete inline data')
                    if is_xor and output_hash.hexdigest()!=op['sha256']:raise ValueError('XOR output hash differs')
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
