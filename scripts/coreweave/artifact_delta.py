#!/usr/bin/env python3
"""Bounded, exact binary deltas for immutable artifacts already present at a peer.

This only changes transport. The receiver must supply the target size and SHA256
from its independently bound actor manifest; reconstructed bytes retain identity.
Uses the standard library so the receiver needs neither Torch nor NumPy.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, tempfile, zlib
from pathlib import Path

CHUNK=1024*1024
SCHEMA='natlang.artifact-xor-zlib/1'


def identity(path):
    s=Path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(CHUNK),b''):h.update(block)
    return h.hexdigest()


def xor_bytes(value, base):
    return (int.from_bytes(value,'little') ^ int.from_bytes(base.ljust(len(value),b'\0'),'little')).to_bytes(len(value),'little')


def publish(pending, target):
    # No replacement of an independently created or already valid artifact.
    os.link(pending,target);Path(pending).unlink()
    fd=os.open(Path(target).parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def build(base, target, out):
    base,target=Path(base).resolve(strict=True),Path(target).resolve(strict=True)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    delta,receipt=out/'delta.zlib',out/'receipt.json'
    if delta.exists() or receipt.exists():raise ValueError('delta publication already exists')
    before=[identity(base),identity(target)]
    if min(before[0][2],before[1][2])<=0:raise ValueError('delta inputs must be nonempty')
    old_hash,new_hash,delta_hash=hashlib.sha256(),hashlib.sha256(),hashlib.sha256()
    compressor=zlib.compressobj(level=1)
    fd,pending=tempfile.mkstemp(prefix='.delta-',dir=out)
    try:
        with os.fdopen(fd,'wb') as writer,base.open('rb') as a,target.open('rb') as b:
            def emit(data):
                writer.write(data);delta_hash.update(data)
            for new in iter(lambda:b.read(CHUNK),b''):
                old=a.read(len(new));old_hash.update(old);new_hash.update(new)
                emit(compressor.compress(xor_bytes(new,old)))
            for old in iter(lambda:a.read(CHUNK),b''):old_hash.update(old)
            emit(compressor.flush());writer.flush();os.fsync(writer.fileno())
        if [identity(base),identity(target)]!=before:raise ValueError('delta input changed during encoding')
        size=Path(pending).stat().st_size
        value={'schema':SCHEMA,'base':{'sha256':old_hash.hexdigest(),'bytes':before[0][2]},
            'target':{'sha256':new_hash.hexdigest(),'bytes':before[1][2]},
            'delta':{'path':delta.name,'sha256':delta_hash.hexdigest(),'bytes':size},
            'ratio':size/before[1][2]}
        publish(pending,delta)
        fd,meta=tempfile.mkstemp(prefix='.receipt-',dir=out)
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            publish(meta,receipt)
        finally:Path(meta).unlink(missing_ok=True)
        return {'path':str(receipt),'sha256':digest_file(receipt),**value}
    finally:Path(pending).unlink(missing_ok=True)


def apply(base, manifest, out, *, expected_sha256, expected_bytes, min_free_bytes=0):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256) or type(expected_bytes) is not int or expected_bytes<=0:
        raise ValueError('receiver requires an independently bound target identity')
    base,manifest=Path(base).resolve(strict=True),Path(manifest).resolve(strict=True)
    raw=manifest.read_bytes();value=json.loads(raw)
    if value.get('schema')!=SCHEMA or value.get('target')!={'sha256':expected_sha256,'bytes':expected_bytes}:
        raise ValueError('delta target differs from receiver binding')
    ref=value['delta'];name=ref['path']
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
        raise ValueError('delta path must be one local filename')
    delta=manifest.parent/name;out=Path(out).absolute()
    if out.exists() or out.is_symlink():raise ValueError('refusing to replace an existing artifact')
    before=[identity(base),identity(delta),identity(manifest)]
    if before[0][2]!=value['base']['bytes'] or digest_file(base)!=value['base']['sha256']:
        raise ValueError('delta base differs from its binding')
    if before[1][2]!=ref['bytes']:raise ValueError('delta payload size differs')
    out.parent.mkdir(parents=True,exist_ok=True);free=os.statvfs(out.parent)
    if free.f_bavail*free.f_frsize<expected_bytes+min_free_bytes:raise ValueError('insufficient artifact reconstruction headroom')
    fd,pending=tempfile.mkstemp(prefix='.reconstruct-',dir=out.parent)
    decoder=zlib.decompressobj();h,dh=hashlib.sha256(),hashlib.sha256();count=0
    try:
        with os.fdopen(fd,'wb') as writer,base.open('rb') as a,delta.open('rb') as source:
            for compressed in iter(lambda:source.read(CHUNK),b''):
                dh.update(compressed)
                while compressed:
                    decoded=decoder.decompress(compressed,min(CHUNK,expected_bytes-count+1))
                    compressed=decoder.unconsumed_tail
                    if decoder.unused_data:raise ValueError('trailing compressed payload')
                    count+=len(decoded)
                    if count>expected_bytes:raise ValueError('delta exceeds receiver size bound')
                    actual=xor_bytes(decoded,a.read(len(decoded)));h.update(actual);writer.write(actual)
            if not decoder.eof or count!=expected_bytes:raise ValueError('incomplete reconstructed artifact')
            if h.hexdigest()!=expected_sha256 or dh.hexdigest()!=ref['sha256']:
                raise ValueError('reconstructed artifact or delta hash differs')
            writer.flush();os.fsync(writer.fileno())
        if [identity(base),identity(delta),identity(manifest)]!=before or manifest.read_bytes()!=raw:
            raise ValueError('delta input changed during reconstruction')
        if digest_file(pending)!=expected_sha256:
            raise ValueError('reconstructed artifact failed stored-byte readback')
        publish(pending,out)
        return {'path':str(out),'sha256':h.hexdigest(),'bytes':count,'delta_sha256':dh.hexdigest()}
    finally:Path(pending).unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--base',required=True);b.add_argument('--target',required=True);b.add_argument('--out',required=True)
    a=sub.add_parser('apply');a.add_argument('--base',required=True);a.add_argument('--manifest',required=True);a.add_argument('--out',required=True)
    a.add_argument('--expected-sha256',required=True);a.add_argument('--expected-bytes',required=True,type=int);a.add_argument('--min-free-bytes',type=int,default=0)
    v=vars(p.parse_args());command=v.pop('command');print(json.dumps(build(**v) if command=='build' else apply(**v)),flush=True)


if __name__=='__main__':main()
