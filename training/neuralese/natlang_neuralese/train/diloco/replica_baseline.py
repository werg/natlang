"""Immutable coordinator baseline pinning without network under its state gate."""
from __future__ import annotations
import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from ...common.jsonio import write_json_durable

# Digests this process computed in full, keyed by the exact inode state that was
# hashed (device, inode, size, mtime, ctime). Any link, chmod or rewrite changes
# ctime and misses, so a reuse claims only bytes this process actually hashed.
# Rotation pins ~30 GB of hardlinks inside the serialized writer; without reuse
# it re-hashed every file twice (pin, then retention verify) on a one-CPU host.
# Retiring a generation unlinks its hardlinks, which changes the ctime of every
# inode still shared with the coordinator root (seed, manifest, unchanged
# fragments). Only a pin inside the serialized writer may reuse a digest by
# (device, inode, size, mtime): the coordinator never writes in place, and an
# in-place write would still fail every strict `verify`, including restore.
_KNOWN=OrderedDict();_UNCHANGED=OrderedDict();_KNOWN_LOCK=threading.Lock();_KNOWN_LIMIT=1<<16


def inode_key(st):
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def remember(key,digest):
    with _KNOWN_LOCK:
        for index,value in ((_KNOWN,key),(_UNCHANGED,key[:4])):
            index[value]=digest;index.move_to_end(value)
            while len(index)>_KNOWN_LIMIT:index.popitem(last=False)


def _pinned_digest(key):
    with _KNOWN_LOCK:return _KNOWN.get(key) or _UNCHANGED.get(key[:4])


def known(key):
    with _KNOWN_LOCK:
        digest=_KNOWN.get(key)
        if digest is not None:_KNOWN.move_to_end(key)
        return digest


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()


def _stable_digest(path,message):
    """Hash once per exact inode state; fail if the file changes while hashed."""
    before=Path(path).stat();key=inode_key(before);digest=known(key)
    if digest is None:
        digest=_hash(path)
        if inode_key(Path(path).stat())!=key:raise ValueError(message)
        remember(key,digest)
    return before,digest


def capture(dispatch, destination, *, captured_monotonic=None, captured_wall=None, pin_extra=None, on_pinned=None):
    """Hardlink exact serialized state inside the queue; hash only after release."""
    target=Path(destination)
    def pin():
        c=dispatch.machine.coordinator;cache=getattr(dispatch.machine,'hash_cache',None) or {}
        target.mkdir(parents=True,exist_ok=False)
        files=[]
        for name in ('manifest.json','seed.safetensors','progress.json','fragments','pending'):
            source=c.root/name
            if not source.exists():
                if name != 'pending':raise ValueError('baseline missing '+name)
                continue
            paths=sorted(source.rglob('*')) if source.is_dir() else [source]
            for path in paths:
                if path.is_symlink():raise ValueError('baseline contains symlink')
                if not path.is_file() or path.name.startswith('.'):continue
                relative=path.relative_to(c.root);dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                prior=inode_key(path.stat());digest=cache.get(prior) or _pinned_digest(prior)
                os.link(path,dest);files.append(str(relative))
                if digest is not None:remember(inode_key(dest.stat()),digest)
        for directory in sorted({target, *(p.parent for p in target.rglob('*') if p.is_file())}, key=lambda p:len(p.parts), reverse=True):
            fd=os.open(directory,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
            try:os.fsync(fd)
            finally:os.close(fd)
        extra=pin_extra(c) if pin_extra is not None else None
        return {'files':files,'quorum_since':{str(k):v for k,v in c.quorum_since.items()},
                'captured_monotonic':(__import__('time').monotonic() if captured_monotonic is None else captured_monotonic),
                'captured_wall':(__import__('time').time() if captured_wall is None else captured_wall),'run_id':c.manifest['run_id'],
                'fingerprint':c.manifest['fingerprint'],'extra':extra}
    pinned=dispatch.read(pin)
    if on_pinned is not None:on_pinned(pinned)
    return finish_pinned(target,pinned)


def finish_pinned(target,pinned):
    target=Path(target);pinned=dict(pinned)
    inventory=[]
    for name in pinned.pop('files'):
        path=target/name;after,digest=_stable_digest(path,'pinned baseline changed during hashing')
        inventory.append({'path':name,'bytes':after.st_size,'sha256':digest})
    from ..native_resident import code_inventory, digest
    runtime=digest(code_inventory(Path(__file__).resolve().parents[2]))
    value={'schema':'natlang.diloco-replica-baseline/1',**pinned,'runtime_code_sha256':runtime,'files':inventory}
    write_json_durable(target/'replica-baseline.json',value)
    return {'path':str(target/'replica-baseline.json'),'sha256':_hash(target/'replica-baseline.json'),'value':value}


def verify(root, expected_sha256):
    root=Path(root);path=root/'replica-baseline.json'
    if _hash(path)!=expected_sha256:raise ValueError('baseline receipt digest differs')
    value=json.loads(path.read_text())
    if value.get('schema')!='natlang.diloco-replica-baseline/1':raise ValueError('unsupported baseline')
    seen=set()
    for item in value.get('files',[]):
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:raise ValueError('invalid baseline path')
        seen.add(str(relative));path=root/relative
        if not path.resolve().is_relative_to(root.resolve()):raise ValueError('baseline path escapes root')
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=item['bytes'] or _stable_digest(path,'baseline file differs')[1]!=item['sha256']:
            raise ValueError('baseline file differs')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='replica-baseline.json'}
    if actual != seen:raise ValueError('baseline inventory is incomplete')
    if not {'manifest.json','seed.safetensors','progress.json'} <= seen:raise ValueError('baseline lacks required files')
    from ..native_resident import code_inventory, digest
    if value.get('runtime_code_sha256') != digest(code_inventory(Path(__file__).resolve().parents[2])):
        raise ValueError('baseline runtime inventory differs')
    manifest=json.loads((root/'manifest.json').read_text())
    if value['run_id']!=manifest['run_id'] or value['fingerprint']!=manifest['fingerprint']:
        raise ValueError('baseline run identity differs')
    expected={f"fragments/{int(item['index']):05d}.safetensors" for item in manifest['fragments']}
    if not expected <= seen:raise ValueError('baseline lacks required fragments')
    return value


def reconstruct(source, destination, *, expected_sha256):
    """Clone a verified immutable baseline into a new writable coordinator root."""
    source=Path(source);destination=Path(destination);value=verify(source,expected_sha256)
    destination.mkdir(parents=True,exist_ok=False)
    for item in value['files']:
        relative=Path(item['path']);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        os.link(source/relative,target)
    return value
