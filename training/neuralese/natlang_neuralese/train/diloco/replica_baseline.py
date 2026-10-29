"""Immutable coordinator baseline pinning without network under its state gate."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from ...common.jsonio import write_json_durable


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()


def capture(dispatch, destination):
    """Hardlink exact serialized state inside the queue; hash only after release."""
    target=Path(destination)
    def pin():
        c=dispatch.machine.coordinator
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
                os.link(path,dest);files.append(str(relative))
        for directory in sorted({target, *(p.parent for p in target.rglob('*') if p.is_file())}, key=lambda p:len(p.parts), reverse=True):
            fd=os.open(directory,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
            try:os.fsync(fd)
            finally:os.close(fd)
        return {'files':files,'quorum_since':{str(k):v for k,v in c.quorum_since.items()},
                'captured_monotonic':__import__('time').monotonic(),'run_id':c.manifest['run_id'],
                'fingerprint':c.manifest['fingerprint']}
    pinned=dispatch.read(pin)
    inventory=[]
    for name in pinned.pop('files'):
        path=target/name;before=path.stat();digest=_hash(path);after=path.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('pinned baseline changed during hashing')
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
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=item['bytes'] or _hash(path)!=item['sha256']:
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
