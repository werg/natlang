#!/usr/bin/env python3
"""Seal exact local dependency bytes for a DiLoCo topology recovery candidate."""
from __future__ import annotations
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
import tarfile

from natlang_neuralese.common.hashing import sha256_file_hex, canonical_json_sha256_hex



def control_bindings(archive):
    """Collect path and hash declarations from the pinned resident control JSON."""
    result = {}
    def visit(value):
        if isinstance(value, dict):
            path = value.get('path'); sha = value.get('file_sha256', value.get('sha256'))
            if isinstance(path, str) and isinstance(sha, str) and len(sha) == 64:
                result[(path, sha)] = {'path': path, 'sha256': sha}
            for key, child in value.items():
                if key not in ('by_source_group', 'code'): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    with tarfile.open(archive, 'r') as tar:
        for member in tar:
            if member.isfile() and member.name.endswith('.json'):
                if member.size > 64 * 2**20: raise ValueError('resident control JSON too large')
                visit(json.load(tar.extractfile(member)))
    return list(result.values())

def seal(bindings, candidates, bundle, sources, output, *, runtime_code_sha256, reserve_bytes=60 * 2**30):
    """Fail closed unless every declared binding and launch-bundle object is locally exact.

    The output is a content archive, never evidence of a remotely durable backup.
    """
    bundle = json.loads(Path(bundle).read_text())
    sources = json.loads(Path(sources).read_text())
    # The delivered archive must contain the exact executing native Python inventory.
    inventory = {}
    marker = 'training/neuralese/natlang_neuralese/'
    with tarfile.open(sources[bundle['code']['key']], 'r:*') as code:
        for member in code:
            name = member.name.removeprefix('./')
            if name.startswith(marker) and name.endswith('.py') and member.isfile():
                relative = name[len(marker):]
                if relative in inventory: raise ValueError('duplicate runtime module in code archive')
                inventory[relative] = hashlib.file_digest(code.extractfile(member), 'sha256').hexdigest()
    if not inventory or canonical_json_sha256_hex(inventory) != runtime_code_sha256:
        raise ValueError('code archive differs from executing native runtime')
    declared = list(bindings)
    for item in [bundle['code'], *bundle['files']]:
        declared.append({'path': item.get('path', '@bundle-code'), 'sha256': item['sha256'],
                         'bytes': item['bytes'], 'candidate': sources[item['key']]})
    resolved = {}; index = []
    for item in declared:
        sha = item['sha256']; logical = item['path']
        choices = [item.get('candidate'), logical, *candidates.get(sha, [])]
        if logical.startswith('/work/hf/') and os.environ.get('NATLANG_RECOVERY_HF_CACHE'):
            choices.append(str(Path(os.environ['NATLANG_RECOVERY_HF_CACHE']) / logical.removeprefix('/work/hf/')))
        if logical.startswith('/work/mirror/'):
            choices.append(logical[len('/work/mirror'):])
        path = None
        for choice in filter(None, choices):
            candidate = Path(choice)
            if not candidate.is_file(): continue
            before = candidate.stat()
            if item.get('bytes') is not None and before.st_size != item['bytes']: continue
            if sha256_file_hex(candidate, drop_cache=True) != sha: continue
            after = candidate.stat()
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                continue
            path = candidate; break
        if path is None: raise ValueError('missing exact dependency '+logical+' '+sha)
        resolved.setdefault(sha, path)
        index.append({'path':logical,'sha256':sha,'bytes':path.stat().st_size,'object':'objects/'+sha})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise ValueError('dependency output already exists')
    needed=sum(p.stat().st_size for p in resolved.values())
    if shutil.disk_usage(output.parent).free < needed + reserve_bytes:
        raise OSError('dependency archive would consume the 60 GiB disk reserve')
    temp=output.with_name(output.name+'.tmp')
    try:
        with tarfile.open(temp,'x',dereference=True) as tar:
            for sha,path in sorted(resolved.items()):
                before=path.stat();tar.add(path,arcname='objects/'+sha,recursive=False)
                after=path.stat()
                if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                    raise ValueError('dependency changed during archive '+str(path))
            raw=json.dumps({'schema':'natlang.diloco-recovery-dependencies/1','bindings':index},sort_keys=True).encode()
            info=tarfile.TarInfo('dependency-index.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        # Read the actual archive members back; pre-archive source checks alone do not prove the archive.
        with tarfile.open(temp,'r') as tar:
            for sha in resolved:
                if hashlib.file_digest(tar.extractfile('objects/'+sha),'sha256').hexdigest()!=sha:
                    raise ValueError('archived dependency differs '+sha)
        os.link(temp,output);temp.unlink()
        return {'path':str(output),'sha256':sha256_file_hex(output,drop_cache=True),'bytes':output.stat().st_size,
                'bindings':len(index),'objects':len(resolved)}
    finally:
        temp.unlink(missing_ok=True)
