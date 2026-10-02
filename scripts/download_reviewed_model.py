#!/usr/bin/env python3
"""Authenticated, pinned background model download with byte and LFS checks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from huggingface_hub import HfApi, get_token, snapshot_download

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('repository'); p.add_argument('revision'); p.add_argument('status_directory', type=Path)
    args = p.parse_args(); root = args.status_directory; root.mkdir(parents=True, exist_ok=True)
    def status(state, **extra):
        row = dict(state=state, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   pid=os.getpid(), repository=args.repository, revision=args.revision, **extra)
        tmp = root / 'status.tmp'; tmp.write_text(json.dumps(row, indent=2)+'\n'); tmp.replace(root/'status.json')
        print(json.dumps(row), flush=True)
    try:
        token = get_token()
        if not token: raise ValueError('HF token absent from configured cache')
        api = HfApi(); api.whoami(token=token)
        info = api.model_info(args.repository, revision=args.revision, token=token, files_metadata=True)
        if info.sha != args.revision: raise ValueError('Pinned revision mismatch')
        weights = [f for f in info.siblings if f.rfilename.endswith('.safetensors')]
        if not weights or any(not f.lfs or not f.lfs.sha256 for f in weights):
            raise ValueError('Missing weight LFS verification metadata')
        cache = Path(os.environ.get('HF_HOME', Path.home()/'.cache/huggingface'))/'hub'
        cache.mkdir(parents=True, exist_ok=True)
        candidate = cache/('models--'+args.repository.replace('/', '--'))/'snapshots'/args.revision
        missing = sum(f.size for f in weights if not (candidate/f.rfilename).exists())
        free = shutil.disk_usage(cache).free
        if free < missing + 5*1024**3:
            status('paused_insufficient_disk', available_bytes=free, missing_weight_bytes=missing, reserved_bytes=5*1024**3); return
        metadata = dict(repository=args.repository, revision=info.sha, authenticated=True,
                        weights=[dict(name=f.rfilename,size=f.size,sha256=f.lfs.sha256) for f in weights])
        (root/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        status('downloading', authenticated=True, weight_bytes=sum(f.size for f in weights), workers=4, transport='HTTP; Xet disabled')
        path = Path(snapshot_download(args.repository, revision=args.revision, token=token, max_workers=4,
          allow_patterns=['*.json','*.jinja','*.safetensors','*.txt','*.model','*.py','*.tiktoken','LICENSE*','README.md']))
        for f in weights:
            file = path/f.rfilename; status('verifying', file=f.rfilename)
            if file.stat().st_size != f.size: raise ValueError('Weight size mismatch: '+f.rfilename)
            sha = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''): sha.update(block)
            if sha.hexdigest() != f.lfs.sha256: raise ValueError('Weight hash mismatch: '+f.rfilename)
        status('ready', snapshot=str(path), authenticated=True, verified_weight_files=len(weights), weight_bytes=sum(f.size for f in weights))
    except Exception as error:
        status('failed', error_type=type(error).__name__)
        raise SystemExit(1)  # Do not leak credential-bearing HTTP error details.
if __name__ == '__main__': main()
