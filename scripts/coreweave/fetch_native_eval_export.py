#!/usr/bin/env python3
"""Fetch an existing verified private student export for owner-host task evaluation.

Uses the same private W&B file lookup/stream/SHA path as periodic checkpoint
verification. No sandbox resource changes, credential upload or URL output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import cwctl


def fetch(receipt,destination):
    import requests
    import wandb
    identity=receipt['identity'];proof=receipt.get('independent_verification') or {}
    if (receipt.get('private') is not True or receipt.get('role') not in {'lfm','mellum'}
            or any(proof.get(key)!=identity[key] for key in ('bytes','sha256'))):
        raise ValueError('private student export lacks independent byte/hash verification')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_name(destination.name+'.pending')
    started=time.time();digest=hashlib.sha256();size=0
    downloaded=not destination.exists()
    if not downloaded:
        with destination.open('rb') as stream:
            while data:=stream.read(8<<20):digest.update(data);size+=len(data)
    else:
        cwctl.load_env_file()
        artifact=wandb.Api().run(receipt['run']).file(receipt['name'])
        with requests.get(artifact.url,auth=('api',os.environ['WANDB_API_KEY']),stream=True,timeout=180) as response:
            if response.status_code!=200:raise RuntimeError('private export HTTP status '+str(response.status_code))
            with pending.open('wb') as stream:
                for data in response.iter_content(8<<20):stream.write(data);digest.update(data);size+=len(data)
                stream.flush();os.fsync(stream.fileno())
    if size!=identity['bytes'] or digest.hexdigest()!=identity['sha256']:
        pending.unlink(missing_ok=True)
        raise ValueError('private export download byte/hash mismatch')
    if downloaded:
        pending.replace(destination)
        descriptor=os.open(destination.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    else:pending.unlink(missing_ok=True)
    return {'path':str(destination),'sha256':digest.hexdigest(),'bytes':size,
            'seconds':time.time()-started,'private':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(fetch(json.loads(args.receipt.read_text()),args.out)))
    except Exception as error:
        # Provider exceptions can contain scoped URLs. Keep those out of logs.
        print(json.dumps({'event':'native_eval_export_fetch_failed','error_type':type(error).__name__}),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
