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
import re
import sys
import time
import cwctl


def _hash_file(path):
    digest=hashlib.sha256();size=0
    with path.open('rb') as stream:
        while data:=stream.read(8<<20):digest.update(data);size+=len(data)
    return size,digest


def _content_range(value,expected_start,expected_size):
    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',value or '')
    if not match:return False
    start,end,total=map(int,match.groups())
    return start==expected_start and end==expected_size-1 and total==expected_size


def _stream_download(response,pending,expected_size,offset,digest):
    status=response.status_code
    if offset:
        if status==200:
            # Some object stores ignore Range. This body starts at byte zero,
            # so discard the old prefix before writing it.
            mode='wb';offset=0;digest=hashlib.sha256()
        elif status==206:
            if not _content_range(response.headers.get('Content-Range'),offset,expected_size):
                raise ValueError('private export range response does not match requested byte interval')
            mode='ab'
        else:
            raise RuntimeError('private export HTTP status '+str(status))
    elif status==200:
        mode='wb'
    else:
        raise RuntimeError('private export HTTP status '+str(status))

    content_length=response.headers.get('Content-Length')
    if content_length is not None:
        try:content_length=int(content_length)
        except (TypeError,ValueError):raise ValueError('private export response has an invalid content length')
        if content_length != expected_size-offset:
            raise ValueError('private export response length does not match expected byte interval')

    size=offset
    try:
        with pending.open(mode) as stream:
            try:
                for data in response.iter_content(8<<20):
                    if not data:continue
                    size+=len(data)
                    if size>expected_size:
                        raise ValueError('private export response exceeds receipt byte count')
                    stream.write(data);digest.update(data)
            finally:
                stream.flush();os.fsync(stream.fileno())
    except ValueError:
        if pending.exists() and pending.stat().st_size>expected_size:
            pending.unlink(missing_ok=True)
        raise
    return size,digest


def _publish_verified(pending,destination,identity,size,digest):
    if size!=identity['bytes']:
        raise ValueError('private export download is incomplete; verified prefix retained for resume')
    if digest.hexdigest()!=identity['sha256']:
        pending.unlink(missing_ok=True)
        raise ValueError('private export download byte/hash mismatch')
    pending.replace(destination)
    descriptor=os.open(destination.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return digest.hexdigest()


def fetch(receipt,destination,*,requests_client=None,wandb_api=None):
    identity=receipt['identity'];proof=receipt.get('independent_verification') or {}
    if (receipt.get('private') is not True or receipt.get('role') not in {'lfm','mellum'}
            or any(proof.get(key)!=identity[key] for key in ('bytes','sha256'))):
        raise ValueError('private student export lacks independent byte/hash verification')
    if type(identity.get('bytes')) is not int or identity['bytes']<0:
        raise ValueError('private student export byte count is invalid')
    if not isinstance(identity.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',identity['sha256']):
        raise ValueError('private student export SHA-256 is invalid')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_name(destination.name+'.pending')
    started=time.time();expected=identity['bytes']
    if destination.exists():
        size,digest=_hash_file(destination)
        if size!=expected or digest.hexdigest()!=identity['sha256']:
            raise ValueError('existing private export destination differs from receipt')
        return {'path':str(destination),'sha256':digest.hexdigest(),'bytes':size,
                'seconds':time.time()-started,'private':True,'resumed_bytes':size,'reused_destination':True}

    offset=0;digest=hashlib.sha256()
    if pending.exists():
        offset,digest=_hash_file(pending)
        if offset>expected or (offset==expected and digest.hexdigest()!=identity['sha256']):
            pending.unlink()
            offset=0;digest=hashlib.sha256()
        elif offset==expected:
            sha=_publish_verified(pending,destination,identity,offset,digest)
            return {'path':str(destination),'sha256':sha,'bytes':expected,
                    'seconds':time.time()-started,'private':True,'resumed_bytes':expected}

    if requests_client is None:
        import requests as requests_client
    if wandb_api is None:
        import wandb
        cwctl.load_env_file()
        wandb_api=wandb.Api()
    artifact=wandb_api.run(receipt['run']).file(receipt['name'])
    headers={'Range':f'bytes={offset}-'} if offset else {}
    with requests_client.get(artifact.url,auth=('api',os.environ['WANDB_API_KEY']),
                             headers=headers,stream=True,timeout=180) as response:
        size,digest=_stream_download(response,pending,expected,offset,digest)
    sha=_publish_verified(pending,destination,identity,size,digest)
    return {'path':str(destination),'sha256':sha,'bytes':size,
            'seconds':time.time()-started,'private':True,'resumed_bytes':offset}


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
