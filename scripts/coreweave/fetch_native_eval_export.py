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
import uuid
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


def external_payload_path(payload_root,identity):
    """Return the content-addressed durable payload location for a receipt."""
    root=Path(payload_root).resolve()
    if not root.is_absolute():raise ValueError('external payload root must be absolute')
    return root/identity['sha256']/('checkpoint-'+identity['sha256']+'.pt')


def publish_payload_link(payload,destination,identity):
    """Atomically expose an already verified external payload at its logical path."""
    payload=Path(payload).resolve(strict=True);destination=Path(destination)
    size,digest=_hash_file(payload)
    if size!=identity['bytes']or digest.hexdigest()!=identity['sha256']:
        raise ValueError('external payload differs from private receipt')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink():
        if destination.resolve(strict=True)!=payload:
            raise ValueError('logical export symlink points at a different payload')
        return {'path':str(destination),'payload_path':str(payload),'sha256':digest.hexdigest(),
                'bytes':size,'reused_destination':True}
    if destination.exists():
        existing_size,existing_digest=_hash_file(destination)
        if existing_size!=size or existing_digest.hexdigest()!=digest.hexdigest():
            raise ValueError('logical export destination already contains different bytes')
        return {'path':str(destination),'payload_path':str(payload),'sha256':digest.hexdigest(),
                'bytes':size,'reused_destination':True}
    if destination.is_symlink():raise ValueError('logical export destination is a dangling symlink')
    temporary=destination.with_name(destination.name+'.link-'+str(os.getpid())+'-'+uuid.uuid4().hex)
    try:
        os.symlink(str(payload),temporary)
        os.replace(temporary,destination)
        descriptor=os.open(destination.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path':str(destination),'payload_path':str(payload),'sha256':digest.hexdigest(),'bytes':size}


def fetch(receipt,destination,*,payload_root=None,minimum_external_free_bytes=20<<30,
          requests_client=None,wandb_api=None,available=None,require_separate_filesystem=True):
    identity=receipt['identity'];proof=receipt.get('independent_verification') or {}
    if (receipt.get('private') is not True or receipt.get('role') not in {'lfm','mellum'}
            or any(proof.get(key)!=identity[key] for key in ('bytes','sha256'))):
        raise ValueError('private student export lacks independent byte/hash verification')
    if type(identity.get('bytes')) is not int or identity['bytes']<0:
        raise ValueError('private student export byte count is invalid')
    if not isinstance(identity.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',identity['sha256']):
        raise ValueError('private student export SHA-256 is invalid')
    logical_destination=Path(destination);logical_destination.parent.mkdir(parents=True,exist_ok=True)
    if type(minimum_external_free_bytes)is not int or minimum_external_free_bytes<0:
        raise ValueError('external payload free-space floor is invalid')
    if payload_root is not None and logical_destination.is_symlink() and not logical_destination.exists():
        raise ValueError('logical export destination is a dangling symlink')
    if payload_root is not None and logical_destination.is_file() and not logical_destination.is_symlink():
        size,digest=_hash_file(logical_destination)
        if size!=identity['bytes']or digest.hexdigest()!=identity['sha256']:
            raise ValueError('existing logical export differs from private receipt')
        return {'path':str(logical_destination),'sha256':digest.hexdigest(),'bytes':size,
                'seconds':0.0,'private':True,'resumed_bytes':size,'reused_destination':True,
                'external_payload':False}
    if payload_root is not None:
        if minimum_external_free_bytes<20<<30:
            raise ValueError('external payload free-space floor cannot be below 20 GiB')
        root=Path(payload_root).resolve()
        if not root.is_dir():raise ValueError('external payload root must already exist')
        if (require_separate_filesystem
                and root.stat().st_dev==logical_destination.parent.stat().st_dev):
            raise ValueError('external payload root shares the logical path filesystem')
        destination=external_payload_path(root,identity)
    else:destination=logical_destination
    if payload_root is not None and destination.is_symlink():
        raise ValueError('content-addressed external payload must be a regular file')
    destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_name(destination.name+'.pending')
    if payload_root is not None:
        legacy_pending=logical_destination.with_name(logical_destination.name+'.pending')
        legacy_control=legacy_pending.with_name(legacy_pending.name+'.aria2')
        if legacy_pending.exists() or legacy_control.exists():
            raise ValueError('logical partial/control requires an explicit controlled external migration')
        if logical_destination.is_symlink() and logical_destination.resolve(strict=True)!=destination:
            raise ValueError('logical export symlink points at a different content-addressed payload')
    started=time.time();expected=identity['bytes']
    if destination.exists():
        size,digest=_hash_file(destination)
        if size!=expected or digest.hexdigest()!=identity['sha256']:
            raise ValueError('existing private export destination differs from receipt')
        result={'path':str(destination),'sha256':digest.hexdigest(),'bytes':size,
                'seconds':time.time()-started,'private':True,'resumed_bytes':size,'reused_destination':True}
        if payload_root is not None:
            link=publish_payload_link(destination,logical_destination,identity)
            result.update(path=link['path'],payload_path=link['payload_path'])
        return result

    if payload_root is not None:
        free=(available or __import__('shutil').disk_usage)(destination.parent).free
        retained=(pending.stat().st_blocks*512 if pending.exists() else 0)
        needed=max(0,expected-retained)+minimum_external_free_bytes
        if free<needed:raise OSError('external payload root lacks transfer bytes plus the configured free-space floor')

    offset=0;digest=hashlib.sha256()
    if pending.exists():
        offset,digest=_hash_file(pending)
        if offset>expected or (offset==expected and digest.hexdigest()!=identity['sha256']):
            pending.unlink()
            offset=0;digest=hashlib.sha256()
        elif offset==expected:
            sha=_publish_verified(pending,destination,identity,offset,digest)
            result={'path':str(destination),'sha256':sha,'bytes':expected,
                    'seconds':time.time()-started,'private':True,'resumed_bytes':expected}
            if payload_root is not None:
                link=publish_payload_link(destination,logical_destination,identity)
                result.update(path=link['path'],payload_path=link['payload_path'])
            return result

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
    result={'path':str(destination),'sha256':sha,'bytes':size,
            'seconds':time.time()-started,'private':True,'resumed_bytes':offset}
    if payload_root is not None:
        link=publish_payload_link(destination,logical_destination,identity)
        result.update(path=link['path'],payload_path=link['payload_path'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--payload-root',type=Path,
                        help='optional external content-addressed storage; logical --out becomes a verified symlink')
    parser.add_argument('--minimum-external-free-bytes',type=int,default=20<<30)
    args=parser.parse_args()
    try:
        print(json.dumps(fetch(json.loads(args.receipt.read_text()),args.out,payload_root=args.payload_root,
                               minimum_external_free_bytes=args.minimum_external_free_bytes)))
    except Exception as error:
        # Provider exceptions can contain scoped URLs. Keep those out of logs.
        print(json.dumps({'event':'native_eval_export_fetch_failed','error_type':type(error).__name__}),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
