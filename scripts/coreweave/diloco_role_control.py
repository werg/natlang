#!/usr/bin/env python3
"""Serialize role requests against one exact persistent sandbox owner."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import time


def _ticks(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (OSError,IndexError):return None


def _owner(directory, *, require_pid1, entrypoint_sha256):
    state=json.loads((directory/'state.json').read_text())
    if state.get('schema')!='natlang.diloco-role-supervisor/1':raise ValueError('invalid role owner state')
    pid=state.get('supervisor_pid');boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if (type(pid) is not int or state.get('boot_id')!=boot or _ticks(pid)!=state.get('supervisor_start_ticks')
            or not isinstance(state.get('owner_id'),str) or not state['owner_id']):
        raise ValueError('role owner boot/PID/start identity changed')
    if require_pid1 and pid!=1:raise ValueError('warm sandbox role owner is not PID 1')
    if require_pid1 and (not isinstance(entrypoint_sha256,str) or not re.fullmatch('[0-9a-f]{64}',entrypoint_sha256)):
        raise ValueError('warm sandbox role owner needs its exact creation entrypoint hash')
    if entrypoint_sha256 is not None:
        argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if b'-c' not in argv:raise ValueError('role owner does not use its declared inline entrypoint')
        index=argv.index(b'-c')
        if index+1>=len(argv) or hashlib.sha256(argv[index+1]).hexdigest()!=entrypoint_sha256:
            raise ValueError('role owner inline entrypoint differs from its creation intent')
    return state


def submit(directory, request, *, timeout=60, require_pid1=False, entrypoint_sha256=None):
    directory=Path(directory)
    if directory.is_symlink():raise ValueError('invalid role control directory')
    directory=directory.resolve(strict=True)
    if not directory.is_dir():raise ValueError('invalid role control directory')
    if timeout<=0:raise ValueError('role control timeout must be positive')
    if (not isinstance(request,dict) or request.get('schema')!='natlang.diloco-role-request/1'
            or not isinstance(request.get('id'),str) or not re.fullmatch(r'[A-Za-z0-9-]{1,128}',request['id'])
            or type(request.get('expected_generation')) is not int or request['expected_generation']<0
            or request.get('action') not in ('start','stop') or request.get('role') not in ('worker','coordinator','actor')):
        raise ValueError('invalid role request')
    deadline=time.monotonic()+timeout
    with (directory/'publisher.lock').open('a') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('role request publisher lock timed out')
                time.sleep(.05)
        state=_owner(directory,require_pid1=require_pid1,entrypoint_sha256=entrypoint_sha256)
        if state['owner_id']!=request.get('owner_id'):
            raise ValueError('role request owner or generation is stale')
        request_hash=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        intent=directory/('publisher-'+request['id']+'.json')
        events=[event for event in state.get('events',[]) if event.get('request_id')==request['id']]
        if events:
            if not intent.is_file() or json.loads(intent.read_text())!={'request_sha256':request_hash}:
                raise ValueError('recorded role request does not have the exact publisher intent')
            if len(events)!=1:raise ValueError('role request has ambiguous history')
            event=events[0];role=state.get('roles',{}).get(request['role'],{})
            history=state.get('events',[]);position=history.index(event)
            if any(item.get('role')==request['role'] and item.get('event') in ('start','stop') for item in history[position+1:]):
                raise ValueError('recorded role request was superseded by a later operation')
            digest=hashlib.sha256(json.dumps(request.get('argv'),separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            if (event.get('event')!=request['action'] or event.get('role')!=request['role']
                    or role.get('pid')!=event.get('pid') or
                    (request['action']=='start' and (event.get('argv_sha256')!=digest or not role.get('alive') or _ticks(event['pid']) is None)) or
                    (request['action']=='stop' and event.get('pid')!=request.get('expected_pid'))):
                raise ValueError('recorded role request no longer matches its actual child')
            return {'id':request['id'],'accepted':True,'already_applied':True,'generation':state['generation'],
                    'pid':event['pid'],'boot_id':state['boot_id'],'owner_id':state['owner_id'],
                    'supervisor_pid':state['supervisor_pid'],'supervisor_start_ticks':state['supervisor_start_ticks'],
                    'process_start_ticks':_ticks(event['pid'])}
        if state.get('generation')!=request.get('expected_generation'):
            raise ValueError('role request owner or generation is stale')
        if request['action']=='stop':
            pid=request.get('expected_pid');ticks=request.get('expected_process_start_ticks')
            if (type(pid) is not int or not isinstance(ticks,str) or _ticks(pid)!=ticks
                    or state.get('roles',{}).get(request['role'],{}).get('pid')!=pid):
                raise ValueError('role stop does not bind the current child PID and start identity')
        if intent.exists():
            if json.loads(intent.read_text())!={'request_sha256':request_hash}:
                raise ValueError('role request ID was bound to a different publisher intent')
        else:
            fd=os.open(intent,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as stream:
                json.dump({'request_sha256':request_hash},stream);stream.flush();os.fsync(stream.fileno())
            parent=os.open(directory,os.O_DIRECTORY)
            try:os.fsync(parent)
            finally:os.close(parent)
        pending=directory/'request.json'
        while pending.exists():
            if time.monotonic()>=deadline:raise TimeoutError('previous role request has not been consumed')
            time.sleep(.05)
        temporary=directory/('.request-'+request['id']+'-'+str(os.getpid()))
        flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
        fd=os.open(temporary,flags,0o600)
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(request,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
            os.link(temporary,pending)
            parent=os.open(directory,os.O_DIRECTORY)
            try:os.fsync(parent)
            finally:os.close(parent)
        finally:temporary.unlink(missing_ok=True)
        while time.monotonic()<deadline:
            try:result=json.loads((directory/'result.json').read_text())
            except (OSError,ValueError):result={}
            current=_owner(directory,require_pid1=require_pid1,entrypoint_sha256=entrypoint_sha256)
            if current['owner_id']!=request['owner_id']:raise ValueError('role owner changed during request')
            if result.get('id')==request['id']:
                if result.get('accepted') is not True:raise ValueError('role owner rejected request: '+str(result))
                if result.get('generation')!=request['expected_generation']+1:
                    raise ValueError('role acknowledgement has unexpected generation')
                pid=result['pid'];ticks=_ticks(pid)
                if request['action']=='start' and ticks is None:raise ValueError('started role exited before custody')
                return {**result,'boot_id':current['boot_id'],'owner_id':current['owner_id'],
                        'supervisor_pid':current['supervisor_pid'],'supervisor_start_ticks':current['supervisor_start_ticks'],
                        'process_start_ticks':ticks}
            time.sleep(.05)
        raise TimeoutError('role owner did not acknowledge its exact request')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--request',required=True)
    parser.add_argument('--timeout',type=float,default=60);parser.add_argument('--require-pid1',action='store_true')
    parser.add_argument('--entrypoint-sha256')
    args=parser.parse_args()
    print(json.dumps(submit(args.directory,json.loads(args.request),timeout=args.timeout,
                            require_pid1=args.require_pid1,entrypoint_sha256=args.entrypoint_sha256),sort_keys=True))


if __name__=='__main__':main()
