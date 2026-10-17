"""Publish cursor-produced inference views using an existing private export adapter."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import time


class ViewPublicationError(ValueError):
    """A verifier rejection whose literal message is safe to persist."""


def save(path, value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.owned-'+str(os.getpid()))
    with tmp.open('w')as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def full_identity(state,role,sandbox_id):
    receipt=state['receipt'];identity=receipt['identity'];verified=receipt['independent_verification']
    if (receipt.get('role')!=role or receipt.get('sandbox_id')!=sandbox_id or receipt.get('private')is not True
            or re.fullmatch('[0-9a-f]{64}',identity['sha256'])is None
            or type(identity['bytes'])is not int or identity['bytes']<=0
            or verified.get('sha256')!=identity['sha256']or verified.get('bytes')!=identity['bytes']
            or type(state.get('last_verified_step'))is not int or state['last_verified_step']<0):
        raise ViewPublicationError('actor view requires an independently verified full training backup')
    return identity


def verified_actor_view(reference,full,role,producer_sha256,pin_directory):
    """Runs on the source host; no source training payload or model is opened."""
    proof_path=pathlib.Path(reference['path']).resolve(strict=True)
    raw=proof_path.read_bytes()
    if len(raw)!=reference['bytes']or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ViewPublicationError('actor view proof changed')
    proof=json.loads(raw)
    if (proof.get('schema')!='natlang.native-actor-view/1' or proof.get('step')!=full['step']
            or proof.get('source',{}).get('sha256')!=full['sha256']
            or proof.get('source',{}).get('bytes')!=full['bytes']
            or proof.get('mechanism',{}).get('sha256')!=producer_sha256):
        raise ViewPublicationError('actor view producer or full-parent identity differs')
    view=proof['view'];relative=pathlib.Path(view['path'])
    if relative.is_absolute()or '..'in relative.parts:raise ViewPublicationError('actor view path must stay beside its proof')
    path=(proof_path.parent/relative).resolve(strict=True)
    if not path.is_relative_to(proof_path.parent)or not path.is_file():raise ViewPublicationError('actor view escaped its proof root')
    if (type(view['bytes'])is not int or not 0<view['bytes']<32*1024**3
            or path.stat().st_size!=view['bytes'] or re.fullmatch('[0-9a-f]{64}',view['sha256'])is None):
        raise ViewPublicationError('actor view size or identity is invalid')
    with path.open('rb')as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=view['sha256']:raise ViewPublicationError('actor view content changed')
    pin_directory=pathlib.Path(pin_directory).resolve(strict=True)
    alias=pin_directory/(role+'-native-inference-step-'+str(full['step'])+'.pt')
    try:os.link(path,alias)
    except FileExistsError:
        if alias.is_symlink()or alias.resolve().parent!=pin_directory:raise ViewPublicationError('actor view alias escaped its pin root')
        with alias.open('rb')as stream:existing=hashlib.file_digest(stream,'sha256').hexdigest()
        if alias.stat().st_size!=view['bytes']or existing!=actual:raise ViewPublicationError('immutable actor view alias conflicts')
    fd=os.open(pin_directory,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return {'step':full['step'],'source':str(alias),'identity':{'sha256':actual,'bytes':view['bytes']},
            'actor_view':reference,'full_training_checkpoint':full,'producer_sha256':producer_sha256}


def run_once(directory,role,sandbox_id,full_state_path,view_state_path,*,observe,export):
    directory=pathlib.Path(directory);directory.mkdir(parents=True,exist_ok=True)
    pending=directory/(role+'-actor-view-export-pending.json')
    view_state_path=pathlib.Path(view_state_path)
    previous=json.loads(view_state_path.read_text())if view_state_path.exists()else{}
    if pending.exists():chosen=json.loads(pending.read_text())
    else:
        full_state=json.loads(pathlib.Path(full_state_path).read_text())
        identity=full_identity(full_state,role,sandbox_id)
        step=full_state['last_verified_step']
        if previous.get('last_verified_step',-1)>=step or not full_state.get('actor_view'):return None
        chosen=observe(full_state['actor_view'],{'step':step,**identity})
        if chosen['step']!=step or chosen['full_training_checkpoint']!={'step':step,**identity}:
            raise ViewPublicationError('observed view changed its verified full-parent identity')
        chosen['full_private_receipt']=full_state['receipt'];save(pending,chosen)
    # Recheck the captured full backup even on a recovered pending operation.
    full=full_identity({'receipt':chosen['full_private_receipt'],'last_verified_step':chosen['step']},role,sandbox_id)
    if chosen['full_training_checkpoint']!={'step':chosen['step'],**full}:raise ViewPublicationError('pending actor view full-parent identity differs')
    # The adapter independently rechecks the immutable physical view.
    receipt=export(chosen)
    verification=receipt.get('independent_verification',{})
    if (receipt.get('private')is not True or receipt.get('role')!=role or receipt.get('sandbox_id')!=sandbox_id
            or receipt.get('source')!=chosen['source']or receipt.get('identity')!=chosen['identity']
            or verification.get('sha256')!=chosen['identity']['sha256']
            or verification.get('bytes')!=chosen['identity']['bytes']):
        raise ViewPublicationError('actor view lacks exact independent private full GET')
    current=json.loads(view_state_path.read_text())if view_state_path.exists()else{}
    if current.get('last_verified_step',-1)==chosen['step']and current['receipt']['identity']!=chosen['identity']:
        raise ViewPublicationError('published actor view step conflicts')
    result=None
    if current.get('last_verified_step',-1)<chosen['step']:
        full=chosen['full_training_checkpoint']
        result={'at':time.time(),'last_verified_step':chosen['step'],'receipt':receipt,
                'inference_source':{'schema':'natlang.native-inference-source/1','training_checkpoint_sha256':full['sha256'],
                    'training_checkpoint_bytes':full['bytes'],'step':full['step']},
                'derivation':{'actor_view':chosen['actor_view'],'producer_sha256':chosen['producer_sha256'],
                    'full_private_receipt':chosen['full_private_receipt'],'complete_training_parent_retained':True,
                    'training_resume':False}}
        save(view_state_path,result)
    pending.unlink();fd=os.open(directory,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['lfm','mellum'],required=True);p.add_argument('--sandbox-id',required=True)
    p.add_argument('--full-state',type=pathlib.Path,required=True);p.add_argument('--view-state',type=pathlib.Path,required=True)
    p.add_argument('--state-dir',type=pathlib.Path,required=True);p.add_argument('--producer-sha256',required=True)
    p.add_argument('--pin-directory',default='/work/production/durable-exports');p.add_argument('--export-command',required=True)
    p.add_argument('--interval',type=float,default=30);p.add_argument('--once',action='store_true');args=p.parse_args(argv)
    command=json.loads(args.export_command)
    if not isinstance(command,list)or not command or any(not isinstance(x,str)for x in command):p.error('export command needs explicit argv')
    if re.fullmatch('[0-9a-f]{64}',args.producer_sha256)is None or args.interval<=0:p.error('invalid producer identity or interval')
    args.state_dir.mkdir(parents=True,exist_ok=True)
    lease=(args.state_dir/(args.role+'-actor-view-publication-owner.lock')).open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:from . import cwctl
    except ImportError:import cwctl
    cwctl.load_env_file();cwctl.require_key()
    def observe(reference,full):
        code='import pathlib,json,hashlib,os,re\n'+inspect.getsource(ViewPublicationError)+'\n'+inspect.getsource(verified_actor_view)
        code+='\ntry:\n print(json.dumps(verified_actor_view('+','.join(repr(x)for x in(reference,full,args.role,args.producer_sha256,args.pin_directory))+')))\nexcept ViewPublicationError as error:\n print(json.dumps({"verifier_rejection":str(error)}));raise SystemExit(23)'
        result=cwctl.read_sandbox(args.sandbox_id,patience=60).exec(['nice','-n','19','python','-c',code],timeout_seconds=600).result(timeout=690)
        if result.returncode:
            try:rejection=json.loads(result.stdout).get('verifier_rejection')
            except (ValueError,AttributeError):rejection=None
            if result.returncode==23 and isinstance(rejection,str):raise ViewPublicationError(rejection)
            error=RuntimeError('source view verification subprocess failed');error.source_returncode=result.returncode;raise error
        return json.loads(result.stdout)
    def export(chosen):
        receipt=args.state_dir/(args.role+'-actor-view-current-private-receipt.json')
        actual=[x.replace('{role}',args.role).replace('{pin}',chosen['source'])for x in command]
        subprocess.run(actual,check=True,env={**os.environ,'CW_EXPORT_RECEIPT_FILE':str(receipt),'CW_EXPORT_SANDBOX_ID':args.sandbox_id,
            'CW_EXPORT_EXPECTED_SHA256':chosen['identity']['sha256'],'CW_EXPORT_EXPECTED_BYTES':str(chosen['identity']['bytes']),
            'CW_EXPORT_REUSE_CONTENT_ALIAS':'1','CW_EXPORT_VERIFY_REMOTE':'1'})
        return json.loads(receipt.read_text())
    while True:
        try:
            result=run_once(args.state_dir,args.role,args.sandbox_id,args.full_state,args.view_state,observe=observe,export=export)
            if result:print(json.dumps({'at':time.time(),'verified_step':result['last_verified_step'],'physical_view_sha256':result['receipt']['identity']['sha256'],'source_training_checkpoint_sha256':result['inference_source']['training_checkpoint_sha256']}),flush=True)
            if args.once:return 0
            time.sleep(args.interval)
        except Exception as error:
            failure={'at':time.time(),'type':type(error).__name__,'error':'view publication retry; full training backup and learner remain independent'}
            if isinstance(error,ViewPublicationError):failure['verifier_reason']=str(error)
            if isinstance(error,subprocess.CalledProcessError):failure['subprocess_returncode']=error.returncode
            if isinstance(getattr(error,'source_returncode',None),int):failure['source_returncode']=error.source_returncode
            save(args.state_dir/(args.role+'-actor-view-last-error.json'),failure)
            print(json.dumps({'at':time.time(),'retry':type(error).__name__}),flush=True)
            if args.once:raise
            time.sleep(min(60,args.interval))


if __name__=='__main__':raise SystemExit(main())
