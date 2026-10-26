#!/usr/bin/env python3
"""Prepare and durably index exact shared participant recovery artifacts.

The global checkpoint and selected participant closure remain in their native
sharded format. A caller supplies the existing private uploader as a callback;
each entry is indexed only after its exact bytes have a full independent GET
receipt. The inventory is append-only by content identity and can be read by
the local/cloud source transport while training continues.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.model.native_bundle import export_bundle, verify_bundle
try:
    from .private_export_guard import validate_receipt
except ImportError:
    from private_export_guard import validate_receipt

INVENTORY_SCHEMA='natlang.native-private-export-inventory/1'
EXPORT_SCHEMA='natlang.native-shared-participant-export/1'


def _binding(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}


def prepare_bundle(checkpoint, participant, out, *, semantic_adapter=None):
    """Export the selected actual participant plus shared/global dependencies."""
    if not isinstance(participant,str) or not participant:
        raise ValueError('shared participant export needs its exact model ID')
    checkpoint=Path(checkpoint).resolve(strict=True)
    source_before=_binding(checkpoint)
    from natlang_neuralese.model.native_shared_checkpoint import shared_manifest
    shared=shared_manifest(checkpoint)
    if shared is not None and semantic_adapter is None:
        raise ValueError('shared participant recovery must retain its exact semantic adapter/control')
    out=Path(out).resolve()
    bundle=export_bundle(checkpoint,out,semantic_adapter=semantic_adapter,participant=participant)
    verified=verify_bundle(out)
    if verified.get('participant')!=participant:
        raise ValueError('portable shared bundle selected another optimizer participant')
    files=[]
    for path in sorted(out.rglob('*')):
        if path.is_symlink():raise ValueError('shared recovery bundle may not contain symlinks')
        if path.is_file():files.append(_binding(path))
    if not files or not any(row['path']==str((out/'bundle.json').resolve()) for row in files):
        raise ValueError('selected shared recovery bundle has no exact portable manifest')
    source_config=verified['checkpoint']
    source_after=_binding(checkpoint)
    if source_after!=source_before or source_config['sha256']!=source_before['sha256']:
        raise ValueError('shared checkpoint changed while its participant closure was being prepared')
    return {'schema':EXPORT_SCHEMA,'participant':participant,
        'checkpoint':source_config,'source_checkpoint':_binding(checkpoint),
        'semantic_adapter':bundle.get('semantic_adapter'),'bundle':_binding(out/'bundle.json'),
        'bundle_root':str(out),'hf_base':'separately pinned; exact base inventory remains in model_config',
        'files':files,'training_admission':False,'training_credit':0}


def prepare_current_from_owner(state_path, participant, out_root):
    """Resolve the live owner checkpoint/semantic route at call time, without a future SHA."""
    state_path=Path(state_path).resolve(strict=True)
    before=(state_path.stat().st_dev,state_path.stat().st_ino,state_path.stat().st_size,
        state_path.stat().st_mtime_ns,state_path.stat().st_ctime_ns)
    state_bytes=state_path.read_bytes()
    state=json.loads(state_bytes)
    read_stat=state_path.stat()
    read_identity=(read_stat.st_dev,read_stat.st_ino,read_stat.st_size,read_stat.st_mtime_ns,read_stat.st_ctime_ns)
    if before!=read_identity or len(state_bytes)!=before[2]:
        raise ValueError('owner checkpoint pointer changed while reading its source state')
    checkpoint_value=state.get('checkpoint')
    checkpoint_path=(checkpoint_value.get('path')if isinstance(checkpoint_value,dict)else checkpoint_value)
    if not isinstance(checkpoint_path,str):
        raise ValueError('current owner state has no exact full-state checkpoint path')
    checkpoint=Path(checkpoint_path).resolve(strict=True)
    semantic_value=state.get('semantic_adapter')
    semantic_adapter=None
    if isinstance(semantic_value,dict):
        semantic_adapter=Path(semantic_value['path']).resolve(strict=True)
        if _binding(semantic_adapter)['sha256']!=semantic_value.get('sha256'):
            raise ValueError('current owner semantic adapter changed its exact state binding')
    elif isinstance(semantic_value,str):
        semantic_adapter=Path(semantic_value).resolve(strict=True)
    out_root=Path(out_root).resolve()
    checkpoint_sha=_binding(checkpoint)['sha256']
    destination=out_root/participant/checkpoint_sha
    prepared=prepare_bundle(checkpoint,participant,destination,semantic_adapter=semantic_adapter)
    state_after=state_path.stat()
    after=(state_after.st_dev,state_after.st_ino,state_after.st_size,state_after.st_mtime_ns,state_after.st_ctime_ns)
    if before!=after:
        raise ValueError('owner checkpoint pointer changed while preparing shared recovery bundle')
    prepared['owner_state']={'path':str(state_path),'sha256':hashlib.sha256(state_bytes).hexdigest(),
        'checkpoint':prepared['source_checkpoint']}
    return prepared


def _load_inventory(path, role):
    path=Path(path)
    if not path.exists():
        return {'schema':INVENTORY_SCHEMA,'role':role,'entries':[]}
    if path.is_symlink() or not path.is_file():
        raise ValueError('private export inventory must be a regular file')
    value=json.loads(path.read_text())
    if (value.get('schema')!=INVENTORY_SCHEMA or value.get('role')!=role
            or not isinstance(value.get('entries'),list)):
        raise ValueError('private export inventory has another schema or owner')
    return value


def _write_inventory(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    fd,pending=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb')as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        if path.is_symlink():raise ValueError('private export inventory may not be a symlink')
        os.replace(pending,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(pending):os.unlink(pending)


def _prepared_file(prepared, source_path):
    source=Path(source_path).resolve(strict=True)
    matches=[row for row in prepared.get('files',[])if Path(row.get('path','')).resolve()==source]
    if len(matches)!=1:
        raise ValueError('private receipt source is not exactly one member of this selected participant bundle')
    expected=matches[0];actual=_binding(source)
    if any(actual.get(key)!=expected.get(key)for key in ('sha256','bytes')):
        raise ValueError('shared bundle member changed before receipt registration')
    return source,actual


def record_private_receipt(prepared, inventory_path, *, role, source_path, receipt):
    """Append one selected bundle member after its exact independent private GET."""
    if prepared.get('schema')!=EXPORT_SCHEMA:
        raise ValueError('private receipt registration needs a selected shared bundle manifest')
    source,actual=_prepared_file(prepared,source_path)
    validate_receipt(receipt,actual)
    if receipt.get('source')!=str(source) or receipt.get('role')!=role:
        raise ValueError('private receipt names another shared bundle member')
    path=Path(inventory_path)
    with (path.with_name(path.name+'.lock')).open('a+b')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        inventory=_load_inventory(path,role)
        key=(actual['sha256'],actual['bytes'],str(source))
        for row in inventory['entries']:
            row_key=(row.get('identity',{}).get('sha256'),row.get('identity',{}).get('bytes'),
                row.get('source_path'))
            if row_key==key:
                validate_receipt(row.get('receipt'),actual)
                if row['receipt']!=receipt:
                    raise ValueError('shared bundle member already has a different private receipt')
                return row
        row={'source_path':str(source),'identity':{'sha256':actual['sha256'],'bytes':actual['bytes']},
            'receipt':receipt,'participant':prepared['participant'],
            'bundle_checkpoint_sha256':prepared['checkpoint']['sha256']}
        inventory['entries'].append(row);_write_inventory(path,inventory)
        return row


def pending_private_exports(prepared, inventory_path, *, role):
    """Return exact bundle members awaiting private readback without future SHA inputs."""
    inventory=_load_inventory(inventory_path,role)
    known={(row.get('identity',{}).get('sha256'),row.get('identity',{}).get('bytes'),row.get('source_path'))
        for row in inventory['entries']}
    pending=[]
    for reference in prepared.get('files',[]):
        source,actual=_prepared_file(prepared,reference['path'])
        if (actual['sha256'],actual['bytes'],str(source))not in known:pending.append(actual)
    return pending


def export_remote_bundle_files(prepared, inventory_path, *, role, export_file):
    """Record a sandbox-owned bundle without downloading its tensors to the host.

    The source owner's ``prepare-current`` command has already run native_bundle's
    complete verifier and emitted every exact member binding. The existing
    sandbox-aware exporter rehashes each remote member against that binding and
    independently GET-verifies the private object before this append-only index
    advances.
    """
    if (prepared.get('schema')!=EXPORT_SCHEMA or prepared.get('training_credit')!=0
            or prepared.get('training_admission') is not False
            or not isinstance(prepared.get('participant'),str)):
        raise ValueError('remote shared participant export lacks its exact custody envelope')
    root=prepared.get('bundle_root')
    if not isinstance(root,str) or not Path(root).is_absolute() or '..' in Path(root).parts:
        raise ValueError('remote shared bundle root must be an exact absolute path')
    root=Path(root)
    files=prepared.get('files')
    if not isinstance(files,list) or not files:
        raise ValueError('remote shared bundle has no exact member inventory')
    refs={}
    for reference in files:
        if (not isinstance(reference,dict) or not isinstance(reference.get('path'),str)
                or not Path(reference['path']).is_absolute() or '..' in Path(reference['path']).parts
                or root not in Path(reference['path']).parents
                or type(reference.get('bytes')) is not int or reference['bytes']<=0
                or not isinstance(reference.get('sha256'),str) or len(reference['sha256'])!=64):
            raise ValueError('remote shared bundle member has an invalid absolute path or identity')
        if reference['path'] in refs:
            raise ValueError('remote shared bundle repeats one member path')
        refs[reference['path']]=reference
    bundle=prepared.get('bundle')
    if (not isinstance(bundle,dict) or bundle.get('path') not in refs
            or refs[bundle['path']].get('sha256')!=bundle.get('sha256')):
        raise ValueError('remote shared bundle does not bind its exact native manifest member')
    inventory_path=Path(inventory_path)
    with (inventory_path.with_name(inventory_path.name+'.lock')).open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        inventory=_load_inventory(inventory_path,role)
        identities={(row.get('identity',{}).get('sha256'),row.get('identity',{}).get('bytes'),
            row.get('source_path')):row for row in inventory['entries']}
        completed=[]
        for path,reference in refs.items():
            identity={'sha256':reference['sha256'],'bytes':reference['bytes']}
            key=(identity['sha256'],identity['bytes'],path)
            existing=identities.get(key)
            if existing is not None:
                validate_receipt(existing.get('receipt'),identity)
                if existing['receipt'].get('source')!=path or existing['receipt'].get('role')!=role:
                    raise ValueError('existing remote private receipt names another role or source')
                completed.append(existing);continue
            if any(row.get('source_path')==path for row in inventory['entries']):
                raise ValueError('remote shared bundle member path was reused for different content')
            receipt=export_file(path,identity)
            validate_receipt(receipt,identity)
            if receipt.get('source')!=path or receipt.get('role')!=role:
                raise ValueError('remote private receipt names another role or source')
            row={'source_path':path,'identity':identity,'receipt':receipt,
                'participant':prepared['participant'],'bundle_checkpoint_sha256':prepared['checkpoint']['sha256']}
            inventory['entries'].append(row);identities[key]=row;completed.append(row)
            _write_inventory(inventory_path,inventory)
        return {'schema':INVENTORY_SCHEMA,'role':role,'entries':completed,
            'participant':prepared['participant'],'bundle_checkpoint_sha256':prepared['checkpoint']['sha256'],
            'all_bundle_files_independently_verified':True,'training_credit':0}


def export_bundle_files(prepared, inventory_path, *, role, export_file):
    """Export missing bundle members and atomically append each verified receipt.

    ``export_file(path, exact_identity)`` is the existing private uploader
    adapter. It must return its normal receipt after complete independent GET;
    this function never treats an upload acknowledgement as custody.
    """
    if (prepared.get('schema')!=EXPORT_SCHEMA or prepared.get('training_credit')!=0
            or prepared.get('training_admission') is not False
            or not isinstance(prepared.get('participant'),str)):
        raise ValueError('shared participant export lacks its exact custody envelope')
    bundle=verify_bundle(prepared['bundle_root'])
    if (bundle.get('participant')!=prepared['participant']
            or _binding(Path(prepared['bundle']['path']))['sha256']!=prepared['bundle']['sha256']):
        raise ValueError('shared participant bundle changed after preparation')
    path=Path(inventory_path)
    with (path.with_name(path.name+'.lock')).open('a+b')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        inventory=_load_inventory(path,role)
        identities={(row.get('identity',{}).get('sha256'),row.get('identity',{}).get('bytes'),
            row.get('source_path')):row for row in inventory['entries']}
        completed=[]
        for reference in prepared['files']:
            source=Path(reference['path']).resolve(strict=True)
            before=source.stat()
            actual=_binding(source)
            if any(actual.get(key)!=reference.get(key)for key in ('sha256','bytes')):
                raise ValueError('shared participant bundle member changed before private export')
            key=(actual['sha256'],actual['bytes'],str(source))
            existing=identities.get(key)
            if existing is not None:
                validate_receipt(existing.get('receipt'),actual)
                if existing['receipt'].get('role')!=role or existing['receipt'].get('source')!=str(source):
                    raise ValueError('existing private receipt names another role or bundle member')
                completed.append(existing)
                continue
            receipt=export_file(str(source),{'sha256':actual['sha256'],'bytes':actual['bytes']})
            validate_receipt(receipt,actual)
            if receipt.get('source')!=str(source) or receipt.get('role')!=role:
                raise ValueError('private receipt names another shared bundle source')
            after=source.stat()
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
                    after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('shared bundle member changed during private export')
            row={'source_path':str(source),'identity':{'sha256':actual['sha256'],'bytes':actual['bytes']},
                'receipt':receipt,'participant':prepared['participant'],
                'bundle_checkpoint_sha256':prepared['checkpoint']['sha256']}
            inventory['entries'].append(row);identities[key]=row;completed.append(row)
            _write_inventory(path,inventory)
        return {'schema':INVENTORY_SCHEMA,'role':role,'entries':completed,
            'participant':prepared['participant'],'bundle_checkpoint_sha256':prepared['checkpoint']['sha256'],
            'all_bundle_files_independently_verified':True,'training_credit':0}


def export_with_runfile_exporter(prepared, inventory_path, *, role, exporter,
                                 source_mode, sandbox_id=None):
    """Invoke the existing owner-host exporter for every missing closure file.

    ``source_mode=local`` reads the DGX-owned file directly. ``source_mode=sandbox``
    asks the existing exporter to read the exact source path from the named
    CoreWeave sandbox through its already scoped SDK path. Credentials stay in
    the owner-host process; none are added to prepared bundle metadata.
    """
    exporter=Path(exporter).resolve(strict=True)
    if not exporter.is_file() or source_mode not in ('local','sandbox'):
        raise ValueError('private runfile export needs an exact exporter and source mode')
    if source_mode=='sandbox' and (not isinstance(sandbox_id,str) or not sandbox_id):
        raise ValueError('sandbox source export needs its exact sandbox identity')
    if source_mode=='local' and sandbox_id is not None:
        raise ValueError('local source export may not declare a sandbox')

    def upload(source, identity):
        receipt_dir=Path(inventory_path).resolve().parent/'.private-receipts'
        receipt_dir.mkdir(parents=True,exist_ok=True)
        fd,receipt_name=tempfile.mkstemp(prefix=identity['sha256']+'-',suffix='.json',dir=receipt_dir)
        os.close(fd);receipt_path=Path(receipt_name)
        env=dict(os.environ)
        env['CW_EXPORT_LOCAL']='1' if source_mode=='local' else '0'
        env['CW_EXPORT_EXPECTED_SHA256']=identity['sha256']
        env['CW_EXPORT_EXPECTED_BYTES']=str(identity['bytes'])
        env['CW_EXPORT_RECEIPT_FILE']=str(receipt_path)
        if source_mode=='sandbox':env['CW_EXPORT_SANDBOX_ID']=sandbox_id
        else:env.pop('CW_EXPORT_SANDBOX_ID',None)
        try:
            subprocess.run([sys.executable,str(exporter),role,source],check=True,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            receipt=json.loads(receipt_path.read_text())
            validate_receipt(receipt,identity)
            if receipt.get('source')!=source or receipt.get('role')!=role:
                raise ValueError('existing private exporter receipt names another role or source')
            return receipt
        finally:
            receipt_path.unlink(missing_ok=True)
            receipt_path.with_suffix('.phase.json').unlink(missing_ok=True)

    if source_mode=='sandbox':
        return export_remote_bundle_files(prepared,inventory_path,role=role,export_file=upload)
    return export_bundle_files(prepared,inventory_path,role=role,export_file=upload)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest='action',required=True)
    prepare=actions.add_parser('prepare')
    prepare.add_argument('--checkpoint',required=True);prepare.add_argument('--participant',required=True)
    prepare.add_argument('--out',required=True);prepare.add_argument('--manifest',required=True)
    prepare.add_argument('--semantic-adapter')
    current=actions.add_parser('prepare-current')
    current.add_argument('--state',required=True);current.add_argument('--participant',required=True)
    current.add_argument('--out-root',required=True);current.add_argument('--manifest',required=True)
    pending=actions.add_parser('pending')
    pending.add_argument('--manifest',required=True);pending.add_argument('--inventory',required=True)
    pending.add_argument('--role',choices=('lfm','mellum'),required=True)
    record=actions.add_parser('record')
    record.add_argument('--manifest',required=True);record.add_argument('--inventory',required=True)
    record.add_argument('--role',choices=('lfm','mellum'),required=True)
    record.add_argument('--source',required=True);record.add_argument('--receipt',required=True)
    export=actions.add_parser('export')
    export.add_argument('--manifest',required=True);export.add_argument('--inventory',required=True)
    export.add_argument('--role',choices=('lfm','mellum'),required=True)
    export.add_argument('--exporter',required=True)
    export.add_argument('--source-mode',choices=('local','sandbox'),required=True)
    export.add_argument('--sandbox-id')
    args=parser.parse_args(argv)
    if args.action in ('prepare','prepare-current'):
        value=(prepare_bundle(args.checkpoint,args.participant,args.out,semantic_adapter=args.semantic_adapter)
            if args.action=='prepare' else prepare_current_from_owner(args.state,args.participant,args.out_root))
        write_json_durable(Path(args.manifest),value)
    elif args.action=='pending':
        value=pending_private_exports(json.loads(Path(args.manifest).read_text()),args.inventory,role=args.role)
    elif args.action=='export':
        value=export_with_runfile_exporter(json.loads(Path(args.manifest).read_text()),args.inventory,
            role=args.role,exporter=args.exporter,source_mode=args.source_mode,sandbox_id=args.sandbox_id)
    else:
        value=record_private_receipt(json.loads(Path(args.manifest).read_text()),args.inventory,
            role=args.role,source_path=args.source,receipt=json.loads(Path(args.receipt).read_text()))
    print(json.dumps(value,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
