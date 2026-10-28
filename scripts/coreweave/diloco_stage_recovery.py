#!/usr/bin/env python3
"""Stage an independently verified DiLoCo topology for explicit restore; never launch a sandbox."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.coreweave.private_export_guard import validate_receipt


def _check(path, identity):
    path=Path(path)
    if not path.is_file() or path.stat().st_size!=identity['bytes'] or sha256_file_hex(path,drop_cache=True)!=identity['sha256']:
        raise ValueError('local recovery component differs from its exact identity: '+str(path))
    return path


def _extract(archive,destination,*,prefix):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        seen=set()
        for member in tar:
            name=PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or member.name in seen or
                    not (member.isfile() or member.isdir()) or not name.parts or
                    (name.parts[0]!=prefix and not (prefix=='objects' and member.name=='dependency-index.json'))):
                raise ValueError('unsafe recovery archive member')
            seen.add(member.name)
            tar.extract(member,destination,filter='data')
            if prefix=='objects' and member.isfile() and name.parts[0]=='objects':
                if len(name.parts)!=2 or sha256_file_hex(destination/member.name,drop_cache=True)!=name.parts[1]:
                    raise ValueError('staged dependency object differs from its name')


def archived_direct_control(archive,control_sha,run_id):
    """Read exact controls from the authenticated dependency archive, not live host paths."""
    from scripts.coreweave.diloco_direct_recovery import record_from_control,load_control
    with tarfile.open(archive,'r') as tar:
        names=tar.getnames()
        if len(names)!=len(set(names)):raise ValueError('duplicate recovery archive members')
        index=json.load(tar.extractfile('dependency-index.json'))
        matches=[x for x in index['bindings'] if x.get('sha256')==control_sha and x.get('object')=='objects/'+control_sha]
        if len(matches)!=1:raise ValueError('sealed dependencies lack the exact direct control')
        def read_object(sha):
            member=tar.getmember('objects/'+sha)
            if not member.isfile():raise ValueError('direct control archive member is not a file')
            data=tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('archived direct control object differs')
            return data
        raw=read_object(control_sha)
        def read_bound(item):
            if not any(row.get('sha256')==item['sha256'] and row.get('object')=='objects/'+item['sha256']
                       and row.get('path')==item['path'] for row in index['bindings']):
                raise ValueError('direct control binding is absent from dependency archive')
            return read_object(item['sha256'])
        logical=matches[0]['path']
        record=record_from_control(logical,run_id,control_bytes=raw,read_bound=read_bound,verify_helpers=False)
        return load_control(logical,record,control_bytes=raw,read_bound=read_bound,verify_helpers=False)


def archived_binding(archive,binding):
    with tarfile.open(archive,'r') as tar:
        index=json.load(tar.extractfile('dependency-index.json'));sha=binding['sha256']
        if not any(x.get('path')==binding['path'] and x.get('sha256')==sha and x.get('object')=='objects/'+sha
                   for x in index['bindings']):raise ValueError('required recovery proof missing from archive')
        member=tar.getmember('objects/'+sha)
        if not member.isfile():raise ValueError('recovery proof is not a file')
        raw=tar.extractfile(member).read()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('archived recovery proof differs')
        return json.loads(raw)


def stage(state_path,cycle_id,storage,parent_pin,parent_receipt,parent_local,target,*,present,reserve_bytes=80*2**30,external_required=True,inputs=None):
    state=json.loads(Path(state_path).read_text());cycles=[item for item in state['cycles'] if item['id']==cycle_id]
    if len(cycles)!=1:raise ValueError('recovery cycle is absent or duplicated')
    cycle=cycles[0]
    if cycle.get('phase')!='verified-components-and-dependencies' or not cycle.get('compatibility',{}).get('compatible'):
        raise ValueError('recovery cycle is not a compatible exported cohort')
    components=cycle['components'];receipts=cycle['receipts'];storage=Path(storage);target=Path(target)
    from scripts.coreweave.diloco_periodic_recovery import compatibility
    dependency=cycle['dependency_intent'];dependency_identity={'sha256':dependency['sha256'],'bytes':dependency['bytes']}
    validate_receipt(cycle['dependency_receipt'],dependency_identity)
    if not cycle['dependency_receipt'].get('verified_at') or not present(cycle['dependency_receipt']):raise ValueError('private dependency archive is absent')
    overrides=inputs or {};dependency_path=_check(overrides.get('dependencies',dependency['path']),dependency_identity)
    direct=None
    contract=state.get('recovery_contract') or {}
    if contract.get('mode')=='direct':
        direct=archived_direct_control(dependency_path,contract['control_sha256'],state['coordinator_run_id'])
        if direct['control_sha256']!=contract['control_sha256']:raise ValueError('direct recovery control does not match exported cycle')
    if not compatibility(components,direct=direct)['compatible']:raise ValueError('recovery sequence or component identities no longer match')
    names={'coordinator',*(f'worker-{i}' for i in range(4))}
    if not direct:names.update(f'resident-{i}' for i in range(4))
    if set(components)!=names or set(receipts)!=names:raise ValueError('topology components are incomplete')
    if external_required:
        from natlang_neuralese.common import paths
        external=paths.root('data_hdd').resolve()
        if not external.is_mount() or not target.resolve().is_relative_to(external):
            raise ValueError('restore staging must use the mounted external data root')
    if direct:
        refs=cycle.get('parent_reference') or {};transfer=direct.get('parent_transfer') or {}
        if (refs.get('pin')!={'path':transfer.get('pin'),'sha256':transfer.get('pin_sha256')} or
                refs.get('receipt')!={'path':transfer.get('receipt'),'sha256':transfer.get('receipt_sha256')}):
            raise ValueError('direct parent proof differs from the sealed control')
        pin=archived_binding(dependency_path,refs['pin']);old_receipt=archived_binding(dependency_path,refs['receipt'])
        qualification=archived_binding(dependency_path,refs['qualification']);copy=archived_binding(dependency_path,refs['copy_receipt'])
    else:
        if not parent_pin or not parent_receipt:raise ValueError('resident recovery requires parent pin and receipt')
        pin=json.loads(Path(parent_pin).read_text());old_receipt=json.loads(Path(parent_receipt).read_text())
    parent_identity={'sha256':pin['sha256'],'bytes':pin['bytes']}
    if direct and (qualification.get('schema')!='natlang.lfm3-seed-independent-full-state/1' or
        qualification.get('passed') is not True or qualification.get('global_step')!=pin['global_step'] or
        any(qualification.get(k)!=v or copy.get(k)!=v for k,v in parent_identity.items()) or
        (copy.get('local_copy') or {}).get('committed') is not True or
        any((copy.get('local_copy') or {}).get(k)!=v for k,v in parent_identity.items())):
        raise ValueError('archived parent full-state or retained-copy proof differs')
    manifest=components.get('coordinator',{}).get('manifest') or {}
    if manifest.get('run_id')!=state.get('coordinator_run_id') or (manifest.get('identity') or {}).get('parent_checkpoint_sha256')!=pin['sha256']:
        raise ValueError('original parent does not match coordinator identity')
    validate_receipt(old_receipt,parent_identity)
    if not old_receipt.get('verified_at') or not present(old_receipt):raise ValueError('private original parent is absent')
    _check(parent_local,parent_identity)
    inputs={}
    for name in sorted(names):
        intent=components[name];identity={'sha256':intent['sha256'],'bytes':intent['bytes']}
        validate_receipt(receipts[name],identity)
        if not receipts[name].get('verified_at') or not present(receipts[name]):raise ValueError('private recovery component is absent: '+name)
        root=Path((direct['component_placement'].get(name) or {}).get('root',str(storage))).resolve() if direct else storage
        expected=root/'objects'/identity['sha256']/Path(intent['path']).name
        if direct and (cycle.get('component_local_paths') or {}).get(name)!=str(expected):
            raise ValueError('direct local component placement differs: '+name)
        inputs[name]=_check(overrides.get(name,expected),identity)
    inputs['dependencies']=dependency_path
    if target.exists():raise ValueError('restore staging target already exists')
    target.parent.mkdir(parents=True,exist_ok=True)
    estimate=parent_identity['bytes']+sum(components[name]['bytes'] for name in names)+dependency_identity['bytes']
    if shutil.disk_usage(target.parent).free < estimate + reserve_bytes:
        raise OSError('restore staging would consume the 80 GiB disk reserve')
    temp=Path(tempfile.mkdtemp(prefix='.'+target.name+'-',dir=target.parent))
    try:
        for i in range(4):
            worker=temp/'workers'/str(i);worker.mkdir(parents=True)
            shutil.copyfile(inputs['worker-'+str(i)],worker/'checkpoint.pt')
            _check(worker/'checkpoint.pt',{'sha256':components['worker-'+str(i)]['sha256'],'bytes':components['worker-'+str(i)]['bytes']})
            if not direct:_extract(inputs['resident-'+str(i)],worker,prefix='diloco-resident')
        _extract(inputs['coordinator'],temp,prefix='diloco-coordinator')
        shutil.copyfile(parent_local,temp/'parent.pt');_check(temp/'parent.pt',parent_identity)
        _extract(inputs['dependencies'],temp/'dependencies',prefix='objects')
        index=json.loads((temp/'dependencies/dependency-index.json').read_text())
        if index.get('schema')!='natlang.diloco-recovery-dependencies/1':raise ValueError('invalid dependency index')
        if direct and not any(item.get('sha256')==direct['control_sha256'] and
                              item.get('object')=='objects/'+direct['control_sha256'] for item in index['bindings']):
            raise ValueError('sealed dependencies omit the exact direct control')
        bindings={};external_objects={}
        for item in index['bindings']:
            path=item['path'];sha=item['sha256']
            if path in bindings and bindings[path]!=sha:
                raise ValueError('conflicting dependency identity')
            if 'external' in item and direct:
                ext=item['external'];identity={'sha256':sha,'bytes':item['bytes']}
                if any(ext.get(k)!=v for k,v in identity.items()):raise ValueError('external dependency identity differs')
                receipt_ref=ext['receipt'];stored=temp/'dependencies'/'objects'/receipt_ref['sha256']
                _check(stored,{'sha256':receipt_ref['sha256'],'bytes':stored.stat().st_size})
                external_receipt=json.loads(stored.read_text());validate_receipt(external_receipt,identity)
                if not external_receipt.get('verified_at') or not present(external_receipt):raise ValueError('external private dependency is absent')
                local=Path((overrides.get('external') or {}).get(sha,ext['local_path']))
                if external_required:
                    from natlang_neuralese.common import paths
                    hdd=paths.root('data_hdd').resolve();nvme=paths.root('data_nvme').resolve()
                    if local.is_relative_to(hdd) and not hdd.is_mount():raise ValueError('external dependency mount is unavailable')
                    resolved=local.resolve()
                    if not (resolved.is_relative_to(nvme) or (hdd.is_mount() and resolved.is_relative_to(hdd))):
                        raise ValueError('external dependency is outside local recovery roots')
                _check(local,identity);external_objects[path]={**ext,'resolved_local_path':str(local)}
            else:
                if item.get('object')!='objects/'+sha:raise ValueError('invalid dependency object')
                _check(temp/'dependencies'/item['object'],{'sha256':sha,'bytes':item['bytes']})
            bindings[path]=sha
        manifest={'schema':'natlang.diloco-staged-recovery/1','cycle':cycle_id,'run_id':state['run_id'],
                  'coordinator_run_id':state['coordinator_run_id'],'parent':{**pin,'staged':'parent.pt'},'components':components,
                  'coordinator_staged':'diloco-coordinator','worker_staged':{str(i):f'workers/{i}' for i in range(4)},
                  **({'direct_control_sha256':direct['control_sha256'],'external_dependencies':external_objects} if direct else {}),
                  'scope':'separate worker optimizer/RNG/exposure and coordinator state; not a consolidated checkpoint',
                  'recovery_restore_qualified':False}
        write_json_durable(temp/'staging.json',manifest)
        os.rename(temp,target)
        return {'target':str(target),'cycle':cycle_id,'parent_sha256':pin['sha256'],
                'scope':manifest['scope'],'recovery_restore_qualified':False}
    finally:
        if temp.exists():shutil.rmtree(temp)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state',required=True);parser.add_argument('--cycle',required=True)
    parser.add_argument('--storage',required=True);parser.add_argument('--parent-pin');parser.add_argument('--parent-receipt')
    parser.add_argument('--parent-local',required=True);parser.add_argument('--out',required=True);parser.add_argument('--inputs-json');args=parser.parse_args()
    from scripts.coreweave import cwctl
    cwctl.load_env_file()
    import wandb
    api=wandb.Api()
    def present(receipt):
        files=list(api.run(receipt['run']).files(names=[receipt['name']]))
        return len(files)==1 and files[0].name==receipt['name'] and files[0].size==receipt['identity']['bytes']
    inputs=json.loads(Path(args.inputs_json).read_text()) if args.inputs_json else None
    print(json.dumps(stage(args.state,args.cycle,args.storage,args.parent_pin,args.parent_receipt,args.parent_local,args.out,present=present,inputs=inputs),sort_keys=True))


if __name__=='__main__':main()
