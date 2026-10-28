#!/usr/bin/env python3
"""Stage an independently verified DiLoCo topology for explicit restore; never launch a sandbox."""
from __future__ import annotations
import argparse
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


def stage(state_path,cycle_id,storage,parent_pin,parent_receipt,parent_local,target,*,present,reserve_bytes=80*2**30,external_required=True,direct_control=None):
    state=json.loads(Path(state_path).read_text());cycles=[item for item in state['cycles'] if item['id']==cycle_id]
    if len(cycles)!=1:raise ValueError('recovery cycle is absent or duplicated')
    cycle=cycles[0]
    if cycle.get('phase')!='verified-components-and-dependencies' or not cycle.get('compatibility',{}).get('compatible'):
        raise ValueError('recovery cycle is not a compatible exported cohort')
    components=cycle['components'];receipts=cycle['receipts'];storage=Path(storage);target=Path(target)
    from scripts.coreweave.diloco_periodic_recovery import compatibility
    direct=None
    if direct_control:
        from scripts.coreweave.diloco_direct_recovery import record_from_control,load_control
        record=record_from_control(direct_control,state['coordinator_run_id']);direct=load_control(direct_control,record)
        if state.get('recovery_contract')!={'mode':'direct','control_sha256':direct['control_sha256']}:
            raise ValueError('direct recovery control does not match exported cycle')
    elif (state.get('recovery_contract') or {}).get('mode')=='direct':
        raise ValueError('direct recovery requires its exact control descriptor')
    if not compatibility(components,direct=direct)['compatible']:raise ValueError('recovery sequence or component identities no longer match')
    names={'coordinator',*(f'worker-{i}' for i in range(4))}
    if not direct:names.update(f'resident-{i}' for i in range(4))
    if set(components)!=names or set(receipts)!=names:raise ValueError('topology components are incomplete')
    if external_required:
        from natlang_neuralese.common import paths
        external=paths.root('data_hdd').resolve()
        if not external.is_mount() or not target.resolve().is_relative_to(external):
            raise ValueError('restore staging must use the mounted external data root')
    pin=json.loads(Path(parent_pin).read_text());parent_identity={'sha256':pin['sha256'],'bytes':pin['bytes']}
    manifest=components.get('coordinator',{}).get('manifest') or {}
    if manifest.get('run_id')!=state.get('coordinator_run_id') or (manifest.get('identity') or {}).get('parent_checkpoint_sha256')!=pin['sha256']:
        raise ValueError('original parent does not match coordinator identity')
    old_receipt=json.loads(Path(parent_receipt).read_text());validate_receipt(old_receipt,parent_identity)
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
        inputs[name]=_check(expected,identity)
    dependency=cycle['dependency_intent'];dependency_identity={'sha256':dependency['sha256'],'bytes':dependency['bytes']}
    validate_receipt(cycle['dependency_receipt'],dependency_identity)
    if not cycle['dependency_receipt'].get('verified_at') or not present(cycle['dependency_receipt']):raise ValueError('private dependency archive is absent')
    inputs['dependencies']=_check(dependency['path'],dependency_identity)
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
                _check(ext['local_path'],identity);external_objects[path]=ext
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
    parser.add_argument('--storage',required=True);parser.add_argument('--parent-pin',required=True);parser.add_argument('--parent-receipt',required=True)
    parser.add_argument('--parent-local',required=True);parser.add_argument('--out',required=True);parser.add_argument('--direct-control');args=parser.parse_args()
    from scripts.coreweave import cwctl
    cwctl.load_env_file()
    import wandb
    api=wandb.Api()
    def present(receipt):
        files=list(api.run(receipt['run']).files(names=[receipt['name']]))
        return len(files)==1 and files[0].name==receipt['name'] and files[0].size==receipt['identity']['bytes']
    print(json.dumps(stage(args.state,args.cycle,args.storage,args.parent_pin,args.parent_receipt,args.parent_local,args.out,present=present,direct_control=args.direct_control),sort_keys=True))


if __name__=='__main__':main()
