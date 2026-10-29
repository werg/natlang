#!/usr/bin/env python3
"""Create a compact, source-bound native checkpoint publication control archive."""
from __future__ import annotations
import argparse, hashlib, io, json, os, sys, tarfile, types
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(reference):
    path = Path(reference['path'])
    if not path.is_absolute() or not path.is_file() or digest(path) != reference['sha256']:
        raise ValueError('selected publication dependency differs: '+str(path))
    if 'bytes' in reference and path.stat().st_size != reference['bytes']:
        raise ValueError('selected publication dependency size differs: '+str(path))
    return path


def retirement_proofs(training):
    """Collect committed proof references from the real active and carried frames."""
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    frames = [training, *collect_sft_progress(training, borrow_frames=True).values()]
    result = {}
    for frame in frames:
        for item in (frame.get('options') or {}).get('source_predecessor_retirement_history') or []:
            ref = {'path': item['proof_path'], 'sha256': item['proof_sha256']}
            path = checked(ref)
            proof = json.loads(path.read_text())
            if any(item.get(k) != value for k, value in proof.items()):
                raise ValueError('committed predecessor retirement proof differs')
            result[(str(path),ref['sha256'])] = ref
    return list(result.values())


def seal_archive(out, checkpoint_sha256, control, producer, provider, selection, helper, proofs):
    """Seal members and a logical restoration map; verify every archived byte."""
    out=Path(out).resolve(); control_path=checked(control); doc=json.loads(control_path.read_text())
    members={}; restore={}
    def add(name,path,logical=None):
        path=Path(path);data=path.read_bytes();ref={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        if name in members and members[name][0]!=data:raise ValueError('archive member collision')
        members[name]=(data,ref)
        if logical is not None:
            if logical in restore and restore[logical] != {'member':name,**ref}:raise ValueError('restore path collision')
            restore[logical]={'member':name,**ref}
    for item in sorted(control_path.parent.rglob('*')):
        if item.is_file():add('control/'+str(item.relative_to(control_path.parent)),item,str(item))
    add('mechanisms/native_checkpoint_control.py',producer,str(producer))
    add('mechanisms/provider_dependencies.py',provider,str(provider))
    add('mechanisms/native_control_archive.py',helper,str(helper))
    add('selection/selection.json',selection,str(selection))
    side=doc.get('hashing_sidecar')
    if side:add('mechanisms/hashing_sidecar.py',checked(side),side['path'])
    # Provider must have authenticated each exact proof, not merely listed it in a saved frame.
    files=doc.get('dependencies',{}).get('files',[]) + doc.get('dependencies',{}).get('assets',[])
    for ref in proofs:
        if not any(f.get('path')==ref['path'] and f.get('sha256')==ref['sha256'] for f in files):
            raise ValueError('committed retirement proof missing from provider closure')
        add('proofs/'+ref['sha256']+'.json',checked(ref),ref['path'])
    catalog={name:ref for name,(_,ref) in members.items()};meta={'schema':'natlang.native-control-restore-map/1','entries':restore}
    archive=out/(checkpoint_sha256+'-publication-control.tar');receipt_path=archive.with_suffix('.receipt.json')
    def verify():
        with tarfile.open(archive,'r') as tar:
            entries={x.name:x for x in tar.getmembers()}
            if set(entries)!=set(catalog)|{'catalog.json','restore-map.json'}:raise ValueError('control archive member set differs')
            if json.load(tar.extractfile(entries['catalog.json']))!=catalog or json.load(tar.extractfile(entries['restore-map.json']))!=meta:raise ValueError('control archive catalog or restore map differs')
            for name,ref in catalog.items():
                member=entries[name]
                if not member.isfile() or member.size!=ref['bytes']:raise ValueError('control archive member size differs')
                if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=ref['sha256']:raise ValueError('control archive member hash differs')
    if archive.exists():
        if not receipt_path.exists():raise ValueError('existing archive lacks sealed receipt')
        receipt=json.loads(receipt_path.read_text())
        if receipt['checkpoint_sha256']!=checkpoint_sha256 or receipt['control']!=control or receipt['catalog']!=catalog or receipt['archive']['sha256']!=digest(archive):raise ValueError('existing control archive binding differs')
        verify();return receipt
    with tarfile.open(archive,'x') as tar:
        for name,(data,_) in members.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
        for name,value in [('catalog.json',catalog),('restore-map.json',meta)]:
            data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
    verify();receipt={'schema':'natlang.native-control-archive/1','checkpoint_sha256':checkpoint_sha256,'control':control,'archive':{'path':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size},'catalog':catalog,'restore_map':meta,'scope':'Compact control, committed retirement proofs and exact mechanisms; full checkpoint and separately verified source archives remain required.'}
    data=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode();fd=os.open(receipt_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--checkpoint-sha256',required=True);parser.add_argument('--step',type=int,required=True);parser.add_argument('--out',required=True);parser.add_argument('--selection',required=True);parser.add_argument('--selection-sha256',required=True);parser.add_argument('--helper-path',required=True);parser.add_argument('--helper-sha256',required=True);args=parser.parse_args()
    selection_path=checked({'path':args.selection,'sha256':args.selection_sha256});helper=checked({'path':args.helper_path,'sha256':args.helper_sha256});selection=json.loads(selection_path.read_text())
    if selection.get('schema')!='natlang.native-control-publication-selection/1':raise ValueError('invalid control selection')
    runtime=Path(selection['runtime_root']);producer=checked(selection['producer']);provider=checked(selection['dependency_producer'])
    if producer != runtime/'training/neuralese/natlang_neuralese/model/native_checkpoint_control.py':raise ValueError('producer is not from selected runtime')
    sys.path[:0]=[str(runtime/'training/neuralese'),str(runtime)]
    from natlang_neuralese.model.native_checkpoint_control import source_stat,write_checkpoint_control
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    path=Path(args.checkpoint).resolve(strict=True);before=source_stat(path)
    if digest(path)!=args.checkpoint_sha256:raise ValueError('checkpoint hash differs')
    state=CheckpointPolicy.load(path)
    if state.get('step')!=args.step or state.get('training',{}).get('global_step')!=args.step or source_stat(path)!=before:raise ValueError('pin step or identity changed')
    provider_bytes=provider.read_bytes()
    if hashlib.sha256(provider_bytes).hexdigest()!=selection['dependency_producer']['sha256']:raise ValueError('provider changed before execution')
    name='bound_provider_'+selection['dependency_producer']['sha256'];module=types.ModuleType(name);module.__file__=str(provider);sys.modules[name]=module;exec(compile(provider_bytes,str(provider),'exec'),module.__dict__)
    def collector(*values,**kwargs):
        print('NATLANG_CONTROL_COLLECTOR_INVOKED',flush=True)
        return module.dependency_closure(*values,**kwargs)
    collector.__module__=name
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    control=write_checkpoint_control(path,out,validated_state=state,decoded_source_stat=before,dependency_collector=collector,publication_only=True,namespace=selection['namespace'],readonly_input_roots=selection['readonly_input_roots'])
    if control is None or source_stat(path)!=before:raise ValueError('control was not produced from unchanged pin')
    result=seal_archive(out,args.checkpoint_sha256,control,producer,provider,selection_path,helper,retirement_proofs(state['training']))
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
