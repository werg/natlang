#!/usr/bin/env python3
"""Exercise an independent local recovery closure before enabling its queue.

Copies immutable inputs to the independent content store, restores runtime and
Python archives into an isolated scratch tree, and runs the real CPU native
checkpoint loader, a staged graph loader, and the Node collector import. It
never changes the original checkpoint or any live control.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,tarfile,tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from scripts.coreweave.private_export_guard import CacheBoundedReader
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.checkpoint_offload import (file_sha256, recovery_closure_identity,
    validate_recovery_plan_dependencies, validate_recovery_plan_shape, verified_recovery_blob)


def extract_archive(archive_path, destination, *, environment=False):
    """Extract only safe members, then independently verify every archived entry."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest_name='environment-manifest.json' if environment else 'runtime-manifest.json'
    prefix='venv/' if environment else 'runtime/'
    observed={};manifest=None
    known={'venv/bin/python':'/usr/bin/python3.12','venv/bin/python3':'python','venv/bin/python3.12':'python'}
    def allow(member,target):
        if environment and member.issym() and known.get(member.name)==member.linkname:return member
        return tarfile.data_filter(member,target)
    with CacheBoundedReader(str(archive_path)) as source:
        with tarfile.open(fileobj=source,mode='r|') as archive:
            for member in archive:
                if member.name==manifest_name:
                    if manifest is not None or not member.isfile():raise ValueError('duplicate or invalid archive manifest')
                    manifest=json.load(archive.extractfile(member));continue
                if not member.name.startswith(prefix) or member.name in observed:
                    raise ValueError('unexpected or duplicate restore archive member')
                if not (member.isfile() or member.isdir() or member.issym()):
                    raise ValueError('unsupported restore archive member')
                archive.extract(member,destination,filter=allow)
                path=destination/member.name
                if member.issym():observed[member.name]={'name':member.name,'link':member.linkname}
                elif member.isdir():observed[member.name]={'name':member.name,'directory':True,'mode':member.mode & 0o7777}
                else:observed[member.name]={'name':member.name,'bytes':member.size,'sha256':file_sha256(path)}
    expected={}
    if not isinstance(manifest,dict) or not isinstance(manifest.get('entries'),list):raise ValueError('archive manifest is missing')
    for item in manifest['entries']:
        value={k:v for k,v in item.items() if k!='original_link'}
        if value.get('directory'):value['mode'] &= 0o7777
        if value['name'] in expected:raise ValueError('duplicate archive manifest entry')
        expected[value['name']]=value
    if observed!=expected:raise ValueError('restored archive members differ from exact inventory')
    return manifest


def _mapped_path(tree, absolute):
    value=Path(absolute)
    if not value.is_absolute() or '..' in value.parts:raise ValueError('recovery source path must be normalized and absolute')
    return tree/value.relative_to('/')


def _host_contract():
    import platform
    result={'machine':platform.machine(),'kernel':platform.release(),'python_interpreter':str(Path('/usr/bin/python3.12').resolve())}
    driver=Path('/proc/driver/nvidia/version')
    result['nvidia_driver']=driver.read_text().strip() if driver.is_file() else None
    return result


def copy_closure_blobs(base_closure, store):
    def copy_one(item):
        key=(item['path'],item['sha256']);target=store/'closure'/item['sha256']
        return key, {'path':str(target),'identity':verified_recovery_blob(item['path'],target,item['sha256'])}
    # At most four copies and eight queued rows; never buffer payloads in RAM.
    with ThreadPoolExecutor(max_workers=4) as pool:
        iterator=iter(enumerate(base_closure)); pending={}; results={}
        for _ in range(8):
            row=next(iterator,None)
            if row is None:break
            index,item=row;pending[pool.submit(copy_one,item)]=index
        while pending:
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                index=pending.pop(future);results[index]=future.result()
                row=next(iterator,None)
                if row is not None:
                    index,item=row;pending[pool.submit(copy_one,item)]=index
    return dict(results[index] for index in sorted(results))


def qualify(draft, checkpoint, scratch_parent, output_plan, qualification_path, evidence_path, environment_scratch=None):
    draft=Path(draft).resolve(strict=True);checkpoint=Path(checkpoint).resolve(strict=True)
    output_plan=Path(output_plan).resolve();qualification_path=Path(qualification_path).resolve();evidence_path=Path(evidence_path).resolve()
    if any(path.exists() for path in (output_plan,qualification_path,evidence_path)):
        raise ValueError('recovery qualification outputs already exist')
    plan=json.loads(draft.read_text());plan['restore_qualification']={'path':str(qualification_path),'sha256':'0'*64}
    plan['closure']=[*plan['closure'],{'role':'controls',**plan['restore_qualification']}]
    validate_recovery_plan_shape(plan);validate_recovery_plan_dependencies(plan,require_restore=False)
    store=Path(plan['destination']).resolve(strict=True);scratch_parent=Path(scratch_parent).resolve(strict=True)
    if store.stat().st_dev!=scratch_parent.stat().st_dev:
        raise ValueError('recovery qualification scratch must share the independent destination filesystem')
    def checkpoint_identity():
        stat=checkpoint.stat();return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    checkpoint_before=checkpoint_identity();checkpoint_sha=file_sha256(checkpoint)
    if checkpoint_identity()!=checkpoint_before:raise ValueError('recovery checkpoint changed while hashing')
    placeholder=plan['restore_qualification'];base_closure=[x for x in plan['closure'] if not (x['role']=='controls' and x['path']==placeholder['path'] and x['sha256']==placeholder['sha256'])]
    blobs=copy_closure_blobs(base_closure,store)
    restore=plan.get('restore') or {}
    def find(role,path):
        items=[x for x in base_closure if x['role']==role and x['path']==path]
        if len(items)!=1:raise ValueError('restore artifact is missing or ambiguous: '+role)
        return Path(blobs[(items[0]['path'],items[0]['sha256'])]['path'])
    runtime_archive=find('runtime',restore['runtime_archive']);environment_archive=find('environment',restore['environment_archive']);stdlib_archive=find('system-python',restore['system_python_archive'])
    asset_archives=[(item,find('task-assets',item['archive'])) for item in restore['asset_archives']]
    environment_scratch=Path(environment_scratch or scratch_parent).resolve(strict=True)
    with (tempfile.TemporaryDirectory(prefix='native-full-recovery-restore-',dir=scratch_parent) as tmp,
          tempfile.TemporaryDirectory(prefix='native-recovery-environment-',dir=environment_scratch) as environment_tmp):
        root=Path(tmp);tree=root/'tree';tree.mkdir();source_map={}
        environment_root=Path(environment_tmp)
        archives={str(runtime_archive),str(environment_archive),str(stdlib_archive),*(str(archive) for _,archive in asset_archives)}
        for item in base_closure:
            blob=Path(blobs[(item['path'],item['sha256'])]['path'])
            if str(blob) in archives:continue
            target=_mapped_path(tree,item['path']);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if file_sha256(target)!=item['sha256']:raise ValueError('conflicting restored source path')
            else:os.link(blob,target)
            source_map[item['path']]=str(target)
            source_map[str(Path(item['path']).resolve())]=str(target)
        write_json_durable(root/'.natlang-source-map.json',{'schema':'natlang.staged-source-map/1','paths':source_map})
        runtime_manifest=extract_archive(runtime_archive,root/'restored-runtime')
        env_manifest=extract_archive(environment_archive,environment_root,environment=True)
        stdlib_manifest=extract_archive(stdlib_archive,root/'restored-stdlib')
        asset_roots={}
        for index,(item,archive) in enumerate(asset_archives):
            temporary=root/f'restored-assets-{index}';extract_archive(archive,temporary)
            target=_mapped_path(tree,item['original_root']);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError('restored task asset root collides with another artifact')
            os.rename(temporary/'runtime',target);asset_roots[item['env']]=str(target)
        system_home=root/'system-python';(system_home/'lib').mkdir(parents=True)
        os.symlink(root/'restored-stdlib'/'runtime',system_home/'lib/python3.12')
        interpreter_source=Path(source_map[restore['interpreter_path']]);interpreter=root/'python-bin/python3.12';interpreter.parent.mkdir()
        shutil.copyfile(interpreter_source,interpreter);interpreter.chmod(0o755)
        venv=environment_root/'venv';(venv/'bin/python').unlink();(venv/'bin/python').symlink_to(interpreter)
        node_source=Path(source_map[restore['node_path']]);node=root/'node-bin/node';node.parent.mkdir();shutil.copyfile(node_source,node);node.chmod(0o755)
        runtime=root/'restored-runtime/runtime';base=_mapped_path(tree,restore['base_path'])
        if not base.is_dir():raise ValueError('restored model base is missing')
        env=dict(os.environ,**asset_roots,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                 PYTHONHOME=str(system_home),PYTHONPATH=str(runtime/'training/neuralese')+':'+str(runtime),
                 NATLANG_REPO=str(runtime),NATLANG_HASH_CACHE=str(root/'hash-cache.jsonl'),
                 HF_HUB_OFFLINE='1',NATLANG_ROW_INDEX_CACHE=str(root/'row-index'))
        env.pop('VIRTUAL_ENV',None);env.pop('LD_LIBRARY_PATH',None)
        code=r'''import json,sys
from pathlib import Path
from natlang_neuralese.model.native_model import load_native_checkpoint
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.data.native_graph_schema import communication_classification
from natlang_neuralese.train.checkpoint_offload import recovery_training_custody, _plan_matches_checkpoint
from natlang_neuralese.train.recipe import native_source_dependencies
checkpoint,base,train,held,sample,fingerprint,mapping,plan_path=sys.argv[1:]
paths=json.loads(Path(mapping).read_text())['paths']
def route(parent,child):
 p=Path(child);key=str(p);resolved=str(p.resolve())
 if key in paths:return Path(paths[key])
 if resolved in paths:return Path(paths[resolved])
 if str(p).startswith(str(Path(mapping).parent)):return p
 raise ValueError('restore source escaped staged closure: '+key)
train_deps=native_source_dependencies(train,source_route=route);held_deps=native_source_dependencies(held,source_route=route)
model,tokenizer,roles,config,state=load_native_checkpoint(checkpoint,base=base,device='cpu')
if not _plan_matches_checkpoint(json.loads(Path(plan_path).read_text()),state):raise ValueError('restored checkpoint source and frame custody differs from recovery plan')
if config['tokenizer_fingerprint']!=fingerprint:raise ValueError('restored tokenizer fingerprint differs')
source=NativeGraphSource(sample,fingerprint,required_split='train',role_ids=tuple(config['marker_ids']))
example=next(iter(source));classification=communication_classification(example)
if not classification['communication']:raise ValueError('restored sample is not an actual communication graph')
mapped=[]
for line in Path('/proc/self/maps').read_text().splitlines():
 parts=line.split()
 if len(parts)>=6 and 'x' in parts[1] and parts[-1].startswith('/'):
  p=Path(parts[-1]);
  if p.is_file():mapped.append(str(p.resolve()))
print('RECOVERY_SMOKE='+json.dumps({'custody':recovery_training_custody(state),'tokenizer_fingerprint':config['tokenizer_fingerprint'],'train_dependencies':len(train_deps),'held_dependencies':len(held_deps),'sample_id':example.id,'sample_classification':classification,'mapped_executables':sorted(set(mapped))}),flush=True)
'''
        train=Path(source_map[plan['inputs']['train']['path']]);held=Path(source_map[plan['inputs']['held']['path']]);sample=Path(source_map[restore['sample_graph']])
        staged_plan=root/'restore-plan.json';write_json_durable(staged_plan,plan)
        command=[str(venv/'bin/python'),'-c',code,str(checkpoint),str(base),str(train),str(held),str(sample),restore['tokenizer_fingerprint'],str(root/'.natlang-source-map.json'),str(staged_plan)]
        py=subprocess.run(command,cwd=runtime,env=env,capture_output=True,text=True,timeout=1800)
        if py.returncode:raise RuntimeError('restored native checkpoint smoke failed: '+py.stderr[-5000:])
        lines=[line[len('RECOVERY_SMOKE='):] for line in py.stdout.splitlines() if line.startswith('RECOVERY_SMOKE=')]
        if len(lines)!=1:raise ValueError('restored native checkpoint smoke omitted exact result')
        if checkpoint_identity()!=checkpoint_before:raise ValueError('recovery checkpoint changed during isolated restore')
        smoke=json.loads(lines[0]);collector=(runtime/'ts-host/dist/teacher/collector.js').resolve(strict=True)
        js="import fs from 'node:fs'; await import(process.argv[1]); console.log('NODE_RECOVERY='+JSON.stringify(fs.readFileSync('/proc/self/maps','utf8')));"
        node_result=subprocess.run([str(node),'--input-type=module','-e',js,collector.as_uri()],cwd=runtime,env=env,capture_output=True,text=True,timeout=120)
        if node_result.returncode:raise RuntimeError('restored Node collector import failed: '+node_result.stderr[-3000:])
        node_lines=[line[len('NODE_RECOVERY='):] for line in node_result.stdout.splitlines() if line.startswith('NODE_RECOVERY=')]
        if len(node_lines)!=1:raise ValueError('restored Node import omitted process map')
        asset_inventory=json.loads(Path(source_map[restore['asset_inventory']]).read_text())
        expected_roots={item['env']:item['original_root'] for item in restore['asset_archives']}
        if set(asset_inventory.get('roots',{}))!=set(expected_roots):raise ValueError('restored task asset inventory roots differ')
        for name,original in expected_roots.items():
            if asset_inventory['roots'][name].get('path')!=original:raise ValueError('task asset original root differs')
            asset_inventory['roots'][name]['path']=asset_roots[name]
        staged_inventory=root/'restored-asset-inventory.json';write_json_durable(staged_inventory,asset_inventory)
        asset_env=dict(env,**asset_roots)
        guard=Path(source_map[restore['asset_guard']])
        asset_result=subprocess.run([str(node),str(guard),str(staged_inventory)],cwd=runtime,env=asset_env,capture_output=True,text=True,timeout=120)
        if asset_result.returncode:raise RuntimeError('restored task asset reads failed: '+asset_result.stderr[-3000:])
        node_mapped=[]
        for line in json.loads(node_lines[0]).splitlines():
            parts=line.split()
            if len(parts)>=6 and 'x' in parts[1] and parts[-1].startswith('/') and Path(parts[-1]).is_file():node_mapped.append(str(Path(parts[-1]).resolve()))
        allowed={str(Path(item['path']).resolve()):item['sha256'] for item in base_closure if item['role'] in ('system-libraries','interpreter','node')}
        used=[]
        for raw in set(smoke['mapped_executables'])|set(node_mapped):
            path=Path(raw)
            if path.is_relative_to(root) or path.is_relative_to(environment_root) or path==checkpoint:continue
            if raw not in allowed or file_sha256(path)!=allowed[raw]:raise ValueError('unbound host executable or library in restored process: '+raw)
            used.append({'path':raw,'sha256':allowed[raw]})
        evidence={'schema':'natlang.local-recovery-restore-evidence/1','checkpoint':{'path':str(checkpoint),'sha256':checkpoint_sha},
                  'inputs':plan['inputs'],'sample':smoke,'node_collector':str(collector.relative_to(root)),
                  'runtime_manifest':runtime_manifest['schema'],'environment_manifest':env_manifest['schema'],
                  'system_python_manifest':stdlib_manifest['schema'],'task_asset_guard_output':asset_result.stdout.strip(),'used_host_executables':sorted(used,key=lambda x:x['path']),
                  'host_contract':_host_contract(),'qualifier':{'path':str(Path(__file__).resolve()),'sha256':file_sha256(Path(__file__))},'scope':'independent closure bytes restored; CPU checkpoint load and staged communication graph; host libraries are hash-checked, no optimizer update or capability claim'}
    evidence_path.parent.mkdir(parents=True,exist_ok=True);write_json_durable(evidence_path,evidence)
    evidence_ref={'path':str(evidence_path),'sha256':file_sha256(evidence_path)}
    plan['closure']=base_closure+[{'role':'controls',**evidence_ref}]
    checks={name:True for name in ('all_payloads_reread','source_closure_resolved','environment_imports','tokenizer_loaded','checkpoint_full_state_loaded','checkpoint_plan_custody','node_collector_imported','task_assets_read')}
    plan['restore_qualification']={'path':str(qualification_path),'sha256':'0'*64}
    qualification={'schema':'natlang.local-recovery-restore-qualification/1','status':'passed','inputs':plan['inputs'],
                   'checkpoint':{'sha256':checkpoint_sha,'training_custody':smoke['custody']},
                   'closure_identity':recovery_closure_identity(plan),'restore':plan['restore'],
                   'additional_inputs':plan.get('additional_inputs',[]),'controller_config':plan['controller']['config'],'checks':checks,
                   'evidence':{name:evidence_ref for name in checks},'host_contract':evidence['host_contract']}
    qualification_path.parent.mkdir(parents=True,exist_ok=True);write_json_durable(qualification_path,qualification)
    plan['restore_qualification']['sha256']=file_sha256(qualification_path)
    plan['closure'].append({'role':'controls',**plan['restore_qualification']})
    validate_recovery_plan_shape(plan);validate_recovery_plan_dependencies(plan)
    output_plan.parent.mkdir(parents=True,exist_ok=True);write_json_durable(output_plan,plan)
    return {'plan':{'path':str(output_plan),'sha256':file_sha256(output_plan)},'qualification':plan['restore_qualification'],'evidence':evidence_ref}


def main():
    p=argparse.ArgumentParser();p.add_argument('--draft-plan',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--scratch-parent',required=True);p.add_argument('--environment-scratch');p.add_argument('--output-plan',required=True);p.add_argument('--qualification',required=True);p.add_argument('--evidence',required=True)
    a=p.parse_args();print(json.dumps(qualify(a.draft_plan,a.checkpoint,a.scratch_parent,a.output_plan,a.qualification,a.evidence,a.environment_scratch)),flush=True)

if __name__=='__main__':main()
