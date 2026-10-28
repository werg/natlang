#!/usr/bin/env python3
"""Supervise private full-state recovery of each DiLoCo learner and the coordinator topology.

A cycle pins each worker's ordinary full checkpoint, then a stable coordinator file set. Worker and resident-control components and coordinator state are exported independently with
exact intents, owner GET verification and local copies; dependency bytes are sealed and exported separately.
The component receipts alone do not establish a complete topology recovery: resident control and
source/runtime/base dependencies must also be independently preserved.
No old parent, older private object or prior recovery set is deleted by this service.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/neuralese'))
import cwctl
from recovery_inputs import durable_json
from diloco_recovery_dependencies import seal, control_bindings
from natlang_neuralese.common import paths
from natlang_neuralese.common.hashing import sha256_file_hex


class Backend:
    def __init__(self,record,storage):
        self.record,self.storage=record,Path(storage).resolve()
        external=paths.root('data_hdd').resolve()
        if not external.is_mount() or not self.storage.is_relative_to(external):
            raise ValueError('recovery storage must be on the mounted external data root')
        self.external=external
        self.repo=paths.root('repo'); self.helper=Path(__file__).with_name('diloco_recovery_pin.py').read_text()
        self.parent=record['run']['coordinator']['parent'];self.coordinator=record['coordinator']
        self.workers=record['learners'];self.out='/work/diloco-learner'
        self.record_path=cwctl.STATE/'diloco'/(record['run']['id']+'.json')
        self.root='/work/diloco-coordinator'
        if len(self.workers)!=record['run']['settings']['workers'] or len(set(self.workers))!=len(self.workers):
            raise ValueError('recovery requires distinct actual sandbox IDs for every learner')
        self.manifest=json.loads(cwctl.read_sandbox(self.coordinator).read_file(self.root+'/manifest.json').result())
        if self.manifest.get('run_id')!=record['run']['settings']['run_id']:
            raise ValueError('actual coordinator manifest belongs to another run')
        self.fingerprint=self.manifest['fingerprint']

    def verify_ownership(self):
        if not self.external.is_mount():raise ValueError('external recovery storage is no longer mounted')
        live=json.loads(self.record_path.read_text())
        if live.get('coordinator')!=self.coordinator or live.get('learners')!=self.workers:
            raise ValueError('sandbox ownership changed; stop and explicitly bind a new recovery journal')

    def _exec(self,sandbox_id,args):
        command=['nice','-n','19','env','PYTHONPATH='+str(self.repo/'training/neuralese')+os.pathsep+str(self.repo),
                 'python3','-c',self.helper,*map(str,args)]
        result=cwctl.read_sandbox(sandbox_id).exec(command,timeout_seconds=3600).result(timeout=3660)
        if result.returncode:raise RuntimeError(f'recovery pin {sandbox_id} failed: {result.stderr[-2000:]}')
        return json.loads(result.stdout)

    def pin_worker(self,worker,cycle):
        pin=f'/work/.coreweave/recovery-pins/{cycle}/worker-{worker}-{uuid.uuid4().hex}/checkpoint.pt'
        package=self.repo/'training/neuralese/natlang_neuralese'
        result=self._exec(self.workers[worker],['worker',self.out+'/checkpoint.pt',pin,worker,self.fingerprint,package])
        return {**result,'sandbox_id':self.workers[worker],'source':self.out+'/checkpoint.pt'}

    def pin_resident(self,worker,cycle):
        pin=f'/work/.coreweave/recovery-pins/{cycle}/resident-{worker}-{uuid.uuid4().hex}/resident.tar'
        result=self._exec(self.workers[worker],['resident','/work/diloco-resident/resident-data',pin])
        return {**result,'sandbox_id':self.workers[worker],'source':'/work/diloco-resident/resident-data'}

    def pin_coordinator(self,cycle):
        pin=f'/work/.coreweave/recovery-pins/{cycle}/coordinator-{uuid.uuid4().hex}/coordinator.tar'
        result=self._exec(self.coordinator,['coordinator',self.root,pin,self.repo/'training/neuralese/natlang_neuralese'])
        return {**result,'sandbox_id':self.coordinator,'source':self.root}

    def export(self,name,intent):
        dest=self.storage/'objects'/intent['sha256']/Path(intent['path']).name
        receipt=self.storage/'receipts'/(name+'-'+intent['sha256']+'.json'); receipt.parent.mkdir(parents=True,exist_ok=True)
        exporter=paths.resolve('data_nvme','dgxops/coreweave-private-runfile-export-20261014.py')
        env={**os.environ,'CW_EXPORT_SANDBOX_ID':intent['sandbox_id'],'CW_EXPORT_EXPECTED_SHA256':intent['sha256'],
             'CW_EXPORT_EXPECTED_BYTES':str(intent['bytes']),'CW_EXPORT_CACHE_BOUNDED_READS':'1',
             'CW_EXPORT_VERIFY_ON_OWNER':'1','CW_EXPORT_OWNER_DESTINATION':str(dest),
             'CW_EXPORT_RECEIPT_FILE':str(receipt),'CW_EXPORT_LOCAL':'0','CW_EXPORT_TARGET_VERIFIES':'0'}
        for key in ('CW_EXPORT_SEALED_PIN','CW_EXPORT_VERIFY_EXISTING_ONLY','CW_EXPORT_VERIFIED_LEGACY_READER'):
            env.pop(key,None)
        sandbox=cwctl.read_sandbox(intent['sandbox_id']);made=sandbox.exec(['mkdir','-p','/work/production']).result()
        if made.returncode:raise RuntimeError('private export capability directory unavailable')
        subprocess.run([sys.executable,str(exporter),'lfm3-diloco-'+name,intent['path']],env=env,check=True)
        value=json.loads(receipt.read_text());identity=value.get('identity') or {};checked=value.get('independent_verification') or {}
        if (value.get('private') is not True or value.get('source')!=intent['path']
                or (value.get('reused_verified_export') or {}).get('source_sandbox_id', value.get('sandbox_id')) != intent['sandbox_id']
                or identity.get('sha256')!=intent['sha256']
                or identity.get('bytes')!=intent['bytes'] or checked.get('sha256')!=intent['sha256']
                or checked.get('bytes')!=intent['bytes'] or not value.get('verified_at')):
            raise ValueError('private recovery export lacks independent complete-object GET')
        local=checked.get('local_copy') or {}
        if local.get('committed') is True and local.get('sha256')!=intent['sha256']:
            raise ValueError('recovery export local-copy receipt differs')
        if not dest.is_file() or sha256_file_hex(dest,drop_cache=True)!=intent['sha256'] or dest.stat().st_size!=intent['bytes']:
            raise ValueError('local full-state readback differs')
        return value

    def prepare_dependencies(self,cycle):
        bundle=Path(self.record['run']['learners']['bundle_file'])
        sources=bundle.with_name(bundle.stem+'.sources.json')
        candidates={}
        for key,value in json.loads(sources.read_text()).items():
            candidates.setdefault(key.rsplit('/',1)[-1],[]).append(value)
        bindings=[]
        for i in range(len(self.workers)):
            required=cycle['components']['worker-'+str(i)]['required_dependencies']
            kinds={kind for item in required for kind in item['kinds']}
            if 'base-model-file' not in kinds or not ({'source-manifest','source-payload'} & kinds):
                raise ValueError('worker lacks explicit base or source dependency inventory')
            bindings.extend(required)
            resident=cycle['components']['resident-'+str(i)]
            archive=self.storage/'objects'/resident['sha256']/Path(resident['path']).name
            bindings.extend(control_bindings(archive))
        target=self.storage/'dependencies'/(cycle['id']+'-'+uuid.uuid4().hex+'.tar')
        return seal(bindings,candidates,bundle,sources,target,
                    runtime_code_sha256=cycle['components']['worker-0']['runtime_code_sha256'])

    def export_dependencies(self,intent):
        receipt=self.storage/'receipts'/('dependencies-'+intent['sha256']+'.json')
        exporter=paths.resolve('data_nvme','dgxops/coreweave-private-runfile-export-20261014.py')
        env={**os.environ,'CW_EXPORT_LOCAL':'1','CW_EXPORT_EXPECTED_SHA256':intent['sha256'],
             'CW_EXPORT_EXPECTED_BYTES':str(intent['bytes']),'CW_EXPORT_CACHE_BOUNDED_READS':'1',
             'CW_EXPORT_VERIFY_ON_OWNER':'1','CW_EXPORT_RECEIPT_FILE':str(receipt)}
        for key in ('CW_EXPORT_SEALED_PIN','CW_EXPORT_VERIFY_EXISTING_ONLY','CW_EXPORT_VERIFIED_LEGACY_READER',
                    'CW_EXPORT_OWNER_DESTINATION','CW_EXPORT_TARGET_VERIFIES'):
            env.pop(key,None)
        subprocess.run([sys.executable,str(exporter),'lfm3-diloco-dependencies',intent['path']],env=env,check=True)
        value=json.loads(receipt.read_text()); checked=value.get('independent_verification') or {}
        if (not value.get('private') or not value.get('owner_host_source') or value.get('source')!=intent['path']
                or value.get('identity')!={'sha256':intent['sha256'],'bytes':intent['bytes']}
                or checked.get('sha256')!=intent['sha256'] or checked.get('bytes')!=intent['bytes']
                or not value.get('verified_at')):
            raise ValueError('dependency export lacks independent complete-object GET')
        return value

    def present(self, receipt):
        import wandb
        identity=receipt['identity']
        files=list(wandb.Api().run(receipt['run']).files(names=[receipt['name']]))
        return len(files)==1 and files[0].name==receipt['name'] and files[0].size==identity['bytes']

    def progress(self,worker):
        raw=cwctl.read_sandbox(self.workers[worker]).read_file(self.out+'/progress.json').result()
        value=json.loads(raw)
        return {'global_step':value.get('global_step'),'checkpoint_step':value.get('checkpoint_step'),
                'diloco':value.get('diloco')}


def compatibility(components):
    workers=[components.get('worker-'+str(i)) for i in range(4)];coord=components.get('coordinator')
    if not coord or any(not w for w in workers):return {'compatible':False,'reason':'incomplete topology'}
    if any(w['fingerprint']!=coord['fingerprint'] for w in workers):return {'compatible':False,'reason':'coordinator fingerprint differs'}
    if (len({w['source_hash'] for w in workers})!=1
            or len({w['runtime_code_sha256'] for w in workers})!=1
            or workers[0]['runtime_code_sha256']!=coord['runtime_code_sha256']
            or len({json.dumps(w.get('recipe_binding'),sort_keys=True) for w in workers})!=1
            or len({json.dumps(w.get('ce_supervision_policy_binding'),sort_keys=True) for w in workers})!=1):
        return {'compatible':False,'reason':'worker source or runtime transitions differ'}
    residents=[components.get('resident-'+str(i)) for i in range(4)]
    if any(not r or not w.get('resident_generation') or r['generation']!=w['resident_generation'] for w,r in zip(workers,residents)):
        return {'compatible':False,'reason':'resident state is missing or differs from the worker checkpoint'}
    versions=coord['versions']
    if any(len(w['versions'])!=len(versions) or any(a>b for a,b in zip(w['versions'],versions)) for w in workers):
        return {'compatible':False,'reason':'a worker checkpoint is ahead of the pinned coordinator'}
    return {'compatible':True,'scope':'separate exact worker full states and stable coordinator state; '
            'not a consolidated global checkpoint; replay and staleness remain subject to normal DiLoCo checks'}


def run_once(record,state_path,backend):
    state_path=Path(state_path);state_path.parent.mkdir(parents=True,exist_ok=True)
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(state_path.read_text()) if state_path.exists() else {'schema':'natlang.diloco-recovery/1',
            'run_id':record['run']['id'],'cycles':[],'latest_verified':None}
        if state['run_id']!=record['run']['id']:raise ValueError('recovery state belongs to a different run')
        def save():durable_json(state_path,state)
        if hasattr(backend,'verify_ownership'):backend.verify_ownership()
        cycle=state.get('pending')
        if cycle is None:
            cycle={'id':uuid.uuid4().hex,'phase':'pinning','components':{},'receipts':{},'errors':{}}
            state['pending']=cycle;save()
        missing=[i for i in range(len(backend.workers)) if 'worker-'+str(i) not in cycle['components']]
        if missing:
            # A later worker checkpoint requires a newer coordinator pin; keep any old artifacts, but do not
            # bind them into this candidate as though they followed the newly pinned checkpoint.
            old=cycle['components'].pop('coordinator',None); old_receipt=cycle['receipts'].pop('coordinator',None)
            if old:cycle.setdefault('superseded_coordinator_pins',[]).append({'pin':old,'receipt':old_receipt})
            with ThreadPoolExecutor(max_workers=len(missing)) as pool:
                futures={i:pool.submit(backend.pin_worker,i,cycle['id']) for i in missing}
                for i,future in futures.items():
                    name='worker-'+str(i)
                    try:cycle['components'][name]=future.result();cycle['errors'].pop(name,None)
                    except Exception as error:cycle['errors'][name]=str(error)
                    save()
        for i in range(len(backend.workers)):
            name='resident-'+str(i)
            if name not in cycle['components']:
                try:cycle['components'][name]=backend.pin_resident(i,cycle['id']);cycle['errors'].pop(name,None)
                except Exception as error:cycle['errors'][name]=str(error)
        if 'coordinator' not in cycle['components']:
            try:cycle['components']['coordinator']=backend.pin_coordinator(cycle['id']);cycle['errors'].pop('coordinator',None)
            except Exception as error:cycle['errors']['coordinator']=str(error)
        for name in list(cycle['receipts']):
            if str(cycle['errors'].get(name,'')).startswith('presence: '):cycle['errors'].pop(name)
        cycle['compatibility']=compatibility(cycle['components']);cycle['phase']='exporting';save()
        # Each pin and actual step/source/runtime identity is durably recorded before its external PUT.
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures={name:pool.submit(backend.export,name,intent) for name,intent in cycle['components'].items()
                     if name not in cycle['receipts']}
            for name,future in futures.items():
                try:cycle['receipts'][name]=future.result();cycle['errors'].pop(name,None)
                except Exception as error:cycle['errors'][name]=str(error)
                save()
        cycle['observed_progress']={}
        for i in range(len(backend.workers)):
            try:cycle['observed_progress'][str(i)]=backend.progress(i)
            except Exception as error:cycle['observed_progress'][str(i)]={'unavailable':str(error)}
        for i in range(len(backend.workers)):
            worker=cycle['components'].get('worker-'+str(i));observed=cycle['observed_progress'][str(i)]
            if worker and type(observed.get('global_step')) is int:
                observed['updates_since_pinned_checkpoint']=max(0,observed['global_step']-worker['global_step'])
        cycle['errors'].pop('dependencies',None)
        all_components = len(cycle['components']) == 2*len(backend.workers)+1
        all_exported = all_components and set(cycle['receipts'])==set(cycle['components']) and not cycle['errors']
        if all_exported and cycle['compatibility']['compatible']:
            try:
                if 'dependency_intent' not in cycle:
                    cycle['dependency_intent']=backend.prepare_dependencies(cycle);save()
                if 'dependency_receipt' not in cycle:
                    cycle['dependency_receipt']=backend.export_dependencies(cycle['dependency_intent']);save()
                if not backend.present(cycle['dependency_receipt']):
                    raise ValueError('dependency private object is absent')
                cycle['errors'].pop('dependencies',None)
            except Exception as error:
                cycle['errors']['dependencies']=str(error);all_exported=False;save()
        if all_exported:
            for name, receipt in cycle['receipts'].items():
                try:
                    if not backend.present(receipt): raise ValueError('private object is absent')
                except Exception as error:
                    cycle['errors'][name]='presence: '+str(error)
            if cycle['errors']:
                cycle['phase']='incomplete';save()
            else:
                cycle['phase']='verified-components-and-dependencies' if cycle['compatibility']['compatible'] else 'verified-components-incompatible'
                cycle['verified_at']=time.time()
                cycle['recovery_blocker']='cross-component restore and coordinator sequence replay have not been qualified'
                if cycle['compatibility']['compatible']:state['latest_exported']=cycle['id']
                state['cycles'].append(cycle);state.pop('pending');save()
        else:
            cycle['phase']='incomplete';save()
        return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_id');p.add_argument('--state',type=Path,required=True)
    p.add_argument('--storage',type=Path,required=True);p.add_argument('--interval',type=float,default=1800)
    p.add_argument('--once',action='store_true');args=p.parse_args()
    if args.interval <= 0:p.error('interval must be positive')
    cwctl.load_env_file()
    import diloco
    record=json.loads((diloco.RUNS/(args.run_id+'.json')).read_text())
    backend=Backend(record,args.storage)
    while True:
        try:
            state=run_once(record,args.state,backend)
            cycle=state.get('pending') or state['cycles'][-1]
            print(json.dumps({'cycle':cycle['id'],'phase':cycle['phase'],'errors':cycle['errors'],
                              'latest_exported':state.get('latest_exported'),
                              'recovery_blocker':cycle.get('recovery_blocker'),
                              'observed_progress':cycle.get('observed_progress')},sort_keys=True),flush=True)
        except Exception as error:
            print(json.dumps({'error':type(error).__name__,'detail':str(error)}),flush=True)
            if args.once:raise
        if args.once:break
        time.sleep(args.interval)


if __name__=='__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):cwctl.reexec_in_venv()
    main()
