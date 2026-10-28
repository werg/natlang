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
import threading
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/neuralese'))
import cwctl
from recovery_inputs import durable_json
from diloco_recovery_dependencies import seal, control_bindings
from natlang_neuralese.common import paths
from natlang_neuralese.common.hashing import sha256_file_hex


class FenceUnavailable(RuntimeError):
    pass


class Backend:
    def __init__(self,record,storage,parent_local=None,parent_qualification=None,parent_copy_receipt=None,runtime_bundle=None):
        self.parent_local=Path(parent_local) if parent_local else None
        self.parent_qualification=Path(parent_qualification) if parent_qualification else None
        self.parent_copy_receipt=Path(parent_copy_receipt) if parent_copy_receipt else None
        self.runtime_bundle=Path(runtime_bundle) if runtime_bundle else None
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

    def pin_worker(self,worker,cycle,source=None):
        pin=f'/work/.coreweave/recovery-pins/{cycle}/worker-{worker}-{uuid.uuid4().hex}/checkpoint.pt'
        package=self.repo/'training/neuralese/natlang_neuralese'
        result=self._exec(self.workers[worker],['worker',source or self.out+'/checkpoint.pt',pin,worker,self.fingerprint,package])
        return {**result,'sandbox_id':self.workers[worker],'source':source or self.out+'/checkpoint.pt'}

    def pin_resident(self,worker,cycle):
        pin=f'/work/.coreweave/recovery-pins/{cycle}/resident-{worker}-{uuid.uuid4().hex}/resident.tar'
        result=self._exec(self.workers[worker],['resident','/work/diloco-resident/resident-data',pin])
        return {**result,'sandbox_id':self.workers[worker],'source':'/work/diloco-resident/resident-data'}

    def pin_coordinator(self,cycle):
        pin=f'/work/.coreweave/recovery-pins/{cycle}/coordinator-{uuid.uuid4().hex}/coordinator.tar'
        result=self._exec(self.coordinator,['coordinator',self.root,pin,self.repo/'training/neuralese/natlang_neuralese'])
        return {**result,'sandbox_id':self.coordinator,'source':self.root}

    def pin_coordinator_tree(self,cycle):
        pin=f'/work/.coreweave/recovery-pins/{cycle}/coordinator-tree-{uuid.uuid4().hex}/tree'
        result=self._exec(self.coordinator,['coordinator-tree',self.root,pin,self.repo/'training/neuralese/natlang_neuralese'])
        return {**result,'sandbox_id':self.coordinator,'source':self.root}

    def archive_coordinator_tree(self,component):
        tree=component['path'];identity=tree+'.identity.json';archive=tree+'.tar'
        box=cwctl.read_sandbox(self.coordinator)
        box.write_file(identity,json.dumps(component['file_identity'],sort_keys=True).encode()).result()
        result=self._exec(self.coordinator,['coordinator-tree-archive',tree,archive,identity])
        return {**component,**result,'binding_kind':'immutable-hardlink-tree-archive','tree':tree}

    def capture_fenced(self, cycle, *, timeout=600):
        """Fence all learners at completed updates, pin controls and coordinator, then release."""
        fence=uuid.uuid4().hex;deadline=time.time()+timeout
        directory=self.out+'/diloco/recovery-fence'
        boxes=[cwctl.read_sandbox(ident) for ident in self.workers]
        # Fail quickly before requesting anything if the actual learner runtime lacks the fence mechanism.
        completed={}
        for worker,box in enumerate(boxes):
            try:cap=json.loads(box.read_file(directory+'/capabilities.json').result())
            except Exception:raise FenceUnavailable('worker runtime has no active recovery-fence capability') from None
            if (cap.get('schema')!='natlang.diloco-recovery-fence/1' or cap.get('worker')!=worker
                    or cap.get('fingerprint')!=self.fingerprint):
                raise FenceUnavailable('worker recovery-fence capability belongs to another owner')
            check=box.exec(['python3','-c',
                "import json,os,pathlib,sys; o=json.loads(sys.argv[1]); p=pathlib.Path('/proc'); "
                "sys.exit(not (o['boot_id']==(p/'sys/kernel/random/boot_id').read_text().strip() and "
                "o['start_ticks']==(p/str(o['pid'])/'stat').read_text().split(') ',1)[1].split()[19]))",
                json.dumps(cap['owner'])],timeout_seconds=30).result()
            if check.returncode:
                # A dead trainer is eligible only with a new-protocol, exact final report.
                try: report=json.loads(box.read_file(self.out+'/report.json').result())
                except Exception:raise FenceUnavailable('worker owner died without a final report') from None
                delivery=report.get('diloco_final_delivery') or {}
                if (report.get('active_stage_complete') is not True or
                    delivery.get('schema')!='natlang.diloco-final-delivery/1' or
                    delivery.get('worker')!=worker or delivery.get('fingerprint')!=self.fingerprint or
                    not isinstance(delivery.get('sequences'),list) or
                    not isinstance(report.get('checkpoint_sha256'),str)):
                    raise FenceUnavailable('completed worker lacks exact final delivery custody')
                source=self.out+'/checkpoint.pt';pin=directory+'/pins/'+fence+'/checkpoint.pt'
                program=("import hashlib,os,pathlib,sys; source,pin,want=sys.argv[1:]; p=pathlib.Path(pin); "
                         "p.parent.mkdir(parents=True,exist_ok=True); os.link(source,pin); "
                         "h=hashlib.sha256(); f=p.open('rb'); "
                         "[h.update(b) for b in iter(lambda:f.read(8388608),b'')]; "
                         "sys.exit(0 if h.hexdigest()==want else 1)")
                result=box.exec(['python3','-c',program,source,pin,report['checkpoint_sha256']],timeout_seconds=3600).result(timeout=3660)
                if result.returncode:raise ValueError('completed worker final checkpoint hash differs from report')
                completed[worker]={'id':fence,'worker':worker,'fingerprint':self.fingerprint,
                    'checkpoint':pin,'deadline':deadline,'completed':True,'sequences':delivery['sequences'],
                    'checkpoint_sha256':report['checkpoint_sha256']}
        try:
            for worker,box in enumerate(boxes):
                if worker in completed:continue
                made=box.exec(['mkdir','-p',directory]).result()
                if made.returncode:raise RuntimeError('worker fence directory unavailable')
                value={'schema':'natlang.diloco-recovery-fence/1','id':fence,'worker':worker,
                       'fingerprint':self.fingerprint,'deadline':deadline}
                pending=directory+'/request-'+fence+'.pending'
                box.write_file(pending,json.dumps(value).encode()).result()
                moved=box.exec(['mv',pending,directory+'/request.json']).result()
                if moved.returncode:raise RuntimeError('worker fence request publication failed')
            ready=dict(completed)
            while len(ready)<len(boxes):
                if time.time()>=deadline:raise TimeoutError('worker recovery fence expired')
                for worker,box in enumerate(boxes):
                    if worker in ready:continue
                    try:
                        value=json.loads(box.read_file(directory+'/'+fence+'.ready.json').result())
                    except Exception:
                        try:failure=json.loads(box.read_file(directory+'/'+fence+'.failed.json').result())
                        except Exception:continue
                        raise RuntimeError('worker fence failed: '+str(failure.get('error_type')))
                    if (value.get('id')!=fence or value.get('worker')!=worker
                            or value.get('fingerprint')!=self.fingerprint or value.get('deadline')!=deadline
                            or value.get('checkpoint')!=directory+'/pins/'+fence+'/checkpoint.pt'):
                        raise ValueError('worker fence readiness binding differs')
                    ready[worker]=value
                if len(ready)<len(boxes):time.sleep(1)
            with ThreadPoolExecutor(max_workers=5) as pool:
                coordinator=pool.submit(self.pin_coordinator_tree,cycle)
                residents={i:pool.submit(self.pin_resident,i,cycle) for i in range(len(boxes))}
                components={'coordinator':coordinator.result()}
                for i,future in residents.items():components['resident-'+str(i)]=future.result()
            if time.time()>=deadline:raise TimeoutError('coordinator pin finished after fence deadline')
            return {'id':fence,'deadline':deadline,'ready':ready,'components':components}
        finally:
            release=json.dumps({'schema':'natlang.diloco-recovery-fence/1','id':fence}).encode()
            for box in boxes:
                try:box.write_file(directory+'/'+fence+'.release.json',release).result()
                except Exception:pass  # Each trainer also has the bounded deadline.

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

    def release_remote_pins(self,cycle):
        """Release only cycle-owned pins after private GET and local copy verification."""
        self.verify_ownership()
        components=cycle['components'];receipts=cycle['receipts']
        if (cycle.get('phase')!='verified-components-and-dependencies' or
                not cycle.get('compatibility',{}).get('compatible') or
                not self.present(cycle['dependency_receipt'])):
            raise ValueError('unverified recovery cycle cannot release pins')
        for name,intent in components.items():
            if not self.present(receipts[name]):raise ValueError('private recovery component disappeared: '+name)
            local=self.storage/'objects'/intent['sha256']/Path(intent['path']).name
            if not local.is_file() or local.stat().st_size!=intent['bytes']:
                raise ValueError('local recovery component is absent: '+name)
        deps=cycle['dependency_intent']
        if not Path(deps['path']).is_file() or Path(deps['path']).stat().st_size!=deps['bytes']:
            raise ValueError('local recovery dependencies are absent')
        root='/work/.coreweave/recovery-pins/'+cycle['id']
        def release(name,intent):
            args=['release',intent['path'],intent['sha256'],intent['bytes'],root]
            if name.startswith('worker-') and 'fenced' in cycle:
                worker=int(name.removeprefix('worker-'));fence=cycle['fenced']
                args += ['--fenced-source',fence['ready'][str(worker)]['checkpoint'],
                         '--fence-root',self.out+'/diloco/recovery-fence/pins/'+fence['id']]
            return self._exec(intent['sandbox_id'],args)
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures={name:pool.submit(release,name,intent) for name,intent in components.items()}
            for future in futures.values():future.result()
        return {'status':'released','at':time.time(),'scope':'only verified cycle-owned remote pins'}

    def prepare_dependencies(self,cycle):
        bundle=self.runtime_bundle or Path(self.record['run']['learners']['bundle_file'])
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

    def download_parent(self,receipt,identity,destination):
        """Retain one independently verified parent GET when no local copy was supplied."""
        import requests, wandb
        from urllib.parse import urlparse
        from verified_download import verify_download_blocks
        remote=wandb.Api().run(receipt['run']).file(receipt['name'])
        if remote.size!=identity['bytes']:raise ValueError('private original parent size changed')
        url=remote.url
        if urlparse(url).hostname=='api.wandb.ai':
            response=requests.get(url,auth=('api',os.environ['WANDB_API_KEY']),stream=True,timeout=60)
            try:response.raise_for_status();url=response.url
            finally:response.close()
        parsed=urlparse(url)
        if parsed.scheme!='https' or parsed.hostname!='storage.googleapis.com':
            raise ValueError('original parent scoped GET unavailable')
        destination.parent.mkdir(parents=True,exist_ok=True)
        try:
            with requests.get(url,stream=True,timeout=1800) as response:
                response.raise_for_status()
                result=verify_download_blocks(response.iter_content(8<<20),identity,destination=destination,
                                              reserve_bytes=80<<30)
        except requests.RequestException:
            raise RuntimeError('original parent private GET failed') from None
        if result.get('local_copy',{}).get('committed') is not True:
            raise ValueError('original parent GET did not commit its external local copy')

    def verify_parent(self):
        from private_export_guard import validate_receipt
        transfer=self.record['run'].get('parent_transfer') or {}
        pin_path=Path(transfer['pin']);receipt_path=Path(transfer['receipt'])
        if (sha256_file_hex(pin_path)!=transfer['pin_sha256'] or
                sha256_file_hex(receipt_path)!=transfer['receipt_sha256']):
            raise ValueError('original parent pin or private receipt changed')
        pin=json.loads(pin_path.read_text());receipt=json.loads(receipt_path.read_text())
        identity={'sha256':pin['sha256'],'bytes':pin['bytes']};validate_receipt(receipt,identity)
        if (not receipt.get('verified_at') or pin['path']!=self.parent or
                self.manifest['identity']['parent_checkpoint_sha256']!=pin['sha256'] or not self.present(receipt)):
            raise ValueError('original parent private object or coordinator binding differs')
        path=self.parent_local or self.storage/'original-parent'/pin['sha256']/'checkpoint.pt'
        if self.parent_qualification is None or self.parent_copy_receipt is None:
            raise ValueError('original parent local copy and full-state qualification receipts are required')
        qualified=json.loads(self.parent_qualification.read_text())
        copied=json.loads(self.parent_copy_receipt.read_text());local=copied.get('local_copy') or {}
        if (qualified.get('schema')!='natlang.lfm3-seed-independent-full-state/1' or qualified.get('passed') is not True
                or qualified.get('sha256')!=pin['sha256'] or qualified.get('bytes')!=pin['bytes']
                or qualified.get('global_step')!=pin['global_step'] or Path(qualified.get('checkpoint','')).resolve()!=path.resolve()
                or copied.get('sha256')!=pin['sha256'] or copied.get('bytes')!=pin['bytes']
                or local.get('committed') is not True or Path(local.get('path','')).resolve()!=path.resolve()):
            raise ValueError('original parent local copy or full-state qualification receipt differs')
        if not path.resolve().is_relative_to(self.external):raise ValueError('original parent copy must be on external storage')
        if not path.exists():self.download_parent(receipt,identity,path)
        if (not path.is_file() or
                path.stat().st_size!=pin['bytes'] or sha256_file_hex(path,drop_cache=True)!=pin['sha256']):
            raise ValueError('original parent local full-state recovery copy differs')
        return {'pin':{'path':str(pin_path),'sha256':transfer['pin_sha256']},
                'receipt':{'path':str(receipt_path),'sha256':transfer['receipt_sha256']},
                'local':{'path':str(path),'sha256':pin['sha256'],'bytes':pin['bytes']},
                'qualification':{'path':str(self.parent_qualification),'sha256':sha256_file_hex(self.parent_qualification)},
                'copy_receipt':{'path':str(self.parent_copy_receipt),'sha256':sha256_file_hex(self.parent_copy_receipt)},
                'global_step':pin['global_step']}

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
    for w,r in zip(workers,residents):
        if not r:
            return {'compatible':False,'reason':'resident state is missing or differs from the worker checkpoint'}
        if w.get('resident_generation'):
            if r['generation'] != w['resident_generation']:
                return {'compatible':False,'reason':'resident state is missing or differs from the worker checkpoint'}
            continue
        initial=r.get('initial_binding') or {}
        try:
            recipe=json.loads(w['recipe_binding']) if isinstance(w['recipe_binding'],str) else w['recipe_binding']
            sources=w['source_receipt_sha256']
            matching={'train':sources[0], 'held':sources[1]} in initial['roles']
            valid=(r.get('binding_kind')=='initial-only' and r['generation']==initial['plan_sha256']
                   and recipe['lineage_sha256']==initial['recipe_sha256'] and matching)
        except (KeyError,TypeError,IndexError,ValueError):
            valid=False
        if not valid:
            return {'compatible':False,'reason':'resident state is missing or differs from the worker checkpoint'}
    versions=coord['versions']
    if any(len(w['versions'])!=len(versions) or any(a>b for a,b in zip(w['versions'],versions)) for w in workers):
        return {'compatible':False,'reason':'a worker checkpoint is ahead of the pinned coordinator'}
    ledgers=coord.get('last_seq');queued=coord.get('queued')
    if not isinstance(ledgers,list) or not isinstance(queued,list) or len(ledgers)!=len(versions) or len(queued)!=len(versions):
        return {'compatible':False,'reason':'coordinator sequence ledger is missing'}
    for worker, state in enumerate(workers):
        if len(state.get('sequences',[]))!=len(versions):
            return {'compatible':False,'reason':'worker sequence ledger is missing'}
        for index,sequence in enumerate(state['sequences']):
            merged=int(ledgers[index].get(str(worker),0));entry=queued[index].get(str(worker))
            held=max(merged,entry['sequence'] if entry else 0)
            if held>sequence:
                return {'compatible':False,'reason':f'coordinator has uncheckpointed worker {worker} fragment {index} sequence'}
            pending=state.get('pending',{}).get(str(index))
            if index in state.get('awaiting_global',[]) and held<sequence:
                return {'compatible':False,'reason':f'coordinator lacks accepted worker {worker} fragment {index} sequence'}
            if entry and pending and entry['sequence']==pending['sequence'] and entry['delta_sha256']!=pending['delta_sha256']:
                return {'compatible':False,'reason':f'queued worker {worker} fragment {index} payload differs'}
    return {'compatible':True,'scope':'separate exact worker full states and stable coordinator state; '
            'not a consolidated global checkpoint; replay and staleness remain subject to normal DiLoCo checks'}


def run_once(record,state_path,backend):
    state_path=Path(state_path);state_path.parent.mkdir(parents=True,exist_ok=True)
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        coordinator_run_id=(record['run'].get('settings') or {}).get('run_id',record['run']['id'])
        state=json.loads(state_path.read_text()) if state_path.exists() else {'schema':'natlang.diloco-recovery/1',
            'run_id':record['run']['id'],'coordinator_run_id':coordinator_run_id,'cycles':[],'latest_verified':None}
        if state['run_id']!=record['run']['id'] or state.get('coordinator_run_id')!=coordinator_run_id:
            raise ValueError('recovery state belongs to a different launch or coordinator run')
        def save():durable_json(state_path,state)
        if hasattr(backend,'verify_ownership'):backend.verify_ownership()
        if hasattr(backend,'release_remote_pins'):
            for previous in state['cycles']:
                if previous.get('pin_cleanup',{}).get('status')=='pending':
                    try:previous['pin_cleanup']=backend.release_remote_pins(previous)
                    except Exception as error:previous['pin_cleanup']={'status':'pending','error':str(error)}
                    save()
        cycle=state.get('pending')
        if cycle is None:
            cycle={'id':uuid.uuid4().hex,'phase':'pinning','components':{},'receipts':{},'errors':{}}
            state['pending']=cycle;save()
        if 'fenced' not in cycle and not cycle.get('unfenced') and hasattr(backend,'capture_fenced'):
            try:
                captured=backend.capture_fenced(cycle['id'])
                cycle['fenced']={'id':captured['id'],'deadline':captured['deadline'],
                                 'ready':{str(k):v for k,v in captured['ready'].items()}}
                cycle['components'].update(captured['components']);cycle['errors'].pop('fence',None)
                state.pop('recovery_blocker',None);save()
            except FenceUnavailable as error:
                if any(old.get('unfenced') and old.get('phase')=='incompatible-pins' for old in state['cycles']):
                    state['recovery_blocker']='unsafe unfenced cohort retained; live fence runtime required before another attempt'
                    state.pop('pending');save();return state
                cycle['unfenced']={'reason':str(error),'scope':'opportunistic sequence-checked component pins'};save()
            except Exception as error:
                cycle['phase']='incomplete';cycle['errors']['fence']=str(error);save();return state
        missing=[i for i in range(len(backend.workers)) if 'worker-'+str(i) not in cycle['components']]
        if missing:
            # A later worker checkpoint requires a newer coordinator pin; keep any old artifacts, but do not
            # bind them into this candidate as though they followed the newly pinned checkpoint.
            old=cycle['components'].pop('coordinator',None) if 'fenced' not in cycle else None
            old_receipt=cycle['receipts'].pop('coordinator',None) if old else None
            if old:cycle.setdefault('superseded_coordinator_pins',[]).append({'pin':old,'receipt':old_receipt})
            with ThreadPoolExecutor(max_workers=len(missing)) as pool:
                futures={i:pool.submit(backend.pin_worker,i,cycle['id'],cycle['fenced']['ready'][str(i)]['checkpoint'])
                         if 'fenced' in cycle else pool.submit(backend.pin_worker,i,cycle['id']) for i in missing}
                for i,future in futures.items():
                    name='worker-'+str(i)
                    try:
                        component=future.result()
                        ready=(cycle.get('fenced',{}).get('ready',{}).get(str(i)) or {})
                        if ready.get('completed') and (component['sha256']!=ready['checkpoint_sha256'] or
                                component['sequences']!=ready['sequences'] or component['pending']):
                            raise ValueError('completed worker checkpoint differs from final delivery report')
                        cycle['components'][name]=component;cycle['errors'].pop(name,None)
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
        coordinator=cycle['components'].get('coordinator') or {}
        if coordinator.get('binding_kind')=='immutable-hardlink-tree':
            try:
                cycle['components']['coordinator']=backend.archive_coordinator_tree(coordinator)
                cycle['errors'].pop('coordinator-archive',None)
            except Exception as error:
                cycle['errors']['coordinator-archive']=str(error)
                cycle['phase']='incomplete';save();return state
            save()
        for name in list(cycle['receipts']):
            if str(cycle['errors'].get(name,'')).startswith('presence: '):cycle['errors'].pop(name)
        cycle['compatibility']=compatibility(cycle['components']);cycle['phase']='exporting';save()
        if (len(cycle['components'])==2*len(backend.workers)+1 and not cycle['compatibility']['compatible']):
            cycle['phase']='incompatible-pins';cycle['recovery_blocker']=cycle['compatibility']['reason']
            state['cycles'].append(cycle);state.pop('pending');save();return state
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
                if hasattr(backend,'verify_parent'):
                    cycle['parent_reference']=backend.verify_parent();save()
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
                if cycle['compatibility']['compatible'] and hasattr(backend,'release_remote_pins'):
                    try:cycle['pin_cleanup']=backend.release_remote_pins(cycle)
                    except Exception as error:cycle['pin_cleanup']={'status':'pending','error':str(error)}
                    save()
        else:
            cycle['phase']='incomplete';save()
        return state



_HEALTH_PROBE = r"""import hashlib,json,pathlib,sys
expected=json.loads(sys.argv[1]);boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
p=pathlib.Path('/proc')/str(expected['pid']); alive=False
try:
 fields=(p/'stat').read_text().split(') ',1)[1].split();alive=boot==expected['boot_id'] and fields[19]==str(expected['start_ticks']) and fields[0]!='Z'
except (OSError,ValueError,IndexError):pass
progress=pathlib.Path('/work/diloco-learner/progress.json');value=None
try:
 value=json.loads(progress.read_text());value={k:value.get(k) for k in ('global_step','step','timestamp','checkpoint_step')}
except (OSError,ValueError):pass
module=pathlib.Path(sys.argv[2])
try: module_sha=hashlib.sha256(module.read_bytes()).hexdigest()
except OSError: module_sha=None
print(json.dumps({'boot_id':boot,'pid_alive':alive,'progress':value,'runtime_module_sha256':module_sha}))"""

def health_snapshot(record, ownership, *, read_sandbox=None):
    """Actual sandbox status, boot/PID ownership and last optimizer progress."""
    read_sandbox=read_sandbox or (lambda ident: cwctl.read_sandbox(ident, patience=30))
    workers=ownership.get('workers') or {};ids=record['learners']
    def check(i):
        role=workers.get(str(i)) or {};ident=ids[i]
        if role.get('sandbox_id')!=ident or not all(k in role for k in ('pid','boot_id','start_ticks')):
            return str(i),{'sandbox_id':ident,'state':'ownership-unavailable'}
        try:
            box=read_sandbox(ident);box.get_status();status=str(getattr(box.status,'value',box.status)).lower()
            module=(ownership.get('runtime') or {}).get('module_path') or str(
                paths.root('repo')/'training/neuralese/natlang_neuralese/train/native_state_sft.py')
            r=box.exec(['python3','-c',_HEALTH_PROBE,json.dumps(role),module],timeout_seconds=30).result(timeout=40)
            if r.returncode:raise ValueError('health probe failed')
            data=json.loads(r.stdout);data.update(sandbox_id=ident,sandbox_status=status)
            expected_sha=(ownership.get('runtime') or {}).get('new_module_sha256')
            data['runtime_matches']=bool(expected_sha and data.get('runtime_module_sha256')==expected_sha)
            data['state']='live' if status=='running' and data['pid_alive'] and data['runtime_matches'] else 'dead-replaced-or-runtime-mismatch'
            return str(i),data
        except Exception as error:return str(i),{'sandbox_id':ident,'state':'unavailable','error_type':type(error).__name__}
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:return dict(pool.map(check,range(len(ids))))

def health_loop(record, ownership_path, destination, interval):
    while True:
        try:
            ownership=json.loads(Path(ownership_path).read_text());workers=health_snapshot(record,ownership)
            current=json.loads((cwctl.STATE/'diloco'/(record['run']['id']+'.json')).read_text())
            value={'schema':'natlang.diloco-worker-health/1','run_id':record['run']['id'],'journal_state':current.get('state'),'at':time.time(),'workers':workers}
            durable_json(destination,value)
            print(json.dumps({'event':'worker_health','workers':workers},sort_keys=True),flush=True)
        except Exception as error:
            print(json.dumps({'event':'worker_health_unavailable','error_type':type(error).__name__}),flush=True)
        time.sleep(interval)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_id');p.add_argument('--state',type=Path,required=True)
    p.add_argument('--storage',type=Path,required=True);p.add_argument('--parent-local',type=Path,required=True);
    p.add_argument('--parent-qualification',type=Path,required=True);p.add_argument('--parent-copy-receipt',type=Path,required=True);p.add_argument('--interval',type=float,default=1800)
    p.add_argument('--once',action='store_true');p.add_argument('--ownership-journal',type=Path);p.add_argument('--health-interval',type=float,default=60);p.add_argument('--wait-worker-progress',action='store_true');p.add_argument('--runtime-bundle',type=Path);p.add_argument('--health-only',action='store_true');args=p.parse_args()
    if args.interval <= 0:p.error('interval must be positive')
    cwctl.load_env_file()
    import diloco
    record=json.loads((diloco.RUNS/(args.run_id+'.json')).read_text())
    if args.health_interval<=0:p.error('health interval must be positive')
    if args.ownership_journal:
        health_path=args.state.with_name(args.state.stem+'-health.json')
        threading.Thread(target=health_loop,args=(record,args.ownership_journal,health_path,args.health_interval),daemon=True).start()
    if args.health_only:
        if not args.ownership_journal:p.error('health-only requires an ownership journal')
        while True:time.sleep(args.health_interval)
    if args.wait_worker_progress:
        if not args.ownership_journal:p.error('waiting for worker progress requires an ownership journal')
        while True:
            try:
                health=json.loads(health_path.read_text());workers=health['workers']
                if len(workers)==len(record['learners']) and all(v['state']=='live' and isinstance(v.get('progress'),dict) and type(v['progress'].get('global_step')) is int for v in workers.values()):break
            except (OSError,ValueError,KeyError):pass
            time.sleep(args.health_interval)
    backend=Backend(record,args.storage,args.parent_local,args.parent_qualification,args.parent_copy_receipt,runtime_bundle=args.runtime_bundle)
    while True:
        try:
            state=run_once(record,args.state,backend)
            cycle=state.get('pending') or state['cycles'][-1]
            print(json.dumps({'cycle':cycle['id'],'phase':cycle['phase'],'errors':cycle['errors'],
                              'latest_exported':state.get('latest_exported'),
                              'recovery_blocker':state.get('recovery_blocker') or cycle.get('recovery_blocker'),
                              'observed_progress':cycle.get('observed_progress')},sort_keys=True),flush=True)
        except Exception as error:
            print(json.dumps({'error':type(error).__name__,'detail':str(error)}),flush=True)
            if args.once:raise
        if args.once:break
        time.sleep(args.interval)


if __name__=='__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):cwctl.reexec_in_venv()
    main()
