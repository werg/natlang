#!/usr/bin/env python3
"""Continue a paused cohosted DiLoCo stage after a verified seed and live-volume snapshot.

Run on the supervisor, through its memory ledger. This command never deletes or restarts the
cohost. It caches exact bundle objects without modifying existing live files, captures an idempotent
snapshot, starts workers 1..N-1 from that snapshot with isolated control directories, verifies the
actual restored parent bytes, and independently downloads a full parent copy. A READY snapshot is
not a restore proof; a verified local download is not a private object-store export.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwctl
import diloco
import recovery_snapshots
from recovery_inputs import durable_json

CACHE_PROGRAM = r'''import hashlib,json,os,pathlib,shutil,sys,tempfile
work=pathlib.Path(sys.argv[1]); cw=work/'.coreweave'; repo=pathlib.Path(sys.argv[2])
manifest=json.loads((cw/'bundle.json').read_text()); cache=cw/'cas'; cache.mkdir(parents=True,exist_ok=True)
items=[i for i in manifest['files'] if not i.get('url')]
if manifest.get('code'): items.append(dict(manifest['code'],path=str(cw/'code.tar.gz')))
records=[]
for item in items:
 source=pathlib.Path(item['path']); source=source if source.is_absolute() else repo/source
 target=cache/item['sha256']; expected=item['sha256']; size=item['bytes']
 def verify(p):
  if p.stat().st_size!=size:return False
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==expected
 if target.exists():
  if not verify(target):raise ValueError('cache object corrupt: '+str(target))
 else:
  if not verify(source):raise ValueError('bundle input unavailable or changed: '+str(source))
  fd,name=tempfile.mkstemp(prefix=expected+'.',suffix='.partial',dir=cache)
  try:
   with os.fdopen(fd,'wb') as o,source.open('rb') as i:
    shutil.copyfileobj(i,o,length=1024*1024);o.flush();os.fsync(o.fileno())
   if not verify(pathlib.Path(name)):raise ValueError('cache copy mismatch')
   try:os.link(name,target)
   except FileExistsError:
    if not verify(target):raise ValueError('concurrent cache object mismatch')
  finally:pathlib.Path(name).unlink(missing_ok=True)
 records.append({'sha256':expected,'bytes':size})
print(json.dumps({'objects':len(records),'bytes':sum(i['bytes'] for i in records),'inventory':records}))
'''


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}


class Cloud:
    def __init__(self, sandbox_id, parent, backup_dir):
        self.sandbox_id, self.parent, self.backup_dir = sandbox_id, parent, Path(backup_dir)

    def sandbox(self, ident=None): return cwctl.read_sandbox(ident or self.sandbox_id, patience=45)

    def seed_receipt(self):
        raw = self.sandbox().read_file(self.parent + '.seed.json').result()
        value = json.loads(raw)
        if (value.get('checkpoint') != self.parent or value.get('global_step', 0) < value.get('requested_stop', 1)
                or not value.get('full_optimizer_rng_exposure') or len(value.get('sha256', '')) != 64):
            raise ValueError('cohost seed receipt does not bind a full earned checkpoint')
        return value

    def cache(self):
        result = self.sandbox().exec(['python3','-c',CACHE_PROGRAM,'/work','/home/werg/natlang']).result()
        if result.returncode: raise RuntimeError('live bundle cache failed: ' + result.stderr[-1000:])
        return json.loads(result.stdout)

    def capture(self, request_id):
        return self.sandbox().snapshot(wait_for_ready=False, request_id=request_id).result(timeout=45)

    def inspect(self, snapshot_id):
        from cwsandbox import Sandbox
        s = Sandbox.get_snapshot(snapshot_id, auth=cwctl.sdk_auth(), timeout_seconds=30).result(timeout=45)
        return {'snapshot_id':snapshot_id, 'status':str(s.status).upper(),
                'source_sandbox_id':s.source_sandbox_id, 'size_bytes':s.size_bytes}

    def download(self, receipt):
        script = Path(__file__).with_name('recovery_pull.py')
        command = [sys.executable, str(script), 'file', self.sandbox_id, self.parent, str(self.backup_dir),
                   '--owner', 'lfm3-v3-seed', '--keep', '0']
        subprocess.run(command, check=True)
        actual = json.loads((self.backup_dir/'latest'/'receipt.json').read_text())
        if not actual.get('verified') or actual['sha256'] != receipt['sha256'] or actual['bytes'] != receipt['bytes']:
            raise ValueError('independent seed download differs from earned checkpoint')
        # Decode real full-state metadata from the independently downloaded bytes on the workstation.
        code = ("import json,sys; from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy; "
                "s=CheckpointPolicy.load(sys.argv[1]);t=s['training']; "
                "assert t['global_step']==int(sys.argv[2]) and t['optimizer'] and t['rng'] and t['exposure_counts']; "
                "print(json.dumps({'step':t['global_step']}))")
        env = {**os.environ, 'PYTHONPATH':'/home/werg/natlang/training/neuralese:/home/werg/natlang'}
        subprocess.run(['/home/werg/natlang/.venv-neuralese/bin/python','-c',code,actual['local'],str(receipt['global_step'])],
                       env=env,check=True,capture_output=True,text=True)
        return actual

    def verify_restore(self, snapshot, worker, receipt):
        from cwsandbox import Sandbox
        sandbox = self.sandbox(worker)
        def inspect_destination():
            value = Sandbox.from_id(worker,auth=cwctl.sdk_auth()).result(timeout=45)
            return [v.scratch.restore_from_snapshot_id for v in value.spec.volumes
                    if v.scratch and v.scratch.restore_from_snapshot_id]
        def execute(command):
            result = sandbox.exec(command).result(timeout=1800)
            if result.returncode: raise RuntimeError('restored file verification failed: '+result.stderr)
            return json.loads(result.stdout)
        return recovery_snapshots.verify_restored_files(self.sandbox_id,snapshot,worker,
            [{'path':self.parent,'sha256':receipt['sha256'],'bytes':receipt['bytes']}],
            inspect_destination=inspect_destination,execute=execute)


def run(record_path, journal_path, cloud, *, wait_seconds=21600, interval=5):
    record_path, journal_path = Path(record_path), Path(journal_path)
    journal_path.parent.mkdir(parents=True,exist_ok=True)
    with journal_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(journal_path.read_text()) if journal_path.exists() else {
            'schema':'natlang.diloco-post-seed/1','sandbox_id':cloud.sandbox_id,'parent':cloud.parent,'phase':'waiting'}
        if state['sandbox_id']!=cloud.sandbox_id or state['parent']!=cloud.parent:
            raise ValueError('post-seed journal belongs to a different sandbox or parent')
        def save(): durable_json(journal_path,state)
        save(); deadline=time.monotonic()+wait_seconds
        while True:
            record=json.loads(record_path.read_text())
            owner = record.get('coordinator')
            if owner is not None and owner!=cloud.sandbox_id: raise ValueError('coordinator sandbox changed')
            if record['run']['coordinator']['parent'] != cloud.parent: raise ValueError('coordinator parent changed')
            if owner and record.get('state') in ('coordinator-ready','running'): break
            if record.get('state','').startswith('failed'): raise ValueError('cohost launch failed: '+str(record.get('error')))
            if time.monotonic()>deadline: raise TimeoutError('cohost did not reach coordinator-ready')
            time.sleep(interval)
        receipt=cloud.seed_receipt()
        if state.get('seed') and state['seed']!=receipt: raise ValueError('seed receipt changed on resume')
        state['seed']=receipt;state['phase']='seed-verified';save()
        with ThreadPoolExecutor(max_workers=2) as pool:
            future=pool.submit(cloud.download,receipt) if not state.get('download') else None
            # The same sandbox ID can have an empty workspace after a reset: recheck the live cache
            # until an actual READY snapshot is recorded, rather than trusting an old enqueue acknowledgement.
            if not state.get('snapshot'):
                state['cache']=cloud.cache();state['phase']='cache-verified';save()
            if not state.get('snapshot_request_id'):
                state['snapshot_request_id']=uuid.uuid4().hex;state['phase']='snapshot-intent';save()
            if not state.get('snapshot_id'):
                state['snapshot_id']=cloud.capture(state['snapshot_request_id']);state['phase']='snapshot-requested';save()
            if not isinstance(state['snapshot_id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', state['snapshot_id']):
                raise ValueError('invalid snapshot identifier')
            while True:
                snapshot=cloud.inspect(state['snapshot_id'])
                if snapshot['source_sandbox_id']!=cloud.sandbox_id: raise ValueError('snapshot source differs')
                if snapshot['status']=='FAILED': raise RuntimeError('seed snapshot failed')
                if snapshot['status']=='READY': break
                if time.monotonic()>deadline: raise TimeoutError('seed snapshot not READY')
                time.sleep(interval)
            state['snapshot']=snapshot;state['phase']='snapshot-ready';save()
            record=json.loads(record_path.read_text())
            if record['state']=='coordinator-ready':
                old_path=Path(record['run']['learners']['job'])
                if state.get('restored_job'):
                    new_path=Path(state['restored_job']['path'])
                    if bind(new_path)!=state['restored_job']: raise ValueError('restored job changed')
                else:
                    original=bind(old_path); definition=json.loads(old_path.read_text())
                    definition['scratch']['restore_from_snapshot_id']=state['snapshot_id']
                    definition['env']={**definition.get('env',{}),'NATLANG_DILOCO_SEED_SHA256':receipt['sha256']}
                    definition['lifetime_hours']=720
                    new_path=journal_path.parent/('learner-restored-'+state['snapshot_id']+'.json')
                    if new_path.exists() and json.loads(new_path.read_text())!=definition:
                        raise ValueError('immutable restored learner job differs')
                    if not new_path.exists(): durable_json(new_path,definition)
                    state['original_job']=original;state['restored_job']=bind(new_path);save()
                record['run']['learners']['job']=str(new_path)
                record['seed_snapshot']={'id':state['snapshot_id'],'status':'READY','source_sandbox_id':cloud.sandbox_id,
                                         'original_job':state['original_job'],'restored_job':state['restored_job']}
                durable_json(record_path,record)
                state['phase']='launching-remaining';save()
                record=diloco.launch_remaining(record['run']['id'])
            elif record['state']!='running':
                raise ValueError('cannot safely relaunch partial worker assignments: '+record['state'])
            state['learners']=record['learners'];state['phase']='workers-started';save()
            restored=state.setdefault('restored',{})
            for i, worker in enumerate(record['learners'][1:],1):
                if str(i) not in restored:
                    restored[str(i)]=cloud.verify_restore(state['snapshot_id'],worker,receipt);save()
            if future:
                state['download']=future.result();save()
        state['phase']='complete';save()
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id');parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--backup-dir',type=Path,required=True);parser.add_argument('--wait-seconds',type=float,default=21600)
    args=parser.parse_args();cwctl.load_env_file()
    path=diloco.RUNS/(args.run_id+'.json');record=json.loads(path.read_text())
    cloud=Cloud(record['coordinator'],record['run']['coordinator']['parent'],args.backup_dir)
    print(json.dumps(run(path,args.journal,cloud,wait_seconds=args.wait_seconds),indent=1))


if __name__=='__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):cwctl.reexec_in_venv()
    main()
