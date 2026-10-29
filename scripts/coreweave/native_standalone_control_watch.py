#!/usr/bin/env python3
"""External standalone checkpoint watcher with an exact per-role control selection."""
import hashlib,json,subprocess,sys,time,fcntl,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));from native_remote_budget import bounded_remote
sys.path.insert(0,'/home/werg/natlang/scripts/coreweave');import cwctl
from natlang_neuralese.common.jsonio import write_json_durable

def bound(ref):
 path=Path(ref['path']);raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
 if actual!=ref['sha256']:raise ValueError('selected watcher dependency changed: '+str(path))
 return raw
selection_path=Path(sys.argv[1]);selection_sha=sys.argv[2];selector_receipt=Path(sys.argv[3]) if len(sys.argv)>3 else None;selector_receipt_sha=sys.argv[4] if len(sys.argv)>4 else None;raw=selection_path.read_bytes()
if hashlib.sha256(raw).hexdigest()!=selection_sha:raise ValueError('watcher selection changed')
selection=json.loads(raw)
if selection.get('schema')!='natlang.native-standalone-control-watcher/1':raise ValueError('invalid watcher selection')
rank=selection['rank'];root=Path(selection['state_root']);journal=Path(selection['journal']);owner=Path(selection['owner_journal'])
s=json.loads(journal.read_text()) if journal.exists() else {'last':selection['initial_step'],'exports':[]}
for prior in s['exports']:
 if prior.get('control_required') and prior.get('control',{}).get('status')!='independently_verified' and prior.get('control_selection_sha256')!=selection_sha:
  raise ValueError('unfinished historical control belongs to a different or unbound selection')
code=bound(selection['scanner']).decode()
OWNER_GUARD = r'''import json,os,pathlib,sys
expected=json.loads(sys.argv[3]);pid=expected['pid']
try:
 boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 fields=pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
 if boot!=expected['boot_id'] or fields[0]=='Z' or fields[19]!=str(expected['process_start_ticks']):raise ValueError('owner identity differs')
except (OSError,ValueError,IndexError,KeyError,TypeError):
 print(json.dumps({'ready':False,'reason':'owner process identity unavailable or changed'}));raise SystemExit(0)
'''
code=bounded_remote(OWNER_GUARD+code);compile(code,'<remote-scanner>','exec')
hook=None;hook_error=None
try:
 hook_ref=selection['hook'];hook_raw=bound(hook_ref);name='selected_native_control_hook_'+hook_ref['sha256'];hook=types.ModuleType(name);hook.__file__=hook_ref['path'];sys.modules[name]=hook;exec(compile(hook_raw,hook_ref['path'],'exec'),hook.__dict__)
except Exception as error:
 hook=None;hook_error=type(error).__name__+': '+str(error)[:400]
cwctl.load_env_file()

def validate_owner(actual, expected):
 for key in ('sandbox_id','plan_sha256','runtime_commit','source_sha256'):
  if actual.get(key)!=expected.get(key):raise ValueError('owner '+key+' differs from selected role')
 accepted=actual.get('accepted') or {};identity=expected.get('accepted') or {}
 for key in ('owner_id','generation','boot_id','pid','process_start_ticks'):
  if accepted.get(key)!=identity.get(key):raise ValueError('owner '+key+' differs from selected role')
 if accepted.get('accepted') is not True:raise ValueError('selected owner is not accepted')

def minimum_step(saved, selected, parent):
 if any(type(x) is not int or x < 0 for x in (saved, selected, parent)):
  raise ValueError('invalid saved, selected or owner parent step')
 return max(saved,selected,parent)

initial_owner=json.loads(owner.read_text());validate_owner(initial_owner,selection['owner_binding'])
s['last']=minimum_step(s.get('last',0),selection['initial_step'],initial_owner['parent_step']);write_json_durable(journal,s)

def control_attempt(v,b,e):
 if hook is None:raise ValueError('selected control hook unavailable: '+hook_error)

 def presence(role,archive):
  try:
   import wandb
   registry=Path('/home/werg/data/dgxops')/('coreweave-production-'+role+'-private-export-run-20261014.json')
   rid=json.loads(registry.read_text())['run_id'];name=role+'/'+archive['sha256']+'/'+Path(archive['path']).name
   return wandb.Api().run('werg/natlang/'+rid).file(name).size==archive['bytes']
  except Exception:
   if b is None:raise
   return False
 def launch(role,unit,sandbox_id,archive,existing):
  env={'CW_EXPORT_SANDBOX_ID':sandbox_id,'CW_EXPORT_EXPECTED_SHA256':archive['sha256'],'CW_EXPORT_EXPECTED_BYTES':str(archive['bytes']),'CW_EXPORT_CACHE_BOUNDED_READS':'1','CW_EXPORT_VERIFY_ON_OWNER':'1'}
  if existing:env['CW_EXPORT_VERIFY_EXISTING_ONLY']='1'
  subprocess.run(['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G',*[f'--setenv={k}={x}' for k,x in env.items()],'/home/werg/.local/share/natlang/cw-venv/bin/python','/home/werg/data/dgxops/coreweave-private-runfile-export-20261014.py',role,archive['path']],check=True)
 hook.attempt_control(v,b,e,s,journal,selection['hook_selection'],write_json_durable,presence=presence,launch=launch)

def refresh_publication(e):
 import hashlib,wandb
 dependencies=[(v['role'],v['receipt_sha256']) for v in selection['publication_dependencies']]
 if not e.get('full_backup_verified') or e.get('control',{}).get('status')!='independently_verified':e['publication_complete']=False;return
 try:
  if selector_receipt is None or selector_receipt_sha is None:raise ValueError('independent selector receipt not declared')
  selector_raw=selector_receipt.read_bytes()
  if hashlib.sha256(selector_raw).hexdigest()!=selector_receipt_sha:raise ValueError('selector receipt changed')
  selector=json.loads(selector_raw);selector_identity=selector['identity'];selector_get=selector['independent_verification']
  if not selector.get('private') or selector_identity['sha256']!=selection_sha or selector_get.get('readback_location')!='owner' or selector_get.get('sha256')!=selection_sha or selector_get.get('bytes')!=selector_identity['bytes']:
   raise ValueError('selector lacks exact private owner readback')
  checks=dependencies+[(e['role'],None),(e['control']['role'],None)];bindings=[];api=wandb.Api()
  if api.run(selector['run']).file(selector['name']).size!=selector_identity['bytes']:raise ValueError('selector private object missing')
  bindings.append({'role':selector['role'],'receipt':str(selector_receipt),'receipt_sha256':selector_receipt_sha,'sha256':selection_sha,'bytes':selector_identity['bytes']})
  for role,expected in checks:
   path=Path('/home/werg/data/dgxops')/('coreweave-production-'+role+'-private-export-receipt-20261014.json');raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
   if expected and digest!=expected:raise ValueError('dependency receipt changed: '+role)
   d=json.loads(raw);identity=d['identity'];v=d['independent_verification']
   if not d.get('private') or v.get('readback_location')!='owner' or v.get('sha256')!=identity['sha256'] or v.get('bytes')!=identity['bytes']:raise ValueError('receipt lacks owner full GET: '+role)
   if role==e['role'] and identity['sha256']!=json.loads(Path(e['intent']).read_text())['sha256']:raise ValueError('full receipt checkpoint differs')
   if role==e['control']['role'] and identity['sha256']!=e['control']['archive']['sha256']:raise ValueError('control receipt differs')
   if api.run(d['run']).file(d['name']).size!=identity['bytes']:raise ValueError('current private object missing: '+role)
   bindings.append({'role':role,'receipt':str(path),'receipt_sha256':digest,'sha256':identity['sha256'],'bytes':identity['bytes']})
  e['publication_dependencies']=bindings;e['publication_complete']=True;e.pop('publication_error',None)
 except Exception as error:e['publication_complete']=False;e['publication_error']=str(error)[:500]
 write_json_durable(journal,s)

while True:
 try:
  for e in s['exports']:
   if e.get('control_required') and e.get('control_selection_sha256')==selection_sha and e.get('control',{}).get('status')!='independently_verified':
    try:
     v=json.loads(Path(e['intent']).read_text());o=json.loads(owner.read_text());b=None
     if o['sandbox_id']==v['sandbox_id']:
      try:b=cwctl.read_sandbox(o['sandbox_id'])
      except Exception:pass
     control_attempt(v,b,e)
    except Exception as error:
     e.setdefault('control',{})['error']=str(error)[:500];write_json_durable(journal,s);print('control retry pending',type(error).__name__,str(error)[:500],flush=True)
  if s['exports']:
   e=s['exports'][-1]
   if e.get('full_launch_pending'):
    v=json.loads(Path(e['intent']).read_text());env={'CW_EXPORT_SANDBOX_ID':v['sandbox_id'],'CW_EXPORT_EXPECTED_SHA256':v['sha256'],'CW_EXPORT_EXPECTED_BYTES':str(v['bytes']),'CW_EXPORT_CACHE_BOUNDED_READS':'1'}
    if subprocess.run(['systemctl','--user','is-active','--quiet',e['unit']]).returncode!=0:
     import uuid
     e['unit']=e['role']+'-export-recover-'+uuid.uuid4().hex;write_json_durable(journal,s)
     subprocess.run(['systemd-run','--user','--unit='+e['unit'],'--property=MemoryMax=2G',*[f'--setenv={k}={x}' for k,x in env.items()],'/home/werg/.local/share/natlang/cw-venv/bin/python','/home/werg/data/dgxops/coreweave-private-runfile-export-20261014.py',e['role'],v['path']],check=True)
    e['full_launch_pending']=False;write_json_durable(journal,s)
   p=Path(e['phase_path']);phase=json.loads(p.read_text()).get('phase') if p.exists() else None
   e['full_backup_verified']=phase in ('independently_verified','reused_independently_verified');e.setdefault('publication_complete',False);refresh_publication(e);write_json_durable(journal,s)
   if phase not in ('independently_verified','reused_independently_verified'):
    unit=e.get('retry_unit',e['unit']);active=subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0
    if not active:
     v=json.loads(Path(e['intent']).read_text());import wandb
     try:
      rid=json.loads((Path('/home/werg/data/dgxops')/('coreweave-production-'+e['role']+'-private-export-run-20261014.json')).read_text())['run_id'];name=e['role']+'/'+v['sha256']+'/'+Path(v['path']).name;present=wandb.Api().run('werg/natlang/'+rid).file(name).size==v['bytes']
     except Exception as err:print('object check unavailable',type(err).__name__,flush=True);time.sleep(60);continue
     n=int(e.get('retry_count',0))+1;unit=e['unit']+'-retry-'+str(n);env={'CW_EXPORT_SANDBOX_ID':v['sandbox_id'],'CW_EXPORT_EXPECTED_SHA256':v['sha256'],'CW_EXPORT_EXPECTED_BYTES':str(v['bytes']),'CW_EXPORT_CACHE_BOUNDED_READS':'1'}
     if present:env['CW_EXPORT_VERIFY_EXISTING_ONLY']='1'
     subprocess.run(['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G',*[f'--setenv={k}={x}' for k,x in env.items()],'/home/werg/.local/share/natlang/cw-venv/bin/python','/home/werg/data/dgxops/coreweave-private-runfile-export-20261014.py',e['role'],v['path']],check=True);e.update(retry_count=n,retry_unit=unit);write_json_durable(journal,s)
    time.sleep(60);continue
  with (root/'standalone-cloud-retention.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   owner_state_path=root/'standalone-cloud-owner-verification.json';verified=set(json.loads(owner_state_path.read_text()).get('verified_roles',[])) if owner_state_path.exists() else set();retired_path=root/'standalone-retired-cloud-objects.json';retired=json.loads(retired_path.read_text()) if retired_path.exists() else {};blocked=set(retired.get('retired_roles',[]))|{x['role'] for x in retired.get('operations',[]) if x.get('phase') in ('intent','deleted')};unverified=[e for e in s['exports'] if e['role'] not in verified and e['role'] not in blocked]
  if len(unverified)>=4:print('cloud queue capped: owner verification behind',rank,flush=True);time.sleep(60);continue
  o=json.loads(owner.read_text());validate_owner(o,selection['owner_binding']);b=cwctl.read_sandbox(o['sandbox_id']);r=b.exec(['python3','-c',code,str(rank),str(s['last']),json.dumps(o['accepted'],sort_keys=True)],timeout_seconds=900).result(timeout=930)
  if r.returncode:print('scan failed',r.stderr[-500:],flush=True);time.sleep(60);continue
  v=json.loads(r.stdout.strip().splitlines()[-1]);
  if not v['ready']:time.sleep(60);continue
  v.update(sandbox_id=o['sandbox_id'],worker=rank,runtime_commit=o['runtime_commit'],plan_sha256=o['plan_sha256'],owner=o['accepted'],outer_credit=False,scope='cloud readback intended; owner GET remains separately required');intent=root/('outerloss-cloud-peer%d-step%d-intent.json'%(rank,v['step']));write_json_durable(intent,v);name='lfm3-outerloss-cloud-peer%d-step%d'%(rank,v['step']);unit=name+'-export';env={'CW_EXPORT_SANDBOX_ID':v['sandbox_id'],'CW_EXPORT_EXPECTED_SHA256':v['sha256'],'CW_EXPORT_EXPECTED_BYTES':str(v['bytes']),'CW_EXPORT_CACHE_BOUNDED_READS':'1'};e={'worker':rank,'step':v['step'],'intent':str(intent),'unit':unit,'role':name,'phase_path':'/home/werg/data/dgxops/coreweave-'+name+'-'+Path(v['path']).name+'-export-progress-20261014.json','control_required':True,'control_selection_sha256':selection_sha,'full_launch_pending':True,'full_backup_verified':False,'publication_complete':False};s['exports'].append(e);s['last']=v['step'];write_json_durable(journal,s)
  try:control_attempt(v,b,e)
  except Exception as error:e.setdefault('control',{})['error']=str(error)[:500];write_json_durable(journal,s);print('control pending; journaled full-state fallback proceeds',type(error).__name__,str(error)[:500],flush=True)
  subprocess.run(['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G',*[f'--setenv={k}={x}' for k,x in env.items()],'/home/werg/.local/share/natlang/cw-venv/bin/python','/home/werg/data/dgxops/coreweave-private-runfile-export-20261014.py',name,v['path']],check=True);e['full_launch_pending']=False;write_json_durable(journal,s);print('cloud export started',rank,v['step'],flush=True)
 except Exception as e:print('watch error',type(e).__name__,str(e),flush=True)
 time.sleep(60)
