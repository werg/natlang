"""Export owned full native student checkpoints without pausing training."""
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


def bound_output_checkpoint(reference,role,*,production_root='/work/native-production',control_root='/work/native-production/control'):
    """Resolve one byte-bound owner output within its configured role namespace."""
    if role not in ('lfm','mellum'):raise ValueError('unknown checkpoint role')
    raw=pathlib.Path(reference['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('owner argv changed')
    argv=json.loads(raw)
    if not isinstance(argv,list) or argv.count('--out')!=1:raise ValueError('owner needs exactly one output')
    index=argv.index('--out')
    if index+1>=len(argv) or not isinstance(argv[index+1],str):raise ValueError('owner output is missing')
    output=pathlib.Path(argv[index+1]).resolve()
    standard=pathlib.Path(production_root)/('mellum21'if role=='mellum'else'lfm25')
    allowed=pathlib.Path(reference.get('allowed_output_root',str(standard))).resolve()
    if allowed!=standard:
        if (allowed.parent!=pathlib.Path(control_root).resolve()
                or re.fullmatch(re.escape(role)+r'(?:[0-9a-f]{3,40})?-.+',allowed.name)is None):
            raise ValueError('configured control output belongs to another role')
    if not output.is_relative_to(allowed):raise ValueError('owner output escapes configured root')
    return output/'checkpoint.pt'


REMOTE_SOURCE = "import pathlib,json,os,time,torch,subprocess,gc\nimport fcntl,uuid\nroot=pathlib.Path(ROOT);remote_lock=(root/f'{ROLE}-cursor-durable-export.lock').open('a');fcntl.flock(remote_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);paths=set(root.glob(f'rounds/{ROLE}/production/round-*/learner/checkpoint.pt'))|set(root.glob(f'rounds/{ROLE}/cycles/*/supervised/*/learner/checkpoint.pt'))\nfor candidate in paths:\n if not candidate.resolve().is_relative_to(root.resolve()/'rounds'/ROLE):raise ValueError('canonical checkpoint escaped its role namespace')\ncursor_path=root/f'rounds/{ROLE}/cycle.json';cursor=json.loads(cursor_path.read_text())if cursor_path.exists()else {};c=cursor.get('current_checkpoint');c=c.get('path')if isinstance(c,dict)else c\nif EXTRA is not None:paths.add(bound_output_checkpoint(EXTRA,ROLE))\nif isinstance(c,str):\n p=pathlib.Path(c).resolve()\n if p.is_relative_to(root/f'rounds/{ROLE}'):paths.add(p)\nchanged=[]\nfor p in paths:\n if not p.is_file():continue\n s=p.stat();identity=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]\n if KNOWN.get(str(p),{}).get('stat')!=identity:changed.append((s.st_mtime_ns,p,identity))\nupdates={};chosen=None\nfor _,p,identity in sorted(changed,reverse=True):\n available=int(next(x.split()[1]for x in pathlib.Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024\n if available<MIN_AVAILABLE_BYTES:break\n # Copy first: atomic slot replacement may race, but the snapshot's own saved step is decisive.\n temp=pathlib.Path(PIN_DIRECTORY)/('.'+ROLE+'-cursor-observation-'+uuid.uuid4().hex+'.pt');temp.parent.mkdir(parents=True,exist_ok=True)\n try:\n  subprocess.run(['cp','--reflink=auto','--preserve=mode,timestamps','--',str(p),str(temp)],check=True);assert temp.stat().st_ino!=p.stat().st_ino\n  with temp.open('rb')as snapshot:os.fsync(snapshot.fileno())\n  with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):value=torch.load(temp,map_location='cpu',mmap=True,weights_only=True)\n  training=value['training'];assert training['optimizer']and training['rng']and training['source_receipts'];step=int(value['step']);del value,training;gc.collect();updates[str(p)]={'stat':identity,'step':step}\n  if step>LAST:\n   target=temp.with_name(f'{ROLE}-native-checkpoint-step-'+str(step)+'.pt')\n   if target.exists():\n    import hashlib\n    assert target.stat().st_size==temp.stat().st_size and hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==hashlib.file_digest(temp.open('rb'),'sha256').hexdigest(),'existing immutable step pin differs'\n    temp.unlink()\n   else:os.replace(temp,target)\n   fd=os.open(target.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd);chosen={'source':str(p),'pin':str(target),'step':step,'bytes':target.stat().st_size};break\n  temp.unlink()\n finally:\n  if temp.exists():temp.unlink()\nprint(json.dumps({'at':time.time(),'observations':updates,'chosen':chosen,'cursor_binding_only':c}))"


def save(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def paths_for(directory, role):
    directory = pathlib.Path(directory)
    if role not in ('lfm', 'mellum'): raise ValueError('unknown checkpoint role')
    return {'state': directory/f'coreweave-native-{role}-periodic-export-state-20261015.json',
            'cache': directory/f'{role}-cursor-export-source-observations-20261017.json',
            'pending': directory/f'{role}-cursor-export-pending-20261017.json',
            'receipt': directory/f'{role}-cursor-current-private-receipt-20261017.json',
            'error': directory/f'{role}-cursor-periodic-export-last-error-20261017.json'}


def run_once(directory, role, *, observe, export):
    """Recover a pending operation first; acknowledge only independent full GET."""
    paths = paths_for(directory, role)
    state = json.loads(paths['state'].read_text()) if paths['state'].exists() else {'last_verified_step': -1}
    observed = json.loads(paths['cache'].read_text()) if paths['cache'].exists() else {}
    if paths['pending'].exists():
        chosen = json.loads(paths['pending'].read_text())
        if chosen['step'] <= state['last_verified_step']:
            paths['pending'].unlink()
            return None
    else:
        result = observe(known=observed, last=state['last_verified_step'])
        chosen = result['chosen']
        if chosen: save(paths['pending'], chosen)
        observed.update(result['observations']); save(paths['cache'], observed)
    if not chosen: return None
    receipt = export(chosen, paths['receipt'])
    identity = receipt['identity']; verification = receipt['independent_verification']
    if (receipt['source'] != chosen['pin'] or identity['bytes'] != chosen['bytes']
            or verification.get('sha256') != identity['sha256']
            or verification.get('bytes') != identity['bytes']
            or re.fullmatch('[0-9a-f]{64}', identity['sha256']) is None):
        raise ValueError('checkpoint exporter lacks exact independent full readback')
    result = {'last_verified_step': chosen['step'], 'receipt': receipt, 'at': time.time(),
              'source_authority': f'actual full optimizer/RNG checkpoint in canonical {role} model worker/cursor namespace',
              'source': chosen['source']}
    if chosen.get('actor_view') is not None:
        result['actor_view'] = chosen['actor_view']
    if chosen.get('actor_view_error') is not None:
        result['actor_view_error'] = chosen['actor_view_error']
    if chosen.get('training_summary') is not None:
        result['training_summary'] = chosen['training_summary']
    save(paths['state'], result)
    paths['pending'].unlink()
    fd = os.open(paths['pending'].parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return {'verified_step': chosen['step'], 'sha256': identity['sha256'], 'source': chosen['source']}


def remote_code(*, role, root, known, last, extra, pin_directory, minimum_available_bytes,
                actor_view_directory=None, actor_view_runtime=None,
                actor_view_sidecar=None, actor_view_sidecar_sha256=None,
                training_summary_spec=None, training_summary_runtime=None):
    source = REMOTE_SOURCE
    if bool(actor_view_directory) != bool(actor_view_runtime):
        raise ValueError('actor view export requires its directory and explicit runtime package')
    if bool(actor_view_sidecar) != bool(actor_view_sidecar_sha256):
        raise ValueError('actor view sidecar needs its exact SHA256')
    if actor_view_sidecar and (not actor_view_directory or not re.fullmatch(r'[0-9a-f]{64}',actor_view_sidecar_sha256)):
        raise ValueError('actor view sidecar needs its directory/runtime and lowercase SHA256')
    if actor_view_directory:
        helper_loading = ("from natlang_neuralese.model import native_bundle as view_module\n    "
            +"if pathlib.Path(view_module.__file__).resolve()!=pathlib.Path("+repr(str(actor_view_runtime))+",'natlang_neuralese/model/native_bundle.py').resolve():raise ValueError('optional view helper origin differs')\n    ")
        if actor_view_sidecar:
            helper_loading = ("import types,hashlib,natlang_neuralese\n    "
                +"if pathlib.Path(natlang_neuralese.__file__).resolve().parent!=pathlib.Path("+repr(str(actor_view_runtime))+",'natlang_neuralese').resolve():raise ValueError('optional view dependency package origin differs')\n    "
                +"sidecar=pathlib.Path("+repr(str(actor_view_sidecar))+").resolve(strict=True);sidecar_raw=sidecar.read_bytes()\n    "
                +"if hashlib.sha256(sidecar_raw).hexdigest()!="+repr(actor_view_sidecar_sha256)+":raise ValueError('optional view sidecar SHA256 differs')\n    "
                +"view_module=types.ModuleType('natlang_neuralese.model.native_bundle_actor_view_"+actor_view_sidecar_sha256+"');view_module.__package__='natlang_neuralese.model';view_module.__file__=str(sidecar)\n    "
                +"exec(compile(sidecar_raw,str(sidecar),'exec'),view_module.__dict__)\n    "
                +"sidecar_writer=view_module.write_actor_view\n    "
                +"def checked_sidecar_writer(*args,**kwargs):\n     result=sidecar_writer(*args,**kwargs)\n     if hashlib.sha256(sidecar.read_bytes()).hexdigest()!="+repr(actor_view_sidecar_sha256)+":raise ValueError('optional view sidecar changed during extraction')\n     if result is not None and json.loads(pathlib.Path(result['path']).read_text())['mechanism']['sha256']!="+repr(actor_view_sidecar_sha256)+":raise ValueError('optional view receipt changed its executed sidecar identity')\n     return result\n    "
                +"view_module.write_actor_view=checked_sidecar_writer\n    ")
        # Keep the already-validated actual mmap state alive only until the
        # selected immutable full pin has also produced its optional actor view.
        source = source.replace('del value,training;gc.collect();','')
        source = source.replace("  with torch.serialization.safe_globals(",
            "  decoded_stat=temp.stat();decoded_source_stat=(decoded_stat.st_dev,decoded_stat.st_ino,decoded_stat.st_size,decoded_stat.st_mtime_ns)\n  with torch.serialization.safe_globals(")
        marker="'bytes':target.stat().st_size};break"
        assert source.count(marker)==1
        source = source.replace(marker,"'bytes':target.stat().st_size}\n   "
            +"import sys;sys.path.insert(0,"+repr(str(actor_view_runtime))+")\n   "
            +"pin_stat=target.stat();pin_signature=(pin_stat.st_dev,pin_stat.st_ino,pin_stat.st_size,pin_stat.st_mtime_ns)\n   "
            +"try:\n    "+helper_loading
            +"chosen['actor_view']=view_module.write_actor_view(target,"+repr(str(actor_view_directory))
            +",validated_state=value,decoded_source_stat=decoded_source_stat)\n   "
            +"except Exception as error:chosen['actor_view_error']={'type':type(error).__name__,'fallback':'complete-checkpoint','assigned_repair':True}\n   "
            +"pin_stat=target.stat()\n   if (pin_stat.st_dev,pin_stat.st_ino,pin_stat.st_size,pin_stat.st_mtime_ns)!=pin_signature:raise ValueError('validated full pin changed during optional view extraction')\n   "
            +"del value,training;gc.collect();break")
        source = source.replace('  temp.unlink()\n finally:',
                                '  del value,training;gc.collect()\n  temp.unlink()\n finally:')
        source = source.replace('    temp.unlink()\n   else:',
            '    actual_stat=target.stat();decoded_source_stat=(actual_stat.st_dev,actual_stat.st_ino,actual_stat.st_size,actual_stat.st_mtime_ns)\n    temp.unlink()\n   else:')
    summary_prelude=''
    if bool(training_summary_spec) != bool(training_summary_runtime):
        raise ValueError('training summary needs its bound specification and qualified runtime')
    if training_summary_spec:
        try:from .checkpoint_exposure_summary import summarize_cohorts
        except ImportError:from checkpoint_exposure_summary import summarize_cohorts
        summary_prelude=inspect.getsource(summarize_cohorts)+'\n'
        summary_prelude+='SUMMARY_SPEC='+repr(training_summary_spec)+'\nSUMMARY_RUNTIME='+repr(str(training_summary_runtime))+'\n'
        summary_prelude+='''def capture_training_summary(value):
 try:
  import sys
  spec_path=pathlib.Path(SUMMARY_SPEC['path']);raw=spec_path.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=SUMMARY_SPEC['sha256']:raise ValueError('training summary specification changed')
  spec=json.loads(raw);sys.path.insert(0,SUMMARY_RUNTIME)
  os.environ['NATLANG_ROW_INDEX_CACHE']=spec['row_index_cache']
  os.environ['NATLANG_HASH_CACHE']=spec['hash_cache']
  from natlang_neuralese.data.native_sources import NativeGraphSource
  import natlang_neuralese.data.native_sources as sources_module
  if pathlib.Path(sources_module.__file__).resolve()!=pathlib.Path(SUMMARY_RUNTIME,'natlang_neuralese/data/native_sources.py').resolve():raise ValueError('training summary source verifier origin differs')
  from natlang_neuralese.train.native_resident import _check_indexes
  _check_indexes(spec['indexes'])
  def load_source(reference):
   from natlang_neuralese.common.hashing import sha256_file_hex_cached
   path=pathlib.Path(reference['path'])
   if sha256_file_hex_cached(path)!=reference['sha256']:raise ValueError('training summary cohort source changed')
   return NativeGraphSource(path,spec['tokenizer_fingerprint'],required_split='train',role_ids=spec['role_ids'])
  import contextlib
  with contextlib.redirect_stdout(sys.stderr):result=summarize_cohorts(value,spec,load_source)
  result['specification']=SUMMARY_SPEC
  return result
 except Exception as error:return {'error_type':type(error).__name__,'assigned_repair':True,'full_backup_retained':True}
'''
        marker='step=int(value[\'step\']);'
        assert source.count(marker)==1
        source=source.replace(marker,marker+'training_summary=capture_training_summary(value)if step>LAST else None;')
        marker="'bytes':target.stat().st_size}"
        assert source.count(marker)==1
        source=source.replace(marker,"'bytes':target.stat().st_size,'training_summary':training_summary}")
    return ('import pathlib,json,hashlib,re\n'+inspect.getsource(bound_output_checkpoint)+'\n'+summary_prelude
            +'ROLE='+repr(role)+'\nROOT='+repr(str(root))+'\nKNOWN='+repr(known)
            +'\nLAST='+repr(last)+'\nEXTRA='+repr(extra)+'\nPIN_DIRECTORY='+repr(str(pin_directory))
            +'\nMIN_AVAILABLE_BYTES='+repr(minimum_available_bytes)+'\n'+source)


def recoverable_remote_code(code, *, role, root):
    """Durably replay an exact completed RPC observation without another decode."""
    request = hashlib.sha256(code.encode()).hexdigest()
    report = pathlib.Path(root)/f'{role}-cursor-observation-{request}.json'
    prefix = 'import pathlib,json,os,sys\nREPORT='+repr(str(report))+'\nREQUEST='+repr(request)+'\n'
    prefix += """def completed_result():
 if not pathlib.Path(REPORT).exists():return None
 record=json.loads(pathlib.Path(REPORT).read_text())
 if record.get('request_sha256')!=REQUEST:raise ValueError('completed observation request differs')
 result=record['result'];chosen=result.get('chosen')
 if not chosen:return None
 if chosen:
  p=pathlib.Path(chosen['pin']);s=p.stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=record['pin_stat']:raise ValueError('completed observation pin changed')
 return result
recovered=completed_result()
if recovered is not None:print(json.dumps(recovered));raise SystemExit(0)
"""
    marker="print(json.dumps({'at':time.time(),'observations':updates,'chosen':chosen,'cursor_binding_only':c}))"
    if code.count(marker)!=1:raise ValueError('remote observation completion marker differs')
    final="""result={'at':time.time(),'observations':updates,'chosen':chosen,'cursor_binding_only':c}
pin_stat=None
if chosen:
 s=pathlib.Path(chosen['pin']).stat();pin_stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
if chosen:
 record={'request_sha256':REQUEST,'result':result,'pin_stat':pin_stat}
 p=pathlib.Path(REPORT);temporary=p.with_name(p.name+'.tmp-'+str(os.getpid()))
 try:
  with temporary.open('w')as stream:json.dump(record,stream);stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 finally:temporary.unlink(missing_ok=True)
print(json.dumps(result))"""
    lock_marker='fcntl.flock(remote_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);paths='
    if lock_marker in code:
        code=code.replace(lock_marker,'fcntl.flock(remote_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\nrecovered=completed_result()\nif recovered is not None:print(json.dumps(recovered));raise SystemExit(0)\npaths=',1)
    return prefix+code.replace(marker,final)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('lfm', 'mellum'), required=True)
    parser.add_argument('--sandbox-id', required=True)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--state-dir', type=pathlib.Path, required=True)
    parser.add_argument('--export-command', required=True, help='JSON argv with {role} and {pin}')
    parser.add_argument('--bound-output', help='JSON immutable argv binding and allowed_output_root')
    parser.add_argument('--pin-directory', default='/work/production/durable-exports')
    parser.add_argument('--minimum-available-bytes', type=int, default=16*1024**3)
    parser.add_argument('--actor-view-directory', type=pathlib.Path)
    parser.add_argument('--actor-view-runtime', type=pathlib.Path,
                        help='explicit qualified training/neuralese package for optional view extraction')
    parser.add_argument('--actor-view-sidecar', type=pathlib.Path, help='optional immutable native_bundle.py writer using the explicit existing runtime dependencies')
    parser.add_argument('--actor-view-sidecar-sha256', help='exact SHA256 of the optional writer sidecar')
    parser.add_argument('--training-summary-spec', help='JSON immutable binding of declared cohort sources')
    parser.add_argument('--training-summary-runtime', type=pathlib.Path, help='qualified runtime training/neuralese directory')
    parser.add_argument('--interval', type=float, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if bool(args.actor_view_directory) != bool(args.actor_view_runtime):
        parser.error('actor view export needs directory and explicit runtime together')
    if bool(args.actor_view_sidecar) != bool(args.actor_view_sidecar_sha256) or (args.actor_view_sidecar and (not args.actor_view_directory or not re.fullmatch(r'[0-9a-f]{64}',args.actor_view_sidecar_sha256))):
        parser.error('optional actor view sidecar requires directory/runtime and its exact lowercase SHA256')
    if bool(args.training_summary_spec) != bool(args.training_summary_runtime):
        parser.error('training summary needs its specification and qualified runtime together')
    summary=json.loads(args.training_summary_spec) if args.training_summary_spec else None
    command = json.loads(args.export_command)
    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
        parser.error('export command must be an explicit nonempty JSON argv')
    if args.interval <= 0 or args.minimum_available_bytes < 0: parser.error('invalid polling/reserve')
    extra = json.loads(args.bound_output) if args.bound_output else None
    args.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        from . import cwctl
    except ImportError:
        import cwctl
    cwctl.load_env_file(); cwctl.require_key()
    lease = (args.state_dir/f'{args.role}-cursor-periodic-export-owner.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def observe(**kwargs):
        code = remote_code(role=args.role, root=args.root, extra=extra,
                           pin_directory=args.pin_directory, minimum_available_bytes=args.minimum_available_bytes,
                           actor_view_directory=args.actor_view_directory,actor_view_runtime=args.actor_view_runtime,
                           actor_view_sidecar=args.actor_view_sidecar,actor_view_sidecar_sha256=args.actor_view_sidecar_sha256,
                           training_summary_spec=summary,training_summary_runtime=args.training_summary_runtime,**kwargs)
        code = recoverable_remote_code(code, role=args.role, root=args.root)
        result = cwctl.read_sandbox(args.sandbox_id, patience=60).exec(
            ['python', '-c', code], timeout_seconds=3600).result(timeout=3690)
        if result.returncode: raise RuntimeError('owned remote checkpoint observation failed: '+str(result.stderr)[-3000:])
        return json.loads(result.stdout)
    def export(chosen, receipt_path):
        actual = [item.replace('{role}', args.role).replace('{pin}', chosen['pin']) for item in command]
        subprocess.run(actual, check=True, env={**os.environ, 'CW_EXPORT_RECEIPT_FILE': str(receipt_path),
                       'CW_EXPORT_SANDBOX_ID': args.sandbox_id, 'CW_EXPORT_VERIFY_REMOTE': '1'})
        return json.loads(receipt_path.read_text())
    while True:
        try:
            result = run_once(args.state_dir, args.role, observe=observe, export=export)
            if result: print(json.dumps({'at': time.time(), **result}), flush=True)
            if args.once: return 0
            time.sleep(args.interval)
        except Exception as error:
            save(paths_for(args.state_dir,args.role)['error'], {'at':time.time(),'type':type(error).__name__,
                'error':'source export retry; inspect owned unit log','traceback_class_only':True})
            print(json.dumps({'at':time.time(),'retry':type(error).__name__}),flush=True)
            if args.once: raise
            time.sleep(min(60,args.interval))


if __name__ == '__main__':
    raise SystemExit(main())
