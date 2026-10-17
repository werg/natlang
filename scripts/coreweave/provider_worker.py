#!/usr/bin/env python3
"""DGX SDK owner for the native file queue; no credentials enter sandboxes.

--config binds existing sandbox IDs, published runtime paths, receipt directory,
and the existing scoped artifact ferry argv. The native controller owns policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwctl
from provider_bridge import ProviderBridge, checked_binding, safe_path

METADATA_ROOT = '/work/native-production/shared/bridge-metadata'


RUNNER = r'''import pathlib,json,fcntl,os,sys,time,signal
p=pathlib.Path(__file__).parent
lock=(p/'worker.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit(0)
spec=json.loads((p/'worker-spec.json').read_text());os.environ.update(spec['env']);sys.path.insert(0,spec['runtime']+'/training/neuralese')
from natlang_neuralese.train.native_readiness import _launch_child,_recover_child,_RecoveredChild
from natlang_neuralese.common.jsonio import write_json_durable
child=None;stop=False

def requested(signum,frame):
 global stop
 stop=True
 if child is not None:child.terminate()
signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
launch_path=p/'child-launch.json'
if launch_path.exists():
 launch=json.loads(launch_path.read_text());child=_recover_child(launch)
else:child,launch=_launch_child(spec['argv'],'native-provider-'+spec['kind'],{'workdir':spec['runtime']},p,spec['request_id'],None)
wrapper_start=int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
write_json_durable(p/'launch.json',{'pid':child.pid if child is not None else None,'launch':launch,'request_id':spec['request_id'],'directory':str(p),'runtime':spec['runtime'],'wrapper_pid':os.getpid(),'wrapper_start_ticks':wrapper_start,'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
if child is None:
 write_json_durable(p/'child-exit.json',{'exit_code':None,'at':time.time(),'reason':'previous owned launch already terminal; no duplicate execution'})
 sys.exit(0)
if stop:child.terminate()
while True:
 code=child.poll()
 if code is not None:
  if not isinstance(child,_RecoveredChild):
   surviving=_recover_child(launch)
   if surviving is not None:child=surviving;continue
  break
 time.sleep(.2)
write_json_durable(launch_path,{**launch,'status':'exited','exit_code':code})
write_json_durable(p/'child-exit.json',{'exit_code':code,'at':time.time()})
'''



class SDKIO:
    def __init__(self, config):
        self.config, self.clients = config, {}
        self.root = Path(config['receipts']); self.root.mkdir(parents=True, exist_ok=True)

    def sandbox(self, sid):
        if sid not in self.config['sandboxes']:
            raise ValueError('sandbox is outside configured existing-resource allowlist')
        if sid not in self.clients:
            self.clients[sid] = cwctl.read_sandbox(sid, patience=60)
        return self.clients[sid]

    def _runtime_environment(self, sid):
        resource = self.config['sandboxes'].get(sid, {})
        env = resource.get('env', {})
        if (not isinstance(env, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                             for key, value in env.items())):
            raise ValueError('provider metadata environment must contain string names and values')
        env = dict(env)
        if resource.get('runtime'):
            env['NATLANG_REPO'] = resource['runtime']
        return env

    def _metadata_environment(self, sid):
        env = self._runtime_environment(sid)
        return 'import os;os.environ.update('+repr(env)+');' if env else ''

    def _record_metadata_timing(self, timing):
        try:
            import fcntl
            with (self.root/'metadata-timings.jsonl').open('a') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write(json.dumps(timing, sort_keys=True)+'\n')
        except Exception as error:
            # Observability must not replace a useful result or its actual error.
            print('provider metadata timing unavailable: '+type(error).__name__, file=sys.stderr)

    def remote(self, sid, code, *, timeout=120):
        started = time.monotonic()
        timing = {'schema': 'natlang.provider-metadata-timing/1', 'at': time.time(),
                  'sandbox_id': sid, 'operation': sys._getframe(1).f_code.co_name,
                  'phase_seconds': {}, 'status': 'failed'}
        try:
            prefix = self._metadata_environment(sid)
            if prefix and not code.startswith(prefix):code = prefix+code
            timing['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
            timing['code_bytes'] = len(code.encode())
            result = self._remote(sid, code, timeout=timeout, timing=timing)
            timing['status'] = 'completed'
            return result
        except BaseException as error:
            timing['error_type'] = type(error).__name__
            raise
        finally:
            timing['total_seconds'] = time.monotonic()-started
            self._record_metadata_timing(timing)

    def _remote(self, sid, code, *, timeout, timing):
        started = time.monotonic()
        try:sandbox = self.sandbox(sid)
        finally:timing['phase_seconds']['sandbox_bind'] = time.monotonic()-started
        temporary = None
        try:
            if len(code.encode()) > 64 * 1024:
                temporary = METADATA_ROOT + '/' + uuid.uuid4().hex + '.py'
                # Large inventory literals otherwise exceed Linux MAX_ARG_STRLEN.
                started = time.monotonic()
                try:
                    self.remote(sid, 'import pathlib;pathlib.Path('+repr(METADATA_ROOT)+').mkdir(parents=True,exist_ok=True)')
                    sandbox.write_file(temporary, code.encode()).result(timeout=120)
                finally:timing['phase_seconds']['script_upload'] = time.monotonic()-started
                digest = hashlib.sha256(code.encode()).hexdigest()
                command = ('import pathlib,hashlib;p=pathlib.Path('+repr(temporary)+');'
                           'raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='+repr(digest)+';'
                           'exec(compile(raw,str(p),"exec"))')
            else:
                command = code
            started = time.monotonic()
            try:result = sandbox.exec(['python', '-c', command], timeout_seconds=timeout).result(timeout=timeout+60)
            finally:timing['phase_seconds']['exec_rpc'] = time.monotonic()-started
            timing['returncode'] = result.returncode
        finally:
            if temporary is not None:
                started = time.monotonic()
                try:
                    self.remote(sid, 'import pathlib;pathlib.Path('+repr(temporary)+').unlink(missing_ok=True)')
                except Exception:
                    timing['cleanup_failed'] = True  # Preserve the original SDK outcome.
                finally:timing['phase_seconds']['cleanup'] = time.monotonic()-started
        if result.returncode:
            evidence = self.root / ('metadata-error-' + uuid.uuid4().hex + '.json')
            evidence.write_text(json.dumps({'sandbox_id': sid,
                'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                'returncode': result.returncode,
                'stdout': result.stdout[-65536:], 'stderr': result.stderr[-65536:]}))
            raise RuntimeError('remote provider metadata operation failed; returncode=' + str(result.returncode)
                               + '; private evidence=' + str(evidence))
        return json.loads(result.stdout) if result.stdout.strip() else None

    def read(self, sid, path):
        safe_path(path)
        size = self.remote(sid, 'import pathlib,json;p=pathlib.Path('+repr(path)+');print(json.dumps(p.stat().st_size))')
        if size > 16*1024**2:
            raise ValueError('provider metadata exceeds16MiB; use artifact transfer for payloads')
        data = self.sandbox(sid).read_file(path).result(timeout=120)
        if not isinstance(data, bytes):
            raise TypeError('SDK metadata read did not return bytes')
        return data

    def read_optional(self, sid, path):
        safe_path(path)
        exists = self.remote(sid, 'import pathlib,json;print(json.dumps(pathlib.Path('+repr(path)+').is_file()))')
        return self.read(sid, path) if exists else None

    def write_atomic(self, sid, path, data):
        safe_path(path)
        temporary = path + '.provider-' + uuid.uuid4().hex
        self.remote(sid, 'import pathlib;pathlib.Path('+repr(path)+').parent.mkdir(parents=True,exist_ok=True)')
        self.sandbox(sid).write_file(temporary, data).result(timeout=120)
        self.remote(sid, 'import os;os.replace('+repr(temporary)+','+repr(path)+')')

    def receipt(self, key):
        path = self.root/(key+'.json')
        return json.loads(path.read_text()) if path.exists() else None

    def save_receipt(self, key, value):
        from natlang_neuralese.common.jsonio import write_json_durable
        write_json_durable(self.root/(key+'.json'), value)

    def dependency_code(self, sid):
        configured = self.config['sandboxes'][sid]
        runtime = configured['runtime']
        source = Path(__file__).with_name('provider_dependencies.py').read_text()
        code = self._metadata_environment(sid)+'import sys,json;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
        if configured.get('cpu_helpers') or configured.get('cpu_helper_loader'):
            if not configured.get('cpu_helpers') or not configured.get('cpu_helper_loader'):
                raise ValueError('provider CPU composition needs both its loader and module manifest')
            loader = checked_binding(configured['cpu_helper_loader'])
            modules = checked_binding(configured['cpu_helpers'])
            code += ('import pathlib,hashlib;'
                     'p=pathlib.Path('+repr(loader['path'])+');'
                     'raw=p.read_bytes();'
                     'assert len(raw)=='+str(loader['bytes'])+' and hashlib.sha256(raw).hexdigest()=='+repr(loader['sha256'])+',"provider CPU loader changed";'
                     'namespace={"__file__":str(p),"__name__":"provider_cpu_loader"};'
                     'exec(compile(raw,str(p),"exec"),namespace);'
                     'namespace["load"]('+repr(modules)+');')
        return code+'exec(compile('+repr(source)+',"provider-dependencies","exec"));'

    def _readonly_input_arguments(self, sid):
        roots = self.config['sandboxes'][sid].get('readonly_input_roots', ())
        return ',readonly_input_roots='+repr(roots) if roots else ''

    def dependency_closure(self, sid, request):
        return self.remote(sid, self.dependency_code(sid)+
                           'print(json.dumps(dependency_closure('+repr(request)+self._readonly_input_arguments(sid)+')))', timeout=900)

    def verify_assets(self, sid, assets):
        code = self.dependency_code(sid)+'verify_assets('+repr(assets)+self._readonly_input_arguments(sid)+');'
        code += 'import subprocess;'
        code += '\nfor asset in '+repr(assets)+':\n'
        code += (' if asset["kind"]=="task-runtime":\n'
                 '  result=subprocess.run(["node",asset["path"]+"/dist/teacher/cli.js","--help"],capture_output=True,timeout=30)\n'
                 '  if result.returncode: raise RuntimeError("provider compiled task CLI/npm import preflight failed")\n')
        self.remote(sid, code, timeout=900)

    def mirror(self, source, target, item):
        item = checked_binding(item)
        if item['bytes']==0:
            if item['sha256']!=hashlib.sha256(b'').hexdigest():
                raise ValueError('empty provider artifact has a nonempty SHA')
            if not self.remote(source,'import pathlib,json;p=pathlib.Path('+repr(item['path'])+');print(json.dumps(p.is_file() and p.stat().st_size==0))'):
                raise ValueError('empty provider source differs from its binding')
            self.write_atomic(target,item['path'],b'')
            return
        runtime = self.config['sandboxes'][target]['runtime']
        code = ('import pathlib,json,sys;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
                'from natlang_neuralese.common.hashing import sha256_file_hex_cached;'
                'p=pathlib.Path('+repr(item['path'])+');'
                'print(json.dumps(p.is_file() and p.stat().st_size=='+str(item['bytes'])+
                ' and sha256_file_hex_cached(p)=='+repr(item['sha256'])+'))')
        if self.remote(target, code, timeout=600):
            return
        if item['path'].startswith('/work/native-production/runtime/'):
            raise ValueError('provider cannot overwrite published runtime through its alias; stage the matching Git runtime first')
        if source == target:
            raise ValueError('coordinator artifact differs from its own request binding')
        job = self.root/('ferry-'+item['sha256']+'-'+target+'.json')
        job.write_text(json.dumps({'source_id': source, 'target_id': target, 'binding': item}))
        # The configured existing ferry uses only scoped PUT/GET capabilities;
        # no large payload passes through this process or its stdout.
        result = subprocess.run([*self.config['ferry_command'], str(job)], capture_output=True)
        if result.returncode:
            raise RuntimeError('private artifact ferry failed; inspect its private receipt')
        if not self.remote(target, code, timeout=600):
            raise ValueError('target artifact failed exact post-transfer verification')

    def mirror_many(self, source, target, items, *, should_stop=lambda: False):
        """Batch small closed artifacts; retain the reusable large-object ferry."""
        from natlang_neuralese.common.hashing import canonical_json_sha256_hex
        unique={}
        for value in items:
            item=checked_binding(value)
            if item['path'] in unique and unique[item['path']]!=item:
                raise ValueError('provider inventory has conflicting path bindings')
            unique[item['path']]=item
        items=list(unique.values())
        code=('import pathlib,json,sys;sys.path.insert(0,'+repr(self.config['sandboxes'][target]['runtime']+'/training/neuralese')+');'
              'from natlang_neuralese.common.hashing import sha256_file_hex_cached;'
              'items='+repr(items)+';print(json.dumps([x for x in items if not '
              '(pathlib.Path(x["path"]).is_file() and pathlib.Path(x["path"]).stat().st_size==x["bytes"] '
              'and sha256_file_hex_cached(x["path"])==x["sha256"])]))')
        missing=self.remote(target,code,timeout=900)
        if source==target and missing:
            raise ValueError('coordinator artifact differs from its own inventory')
        groups=[];group=[];size=0
        for item in missing:
            if item['path'].startswith('/work/native-production/runtime/'):
                raise ValueError('provider cannot overwrite published runtime through its alias')
            if item['bytes']>16*1024**2:
                if should_stop():return False
                self.mirror(source,target,item)
                continue
            if group and size+item['bytes']>64*1024**2:
                groups.append(group);group=[];size=0
            group.append(item);size+=item['bytes']
        if group:groups.append(group)
        for group in groups:
            if should_stop():return False
            if len(group)==1:
                self.mirror(source,target,group[0]);continue
            batch=canonical_json_sha256_hex(sorted(group,key=lambda x:x['path']))
            directory='/work/native-production/shared/bridge-transfer-batches/'+batch
            name='artifacts-'+batch
            archive_source=Path(__file__).with_name('source_archive.py').read_text()
            common=('import pathlib,json,types,hashlib,os,shutil;'
                    'archive=types.ModuleType("provider_sealed_archive");'
                    'exec(compile('+repr(archive_source)+',"source_archive.py","exec"),archive.__dict__);'
                    'directory=pathlib.Path('+repr(directory)+');name='+repr(name)+';')
            prepare=common+"""
items=ITEMS;tree=directory/'source';spool=directory/'spool';tree.mkdir(parents=True,exist_ok=True)
for item in items:
 source=pathlib.Path(item['path']);target=tree/source.relative_to('/work/native-production');target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  try:os.link(source,target)
  except OSError:shutil.copyfile(source,target)
 if target.stat().st_size!=item['bytes'] or archive.digest_file(target)!=item['sha256']:raise ValueError('provider batch source changed')
receipt=spool/(name+'.receipt.json')
if receipt.exists():
 value=json.loads(receipt.read_text());archive.verify(spool,name)
else:
 value=archive.pack(tree,spool,name,reserve_bytes=0);archive.verify(spool,name)
expected={str(pathlib.Path(x['path']).relative_to('/work/native-production')):(x['sha256'],x['bytes']) for x in items}
if {x['path']:(x['sha256'],x['bytes']) for x in value['manifest']['files']}!=expected:raise ValueError('provider batch member set changed')
print(json.dumps(value))
""".replace('ITEMS',repr(group))
            receipt=self.remote(source,prepare,timeout=900)
            binding={'path':receipt['archive'],'sha256':receipt['archive_sha256'],'bytes':receipt['archive_bytes']}
            self.mirror(source,target,binding)
            restore=common+'directory.mkdir(parents=True,exist_ok=True);archive._write_json(directory/"spool"/(name+".receipt.json"),'+repr(receipt)+');print(json.dumps(archive.restore(directory/"spool",name,"/work/native-production")))'
            result=self.remote(target,restore,timeout=900)
            if result['restored_files']!=len(group):
                raise ValueError('provider batch target member count differs')
        if self.remote(target,code,timeout=900):
            raise ValueError('provider batch target failed exact final verification')
        return True

    def launch(self, sid, request, request_path, result_path):
        resource = self.config['sandboxes'][sid]
        directory = str(Path(result_path).parent)
        safe_path(directory)
        self.remote(sid, 'import pathlib;pathlib.Path('+repr(directory)+').mkdir(parents=True,exist_ok=True)')
        spec = {'request_id': request['request_id'], 'kind': request['kind'], 'runtime': resource['runtime'],
                'argv': [resource.get('python', 'python'), '-u', '-m', 'natlang_neuralese.rl.native_bridge',
                         '--request', request_path, '--result', result_path],
                'env': {**self._runtime_environment(sid), 'PYTHONSAFEPATH': '1',
                        'PYTHONPATH': resource['runtime']+'/training/neuralese'}}
        if resource.get('worker_entry'):
            entry = checked_binding(resource['worker_entry'])
            code = ('import pathlib,hashlib,json;p=pathlib.Path('+repr(entry['path'])+');'
                    'print(json.dumps(p.is_file() and p.stat().st_size=='+str(entry['bytes'])+
                    ' and hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(entry['sha256'])+'))')
            if not self.remote(sid, code):
                raise ValueError('provider bound worker control entry changed')
            spec['argv'] = [resource.get('python', 'python'), '-u', entry['path'],
                            '--request', request_path, '--result', result_path]
            spec['env']['NATLANG_WORKER_ENTRY_SHA256'] = entry['sha256']
        old = self.read_optional(sid, directory+'/launch.json')
        if old:
            value = json.loads(old)
            if value['request_id'] != request['request_id']:
                raise ValueError('existing worker process belongs to another request')
            return value
        self.sandbox(sid).write_file(directory+'/runner.py', RUNNER.encode()).result(timeout=120)
        self.sandbox(sid).write_file(directory+'/worker-spec.json', json.dumps(spec).encode()).result(timeout=120)
        code = ('import subprocess,pathlib; p=pathlib.Path('+repr(directory)+');'
                "log=(p/'worker.log').open('ab');subprocess.Popen(['python',str(p/'runner.py')],"
                'stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)')
        self.remote(sid, code)
        for _ in range(30):
            data = self.read_optional(sid, directory+'/launch.json')
            if data:
                return json.loads(data)
            time.sleep(.2)
        raise RuntimeError('detached worker did not publish its owned PID receipt')

    def ownership_code(self, process):
        runtime = process['runtime']
        return ('import pathlib,json,sys;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
                'from natlang_neuralese.train.native_readiness import _recover_child;'
                'process='+repr(process)+';child=_recover_child(process[\'launch\'])')

    def poll(self, sid, process, result_path):
        code = self.ownership_code(process) + ';\nif child is not None: child.poll()\n'
        code += ("wrapper=pathlib.Path('/proc/'+str(process['wrapper_pid'])+'/stat');"
                 "wrapper_alive=(wrapper.exists() and wrapper.read_text().rsplit(')',1)[1].split()[0]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id'] and int(wrapper.read_text().rsplit(')',1)[1].split()[19])==process['wrapper_start_ticks']);"
                 "print(json.dumps({'alive':child is not None and child.poll() is None,'wrapper_alive':wrapper_alive}))")
        state = self.remote(sid, code)
        if state['alive'] or state['wrapper_alive']:
            return {'terminal': False}
        return {'terminal': True, 'result': self.read_optional(sid, result_path)}

    def closed_artifacts(self, sid, out):
        safe_path(out)
        runtime = self.config['sandboxes'][sid]['runtime']
        return self.remote(sid, 'import sys,json;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
                           'from natlang_neuralese.rl.native_bridge import inventory;'
                           'print(json.dumps(inventory('+repr(out)+'))) ', timeout=900)

    def terminate(self, sid, process):
        code = self.ownership_code(process) + ';\nif child is not None:child.terminate()'
        self.remote(sid, code)

    def pending(self, coordinator, queue):
        safe_path(queue)
        runtime = self.config['sandboxes'][coordinator]['runtime']
        code = ('import sys,json;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
                'from natlang_neuralese.rl.native_bridge import queued_requests;'
                'print(json.dumps(queued_requests('+repr(queue)+')))')
        return self.remote(coordinator, code)


def record_provider_error(io, request_id, error):
    """An SDK failure before validation must not manufacture a request owner."""
    current = io.receipt(request_id)
    failure = {'type': type(error).__name__, 'at': time.time()}
    if current:
        io.save_receipt(request_id, {**current, 'last_provider_error': failure})
    else:
        io.save_receipt(request_id+'-provider-error', {'last_provider_error': failure})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    coordinator = config['coordinator_id']
    if coordinator not in config['sandboxes'] or not config.get('ferry_command'):
        raise ValueError('provider configuration lacks exact coordinator/resources/ferry')
    cwctl.load_env_file(); cwctl.require_key()
    io = SDKIO(config); bridge = ProviderBridge(io, coordinator, config['sandboxes'])
    stop = False
    def requested(signum, frame):
        nonlocal stop
        stop = True
        # Forward promptly even while another request is transferring a large
        # artifact. Never exit this host bridge before owned workers drain.
        for path in io.root.glob('*.json'):
            try:
                receipt = json.loads(path.read_text())
                if receipt.get('phase') == 'launched' and receipt.get('process'):
                    io.terminate(receipt['target'], receipt['process'])
                    receipt['phase'] = 'stop_sent'; io.save_receipt(path.stem, receipt)
            except Exception as error:
                print(json.dumps({'event': 'stop_forward_retry_required', 'error_type': type(error).__name__}), flush=True)
    signal.signal(signal.SIGTERM, requested); signal.signal(signal.SIGINT, requested)
    once_ids = None
    while True:
        try:
            pending = io.pending(coordinator, config['queue'])
        except Exception as error:
            record_provider_error(io, "queue-poll", error)
            print(json.dumps({"event": "provider_queue_poll_retry", "error_type": type(error).__name__}), flush=True)
            time.sleep(config.get("poll_seconds", 10))
            continue
        if args.once:
            if once_ids is None:
                once_ids = {envelope["request_id"] for envelope in pending}
            pending = [envelope for envelope in pending if envelope["request_id"] in once_ids]
        if stop and not pending:
            return 0
        active = False
        for envelope in pending:
            receipt = io.receipt(envelope['request_id'])
            # A host stop drains owned workers; unstarted requests remain queued.
            if stop and not receipt:
                continue
            try:
                complete = bridge.tick(envelope, host_stop=lambda: stop)
            except Exception as error:
                # SDK timeout may occur after detached launch. Keep the exact
                # staging/ownership receipt and reattach on the next tick.
                record_provider_error(io, envelope['request_id'], error)
                print(json.dumps({'event': 'provider_attempt_requires_retry',
                    'request_id': envelope['request_id'], 'error_type': type(error).__name__}), flush=True)
                owned = io.receipt(envelope["request_id"]) or {}
                active |= bool(owned.get("process")) or not stop
                continue
            active |= not complete
        if (args.once and not active) or (stop and not active):
            return 0
        time.sleep(config.get('poll_seconds', 10))


if __name__ == '__main__':
    raise SystemExit(main())
