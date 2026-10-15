#!/usr/bin/env python3
"""DGX SDK owner for the native file queue; no credentials enter sandboxes.

--config binds existing sandbox IDs, published runtime paths, receipt directory,
and the existing scoped artifact ferry argv. The native controller owns policy.
"""
from __future__ import annotations
import argparse
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

    def remote(self, sid, code, *, timeout=120):
        result = self.sandbox(sid).exec(['python', '-c', code], timeout_seconds=timeout).result(timeout=timeout+60)
        if result.returncode:
            raise RuntimeError('remote provider metadata operation failed; returncode=' + str(result.returncode))
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
        runtime = self.config['sandboxes'][sid]['runtime']
        source = Path(__file__).with_name('provider_dependencies.py').read_text()
        return ('import sys,json;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
                'exec(compile('+repr(source)+',"provider-dependencies","exec"));')

    def dependency_closure(self, sid, request):
        return self.remote(sid, self.dependency_code(sid)+
                           'print(json.dumps(dependency_closure('+repr(request)+')))', timeout=900)

    def verify_assets(self, sid, assets):
        self.remote(sid, self.dependency_code(sid)+'verify_assets('+repr(assets)+')', timeout=900)

    def mirror(self, source, target, item):
        item = checked_binding(item)
        runtime = self.config['sandboxes'][target]['runtime']
        code = ('import pathlib,json,sys;sys.path.insert(0,'+repr(runtime+'/training/neuralese')+');'
                'from natlang_neuralese.common.hashing import sha256_file_hex_cached;'
                'p=pathlib.Path('+repr(item['path'])+');'
                'print(json.dumps(p.is_file() and p.stat().st_size=='+str(item['bytes'])+
                ' and sha256_file_hex_cached(p)=='+repr(item['sha256'])+'))')
        if self.remote(target, code, timeout=600):
            return
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

    def launch(self, sid, request, request_path, result_path):
        resource = self.config['sandboxes'][sid]
        directory = str(Path(result_path).parent)
        safe_path(directory)
        self.remote(sid, 'import pathlib;pathlib.Path('+repr(directory)+').mkdir(parents=True,exist_ok=True)')
        spec = {'request_id': request['request_id'], 'kind': request['kind'], 'runtime': resource['runtime'],
                'argv': [resource.get('python', 'python'), '-u', '-m', 'natlang_neuralese.rl.native_bridge',
                         '--request', request_path, '--result', result_path],
                'env': {**resource.get('env', {}), 'PYTHONSAFEPATH': '1',
                        'PYTHONPATH': resource['runtime']+'/training/neuralese'}}
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
