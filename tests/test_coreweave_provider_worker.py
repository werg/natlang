import importlib.util
import json
import os
import pytest
from pathlib import Path
import subprocess
import sys


spec = importlib.util.spec_from_file_location('cw_provider_worker', Path(__file__).parents[1] / 'scripts/coreweave/provider_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_detached_runner_lock_and_receipt_prevent_duplicate_execution(tmp_path):
    script = tmp_path / 'runner.py'; script.write_text(worker.RUNNER)
    counter = tmp_path / 'executions.txt'
    command = [sys.executable, '-c', 'import pathlib,time;p=pathlib.Path('+repr(str(counter))+');'
               "p.open('a').write('once\\n');time.sleep(.2)"]
    (tmp_path / 'worker-spec.json').write_text(json.dumps({'request_id': 'exact-request',
        'runtime': str(Path(__file__).parents[1]), 'kind': 'round', 'argv': command, 'env': {'CUDA_VISIBLE_DEVICES': ''}}))
    first = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    second = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first.communicate(timeout=10); second.communicate(timeout=10)
    assert counter.read_text() == 'once\n'
    receipt = json.loads((tmp_path / 'launch.json').read_text())
    assert receipt['wrapper_pid'] in {first.pid, second.pid}
    assert receipt['launch']['token'] and receipt['launch']['identity'] is not None
    subprocess.run([sys.executable, str(script)], check=True)
    assert counter.read_text() == 'once\n'


def test_shared_exact_nonce_and_start_identity_owns_model_signals():
    command = [sys.executable, '-c', 'import time;time.sleep(30)']
    token = 'test-provider-worker-owned-child'
    cwd = str(Path(__file__).parents[1])
    child = subprocess.Popen(command, cwd=cwd,
        env={**os.environ, 'NATLANG_NATIVE_CONTROLLER_LAUNCH': token})
    try:
        start = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        launch = {'status': 'running', 'token': token, 'argv': command, 'cwd': cwd,
            'identity': {'pid': child.pid, 'start_ticks': start, 'boot_id': boot, 'argv': command, 'cwd': cwd}}
        process = {'pid': child.pid, 'launch': launch, 'runtime': cwd,
            'wrapper_pid': 99999999, 'wrapper_start_ticks': 0, 'boot_id': boot}
        io = worker.SDKIO.__new__(worker.SDKIO)
        def remote(sid, code):
            result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
            return json.loads(result.stdout) if result.stdout.strip() else None
        io.remote = remote
        io.read_optional = lambda *args: None
        wrong = {**process, 'launch': {**launch, 'token': 'not-the-owned-launch'}}
        io.terminate('allowed', wrong)
        assert child.poll() is None
        assert io.poll('allowed', wrong, '/unused')['terminal'] is True
        assert io.poll('allowed', process, '/unused') == {'terminal': False}
        mismatched = {**process, 'launch': {**launch, 'identity': {**launch['identity'], 'start_ticks': str(int(start)+1)}}}
        with pytest.raises(subprocess.CalledProcessError):
            io.terminate('allowed', mismatched)
        assert child.poll() is None
        io.terminate('allowed', process)
        assert child.wait(timeout=5) == -15
    finally:
        if child.poll() is None:
            child.kill(); child.wait()


def test_staging_failure_before_ownership_does_not_poison_request_retry():
    class Receipts:
        def __init__(self): self.rows = {}
        def receipt(self, key): return self.rows.get(key)
        def save_receipt(self, key, value): self.rows[key] = value
    io = Receipts()
    worker.record_provider_error(io, 'request-1', TimeoutError())
    assert io.receipt('request-1') is None
    assert io.receipt('request-1-provider-error')['last_provider_error']['type'] == 'TimeoutError'
    owner = {'phase': 'launched', 'target': 'owned', 'request': {'sha256': 'exact'}, 'process': {'pid': 3}}
    io.save_receipt('request-1', owner)
    worker.record_provider_error(io, 'request-1', ConnectionError())
    assert all(io.receipt('request-1')[key] == value for key, value in owner.items())

def test_runner_death_before_provider_launch_receipt_recovers_existing_child(tmp_path):
    script=tmp_path/'runner.py'
    script.write_text(worker.RUNNER.replace("write_json_durable(p/'launch.json'", "os._exit(19)\nwrite_json_durable(p/'launch.json'",1))
    counter=tmp_path/'executions.txt'
    command=[sys.executable,'-c','import pathlib,time;p=pathlib.Path('+repr(str(counter))+');p.open("a").write("once\\n");time.sleep(1)']
    (tmp_path/'worker-spec.json').write_text(json.dumps({'request_id':'interrupted-exact-request',
        'runtime':str(Path(__file__).parents[1]),'kind':'round','argv':command,'env':{'CUDA_VISIBLE_DEVICES':''}}))
    first=subprocess.run([sys.executable,str(script)],capture_output=True,timeout=15)
    assert first.returncode==19 and (tmp_path/'child-launch.json').exists()
    assert not (tmp_path/'launch.json').exists()
    script.write_text(worker.RUNNER)
    subprocess.run([sys.executable,str(script)],check=True,capture_output=True,timeout=15)
    assert counter.read_text()=='once\n'
    assert json.loads((tmp_path/'child-exit.json').read_text())['reason'].startswith('previous owned launch already terminal')

def test_once_drains_initial_request_after_queue_timeout_without_launching_arrivals(tmp_path,monkeypatch):
    config=tmp_path/'provider.json';config.write_text(json.dumps({'coordinator_id':'mellum','sandboxes':{'mellum':{},'lfm':{}},'ferry_command':['existing-ferry'],'queue':'/work/native-production/queue','poll_seconds':0}))
    class IO:
        def __init__(self):self.rows={};self.polls=0
        def receipt(self,key):return self.rows.get(key)
        def save_receipt(self,key,value):self.rows[key]=value
        def pending(self,*args):
            self.polls+=1
            if self.polls==1:raise TimeoutError('transient SDK queue read')
            return [{'request_id':'initial'}]+([{'request_id':'later'}] if self.polls>2 else [])
    io=IO();calls=[]
    class Bridge:
        def __init__(self,*args):pass
        def tick(self,envelope,**kwargs):
            calls.append(envelope['request_id']);return len(calls)>1
    monkeypatch.setattr(worker,'SDKIO',lambda config:io)
    monkeypatch.setattr(worker,'ProviderBridge',Bridge)
    monkeypatch.setattr(worker.cwctl,'load_env_file',lambda:None)
    monkeypatch.setattr(worker.cwctl,'require_key',lambda:None)
    assert worker.main(['--config',str(config),'--once'])==0
    assert calls==['initial','initial'] and io.polls==3
    assert io.receipt('queue-poll-provider-error')['last_provider_error']['type']=='TimeoutError'

def test_provider_never_ferries_over_a_published_runtime_alias():
    io=worker.SDKIO.__new__(worker.SDKIO)
    io.config={'sandboxes':{'target':{'runtime':'/published/runtime'}}}
    io.remote=lambda *args,**kwargs:False
    with pytest.raises(ValueError,match='cannot overwrite published runtime'):
        io.mirror('source','target',{'path':'/work/native-production/runtime/training/neuralese/recipes/live.json','bytes':1,'sha256':'a'*64})
