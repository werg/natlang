import importlib.util
import json
import os
import pytest
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import contextlib
import io
import shutil


spec = importlib.util.spec_from_file_location('cw_provider_worker', Path(__file__).parents[1] / 'scripts/coreweave/provider_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_empty_closed_artifact_is_exact_without_zero_byte_store_upload(tmp_path):
    class Empty(worker.SDKIO):
        def remote(self,source,code,**kwargs):return True
        def write_atomic(self,target,path,data):self.written=(target,path,data)
    client=Empty({'receipts':str(tmp_path),'sandboxes':{}})
    binding={'path':'/work/native-production/closed.lock','bytes':0,'sha256':worker.hashlib.sha256(b'').hexdigest()}
    client.mirror('source','target',binding)
    assert client.written==('target',binding['path'],b'')
    with pytest.raises(ValueError,match='nonempty SHA'):
        client.mirror('source','target',{**binding,'sha256':'f'*64})


def metadata_client(tmp_path):
    """Execute real remote code and count both exec and file RPCs."""
    calls=[]
    class Sandbox:
        def exec(self,argv,**kwargs):
            calls.append(argv)
            rewritten=[sys.executable,*argv[1:-1],argv[-1].replace('/work/native-production',str(tmp_path/'guest'))]
            return SimpleNamespace(result=lambda **kw:subprocess.run(rewritten,capture_output=True,text=True))
        def read_file(self,path):
            calls.append(('read_file',path))
            physical=Path(path.replace('/work/native-production',str(tmp_path/'guest')))
            return SimpleNamespace(result=lambda **kwargs:physical.read_bytes())
    client=worker.SDKIO({'receipts':str(tmp_path/'receipts'),'sandboxes':{}})
    client.sandbox=lambda sid:Sandbox()
    (tmp_path/'guest').mkdir()
    return client,calls


@pytest.mark.parametrize('optional',[False,True])
@pytest.mark.parametrize('payload',[b'',b'\x00\xff\xfe\r\nexact JSON bytes\n',b'{"stop":true}\n'])
def test_metadata_reads_return_exact_bytes_in_one_rpc(tmp_path,optional,payload):
    client,calls=metadata_client(tmp_path);(tmp_path/'guest/value').write_bytes(payload)
    read=client.read_optional if optional else client.read
    assert read('guest','/work/native-production/value')==payload
    assert len(calls)==1
    timing=json.loads((client.root/'metadata-timings.jsonl').read_text())
    assert timing['status']=='completed'and timing['operation']=='_read_metadata'


def test_optional_metadata_missing_directory_and_empty_are_distinct(tmp_path):
    client,calls=metadata_client(tmp_path)
    assert client.read_optional('guest','/work/native-production/missing')is None
    (tmp_path/'guest/directory').mkdir()
    assert client.read_optional('guest','/work/native-production/directory')is None
    (tmp_path/'guest/empty').touch()
    assert client.read_optional('guest','/work/native-production/empty')==b''
    assert len(calls)==3
    with pytest.raises(RuntimeError,match='private evidence'):
        client.read('guest','/work/native-production/missing')
    assert len(calls)==4 and len(list(client.root.glob('metadata-error-*.json')))==1
    with pytest.raises(ValueError,match='namespace'):
        client.read_optional('guest','/work/native-production/../escape')
    assert len(calls)==4


@pytest.mark.parametrize('extra',[0,1])
def test_metadata_existing_size_boundary_is_preserved(tmp_path,extra):
    client,calls=metadata_client(tmp_path);path=tmp_path/'guest/boundary'
    with path.open('wb')as stream:stream.truncate(worker.MAX_METADATA_BYTES+extra)
    if extra:
        with pytest.raises(ValueError,match='exceeds16MiB'):client.read('guest','/work/native-production/boundary')
    else:assert client.read('guest','/work/native-production/boundary')==bytes(worker.MAX_METADATA_BYTES)
    assert len(calls)==(1 if extra else 2)


@pytest.mark.parametrize('extra',[0,1])
def test_metadata_inline_boundary_preserves_safe_file_transport(tmp_path,extra):
    client,calls=metadata_client(tmp_path);payload=b'x'*(worker.INLINE_METADATA_BYTES+extra)
    (tmp_path/'guest/value').write_bytes(payload)
    assert client.read_optional('guest','/work/native-production/value')==payload
    assert len(calls)==1+extra
    if extra:assert calls[-1]==('read_file','/work/native-production/value')


@pytest.mark.parametrize('changed',[b'truncated',bytearray(b'wrong SDK type')])
def test_metadata_file_fallback_rejects_changed_length_or_type(tmp_path,changed):
    client=worker.SDKIO({'receipts':str(tmp_path),'sandboxes':{}})
    client.remote=lambda *args,**kwargs:{'sdk_file_bytes':worker.INLINE_METADATA_BYTES+1}
    client.sandbox=lambda sid:SimpleNamespace(read_file=lambda path:SimpleNamespace(result=lambda **kwargs:changed))
    with pytest.raises((ValueError,TypeError)):client.read('guest','/work/native-production/value')


@pytest.mark.parametrize('value',[None,{}, {'bytes':True,'base64':''}, {'bytes':-1,'base64':''},
    {'bytes':0,'base64':'AAAA'}, {'bytes':1,'base64':'AA==' ,'extra':1},
    {'bytes':1,'base64':'!@#$'}, {'bytes':3,'base64':'AA=='},
    {'bytes':worker.MAX_METADATA_BYTES+1,'base64':''}, {'sdk_file_bytes':True},
    {'sdk_file_bytes':worker.MAX_METADATA_BYTES+1}, {'sdk_file_bytes':worker.INLINE_METADATA_BYTES}])
def test_metadata_malformed_or_truncated_envelope_rejects(tmp_path,value):
    client=worker.SDKIO({'receipts':str(tmp_path),'sandboxes':{}})
    client.remote=lambda *args,**kwargs:value
    with pytest.raises(ValueError):client.read('guest','/work/native-production/value')


def test_metadata_inventory_uses_one_exact_archive_and_preserves_original_namespace(tmp_path):
    class Local(worker.SDKIO):
        def __init__(self):
            super().__init__({'receipts':str(tmp_path/'receipts'),'sandboxes':{
                name:{'runtime':str(Path(__file__).parents[1])} for name in ('source','target')}})
            self.ferries=[]
        def remote(self,sid,code,**kwargs):
            physical=str(tmp_path/sid)
            capture=io.StringIO()
            code=code.replace('/work/native-production',physical)
            if sid=='target':
                # Both real sandboxes share a lexical namespace. Preserve the
                # source-origin field inside the sealed archive receipt while
                # virtualizing only the target's filesystem accesses.
                code=code.replace("'source': '"+physical,"'source': '"+str(tmp_path/'source'))
            with contextlib.redirect_stdout(capture):
                exec(code,{'__name__':'local_sdk_control'})
            return json.loads(capture.getvalue().replace(physical,'/work/native-production'))
        def mirror(self,source,target,item):
            relative=Path(item['path']).relative_to('/work/native-production')
            original=tmp_path/source/relative;destination=tmp_path/target/relative
            assert original.stat().st_size==item['bytes']
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,destination)
            self.ferries.append(item)
    client=Local();items=[]
    for name,payload in [('state.json',b'{"actual":"RNG"}\r\n'),('nested/optimizer.pt',b'optimizer\x00')]:
        path=tmp_path/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        items.append({'path':'/work/native-production/'+name,'bytes':len(payload),
            'sha256':worker.hashlib.sha256(payload).hexdigest()})
    assert client.mirror_many('source','target',items)
    assert len(client.ferries)==1 and client.ferries[0]['path'].endswith('.tar.zst')
    for item in items:
        relative=Path(item['path']).relative_to('/work/native-production')
        assert (tmp_path/'target'/relative).read_bytes()==(tmp_path/'source'/relative).read_bytes()
    # A retry verifies the inventory in one probe and does not ferry again.
    assert client.mirror_many('source','target',items) and len(client.ferries)==1
    with pytest.raises(ValueError,match='conflicting'):
        client.mirror_many('source','target',[items[0],{**items[0],'bytes':999}])


def test_large_metadata_uses_hash_checked_file_and_removes_it(tmp_path, monkeypatch):
    directory = tmp_path / 'remote-metadata'
    monkeypatch.setattr(worker, 'METADATA_ROOT', str(directory))
    calls = []
    class Sandbox:
        def write_file(self, path, data):
            Path(path).write_bytes(data)
            return SimpleNamespace(result=lambda **kwargs: None)
        def exec(self, argv, **kwargs):
            calls.append(argv)
            assert len(argv[-1].encode()) < 65536
            return SimpleNamespace(result=lambda **kw: subprocess.run(
                [sys.executable, *argv[1:]], capture_output=True, text=True))
    io = worker.SDKIO({'receipts': str(tmp_path / 'receipts'), 'sandboxes': {}})
    io.sandbox = lambda sid: Sandbox()
    code = '# actual large inventory padding\n' * 5000 + '\nprint('+repr('{"verified":true}')+')'
    assert io.remote('fixture', code) == {'verified': True}
    assert len(calls) == 3 and not list(directory.glob('*.py'))


def test_large_metadata_corruption_fails_before_execution_and_keeps_private_evidence(tmp_path, monkeypatch):
    directory = tmp_path / 'remote-metadata'
    monkeypatch.setattr(worker, 'METADATA_ROOT', str(directory))
    executed = tmp_path / 'should-not-exist'
    class Sandbox:
        def write_file(self, path, data):
            Path(path).write_bytes(data + b'\n# truncated or changed transfer\n')
            return SimpleNamespace(result=lambda **kwargs: None)
        def exec(self, argv, **kwargs):
            return SimpleNamespace(result=lambda **kw: subprocess.run(
                [sys.executable, *argv[1:]], capture_output=True, text=True))
    io = worker.SDKIO({'receipts': str(tmp_path / 'receipts'), 'sandboxes': {}})
    io.sandbox = lambda sid: Sandbox()
    code = '# inventory\n' * 7000 + '\nfrom pathlib import Path;Path('+repr(str(executed))+').touch()'
    with pytest.raises(RuntimeError, match='private evidence='):
        io.remote('fixture', code)
    assert not executed.exists() and not list(directory.glob('*.py'))
    evidence = json.loads(next(io.root.glob('metadata-error-*.json')).read_text())
    assert 'AssertionError' in evidence['stderr']


def test_detached_runner_lock_and_receipt_prevent_duplicate_execution(tmp_path,monkeypatch):
    script = tmp_path / 'runner.py'; script.write_text(worker.RUNNER)
    counter = tmp_path / 'executions.txt'
    repo=str(Path(__file__).parents[1])
    monkeypatch.setenv('NATLANG_REPO','/retired/host-only-checkout')
    client=worker.SDKIO({'receipts':str(tmp_path/'receipts'),'sandboxes':{
        'worker':{'runtime':repo,'env':{'NATLANG_REPO':'/another/retired-checkout'}}}})
    command = [sys.executable, '-c', 'import pathlib,time;from natlang_neuralese.common.paths import root;'
               'from scripts.coreweave.native_controller_continuation import runtime_child_environment;'
               'assert callable(runtime_child_environment);'
               'assert root("repo")==pathlib.Path('+repr(repo)+');'
               'assert (root("repo")/"scripts/run_bonsai_queue.py").is_file();'
               'p=pathlib.Path('+repr(str(counter))+');'
               "p.open('a').write('once\\n');time.sleep(.2)"]
    (tmp_path / 'worker-spec.json').write_text(json.dumps({'request_id': 'exact-request',
        'runtime': repo, 'kind': 'round', 'argv': command,
        'env': {**client._runtime_environment('worker'),'CUDA_VISIBLE_DEVICES': ''}}))
    first = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    second = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first_output=first.communicate(timeout=10);second_output=second.communicate(timeout=10)
    assert first.returncode==second.returncode==0,(first_output,second_output)
    assert json.loads((tmp_path/'child-exit.json').read_text())['exit_code']==0
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


def test_dependency_cpu_composition_loads_before_resolver_and_checks_bytes(tmp_path, monkeypatch):
    import builtins
    root='/work/native-production/cpu-metadata-test'
    manifest=tmp_path/'modules.json';manifest.write_text('{}')
    loader=tmp_path/'loader.py'
    loader.write_text('import pathlib,hashlib,builtins\ndef load(reference):\n'
        ' raw=pathlib.Path(reference["path"]).read_bytes()\n'
        ' assert len(raw)==reference["bytes"] and hashlib.sha256(raw).hexdigest()==reference["sha256"]\n'
        ' builtins._provider_cpu_test_loaded=True\n')
    def binding(path):
        return {'path':root+'/'+path.name,'bytes':path.stat().st_size,
                'sha256':worker.hashlib.sha256(path.read_bytes()).hexdigest()}
    config={'runtime':'/unchanged/numerical-runtime','cpu_helper_loader':binding(loader),
            'cpu_helpers':binding(manifest)}
    client=worker.SDKIO({'receipts':str(tmp_path/'receipts'),'sandboxes':{'worker':config}})
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda path,*a,**kw:
        'assert __import__("builtins")._provider_cpu_test_loaded' if path.name=='provider_dependencies.py'
        else original(path,*a,**kw))
    monkeypatch.setattr(builtins,'_provider_cpu_test_loaded',False,raising=False)
    code=client.dependency_code('worker').replace(root,str(tmp_path))
    exec(code,{})
    assert builtins._provider_cpu_test_loaded is True
    builtins._provider_cpu_test_loaded=False
    loader.write_text(loader.read_text()+'# changed after binding\n')
    with pytest.raises(AssertionError,match='CPU loader changed'):
        exec(code,{})
    assert builtins._provider_cpu_test_loaded is False

    loader.write_text('import builtins\ndef load(reference):\n builtins._provider_cpu_test_loaded=True\n')
    config['cpu_helper_loader']=binding(loader)
    code=client.dependency_code('worker').replace(root,str(tmp_path))
    read_bytes=Path.read_bytes
    def replaced_after_read(path):
        raw=read_bytes(path)
        if path==loader:
            path.write_text('raise RuntimeError("unverified replacement executed")\n')
        return raw
    monkeypatch.setattr(Path,'read_bytes',replaced_after_read)
    exec(code,{})
    assert builtins._provider_cpu_test_loaded is True


def test_dependency_cpu_composition_rejects_incomplete_or_unscoped_bindings(tmp_path):
    config={'runtime':'/unchanged/numerical-runtime','cpu_helpers':{'path':'unused'}}
    client=worker.SDKIO({'receipts':str(tmp_path),'sandboxes':{'worker':config}})
    with pytest.raises(ValueError,match='both its loader'):
        client.dependency_code('worker')
    config['cpu_helper_loader']={'path':'/tmp/unscoped.py','bytes':1,'sha256':'a'*64}
    with pytest.raises(ValueError,match='namespace'):
        client.dependency_code('worker')


def test_dependency_code_imports_selected_package_and_repository_helpers_in_isolated_python(tmp_path,monkeypatch):
    runtime=tmp_path/'runtime'
    package=runtime/'training/neuralese/natlang_neuralese';package.mkdir(parents=True)
    (package/'__init__.py').write_text('SELECTED="actual-package"\n')
    helper=runtime/'scripts/neuralese_data';helper.mkdir(parents=True)
    (helper/'probe.py').write_text('SELECTED="actual-repository-helper"\n')
    client=worker.SDKIO({'receipts':str(tmp_path/'receipts'),'sandboxes':{
        'worker':{'runtime':str(runtime),'env':{'PYTHONPATH':'/retired/checkout'}}}})
    original=Path.read_text
    body='from scripts.neuralese_data.probe import SELECTED;assert SELECTED=="actual-repository-helper";import natlang_neuralese;assert natlang_neuralese.SELECTED=="actual-package"'
    monkeypatch.setattr(Path,'read_text',lambda path,*a,**kw:body if path.name=='provider_dependencies.py'else original(path,*a,**kw))
    result=subprocess.run([sys.executable,'-I','-c',client.dependency_code('worker')],
        cwd=tmp_path,capture_output=True,text=True,env={**os.environ,'PYTHONSAFEPATH':'1'})
    assert result.returncode==0,result.stderr


def test_metadata_environment_precedes_dependency_import_and_cpu_loader(tmp_path,monkeypatch):
    root='/work/native-production/metadata-env-test'
    loader=tmp_path/'loader.py';loader.write_text('import os\nassert os.environ["PROVIDER_CACHE_TEST"]=="declared-cache"\ndef load(reference):\n assert os.environ["PROVIDER_CACHE_TEST"]=="declared-cache"\n')
    modules=tmp_path/'modules.json';modules.write_text('{}')
    def bound(path):return {'path':root+'/'+path.name,'bytes':path.stat().st_size,'sha256':worker.hashlib.sha256(path.read_bytes()).hexdigest()}
    config={'runtime':'/same/runtime','env':{'PROVIDER_CACHE_TEST':'declared-cache'},
        'cpu_helper_loader':bound(loader),'cpu_helpers':bound(modules)}
    client=worker.SDKIO({'receipts':str(tmp_path/'receipts'),'sandboxes':{'worker':config}})
    read_text=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda path,*a,**kw:
        'import os;assert os.environ["PROVIDER_CACHE_TEST"]=="declared-cache"' if path.name=='provider_dependencies.py'
        else read_text(path,*a,**kw))
    monkeypatch.delenv('PROVIDER_CACHE_TEST',raising=False)
    code=client.dependency_code('worker').replace(root,str(tmp_path))
    try:exec(code,{})
    finally:os.environ.pop('PROVIDER_CACHE_TEST',None)


@pytest.mark.parametrize('large',[False,True])
def test_remote_uses_declared_env_before_import_and_records_only_phase_metadata(tmp_path,monkeypatch,large):
    monkeypatch.setattr(worker,'METADATA_ROOT',str(tmp_path/'remote'))
    class Sandbox:
        def write_file(self,path,raw):
            Path(path).write_bytes(raw);return SimpleNamespace(result=lambda **kw:None)
        def exec(self,argv,**kwargs):
            return SimpleNamespace(result=lambda **kw:subprocess.run([sys.executable,*argv[1:]],capture_output=True,text=True))
    secret='private-environment-value-not-for-timing-logs'
    client=worker.SDKIO({'receipts':str(tmp_path/'receipts'),'sandboxes':{
        'worker':{'env':{'PROVIDER_CACHE_TEST':secret},'runtime':'unused'}}})
    client.sandbox=lambda sid:Sandbox()
    module=tmp_path/'early_import.py';module.write_text('import os\nassert os.environ["PROVIDER_CACHE_TEST"]=='+repr(secret)+'\n')
    code=('import sys;sys.path.insert(0,'+repr(str(tmp_path))+');import early_import;print("{\\\"passed\\\":true}")')
    if large:code='# padding inventory\n'*6000+code
    assert client.remote('worker',code)=={'passed':True}
    records=[json.loads(line) for line in (client.root/'metadata-timings.jsonl').read_text().splitlines()]
    record=next(row for row in records if row['code_bytes']>65536) if large else records[0]
    assert record['status']=='completed' and record['returncode']==0
    assert record['phase_seconds']['sandbox_bind']>=0 and record['phase_seconds']['exec_rpc']>=0
    if large:assert record['phase_seconds']['script_upload']>=0 and record['phase_seconds']['cleanup']>=0
    assert record['total_seconds']>=sum(record['phase_seconds'].values())
    assert secret not in (client.root/'metadata-timings.jsonl').read_text()
    assert 'PROVIDER_CACHE_TEST' not in os.environ
    assert not list((tmp_path/'remote').glob('*.py'))


def test_metadata_timing_failure_preserves_original_remote_error(tmp_path,monkeypatch,capsys):
    client=worker.SDKIO({'receipts':str(tmp_path),'sandboxes':{}})
    client.sandbox=lambda sid:(_ for _ in ()).throw(RuntimeError('original SDK failure'))
    original=Path.open
    def denied(path,*a,**kw):
        if path.name=='metadata-timings.jsonl':raise OSError('timing disk unavailable')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',denied)
    with pytest.raises(RuntimeError,match='original SDK failure'):client.remote('fixture','print("{}")')
    assert 'timing unavailable: OSError' in capsys.readouterr().err


def test_metadata_environment_rejects_nonstring_configuration(tmp_path):
    client=worker.SDKIO({'receipts':str(tmp_path),'sandboxes':{'worker':{'runtime':'unused','env':{'NATLANG_HASH_CACHE':False}}}})
    with pytest.raises(ValueError,match='string names and values'):client.dependency_code('worker')



def test_configured_readonly_roots_reach_both_remote_asset_operations(tmp_path):
    class Capture(worker.SDKIO):
        def dependency_code(self,sid):return ''
        def remote(self,sid,code,**kwargs):self.code=code;return {}
    client=Capture({'receipts':str(tmp_path),'sandboxes':{'worker':{'readonly_input_roots':['/exact/d24']}}})
    client.dependency_closure('worker',{'kind':'round'})
    assert "readonly_input_roots=['/exact/d24']" in client.code
    client.verify_assets('worker',[])
    assert "verify_assets([],readonly_input_roots=['/exact/d24'])" in client.code
    client.config['sandboxes']['worker']={}
    client.dependency_closure('worker',{})
    assert 'readonly_input_roots' not in client.code



@pytest.mark.parametrize('changed',[False,True])
def test_bound_worker_control_entry_is_verified_before_detached_launch(tmp_path,changed):
    entry={'path':'/work/native-production/control/shared-entry.py','bytes':123,'sha256':'e'*64}
    class Sandbox:
        def __init__(self):self.files={}
        def write_file(self,path,raw):self.files[path]=raw;return SimpleNamespace(result=lambda **kw:None)
    class Client(worker.SDKIO):
        def remote(self,sid,code,**kw):
            if 'hashlib.sha256(p.read_bytes())'in code:return not changed
            return None
        def read_optional(self,sid,path):
            self.reads=getattr(self,'reads',0)+1
            return json.dumps({'request_id':'bound-request'})if self.reads>1 else None
    client=Client({'receipts':str(tmp_path),'sandboxes':{'worker':{'runtime':'/exact/numeric','worker_entry':entry}}})
    sandbox=Sandbox();client.sandbox=lambda sid:sandbox
    if changed:
        with pytest.raises(ValueError,match='control entry changed'):
            client.launch('worker',{'request_id':'bound-request','kind':'round'},'/work/native-production/request.json','/work/native-production/worker/result.json')
        assert sandbox.files=={}
    else:
        assert client.launch('worker',{'request_id':'bound-request','kind':'round'},'/work/native-production/request.json','/work/native-production/worker/result.json')=={'request_id':'bound-request'}
        saved=json.loads(sandbox.files['/work/native-production/worker/worker-spec.json'])
        assert saved['argv']==['python','-u',entry['path'],'--request','/work/native-production/request.json','--result','/work/native-production/worker/result.json']
        assert saved['env']['NATLANG_WORKER_ENTRY_SHA256']==entry['sha256']
        assert saved['runtime']=='/exact/numeric'
        assert sandbox.files['/work/native-production/worker/runner.py']==worker.RUNNER.encode()
