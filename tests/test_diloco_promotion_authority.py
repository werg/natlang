import json,os,subprocess,sys,time
from pathlib import Path
import pytest
from scripts.coreweave.diloco_promotion_authority import fence,release


def test_fence_rejects_direct_worker_start_until_exact_release(tmp_path):
    script=Path(__file__).parents[1]/'scripts/coreweave/diloco_role_supervisor.py'
    owner=subprocess.Popen([sys.executable,str(script),str(tmp_path)])
    def wait(fn):
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            try:
                value=fn()
                if value:return value
            except (OSError,ValueError,KeyError):pass
            time.sleep(.02)
        raise AssertionError('timeout')
    try:
        state=wait(lambda:json.loads((tmp_path/'state.json').read_text()))
        args={'expected_owner':state['owner_id'],'expected_boot':state['boot_id'],'nonce':'a'*48}
        fence(tmp_path,**args)
        request={'schema':'natlang.diloco-role-request/1','id':'racing-start','owner_id':state['owner_id'],
                 'expected_generation':state['generation'],'action':'start','role':'worker',
                 'argv':[sys.executable,'-c','import time;time.sleep(5)']}
        path=tmp_path/'request.tmp';path.write_text(json.dumps(request));os.replace(path,tmp_path/'request.json')
        result=wait(lambda: json.loads((tmp_path/'result.json').read_text()))
        assert result['accepted'] is False and 'fenced' in result['error']
        with pytest.raises(ValueError):release(tmp_path,**dict(args,nonce='b'*48))
        release(tmp_path,**args)
    finally:
        owner.terminate();owner.wait(timeout=8)


def test_nested_owner_handoff_fences_old_and_new_publishers(tmp_path):
    script=Path(__file__).parents[1]/'scripts/coreweave/diloco_role_supervisor.py'
    old=tmp_path/'old';nested=tmp_path/'nested'
    proc=subprocess.Popen([sys.executable,str(script),str(old)])
    def wait(fn):
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            try:
                v=fn()
                if v:return v
            except (OSError,ValueError,KeyError):pass
            time.sleep(.02)
        raise AssertionError('timeout')
    def state(d):return json.loads((d/'state.json').read_text())
    def submit(d,v):
        t=d/('request-'+v['id']);t.write_text(json.dumps(v));os.replace(t,d/'request.json')
        return wait(lambda: (x if (x:=json.loads((d/'result.json').read_text())).get('id')==v['id'] else None))
    def start(d,id,argv):
        s=state(d);return submit(d,{'schema':'natlang.diloco-role-request/1','id':id,'owner_id':s['owner_id'],
            'expected_generation':s['generation'],'role':'worker','action':'start','argv':argv})
    try:
        oldstate=wait(lambda:state(old))
        import hashlib
        from scripts.coreweave.diloco_nested_owner import prepare
        prepared=prepare(source=script,expected_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),python=sys.executable,control_directory=str(nested),runtime=str(Path(__file__).parents[1]))
        nested_start=start(old,'nested',prepared['argv']);assert nested_start['accepted']
        assert hashlib.sha256(Path(f"/proc/{nested_start['pid']}/cmdline").read_bytes()).hexdigest()==prepared['supervisor_cmdline_sha256']
        nestedstate=wait(lambda:state(nested))
        stale={'schema':'natlang.diloco-role-request/1','id':'stale','owner_id':oldstate['owner_id'],
            'expected_generation':oldstate['generation'],'role':'worker','action':'start','argv':[sys.executable,'-c','pass']}
        assert not submit(old,stale)['accepted']
        assert not start(old,'old-replacement',[sys.executable,'-c','pass'])['accepted']
        authority={'expected_owner':nestedstate['owner_id'],'expected_boot':nestedstate['boot_id'],'nonce':'n'*48}
        fence(nested,**authority)
        assert not start(nested,'nested-fenced',[sys.executable,'-c','pass'])['accepted']
        import socket
        from scripts.coreweave.diloco_promotion_authority import authorize_coordinator,authorize_seed_server
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));seedport=sock.getsockname()[1]
        seed=tmp_path/'seed';seed.mkdir();(seed/'seed-receipt.json').write_text(json.dumps({'files':[]}));tokenfile=tmp_path/'seed-token';tokenfile.write_text('s'*32)
        config=tmp_path/'seed-config.json';config.write_text(json.dumps({'seed':str(seed),'host':'127.0.0.1','port':seedport,'token_file':str(tokenfile)}))
        configsha=hashlib.sha256(config.read_bytes()).hexdigest();ownerstate=state(nested)
        seedrequest={'schema':'natlang.diloco-role-request/1','id':'seed-server','owner_id':ownerstate['owner_id'],
            'expected_generation':ownerstate['generation'],'role':'actor','action':'start','promotion_nonce':authority['nonce'],
            'argv':[sys.executable,'-m','natlang_neuralese.train.diloco.replica_cli','seed-serve','--config',str(config),'--config-sha256',configsha]}
        authorize_seed_server(nested,request=seedrequest,**authority);seedpid=submit(nested,seedrequest)['pid']
        def seed_ready():
            try:
                with socket.create_connection(('127.0.0.1',seedport),timeout=.1):return True
            except OSError:return False
        wait(seed_ready);assert not start(nested,'worker-during-seed',[sys.executable,'-c','pass'])['accepted']
        ownerstate=state(nested);assert submit(nested,{'schema':'natlang.diloco-role-request/1','id':'stop-seed','owner_id':ownerstate['owner_id'],
            'expected_generation':ownerstate['generation'],'role':'actor','action':'stop','expected_pid':seedpid})['accepted']
        wait(lambda:not state(nested)['roles']['actor']['alive'])
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        state_now=state(nested)
        request={'schema':'natlang.diloco-role-request/1','id':'new-coordinator','owner_id':state_now['owner_id'],
                 'expected_generation':state_now['generation'],'role':'coordinator','action':'start','promotion_nonce':authority['nonce'],
                 'argv':[sys.executable,'-c',f'import socket,time;s=socket.socket();s.bind(("127.0.0.1",{port}));s.listen();time.sleep(30)']}
        authorize_coordinator(nested,request=request,**authority)
        assert submit(nested,request)['accepted']
        def ready():
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.1):return True
            except OSError:return False
        wait(ready)
        assert not start(nested,'worker-still-fenced',[sys.executable,'-c','pass'])['accepted']
        release(nested,**authority)
        assert start(nested,'new-worker',[sys.executable,'-c','import time;time.sleep(30)'])['accepted']
    finally:
        proc.terminate();proc.wait(timeout=8)


def test_simultaneous_start_and_fence_cannot_both_succeed(tmp_path):
    import concurrent.futures,threading
    script=Path(__file__).parents[1]/'scripts/coreweave/diloco_role_supervisor.py'
    owner=subprocess.Popen([sys.executable,str(script),str(tmp_path)])
    try:
        limit=time.monotonic()+8
        while not (tmp_path/'state.json').exists() and time.monotonic()<limit:time.sleep(.01)
        state=json.loads((tmp_path/'state.json').read_text());barrier=threading.Barrier(2)
        def attempt_fence():
            barrier.wait()
            try:fence(tmp_path,expected_owner=state['owner_id'],expected_boot=state['boot_id'],nonce='z'*48);return True
            except ValueError:return False
        def attempt_start():
            value={'schema':'natlang.diloco-role-request/1','id':'racing','owner_id':state['owner_id'],
                'expected_generation':state['generation'],'action':'start','role':'worker',
                'argv':[sys.executable,'-c','import time;time.sleep(10)']}
            path=tmp_path/'request.tmp';path.write_text(json.dumps(value));barrier.wait();os.replace(path,tmp_path/'request.json')
            end=time.monotonic()+8
            while time.monotonic()<end:
                try:
                    result=json.loads((tmp_path/'result.json').read_text())
                    if result.get('id')=='racing':return result['accepted']
                except (OSError,ValueError):pass
                time.sleep(.02)
            raise AssertionError('request did not complete')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(attempt_fence);b=pool.submit(attempt_start)
            assert not (a.result() and b.result())
    finally:
        owner.terminate();owner.wait(timeout=8)
