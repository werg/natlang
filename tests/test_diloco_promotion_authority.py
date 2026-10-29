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
        assert start(old,'nested',[sys.executable,str(script),str(nested)])['accepted']
        nestedstate=wait(lambda:state(nested))
        stale={'schema':'natlang.diloco-role-request/1','id':'stale','owner_id':oldstate['owner_id'],
            'expected_generation':oldstate['generation'],'role':'worker','action':'start','argv':[sys.executable,'-c','pass']}
        assert not submit(old,stale)['accepted']
        assert not start(old,'old-replacement',[sys.executable,'-c','pass'])['accepted']
        authority={'expected_owner':nestedstate['owner_id'],'expected_boot':nestedstate['boot_id'],'nonce':'n'*48}
        fence(nested,**authority)
        assert not start(nested,'nested-fenced',[sys.executable,'-c','pass'])['accepted']
        release(nested,**authority)
        assert start(nested,'new-worker',[sys.executable,'-c','import time;time.sleep(30)'])['accepted']
    finally:
        proc.terminate();proc.wait(timeout=8)
