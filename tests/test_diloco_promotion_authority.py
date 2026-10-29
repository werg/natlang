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
