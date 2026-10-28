import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/coreweave'))
import diloco_role_control as control


def test_concurrent_role_publishers_never_overwrite_or_duplicate(tmp_path):
    repo=Path(__file__).parents[1];directory=tmp_path/'control'
    owner=subprocess.Popen([sys.executable,str(repo/'scripts/coreweave/diloco_role_supervisor.py'),str(directory)])
    try:
        deadline=time.monotonic()+8
        while not (directory/'state.json').exists() and time.monotonic()<deadline:time.sleep(.02)
        state=json.loads((directory/'state.json').read_text())
        def request(role):
            return {'schema':'natlang.diloco-role-request/1','id':'start-'+role,'action':'start','role':role,
                    'owner_id':state['owner_id'],'expected_generation':0,
                    'argv':[sys.executable,'-c','import time;time.sleep(60)']}
        requests=[request('worker'),request('coordinator')]
        results=[];barrier=threading.Barrier(2)
        def send(value):
            barrier.wait()
            try:results.append(('ok',value,control.submit(directory,value,timeout=8)))
            except ValueError as error:results.append(('stale',value,str(error)))
        threads=[threading.Thread(target=send,args=(value,)) for value in requests]
        for thread in threads:thread.start()
        for thread in threads:thread.join(timeout=10);assert not thread.is_alive()
        assert sorted(x[0] for x in results)==['ok','stale']
        applied=next(x for x in results if x[0]=='ok')
        assert control.submit(directory,applied[1],timeout=8)['already_applied']
        retry=next(x[1] for x in results if x[0]=='stale')
        retry['expected_generation']=json.loads((directory/'state.json').read_text())['generation']
        assert control.submit(directory,retry,timeout=8)['accepted']
        latest=json.loads((directory/'state.json').read_text())
        assert latest['roles']['worker']['alive'] and latest['roles']['coordinator']['alive']
        with pytest.raises(ValueError,match='not PID 1'):
            control._owner(directory,require_pid1=True,entrypoint_sha256=None)
    finally:
        owner.terminate();owner.wait(timeout=8)
