import json,os,signal,subprocess,sys,time
from pathlib import Path


def test_persistent_role_replacement_and_orderly_shutdown(tmp_path):
    supervisor=Path(__file__).parents[1]/'scripts/coreweave/diloco_role_supervisor.py'
    role=tmp_path/'role.py'
    role.write_text('''import pathlib,signal,sys,time\np=pathlib.Path(sys.argv[1]); role=sys.argv[2]\ndef stop(*_):\n if role=="worker":time.sleep(.2);(p/"checkpoint").write_text("saved")\n else:(p/"coord-stop").write_text(str((p/"checkpoint").exists()))\n sys.exit(0)\nsignal.signal(signal.SIGTERM,stop)\nwhile True:time.sleep(.05)\n''')
    control=tmp_path/'control';proc=subprocess.Popen([sys.executable,str(supervisor),str(control)])
    def wait(predicate):
        end=time.monotonic()+8
        while time.monotonic()<end:
            try:
                if predicate():return
            except (OSError,ValueError,KeyError):pass
            time.sleep(.02)
        raise AssertionError('timed out waiting for supervisor')
    def state():return json.loads((control/'state.json').read_text())
    def request(n,action,role,pid=None):
        value={'schema':'natlang.diloco-role-request/1','id':str(n),'expected_generation':n-1,'owner_id':state()['owner_id'],'action':action,'role':role}
        if action=='start':value['argv']=[sys.executable,str(rolefile),str(tmp_path),role]
        else:value['expected_pid']=pid
        temp=control/'request.tmp';temp.write_text(json.dumps(value));os.replace(temp,control/'request.json')
        wait(lambda:json.loads((control/'result.json').read_text()).get('id')==str(n))
        result=json.loads((control/'result.json').read_text());assert result['accepted'];return result['pid']
    rolefile=role
    try:
        wait(lambda:control.joinpath('state.json').exists());c=request(1,'start','coordinator');w=request(2,'start','worker')
        request(3,'stop','coordinator',c);wait(lambda:state()['roles']['coordinator']['alive'] is False)
        assert proc.poll() is None and state()['roles']['worker']['alive']
        c2=request(4,'start','coordinator');assert c2!=c and proc.poll() is None
        proc.send_signal(signal.SIGTERM);assert proc.wait(timeout=8)==0
        assert (tmp_path/'checkpoint').exists() and (tmp_path/'coord-stop').read_text()=='True'
    finally:
        if proc.poll() is None:proc.kill();proc.wait()


def test_real_bootstrap_handoff_waits_for_role_owner(tmp_path):
    repo = Path(__file__).parents[1]
    work = tmp_path / 'work'
    env = {**os.environ, 'NATLANG_REPO': str(repo), 'NATLANG_CW_WORK': str(work),
           'NATLANG_CW_MIRROR_ROOTS': str(tmp_path / 'mirror'),
           'NATLANG_DILOCO_ROLE_SUPERVISION': '1', 'NATLANG_CW_COLLECT_WAIT': '0'}
    # Avoid package installation if the host does not have zstd; this test uses no bundle.
    fakebin = tmp_path / 'bin'; fakebin.mkdir()
    (fakebin / 'zstd').symlink_to('/bin/true')
    env['PATH'] = str(fakebin) + os.pathsep + env['PATH']
    child_marker = tmp_path / 'original-exited'
    proc = subprocess.Popen(['bash', str(repo / 'scripts/coreweave/bootstrap.sh'), sys.executable,
                             '-c', 'from pathlib import Path;import sys;Path(sys.argv[1]).write_text("done")', str(child_marker)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    control = work / '.coreweave/role-control'
    try:
        end = time.monotonic() + 10
        while time.monotonic() < end and not (control / 'state.json').exists():
            assert proc.poll() is None
            time.sleep(.05)
        assert child_marker.read_text() == 'done'
        state = json.loads((control / 'state.json').read_text())
        assert state['generation'] == 0 and state['roles'] == {}
        request = {'schema': 'natlang.diloco-role-request/1', 'id': 'replacement',
                   'expected_generation': 0, 'owner_id': state['owner_id'], 'role': 'worker', 'action': 'start',
                   'argv': [sys.executable, '-c', 'import time;time.sleep(30)']}
        temp = control / 'request.tmp'; temp.write_text(json.dumps(request)); os.replace(temp, control / 'request.json')
        while time.monotonic() < end and not (control / 'result.json').exists(): time.sleep(.05)
        assert json.loads((control / 'result.json').read_text())['accepted']
        assert proc.poll() is None
        proc.terminate(); assert proc.wait(timeout=8) == 0
    finally:
        if proc.poll() is None: proc.kill(); proc.wait()


def test_restarted_owner_rejects_stale_requests(tmp_path):
    supervisor = Path(__file__).parents[1] / 'scripts/coreweave/diloco_role_supervisor.py'
    control = tmp_path / 'control'
    def wait_for(predicate):
        end = time.monotonic() + 8
        while time.monotonic() < end:
            try:
                if predicate(): return
            except (OSError, ValueError, KeyError): pass
            time.sleep(.03)
        raise AssertionError('supervisor timed out')
    def state(): return json.loads((control / 'state.json').read_text())
    first = subprocess.Popen([sys.executable, str(supervisor), str(control)])
    second = None
    try:
        wait_for(lambda: (control / 'state.json').exists())
        old = state(); first.terminate(); assert first.wait(timeout=8) == 0
        second = subprocess.Popen([sys.executable, str(supervisor), str(control)])
        wait_for(lambda: state()['owner_id'] != old['owner_id'])
        request = {'schema':'natlang.diloco-role-request/1', 'id':'stale', 'owner_id':old['owner_id'],
                   'expected_generation':old['generation'], 'role':'worker', 'action':'start',
                   'argv':[sys.executable, '-c', 'pass']}
        tmp=control/'request.tmp'; tmp.write_text(json.dumps(request)); os.replace(tmp, control/'request.json')
        wait_for(lambda: (control/'result.json').exists())
        assert not json.loads((control/'result.json').read_text())['accepted']
        assert not state()['roles']
    finally:
        if first.poll() is None: first.kill(); first.wait()
        if second and second.poll() is None: second.terminate(); second.wait(timeout=8)
