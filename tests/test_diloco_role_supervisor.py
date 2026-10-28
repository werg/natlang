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
        value={'schema':'natlang.diloco-role-request/1','id':str(n),'expected_generation':n-1,'action':action,'role':role}
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
