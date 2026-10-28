import hashlib,json,os,subprocess,sys
from test_diloco_recovery import load


def test_health_identifies_actual_live_and_dead_owner():
    service=load('diloco_periodic_recovery')
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
    boot=open('/proc/sys/kernel/random/boot_id').read().strip();ticks=open(f'/proc/{p.pid}/stat').read().split(') ',1)[1].split()[19]
    module=service.paths.root('repo')/'training/neuralese/natlang_neuralese/train/native_state_sft.py'
    record={'learners':['one']};ownership={'runtime':{'module_path':str(module),
        'new_module_sha256':hashlib.sha256(module.read_bytes()).hexdigest()},
        'workers':{'0':{'sandbox_id':'one','pid':p.pid,'start_ticks':ticks,'boot_id':boot}}}
    class Future:
        def __init__(self,value):self.value=value
        def result(self,**kw):return self.value
    class Box:
        status='running'
        def get_status(self):pass
        def exec(self,args,**kw):return Future(subprocess.run(args,capture_output=True,text=True))
    try:
        value=service.health_snapshot(record,ownership,read_sandbox=lambda ident:Box())
        assert value['0']['state']=='live' and value['0']['progress'] is None
        p.terminate();p.wait()
        value=service.health_snapshot(record,ownership,read_sandbox=lambda ident:Box())
        assert value['0']['state']=='dead-replaced-or-runtime-mismatch' and not value['0']['pid_alive']
    finally:
        if p.poll() is None:p.terminate();p.wait()
