import hashlib,json,os,subprocess,sys
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from test_diloco_recovery import load


def test_health_identifies_actual_live_and_dead_owner(tmp_path):
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
        assert 'one trainer module' in value['0']['runtime_check_scope']
        progress=tmp_path/'fresh-learner'/'progress.json';progress.parent.mkdir()
        progress.write_text(json.dumps({'global_step':2513,'checkpoint_step':2512,'timestamp':123.}))
        ownership['workers']['0']['progress_path']=str(progress)
        value=service.health_snapshot(record,ownership,read_sandbox=lambda ident:Box())
        assert value['0']['progress']['global_step']==2513
        assert value['0']['progress_path']==str(progress)
        # A new worker can use a different exact package from an older cohort.
        bound=tmp_path/'current-trainer.py';bound.write_text('current runtime\n')
        ownership['workers']['0'].update(module_path=str(bound),module_sha256=hashlib.sha256(bound.read_bytes()).hexdigest())
        ownership['runtime']['new_module_sha256']='0'*64
        value=service.health_snapshot(record,ownership,read_sandbox=lambda ident:Box())
        assert value['0']['state']=='live' and value['0']['runtime_matches']
        ownership['workers']['0']['progress_path']='relative/progress.json'
        value=service.health_snapshot(record,ownership,read_sandbox=lambda ident:Box())
        assert value['0']['state']=='unavailable'
        ownership['workers']['0']['progress_path']=str(progress)
        p.terminate();p.wait()
        value=service.health_snapshot(record,ownership,read_sandbox=lambda ident:Box())
        assert value['0']['state']=='dead-replaced-or-runtime-mismatch' and not value['0']['pid_alive']
    finally:
        if p.poll() is None:p.terminate();p.wait()


def test_coordinator_health_checks_authenticated_http_and_exact_run(tmp_path):
    service=load('diloco_periodic_recovery')
    token='secret-test-token';(tmp_path/'diloco-token').write_text(token)
    status={'run_id':'current-run','merges':3,'versions':[1,2],'global_tokens':123,'live_workers':4}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.headers['Authorization']=='Bearer '+token
            assert self.path=='/v1/status'
            data=json.dumps(status).encode();self.send_response(200)
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'])
    module=tmp_path/'coordinator.py';module.write_text('bound coordinator\n')
    role={'sandbox_id':'one','pid':p.pid,'boot_id':open('/proc/sys/kernel/random/boot_id').read().strip(),
          'start_ticks':open(f'/proc/{p.pid}/stat').read().rsplit(')',1)[1].split()[19],
          'module_path':str(module),'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
          'progress_path':str(tmp_path/'absent-progress.json'),'http_port':server.server_port,'secrets_directory':str(tmp_path)}
    record={'coordinator':'one','run':{'settings':{'run_id':'current-run'}}}
    class Future:
        def __init__(self,result):self.value=result
        def result(self,**kwargs):return self.value
    class Box:
        status='running'
        def get_status(self):pass
        def exec(self,args,**kwargs):return Future(subprocess.run(args,capture_output=True,text=True))
    try:
        value=service.coordinator_health_snapshot(record,{'coordinator':role},read_sandbox=lambda _:Box())
        assert value['state']=='live' and value['http_ok'] and value['progress'] is None
        assert value['coordinator_status']['merges']==3
        status['run_id']='different-run'
        value=service.coordinator_health_snapshot(record,{'coordinator':role},read_sandbox=lambda _:Box())
        assert value['state']=='coordinator-http-unavailable' and not value['http_ok']
        assert service.coordinator_health_snapshot(record,{'coordinator':{}},read_sandbox=lambda _:Box())['state']=='coordinator-configuration-unavailable'
    finally:
        p.terminate();p.wait();server.shutdown();server.server_close();thread.join()
