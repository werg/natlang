import json
import pytest
from test_diloco_recovery import load


def test_fence_accepts_only_bound_final_report_for_dead_worker(tmp_path, monkeypatch):
    service=load('diloco_periodic_recovery');backend=object.__new__(service.Backend)
    backend.workers=['dead'];backend.fingerprint='fp';backend.out='/work/out';backend.repo=tmp_path
    cap={'schema':'natlang.diloco-recovery-fence/1','worker':0,'fingerprint':'fp','owner':{'pid':123,'boot_id':'old','start_ticks':'1'}}
    report={'active_stage_complete':True,'checkpoint_sha256':'a'*64,'diloco_final_delivery':{'schema':'natlang.diloco-final-delivery/1','worker':0,'fingerprint':'fp','sequences':[2]}}
    class Result:
        def __init__(self,value=None,code=0):self.value=value;self.returncode=code
        def result(self,**kw):return self.value if self.value is not None else self
    class Box:
        def read_file(self,path):return Result(json.dumps(report if path.endswith('report.json') else cap))
        def exec(self,args,**kw):return Result(code=1 if "o['boot_id']" in ' '.join(args) else 0)
        def write_file(self,*args):return Result()
    monkeypatch.setattr(service.cwctl,'read_sandbox',lambda ident:Box())
    monkeypatch.setattr(backend,'pin_coordinator',lambda cycle:{'path':'coord'})
    monkeypatch.setattr(backend,'pin_resident',lambda worker,cycle:{'path':'resident'})
    captured=backend.capture_fenced('cycle',timeout=30)
    assert captured['ready'][0]['completed'] and captured['ready'][0]['sequences']==[2]
    report.pop('diloco_final_delivery')
    with pytest.raises(service.FenceUnavailable,match='final delivery'):
        backend.capture_fenced('cycle',timeout=30)
