import copy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/coreweave'))
import diloco_periodic_recovery as recovery
from diloco_periodic_recovery import observe_optimizer_progress


def test_four_worker_progress_distinguishes_alive_from_advancing_and_owner_change():
    def worker(step,pid):
        return {'state':'live','sandbox_id':f'box-{pid}',
                'process_identity':{'pid':pid,'boot_id':'boot','start_ticks':'1'},
                'progress':{'global_step':step}}
    workers={str(i):worker(100+i,i+1) for i in range(4)}
    first=observe_optimizer_progress(workers,None,1000)
    assert all(item['optimizer_observation']['status']=='baseline' for item in first.values())
    workers['0']['progress']['global_step']+=1
    workers['2']['progress']['global_step']-=1
    workers['3']=worker(4,99)
    second=observe_optimizer_progress(workers,{'workers':first},1060)
    assert second['0']['optimizer_observation']['status']=='advanced'
    assert second['1']['optimizer_observation']['status']=='unchanged'
    assert second['1']['optimizer_observation']['seconds_without_observed_advance']==60
    assert second['2']['optimizer_observation']['status']=='regressed'
    assert second['3']['optimizer_observation']['status']=='baseline'
    third=observe_optimizer_progress(workers,{'workers':second},1120)
    assert third['0']['optimizer_observation']['seconds_without_observed_advance']==60
    assert third['1']['optimizer_observation']['last_observed_advance_at'] is None
    unavailable=copy.deepcopy(workers);unavailable['1']['state']='unavailable'
    assert observe_optimizer_progress(unavailable,{'workers':third},1180)['1']['optimizer_observation']=={'status':'unavailable'}


def test_periodic_recovery_retries_initialization_and_partial_exports_promptly(monkeypatch,tmp_path):
    calls=[];sleeps=[];backend=object()
    def factory():
        calls.append('init')
        if calls.count('init')==1:raise OSError('transient coordinator read failure')
        return backend
    def run_once(record,state_path,value):
        assert value is backend and state_path==tmp_path/'state.json'
        calls.append('cycle')
        phase='incomplete' if calls.count('cycle')==1 else 'verified-components-and-dependencies'
        return {'pending':{'id':'same-journal','phase':phase,'errors':{}},'cycles':[]}
    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps)==3:raise StopIteration
    monkeypatch.setattr(recovery,'run_once',run_once)
    try:recovery.supervise_recovery({},tmp_path/'state.json',factory,interval=1800,sleep=sleep)
    except StopIteration:pass
    assert calls==['init','init','cycle','cycle']
    assert sleeps==[60,60,1800]
