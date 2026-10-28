import copy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/coreweave'))
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
