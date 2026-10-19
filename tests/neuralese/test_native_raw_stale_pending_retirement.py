"""Stale RAW work is retired as debt, never acknowledged as completion."""
import copy
import fcntl
import json
import time
from pathlib import Path

import pytest

from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_raw_handoff as handoff
from natlang_neuralese.rl import native_raw_rounds as raw
from natlang_neuralese.train import native_resident as resident


@pytest.fixture(params=[('lfm',3,0,'updating'),('mellum',4,1,'stopped_collecting')],ids=['lfm','mellum'])
def stale_round(tmp_path,monkeypatch,request):
    model,round_number,attempt,phase=request.param
    root=tmp_path/'rounds'/model;root.mkdir(parents=True)
    out=root/'production';out.mkdir()
    controller=root/'config.json';write(controller,{'out':str(out),'raw_model_id':'fixture-model'})
    checkpoint=root/'cycles'/'000004'/'supervised'/'001'/'learner'/'checkpoint.pt'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'checkpoint stays unchanged')
    current=resident.binding(checkpoint)
    active=out/'round-000003';active.mkdir()
    bridge=root/'cycles'/'000004'/'rl-bridge';bridge.mkdir(parents=True)
    (bridge/'sealed-request.json').write_text('{"immutable":true}\n')
    state_path=out/'state.json'
    state={'phase':phase,'round':round_number,'offset':17,'trained_rounds':round_number,'collection_attempt':attempt,
        'checkpoint':str(checkpoint),'production_complete':False,'selection':['case-a','case-b'],
        'task_curriculum_history':{'tool-use':{'attempts':8}},'group_stats':[{'success':0.25}],
        'active_update_round':str(active),'update_mode':'grpo','update_rollouts':str(active/'rollouts.jsonl'),
        'update_stages':['all'],'pending_native_checkpoint':None,'pending_native_round_dir':None,
        'future_debt':{'ordinary_rows':123,'held_rows':19}}
    write(state_path,state)
    cursor_path=root/'cycle.json'
    cursor={'cycle':4,'phase':'rl','next_stage':0,'pending_rl_round':round_number+1,'current_checkpoint':current,
        'resident_current':{'path':'resident/current.json','sha256':'0'*64},
        'readiness':{'stale':True},'future_case_debt':['a','b']}
    write(cursor_path,cursor)
    selection_path=out/f'round-{round_number:06d}-selection.json'
    write(selection_path,{'source_sha256':'1'*64,'selection':[{'row':0},{'row':1}],
        'cases':[{'split':'train','id':'a'},{'split':'train','id':'b'}]})
    q=root.parent/'successor';q.mkdir();(q/'learner').mkdir()
    wrapper=q/'owned-entry.py';wrapper.write_text('# exact in-process wrapper\n')
    gate=q/'gate-spec-preflight.json';write(gate,{'out':str(q/'learner')})
    pid=98765431;boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    argv=['/usr/bin/python','-u',str(wrapper)]
    observed={'pid':pid,'boot_id':boot,'start_ticks':'100','state':'S',
        'argv_hex':b'\0'.join(x.encode() for x in argv).hex()+'00'}
    monkeypatch.setattr(handoff,'observe',lambda p: observed if p==pid else None)
    census_path=tmp_path/'process-census.json'
    write(census_path,{'boot_id':boot,'at':time.time(),'processes':[{'pid':pid,'boot_id':boot,
        'birth':'100','state':'S','argv':argv}]})
    quiescence_path=tmp_path/'provider-quiescence.json'
    write(quiescence_path,{'schema':'natlang.supervisor-legacy-provider-quiescence/1',
        'at':time.time(),'boot_id':boot,
        'training_credit':0,'learner_checkpoint_mutations':[],
        'units':[{'Id':unit,'ActiveState':'inactive','SubState':'dead','MainPID':'0'} for unit in (
            'cw-lfm-cpu26-keep-provider-20261018.service','cw-lfm-cpu26-keep-heartbeat-20261018.service',
            'cw-mellum-cpu26-actual-queue-provider-20261018.service','cw-mellum-cpu26-actual-queue-heartbeat-20261018.service')]})
    authority={'schema':'natlang.native-raw-stale-pending-authority/2',
        'config':resident.binding(controller),'state':resident.binding(state_path),
        'cursor':resident.binding(cursor_path),'selection':resident.binding(selection_path),
        'historical_owner_identity_available':False,'process_census':resident.binding(census_path),
        'permitted_owner':{'identity':{'pid':pid,'boot_id':boot,'start_ticks':'100'},'q_root':str(q),
            'gate_spec':resident.binding(gate),'wrapper':resident.binding(wrapper),'argv':argv},
        'operation_directories':[str(bridge),str(active)],
        'provider_quiescence':resident.binding(quiescence_path),
        'reason':'owner-authorized recovery of an exact stale pending round; no training credit'}
    # Model the actual inherited root controller lease and the separate
    # production lease. The helper must verify and reuse, never unlock, fd.
    canonical=root.parent/'controller.lock';canonical.parent.mkdir(parents=True,exist_ok=True)
    fd=canonical.open('a');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=tmp_path/'retire.json'
    yield {'model':model,'round':round_number,'attempt':attempt,'root':root,'out':out,'config':controller,'state':state_path,'cursor':cursor_path,
        'selection':selection_path,'authority':authority,'receipt':receipt,'fd':fd,'checkpoint':checkpoint,'bridge':bridge,
        'permitted_pid':pid,'observed':observed}
    fd.close()


def retire(x):
    return raw.retire_stale_pending_raw_round(x['config'],x['root']/'cycle.json',x['authority'],
        x['receipt'],owned_controller_fd=x['fd'].fileno())


def test_retirement_preserves_debt_and_sealed_artifacts_then_returns_to_sft(stale_round):
    x=stale_round;state_before=json.loads(x['state'].read_text());cursor_before=json.loads(x['cursor'].read_text())
    checkpoint=x['checkpoint'].read_bytes();bridge=(x['bridge']/'sealed-request.json').read_bytes()
    result=retire(x);state=json.loads(x['state'].read_text());cursor=json.loads(x['cursor'].read_text())
    assert state['phase']=='collect' and state['collection_attempt']==x['attempt']+1
    assert all(state[key]==state_before[key] for key in ('round','offset','trained_rounds','selection',
        'task_curriculum_history','group_stats','future_debt','checkpoint'))
    assert state['stale_pending_raw_retirement']['retired_state_fields']=={
        'active_update_round':state_before['active_update_round'],'update_mode':'grpo',
        'update_rollouts':state_before['update_rollouts'],'update_stages':['all']}
    assert cursor['phase']=='sft' and cursor['next_stage']==0 and 'pending_rl_round' not in cursor
    assert cursor['cycle']==cursor_before['cycle'] and cursor['future_case_debt']==cursor_before['future_case_debt']
    assert cursor['raw_retry_generation']==x['attempt']+1 and 'readiness' not in cursor
    receipt=json.loads(x['receipt'].read_text())
    assert receipt['whole_recipe_completed'] is False and receipt['training_credit']==0
    assert x['checkpoint'].read_bytes()==checkpoint and (x['bridge']/'sealed-request.json').read_bytes()==bridge
    assert retire(x)==result
    assert json.loads(x['state'].read_text())['round']==x['round']


def test_crash_between_pointer_replacements_retries_exact_candidates(stale_round,monkeypatch):
    x=stale_round;actual=raw.write_bytes_durable;cursor=x['cursor']
    def crash(path,data):
        if Path(path)==cursor:raise OSError('simulated cursor publication crash')
        return actual(path,data)
    with monkeypatch.context() as patcher:
        patcher.setattr(raw,'write_bytes_durable',crash)
        with pytest.raises(OSError):retire(x)
    assert json.loads(x['state'].read_text())['phase']=='collect'
    assert json.loads(cursor.read_text())['pending_rl_round']==x['round']+1
    result=retire(x)
    assert json.loads(cursor.read_text())['phase']=='sft'
    assert json.loads(Path(result['path']).read_text())['training_credit']==0


@pytest.mark.parametrize('mutation',['owner-live','live-old-argv','source-tamper','provider-live','held-pending','unknown-pointer','selection-change','lock-path'])
def test_stale_retirement_fails_closed_without_mutating_controller(stale_round,monkeypatch,mutation):
    x=stale_round
    if mutation=='owner-live':
        x['observed']['argv_hex']=b'\0'.join([b'/usr/bin/python',b'old-worker',str(x['bridge']).encode()]).hex()+'00'
    elif mutation=='live-old-argv':
        owner=x['authority']['permitted_owner']['identity']
        pid=__import__('os').getpid();boot=owner['boot_id'];argv=['/usr/bin/python','old-collector',str(x['bridge'])]
        census_path=Path(x['authority']['process_census']['path'])
        write(census_path,{'boot_id':boot,'at':time.time(),'processes':[
            {'pid':owner['pid'],'boot_id':boot,'birth':owner['start_ticks'],'state':'S',
             'argv':x['authority']['permitted_owner']['argv']},
            {'pid':pid,'boot_id':boot,'birth':'999999','state':'S','argv':argv}]})
        x['authority']['process_census']=resident.binding(census_path)
        old_observe=handoff.observe
        def observe(p):
            if p==pid:return {'pid':pid,'boot_id':boot,'start_ticks':'999999','state':'S',
                'argv_hex':b'\0'.join(v.encode() for v in argv).hex()+'00'}
            return old_observe(p)
        monkeypatch.setattr(handoff,'observe',observe)
    elif mutation=='source-tamper':
        source=Path(x['authority']['process_census']['path']);source.write_text('{"boot_id":"wrong","at":1,"processes":[]}')
    elif mutation=='provider-live':
        source=Path(x['authority']['provider_quiescence']['path']);value=json.loads(source.read_text())
        value['units'][0]['ActiveState']='active';write(source,value)
        x['authority']['provider_quiescence']=resident.binding(source)
    elif mutation=='held-pending':
        value=json.loads(x['state'].read_text());value['pending_native_checkpoint']='sealed';write(x['state'],value)
    elif mutation=='unknown-pointer':
        value=json.loads(x['cursor'].read_text());value['pending_round_owner']={'path':'unknown'};write(x['cursor'],value)
    elif mutation=='selection-change':
        value=json.loads(x['selection'].read_text());value['selection'].reverse();write(x['selection'],value)
    else:
        x['fd'].close();fd=(x['root']/'wrong.lock').open('a');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);x['fd']=fd
    before={path:path.read_bytes() for path in (x['state'],x['cursor'])}
    with pytest.raises((ValueError,OSError)):
        retire(x)
    assert not x['receipt'].exists()
    for path,value in before.items():assert path.read_bytes()==value


@pytest.mark.parametrize('generation',['legacy','retry'])
def test_bridge_retry_uses_a_fresh_job_directory(stale_round,generation):
    work=stale_round['root']/'cycles'/'000004'
    cursor={'cycle':4,'raw_retry_generation':0 if generation=='legacy' else 2}
    namespace=raw.cycle_work_directory(stale_round['root'],cursor)
    path=raw._raw_bridge_jobdir(namespace,cursor)
    assert path==(work/'rl-bridge' if generation=='legacy' else work/'retired-attempt-000002'/'rl-bridge')
    assert (path!=stale_round['bridge'])==(generation!='legacy')


def test_retry_namespace_survives_raw_to_sft_boundary_until_cycle_closes(stale_round,monkeypatch):
    x=stale_round;cursor={'cycle':4,'phase':'rl','pending_rl_round':x['round']+1,
        'raw_retry_generation':x['attempt']+1,'current_checkpoint':resident.binding(x['checkpoint'])}
    monkeypatch.setattr(raw,'_current_checkpoint',lambda config:resident.binding(x['checkpoint']))
    next_cursor=raw.completed_raw_cursor({'out':str(x['out'])},cursor,{'phase':'collect','round':x['round']+1})
    assert next_cursor['phase']=='sft' and next_cursor['raw_retry_generation']==x['attempt']+1
    assert raw.cycle_work_directory(x['root'],next_cursor)==x['root']/'cycles'/'000004'/f'retired-attempt-{x["attempt"]+1:06d}'
