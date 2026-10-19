"""Interrupted quanta resume at their same scheduler stage with real earned state."""
import copy
import fcntl
import json
from pathlib import Path
import pytest
from test_native_interrupted_update import stopped,owned,ordinary,bare,later
from natlang_neuralese.rl import native_raw_handoff as handoff, native_raw_rounds as raw
from natlang_neuralese.rl import native_interrupted_resume as resume
from natlang_neuralese.train.native_resident import binding
from natlang_neuralese.common.jsonio import write_json_durable as write


@pytest.fixture
def pending(stopped,owned,tmp_path):
    x={**owned,**stopped}
    ack=handoff.prepare_saved_update_acknowledgement(binding(x['recovery']),x['receipt'],x['ack'])
    cursor_path=x['root']/'cycle.json';cursor=json.loads(cursor_path.read_text())
    cursor.update(phase='sft',next_stage=0,raw_retry_generation=2,
        current_checkpoint=binding(json.loads((x['out']/'state.json').read_text())['checkpoint']),
        previously_completed_dpo={'updates':250,'must_preserve':True})
    cursor.pop('pending_rl_round',None)
    write(cursor_path,cursor)
    x.update(resume_out=tmp_path/'resumed.json',ops=[{'saved_update_recovery':ack,'receipt':x['receipt']}],
             cursor_path=cursor_path,cursor_before=copy.deepcopy(cursor),
             state_before=json.loads((x['out']/'state.json').read_text()))
    return x


def invoke(x):
    return resume.publish_interrupted_resume(x['config'],x['ops'],x['resume_out'],reason='correct actual task controls')


def test_resume_preserves_stage_and_debt_but_uses_earned_parent_in_fresh_namespace(pending):
    x=pending;payload=x['pin'].read_bytes();result=invoke(x)
    receipt=json.loads(Path(result['path']).read_text())
    cursor=json.loads(x['cursor_path'].read_text());state=json.loads((x['out']/'state.json').read_text())
    assert state['checkpoint']==str(x['pin']) and cursor['current_checkpoint']==binding(x['pin'])
    assert cursor['next_stage']==0 and cursor['cycle']==x['cursor_before']['cycle']
    assert cursor['raw_retry_generation']==3
    assert cursor['previously_completed_dpo']==x['cursor_before']['previously_completed_dpo']
    assert cursor['unknown_cursor']==x['cursor_before']['unknown_cursor']
    assert 'readiness'not in cursor and'foundation_qualified'not in cursor
    assert all(state[k]==v for k,v in x['state_before'].items()if k not in('checkpoint','phase','actor_policy_pending_load','readiness'))
    assert receipt['training_credit']==0 and receipt['readiness_granted']is False
    assert receipt['whole_recipe_completed']is False
    assert cursor['interrupted_supervised_operations']==x['ops']
    assert raw.cycle_work_directory(x['root'],cursor)!=raw.cycle_work_directory(x['root'],x['cursor_before'])
    assert x['pin'].read_bytes()==payload
    with pytest.raises(ValueError,match='historical only'):
        raw._owned_operation_chain(x['ops'],binding(x['pin']),receipt.get('model_id','none'),'none')
    state['subsequent_earned_updates']=11;write(x['out']/'state.json',state)
    assert invoke(x)==result
    assert json.loads((x['out']/'state.json').read_text())['subsequent_earned_updates']==11


def test_partial_commit_retries_only_exact_known_successors(pending,monkeypatch):
    x=pending;real=resume.write_bytes_durable;state_path=x['out']/'state.json'
    def fail_once(path,raw_bytes):
        if Path(path)==state_path:raise OSError('actual metadata publication interrupted')
        return real(path,raw_bytes)
    monkeypatch.setattr(resume,'write_bytes_durable',fail_once)
    with pytest.raises(OSError):invoke(x)
    assert not x['resume_out'].exists()
    monkeypatch.setattr(resume,'write_bytes_durable',real)
    assert json.loads(Path(invoke(x)['path']).read_text())['same_stage_retry']
    assert json.loads(state_path.read_text())['checkpoint']==str(x['pin'])


@pytest.mark.parametrize('tamper',['stage','phase','pending-policy','pending-native','namespace','lease'])
def test_resume_rejects_other_work_or_held_controller(pending,tamper):
    x=pending;cursor=json.loads(x['cursor_path'].read_text());state_path=x['out']/'state.json'
    state=json.loads(state_path.read_text());lease=None
    if tamper=='stage':cursor['next_stage']=999
    elif tamper=='phase':cursor['phase']='rl'
    elif tamper=='pending-policy':cursor['pending_rl_round']=10
    elif tamper=='pending-native':state['pending_native_checkpoint']='sealed'
    elif tamper=='namespace':
        next_cursor={**cursor,'raw_retry_generation':3};raw.cycle_work_directory(x['root'],next_cursor).mkdir(parents=True)
    else:
        lease=(x['root'].parent/'controller.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(x['cursor_path'],cursor);write(state_path,state)
    try:
        with pytest.raises((ValueError,BlockingIOError)):invoke(x)
        assert not x['resume_out'].exists()
    finally:
        if lease is not None:lease.close()
