"""Disjoint repair capacity preserves valid live owners and immutable attempts."""
import copy
import json
from types import SimpleNamespace
import pytest
from scripts.coreweave import teacher_generation_guardian as guardian
from test_teacher_generation_guardian import repair_fixture


def fixture(tmp_path, monkeypatch):
    config, receipt = repair_fixture(tmp_path)
    config['repair_parallel_batches'] = 2
    queue = receipt.parent/'queue.jsonl'
    rows = [json.loads(line) for line in queue.read_text().splitlines()]
    other = tmp_path/'repairs/batch-0014/queue.jsonl'
    other.parent.mkdir(parents=True)
    other.write_text(''.join(json.dumps({**row, 'key':'batch-0014:case-0'})+'\n' for row in rows))
    old = {'pid':101,'start_ticks':'10','boot_id':'boot','argv':['python','run_bonsai_queue.py']}
    attempt = {'token':'old-token','owner':old,'state':'launched'}
    guardian.write(receipt, {'source_queue':guardian.binding(queue), 'current':attempt,
                            'attempts':[], 'terminal_keys':{}})
    live = {101}; calls = []
    monkeypatch.setattr(guardian,'alive',lambda owner:bool(owner and owner['pid'] in live))
    monkeypatch.setattr(guardian,'recover_token',lambda token,**kwargs:None)
    def spawn(command, **kwargs):
        calls.append(command); pid=200+len(calls);live.add(pid)
        return SimpleNamespace(pid=pid)
    monkeypatch.setattr(guardian.subprocess,'Popen',spawn)
    monkeypatch.setattr(guardian,'process',lambda pid:{'pid':pid,'start_ticks':str(pid),'boot_id':'boot','argv':[b'python',b'run_bonsai_queue.py']})
    return config,receipt,other,old,live,calls


def test_adds_disjoint_batch_without_signalling_or_mutating_live_queue(tmp_path,monkeypatch):
    config,receipt,other,old,live,calls=fixture(tmp_path,monkeypatch)
    original=receipt.read_bytes(); old_queue=(receipt.parent/'queue.jsonl').read_bytes()
    state={'repair_owner':copy.deepcopy(old)}
    guardian.launch_repair(tmp_path,state,config)
    assert len(calls)==1 and str(other.parent/'collector.lock') in calls[0]
    assert str(other.parent/'attempt-0001/queue.jsonl') in calls[0]
    assert state['repair_owner']==old and len(state['repair_batch_owners'])==2
    assert receipt.read_bytes()==original and (receipt.parent/'queue.jsonl').read_bytes()==old_queue
    # Recovery after a lost final state write discovers the existing receipt
    # owner and never spawns a second attempt for its still-live batch.
    state={'repair_owner':copy.deepcopy(old)}
    guardian.launch_repair(tmp_path,state,config)
    assert len(calls)==1 and len(state['repair_batch_owners'])==2
    config['repair_parallel_batches']=1
    guardian.launch_repair(tmp_path,state,config)
    assert len(calls)==1 and len(state['repair_batch_owners'])==2 and live=={101,201}


def test_unknown_live_legacy_reserves_capacity_and_changed_queue_is_rejected(tmp_path,monkeypatch):
    config,receipt,other,old,live,calls=fixture(tmp_path,monkeypatch)
    unknown={**old,'pid':99};live.add(99)
    guardian.launch_repair(tmp_path,{'repair_owner':unknown},config)
    assert not calls  # The known and unknown obligations consume both slots.
    (receipt.parent/'queue.jsonl').write_text('changed immutable queue\n')
    with pytest.raises(ValueError,match='source queue changed'):
        guardian.launch_repair(tmp_path,{'repair_owner':old},config)
    assert not calls


@pytest.mark.parametrize('limit',[None,False,0,-1,1.5,'2'])
def test_invalid_parallel_setting_rejected_before_spawn(tmp_path,monkeypatch,limit):
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:pytest.fail('spawned'))
    with pytest.raises(ValueError,match='positive integer'):
        guardian.launch_repair(tmp_path,{}, {'repair_parallel_batches':limit})


def test_live_flock_owner_and_recoverable_child_use_one_slot(tmp_path,monkeypatch):
    config,receipt,other,old,live,calls=fixture(tmp_path,monkeypatch)
    old['argv']=['flock','-n','/control/repair-collector.lock','nice','-n','19','python','run_bonsai_queue.py']
    value=json.loads(receipt.read_text());value['current']['owner']=copy.deepcopy(old)
    guardian.write(receipt,value);before=receipt.read_bytes()
    child={'pid':151,'start_ticks':'15','boot_id':'boot','argv':[b'python',b'run_bonsai_queue.py']}
    live.add(151);recoveries=[]
    def recover(token,**kwargs):
        recoveries.append(token)
        return child if token=='old-token'else None
    monkeypatch.setattr(guardian,'recover_token',recover)
    state={'repair_owner':copy.deepcopy(old)}
    guardian.launch_repair(tmp_path,state,config)
    assert len(calls)==1 and len(state['repair_batch_owners'])==2
    assert state['repair_owner']==old and receipt.read_bytes()==before
    assert 'old-token'not in recoveries
    guardian.launch_repair(tmp_path,state,config)
    assert len(calls)==1 and receipt.read_bytes()==before
