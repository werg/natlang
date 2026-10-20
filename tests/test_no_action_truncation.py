"""Exact completed no-action evidence, separate from live/pending model work."""
import copy
import json
from pathlib import Path
import pytest
from scripts import run_bonsai_queue as queue
from natlang_neuralese.eval.task_packet import classify_case


def turn(index,*,invocation='child',progress=None,calls=None,truncated=True):
    value={'invocation_id':invocation,'request_sha256':f'{index:064x}',
        'response':{'calls':calls or[],'text':f'synthetic partial {index}',
            'completion_tokens':8,'truncated':truncated,
            'raw_response':{'choices':[{'finish_reason':'length'if truncated else'stop'}]}}}
    if progress is not None:
        value['execution_progress']={'schema':'natlang.completed-action-progress/1',
            'invocation_id':invocation,'successful_actions':progress,
            'last_success_seq':progress*10 if progress else None}
    return value


def partial(tmp_path,turns,**changes):
    job={'key':'actual-job','program_id':'synthetic-only','digest':'d'*64}
    entry={'count':1,'jobs':str(tmp_path),'_resolved_jobs':[job]}
    value={'program_id':job['program_id'],'provenance':{'program_ir_sha256':job['digest']},'turns':turns,**changes}
    (tmp_path/'actual-job.partial.json').write_text(json.dumps(value))
    return entry


def test_legacy_completed_no_action_needs_no_tool_observation_and_stays_per_invocation(tmp_path):
    rows=[turn(index)for index in range(3)]
    entry=partial(tmp_path,rows)
    evidence=queue.identical_turn_streak(entry,3)
    assert evidence['reason']=='consecutive_truncated_no_action'and evidence['completed_no_action_replies']==3
    assert queue.identical_turn_streak(entry,0)is None
    for change in ('other-invocation','pending','nontruncated','proposal','authored','wrong-source'):
        changed=copy.deepcopy(rows);options={}
        if change=='other-invocation':changed[1]['invocation_id']='different-child'
        elif change=='pending':changed.pop()
        elif change=='nontruncated':changed[-1]['response']['truncated']=False
        elif change=='proposal':changed[1]['response']['calls']=[['eval',{'code':'1'}]]
        elif change=='authored':changed[1]['response']['raw_response']['natlang_action_provenance']={'sampled':False}
        else:options={'program_id':'stale-other-source'}
        assert queue.identical_turn_streak(partial(tmp_path,changed,**options),3)is None


def test_real_progress_resets_but_unexecuted_proposals_do_not(tmp_path):
    rows=[turn(0,progress=0),turn(1,progress=0,calls=[['unknown',{}]],truncated=False),
          turn(2,progress=0),turn(3,progress=0)]
    evidence=queue.identical_turn_streak(partial(tmp_path,rows),3)
    assert evidence['completed_no_action_replies']==3
    # Real successful action receipt, rather than the emitted proposal, resets.
    rows[-1]['execution_progress'].update(successful_actions=1,last_success_seq=10)
    assert queue.identical_turn_streak(partial(tmp_path,rows),3)is None
    # An old failing child's history must not terminate a newer invocation.
    rows.append(turn(4,invocation='new-child',progress=0))
    assert queue.identical_turn_streak(partial(tmp_path,rows),3)is None
    rows[0]['execution_progress']['successful_actions']=2
    with pytest.raises(ValueError,match='progress receipt'):
        queue.identical_turn_streak(partial(tmp_path,rows[:-1]),3)


def test_existing_exact_executed_action_observation_guard_remains(tmp_path):
    rows=[turn(i,calls=[['eval',{'code':'1'}]],truncated=False)for i in range(3)]
    for row in rows:row['last_tool_observation']={'content_sha256':'b'*64}
    evidence=queue.identical_turn_streak(partial(tmp_path,rows),3)
    assert evidence['observation_sha256']=='b'*64 and'reason'not in evidence


def test_explicit_terminal_model_incomplete_is_scored_but_never_overrides_success(tmp_path):
    evidence=queue.identical_turn_streak(partial(tmp_path,[turn(i)for i in range(3)]),3)
    finish={'status':'model_incomplete','resource_limit_reason':'consecutive_truncated_no_action',
            'model_incomplete_evidence':evidence}
    assert classify_case(None,finish)=='model_incomplete'
    assert classify_case({'outcome':{'status':'done','accepted':True}},finish)=='task_success'
    broken=copy.deepcopy(finish);broken['model_incomplete_evidence']['request_sha256'][0]='invalid'
    assert classify_case(None,broken)=='unresolved'
    assert classify_case({'outcome':{'status':'quiesced','detail':'model_incomplete: truncated_without_action (configured repair limit reached)'}},
        {'status':'complete'})=='model_incomplete'


def test_queue_terminates_only_owned_child_and_keeps_terminal_evidence(tmp_path,monkeypatch):
    from test_bonsai_queue import queue_entry,install_child,TS_HOST
    entry=queue_entry(tmp_path);job=queue.resolve_entry_jobs(entry,TS_HOST)[0]
    Path(entry['jobs']).mkdir()
    rows=[turn(i)for i in range(3)]
    Path(entry['jobs'],job['key']+'.partial.json').write_text(json.dumps({
        'program_id':job['program_id'],'provenance':{'program_ir_sha256':job['digest']},'turns':rows}))
    source=tmp_path/'queue.jsonl';source.write_text(json.dumps(entry)+'\n');journal=tmp_path/'journal.jsonl'
    children=[]
    class Child:
        def __init__(self,command,**kwargs):self.command=command;self.terminated=False;self.reaped=False;children.append(self)
        def wait(self,timeout=None):
            if self.terminated:self.reaped=True;return-15
            raise queue.subprocess.TimeoutExpired(self.command,timeout)
        def terminate(self):self.terminated=True
        def kill(self):pytest.fail('cooperative owned-child termination should drain')
    install_child(monkeypatch,Child)
    monkeypatch.setattr(queue,'local_decode_progress',lambda *a:999)
    monkeypatch.setattr(queue,'cooldown_remaining',lambda *a:0,raising=False)
    # Skip a live HTTP health check; this fixture has no model/server.
    class Healthy:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        status=200
    monkeypatch.setattr(queue.urllib.request,'urlopen',lambda *a,**k:Healthy())
    assert queue.run_queue(source,journal,TS_HOST,seconds=60,identical_turn_limit=3)==2
    assert len(children)==1 and children[0].terminated and children[0].reaped
    finish=next(row for row in map(json.loads,journal.read_text().splitlines())if row['event']=='finish')
    assert finish['status']=='model_incomplete'and finish['failure_streak']==0
    assert classify_case(None,finish)=='model_incomplete'
