"""All original contexts, pinned neighbor facts and shared benchmark protection."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from neuralese_data import acquired_agents as acquired,agents
from neuralese_data.common import Reject
from neuralese_data.records import group_key


def context(answer):
    return [{'role':'user','content':'Original benchmark task'},
            {'role':'assistant','content':'Inspect source'},
            {'role':'assistant','content':'[command] cat source.py\nENTIRE OBSERVED SOURCE'},
            {'role':'assistant','content':answer}]


def source(tmp_path,monkeypatch,ref,files):
    root=tmp_path/ref;root.mkdir()
    entries=[]
    for name,row in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        data=json.dumps(row).encode();path.write_bytes(data)
        entries.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    receipt={'status':'acquired','repo':'pinned/'+ref,'revision':'exact-revision','card':{},'files':entries}
    (root/'acquisition.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(agents,'EXPANSION_ROOT',tmp_path)
    monkeypatch.setenv('NATLANG_HASH_CACHE',str(tmp_path/'hash-cache.jsonl'))
    monkeypatch.setattr(agents,'_bench_ids',lambda:{'repo_task:17'})
    return root


def test_team_source_retains_all_contexts_and_observed_command_logs(tmp_path,monkeypatch):
    parent='run/coop/repo_task/17/f1_f2/'
    row={'repo':'repo_task','task_id':17,'feature_id':1,'agent_id':'agent1','status':'Submitted','model':'actual-teacher',
        'mini_swe_trajectory_format':'codex_stream_log','messages':context('Final'),
        'mini_swe_segments':[{'kind':'solver','messages':context('Before compaction')},
                             {'kind':'summarizer','messages':context('Original summary')}],
        'team':{'run_id':'actual-run','lead_agent':'agent1'}}
    evaluation={'repo':'repo_task','task_id':17,'correct':True,'both_passed':True,'verified':True,
                'eval_source':'cooperbench_held_out_tests'}
    root=source(tmp_path,monkeypatch,'D13',{parent+'agent1_traj.json':row,parent+'eval.json':evaluation,
        parent+'conversation.json':[{'from':'agent2','content':'Original communication'}]})
    monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({parent+'agent1_traj.json'}))
    result=list(acquired.team_cooperation_acquired())
    assert len(result)==3 and all(not isinstance(t,Reject) for _,t in result)
    for _,t in result:
        assert t.split=='test' and group_key('cooper-task','repo_task:17') in t.split_groups
        assert t.messages[2]['role']=='tool' and t.messages[2]['content']==row['messages'][2]['content']
        assert 'tool_calls' not in t.messages[2],'no invented typed boundary inside combined source log'
        assert t.outcome['label']=='teacher' and t.outcome['checked'] is None
        details=t.lineage['notes']['original_cooperation_context']
        assert details['neighbor_bindings']['eval.json']['sha256']==hashlib.sha256((root/(parent+'eval.json')).read_bytes()).hexdigest()
        assert details['neighbors']['conversation.json'][0]['content']=='Original communication'
        assert details['team']['run_id']=='actual-run' and details['observed_command_logs'][0]['index']==2
    # Changing an adjacent quality document cannot reuse a cached positive
    # classification, even though the trajectory itself has identical bytes.
    (root/(parent+'eval.json')).write_text('tampered')
    rejected=list(acquired.team_cooperation_acquired())
    assert isinstance(rejected[0][1],Reject) and 'differs' in str(rejected[0][1])


def test_extra_qwen_agents_without_full_copy_are_not_omitted(tmp_path,monkeypatch):
    parent='coop/repo_task/17/f1_f4/'
    row={'repo':'repo_task','task_id':17,'agent_id':'source-agent2','feature_id':4,'status':'Submitted',
         'model':'source-teacher','messages':context('Extra original agent')}
    files={parent+'agent4_traj.json':row,parent+'eval.json':{'repo':'repo_task','task_id':17,'both_passed':False}}
    source(tmp_path,monkeypatch,'D14',files)
    monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({parent+'agent4_traj.json'}))
    result=list(acquired.qwen_cooperation_acquired())
    assert len(result)==1 and not isinstance(result[0][1],Reject)
    t=result[0][1]
    assert t.messages[-1]['content']=='Extra original agent' and t.outcome['label']=='failed'
    assert t.split=='test' and group_key('cooper-task','repo_task:17') in t.split_groups
    details=t.lineage['notes']['original_cooperation_context']
    assert details['agent_id']=='source-agent2' and details['path'].endswith('agent4_traj.json')
    assert acquired.cooperation_trajectory_file('D14',parent+'agent4_traj.json',files)
    assert not acquired.cooperation_trajectory_file('D14',parent+'agent4_traj.json',set(files)|{parent+'agent4_full_traj.json'})


def test_wrong_task_neighbor_is_not_authenticated_by_filename_alone(tmp_path,monkeypatch):
    parent='run/coop/repo_task/17/f1_f2/'
    source(tmp_path,monkeypatch,'D13',{parent+'agent1_traj.json':{'repo':'repo_task','task_id':17,'messages':context('Final')},
        parent+'eval.json':{'repo':'different','task_id':17,'both_passed':True}})
    monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({parent+'agent1_traj.json'}))
    _,rejected=next(acquired.team_cooperation_acquired())
    assert isinstance(rejected,Reject) and 'another original benchmark task' in str(rejected)
