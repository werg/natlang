"""Exact format boundaries, without inferred tool calls or source success."""
import copy
import json

import pytest

from scripts.neuralese_data.acquired_granted import complete_messages
from scripts.neuralese_data.common import Reject


def test_native_blocks_preserve_order_reasoning_dispatch_and_complete_observations():
    raw=[{'role':'system','content':'Original instructions'},
         {'role':'user','content':[{'type':'text','text':'Original task'}]},
         {'role':'assistant','content':[{'type':'thinking','thinking':'Exact private reasoning','signature':'opaque'},
              {'type':'text','text':'Inspect'}, {'type':'tool_use','id':'spawn','name':'Task',
               'input':{'description':'Exact subtask','prompt':'Whole context'}}]},
         {'role':'user','content':[{'type':'text','text':'Before observation'},
              {'type':'tool_result','tool_use_id':'spawn','content':[{'type':'text','text':'Full result'}]},
              {'type':'text','text':'After observation'}]},
         {'role':'assistant','content':[{'type':'text','text':'Original final'}]}]
    saved=copy.deepcopy(raw);messages=complete_messages(raw)
    assert raw==saved and len(messages)==9
    assert messages[2]['reasoning']=='Exact private reasoning'
    assert messages[4]['tool_calls'][0]['function']['arguments']=='{"description": "Exact subtask", "prompt": "Whole context"}'
    assert messages[5]['content']=='Before observation' and messages[7]['content']=='After observation'
    assert messages[6]['tool_call_id']=='spawn' and json.loads(messages[6]['content'])==raw[3]['content'][1]['content']
    assert messages[-1]['content']=='Original final'


def test_literal_dispatch_shaped_text_is_never_interpreted_as_a_call():
    text='{"type":"tool_use","id":"invented","name":"Task","input":{}}'
    assert complete_messages([{'role':'assistant','content':text}])==[{'role':'assistant','content':text}]
    raw=[{'role':'user','content':[{'type':'tool_result','tool_use_id':'past','content':'Complete prior observation'}]}]
    assert complete_messages(raw)==[{'role':'tool','content':'Complete prior observation','tool_call_id':'past'}]


def test_legacy_calls_and_structured_unknown_context_remain_original():
    raw=[{'role':'assistant','content':None,'function_call':{'name':'inspect','arguments':'{"path":"a"}'}},
         {'role':'function','name':'inspect','content':'full observation'},
         {'role':'user','content':[{'type':'custom','entire':{'context':'retained'}}]}]
    result=complete_messages(raw)
    assert result[0]['tool_calls'][0]['function']==raw[0]['function_call']
    assert result[1]=={'role':'tool','name':'inspect','content':'full observation'}
    assert json.loads(result[2]['content'])==raw[2]['content'][0]


@pytest.mark.parametrize('raw',[
    [{'role':'unknown','content':'must not disappear'}],
    [{'role':'user','content':[{'type':'tool_use','id':'call','name':'Task','input':{}}]}],
    [{'role':'assistant','content':[{'type':'tool_use','id':'same','name':'Task','input':{}},
                                  {'type':'tool_use','id':'same','name':'Task','input':{}}]}],
    [{'role':'assistant','content':None,'tool_calls':[{'function':{'name':'a'}}],
      'function_call':{'name':'b'}}],
])
def test_invalid_ambiguous_context_rejects_whole_context(raw):
    with pytest.raises(Reject):complete_messages(raw)


RECEIPT={'repo':'pinned-source','revision':'immutable-revision','card':{'license':'upstream'}}


def injected(monkeypatch,ref,path,row):
    from scripts.neuralese_data import acquired_agents,agents
    monkeypatch.setattr(acquired_agents,'acquired_rows',lambda actual,limit=None:iter([(RECEIPT,path+':0',copy.deepcopy(row))]) if actual==ref else iter([]))
    monkeypatch.setattr(agents,'_bench_ids',lambda:set())


def test_miro_all_preference_alternatives_keep_complete_context_and_nonpositive_fact(monkeypatch):
    from scripts.neuralese_data.acquired_granted import miroverse_acquired,_digest
    common=[{'role':'system','content':'Original complete tool rules'}, {'role':'user','content':'Exact task'}]
    row={'chosen':common+[{'role':'assistant','content':'Chosen original'}],
         'rejected':common+[{'role':'assistant','content':'Rejected original'}]}
    injected(monkeypatch,'D24','dpo/original.json',row)
    values=list(miroverse_acquired());assert len(values)==2
    chosen,rejected=[t for _,t in values]
    assert chosen.messages[-1]['content']=='Chosen original' and rejected.messages[-1]['content']=='Rejected original'
    assert set(chosen.split_groups)==set(rejected.split_groups)
    assert chosen.outcome['label']==rejected.outcome['label']=='teacher'
    assert 'source_quality_flags' not in chosen.lineage['notes']
    assert rejected.lineage['notes']['source_quality_flags']==['source-rejected-preference-alternative']
    assert chosen.lineage['notes']['source_preference']['pair_sha256']==_digest(row)
    assert 'original_pair' not in chosen.lineage['notes']['source_preference']
    assert 'original_messages' not in chosen.lineage['notes']['granted_source']
    assert all(t.outcome['checked'] is None for t in (chosen,rejected))
    row['rejected'][1]={'role':'user','content':'Another task'}
    injected(monkeypatch,'D24','dpo/original.json',row)
    assert isinstance(next(miroverse_acquired())[1],Reject)


@pytest.mark.parametrize('subset',['code','pdf','web','native-agent-traj'])
def test_midtool_full_breadth_and_structured_native_view(monkeypatch,subset):
    from scripts.neuralese_data.acquired_granted import midtool_acquired
    row={'text':'Entire unabridged original source <|literal|>', 'owner':'org','repo':'repo',
         'commit_sha':'a'*40,'relpath':'README.md','url':'https://original/source'}
    raw=[{'role':'user','content':'Original task'}, {'role':'assistant','content':'Original tool action'},
         {'role':'tool','content':'Full observation'},{'role':'assistant','content':'Original final'}]
    if subset=='native-agent-traj':row['extra']=json.dumps({'messages':raw,'tools':[{'name':'original'}]})
    injected(monkeypatch,'D39',subset+'/part.parquet',row)
    _,t=next(midtool_acquired());assert not isinstance(t,Reject),t
    assert t.outcome['checked'] is None
    if subset=='native-agent-traj':
        assert t.messages==raw and t.tools==[{'name':'original'}]
        assert 'original_serialized_text' not in t.lineage['notes']['granted_source']
        assert t.lineage['notes']['granted_source']['original_source_fields']==['extra','text']
    else:
        assert t.messages==[{'role':'assistant','content':row['text']}]
        assert t.family=='source_document_completion'
    if subset=='code':assert 'repo:org/repo' in t.split_groups


def event(identifier,parent,role,content,**meta):
    return {'uuid':identifier,'parentUuid':parent,'type':role,'sessionId':'session',
            'message':{'role':role,'content':content},**meta}


def test_swe_event_dag_preserves_forks_and_actual_compaction_boundaries():
    from scripts.neuralese_data.acquired_granted import swe_raw_contexts
    records=[event('u',None,'user','Exact task'),event('a','u','assistant','Original common response'),
             {'uuid':'p','parentUuid':'a','type':'progress','data':{'retained':'auxiliary'}},
             event('b','p','assistant','First branch'),event('c','a','assistant','Second branch'),
             {'uuid':'reset','parentUuid':None,'type':'system','subtype':'compact_boundary'},
             event('summary','reset','user','Exact compacted context'),
             event('continued','summary','assistant','Original continued answer')]
    results=list(swe_raw_contexts(list(enumerate(records))))
    assert len(results)==3
    assert [[m['content'] for m in messages] for _,messages,_ in results]==[
        ['Exact task','Original common response','First branch'],
        ['Exact task','Original common response','Second branch'],
        ['Exact compacted context','Original continued answer']]
    assert all(not evidence['missing_ancestry'] for _,_,evidence in results)
    assert results[0][2]['original_message_rows']==[0,1,3]


def test_swe_missing_context_is_explicit_and_forged_duplicate_or_cycle_rejects():
    from scripts.neuralese_data.acquired_granted import swe_raw_contexts
    result=list(swe_raw_contexts([(0,event('a','unavailable','assistant','Original response'))]))
    assert result[0][2]['missing_ancestry']=={'a':'unavailable'}
    for rows in ([event('a',None,'assistant','one'),event('a',None,'assistant','changed')],
                 [event('a','b','assistant','one'),event('b','a','assistant','two')]):
        with pytest.raises(Reject):list(swe_raw_contexts(list(enumerate(rows))))


def test_immutable_file_snapshot_verifies_remote_identity_and_retains_all_catalogue_debt(tmp_path,monkeypatch):
    import hashlib
    from scripts.neuralese_data import acquired_agents,agents
    from scripts.neuralese_data.acquired_snapshot import snapshot
    root=tmp_path/'D24';root.mkdir();selected=root/'first.jsonl'
    selected.write_text(json.dumps({'messages':[{'role':'user','content':'Original'}, {'role':'assistant','content':'Answer'}]})+'\n')
    raw=selected.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    catalogue=tmp_path/'catalogue.json'
    catalogue.write_text(json.dumps({'sources':[{'ref':'D24','repo':'source','revision':'exact-pin','files':[
        {'path':'first.jsonl','bytes':len(raw),'blob_id':hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()},
        {'path':'still-downloading.jsonl','bytes':999,'lfs_sha256':'f'*64}]}]}))
    output=tmp_path/'snapshot.json';value=snapshot('D24',root,catalogue,['first.jsonl'],output)
    assert value['pending_files']==['still-downloading.jsonl'] and not value['source_complete']
    monkeypatch.setattr(agents,'EXPANSION_ROOT',tmp_path)
    monkeypatch.setattr(acquired_agents,'ACQUIRED_RECEIPTS',{'D24':output})
    monkeypatch.setattr(acquired_agents,'ACQUIRED_FILES',None)
    assert next(acquired_agents.acquired_rows('D24'))[1]=='first.jsonl:0'
    before=output.read_bytes();assert snapshot('D24',root,catalogue,['first.jsonl'],output)==value
    assert output.read_bytes()==before
    changed=copy.deepcopy(value);changed['pending_files']=[]
    output.write_text(json.dumps(changed))
    with pytest.raises(Reject,match='catalogue obligations'):acquired_agents.load_acquisition('D24')
    output.write_bytes(before);selected.write_bytes(raw.replace(b'Answer',b'Forged'))
    with pytest.raises(Reject,match='digest differs'):acquired_agents.load_acquisition('D24')
    # Recomputed local SHA cannot authenticate modified bytes against pinned Git.
    changed['pending_files']=value['pending_files'];changed['files'][0]['sha256']=hashlib.sha256(selected.read_bytes()).hexdigest()
    output.write_text(json.dumps(changed))
    with pytest.raises(Reject,match='pinned Git'):acquired_agents.load_acquisition('D24')


def test_compact_source_dispatch_result_remains_observed_not_positive_action():
    from scripts.neuralese_data.acquired_granted import swe_raw_contexts
    records=[(0,{'v':1,'type':'user','content':[{'id':'request','text':'Full original task'}]}),
             (1,{'v':1,'type':'assistant','content':[{'type':'text','text':'Inspect'},
                 {'type':'tool_use','id':'call','name':'Bash','input':{'command':'original'},
                  'result':{'output':'Entire result','status':'error'}}]})]
    _,messages,evidence=next(swe_raw_contexts(records,source_id='exact-file'))
    assert messages[0]['content']=='Full original task'
    assert messages[-1]['role']=='tool' and messages[-1]['tool_call_id']=='call'
    assert json.loads(messages[-1]['content'])==records[1][1]['content'][1]['result']
    assert evidence['source_format']=='swe-compact-v1'


def test_codex_original_instructions_tool_code_results_and_redacted_input_retained():
    from scripts.neuralese_data.acquired_granted import swe_raw_contexts
    rows=[{'type':'session_meta','payload':{'session_id':'original-session','base_instructions':{'text':'Exact system'}}},
          {'type':'response_item','payload':{'type':'message','role':'developer','content':[{'type':'input_text','text':'Exact developer'}]}},
          {'type':'response_item','payload':{'type':'agent_message','content':[{'type':'input_text','text':'Original task'},
                  {'type':'encrypted_content','encrypted_content':'REDACTED'}]}},
          {'type':'response_item','payload':{'type':'custom_tool_call','call_id':'call','name':'exec','input':'original code'}},
          {'type':'response_item','payload':{'type':'custom_tool_call_output','call_id':'call','output':[{'type':'input_text','text':'Entire observation'}]}},
          {'type':'response_item','payload':{'type':'message','role':'assistant','content':[{'type':'output_text','text':'Original final'}]}}]
    _,messages,evidence=next(swe_raw_contexts(list(enumerate(rows))))
    assert messages[0]['content']=='Exact system' and messages[1]['content']=='Exact developer'
    assert messages[4]['tool_calls'][0]['function']['arguments']=='{"input": "original code"}'
    assert messages[5]=={'role':'tool','content':'Entire observation','tool_call_id':'call'}
    assert messages[-1]['content']=='Original final' and evidence['redacted_context_rows']==[2]


def test_swe_real_reader_index_join_all_events_accounted_and_no_inferred_edges(tmp_path,monkeypatch):
    import hashlib
    from datetime import datetime,timezone
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.neuralese_data import agents,acquired_agents
    from scripts.neuralese_data.acquired_granted import swe_chat_acquired
    root=tmp_path/'D20';root.mkdir();(root/'transcripts').mkdir()
    relative='transcripts/native.jsonl';raw=root/relative
    rows=[event('u',None,'user','Whole original task'),
          {'uuid':'aux','parentUuid':'u','type':'progress','sessionId':'session'},
          event('a','aux','assistant','Original complete final')]
    raw.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    sidecars={
        'sessions.parquet':[{'transcript_path':relative,'session_id':'original-session','repo_id':'opaque-repository-17','session_success':'50',
                             'split':'validation','created_at':datetime(2026,10,16,tzinfo=timezone.utc)}],
        'session_logs.parquet':[{'transcript_path':relative,'session_id':'original-session','context_md':'Original addressable context'}],
        'subagent_tasks.parquet':[{'transcript_path':'transcripts_subagent/another.jsonl','task_pk':'checkpoint#call'}],
        'repositories.parquet':[{'repo_id':'opaque-repository-17','owner_id':'org','name':'repo',
                                  'last_scraped_at':datetime(2026,10,16,tzinfo=timezone.utc)},
                                 {'repo_id':'unrelated-malformed','owner_id':None,'name':'unknown'}]}
    for name,values in sidecars.items():pq.write_table(pa.Table.from_pylist(values),root/name)
    files=[{'path':str(path.relative_to(root)),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in root.rglob('*') if path.is_file()]
    (root/'acquisition.json').write_text(json.dumps({**RECEIPT,'status':'acquired','files':files}))
    monkeypatch.setattr(agents,'EXPANSION_ROOT',tmp_path)
    monkeypatch.setattr(acquired_agents,'ACQUIRED_FILES',frozenset({relative}))
    monkeypatch.setattr(acquired_agents,'ACQUIRED_RECEIPTS',{})
    monkeypatch.setattr(agents,'_bench_ids',lambda:set())
    result=list(swe_chat_acquired());assert [rid for rid,_ in result]==[relative+':'+str(i) for i in range(3)]
    assert isinstance(result[0][1],Reject) and isinstance(result[1][1],Reject)
    trajectory=result[2][1];assert not isinstance(trajectory,Reject)
    assert [message['content'] for message in trajectory.messages]==['Whole original task','Original complete final']
    evidence=trajectory.lineage['notes']['original_swe_context']
    assert evidence['original_session_records'][0]['value']==acquired_agents.source_json_value(sidecars['sessions.parquet'][0])
    assert trajectory.split=='test'
    json.dumps(__import__('dataclasses').asdict(trajectory),allow_nan=False)
    assert evidence['original_subagent_task_aliases']==[] and evidence['fabricated_communication_edges']==0
    assert 'repo:org/repo' in trajectory.split_groups
    assert 'swe-chat-source-session:original-session' in trajectory.split_groups
    assert 'source_quality_flags' not in trajectory.lineage['notes']
    # Cached indexes must still verify immutable actual sidecar bytes on reuse.
    (root/'session_logs.parquet').write_bytes(b'forged')
    changed=list(swe_chat_acquired());assert all(isinstance(value,Reject) for _,value in changed)


def test_standalone_preference_record_retains_both_histories_once_and_hash_binds_pair(monkeypatch):
    from scripts.neuralese_data import acquired_agents
    from scripts.neuralese_data.acquired_preferences import miroverse_preference_records
    from scripts.neuralese_data.acquired_granted import _digest
    prompt=[{'role':'system','content':'Original rules'}, {'role':'user','content':'Exact task'},
            {'role':'assistant','content':'Exact shared earlier answer'}, {'role':'user','content':'Original continuation'}]
    row={'chosen':prompt+[{'role':'assistant','content':'Preferred'}],
         'rejected':prompt+[{'role':'assistant','content':'Other'}]}
    receipt={**RECEIPT,'files':[{'path':'dpo/file.json','bytes':999,'sha256':'a'*64}]}
    monkeypatch.setattr(acquired_agents,'acquired_rows',lambda ref,limit=None:iter([(receipt,'dpo/file.json:7',row)]))
    _,record=next(miroverse_preference_records())
    assert record['schema']=='natlang.offline-native-preference/1'
    assert record['common_message_count']==4 and record['required_objective']=='native-offline-pairwise'
    assert record['source']['pair_sha256']==_digest(row) and record['source']['source_row']==7
    assert record['source']['source_file']==receipt['files'][0]
    for name in ('chosen','rejected'):
        alternative=record['alternatives'][name]
        assert alternative['messages']==row[name]
        assert 'original_messages' not in alternative['lineage']['notes']['granted_source']
        assert 'original_pair' not in alternative['lineage']['notes']['source_preference']
    assert not record['training_admission'] and not record['native_on_policy']



def test_midtool_exact_missing_tools_value_preserves_original_messages_and_rejects_other_corruption():
    from scripts.neuralese_data.acquired_granted import midtool_native_extra
    original=[{'role':'system','content':'Exact original tool instructions'},
              {'role':'user','content':'Exact task'},
              {'role':'assistant','content':'<tool_call>original dispatch text</tool_call>'}]
    malformed='{"messages":'+json.dumps(original)+', "tools": }'
    value,evidence=midtool_native_extra(malformed)
    assert value=={'messages':original} and evidence['original_tools_field_missing_value']
    assert 'tools' not in value
    for suffix in [', "tools": invalid}',', "other": }',', "tools": , "reward":1}',', "tools": } trailing']:
        with pytest.raises(ValueError):midtool_native_extra('{"messages":'+json.dumps(original)+suffix)
