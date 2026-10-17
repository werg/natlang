"""Complete contexts and exact channel/call provenance, without success invention."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from neuralese_data import acquired_agents as acquired,agents
from neuralese_data.common import Reject

RECEIPT={'repo':'pinned-source','revision':'immutable-revision','card':{'license':'upstream'}}


def rows(monkeypatch,ref,row,path):
    monkeypatch.setattr(acquired,'acquired_rows',lambda actual,limit=None:iter([(RECEIPT,path+':0',copy.deepcopy(row))]) if actual==ref else iter([]))
    monkeypatch.setattr(agents,'_bench_ids',lambda:set())


def conversation(answer):
    return [{'role':'system','content':'Original instructions'}, {'role':'user','content':'Original task'},
            {'role':'assistant','content':answer}]


def test_cooperation_preserves_all_compaction_contexts_and_shared_task(monkeypatch):
    row={'info':{'config':{'model':{'model_name':'source-model'}},'exit_status':'LimitsExceeded'},
        'messages':conversation('Final distinct context'),
        'segments':[{'kind':'solver','messages':conversation('First unabridged solver')},
                    {'kind':'summarizer','messages':conversation('Original compaction')},
                    {'kind':'solver','messages':conversation('Continued solver')}]}
    saved=copy.deepcopy(row);rows(monkeypatch,'D14',row,'coop/repo_task/17/pair/agent1_full_traj.json')
    result=list(acquired.qwen_cooperation_acquired())
    assert len(result)==4 and all(not isinstance(t,Reject) for _,t in result)
    assert [t.messages[-1]['content'] for _,t in result]==['First unabridged solver','Original compaction','Continued solver','Final distinct context']
    assert len({tuple(t.split_groups) for _,t in result})==1
    assert all(t.outcome['label']=='failed' and t.outcome['checked'] is None for _,t in result)
    assert all(t.lineage['teacher']['model']=='source-model' for _,t in result) and row==saved
    row['messages'].append({'role':'exit','content':'','extra':{'exit_status':'LimitsExceeded'}})
    rows(monkeypatch,'D14',row,'coop/repo_task/17/pair/agent1_full_traj.json')
    _,final=list(acquired.qwen_cooperation_acquired())[-1]
    receipt=final.lineage['notes']['original_cooperation_context']
    assert receipt['original_message_count']==4 and receipt['model_message_count']==3
    assert receipt['terminal_events'][0]['event']==row['messages'][-1]
    row['messages']=row['segments'][-1]['messages'];rows(monkeypatch,'D14',row,'coop/repo_task/17/pair/agent1_full_traj.json')
    assert len(list(acquired.qwen_cooperation_acquired()))==3,'only byte-equivalent final context is redundant'


def test_d14_bounded_projection_preserves_all_contexts_and_binds_raw_source(monkeypatch,tmp_path):
    segments=[]
    total=0
    escaped_content='line\nnext\ttab \\ slash \\" quoted {} braces ☃'
    for segment_index in range(11):
        messages=[{'role':'system','content':'Original instructions'},
                  {'role':'user','content':f'Original task {segment_index}'}]
        for message_index in range(6):
            if segment_index==4 and message_index==2:
                messages.append({'role':'tool','name':'inspect','content':'Exact complete observation',
                                 'extra':'x'*(2*1024*1024)+escaped_content,
                                 'content_parts':[{'role':'tool','extra':'semantic payload field',
                                                   'content':'preserve nested tool payload'}]})
            else:
                content=(escaped_content if segment_index==0 and message_index==0
                         else f'Original response {segment_index}-{message_index}')
                messages.append({'role':'assistant','content':content,
                                 'extra':{'ignored_by_normalizer':True}})
        segments.append({'kind':f'context-{segment_index}','messages':messages})
        total+=len(messages)
    final=[{'role':'user','content':'Distinct original final task'},
           {'role':'assistant','content':'Distinct original final response'}]
    total+=len(final)
    assert total==90
    row={'info':{'config':{'model':{'model_name':'pinned-source-teacher'}},'exit_status':'Completed'},
         'segments':segments,'messages':final}
    raw=json.dumps(row,separators=(',',':'),ensure_ascii=False).encode()
    source=tmp_path/'agent7_full_traj.json';source.write_bytes(raw)
    entry={'path':source.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    index,projected=next(acquired._iter_d14_full_trajectory_json(source,entry))
    assert index==0 and len(projected['segments'])==11
    assert sum(len(segment['messages']) for segment in projected['segments'])+len(projected['messages'])==90
    tool=projected['segments'][4]['messages'][4]
    assert tool=={'role':'tool','name':'inspect','content':'Exact complete observation',
                  'content_parts':[{'role':'tool','extra':'semantic payload field',
                                    'content':'preserve nested tool payload'}]}
    assert projected['segments'][0]['messages'][2]['extra']=={'ignored_by_normalizer':True}
    assert projected['segments'][0]['messages'][2]['content']==escaped_content
    proof=projected['_natlang_source_projection']
    assert proof['source_file_sha256']==entry['sha256'] and proof['source_file_bytes']==len(raw)
    assert proof['omitted_message_extra_fields']==1 and proof['omitted_raw_value_bytes']>2*1024*1024
    assert 'not independently decoded' in proof['discarded_value_validation']
    with pytest.raises(Reject,match='acquisition binding'):
        next(acquired._iter_d14_full_trajectory_json(source,{**entry,'sha256':'0'*64}))
    rows(monkeypatch,'D14',projected,'coop/repo_task/17/pair/agent7_full_traj.json')
    result=list(acquired.qwen_cooperation_acquired())
    assert len(result)==12 and all(not isinstance(trajectory,Reject) for _,trajectory in result)
    assert [trajectory.lineage['notes']['original_cooperation_context']['kind'] for _,trajectory in result[:11]]==[
        f'context-{index}' for index in range(11)]
    assert all(trajectory.lineage['notes']['source_projection']==proof for _,trajectory in result)
    assert result[-1][1].messages[-1]['content']=='Distinct original final response'


def block(text):return [{'type':'text','text':text,'tools':None}]


def research():
    return {'qid':12,'chunk_idx':1,'num_chunks':2,'status':'success','messages':[
        {'role':'system','content':[{'type':'system_content','model_identity':'Original generic system identity',
            'tools':{'browser':{'tools':[{'name':'search','parameters':{'type':'object'}}]}}}]},
        {'role':'developer','content':block('Original tool instructions')},
        {'role':'user','content':block('Original research question')},
        {'role':'assistant','channel':'analysis','content':block('Exact reasoning')},
        {'role':'assistant','channel':'analysis','content_type':'code','recipient':'browser.search','content':block('{"query":"exact"}')},
        {'role':'tool','channel':'analysis','recipient':'assistant','content':block('Exact complete observation')},
        {'role':'assistant','channel':'final','content':block('Exact final answer')}]}


def test_research_blocks_reasoning_calls_observations_and_sibling_groups(monkeypatch):
    row=research();saved=copy.deepcopy(row);rows(monkeypatch,'D32',row,'seed_42/train.parquet')
    _,t=next(acquired.openresearcher_structured_acquired())
    assert not isinstance(t,Reject),t
    assert len(t.messages)==7 and t.messages[3]['reasoning']=='Exact reasoning'
    call=t.messages[4]['tool_calls'][0]
    assert call['function']=={'name':'browser.search','arguments':'{"query":"exact"}'}
    assert t.messages[5]['tool_call_id']==call['id'] and t.messages[5]['content']=='Exact complete observation'
    assert 'Original generic system identity' in t.messages[0]['content']
    assert t.messages[1]['content']=='Original tool instructions' and t.messages[-1]['content']=='Exact final answer'
    assert t.tools[0]['function']['name']=='browser.search'
    assert t.lineage['teacher']=='unknown','generic system identity is not an actual teacher model identifier'
    assert t.outcome['checked'] is None and row==saved
    row.update(chunk_idx=0,status='failed');rows(monkeypatch,'D32',row,'seed_32/train.parquet')
    _,sibling=next(acquired.openresearcher_structured_acquired())
    assert sibling.id!=t.id and sibling.split_groups==t.split_groups and sibling.outcome['label']=='failed'
    row['messages'].insert(3,{'role':'tool','content':block('unbound observation')})
    rows(monkeypatch,'D32',row,'seed_42/train.parquet')
    assert isinstance(next(acquired.openresearcher_structured_acquired())[1],Reject)


def test_shared_openai_normalization_never_silently_drops_structured_observation():
    original={'role':'tool','content':[{'type':'text','text':'Entire observation'},{'type':'custom','evidence':17}]}
    result=agents._openai_messages([original])[0]
    assert json.loads(result['content'])==original['content']


def collusion():
    messages=[{'role':'system','content':'Original game'}, {'role':'user','content':'Current exact task'},
        {'role':'assistant','content':'answer'}, {'role':'user','content':'Original feedback'},
        {'role':'assistant','content':'reflection'}]
    episode={'episode_id':'episode-1','task_ids':['task-a','task-b'],'agents':{
        'alice':{'messages':messages,'own_task_prompt':'Current exact task','true_expected_verdict':'pass','verdict_correct':True,'reward':10}},
        'llm_usage':[{'actor':'alice','phase':'task','round':None,'agent_attempt':1,'response_status':'accepted',
                      'visible_content_chars':6,'tool_call_count':0},
                     {'actor':'alice','phase':'private_outcome_reflection','round':None,'agent_attempt':None,'response_status':'accepted',
                      'visible_content_chars':10,'tool_call_count':0}],
        'llm_reasoning_traces':[{'agent_id':'alice','phase':'task','round':None,'attempt':0,
            'reasoning_trace':{'reasoning_content':'Actual private reasoning','provider_reasoning':'Actual private reasoning'}}]}
    return {'schema_version':'1.0.0','run_id':'run','run_config':{'models':{'alice':'original-teacher'}},'results':[episode]}


def test_collusion_uses_original_full_history_and_exact_sparse_reasoning_evidence(monkeypatch):
    row=collusion();saved=copy.deepcopy(row);rows(monkeypatch,'D19',row,'traces/condition/rep.json.gz')
    _,t=next(acquired.agent_collusion_raw_acquired())
    assert not isinstance(t,Reject),t
    assert len(t.messages)==5 and t.messages[2]['reasoning']=='Actual private reasoning'
    assert t.messages[-1]['content']=='reflection' and t.outcome['label']=='teacher' and t.outcome['checked'] is None
    alignment=t.lineage['notes']['original_collusion_session']['reasoning_alignment']
    assert alignment['complete'] and alignment['bindings'][0]['message_index']==2
    assert row==saved
    row['results'][0]['agents']['alice']['verdict_correct']=False
    row['results'][0]['llm_usage'][0]['visible_content_chars']=999
    rows(monkeypatch,'D19',row,'traces/condition/rep.json.gz')
    _,failed=next(acquired.agent_collusion_raw_acquired())
    assert failed.outcome['label']=='failed'
    assert 'reasoning' not in failed.messages[2],'forged response-shape evidence must not assign reasoning to a guessed target'
    assert failed.lineage['notes']['original_collusion_session']['reasoning_alignment']['complete'] is False
    assert failed.lineage['notes']['original_collusion_session']['llm_reasoning_traces']
    rows(monkeypatch,'D19',row,'data/agents/reordered-summary.parquet')
    assert isinstance(next(acquired.agent_collusion_raw_acquired())[1],Reject)


def test_malformed_context_is_explicit_and_other_original_contexts_survive(monkeypatch):
    row={'info':{},'messages':conversation('Final'),
         'segments':[None,{'kind':'solver','messages':conversation('Good segment')}]}
    rows(monkeypatch,'D14',row,'coop/repo/17/pair/agent1_full_traj.json')
    result=list(acquired.qwen_cooperation_acquired())
    assert isinstance(result[0][1],Reject) and [t.messages[-1]['content'] for _,t in result[1:]]==['Good segment','Final']
    row=collusion();row['results'].insert(0,None)
    rows(monkeypatch,'D19',row,'traces/condition/rep.json.gz')
    result=list(acquired.agent_collusion_raw_acquired())
    assert isinstance(result[0][1],Reject) and result[1][1].messages[-1]['content']=='reflection'


def test_legacy_function_calls_and_named_feedback_preserve_source_bytes():
    raw=[{'role':'assistant','content':None,'function_call':{'name':'inspect','arguments':'{"file":"a"}'}},
         {'role':'function','name':'inspect','content':'{"name":"observed","arguments":"this is feedback"}'}]
    for normalized in (agents._openai_messages(raw),agents.expansion_messages(raw)):
        assert len(normalized)==2
        assert normalized[0]['tool_calls'][0]['function']==raw[0]['function_call']
        assert normalized[1]['role']=='tool' and normalized[1]['name']=='inspect'
        assert normalized[1]['content']==raw[1]['content']
    with pytest.raises(Reject,match='conflicting'):
        agents._openai_messages([{**raw[0],'tool_calls':[{'function':{'name':'other'}}]}])


def test_correct_negative_verdict_is_not_itself_an_actor_failure(monkeypatch):
    row=collusion();actor=row['results'][0]['agents']['alice']
    actor.update(true_expected_verdict='fail',verdict_correct=True,reward=10)
    rows(monkeypatch,'D19',row,'traces/condition/rep.json.gz')
    _,t=next(acquired.agent_collusion_raw_acquired())
    assert t.outcome['label']=='teacher' and t.outcome['details']['reported_true_expected_verdict']=='fail'
    actor['extraction_eval']={'quality_verdict':'fail'}
    rows(monkeypatch,'D19',row,'traces/condition/rep.json.gz')
    assert next(acquired.agent_collusion_raw_acquired())[1].outcome['label']=='failed'
