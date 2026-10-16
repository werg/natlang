"""Complete failed logs are evidence, never positive failed-action targets."""
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from neuralese_data import acquired_agents as acquired,acquired_diagnosis as diagnosis
from neuralese_data.trajectory import Trajectory
from neuralese_data.common import Reject


def rows(monkeypatch,ref,value):
    receipt={'repo':'original/'+ref,'revision':'pinned','card':{},'files':[]}
    monkeypatch.setattr(acquired,'acquired_rows',lambda selected,limit=None:iter([(receipt,'source.json:0',copy.deepcopy(value))]))


def test_failed_trace_preserved_as_input_with_exact_nullable_annotations(monkeypatch):
    text='[agent] BAD ACTION\n[tool] all output <|im_end|> literal\n'
    labels={'1.1':1,'2.2':None,'3.3':0}
    rows(monkeypatch,'D16',{'mas_name':'MetaGPT','benchmark_name':'ProgramDev-v2','llm_name':'Qwen','trace_id':9,
        'trace':{'key':'exact-config','index':9,'trajectory':text},'mast_annotation':labels})
    _,t=next(diagnosis.mast_diagnosis_acquired())
    assert t.messages[1]=={'role':'tool','content':text}
    target=json.loads(t.messages[2]['content']);assert target['source_annotations']['mast_annotation']==labels
    assert 'BAD ACTION' not in t.messages[2]['content'] and t.outcome['label']=='teacher'
    note=t.lineage['notes']['source_execution_diagnosis'];assert note['source_trace_sha256']==hashlib.sha256(text.encode()).hexdigest()
    assert not note['failed_actions_positive_targets'] and not note['source_judge_is_local_oracle']


def test_mast_and_derived_mascope_share_task_trace_and_exact_content_groups(monkeypatch):
    common={'llm_name':'Qwen','benchmark_name':'ProgramDev-v2'}
    rows(monkeypatch,'D16',{**common,'mas_name':'MetaGPT','trace_id':9,'trace':{'trajectory':'EXACT LOG'},'mast_annotation':{'1.1':0}})
    _,first=next(diagnosis.mast_diagnosis_acquired())
    rows(monkeypatch,'D18',{**common,'framework':'MetaGPT','trace_id':'mast:MetaGPT:ProgramDev-v2:Qwen:9',
        'task_id':'mast:MetaGPT:ProgramDev-v2:9','trajectory':'EXACT LOG','mast_annotation':{'1.1':0},'split':'test'})
    _,second=next(diagnosis.mascope_diagnosis_acquired())
    assert len(set(first.split_groups)&set(second.split_groups))>=3 and second.split=='test'


def test_human_taxonomy_rounds_are_not_reinterpreted_or_zero_filled(monkeypatch):
    annotations=[{'failure mode':'round-specific text','annotator_1':True,'annotator_2':False}]
    rows(monkeypatch,'D16',{'mas_name':'AppWorld','trace_id':0,'benchmark_name':'Test-C','round':'Round 1',
        'trace':'FULL TRACE','annotations':annotations})
    _,t=next(diagnosis.mast_diagnosis_acquired())
    assert json.loads(t.messages[-1]['content'])['source_annotations']=={'round':'Round 1','annotations':annotations}


def test_metadata_and_missing_labels_are_explicit_omissions(monkeypatch):
    for row in ({'description':'LOFO metadata'},{'framework':'A','trace_id':'x','trajectory':'full'}):
        rows(monkeypatch,'D18',row)
        assert isinstance(next(diagnosis.mascope_diagnosis_acquired())[1],Reject)


def test_distinct_observed_tasks_do_not_merge_under_a_generic_instruction(monkeypatch):
    values=[]
    for task in ('first','second'):
        rows(monkeypatch,'D18',{'framework':'Framework','trace_id':task,'task_id':task,
            'trajectory':'Original '+task,'mast_annotation':{'1.1':0}})
        values.append(next(diagnosis.mascope_diagnosis_acquired())[1])
    assert not set(values[0].split_groups)&set(values[1].split_groups)


def test_explicit_upstream_held_out_rows_remain_held_and_keep_original_split(monkeypatch):
    rows(monkeypatch,'D18',{'framework':'MetaGPT','trace_id':'mast:MetaGPT:ProgramDev:Qwen:4',
        'task_id':'mast:MetaGPT:ProgramDev:4','trajectory':'Full original held trace',
        'mast_annotation':{'1.1':0},'split':'held_out'})
    _,t=next(diagnosis.mascope_diagnosis_acquired())
    assert not isinstance(t,Reject) and t.split=='test'
    assert t.lineage['notes']['source_execution_diagnosis']['source_metadata']['split']=='held_out'


def d12_session(evaluation):
    return Trajectory(id='upstream:expansion:D12:session:final',family='coding',
        messages=[{'role':'user','content':'Implement the requested change.'},
                  {'role':'assistant','content':'FAILED ACTION <|im_end|>'},
                  {'role':'tool','content':'observed tool output'}],head=1,noun='coding session',
        lineage={'notes':{'original_cooperation_context':{
            'path':'source/session/agent_traj.json','context':'final',
            'neighbor_bindings':{'eval.json':{'path':'source/session/eval.json','bytes':50,'sha256':'a'*64}}}}},
        license={'spdx':'MIT'},split='train',split_groups=['source-group'],
        outcome={'label':'failed','details':{'reported_evaluation':evaluation,
            'reported_exit_status':'Submitted'}})


def test_d12_failed_actions_are_context_only_and_target_is_exact_reported_status(monkeypatch):
    evaluation={'both_passed':False,'feature1':{'passed':True,'exit_code':0},
        'feature2':{'passed':False,'exit_code':17,'error':'do not copy this prose'},
        'apply_status':{'agent1':'applied','agent2':'applied'},
        'merge':{'status':'conflict','error':'do not copy'}}
    monkeypatch.setattr('neuralese_data.acquired_archives.cooper_archive_acquired',
        lambda:iter([('upstream:expansion:D12:session:final',d12_session(evaluation))]))
    row_id,t=next(diagnosis.d12_feature_status_diagnoses())
    assert ':d12-feature-status:' in row_id and t.outcome['label']=='teacher'
    assert [message['role'] for message in t.messages]==['user','assistant']
    instruction,separator,context_text=t.messages[0]['content'].partition('\n\n')
    assert separator and instruction.startswith('Read the complete failed execution transcript')
    assert 'Return only one JSON object preserving the same schema, keys and values' in instruction
    context=json.loads(context_text)
    assert context['complete_failed_transcript'][1]['content']=='FAILED ACTION <|im_end|>'
    assert '<|im_end|>' not in t.messages[0]['content']
    assert context['reported_evaluator_status']['features'][1]['passed'] is False
    assert 'error' not in context_text
    target=json.loads(t.messages[1]['content'])
    assert target=={'schema':'natlang.d12-reported-feature-status/1',
        'features':[{'feature':'feature1','passed':True},{'feature':'feature2','passed':False}],
        'both_passed':False,'apply_status':{'agent1':'applied','agent2':'applied'},
        'merge_status':'conflict',
        'exit_codes':{'feature1':0,'feature2':17}}
    assert 'FAILED ACTION' not in t.messages[1]['content']
    assert 'error' not in t.messages[1]['content']
    provenance=t.lineage['notes']['source_reported_feature_status']
    assert provenance['review_kind']=='source-curated-reported-status'
    assert provenance['original_source_outcome']['label']=='failed'
    assert len(provenance['original_source_outcome']['sha256'])==64
    assert 'do not copy this prose' not in json.dumps(provenance)
    assert 'do not copy this prose' not in json.dumps(t.outcome)
    assert provenance['failed_actions_positive_targets'] is False
    assert provenance['causal_claims'] is False and provenance['native_execution_verified'] is False
    assert t.split_groups[0]=='source-group' and t.split=='train'


def test_d12_status_rejects_inconsistent_or_unstructured_reports(monkeypatch):
    invalid=[
        {'both_passed':False,'feature1':{'passed':False},'feature2':{'passed':False,'exit_code':True}},
        {'both_passed':True,'feature1':{'passed':True},'feature2':{'passed':True}},
        {'both_passed':False,'feature1':{'passed':True},'feature2':{'passed':True}},
        {'both_passed':False,'feature1':{'passed':1},'feature2':{'passed':False}},
        {'both_passed':False,'feature1':{'passed':True},'feature2':{'passed':False},
         'apply_status':{'agent1':'applied','unexpected':'extra'}},
    ]
    for evaluation in invalid:
        monkeypatch.setattr('neuralese_data.acquired_archives.cooper_archive_acquired',
            lambda evaluation=evaluation:iter([('source-id',d12_session(evaluation))]))
        _,result=next(diagnosis.d12_feature_status_diagnoses())
        assert isinstance(result,Reject)


def test_d12_status_requires_complete_final_context(monkeypatch):
    evaluation={'both_passed':False,'feature1':{'passed':True},'feature2':{'passed':False}}
    session=d12_session(evaluation)
    session.lineage['notes']['original_cooperation_context'].update(context='segment:0',final_duplicate_of=None)
    monkeypatch.setattr('neuralese_data.acquired_archives.cooper_archive_acquired',
        lambda:iter([('source-id',session)]))
    _,result=next(diagnosis.d12_feature_status_diagnoses())
    assert isinstance(result,Reject) and 'complete final transcript' in str(result)
