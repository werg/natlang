"""Complete failed logs are evidence, never positive failed-action targets."""
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from neuralese_data import acquired_agents as acquired,acquired_diagnosis as diagnosis
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
