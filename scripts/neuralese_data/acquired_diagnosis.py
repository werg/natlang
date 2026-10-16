"""Exact source-label supervision over complete observed multi-agent logs.

The source annotations are existing teacher data, not a local task oracle or
permission to imitate failed agent actions. Raw logs remain observed input.
"""
import hashlib
import json

from .common import Reject
from .records import group_key


def _trace_and_annotations(ref,row):
    if ref=='D16':
        trace=row.get('trace')
        text=trace.get('trajectory') if isinstance(trace,dict) else trace
        if 'mast_annotation' in row:annotations={'mast_annotation':row['mast_annotation']}
        elif 'annotations' in row:annotations={'round':row.get('round'),'annotations':row['annotations']}
        else:raise Reject('source diagnosis lacks original annotation inventory')
        return text,annotations
    return row.get('trajectory'),{key:row[key] for key in
        ('mast_annotation','anomaly_type','anomaly_injected','note') if key in row}


def source_diagnoses(ref,limit=None):
    from . import agents
    from .acquired_agents import acquired_rows,_reported_session,_bound_acquired_path
    previous=None;split_document=None;split_binding=None
    for receipt,rid,row in acquired_rows(ref,limit):
        try:
            if receipt is not previous:
                previous=receipt;split_document=split_binding=None
                entry=next((e for e in receipt['files'] if e['path']=='lofo_splits.json'),None)
                if entry is not None:
                    path=_bound_acquired_path(agents.EXPANSION_ROOT/ref,entry)
                    split_document=json.loads(path.read_text());split_binding=entry
            if not isinstance(row,dict) or ('trace' not in row and 'trajectory' not in row):
                raise Reject('non-trajectory source metadata; preserved in pinned acquisition')
            text,annotations=_trace_and_annotations(ref,row)
            if not isinstance(text,str) or not text:raise Reject('diagnosis source lacks complete observed trace text')
            if not annotations:raise Reject('diagnosis source lacks original labels')
            if row.get('split') not in (None,'train','test','validation','eval','held_out'):
                raise Reject('unknown original diagnosis split')
            framework=row.get('mas_name') or row.get('framework');benchmark=row.get('benchmark_name')
            trace_id=str(row.get('trace_id')) if row.get('trace_id') is not None else None
            if not framework or trace_id is None:raise Reject('diagnosis source lacks framework/trace identity')
            llm=row.get('llm_name');origin=str(trace_id)
            if ref=='D18' and trace_id.startswith('mast:'):origin=trace_id.removeprefix('mast:')
            elif ref=='D16':origin=f'{framework}:{benchmark}:{llm}:{trace_id}'
            task=row.get('task_id') or f'{framework}:{benchmark}:{trace_id}'
            if isinstance(task,str) and task.startswith('mast:'):task=task.removeprefix('mast:')
            messages=[{'role':'user','content':f'Read the complete observed execution trace for {framework} task {task} and return its source-reported failure annotations.'},
                      {'role':'tool','content':text},
                      {'role':'assistant','content':json.dumps({'schema':'natlang.source-execution-diagnosis/1',
                          'source_annotations':annotations},ensure_ascii=False,separators=(',',':'))}]
            t=_reported_session(ref,receipt,rid,{**row,'split':'test' if row.get('split')=='held_out' else row.get('split')},messages,task_id=str(task),session_id=rid,teacher=None)
            # Generic conversational cleanup must not delete embedded control
            # spellings or parse source log bodies as live dispatches.
            t.messages=messages;t.family='execution_failure_diagnosis';t.noun='observed execution trace'
            digest=hashlib.sha256(text.encode()).hexdigest()
            t.split_groups.extend([group_key('mast-trace',origin),group_key('mast-task',str(task)),
                                   group_key('observed-trace-sha256',digest)])
            metadata={k:v for k,v in row.items() if k not in ('trajectory','trace')}
            trace=row.get('trace')
            if isinstance(trace,dict):metadata['trace']={k:v for k,v in trace.items() if k!='trajectory'}
            t.lineage['notes']['source_execution_diagnosis']={'source_metadata':metadata,
                'source_trace_sha256':digest,'complete_trace':True,'source_labels_unchanged':True,
                'failed_actions_positive_targets':False,'source_judge_is_local_oracle':False,
                'lofo_splits':split_document,'lofo_binding':split_binding}
            t.outcome['details'].update(source_execution_outcome='not asserted',
                supervised_task='exact source-reported diagnosis',executed_locally=False)
            yield rid,t
        except (Reject,ValueError,TypeError,KeyError) as error:yield rid,Reject(str(error))


def mast_diagnosis_acquired(limit=None):
    yield from source_diagnoses('D16',limit)


def mascope_diagnosis_acquired(limit=None):
    yield from source_diagnoses('D18',limit)
