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


def d12_feature_status_diagnoses(limit=None):
    """Derive only explicit D12 evaluator statuses from failed sessions.

    Failed assistant turns remain in the complete user-provided context, never
    targets. The user message is a normal chat message, not an orphan tool reply.
    This is source-curated status supervision, not a cause/fix diagnosis.
    """
    from copy import copy, deepcopy
    from .acquired_archives import cooper_archive_acquired

    emitted=0
    for source_id,original in cooper_archive_acquired():
        if limit is not None and emitted>=limit:
            return
        if isinstance(original,Reject):
            continue
        try:
            if original.outcome.get('label')!='failed':
                raise Reject('D12 status view requires an originally failed session')
            details=original.outcome.get('details')
            evaluation=details.get('reported_evaluation') if isinstance(details,dict) else None
            if not isinstance(evaluation,dict):
                raise Reject('D12 source lacks structured reported evaluation')
            features=[]
            for key in ('feature1','feature2'):
                feature=evaluation.get(key)
                if not isinstance(feature,dict) or type(feature.get('passed')) is not bool:
                    raise Reject('D12 source feature status is missing an exact boolean')
                features.append({'feature':key,'passed':feature['passed']})
            both=evaluation.get('both_passed')
            if type(both) is not bool or both != all(row['passed'] for row in features):
                raise Reject('D12 aggregate status does not agree with feature statuses')
            if both:
                raise Reject('D12 source is successful, not a failed-status example')

            reported={'schema':'natlang.d12-reported-feature-status/1',
                      'features':features,'both_passed':both}
            if 'apply_status' in evaluation:
                apply_status=evaluation['apply_status']
                if (not isinstance(apply_status,dict) or set(apply_status)!={'agent1','agent2'}
                        or any(not isinstance(value,str) for value in apply_status.values())):
                    raise Reject('D12 apply status is not the exact two-agent structured record')
                reported['apply_status']={key:apply_status[key] for key in ('agent1','agent2')}
            merge=evaluation.get('merge')
            if isinstance(merge,dict) and 'status' in merge:
                if not isinstance(merge['status'],str):
                    raise Reject('D12 merge status is not structured text')
                reported['merge_status']=merge['status']
            for key in ('feature1','feature2'):
                code=(evaluation.get(key) or {}).get('exit_code')
                if code is not None:
                    if type(code) is not int:
                        raise Reject('D12 feature exit code is not an integer')
                    reported.setdefault('exit_codes',{})[key]=code

            original_note=original.lineage.get('notes',{}).get('original_cooperation_context',{})
            if not isinstance(original_note,dict):
                raise Reject('D12 status view lacks original source binding')
            if (original_note.get('context')!='final'
                    and original_note.get('final_duplicate_of')!=original_note.get('context')):
                raise Reject('D12 status view requires the complete final transcript, not a partial segment')
            source_binding={'original_id':source_id,
                'source_path':original_note.get('path'),
                'source_context':original_note.get('context'),
                'neighbor_bindings':original_note.get('neighbor_bindings'),
                'evaluation_sha256':hashlib.sha256(json.dumps(evaluation,ensure_ascii=False,
                    sort_keys=True,separators=(',',':')).encode()).hexdigest()}
            if not source_binding['source_path'] or not source_binding['neighbor_bindings']:
                raise Reject('D12 status view lacks exact archive/evaluation member receipts')
            context=json.dumps({'complete_failed_transcript':original.messages,
                'reported_evaluator_status':reported},ensure_ascii=False,separators=(',',':'))
            # Keep literal model control spellings as data inside the context.
            context=context.replace('<','\\u003c')
            target=json.dumps(reported,ensure_ascii=False,separators=(',',':'))
            instruction=('Read the complete failed execution transcript and its source-reported '
                'evaluator status record. Copy only the exact structured feature statuses and '
                'recorded status fields from that record; do not infer a cause or repair.')
            derived=copy(original)
            derived.lineage=deepcopy(original.lineage)
            view_digest=hashlib.sha256((source_id+'\0'+source_binding['evaluation_sha256']).encode()).hexdigest()
            derived.id=f'{source_id}:d12-feature-status:{view_digest[:20]}'
            derived.family='source_reported_feature_status'
            derived.noun='source-reported feature status'
            derived.messages=[
                {'role':'user','content':instruction+'\n\n'+context},
                {'role':'assistant','content':target}]
            derived.split_groups=list(dict.fromkeys([*derived.split_groups,
                group_key('d12-feature-status-view',view_digest)]))
            note=derived.lineage.setdefault('notes',{})
            note['source_reported_feature_status']={**source_binding,
                'view_id_sha256':view_digest,'review_kind':'source-curated-reported-status',
                'original_source_outcome':{'label':'failed',
                    'sha256':hashlib.sha256(json.dumps(original.outcome,ensure_ascii=False,
                        sort_keys=True,separators=(',',':')).encode()).hexdigest()},
                'complete_failed_transcript_is_context':True,'failed_actions_positive_targets':False,
                'causal_claims':False,'repair_claims':False,'source_judge_is_local_oracle':False,
                'native_execution_verified':False,'executed_locally':False,
                'target_fields':sorted(reported.keys())}
            derived.outcome={'label':'teacher','details':{
                'supervised_task':'exact source-reported D12 feature status',
                'source_curated':True,'executed_locally':False,
                'source_execution_outcome':'failed',
                'original_source_outcome_sha256':note['source_reported_feature_status']['original_source_outcome']['sha256'],
                'reported_evaluation_sha256':source_binding['evaluation_sha256']}}
            emitted+=1
            yield derived.id,derived
        except (Reject,ValueError,TypeError,KeyError) as error:
            yield source_id,Reject(str(error))
