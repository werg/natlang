"""Mandatory CE-stage credits carried through shared objectives.

These are schedule metadata, never another optimizer/RNG history. The live
participant owns one current model/optimizer/FP8/RNG continuation stream.
"""
from __future__ import annotations
import copy
import math
import json
from pathlib import Path
from ..common.hashing import sha256_file_hex_cached

TRAINING_SCHEMA='natlang.neuralese-native-sft-training/1'
PROGRESS_SCHEMA='natlang.native-sft-progress/1'
_TRANSIENT={'optimizer','rng','fp8','sft_progress'}


def required_visits(order,epochs):
    if not math.isfinite(epochs) or epochs<=0:
        raise ValueError('native exposure epochs must be finite and positive')
    whole=int(epochs);fraction=math.ceil((epochs-whole)*len(order))
    return {key:whole+int(index<fraction) for index,key in enumerate(order)}


def stage_key(training):
    return str((training.get('options') or {}).get('stage_id') or 'native-graph-ce')


def collect_sft_progress(training):
    """Capture an actual CE schedule, retaining older mandatory stages unchanged."""
    result=copy.deepcopy(training.get('sft_progress') or {})
    if training.get('schema')==TRAINING_SCHEMA and (training.get('identity') or {}).get('objective') is None:
        frame={key:copy.deepcopy(value) for key,value in training.items() if key not in _TRANSIENT}
        order=frame.get('source_row_order') or list(frame.get('source_row_bindings') or {})
        frame['required_visits']=required_visits(order,float((frame.get('options') or {}).get('epochs',1)))
        key=stage_key(training)
        frame.setdefault('completed_foundations',(result.get(key) or {}).get('completed_foundations',[]))
        result[key]=frame
    return result


def restore_sft_progress(current,stage):
    """Restore credits with the current participant optimizer/RNG, not old state."""
    frame=collect_sft_progress(current).get(stage or 'native-graph-ce')
    if frame is None:return None
    result=copy.deepcopy(frame)
    for key in ('optimizer','rng','fp8'):
        if key in current:result[key]=current[key]
    result['global_step']=int(current.get('global_step',current.get('step',0)))
    result['sft_progress']=collect_sft_progress(current)
    return result


def progress_obligations(progress):
    """Truthful deficits for every stage, including sources absent from RL rows."""
    stages={}
    for stage,frame in progress.items():
        quotas=frame.get('required_visits')
        if quotas is None:
            quotas=required_visits(frame.get('source_row_order') or list(frame.get('source_row_bindings') or {}),
                float((frame.get('options') or {}).get('epochs',1)))
        counts=frame.get('exposure_counts') or {}
        missing={key:max(0,required-int(counts.get(key,0))) for key,required in quotas.items()
                 if int(counts.get(key,0))<required}
        stages[stage]={'complete':not missing,'required_graph_visits':sum(quotas.values()),
            'graph_visits':sum(int(counts.get(key,0)) for key in quotas),
            'remaining_graph_visits':sum(missing.values()),'missing_rows':missing,
            'source_hash':frame.get('source_hash'),
            'train_source_sha256':(frame.get('source_receipts') or [{}])[0].get('sha256'),
            'held_source_sha256':(frame.get('source_receipts') or [{},{}])[-1].get('sha256'),
            'epochs':(frame.get('options') or {}).get('epochs',1)}
    return {'schema':PROGRESS_SCHEMA,'complete':bool(stages) and all(x['complete'] for x in stages.values()),
        'remaining_graph_visits':sum(x['remaining_graph_visits'] for x in stages.values()),'stages':stages}


def communication_foundation(source,counts,*,epochs,trained_bindings):
    """Exact full declared communication cohort; ordinary rows do not gate it."""
    from ..common.hashing import canonical_json_sha256_hex
    quotas=required_visits(source.ordered_ids(),epochs)
    rows={}
    for entry in source.entries:
        if not entry.get('communication'):continue
        key=entry['id']
        compatible=trained_bindings.get(key)==entry['binding']
        rows[key]={'binding':entry['binding'],'required_visits':quotas[key],
            'graph_visits':int(counts.get(key,0)) if compatible else 0,
            'admitted_to_training':compatible}
    cohort={key:{name:row[name] for name in ('binding','required_visits')} for key,row in rows.items()}
    return {'schema':'natlang.native-communication-foundation/1',
        'cohort_sha256':canonical_json_sha256_hex(cohort),'sources':source.receipt(),
        'epochs':epochs,'rows':rows,'graphs':len(rows),
        'required_graph_visits':sum(row['required_visits'] for row in rows.values()),
        'graph_visits':sum(row['graph_visits'] for row in rows.values()),
        'complete':bool(rows) and all(row['admitted_to_training'] and
            row['graph_visits']>=row['required_visits']>0 for row in rows.values())}


def raw_communication_rows(progress):
    """Bind row qualification to actual source identity, across declared stages."""
    rows={}
    for frame in progress.values():
        quotas=frame.get('required_visits') or {}
        counts=frame.get('exposure_counts') or {}
        for key in (frame.get('options') or {}).get('_communication_ids',[]):
            binding=frame['source_row_bindings'][key]
            previous=rows.get(key)
            if previous and previous['binding']!=binding:
                raise ValueError('carried communication row changed across supervised stages: '+key)
            row=rows.setdefault(key,{'binding':binding,'qualified':True,'required_visits':0,'graph_visits':0})
            required=quotas.get(key,0);actual=int(counts.get(key,0))
            row['required_visits']+=required;row['graph_visits']+=actual
            row['qualified']=row['qualified'] and required>0 and actual>=required
    return rows

def qualified_refresh_config(source,requested,models):
    """Use the previous semantic cohort until the whole new raw cohort is ready."""
    from ..data.native_sources import NativeGraphSource
    from ..common.hashing import canonical_json_sha256_hex
    proposed=json.loads(Path(requested).read_text())
    pending=[]
    pairs=proposed['pairs']
    if sha256_file_hex_cached(pairs['path'])!=pairs['sha256']:
        raise ValueError('requested semantic communication source changed')
    for name,model in models.items():
        source_index=NativeGraphSource(pairs['path'],model['native_identity']['tokenizer_fingerprint'],
            required_split='train',pair_model=name,role_ids=model['marker_ids'])
        raw=model.get('raw_communication_rows') or {}
        for entry in source_index.entries:
            if not entry.get('communication'):continue
            actual=raw.get(entry['id']) or {}
            if actual.get('binding')!=entry['binding'] or actual.get('qualified') is not True:
                pending.append({'model_id':name,'graph_id':entry['id'],'binding':entry['binding']})
    if not pending:return proposed,[]
    qualified=source.get('qualified_curriculum') or {}
    previous=qualified.get('config')
    if not isinstance(previous,dict) or canonical_json_sha256_hex(previous)!=qualified.get('sha256'):
        raise ValueError('new communication cohort is not raw-trained and no exact previous semantic cohort is available')
    # Keep useful operational tuning, while source membership remains the
    # explicit previously qualified cohort. Pending membership is never lost.
    selected={**proposed,**{key:previous[key] for key in ('pairs','held_pairs','external_bundles') if key in previous}}
    if 'external_bundles' not in previous:selected.pop('external_bundles',None)
    return selected,pending
