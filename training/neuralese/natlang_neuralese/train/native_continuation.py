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


def reject_withdrawn_targets(training,current_bindings):
    """A later CE/objective switch cannot silently reintroduce rejected targets."""
    for item in (training.get('options') or {}).get('withdrawal_history') or []:
        for row in item['rows']:
            if current_bindings.get(row['id'])==row['binding']:
                raise ValueError('native source reintroduced a previously withdrawn target: '+row['id'])


def withdraw_ordinary_rows(training, proof_path, *, current_bindings, model_id, source_origins=None):
    """Verify exact rejected targets, keeping current moments/RNG and good credits."""
    path=Path(proof_path)
    proof=json.loads(path.read_text())
    if proof.get('schema')!='natlang.native-graph-withdrawal/1' or proof.get('model_id')!=model_id:
        raise ValueError('native withdrawal schema/model identity differs')
    evidence=proof.get('evidence') or {}
    evidence_path=Path(evidence.get('path',''))
    if not evidence_path.is_absolute():evidence_path=path.parent/evidence_path
    if not evidence_path.is_file() or sha256_file_hex_cached(evidence_path)!=evidence.get('sha256'):
        raise ValueError('native withdrawal evidence hash differs')
    proof_sha=sha256_file_hex_cached(path)
    history=(training.get('options') or {}).get('withdrawal_history') or []
    if any(item.get('proof_sha256')==proof_sha for item in history):
        if any(row['id'] in current_bindings for row in proof['rows']):
            raise ValueError('native withdrawal reintroduced a previously rejected target')
        return training
    leaves=set()
    def visit(receipt):
        if receipt.get('sources'):
            for child in receipt['sources']:visit(child)
        else:leaves.add(receipt.get('sha256'))
    receipts=training.get('source_receipts') or []
    if receipts:visit(receipts[0])
    bindings=training.get('source_row_bindings') or {}
    forbidden=set((training.get('options') or {}).get('_communication_ids') or [])
    rows=proof.get('rows')
    if not isinstance(rows,list) or not rows:raise ValueError('native withdrawal requires enumerated rows')
    selected={}
    for row in rows:
        key=row.get('id')
        if (not isinstance(key,str) or key in selected or key in forbidden
                or bindings.get(key)!=row.get('binding') or not bindings.get(key)
                or row.get('source_sha256') not in leaves or not row.get('reason')
                or (source_origins or {}).get(key)!=row.get('source_sha256')
                or key in current_bindings):
            raise ValueError('native withdrawal row/source/binding is not an exact removed ordinary target: '+str(key))
        selected[key]=row
    # Copy schedule metadata only; preserve the actual live optimizer/RNG objects.
    result=dict(training)
    result['options']=copy.deepcopy(training.get('options') or {})
    history=list(result['options'].get('withdrawal_history') or [])
    historical_frames=collect_sft_progress(training)
    history.append({'proof_sha256':proof_sha,'evidence':proof['evidence'],
        'rows':[{**row,'consumed_visits':int((training.get('exposure_counts') or {}).get(key,0)),
                 'consumed_visits_by_stage':{stage:int((frame.get('exposure_counts') or {}).get(key,0))
                    for stage,frame in historical_frames.items()
                    if key in (frame.get('source_row_bindings') or {})}}
                for key,row in selected.items()],
        'at_global_step':training.get('global_step',training.get('step',0)),
        'prior_updates_undone':False})
    result['options']['withdrawal_history']=history
    def remove(frame):
        frame['source_row_bindings']={k:v for k,v in (frame.get('source_row_bindings') or {}).items() if k not in selected}
        frame['source_row_order']=[k for k in frame.get('source_row_order') or [] if k not in selected]
        for field in ('exposure_counts','required_visits'):
            if field in frame:frame[field]={k:v for k,v in frame[field].items() if k not in selected}
    remove(result)
    result['sft_progress']=copy.deepcopy(training.get('sft_progress') or {})
    for frame in result['sft_progress'].values():
        for key,row in selected.items():
            if key in (frame.get('source_row_bindings') or {}) and frame['source_row_bindings'][key]!=row['binding']:
                raise ValueError('native withdrawal conflicts with a carried stage binding')
        remove(frame)
        frame.setdefault('options',{})['withdrawal_history']=copy.deepcopy(history)
    return result


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


def verify_initial_foundation_cohort(train, foundation, *, already_declared=False):
    """First activation covers every current communication row, without ordinary CE."""
    if already_declared:
        return  # Later admitted rows remain mandatory without moving this finite cohort.
    rows = lambda source: {entry['id']: entry['binding'] for entry in source.entries
                           if entry.get('communication')}
    expected, actual = rows(train), rows(foundation)
    if not expected or actual != expected:
        raise ValueError('initial communication foundation must bind the entire currently declared communication cohort')


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
