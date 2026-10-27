"""Mandatory CE and offline-pair credits carried through shared objectives.

These are schedule metadata, never another optimizer/RNG history. The live
participant owns one current model/optimizer/FP8/RNG continuation stream.
"""
from __future__ import annotations
import math
import json
from pathlib import Path
from ..common.hashing import sha256_file_hex_cached
from ..common.copying import deepcopy_metadata

TRAINING_SCHEMA='natlang.neuralese-native-sft-training/1'
PROGRESS_SCHEMA='natlang.native-sft-progress/1'
# Live learner state, like optimizer history, belongs to the rolling checkpoint, never a stage frame.
_TRANSIENT={'optimizer','rng','fp8','sft_progress','diloco'}


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
    result['options']=deepcopy_metadata(training.get('options') or {})
    history=list(result['options'].get('withdrawal_history') or [])
    historical_frames=collect_sft_progress(training,borrow_frames=True)
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
    result['sft_progress']=deepcopy_metadata(training.get('sft_progress') or {})
    for frame in result['sft_progress'].values():
        for key,row in selected.items():
            if key in (frame.get('source_row_bindings') or {}) and frame['source_row_bindings'][key]!=row['binding']:
                raise ValueError('native withdrawal conflicts with a carried stage binding')
        remove(frame)
        frame.setdefault('options',{})['withdrawal_history']=deepcopy_metadata(history)
    return result


def required_visits(order,epochs):
    if not math.isfinite(epochs) or epochs<=0:
        raise ValueError('native exposure epochs must be finite and positive')
    whole=int(epochs);fraction=math.ceil((epochs-whole)*len(order))
    return {key:whole+int(index<fraction) for index,key in enumerate(order)}


def stage_key(training):
    return str((training.get('options') or {}).get('stage_id') or 'native-graph-ce')


def _retain_pending_sources(frame,previous):
    """Merge mandatory source declarations into the existing CE frame, with no earned credit."""
    active=frame.get('source_row_bindings') or {}
    pending=deepcopy_metadata(previous.get('pending_source_rows') or {})
    declarations=deepcopy_metadata(previous.get('pending_source_declarations') or {})
    quotas=dict(previous.get('required_visits') or {})
    for key,required in required_visits(frame.get('source_row_order') or list(active),
            float((frame.get('options') or {}).get('epochs',1))).items():
        quotas[key]=max(quotas.get(key,0),required)
    for key,required in ((frame.get('options') or {}).get('_objective_required_visits') or {}).items():
        if key in active:quotas[key]=max(quotas.get(key,0),required)
    for declaration in (frame.get('options') or {}).get('_pending_native_source_declarations') or []:
        reference=declaration['binding'];fingerprint=reference['sha256']
        record={key:deepcopy_metadata(value) for key,value in declaration.items() if key!='rows'}
        if fingerprint in declarations and declarations[fingerprint]!=record:
            raise ValueError('pending source declaration changed without a new immutable binding')
        declarations[fingerprint]=record
        reject_withdrawn_targets(frame,{key:row['binding'] for key,row in declaration['rows'].items()})
        for key,row in declaration['rows'].items():
            prior=pending.get(key)
            if prior and prior['binding']!=row['binding']:
                raise ValueError('pending native source changed an existing row binding: '+key)
            if key in active and active[key]!=row['binding']:
                raise ValueError('active native source differs from its pending admitted row: '+key)
            pending[key]={'binding':row['binding'],'required_visits':max(
                row['required_visits'],(prior or {}).get('required_visits',0))}
            quotas[key]=max(quotas.get(key,0),row['required_visits'])
    counts=frame.setdefault('exposure_counts',{})
    for key,row in list(pending.items()):
        quotas[key]=max(quotas.get(key,0),row['required_visits'])
        if key in active:
            if active[key]!=row['binding']:
                raise ValueError('capacity activation changed pending native row semantics: '+key)
            # Existing exact credits belong to this one row, not a second
            # pending counter. A newly active row earns no visits by admission.
            counts[key]=max(int(counts.get(key,0)),int((previous.get('exposure_counts') or {}).get(key,0)))
            del pending[key]
        else:
            counts[key]=int((previous.get('exposure_counts') or {}).get(key,0))
    frame['required_visits']=quotas
    if declarations or pending:
        frame['pending_source_rows']=pending
        frame['pending_source_declarations']=declarations
    return frame


def collect_sft_progress(training, *, borrow_frames=False):
    """Capture a CE schedule; metadata-only readers may borrow nested values.

    Borrowed results must not be mutated by the caller. Pending-source merging
    still owns its counters and declarations, so discovery never alters the
    decoded checkpoint. Training callers retain fully independent deep copies.
    """
    historical=training.get('sft_progress') or {}
    capture=training.get('schema')==TRAINING_SCHEMA and ((training.get('identity') or {}).get('objective') is None
            or (training.get('identity') or {}).get('objective_kind')=='native-offline-pairwise')
    key=stage_key(training) if capture else None
    # The current frame replaces its historical snapshot. Copy only the other
    # stages; the pending merge below owns the few prior values still needed.
    result=(dict(historical) if borrow_frames else
            deepcopy_metadata({name:None if capture and name==key else frame
                           for name,frame in historical.items()}))
    if capture:
        previous=historical.get(key) or {}
        frame={key:value if borrow_frames else deepcopy_metadata(value)
               for key,value in training.items() if key not in _TRANSIENT}
        if borrow_frames:
            # _retain_pending_sources updates only these existing nested values;
            # quotas and pending declarations are allocated by that helper.
            frame['exposure_counts']=dict(training.get('exposure_counts') or {})
        _retain_pending_sources(frame,previous)
        if 'completed_foundations' not in frame:
            completed=previous.get('completed_foundations',[])
            frame['completed_foundations']=completed if borrow_frames else deepcopy_metadata(completed)
        result[key]=frame
    return result


def sft_frame_complete(frame):
    """Exact completion without allocating the reporting-only deficit map."""
    quotas=frame.get('required_visits')
    if quotas is None:
        quotas=required_visits(frame.get('source_row_order') or list(frame.get('source_row_bindings') or {}),
            float((frame.get('options') or {}).get('epochs',1)))
    counts=frame.get('exposure_counts') or {}
    complete=True
    for key,required in quotas.items():
        # Visit every counter, including malformed counters after a deficit.
        if int(counts.get(key,0))<required:
            complete=False
    if (frame.get('identity') or {}).get('objective_kind'):
        # The full report validates this field even for completed old stages.
        # Dependency discovery must retain that malformed-state rejection.
        frame['identity']['objective']
    return complete


def restore_sft_progress(current,stage,*,collected_progress=None):
    """Restore credits/current state; optional progress is read-only, from current.

    The trainer already owns an independent collection of this history. Reuse
    it without another full copy; the selected active frame remains isolated.
    """
    progress=collect_sft_progress(current)if collected_progress is None else collected_progress
    frame=progress.get(stage or 'native-graph-ce')
    if frame is None:return None
    result=deepcopy_metadata(frame)
    for key in ('optimizer','rng','fp8'):
        if key in current:result[key]=current[key]
    result['global_step']=int(current.get('global_step',current.get('step',0)))
    result['sft_progress']=progress
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
        if (frame.get('identity') or {}).get('objective_kind'):
            stages[stage]['objective_kind']=frame['identity']['objective_kind']
            stages[stage]['objective']=deepcopy_metadata(frame['identity']['objective'])
        if frame.get('pending_source_declarations') or frame.get('pending_source_rows'):
            stages[stage].update(
                pending_source_graph_visits=sum(value for key,value in missing.items() if key in (frame.get('pending_source_rows') or {})),
                active_remaining_graph_visits=sum(value for key,value in missing.items() if key not in (frame.get('pending_source_rows') or {})),
                pending_source_declarations=deepcopy_metadata(frame.get('pending_source_declarations') or {}))
    return {'schema':PROGRESS_SCHEMA,'complete':bool(stages) and all(x['complete'] for x in stages.values()),
        'remaining_graph_visits':sum(x['remaining_graph_visits'] for x in stages.values()),'stages':stages}


def verify_initial_foundation_cohort(train, foundation, *, already_declared=False):
    """First activation covers every current communication row, without ordinary CE."""
    if already_declared:
        return  # Later admitted rows remain mandatory without moving this finite cohort.
    rows = lambda source: {entry['id']: entry['binding'] for entry in source.entries
                           if entry.get('producer_consumer', entry.get('communication', False))}
    expected, actual = rows(train), rows(foundation)
    if not expected or actual != expected:
        raise ValueError('initial communication foundation must bind the entire currently declared communication cohort')


def communication_foundation(source,counts,*,epochs,trained_bindings):
    """Exact full declared communication cohort; ordinary rows do not gate it."""
    from .native_operation_report import source_receipt
    quotas=required_visits(source.ordered_ids(),epochs)
    rows={}
    for entry in source.entries:
        if not entry.get('producer_consumer', entry.get('communication', False)):continue
        key=entry['id']
        compatible=trained_bindings.get(key)==entry['binding']
        rows[key]={'binding':entry['binding'],'required_visits':quotas[key],
            'graph_visits':int(counts.get(key,0)) if compatible else 0,
            'admitted_to_training':compatible}
    return {'schema':'natlang.native-communication-foundation/1',
        'sources':source_receipt(source), 'epochs':epochs, 'rows':rows,
        **_foundation_summary(rows)}


def _foundation_summary(rows):
    """One row-completion law for the producer and transported proof verifier."""
    from ..common.hashing import canonical_json_sha256_hex
    cohort={key:{name:row[name] for name in ('binding','required_visits')} for key,row in rows.items()}
    return {'cohort_sha256':canonical_json_sha256_hex(cohort),'graphs':len(rows),
        'required_graph_visits':sum(row['required_visits'] for row in rows.values()),
        'graph_visits':sum(row['graph_visits'] for row in rows.values()),
        'complete':bool(rows) and all(row['admitted_to_training'] and
            row['graph_visits']>=row['required_visits']>0 for row in rows.values())}


def validate_communication_foundation(value, *, epochs, source_sha256):
    """Verify inline completed row evidence without reading a remote source path."""
    if (not isinstance(source_sha256,str) or len(source_sha256)!=64
            or any(char not in '0123456789abcdef' for char in source_sha256)
            or not isinstance(value,dict) or value.get('schema')!='natlang.native-communication-foundation/1'
            or type(value.get('epochs')) not in (int,float) or not math.isfinite(value['epochs'])
            or value['epochs']<=0 or value['epochs']!=epochs
            or (value.get('sources') or {}).get('sha256')!=source_sha256
            or not isinstance(value.get('rows'),dict) or not value['rows']):
        raise ValueError('communication foundation did not consume the entire declared admitted cohort: source/epochs/rows mismatch')
    for key,row in value['rows'].items():
        if (not isinstance(key,str) or not key or not isinstance(row,dict)
                or not isinstance(row.get('binding'),str) or not row['binding']
                or type(row.get('required_visits')) is not int or row['required_visits']<=0
                or type(row.get('graph_visits')) is not int or row['graph_visits']<row['required_visits']
                or row.get('admitted_to_training') is not True):
            raise ValueError('communication foundation requires completed exact exposure for every row')
    expected=_foundation_summary(value['rows'])
    if (expected['complete'] is not True or any(value.get(key)!=item for key,item in expected.items())
            or any(type(value.get(key)) is not int for key in ('graphs','required_graph_visits','graph_visits'))
            or value.get('complete') is not True):
        raise ValueError('communication foundation aggregate/cohort differs from its per-row exposure')
    return value


def raw_communication_rows(progress):
    """Bind row qualification to actual source identity, across declared stages."""
    rows={}
    for frame in progress.values():
        quotas=frame.get('required_visits') or {}
        counts=frame.get('exposure_counts') or {}
        options = frame.get('options') or {}
        for key in options.get('_producer_consumer_ids', options.get('_communication_ids', [])):
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
