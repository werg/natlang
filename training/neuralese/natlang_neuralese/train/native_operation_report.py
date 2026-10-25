"""Small operation control records backed by the existing full row checkpoint."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached

DETAIL_SCHEMA='natlang.native-operation-detail/1'


def source_identity_receipt(source):
    """Keep every source identity without repeated ancestor-wide group maps."""
    result={'path':str(source.source_identity_path.resolve()),'sha256':source.file_sha256,
            'rows':len(source.entries)}
    if source.kind=='manifest':
        children=[source_identity_receipt(child) for child in source.children]
        result.update(sources=children,
            communication_graphs=sum(child['communication_graphs'] for child in children))
    else:
        result['communication_graphs']=sum(bool(entry.get('communication')) for entry in source.entries)
    return result


def source_receipt(source):
    """Record exact whole-source totals once, plus the complete identity tree."""
    result=source_identity_receipt(source)
    groups={}
    for entry in source.entries:
        for group in entry['groups']:
            count=groups.setdefault(group,{'examples':0,'supervised_tokens':0})
            count['examples']+=1
            count['supervised_tokens']+=entry['supervised_tokens']
    return {**result,'by_source_group':groups,'group_counts_scope':'root-only/1'}


class SourceIdentityView:
    """The classification verifier needs exact entries and source identities.

    Its existing producer module is immutable evidence for migrated caches.
    This consumer view supplies its read interface without manufacturing the
    unrelated full group histograms at every historical manifest ancestor.
    """
    def __init__(self,source):
        self.entries=source.entries
        self._receipt=source_identity_receipt(source)

    def receipt(self):
        return self._receipt


def _digest(value):
    # Stream the inventory, while letting the native encoder handle each
    # bounded batch of records. iterencode's Python scalar generator dominated final reports on
    # million-row recipes; encoding the whole inventory would duplicate it.
    digest=hashlib.sha256()
    encoder=json.JSONEncoder(ensure_ascii=False,sort_keys=True,separators=(',',':'))
    if isinstance(value,dict):
        digest.update(b'{')
        keys=sorted(value)
        for start in range(0,len(keys),1024):
            if start:digest.update(b',')
            # Retain JSON's exact non-string key conversion as well.
            text=encoder.encode({key:value[key] for key in keys[start:start+1024]})
            digest.update(text[1:-1].encode('utf-8'))
        digest.update(b'}')
    elif isinstance(value,(list,tuple)):
        digest.update(b'[')
        for start in range(0,len(value),1024):
            if start:digest.update(b',')
            text=encoder.encode(value[start:start+1024])
            digest.update(text[1:-1].encode('utf-8'))
        digest.update(b']')
    else:
        digest.update(encoder.encode(value).encode('utf-8'))
    return digest.hexdigest()


def _binding(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}


def _detail(value,checkpoint,kind,**selection):
    return {'schema':DETAIL_SCHEMA,'checkpoint':checkpoint,'kind':kind,
            'entries':len(value),'sha256':_digest(value),**selection}


def compact_obligations(value,checkpoint):
    if value is None:return None
    result={**value,'stages':{}}
    for name,stage in value.get('stages',{}).items():
        item=dict(stage)
        if 'missing_rows' in item:
            rows=item.pop('missing_rows')
            item['missing_rows_detail']=_detail(rows,checkpoint,'stage-deficits',stage=name)
        result['stages'][name]=item
    return result


def _source_receipt(value,checkpoint,path):
    result=dict(value)
    if 'by_source_group' in result:
        groups=result.pop('by_source_group')
        result['by_source_group_detail']=_detail(groups,checkpoint,'checkpoint-path',path=path+['by_source_group'])
    if 'sources' in value:
        result['sources']=[_source_receipt(child,checkpoint,path+['sources',index])
                           for index,child in enumerate(value['sources'])]
    return result


def compact_training_report(report,checkpoint):
    """Preserve control fields; replace only exact checkpoint-backed repetitions."""
    result=dict(report)
    result['mandatory_sft']=compact_obligations(report.get('mandatory_sft'),checkpoint)
    for name,index in [('train_source',0),('held_source',1)]:
        if name in report:
            result[name]=_source_receipt(report[name],checkpoint,['training','source_receipts',index])
    return result


def compact_exposure(exposure,checkpoint):
    result=dict(exposure)
    if 'by_source_group' in result:
        value=result.pop('by_source_group')
        result['by_source_group_detail']=_detail(value,checkpoint,'group-exposure')
    return result


def resolve_detail(proof,*,checkpoint_loader=None,source=None):
    """Demand-driven exact row proof; normal controller operation never calls it."""
    if proof.get('schema')!=DETAIL_SCHEMA:raise ValueError('unsupported native operation detail')
    binding=proof['checkpoint'];path=Path(binding['path']).resolve(strict=True)
    if path.stat().st_size!=binding['bytes'] or sha256_file_hex_cached(path)!=binding['sha256']:
        raise ValueError('native operation detail checkpoint changed')
    if checkpoint_loader is None:
        from .checkpoint_policy import CheckpointPolicy
        checkpoint_loader=CheckpointPolicy.load
    state=checkpoint_loader(path)
    kind=proof['kind']
    if kind=='checkpoint-path':
        value=state
        for key in proof['path']:value=value[key]
    elif kind=='stage-deficits':
        from .native_continuation import progress_obligations
        value=progress_obligations(state['training']['sft_progress'])['stages'][proof['stage']]['missing_rows']
    elif kind=='group-exposure':
        from ..data.native_sources import NativeGraphSource
        training=state['training'];config=state['model_config'];receipt=training['source_receipts'][0]
        if source is None:
            source=NativeGraphSource(receipt['path'],config['tokenizer_fingerprint'],required_split='train',role_ids=config['marker_ids'])
        if source.file_sha256!=receipt['sha256'] or source.bindings!=training['source_row_bindings']:
            raise ValueError('native group exposure source/rows changed')
        value={};counts=training['exposure_counts']
        for entry in source.entries:
            visits=counts[entry['id']]
            for group in entry['groups']:
                item=value.setdefault(group,{'graph_visits':0,'supervised_token_visits':0})
                item['graph_visits']+=visits
                item['supervised_token_visits']+=visits*entry['supervised_tokens']
    else:raise ValueError('unsupported native operation detail kind')
    if len(value)!=proof['entries'] or _digest(value)!=proof['sha256']:
        raise ValueError('native operation detail content differs')
    return value


def control_progress(progress,report_path):
    """Keep every control field, including future guards, without bulky payloads."""
    if progress.get('schema')!='natlang.neuralese-native-sft-report/1':return progress
    report=_binding(report_path)
    result=dict(progress)
    checkpoint_path=Path(progress.get('checkpoint') or Path(report_path).parent/'checkpoint.pt')
    checkpoint=_binding(checkpoint_path)
    if progress.get('checkpoint_sha256') not in (None,checkpoint['sha256']):
        raise ValueError('native operation report checkpoint differs')
    result['mandatory_sft']=compact_obligations(progress.get('mandatory_sft'),checkpoint)
    for key in ('metrics','train_source','held_source','model_config','raw_communication_rows'):
        if key in result:
            result.pop(key)
            result.setdefault('report_details',{})[key]={'report':report,'field':key}
    if isinstance(result.get('communication_foundation'),dict):
        foundation=dict(result['communication_foundation'])
        for key in ('rows','sources'):
            if key in foundation:
                foundation.pop(key)
                foundation.setdefault('report_details',{})[key]={'report':report,'field':['communication_foundation',key]}
        result['communication_foundation']=foundation
    result['full_report']=report
    return result


def resolve_control_field(reference):
    binding=reference['report'];path=Path(binding['path']).resolve(strict=True)
    if path.stat().st_size!=binding['bytes'] or sha256_file_hex_cached(path)!=binding['sha256']:
        raise ValueError('native operation control report changed')
    value=json.loads(path.read_text())
    fields=reference['field']
    for key in fields if isinstance(fields,list) else [fields]:value=value[key]
    return value


def saved_progress(checkpoint):
    """Use the trusted trainer's exact current receipt; stale slots fall back."""
    receipt_path=Path(checkpoint['path']).parent/'native-training-receipt.json'
    if not receipt_path.is_file():return None
    receipt=json.loads(receipt_path.read_text())
    if (receipt.get('schema')!='natlang.native-training-readiness/1'
            or (receipt.get('checkpoint') or {}).get('sha256')!=checkpoint['sha256']):return None
    if (receipt.get('native_identity') or {}).get('checkpoint_sha256')!=checkpoint['sha256']:
        raise ValueError('native progress receipt identity differs from its checkpoint')
    binding=receipt['training_receipt'];path=Path(binding['path']).resolve(strict=True)
    if sha256_file_hex_cached(path)!=binding['sha256']:
        raise ValueError('native progress bound report changed')
    value=json.loads(path.read_text())
    progress=value.get('mandatory_sft')
    if (value.get('schema')!='natlang.neuralese-native-sft-report/1'
            or value.get('checkpoint_sha256')!=checkpoint['sha256']
            or not isinstance(progress,dict) or progress!=receipt.get('mandatory_sft')
            or progress.get('schema')!='natlang.native-sft-progress/1'):
        raise ValueError('native progress report/receipt/checkpoint differ')
    for stage in progress.get('stages',{}).values():
        detail=stage.get('missing_rows_detail')
        if detail is not None and (detail.get('schema')!=DETAIL_SCHEMA
                or detail.get('checkpoint',{}).get('sha256')!=checkpoint['sha256']):
            raise ValueError('native progress row detail belongs to another checkpoint')
    actual_checkpoint=_binding(checkpoint['path'])
    if actual_checkpoint['sha256']!=checkpoint['sha256']:
        raise ValueError('native progress current checkpoint changed')
    return compact_obligations(progress,actual_checkpoint)


def compact_controller_state(state,report_path):
    """Pure commit value for an owned boundary; all unrelated cursor data survives."""
    progress=state.get('last_learner_progress') or {}
    if progress.get('schema')!='natlang.neuralese-native-sft-report/1':return dict(state)
    actual=json.loads(Path(report_path).read_text())
    if progress.get('full_report'):
        if progress['full_report']!=_binding(report_path):
            raise ValueError('controller compact progress refers to another report')
        compact=control_progress(actual,report_path)
        if progress!=compact:raise ValueError('controller compact progress changed its report controls')
    else:
        if progress!=actual:raise ValueError('controller progress differs from actual learner report')
        compact=control_progress(actual,report_path)
    result=dict(state);result['last_learner_progress']=compact
    if state.get('mandatory_sft') is not None:
        if state['mandatory_sft'] not in (actual.get('mandatory_sft'),compact.get('mandatory_sft')):
            raise ValueError('controller mandatory sources differ from actual learner report')
        result['mandatory_sft']=compact['mandatory_sft']
    return result


def finite_training_state_receipt(model, optimizer, checkpoint_sha256):
    """Check saved-boundary numerics without granting runtime qualification."""
    import math
    import torch
    from .optim_restore import _children
    tensors = {'model': 0, 'optimizer': 0}
    failures = []
    seen = set()

    def check(value, scope, path):
        if torch.is_tensor(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if value.is_floating_point() or value.is_complex():
                tensors[scope] += 1
                selected = value.detach()
                # Float8 has no isfinite kernel on some supported backends.
                if str(selected.dtype).startswith('torch.float8_'):
                    selected = selected.float()
                if not bool(torch.isfinite(selected).all()):
                    failures.append(path)
        elif isinstance(value, dict):
            for index, item in enumerate(value.values()):
                check(item, scope, path + '/' + str(index))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                check(item, scope, path + '/' + str(index))
        elif isinstance(value, float) and not math.isfinite(value):
            failures.append(path)

    for name, parameter in model.named_parameters():
        check(parameter, 'model', 'parameter/' + name)
    for name, buffer in model.named_buffers():
        check(buffer, 'model', 'buffer/' + name)
    for name, child in _children(optimizer):
        check(child.state, 'optimizer', 'optimizer/' + name)
    return {'schema': 'natlang.native-finite-training-state/1',
            'checkpoint_sha256': checkpoint_sha256,
            'passed': not failures and tensors['model'] > 0,
            'floating_tensors': tensors, 'nonfinite': failures,
            'runtime_qualification_granted': False}


def supervised_progress_numerics_ok(report, checkpoint):
    """A deferred runtime check may accompany genuinely finite earned work."""
    if 'progress_numerics' not in report:
        return report.get('numerics_ok') is True
    from ..train.native_readiness import _bound
    evidence = report.get('progress_numerics')
    if not isinstance(evidence, dict):
        return False
    if (evidence.get('schema') != 'natlang.native-finite-training-state/1'
            or evidence.get('passed') is not True
            or evidence.get('checkpoint_sha256') != checkpoint['sha256']
            or evidence.get('runtime_qualification_granted') is not False
            or evidence.get('nonfinite') != []):
        return False
    counts = evidence.get('floating_tensors') or {}
    if (set(counts) != {'model', 'optimizer'}
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts['model'] == 0):
        return False
    if report.get('numerics_ok') is True:
        return True
    reference = report.get('runtime_certificate') or {}
    if set(reference) != {'path', 'sha256', 'passed'} or reference['passed'] is not False:
        return False
    certificate = json.loads(_bound({key: reference[key] for key in ('path', 'sha256')}).read_text())
    return (certificate.get('passed') is False
            and certificate.get('deferred') is True
            and certificate.get('reason') == 'shared qualification request pending'
            and certificate.get('checkpoint_sha256') == checkpoint['sha256']
            and not certificate.get('error'))


def active_stage_handoff(report,checkpoint,*,stage_id=None):
    """Publish a finished active stage without claiming all recipe/held debt complete."""
    if report.get('schema')!='natlang.neuralese-native-sft-report/1'or report.get('active_stage_complete')is not True:return False
    record=report.get('active_stage_completion')or{}
    if (record.get('schema')!='natlang.native-active-stage-completion/1'
            or not isinstance(record.get('stage_id'),str)or not record['stage_id']
            or (stage_id is not None and record['stage_id']!=stage_id)
            or record.get('checkpoint_sha256')!=checkpoint['sha256']
            or record.get('train_source_sha256')!=(report.get('train_source')or{}).get('sha256')
            or record.get('held_source_sha256')!=(report.get('held_source')or{}).get('sha256')):return False
    import re
    if any(not re.fullmatch('[0-9a-f]{64}',str(record.get(key,'')))
           for key in ('checkpoint_sha256','train_source_sha256','held_source_sha256')):return False
    required=record.get('required_graph_visits');observed=record.get('graph_visits');remaining=record.get('remaining_graph_visits')
    if (type(required)is not int or required<1 or type(observed)is not int or observed<required
            or type(remaining)is not int or remaining!=0):return False
    held=(report.get('metrics')or{}).get('held')or{}
    return (record.get('held_scope')in ('cadence-subset','full')
        and record['held_scope']==held.get('scope')
        and record.get('full_held_qualification_complete')is (held.get('scope')=='full')
        and not held.get('operationally_interrupted')and held.get('examples',0)>0
        and type(held.get('loss'))in (int,float)and math.isfinite(held['loss'])
        and held.get('source_sha256')==record['held_source_sha256']
        and supervised_progress_numerics_ok(report,checkpoint))
