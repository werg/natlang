"""Small operation control records backed by the existing full row checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached

DETAIL_SCHEMA='natlang.native-operation-detail/1'


def source_identity_receipt(source):
    """Keep every source identity without repeated ancestor-wide group maps."""
    result={'path':str(source.path.resolve()),'sha256':source.file_sha256,
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
