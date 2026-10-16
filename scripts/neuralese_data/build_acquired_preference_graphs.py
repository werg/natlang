"""Durable jointly protected offline preference graphs for the shared native trainer."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import shutil

from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_preference import pair_graph,KIND
from natlang_neuralese.train.native_preference import SCHEMA as OBJECTIVE_SCHEMA
from scripts.neuralese_data.acquired_preferences import normalized_preference_records
from scripts.neuralese_data.build_acquired_full_session_graphs import (
    sessions,prepare_tokenizer,completed_shards,runtime_bindings,_sync_concat,
    _stop_writer_without_deleting_temp,_record_uncommitted_cleanup)
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.prepare_acquired_session_admission import verify_global_source_closure

ROOT=Path(__file__).resolve().parents[2]
SHARD_SCHEMA='natlang.acquired-offline-preference-shard/1'
ADDITIONAL=('objective.json','row-proof.jsonl')


def _bound(path):
    path=Path(path)
    return {'path':str(path.resolve()),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}


def _runtime():
    result=runtime_bindings()
    paths={'preference_builder':Path(__file__),
           'normalized_preference_reader':ROOT/'scripts/neuralese_data/acquired_preferences.py',
           'pair_renderer':ROOT/'training/neuralese/natlang_neuralese/data/native_preference.py',
           'pair_objective':ROOT/'training/neuralese/natlang_neuralese/train/native_preference.py',
           'source_admission':ROOT/'scripts/neuralese_data/prepare_acquired_session_admission.py',
           'decision_holds':ROOT/'scripts/neuralese_data/decision_holds.py'}
    result['files'].update({name:{'sha256':sha256_file_hex(path)} for name,path in paths.items()})
    result['fingerprint']=canonical_json_sha256_hex(result['files'])
    return {key:result[key] for key in ('files','fingerprint')}


def _pair_eligibility(record,protection,decision_ids):
    groups=set(record['source_groups'])
    if groups & (protection['quality']|protection['pending']):
        return 'current-source-quality-or-pending-protection'
    held_sources={identifier.split(':decision:',1)[0] for identifier in decision_ids}|decision_ids
    for trajectory in record['alternatives'].values():
        identifiers={trajectory['id'],*(trajectory['id']+':'+str(i) for i,message in enumerate(trajectory['messages'])
                                      if message.get('role')=='assistant')}
        if identifiers & held_sources:return 'current-decision-hold'
    return None



def protected_pair(record,protection,decision_ids):
    """Apply one shared exact whole-pair protection contract before rendering/audit."""
    reason=_pair_eligibility(record,protection,decision_ids)
    if reason:return None,reason
    split='test' if record['split']!='train' or set(record['source_groups']) & protection['held'] else 'train'
    record['split']=split
    for trajectory in record['alternatives'].values():trajectory['split']=split
    return record,None

def main(argv=None,*,prepared=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-receipt',required=True,type=Path)
    parser.add_argument('--base-tokenizer',required=True,type=Path)
    parser.add_argument('--protection',required=True,type=Path)
    parser.add_argument('--source-closure-receipt',type=Path,
        help='durable whole-source closure receipt; its bound inputs and current protections are rechecked')
    parser.add_argument('--decision-hold',required=True,action='append',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--chunk-pairs',type=int,default=100)
    parser.add_argument('--beta',type=float,default=.1)
    parser.add_argument('--margin',type=float,default=0.)
    args=parser.parse_args(argv)
    import math
    if args.chunk_pairs<1 or not math.isfinite(args.beta) or args.beta<=0 or not math.isfinite(args.margin):
        raise ValueError('offline conversion requires positive chunk/beta and finite margin')
    with acquire_output_lease(args.output):return convert(args,prepared)


def convert(args,prepared):
    export=json.loads(args.export_receipt.read_text())
    if export.get('schema')!='natlang.normalized-agent-source-file/1':
        raise ValueError('offline preference needs its complete normalized source export receipt')
    source=args.export_receipt.parent/'full-trajectories.jsonl.zst'
    source_binding=_bound(source)
    if export['outputs'][source.name]!=source_binding['sha256']:
        raise ValueError('offline normalized source changed from its export receipt')
    protection=current_protection(args.protection)
    closure_receipt = args.source_closure_receipt
    if closure_receipt is None:
        verify_global_source_closure(args.protection,protection)
    else:
        verify_global_source_closure(args.protection,protection,receipt_path=closure_receipt)
    projection=json.loads(args.protection.read_text())
    if not any(item['sha256']==sha256_file_hex(args.export_receipt)
               and item['source_sha256']==source_binding['sha256']
               for item in projection['normalized_export_bindings']):
        raise ValueError('offline source is absent from the current complete global held closure')
    holds=sorted((ROOT/'training/decision-holds').glob('*.jsonl'))
    if {path.resolve() for path in args.decision_hold}!={path.resolve() for path in holds}:
        raise ValueError('offline conversion needs the complete current decision hold file set')
    decision_ids,decision_bindings=load_decision_holds(holds)
    source_rows=0
    for value in sessions(source):
        if (value['source_file']!=export['contract']['source']
                or value['source_revision']!=export['contract']['revision']):
            raise ValueError('offline normalized row differs from its original source contract')
        source_rows+=1
    if source_rows!=export['complete_sessions']:
        raise ValueError('offline source session inventory differs from its complete export')
    units=sum(1 for _ in normalized_preference_records(source))
    prepared=prepare_tokenizer(args.base_tokenizer) if prepared is None else prepared
    prepared.validate(args.base_tokenizer)
    binding={'source':source_binding,'export_receipt':_bound(args.export_receipt),
        'protection':protection['proof'],'decision_holds':decision_bindings,
        'prepared_tokenizer':prepared.binding,'runtime':_runtime(),
        'objective':{'kind':KIND,'beta':args.beta,'margin':args.margin},
        'chunk_pairs':args.chunk_pairs,'normalized_sessions':source_rows,'source_units':units}
    if closure_receipt is not None:
        binding['source_closure_receipt'] = _bound(closure_receipt)
    output=args.output;shard_root=output/'pair-shards'
    receipt_path=output/'receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('binding')!=binding:raise ValueError('immutable offline preference output has changed inputs')
    frontier,shards=completed_shards(shard_root,binding,units,schema=SHARD_SCHEMA,additional_outputs=ADDITIONAL)
    if receipt_path.exists():
        for item in [*receipt['outputs'],receipt['objective'],receipt['row_proof'],receipt['omissions']]:
            if _bound(item['path'])!={key:item[key] for key in ('path','sha256','bytes')}:
                raise ValueError('immutable completed offline preference output changed')
        if frontier!=units or receipt.get('shards')!=shards:
            raise ValueError('completed offline preference receipt lacks its exact durable source frontier')
        return receipt
    counts=Counter()
    for shard in shards:counts.update(shard['counts'])
    iterator=itertools.islice(normalized_preference_records(source),frontier,None)
    while frontier<units:
        batch=itertools.islice(iterator,args.chunk_pairs)
        batch_count=0
        directory=shard_root/f'shard-{frontier:08d}';directory.mkdir()
        writers={split:GraphWriter(directory/split/'native-graphs.jsonl.zst') for split in ('train','test')}
        shard_counts=Counter();spec_rows={}
        try:
            with (directory/'omissions.jsonl').open('w') as omissions,(directory/'row-proof.jsonl').open('w') as proof:
                for indices,record,rejection in batch:
                    batch_count+=1
                    shard_counts['normalized_sessions']+=len(indices)
                    record,reason=(None,str(rejection)) if rejection else protected_pair(record,protection,decision_ids)
                    if reason:
                        omissions.write(json.dumps({'normalized_source_rows':indices,'reason':reason})+'\n')
                        shard_counts['omitted_source_units']+=1;continue
                    split=record['split']
                    row,spec=pair_graph(record,prepared.tokenizer,tokenizer_fingerprint=prepared.expanded_fingerprint)
                    if row['id'] in spec_rows:raise ValueError('offline source repeats an immutable graph')
                    writers[split].write(row);spec_rows[row['id']]=spec
                    proof.write(json.dumps({'id':row['id'],'graph_sha256':canonical_json_sha256_hex(row),
                        'source_groups':row['source_groups'],'split':split,'normalized_source_rows':indices,
                        'source_record_sha256':spec['source_record_sha256']})+'\n')
                    shard_counts['graphs:'+split]+=1
                for stream in (omissions,proof):stream.flush();os.fsync(stream.fileno())
            if not batch_count:raise ValueError('offline source ended before its authenticated frontier')
            outputs=[writer.finish() for writer in writers.values()]
            objective={'schema':OBJECTIVE_SCHEMA,'kind':KIND,'beta':args.beta,'margin':args.margin,
                       'offline':True,'native_on_policy':False,'rows':spec_rows}
            write_json_durable(directory/'objective.json',objective)
            outputs=[{**item,'path':str(Path(item['path']).relative_to(directory))} for item in outputs]
            outputs += [{**_bound(directory/name),'path':name,'rows':len(spec_rows)} for name in ADDITIONAL]
            shard_receipt={'schema':SHARD_SCHEMA,'binding_sha256':canonical_json_sha256_hex(binding),
                'source_record_start':frontier,'source_record_stop':frontier+batch_count,'source_record_count':batch_count,
                'counts':dict(shard_counts),'outputs':outputs,'omissions_sha256':sha256_file_hex(directory/'omissions.jsonl')}
            write_json_durable(directory/'receipt.json',shard_receipt)
        except BaseException:
            for writer in writers.values():_stop_writer_without_deleting_temp(writer)
            _record_uncommitted_cleanup(shard_root,directory);shutil.rmtree(directory,ignore_errors=True)
            raise
        frontier+=batch_count;counts.update(shard_counts)
        shards.append({'path':str(directory/'receipt.json'),'sha256':sha256_file_hex(directory/'receipt.json'),
                       'start':frontier-batch_count,'stop':frontier,'counts':dict(shard_counts)})
        write_json_durable(output/'resume-state.json',{'binding':binding,'completed_source_record_frontier':frontier,'shards':shards})
    if counts['normalized_sessions']!=source_rows:
        raise ValueError('offline pair conversion lost original normalized sessions')
    if (_bound(source)!=source_binding or current_protection(args.protection)['proof']!=protection['proof']
            or load_decision_holds(holds)!=(decision_ids,decision_bindings)
            or sorted((ROOT/'training/decision-holds').glob('*.jsonl'))!=holds):
        raise ValueError('offline source/protection/holds changed during conversion')
    if closure_receipt is not None:
        verify_global_source_closure(args.protection,current_protection(args.protection),
                                     receipt_path=closure_receipt)
    prepared.validate(args.base_tokenizer)
    all_rows={};outputs=[]
    for shard in shards:
        directory=Path(shard['path']).parent
        rows=json.loads((directory/'objective.json').read_text())['rows']
        if set(rows)&all_rows.keys():raise ValueError('offline graph identity duplicated across durable shards')
        all_rows.update(rows)
    for split in ('train','test'):
        target=output/split/'native-graphs.jsonl.zst'
        _sync_concat(target,[Path(item['path']).parent/split/'native-graphs.jsonl.zst' for item in shards])
        outputs.append({**_bound(target),'rows':counts['graphs:'+split]})
    objective={'schema':OBJECTIVE_SCHEMA,'kind':KIND,'beta':args.beta,'margin':args.margin,
        'offline':True,'native_on_policy':False,'source':source_binding,'rows':all_rows}
    write_json_durable(output/'objective.json',objective)
    _sync_concat(output/'row-proof.jsonl',[Path(item['path']).parent/'row-proof.jsonl' for item in shards])
    _sync_concat(output/'omissions.jsonl',[Path(item['path']).parent/'omissions.jsonl' for item in shards])
    result={'schema':'natlang.acquired-offline-preference-graphs/1','binding':binding,'counts':dict(counts),
        'outputs':outputs,'objective':_bound(output/'objective.json'),'row_proof':_bound(output/'row-proof.jsonl'),
        'omissions':_bound(output/'omissions.jsonl'),'shards':shards,'training_admission':False,
        'ordinary_positive_rejected_credits':0,'capacity_qualified':False}
    write_json_durable(receipt_path,result)
    return result


if __name__=='__main__':main()
