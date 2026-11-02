#!/usr/bin/env python3
"""Bind each repair export to its original conversion's actual history profile."""
import argparse
import json
from pathlib import Path
from collections import Counter
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable


def checked(ref):
    p=Path(ref['path']).resolve(strict=True)
    if sha256_file_hex(p)!=ref['sha256']:raise ValueError('repair profile evidence changed')
    return p


def build(batch_ref,census_ref,out):
    batch=json.loads(checked(batch_ref).read_text());census=json.loads(checked(census_ref).read_text())
    if (batch.get('schema')!='natlang.acquired-syntax-repair-batch/1' or batch.get('training_admission') is not False
            or batch.get('census')!=census_ref or census.get('schema')!='natlang.acquired-render-pending-census/1'):
        raise ValueError('repair batch and exact pending census differ')
    sources={}
    for item in census['receipts']:
        receipt=json.loads(checked(item).read_text());original_ref=receipt['converter_receipt'];original=json.loads(checked(original_ref).read_text())
        source=receipt['source']
        if (original.get('source')!=source or source['path'] in sources
                or original.get('explicit_history_reasoning_policy') not in (None,'keep')):
            raise ValueError('original source history profile is absent, ambiguous or mismatched')
        sources[source['path']]={'profile':original.get('explicit_history_reasoning_policy'),'converter_receipt':original_ref}
    profiles={};exports=[];bindings=[];counts=Counter()
    for view in batch['views']:
        path=checked(view);export=json.loads(path.read_text());contract=(export.get('contract') or {}).get('source_syntax_repair') or {}
        parent=contract.get('parent_source') or {};source=sources.get(parent.get('path'))
        if source is None or parent.get('sha256')!=next(x['source']['sha256'] for x in census['receipts'] if x['source']['path']==parent['path']):
            raise ValueError('derived source does not bind its original conversion')
        profile=source['profile'];exports.append(str(path));counts[profile or 'default']+=1
        if profile is not None:profiles[str(path)]=profile
        bindings.append({'derived_export':{'path':str(path),'sha256':view['sha256']},
            'parent_source':parent,'original_converter_receipt':source['converter_receipt'],'history_reasoning_policy':profile})
    if len(exports)!=len(set(exports)):raise ValueError('repair batch repeats an export')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    export_path=out/'exports.json';write_json_durable(export_path,{'exports':exports,'history_reasoning_policy':profiles})
    result={'schema':'natlang.acquired-syntax-repair-history-profiles/1','training_admission':False,
        'batch':batch_ref,'census':census_ref,'counts':dict(counts),'bindings':bindings,
        'exports':{'path':str(export_path),'sha256':sha256_file_hex(export_path)},
        'builder':{'path':str(Path(__file__).resolve()),'sha256':sha256_file_hex(__file__)},
        'scope':'Exact original per-source render policy; no graph conversion or training admission.'}
    write_json_durable(out/'receipt.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('batch','batch-sha256','census','census-sha256','out'):p.add_argument('--'+flag,required=True)
    a=p.parse_args();result=build({'path':a.batch,'sha256':a.batch_sha256},{'path':a.census,'sha256':a.census_sha256},a.out)
    print(json.dumps(result['counts'],sort_keys=True))
if __name__=='__main__':main()
