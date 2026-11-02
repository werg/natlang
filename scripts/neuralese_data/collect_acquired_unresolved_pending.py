#!/usr/bin/env python3
"""Retain exact source-bound render omissions outside the strict repair set."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable


def checked(ref):
    path=Path(ref['path']).resolve(strict=True)
    if sha256_file_hex(path)!=ref['sha256']:raise ValueError('pending source evidence changed')
    return path


def collect(census_ref,out):
    census_path=checked(census_ref);census=json.loads(census_path.read_text())
    if census.get('schema')!='natlang.acquired-render-pending-census/1' or census.get('training_admission') is not False:
        raise ValueError('unresolved inventory requires a complete source audit')
    output=Path(out);output.mkdir(parents=True,exist_ok=False)
    rows=[];counts=Counter();sources=[]
    for item in census['receipts']:
        audit_path=checked(item);audit=json.loads(audit_path.read_text())
        candidates_path=audit_path.parent/'candidates.json'
        if sha256_file_hex(candidates_path)!=audit['candidate_sha256']:raise ValueError('candidate inventory changed')
        candidates=json.loads(candidates_path.read_text());selected={row['id'] for row in candidates['rows']}
        if len(selected)!=len(candidates['rows']):raise ValueError('candidate inventory repeats source identity')
        converter_path=checked(audit['converter_receipt']);converter=json.loads(converter_path.read_text())
        omission_path=converter_path.parent/'omissions.jsonl'
        if sha256_file_hex(omission_path)!=converter['omissions_sha256']:raise ValueError('original omission ledger changed')
        found=set();argument=unrepresentable=0
        with omission_path.open() as stream:
            for line in stream:
                if not line.strip():continue
                original=json.loads(line);reason=original.get('reason')
                if reason not in ('source-render-pending:tool-arguments','source-render-pending:tool-call-unrepresentable'):continue
                ident=original['id']
                if ident in found:raise ValueError('original omission ledger repeats an identity')
                found.add(ident)
                if reason=='source-render-pending:tool-arguments' and ident in selected:continue
                if reason=='source-render-pending:tool-arguments':
                    unresolved='ambiguous-or-single-template-render-failure';argument+=1
                else:unresolved='unrepresentable-original-tool-name';unrepresentable+=1
                rows.append({'original_omission':original,'source':audit['source'],'converter_receipt':audit['converter_receipt'],
                             'reason':unresolved,'training_admission':False})
                counts[unresolved]+=1
        if not selected<=found or argument!=audit['counts'].get('ambiguous_or_unsupported',0)+audit['counts'].get('not_both_json_failures',0) or unrepresentable!=audit['counts'].get('unrepresentable_rows',0):
            raise ValueError('pending source obligation accounting differs from audit')
        sources.append({'audit':item,'omissions':{'path':str(omission_path),'sha256':converter['omissions_sha256']},
                        'unresolved_arguments':argument,'unrepresentable':unrepresentable})
    if (counts['ambiguous-or-single-template-render-failure']!=census['counts'].get('ambiguous_or_unsupported',0)+census['counts'].get('not_both_json_failures',0)
            or counts['unrepresentable-original-tool-name']!=census['counts'].get('unrepresentable_rows',0)):
        raise ValueError('unresolved source obligations differ from the complete census')
    ledger=output/'unresolved.jsonl'
    with ledger.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    result={'schema':'natlang.acquired-unresolved-render-obligations/1','training_admission':False,'census':census_ref,
        'counts':dict(counts),'rows':len(rows),'ledger':{'path':str(ledger),'sha256':sha256_file_hex(ledger)},
        'sources':sources,'scope':'Exact original omissions retained for later review; no fabricated tool calls or outcomes.'}
    write_json_durable(output/'receipt.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--census',required=True);p.add_argument('--census-sha256',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(collect({'path':args.census,'sha256':args.census_sha256},args.out)['counts'],sort_keys=True))
if __name__=='__main__':main()
