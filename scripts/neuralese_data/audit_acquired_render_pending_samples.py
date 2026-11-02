#!/usr/bin/env python3
"""Inspect exact omitted source sessions without generating replacement targets."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.serve.chat import RequestError
from scripts.neuralese_data.build_acquired_full_session_graphs import prevalidate_session_tool_arguments


def audit(receipt_paths, output, limit=3):
    all_results=[];counts=Counter()
    for value in receipt_paths:
        receipt_path=Path(value).resolve(strict=True);receipt=json.loads(receipt_path.read_text())
        omission_path=receipt_path.parent/'omissions.jsonl'
        if sha256_file_hex(omission_path)!=receipt['omissions_sha256']:
            raise ValueError('pending omission ledger differs from its converter receipt')
        rows=[json.loads(line) for line in omission_path.open() if line.strip()]
        chosen={};byid={}
        for row in rows:
            reason=row.get('reason','')
            if not reason.startswith('source-render-pending:'):continue
            counts[reason]+=1
            if len(chosen.setdefault(reason,[]))<limit:
                chosen[reason].append(row['id']);byid[row['id']]=row
        if not byid:continue
        source=Path(receipt['source']['path']).resolve(strict=True)
        # The converter's source SHA remains recorded; exact selected rows are
        # independently bound by the original canonical row hashes below.
        before=source.stat();matched=set()
        with open_jsonl_text(source) as stream:
            for line in stream:
                if not line.strip():continue
                value=json.loads(line);trajectory=value['trajectory'];ident=trajectory['id']
                if ident not in byid:continue
                omission=byid[ident]
                if canonical_json_sha256_hex(value)!=omission['normalized_source_sha256']:
                    raise ValueError('pending source occurrence differs from its omission record')
                matched.add(ident);calls=[]
                for index,message in enumerate(trajectory['messages']):
                    for call_index,call in enumerate(message.get('tool_calls') or []):
                        function=call.get('function',call);arg=function.get('arguments');name=function.get('name')
                        try:prevalidate_session_tool_arguments({'trajectory':{'messages':[{**message,'tool_calls':[call]}]}},template_backbone='lfm3');failure=None
                        except RequestError as error:failure={'code':error.code,'detail':str(error)[:300]}
                        if failure:
                            raw=arg if isinstance(arg,str) else json.dumps(arg,ensure_ascii=False,sort_keys=True)
                            calls.append({'message_index':index,'call_index':call_index,'name':name,'argument_type':type(arg).__name__,
                                'argument_sha256':hashlib.sha256(raw.encode()).hexdigest(),'argument_excerpt':raw[:600],
                                'error':failure})
                all_results.append({'id':ident,'source_row':value['source_row'],'reason':omission['reason'],
                    'source_sha256':receipt['source']['sha256'],'normalized_source_sha256':omission['normalized_source_sha256'],
                    'source_outcome':trajectory.get('outcome'),'calls':calls,
                    'scope':'original source values; no repair, execution, or success assertion'})
                if matched==set(byid):break
        after=source.stat();signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if matched!=set(byid) or signature(before)!=signature(after):raise ValueError('sampled source occurrences missing or changed')
    result={'schema':'natlang.acquired-render-pending-sample-audit/1','counts_in_selected_receipts':dict(counts),
            'sampled':all_results,'limitations':['Not a census of unexamined source rows.','Positive labels remain source-reported, not verified outcomes.']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('audit output must be fresh')
    write_json_durable(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',action='append',required=True);p.add_argument('--out',required=True);p.add_argument('--per-reason',type=int,default=3)
    args=p.parse_args();result=audit(args.receipt,args.out,args.per_reason)
    print(json.dumps({'counts':result['counts_in_selected_receipts'],'sampled':len(result['sampled'])},sort_keys=True))

if __name__=='__main__':main()
