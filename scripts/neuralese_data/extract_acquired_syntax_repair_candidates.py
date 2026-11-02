#!/usr/bin/env python3
"""Extract narrowly repairable exact pending occurrences for the existing repair builder."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.serve.chat import RequestError, normalize_tool_call_arguments
from scripts.neuralese_data.acquired_session_syntax_repair import repair_unquoted_query_argument
from scripts.neuralese_data.build_acquired_full_session_graphs import prevalidate_session_tool_arguments

MODULE_SOURCE_BYTES=Path(__file__).read_bytes()


def extract(receipt_path, output):
    receipt_path=Path(receipt_path).resolve(strict=True);receipt=json.loads(receipt_path.read_text())
    omissions_path=receipt_path.parent/'omissions.jsonl'
    if sha256_file_hex(omissions_path)!=receipt['omissions_sha256']:raise ValueError('omission ledger changed')
    pending={row['id']:row for line in omissions_path.open() if line.strip() for row in [json.loads(line)]
             if row.get('reason') in ('source-render-pending:tool-arguments','source-render-pending:tool-call-unrepresentable')}
    source=Path(receipt['source']['path']).resolve(strict=True);actual=sha256_file_hex(source)
    if actual!=receipt['source']['sha256']:raise ValueError('normalized source differs from conversion receipt')
    source_ref={'path':str(source),'sha256':actual};counts=Counter();candidates=[];audits=[];seen=set();unrepresentable=Counter()
    with open_jsonl_text(source) as stream:
        for line in stream:
            if not line.strip():continue
            value=json.loads(line);trajectory=value['trajectory'];ident=trajectory['id']
            if ident not in pending:continue
            if ident in seen or canonical_json_sha256_hex(value)!=pending[ident]['normalized_source_sha256']:
                raise ValueError('pending source occurrence repeated or changed')
            seen.add(ident);counts['pending_rows']+=1
            if pending[ident]['reason']=='source-render-pending:tool-call-unrepresentable':
                counts['unrepresentable_rows']+=1;names=[]
                for message in trajectory['messages']:
                    for call in message.get('tool_calls') or []:
                        try:prevalidate_session_tool_arguments({'trajectory':{'messages':[{**message,'tool_calls':[call]}]}},template_backbone='lfm3')
                        except RequestError as error:
                            if error.code=='tool-call-unrepresentable':names.append(str(call.get('function',call).get('name')))
                if not names:raise ValueError('unrepresentable omission cannot be reproduced')
                unrepresentable.update(set(names));continue
            counts['tool_argument_rows']+=1;failures={}
            for backbone in ('lfm2','mellum'):
                try:prevalidate_session_tool_arguments(value,template_backbone=backbone)
                except RequestError as error:failures[backbone]=str(error)
            if set(failures)!={'lfm2','mellum'} or any(not x.startswith('tool-arguments:') for x in failures.values()):
                counts['not_both_json_failures']+=1;continue
            schemas={}
            for tool in trajectory.get('tools') or []:
                function=tool.get('function',tool);name=function.get('name')
                if name in schemas:raise ValueError('source has ambiguous duplicate tool declarations')
                schemas[name]=function.get('parameters')
            invalid=[];candidate_calls=[];safe=True
            for mi,message in enumerate(trajectory['messages']):
                for ci,call in enumerate(message.get('tool_calls') or []):
                    isolated={**message,'tool_calls':[call]}
                    try:normalize_tool_call_arguments([isolated]);continue
                    except RequestError as error:
                        if error.code!='tool-arguments':raise
                    function=call.get('function',call);name=function.get('name');argument=function.get('arguments');schema=schemas.get(name)
                    if name!='browser.search' or not isinstance(argument,str) or repair_unquoted_query_argument(argument,schema=schema) is None:
                        safe=False;break
                    invalid.append({'message_index':mi,'call_index':ci,'decoded_type':'invalid-json'})
                    candidate_calls.append({'message_index':mi,'call_index':ci,'name':name,'schema':schema,'arguments':argument,
                        'arguments_utf8_sha256':hashlib.sha256(argument.encode()).hexdigest()})
                if not safe:break
            if not safe or not invalid:
                counts['ambiguous_or_unsupported']+=1;continue
            # A source-only syntax repair cannot mask an unrepresentable call.
            candidate={'id':ident,'source_row':value['source_row'],'source_file':value['source_file'],
                'normalized_source_sha256':pending[ident]['normalized_source_sha256'],'calls':candidate_calls}
            candidates.append(candidate);audits.append({'id':ident,'failures':failures,'calls':invalid})
            counts['fully_bounded_candidates']+=1;counts['candidate_calls']+=len(invalid)
    if seen!=set(pending):raise ValueError('normalized source omitted a recorded pending row')
    if sha256_file_hex(source)!=actual:raise ValueError('normalized source changed during audit')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    candidate={'schema':'natlang.source16-tool-repair-candidate-extract/1','source':source_ref,'rows':candidates}
    audit={'schema':'natlang.source-tool-pending-audit/1','source':source_ref,'rows':audits}
    write_json_durable(out/'candidates.json',candidate);write_json_durable(out/'pending-audit.json',audit)
    if Path(__file__).read_bytes()!=MODULE_SOURCE_BYTES:raise ValueError('candidate extractor changed during execution')
    builder=out/('extractor-'+hashlib.sha256(MODULE_SOURCE_BYTES).hexdigest()+'.py')
    builder.write_bytes(MODULE_SOURCE_BYTES)
    summary={'schema':'natlang.acquired-syntax-candidate-census/1','training_admission':False,'source':source_ref,
        'converter_receipt':{'path':str(receipt_path),'sha256':sha256_file_hex(receipt_path)},'counts':dict(counts),
        'builder':{'path':str(builder),'sha256':hashlib.sha256(MODULE_SOURCE_BYTES).hexdigest()},
        'unrepresentable_tool_name_rows':dict(unrepresentable),
        'candidate_sha256':sha256_file_hex(out/'candidates.json'),'audit_sha256':sha256_file_hex(out/'pending-audit.json'),
        'scope':'Only exact bounded browser.search JSON syntax candidates; no executed-success or semantic-quality claim.'}
    write_json_durable(out/'receipt.json',summary);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(extract(a.receipt,a.out)['counts'],sort_keys=True))

if __name__=='__main__':main()
