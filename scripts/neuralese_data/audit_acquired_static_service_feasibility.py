#!/usr/bin/env python3
"""Inspect whether one original session can be represented by recorded service observations.

This is not a native trajectory, an execution proof, or training admission.
"""
import argparse
from collections import Counter
import json
import re
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable
from scripts.neuralese_data.build_acquired_streaming_successor import bind,checked

CODE=Path(__file__).read_bytes()


def inspect_session(row):
    trajectory=row['trajectory']; tools=trajectory.get('tools') or []; definitions={}; errors=[]
    for item in tools:
        function=item.get('function') or {};name=function.get('name')
        if item.get('type')!='function' or not isinstance(name,str) or name in definitions:
            errors.append('missing, duplicate or non-function tool declaration');continue
        definitions[name]=function
    pending={}; calls=[]; observations=[]; seen=set(); names=Counter()
    for index,message in enumerate(trajectory['messages']):
        for call in message.get('tool_calls') or []:
            function=call.get('function') or {};name=function.get('name');ident=call.get('id');args=function.get('arguments')
            try:
                args=json.loads(args) if isinstance(args,str) else args
            except ValueError:
                errors.append(f'message {index}: invalid JSON arguments');args=None
            if message.get('role')!='assistant' or not isinstance(ident,str) or not ident or ident in seen:
                errors.append(f'message {index}: missing or repeated assistant call identity');continue
            if name not in definitions:errors.append(f'message {index}: call has no declared tool schema')
            if not isinstance(args,dict):errors.append(f'message {index}: arguments are not an object')
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z_$][\w$]*\.[A-Za-z_$][\w$]*',name):
                errors.append(f'message {index}: tool name is not a two-part TypeScript service method')
            record={'call_id':ident,'name':name,'arguments':args,'message_index':index}
            pending[ident]=record;calls.append(record);seen.add(ident);names[name or '<missing>']+=1
        if message.get('role')=='tool':
            ident=message.get('tool_call_id');call=pending.pop(ident,None)
            if call is None:
                errors.append(f'message {index}: tool observation lacks a preceding unique call');continue
            if message.get('name') not in (None,call['name']):
                errors.append(f'message {index}: tool observation names another method')
            content=message.get('content')
            if not isinstance(content,str):errors.append(f'message {index}: tool observation is not a string')
            observations.append({'call_id':ident,'name':call['name'],'arguments':call['arguments'],
                'result':content,'call_message_index':call['message_index'],'result_message_index':index})
    if pending:errors.append('one or more calls lack a recorded observation')
    ids=[item['call_id'] for item in observations]
    if ids != [item['call_id'] for item in calls]:
        errors.append('recorded observation order differs from call order; parallel effects need explicit custody')
    repeated=Counter(canonical_json_sha256_hex([item['name'],item['arguments']]) for item in calls)
    outcomes={}
    for item in observations:
        key=canonical_json_sha256_hex([item['name'],item['arguments']])
        outcomes.setdefault(key,set()).add(canonical_json_sha256_hex(item['result']))
    varied=sum(len(values)>1 for values in outcomes.values())
    return {'source_id':trajectory['id'],'normalized_session_sha256':canonical_json_sha256_hex(row),
        'messages':len(trajectory['messages']),'declared_tool_names':sorted(definitions),
        'tool_calls':len(calls),'tool_observations':len(observations),'tool_call_names':dict(names),
        'repeated_identical_calls':sum(n-1 for n in repeated.values() if n>1),
        'repeated_call_keys_with_different_observations':varied,
        'structural_pairing_errors':errors,'source_outcome':trajectory.get('outcome'),
        'schema_check':'not yet performed; source declarations are preserved, not assumed correct',
        'requirements':['Validate the complete declared JSON schemas and exact source arguments.',
            'Preserve call order and per-call original observation; reject unrecorded calls and state changes.',
            'Bind or explicitly report all referenced external files, original source and tool observations.',
            'Construct and replay actual Natlang eval/service graphs and validate every resulting observation.',
            'Separately qualify the task outcome and native learning objective before admission.'],
        'replay_contract':{'schema':'natlang.source-observation-replay-contract/1',
            'matching':'ordered-exact','allow_new_effects':False,
            'source_session_sha256':canonical_json_sha256_hex(row)},
        'source_observations_not_native_execution':True,'native_graph_constructed':False,
        'training_admission':False,'observations':observations,'tool_declarations':tools}


def audit(converter_receipt, output, *, row_index=0):
    converter_path=Path(converter_receipt).resolve(strict=True);converter_ref=bind(converter_path)
    converter=json.loads(converter_path.read_text()); source_ref=converter['source']
    source=checked(source_ref,'original normalized source')
    if type(row_index) is not int or row_index<0:raise ValueError('row index must be nonnegative')
    with open_jsonl_text(source) as stream:
        for number,line in enumerate(stream):
            if number==row_index:row=json.loads(line);break
        else:raise ValueError('requested source row is absent')
    result=inspect_session(row);result.update({'schema':'natlang.acquired-static-service-feasibility/1',
        'source':source_ref,'source_row_index':row_index,'converter_receipt':converter_ref,'builder':bind(Path(__file__)),
        'scope':'Source-derived feasibility review; recorded tool observations are not newly executed Natlang effects.'})
    if bind(converter_path)!=converter_ref or bind(source)['sha256']!=source_ref['sha256'] or Path(__file__).read_bytes()!=CODE:
        raise ValueError('source controls changed during feasibility review')
    output=Path(output)
    if output.exists():raise ValueError('feasibility review must not overwrite an artifact')
    output.parent.mkdir(parents=True,exist_ok=True);write_json_durable(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--converter-receipt',required=True)
    p.add_argument('--row-index',type=int,default=0);p.add_argument('--out',required=True)
    a=p.parse_args();result=audit(a.converter_receipt,a.out,row_index=a.row_index)
    print(json.dumps({k:result[k] for k in ('source_id','tool_calls','tool_observations','tool_call_names',
        'repeated_identical_calls','repeated_call_keys_with_different_observations','structural_pairing_errors','training_admission')},sort_keys=True))


if __name__=='__main__':main()
