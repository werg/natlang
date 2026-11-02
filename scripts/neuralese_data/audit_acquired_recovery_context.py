#!/usr/bin/env python3
"""Inspect adjacent original messages for unrepresentable tool-call omissions."""
import argparse
import json
from pathlib import Path
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable


def audit(sample_path,converter_path,out):
    sample_path=Path(sample_path).resolve(strict=True);converter_path=Path(converter_path).resolve(strict=True)
    sample=json.loads(sample_path.read_text());converter=json.loads(converter_path.read_text())
    selected={x['id']:x for x in sample['sampled'] if x['reason']=='source-render-pending:tool-call-unrepresentable'}
    source=Path(converter['source']['path']).resolve(strict=True);results=[];seen=set()
    def excerpt(message):
        value=message.get('content');return {'role':message.get('role'),'tool_call_id':message.get('tool_call_id'),
            'name':message.get('name'),'content_type':type(value).__name__,
            'content_excerpt':str(value)[:500],'parts_excerpt':str(message.get('parts'))[:700],
            'text_excerpt':str(message.get('text'))[:700],'reasoning_excerpt':str(message.get('reasoning'))[:700],
            'keys':sorted(message),
            'tool_call_count':len(message.get('tool_calls') or [])}
    with open_jsonl_text(source) as stream:
        for line in stream:
            if not line.strip():continue
            value=json.loads(line);ident=value['trajectory']['id']
            if ident not in selected:continue
            prior=selected[ident]
            if canonical_json_sha256_hex(value)!=prior['normalized_source_sha256']:raise ValueError('sample source changed')
            seen.add(ident);messages=value['trajectory']['messages'];windows=[]
            for call in prior['calls']:
                index=call['message_index'];windows.append({'message_index':index,'invalid_name':call['name'],
                    'previous':excerpt(messages[index-1]) if index else None,
                    'next':excerpt(messages[index+1]) if index+1<len(messages) else None,
                    'following':excerpt(messages[index+2]) if index+2<len(messages) else None})
            results.append({'id':ident,'normalized_source_sha256':prior['normalized_source_sha256'],
                'source_outcome':value['trajectory'].get('outcome'),'windows':windows,
                'status':'original context only; no tool execution, repair or outcome verification'})
            if seen==set(selected):break
    if seen!=set(selected):raise ValueError('source omitted selected pending samples')
    result={'schema':'natlang.acquired-unrepresentable-recovery-context-audit/1','training_admission':False,
        'sample':{'path':str(sample_path),'sha256':sha256_file_hex(sample_path)},
        'converter':{'path':str(converter_path),'sha256':sha256_file_hex(converter_path)},
        'source':converter['source'],'rows':results,'scope':'Bounded original transcript observations; no inferred intended call or success.'}
    out=Path(out)
    if out.exists():raise ValueError('recovery-context audit output must be fresh')
    write_json_durable(out,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sample',required=True);p.add_argument('--converter',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps({'rows':len(audit(a.sample,a.converter,a.out)['rows'])}))
if __name__=='__main__':main()
