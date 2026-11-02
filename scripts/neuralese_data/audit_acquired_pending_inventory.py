#!/usr/bin/env python3
"""One bounded pass over each normalized source with unresolved render omissions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.extract_acquired_syntax_repair_candidates import extract

SOURCE_BYTES=Path(__file__).read_bytes()

def audit(summary_path,out):
    summary_path=Path(summary_path).resolve(strict=True);before=summary_path.stat()
    data=json.loads(summary_path.read_text())
    if data.get('schema')!='natlang.acquired-session-student-conversion/1' or data.get('incomplete'):
        raise ValueError('requires the complete current converter inventory')
    receipts={}
    for output in data['by_source']['D32']['outputs']:
        path=Path(output['path'])
        if '/source-syntax-repair-' in str(path):continue
        rp=(path.parent.parent/'receipt.json').resolve(strict=True)
        receipt=json.loads(rp.read_text());counts=receipt.get('counts') or {}
        if any(counts.get('source-render-pending:'+reason,0) for reason in ('tool-arguments','tool-call-unrepresentable')):
            receipts[str(rp)]=receipt
    out=Path(out);out.mkdir(parents=True,exist_ok=False);totals=Counter();names=Counter();items=[]
    for index,(path,receipt) in enumerate(sorted(receipts.items())):
        key=hashlib.sha256(path.encode()).hexdigest()[:20]
        result=extract(path,out/key)
        observed=result['counts'];expected=receipt['counts']
        if observed.get('tool_argument_rows',0)!=expected.get('source-render-pending:tool-arguments',0) or observed.get('unrepresentable_rows',0)!=expected.get('source-render-pending:tool-call-unrepresentable',0):
            raise ValueError('source pending census differs from conversion receipt')
        totals.update(observed);names.update(result['unrepresentable_tool_name_rows'])
        proof=out/key/'receipt.json';items.append({'path':str(proof),'sha256':sha256_file_hex(proof),
            'source':result['source'],'counts':observed})
        print(json.dumps({'kind':'acquired_pending_audit_progress','files':index+1,'of':len(receipts),'pending':totals['pending_rows']}),flush=True)
    for reason,key in (('source-render-pending:tool-arguments','tool_argument_rows'),('source-render-pending:tool-call-unrepresentable','unrepresentable_rows')):
        if data['totals'].get(reason,0)!=totals[key]:raise ValueError('pending corpus counts differ from complete conversion summary')
    after=summary_path.stat();signature=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    if signature(before)!=signature(after) or Path(__file__).read_bytes()!=SOURCE_BYTES:raise ValueError('audit controls changed while running')
    result={'schema':'natlang.acquired-render-pending-census/1','training_admission':False,
        'summary':{'path':str(summary_path),'sha256':sha256_file_hex(summary_path)},'counts':dict(totals),
        'unrepresentable_tool_name_rows':dict(names),'receipts':items,
        'builder_sha256':hashlib.sha256(SOURCE_BYTES).hexdigest(),
        'scope':'Exact original omitted occurrences, one streamed pass per normalized source; no executed-outcome or capability claim.'}
    write_json_durable(out/'receipt.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(audit(args.summary,args.out)['counts'],sort_keys=True))

if __name__=='__main__':main()
