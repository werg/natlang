#!/usr/bin/env python3
"""Resume an exact, one-pass-per-source audit of unresolved acquired sessions."""
from __future__ import annotations
import argparse, copy, hashlib, io, json, os, shutil
from collections import Counter, defaultdict
from pathlib import Path
import zstandard
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable, write_jsonl_text
from natlang_neuralese.serve.chat import RequestError
from scripts.neuralese_data.acquired_session_syntax_repair import repair_unquoted_query_argument
from scripts.neuralese_data.build_acquired_full_session_graphs import prevalidate_session_tool_arguments

MODULE=Path(__file__).read_bytes()

def bind(path):
    path=Path(path).resolve(strict=True);return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}

def checked(ref):
    got=bind(ref['path'])
    if got['sha256']!=ref['sha256'] or ('bytes' in ref and ref['bytes']!=got['bytes']):raise ValueError('bound evidence changed: '+ref['path'])
    return Path(got['path'])

def signature(path):
    s=path.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

class HashReader(io.RawIOBase):
    def __init__(self,raw):self.raw=raw;self.digest=hashlib.sha256()
    def readable(self):return True
    def readinto(self,buf):
        n=self.raw.readinto(buf)
        if n:self.digest.update(memoryview(buf)[:n])
        return n

def classify(value):
    trajectory=value['trajectory']; schemas={};duplicates=[]
    for tool in trajectory.get('tools') or []:
        fn=tool.get('function',tool);name=fn.get('name')
        if name in schemas:duplicates.append(name)
        schemas[name]=fn.get('parameters')
    errors=[];repairs=[]
    for mi,message in enumerate(trajectory['messages']):
        for ci,call in enumerate(message.get('tool_calls') or []):
            try:prevalidate_session_tool_arguments({'trajectory':{'messages':[{**message,'tool_calls':[call]}]}},template_backbone='lfm3');continue
            except RequestError as exc:
                fn=call.get('function',call);name=fn.get('name');arg=fn.get('arguments');schema=schemas.get(name)
                raw=arg if isinstance(arg,str) else json.dumps(arg,ensure_ascii=False,sort_keys=True)
                record={'message_index':mi,'call_index':ci,'tool_name':name,'arguments_type':type(arg).__name__,
                        'arguments_sha256':hashlib.sha256(raw.encode()).hexdigest(),'error_code':exc.code,'detail':str(exc)}
                if exc.code=='tool-arguments' and name=='browser.search' and not duplicates:
                    fixed=repair_unquoted_query_argument(arg,schema=schema)
                    if fixed is not None:
                        record['repaired_arguments']=fixed;record['repaired_arguments_sha256']=hashlib.sha256(fixed.encode()).hexdigest();repairs.append(record)
                errors.append(record)
    try:prevalidate_session_tool_arguments(value,template_backbone='lfm3');whole_error=None
    except RequestError as exc:whole_error={'code':exc.code,'detail':str(exc)}
    if whole_error is None:
        disposition='renderable-original-review-required';reason='current LFM3 validation accepts the unchanged source; original omission remains historical'
    elif not duplicates and errors and len(repairs)==len(errors):
        repaired=copy.deepcopy(value)
        for item in repairs:
            call=repaired['trajectory']['messages'][item['message_index']]['tool_calls'][item['call_index']];call.get('function',call)['arguments']=item['repaired_arguments']
        try:prevalidate_session_tool_arguments(repaired,template_backbone='lfm3')
        except RequestError as exc:
            disposition='quality-excluded';reason='strict source-bound syntax repair leaves LFM3 validation failure: '+str(exc)
        else:disposition='repairable-source-syntax-review-required';reason='all source failures have deterministic schema-bound syntax repairs; no admission or success inferred'
    else:
        disposition='quality-excluded';reason=('ambiguous duplicate declared tool names' if duplicates else 'one or more original calls lack a lossless representable LFM3 syntax repair')
    return {'disposition':disposition,'reason':reason,'errors':errors,'whole_validation':whole_error,'duplicate_tool_names':duplicates,
            'source_outcome':trajectory.get('outcome'),'training_admission':False,'tool_execution_verified':False}

def inputs(ledger_path,extra_path):
    ledger_path=Path(ledger_path).resolve(strict=True);summary=json.loads(ledger_path.read_bytes());checked(summary['ledger'])
    if summary.get('schema')!='natlang.acquired-unresolved-render-obligations/1' or summary.get('training_admission') is not False:raise ValueError('unexpected unresolved ledger')
    grouped=defaultdict(dict);sources={}
    for line in Path(summary['ledger']['path']).open():
        row=json.loads(line);ident=row['original_omission']['id'];ref=row['source'];key=(ref['path'],ref['sha256'])
        if ident in grouped[key]:raise ValueError('repeated unresolved occurrence')
        grouped[key][ident]=row['original_omission'];sources[key]=ref
    extra_path=Path(extra_path).resolve(strict=True);extra=json.loads(extra_path.read_bytes())
    if extra.get('schema')!='natlang.acquired-repair-still-pending/1' or extra.get('training_admission') is not False:raise ValueError('unexpected extra obligation')
    for k in ('original_converter','original_omission_ledger','derived_converter','derived_omission_ledger','original_audit','reproduced_sample'):checked(extra[k])
    ref=extra['original_audit']['source'];key=(ref['path'],ref['sha256']);ident=extra['original_omission']['id']
    if ident in grouped[key]:raise ValueError('extra obligation duplicates ledger')
    grouped[key][ident]=extra['original_omission'];sources[key]=ref
    if sum(map(len,grouped.values()))!=summary['rows']+1:raise ValueError('unresolved count mismatch')
    return grouped,sources,bind(ledger_path),bind(extra_path)

def build(ledger_path,extra_path,output):
    grouped,sources,ledger_ref,extra_ref=inputs(ledger_path,extra_path);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    builder=out/('builder-'+hashlib.sha256(MODULE).hexdigest()+'.py')
    if builder.exists():
        if builder.read_bytes()!=MODULE:raise ValueError('sealed builder differs')
    else:builder.write_bytes(MODULE)
    all_counts=Counter();results=[]
    for index,key in enumerate(sorted(grouped)):
        ref=sources[key];wanted=grouped[key];tag=hashlib.sha256((key[0]+'\0'+key[1]).encode()).hexdigest();directory=out/tag;receipt_path=directory/'receipt.json'
        if receipt_path.exists():
            prior=json.loads(receipt_path.read_bytes());checked(prior['sessions']);checked(prior['classification'])
            if not Path(ref['path']).is_file():
                all_counts['missing-source-recovery-required']+=len(wanted);results.append({'source':ref,'captured_receipt':bind(receipt_path),'missing_rows':len(wanted),'recovery_dependency':'restore exact normalized source bytes with recorded SHA256'});continue
            if prior['source']!=ref or prior['wanted_sha256']!=canonical_json_sha256_hex(wanted) or prior['builder_sha256']!=hashlib.sha256(MODULE).hexdigest() or signature(Path(ref['path']))!=prior['source_signature']:
                raise ValueError('resumed completed source identity differs')
            results.append(bind(receipt_path));all_counts.update(prior['counts']);continue
        if directory.exists():
            incomplete=out/(tag+'.incomplete-'+str(os.getpid()))
            if incomplete.exists():raise ValueError('incomplete capture preservation path already exists')
            os.replace(directory,incomplete)
        source=Path(ref['path'])
        if not source.is_file():
            all_counts['missing-source-recovery-required']+=len(wanted);results.append({'source':ref,'missing_rows':len(wanted),'recovery_dependency':'restore exact normalized source bytes with recorded SHA256'});continue
        tmp=out/('.'+tag+'.partial');
        if tmp.exists():shutil.rmtree(tmp)
        tmp.mkdir();before=signature(source);seen=set();counts=Counter();classifications=[]
        with source.open('rb',buffering=0) as raw:
            hashed=HashReader(raw)
            with zstandard.ZstdDecompressor().stream_reader(hashed,read_across_frames=True,closefd=False) as decoded, io.TextIOWrapper(decoded,encoding='utf-8') as text, write_jsonl_text(tmp/'sessions.jsonl.zst') as selected:
                for line in text:
                    if not line.strip():continue
                    value=json.loads(line);ident=value['trajectory']['id']
                    if ident not in wanted:continue
                    if ident in seen or canonical_json_sha256_hex(value)!=wanted[ident]['normalized_source_sha256']:raise ValueError('source occurrence missing identity or changed')
                    seen.add(ident);decision=classify(value);decision.update({'id':ident,'source_row':value['source_row'],'source_file':value['source_file'],'normalized_source_sha256':wanted[ident]['normalized_source_sha256'],'original_omission':wanted[ident]})
                    classifications.append(decision);counts[decision['disposition']]+=1
                    selected.write(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
            digest=hashed.digest.hexdigest()
        if digest!=ref['sha256'] or signature(source)!=before or seen!=set(wanted):raise ValueError('source stream digest, signature or occurrence count changed')
        write_json_durable(tmp/'classification.json',{'schema':'natlang.acquired-original-render-classification/1','rows':classifications})
        os.replace(tmp,directory)
        result={'schema':'natlang.acquired-original-render-source-capture/1','source':ref,'source_signature':before,'wanted_sha256':canonical_json_sha256_hex(wanted),'rows':len(seen),'counts':dict(counts),'sessions':bind(directory/'sessions.jsonl.zst'),'classification':bind(directory/'classification.json'),'builder_sha256':hashlib.sha256(MODULE).hexdigest(),'scope':'Full original selected sessions; one compressed-source pass with streaming SHA; no source outcome verification.'}
        write_json_durable(receipt_path,result);results.append(bind(receipt_path));all_counts.update(counts)
        print(json.dumps({'source_index':index+1,'sources':len(grouped),'rows':len(seen),'counts':dict(counts)}),flush=True)
    if sum(all_counts.values())!=sum(map(len,grouped.values())):raise ValueError('classification accounting differs')
    if Path(__file__).read_bytes()!=MODULE or bind(ledger_path)!=ledger_ref or bind(extra_path)!=extra_ref:raise ValueError('audit code or controls changed')
    final={'schema':'natlang.acquired-original-render-resolution/1','ledger':ledger_ref,'extra_obligation':extra_ref,'builder':bind(builder),'sources':results,'rows':sum(all_counts.values()),'counts':dict(all_counts),'training_admission':False,'scope':'Exact source-captured classification; quality exclusions preserve originals and do not assert source outcomes.'}
    write_json_durable(out/'receipt.json',final);print(json.dumps({'rows':final['rows'],'counts':final['counts']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ledger',required=True);p.add_argument('--extra',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.ledger,a.extra,a.output)
