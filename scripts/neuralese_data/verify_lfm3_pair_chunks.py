#!/usr/bin/env python3
"""Verify source-bound LFM3 communication pairs into durable closed graph chunks.

Each chunk becomes independently usable only after every pair in it passes the
source execution, current protection, occurrence, renderer and graph checks.
A resume replays exact input identities and preserves only closed chunks.
"""
import argparse,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import canonical_json_str,iter_jsonl_rows,write_json_durable
from natlang_neuralese.common import line_backbones
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
from natlang_neuralese.model.shared_state import ROLE_MARKERS
from natlang_neuralese.serve.chat import build_message
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision,source_occurrences,writer_payload
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.verified_chunk_journal import verify_chunks
from scripts.source_review import pending_family_hold_reason


def digest(x):return hashlib.sha256(canonical_json_str(x).encode()).hexdigest()
def edge_digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_state(p):
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def binding(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':sha256_file_hex(p),'stat':file_state(p)}


def resolve_writes(x):
    if isinstance(x,dict):
        if set(x)=={'$write'} and isinstance(x['$write'],dict):
            d=x['$write'];return json.loads(d['source']) if d.get('type')=='Neuralese<unknown>' else d['source']
        return {k:resolve_writes(v) for k,v in x.items()}
    if isinstance(x,list):return [resolve_writes(v) for v in x]
    return x


def expected_calls(record):
    result=[]
    for c in (record.get('target') or {}).get('tool_calls') or []:
        f=c.get('function') or c;a=f['arguments'];result.append((f['name'],resolve_writes(json.loads(a) if isinstance(a,str) else a)))
    return result


def verify_pair(row, *, records, edges, tokenizer, model_id, fingerprint, roles, blocked, holds):
    if row.get('split')!='train' or set(row.get('models') or {})!={model_id}:raise ValueError('pair split/model differs')
    c=row['communication'];edge=edges.get(c['source_id'])
    if not edge or edge.get('writer_record_id')!=c.get('writer_record_id') or edge.get('reader_record_id')!=c.get('reader_record_id'):
        raise ValueError('pair lacks exact source edge')
    writer,reader=records[c['writer_record_id']],records[c['reader_record_id']]
    groups=set(row.get('source_groups') or [])
    if not groups or groups&blocked or {writer['id'],reader['id']}&holds:raise ValueError('source is protected')
    for record in (writer,reader):
        if record.get('split')!='train' or not set(record.get('source_groups') or [])<=groups:raise ValueError('record split/groups differ')
        admission=record.get('training_admission') or {};outcome=record.get('outcome') or {};trace=record.get('trace_admission') or {}
        if (admission.get('approved') is not True or outcome.get('accepted') is not True
                or outcome.get('status')!='done' or (outcome.get('oracle') or {}).get('accepted') is not True
                or trace.get('admitted') is not True):
            raise ValueError('record lacks accepted execution custody')
        program=(record.get('task') or {}).get('program_ir');prov=record.get('provenance') or {}
        if (not isinstance(program,dict) or digest(program)!=prov.get('program_ir_sha256') or pending_family_hold_reason(program)
                or not prov.get('model') or not prov.get('runtime') or not prov.get('trace_sha256')):
            raise ValueError('record program, actor, runtime or family custody invalid')
    body=writer_payload(writer,edge['write_name'],edge['body_sha256'])
    occurrence=source_occurrences(writer,reader,edge['write_name'],body,edge['body_sha256'],
        writer_occurrence_id=edge.get('writer_occurrence_id'),reader_occurrence_id=edge.get('reader_occurrence_id'))
    witness=json.loads(c['occurrence'])
    if (any(witness.get(k)!=v for k,v in occurrence.items()) or witness.get('edge_id')!=c['source_id']
            or witness.get('trajectory')!=edge.get('trajectory')
            or witness.get('writer_source_row_sha256')!=(writer.get('source_ref') or {}).get('source_row_sha256')
            or witness.get('reader_source_row_sha256')!=(reader.get('source_ref') or {}).get('source_row_sha256')):
        raise ValueError('source occurrence differs')
    view=row['models'][model_id];graph=view['graph'];parsed=parse_graph(graph,tokenizer_fingerprint=fingerprint)
    derivations=(graph.get('source_pairing') or {}).get('authenticated_static_derivations') or {}
    for key,record in (('writer',writer),('reader',reader)):
        witnesses=derivations.get(key) or []
        if not any(x.get('source_record_id')==record['id'] and x.get('source_record_sha256')==digest(record)
                   and x.get('original_target_sha256')==digest(record.get('target')) for x in witnesses):
            raise ValueError('graph lacks exact source derivation custody')
    if graph.get('split')!='train' or set(graph.get('source_groups') or [])!=groups or len(parsed.plan.invocations)!=2:
        raise ValueError('graph split/groups or node topology differs')
    producer,consumer=graph['nodes'];export=view['export'];lo,hi=export['start'],export['stop']
    if (export.get('node')!=producer['name'] or [lo,hi] not in producer.get('producing',[])
            or tokenizer.decode(producer['token_ids'][lo:hi],skip_special_tokens=False)!=body or export['text']!=body):
        raise ValueError('producer export differs from authenticated source')
    count=len(occurrence['reader_json_pointers'])
    imports=consumer.get('imports') or []
    if (consumer['token_ids'].count(roles[2])!=count or len(imports)!=count
            or any(i.get('source')!=producer['name'] or i.get('start')!=lo or i.get('stop')!=hi
                   or i.get('stop_gradient',False) for i in imports)):
        raise ValueError('consumer occurrence count differs')
    top=max(producer['target_weights']);first=next(i for i,w in enumerate(producer['target_weights']) if w==top)
    reply=tokenizer.decode([t for t in producer['token_ids'][first:] if t not in roles],skip_special_tokens=False)
    message=build_message(reply,[],tools=writer.get('tools'))
    calls=[(c['function']['name'],json.loads(c['function']['arguments'])) for c in message.get('tool_calls') or []]
    want=expected_calls(writer)
    if want:
        if calls!=want:raise ValueError('complete producer target differs from source')
    else:
        content=(writer.get('target') or {}).get('content')
        if not isinstance(content,str) or not content or content not in reply:raise ValueError('assistant producer target differs from source')
    return graph


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--protection',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--chunk-rows',type=int,default=128);a=p.parse_args();receipt=json.loads(a.receipt.read_text())
    inputs={k:binding(v['path']) for k,v in receipt['source'].items()}
    if any(inputs[k]['sha256']!=v['sha256'] for k,v in receipt['source'].items()):raise ValueError('source receipt input changed')
    for name,record in (('receipt',binding(a.receipt)),('pairs',binding(a.pairs))):inputs[name]=record
    declared=(receipt.get('outputs') or {}).get('pairs',{}).get('train') or {}
    if inputs['pairs']['sha256']!=declared.get('sha256') or Path(declared.get('path','')).resolve()!=a.pairs.resolve():
        raise ValueError('pair file is not the exact admitted train output in its conversion receipt')
    protection=current_protection(a.protection);blocked=protection['held']|protection['quality']|protection['pending']
    holds,hold_bindings=load_decision_holds(sorted((a.protection.parent/'decision-holds').glob('*.jsonl')))
    tok=AutoTokenizer.from_pretrained(str(a.tokenizer),local_files_only=True,trust_remote_code=False);roles=provision(tok);fp=tokenizer_fingerprint(tok)
    expected=line_backbones.backbone('lfm3');model_id=line_backbones.model_id('lfm3')
    if fp!=expected['expanded_fingerprint'] or list(roles)!=list(expected['role_token_ids']):raise ValueError('tokenizer differs')
    repo=Path(__file__).resolve().parents[2]
    code=[Path(__file__).resolve(),Path(__file__).with_name('verified_chunk_journal.py').resolve(),Path(__file__).with_name('build_native_semantic_pairs_e2e.py').resolve(),Path(__file__).with_name('build_verified_teacher_trajectory_graphs.py').resolve(),
          *(repo/x for x in ('scripts/source_review.py','scripts/neuralese_data/decision_holds.py',
            'training/neuralese/natlang_neuralese/data/native_graph_schema.py','training/neuralese/natlang_neuralese/model/native_graph_plan.py',
            'training/neuralese/natlang_neuralese/data/text_corpus.py','training/neuralese/natlang_neuralese/serve/chat.py',
            'training/neuralese/natlang_neuralese/common/jsonio.py'))]
    protection_manifest=json.loads(a.protection.read_text())
    protection_files=[binding(a.protection.parent/x['path']) for x in protection_manifest['files']]
    for item,expected_file in zip(protection_files,protection_manifest['files'],strict=True):
        if item['sha256']!=expected_file['sha256']:raise ValueError('protection dependency changed')
    tokenizer_files=[binding(a.tokenizer/x) for x in ('tokenizer.json','tokenizer_config.json','chat_template.jinja') if (a.tokenizer/x).is_file()]
    review_names=('holds.jsonl','family-holds.json','datasets.json')
    source_reviews=[binding(root/'training/source-reviews'/x) for root in (repo,Path('/home/werg/natlang')) for x in review_names]
    if any(source_reviews[i]['sha256']!=source_reviews[i+len(review_names)]['sha256'] for i in range(len(review_names))):
        raise ValueError('frozen source reviews differ from the live current reviews')
    identity={'schema':'natlang.lfm3-pair-verifier-identity/1','inputs':inputs,'model_id':model_id,'tokenizer_fingerprint':fp,'roles':list(roles),
              'protection':protection['proof'],'protection_files':protection_files,'decision_holds':hold_bindings,'code':[binding(x) for x in code],
              'tokenizer_files':tokenizer_files,'source_reviews':source_reviews}
    files=[Path(x['path']) for x in list(inputs.values())+identity['code']+identity['source_reviews']+protection_files+tokenizer_files]
    states={str(x):file_state(x) for x in files}
    def validate(current):
        if current!=identity or any(file_state(x)!=states[str(x)] for x in files):raise ValueError('bound verifier dependency changed')
        if sha256_file_hex(a.protection)!=identity['protection']['sha256']:raise ValueError('current protection changed')
        for h in hold_bindings:
            if sha256_file_hex(Path(h['path']))!=h['sha256']:raise ValueError('decision holds changed')
    wanted=set()
    for _,row in iter_jsonl_rows(a.pairs):wanted.update((row['communication']['writer_record_id'],row['communication']['reader_record_id']))
    records={}
    for _,row in iter_jsonl_rows(Path(inputs['records']['path'])):
        if row.get('id') in wanted:
            if row['id'] in records:raise ValueError('duplicate source record')
            records[row['id']]=row
    if set(records)!=wanted:raise ValueError('source record coverage incomplete')
    edges={}
    for _,edge in iter_jsonl_rows(Path(inputs['edges']['path'])):
        key=edge_digest(edge)
        if key in edges:raise ValueError('duplicate source edge')
        edges[key]=edge
    validate(identity)
    rows=(row for _,row in iter_jsonl_rows(a.pairs))
    def checked(row):
        declared=row['communication'].get('source_binding') or {}
        if any((declared.get(k) or {}).get('sha256')!=inputs[k]['sha256'] for k in ('records','edges')):
            raise ValueError('pair source binding differs from the exact verifier inputs')
        return verify_pair(row,records=records,edges=edges,tokenizer=tok,model_id=model_id,fingerprint=fp,
                           roles=roles,blocked=blocked,holds=holds)
    result=verify_chunks(rows,out=a.out,identity=identity,validate_identity=validate,chunk_rows=a.chunk_rows,
        verify=checked)
    if result['rows']!=declared.get('rows'):raise ValueError('verified count differs from conversion receipt')
    validate(identity)
    chunks=[binding(x) for x in sorted(a.out.glob('chunk-*.receipt.json'))]
    report={'schema':'natlang.lfm3-train-pair-verification/1','identity':identity,'identity_sha256':digest(identity),
            'rows':result['rows'],'chunks':chunks,'passed':True,'scope':'train pairs only; held pairs are not verified',
            'training_exposure':0,'optimizer_updates':0}
    path=a.out/'train-verification.json'
    if path.exists() and json.loads(path.read_text())!=report:raise ValueError('existing train verification differs')
    if not path.exists():write_json_durable(path,report)
    print(json.dumps({'result':result,'verification':binding(path)}),flush=True)

if __name__=='__main__':main()
