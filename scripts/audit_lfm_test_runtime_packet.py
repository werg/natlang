#!/usr/bin/env python3
"""Audit a prepared LFM runtime packet against its source and rendered split."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.jsonio import canonical_json_str as canonical  # noqa: E402


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def audit(source, source_manifest, ready, packet, out):
    source,source_manifest,ready,packet,out=map(Path,(source,source_manifest,ready,packet,out))
    prior=json.loads((packet/'packet-receipt.json').read_text())
    sm=json.loads(source_manifest.read_text())
    if sha_file(source)!=prior['source']['sha256'] or sha_file(source)!=sm['sha256']:
        raise ValueError('model-neutral source hash mismatch')
    with (packet/'selection.jsonl').open() as stream:
        wanted={json.loads(line)['id'] for line in stream if line.strip()}
    variants=defaultdict(dict); neutral_aliases=defaultdict(set); neutral_rows=Counter()
    with source.open() as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line); split=row.get('split'); neutral_rows[split]+=1
            if split!='test': continue
            for key in ('source_groups','source_ids','source_program_ids'):
                neutral_aliases[key].update(map(str,row.get(key) or []))
            if row.get('program_id'): neutral_aliases['program_ids'].add(str(row['program_id']))
            ir=(row.get('task') or {}).get('program_ir')
            if isinstance(ir,dict) and ir.get('id') in wanted:
                variants[ir['id']][digest(ir)]=ir
    if set(variants)!=wanted: raise ValueError('not all packet IR ids found in source')
    actual={'train':defaultdict(set),'test':defaultdict(set)}; counts=Counter(); tokens=Counter(); max_tokens=Counter()
    with ready.open() as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line); split=row.get('split')
            if split not in ('train','test'): raise ValueError('rendered row lacks train/test split')
            counts[split]+=1
            n=(row.get('token_counts') or {}).get('total_tokens')
            if isinstance(n,int): tokens[split]+=n;max_tokens[split]=max(max_tokens[split],n)
            for key in ('source_groups','source_ids','source_program_ids'):
                actual[split][key].update(map(str,row.get(key) or []))
            if row.get('program_id'):actual[split]['program_ids'].add(str(row['program_id']))
    # The packet is test-only. Require every supplied alias to remain absent from actual LFM train.
    packet_aliases=defaultdict(set)
    for id,byhash in variants.items():
        for ir in byhash.values():
            for key in ('source_groups','source_ids'):
                packet_aliases[key].update(map(str,ir.get(key) or []))
    for kind in ('source_groups','source_ids','program_ids'):
        packet_aliases[kind].update(neutral_aliases[kind])
    overlaps={k:sorted(packet_aliases[k]&actual['train'][k]) for k in packet_aliases}
    overlaps={k:v for k,v in overlaps.items() if v}
    if overlaps: raise ValueError(f'test packet aliases overlap actual LFM train: { {k:len(v) for k,v in overlaps.items()} }')
    # Record each exact IR digest as an independent evaluation instance. Duplicate
    # original IDs must be dispatched separately and joined by evaluation_case_id.
    out.mkdir(parents=True,exist_ok=False)
    cases_path=out/'cases.ir.jsonl'; gold_path=out/'gold-reference.jsonl'; sel_path=out/'selection.jsonl'; audit_path=out/'variant-audit.jsonl'
    variant_case_count=0; extra=0; metadata_only=0; semantic=0
    with cases_path.open('x') as co,gold_path.open('x') as go,sel_path.open('x') as so,audit_path.open('x') as ao:
        for ident in sorted(variants):
            hashes=sorted(variants[ident]); objs=[variants[ident][x] for x in hashes]
            stripped=[]
            for ir in objs:
                clean=dict(ir);clean.pop('task_modality',None);stripped.append(digest(clean))
            category='single' if len(objs)==1 else ('task_modality_metadata_only' if len(set(stripped))==1 else 'semantic_or_contract_difference')
            if category=='task_modality_metadata_only': metadata_only+=1
            elif category=='semantic_or_contract_difference': semantic+=1
            extra+=max(0,len(objs)-1)
            if len(objs)>1:
                ao.write(json.dumps({'id':ident,'variant_count':len(objs),'variant_sha256':hashes,'comparison_after_removing_only_task_modality':category,'all_expected_sha256':sorted({digest(x.get('semantics',{}).get('expected')) for x in objs})},sort_keys=True,separators=(',',':'))+'\n')
            for fp,ir in zip(hashes,objs):
                expected=ir.get('semantics',{}).get('expected')
                if 'expected' not in ir.get('semantics',{}): raise ValueError(f'missing expected for {ident}')
                eval_id=f'{ident}::irsha256:{fp}'
                co.write(json.dumps(ir,ensure_ascii=False,separators=(',',':'))+'\n')
                go.write(json.dumps({'evaluation_case_id':eval_id,'id':ident,'program_ir_sha256':fp,'expected':expected},ensure_ascii=False,separators=(',',':'))+'\n')
                so.write(json.dumps({'evaluation_case_id':eval_id,'id':ident,'program_ir_sha256':fp,'split':'test','variant_class':category},ensure_ascii=False,separators=(',',':'))+'\n')
                variant_case_count+=1
    receipt={'schema':'lfm-test-runtime-packet-audit/2','status':'prepared_test_only_not_executed','parent_packet':str(packet.resolve()),'parent_receipt_sha256':sha_file(packet/'packet-receipt.json'),'model_neutral_source':{'path':str(source.resolve()),'sha256':sha_file(source),'manifest_sha256':sha_file(source_manifest),'rows':sum(neutral_rows.values())},'actual_lfm_ready':{'path':str(ready.resolve()),'sha256':sha_file(ready),'rows':sum(counts.values()),'split_rows':dict(counts),'split_token_totals':dict(tokens),'max_rendered_row_tokens':dict(max_tokens),'context_length':16384,'all_rendered_rows_within_context':max(max_tokens.values(),default=0)<=16384},'source_and_alias_closure':{'packet_ids':len(variants),'packet_ir_variants':variant_case_count,'additional_distinct_variants':extra,'metadata_only_variant_ids':metadata_only,'semantic_or_contract_variant_ids':semantic,'actual_lfm_train_alias_overlaps':{},'source_groups_and_aliases_are_separate_from_model_neutral_5432_test_decision_rows':True},'variant_handling':'Every distinct canonical program_ir digest is emitted; evaluation_case_id is the composite (original id, exact IR digest). Dispatch variants sharing original id separately; never collapse to first occurrence. task_modality-only differences are explicitly classified metadata-only; all other differences remain semantic/contract review cases.','limitations':['Rendered-row token counts prove training rendering stays within 16,384 tokens; they do not prove every native runtime prompt, after document loading/tool context, fits the same context. Runtime prompt/token preflight is still required before evaluation.','source_program_ids are absent from the model-specific ready rows; source_groups, source_ids, and program_ids were checked for overlap against its train split.']}
    def count_lines(path):
        with path.open() as stream:return sum(1 for _ in stream)
    receipt['artifacts']={p.name:{'path':str(p.resolve()),'sha256':sha_file(p),'rows':count_lines(p)} for p in (cases_path,gold_path,sel_path,audit_path)}
    (out/'audit-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--ready',type=Path,required=True);p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.source,a.source_manifest,a.ready,a.packet,a.out),indent=2,sort_keys=True))
if __name__=='__main__':main()
