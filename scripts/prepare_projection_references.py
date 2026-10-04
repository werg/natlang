#!/usr/bin/env python3
"""Select diverse already-trained, protected-source-disjoint projection references.

This only prepares candidates. Native admission and execution replay remain the
collector's responsibility. No evaluation answer is copied into the output.
"""
import argparse,collections,gzip,hashlib,json,pathlib
from scripts.corpus import file_digest, records

IDENTITY_KEYS={'external_source_id','id','original_source_id','parent_source_id','program_id','program_ids','program_ir_id','source_id','source_ids','source_program_id','source_program_ids','source_ref_id','source_groups','composite_component_ids','dataset_records'}

def aliases(value):
    result=set()
    def strings(v):
        if isinstance(v,str):result.add(v)
        elif isinstance(v,list):
            for x in v:strings(x)
    def walk(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in IDENTITY_KEYS:strings(x)
                if isinstance(x,(dict,list)):walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(value);return result


def label(value):
    if value is True:return 'true'
    if value is False:return 'false'
    if isinstance(value,dict) and value and all(type(v) is bool for v in value.values()):
        return 'all_true' if all(value.values()) else 'all_false' if not any(value.values()) else 'mixed_boolean'
    return 'other'


def prepare(snapshot, parent_split, protected, output, *, per_family=8, max_hint_chars=8000,max_turns=12):
    if per_family<1 or max_hint_chars<1 or max_turns<1:raise ValueError('selection budgets must be positive')
    output=pathlib.Path(output);output.mkdir(parents=True,exist_ok=False)
    protection=json.loads(pathlib.Path(protected).read_text());split=json.loads(pathlib.Path(parent_split).read_text())
    if protection.get('status')!='passed':raise ValueError('protected alias audit is not passed')
    authoritative_pins={}
    for name in ('teacher','split'):
        path=protection.get(name+'_path');expected=protection.get(name+'_sha256')
        if not path or file_digest(path)!=expected:raise ValueError('protected audit authoritative '+name+' input changed')
        authoritative_pins[path]=expected
    group_map=json.loads(pathlib.Path(protection['split_path']).read_text())['groups']
    blocked=set()
    for k in ('protected_identity_alias_values','protected_source_alias_values','protected_program_alias_values','protected_group_alias_values'):
        if not isinstance(protection.get(k),list):raise ValueError('protected audit lacks complete alias sets')
        blocked.update(protection[k])
    train=set(split['train_programs']);held=set(split['held_programs']);buckets=collections.defaultdict(dict);excluded=collections.Counter();scanned=0
    proven=set();unproven=set()
    for row in records(protection['teacher_path']):
        pid=row.get('program_id')
        if pid not in train:continue
        groups=row.get('source_groups') or []
        if row.get('split')!='train' or not groups or any(group_map.get(g)!='train' for g in groups):unproven.add(pid)
        else:proven.add(pid)
    train &= proven-unproven
    # Bound retained raw rows to per-family/label slots. Large outcomes are never
    # accumulated for the entire source bank.
    with (gzip.open(snapshot,'rt') if str(snapshot).endswith('.gz') else open(snapshot)) as stream:
        for line in stream:
            row=json.loads(line);scanned+=1;ir=row.get('task',{}).get('program_ir');reason=None
            if not ir or ir.get('split')!='train' or ir['id'] not in train or ir['id'] in held:reason='not_parent_train'
            elif ir.get('semantics',{}).get('world') or ir.get('semantics',{}).get('services'):reason='external_executor'
            elif not row.get('outcome',{}).get('accepted'):reason='not_verified_teacher'
            elif not row.get('trajectory') or len(row['trajectory'])>max_turns:reason='trajectory_length'
            elif (aliases(ir)|aliases(row.get('provenance',{}))) & blocked:reason='protected_alias'
            if reason:excluded[reason]+=1;continue
            hint=json.dumps([t['assistant'] for t in row['trajectory']],ensure_ascii=False,separators=(',',':'))
            if len(hint)>max_hint_chars:excluded['hint_length']+=1;continue
            family=ir.get('family',ir.get('curriculum',{}).get('family','unknown'));bucket=label(row['outcome'].get('value'))
            key=(family,bucket);slot=buckets[key]
            if ir['id'] in slot:excluded['duplicate_program']+=1;continue
            if len(slot)>=per_family:excluded['bucket_cap']+=1;continue
            slot[ir['id']]=row
    selected=[]
    for family in sorted({k[0] for k in buckets}):
        queues=[list(buckets[(family,k)].values()) for k in ('true','mixed_boolean','all_true','false','all_false','other')]
        for _ in range(per_family):
            nonempty=[q for q in queues if q]
            if not nonempty:break
            # Round robin guarantees that constant false answers cannot consume
            # every slot when positive/mixed candidates exist in the same family.
            q=nonempty[0];selected.append(q.pop(0));queues.remove(q);queues.append(q)
    artifacts=[];irs=[]
    for row in selected:
        ir=row['task']['program_ir'];name=hashlib.sha256(ir['id'].encode()).hexdigest()[:24]+'.json';p=output/name
        p.write_text(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');artifacts.append({'path':str(p.resolve()),'sha256':file_digest(p)});irs.append(ir)
    ir_path=output/'cases.ir.jsonl';ir_path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in irs))
    source_pins={str(pathlib.Path(p).resolve()):file_digest(p) for p in (snapshot,parent_split,protected)}
    source_pins.update(authoritative_pins)
    closure={'schema':'natlang.projection_reference_closure/1','status':'passed','selected_ir_sha256':file_digest(ir_path),
             'combined_identity_overlap_count':0,'selected_groups_not_train':[],'errors':[],
             'basis':'completed-parent train membership, pinned prepared-teacher source groups explicitly mapped train, recursive protected alias exclusion; native admission pending',
             'pins':source_pins,'selected_cases':len(irs)}
    (output/'source-closure.json').write_text(json.dumps(closure,indent=2)+'\n')
    report={'schema':'natlang.projection_reference_selection/1','teacher_artifacts':artifacts,'selected_program_ids':[r['id'] for r in irs],
            'scanned':scanned,'selected':len(irs),'by_family':dict(collections.Counter(r.get('family','unknown') for r in irs)),
            'by_label':dict(collections.Counter(label(r['outcome'].get('value')) for r in selected)),
            'excluded':dict(excluded),'pins':source_pins,'native_admission_pending':True,'training_publication':False}
    (output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('snapshot','parent-split','protected','output'):p.add_argument('--'+k,required=True,type=pathlib.Path)
    p.add_argument('--per-family',type=int,default=8);p.add_argument('--max-hint-chars',type=int,default=8000);p.add_argument('--max-turns',type=int,default=12)
    a=p.parse_args();report=prepare(**vars(a));print(json.dumps({k:v for k,v in report.items() if k not in ('teacher_artifacts','selected_program_ids')}))
