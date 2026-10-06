#!/usr/bin/env python3
"""Assemble reviewed decision worlds with positive target admission and producer/split closure.

This intentionally reviews named constructed-world families, not arbitrary source corpora.
Existing corpora retain their own source policy. Evaluation records remain in a separate file.
"""
import argparse, collections, hashlib, json, subprocess
from pathlib import Path
from audit_neuralese_recurrence import names

REVIEWED = {'decision_skill_catalog', 'decision_extract_chain'}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', nargs='+', type=Path, required=True)
    p.add_argument('--pieces', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hold-source-group', action='append', default=[], help='explicit semantic-source hold, logged per excluded record')
    a=p.parse_args()
    if a.out.exists():raise ValueError('fresh output required')
    # Authored fixture holds also apply to already materialized snapshots whose
    # admission predates review. A successful old outcome cannot clear the hold.
    fixture_policy=Path(__file__).resolve().parents[1]/'ts-host/scripts/inline-curriculum/decision-rich-extra-fixtures.json'
    fixture_holds=json.loads(fixture_policy.read_text()).get('source_holds', [])
    held_groups=set(a.hold_source_group) | {
        f"authored-bounded-decisions-v1:{h['domain']}:fixture-{h['fixture']}" for h in fixture_holds}
    rows={}; rejected=[]; pieces={}
    for path in a.pieces:
        for line in path.open():
            row=json.loads(line)
            if row['name'] in pieces and pieces[row['name']]!=row:raise ValueError('conflicting soft piece')
            pieces[row['name']]=row
    for path in a.records:
        for line in path.open():
            row=json.loads(line); ir=row.get('task',{}).get('program_ir',{}); family=ir.get('curriculum',{}).get('family')
            reason=None
            if row['id'] in rows:raise ValueError('duplicate representation of '+row['id'])
            if set(ir.get('source_groups', [])) & held_groups:reason='explicit semantic source hold'
            elif family not in REVIEWED:reason='source family has not been reviewed by this assembler'
            elif row.get('neuralese_conversion',{}).get('version') not in {'natlang.neuralese-conversion/5', 'natlang.neuralese-conversion/6'}:reason='requires current conversion'
            elif row.get('training_admission',{}).get('approved') is not True:reason='target not positively admitted'
            elif row.get('outcome',{}).get('accepted') is not True:reason='runtime outcome not accepted'
            elif row.get('outcome',{}).get('oracle',{}).get('level')!='exact' or row.get('outcome',{}).get('oracle',{}).get('accepted') is not True:reason='requires accepted exact oracle'
            elif row.get('trace_admission',{}).get('admitted') is not True:reason='trace not admitted'
            elif ir.get('license')!='project-generated' or 'constructed-world-oracle' not in ir.get('gold_sources',[]):reason='unreviewed license or oracle'
            elif row.get('split') not in {'train','test'}:reason='missing explicit split'
            elif not ir.get('source_groups') or not all(g.startswith('authored-bounded-decisions-v1:') for g in ir['source_groups']):reason='missing stable fixture groups'
            if reason:rejected.append({'id':row['id'],'reason':reason});continue
            rows[row['id']]=row
    # Fixed point: an approved reader cannot depend on a held or missing producer.
    changed=True
    while changed:
        changed=False; producers=collections.defaultdict(list)
        for r in rows.values():
            for n in names(r.get('target'),'write'):producers[n].append(r['id'])
        for ident,row in list(rows.items()):
            dependencies=names(row['messages'],'read')-names(row.get('target'),'write')
            bad=[n for n in dependencies if len(producers[n])!=1 or rows[producers[n][0]]['split']!=row['split']]
            if bad:rejected.append({'id':ident,'reason':'producer closure','names':bad});del rows[ident];changed=True
    groups=collections.defaultdict(set)
    for r in rows.values():
        for g in r['source_groups']:groups[g].add(r['split'])
    if any(len(v)>1 for v in groups.values()):raise ValueError('source group crosses splits')
    if not all(any(r['split']==split for r in rows.values()) for split in ['train','test']):raise ValueError('both train and held-out are required')
    a.out.mkdir(parents=True)
    used=set()
    for split,name in [('train','train.jsonl'),('test','heldout.jsonl')]:
        with (a.out/name).open('w') as f:
            for r in rows.values():
                if r['split']==split:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (a.out/'records.jsonl').open('w') as f:
        for r in rows.values():
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
            for m in r['messages']:
                if isinstance(m.get('content'),list):used.update(x['name'] for x in m['content'] if x['type']=='soft')
    if used-pieces.keys():raise ValueError('missing soft pieces')
    (a.out/'pieces.jsonl').write_text(''.join(json.dumps(pieces[n],ensure_ascii=False)+'\n' for n in sorted(used)))
    (a.out/'held-targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rejected))
    subprocess.run(['python3',str(Path(__file__).with_name('audit_neuralese_recurrence.py')),str(a.out/'records.jsonl'),'--out',str(a.out/'recurrence-audit.json')],check=True)
    audit=json.loads((a.out/'recurrence-audit.json').read_text())
    if not audit['structurally_closed']:raise ValueError('graph failed structural closure')
    report={'schema':'natlang.reviewed-recurrence-cohort/1','admission':'reviewed-constructed-world-runtime-positive-source-and-graph-closed',
      'reviewed_families':sorted(REVIEWED),'counts':dict(collections.Counter(r['split'] for r in rows.values())),
      'source_groups':{s:sorted(g for g,v in groups.items() if s in v) for s in ['train','test']},
      'unique_authored_fixtures':len(groups),'held_targets':len(rejected),'recurrence':audit,
      'inputs':{str(path):sha(path) for path in a.records+a.pieces},
      'quality_scope':'exact authored-world decisions and runtime copying; not human/external benchmark labels; permutations and replicas are not independent facts',
      'evaluation_policy':'test is held out from optimization; task-execution diagnostics required in addition to NLL'}
    (a.out/'admission.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['counts']))
if __name__=='__main__':main()
