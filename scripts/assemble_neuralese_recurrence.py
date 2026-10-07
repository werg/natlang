#!/usr/bin/env python3
"""Assemble reviewed decision worlds with positive target admission and producer/split closure.

This intentionally reviews named constructed-world families, not arbitrary source corpora.
Existing corpora retain their own source policy. Evaluation records remain in a separate file.
"""
import argparse, collections, hashlib, json, subprocess
from pathlib import Path
from audit_neuralese_recurrence import names

REVIEWED = {'decision_skill_catalog', 'decision_extract_chain'}
# A source review can admit named v7 records from these families only; membership
# alone never admits a row. New families require an assembler policy update.
SOURCE_REVIEW_FAMILIES = {'authored_semantic_reducers'}
CURRENT_CONVERSIONS = {'natlang.neuralese-conversion/5', 'natlang.neuralese-conversion/6',
                       'natlang.neuralese-conversion/7', 'natlang.neuralese-conversion/8'}
SOURCE_REVIEW_SCHEMA = 'natlang.neuralese-source-review/1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def artifact_key(path):
    path=Path(path).resolve()
    root=Path(__file__).resolve().parents[1]
    try:return path.relative_to(root).as_posix()
    except ValueError:return str(path)


def selector_matches(row, selector):
    return (row.get('id') in selector['target_ids'] or
            bool(set(row.get('source_groups',[])) & set(selector['source_groups'])))


def load_source_review(path, records, pieces):
    """Load an exact-input, explicit-target allow/hold decision; reject stale reviews."""
    review=json.loads(Path(path).read_text())
    if set(review)!={'schema','inputs','allow','hold'} or review.get('schema')!=SOURCE_REVIEW_SCHEMA:
        raise ValueError('invalid source review schema or fields')
    expected={role:[{'path':artifact_key(p),'sha256':sha(p)} for p in paths]
              for role,paths in (('records',records),('pieces',pieces))}
    supplied=review.get('inputs')
    if supplied!=expected:
        raise ValueError('source review artifact paths or SHA-256 digests are stale')
    for key in ('allow','hold'):
        selector=review.get(key)
        if not isinstance(selector,dict) or set(selector)!={'source_groups','target_ids'}:
            raise ValueError('source review selectors require source_groups and target_ids')
        if any(not isinstance(values,list) or any(not isinstance(v,str) or not v for v in values)
               for values in selector.values()):
            raise ValueError('source review selector entries must be nonempty strings')
    if not review['allow']['source_groups'] and not review['allow']['target_ids']:
        raise ValueError('source review must explicitly allow source groups or target IDs')
    if (set(review['allow']['source_groups']) & set(review['hold']['source_groups']) or
        set(review['allow']['target_ids']) & set(review['hold']['target_ids'])):
        raise ValueError('source review cannot both allow and hold the same selector')
    all_rows=[]
    for path in records:
        all_rows.extend(json.loads(line) for line in Path(path).open())
    all_ids={row.get('id') for row in all_rows}
    all_groups={g for row in all_rows for g in row.get('task',{}).get('program_ir',{}).get('source_groups',[])}
    for key in ('allow','hold'):
        selector=review[key]
        stale_ids=set(selector['target_ids'])-all_ids
        stale_groups=set(selector['source_groups'])-all_groups
        if stale_ids or stale_groups:
            raise ValueError(f'stale {key} source review selectors: IDs={sorted(stale_ids)} groups={sorted(stale_groups)}')
    return review, all_rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', nargs='+', type=Path, required=True)
    p.add_argument('--pieces', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hold-source-group', action='append', default=[], help='explicit semantic-source hold, logged per excluded record')
    p.add_argument('--source-review',type=Path,
        help='SHA-pinned JSON review with explicit allow/hold source_groups and target_ids')
    a=p.parse_args(argv)
    if a.out.exists():raise ValueError('fresh output required')
    # Authored fixture holds also apply to already materialized snapshots whose
    # admission predates review. A successful old outcome cannot clear the hold.
    fixture_policy=Path(__file__).resolve().parents[1]/'ts-host/scripts/inline-curriculum/decision-rich-extra-fixtures.json'
    fixture_holds=json.loads(fixture_policy.read_text()).get('source_holds', [])
    held_groups=set(a.hold_source_group) | {
        f"authored-bounded-decisions-v1:{h['domain']}:fixture-{h['fixture']}" for h in fixture_holds}
    source_review,_=(load_source_review(a.source_review,a.records,a.pieces)
                     if a.source_review else (None,[]))
    allow_selector=source_review['allow'] if source_review else {'source_groups':[],'target_ids':[]}
    review_hold=source_review['hold'] if source_review else {'source_groups':[],'target_ids':[]}
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
            elif set(ir.get('source_groups', [])) & held_groups or selector_matches(row,review_hold):reason='explicit semantic source hold'
            elif row.get('neuralese_conversion',{}).get('version') not in CURRENT_CONVERSIONS:reason='requires current conversion'
            elif family not in REVIEWED and not (source_review and family in SOURCE_REVIEW_FAMILIES
                    and selector_matches(row,allow_selector)
                    and row.get('neuralese_conversion',{}).get('version') in {'natlang.neuralese-conversion/7','natlang.neuralese-conversion/8'}):
                reason='source family or target not explicitly reviewed'
            elif row.get('training_admission',{}).get('approved') is not True:reason='target not positively admitted'
            elif row.get('outcome',{}).get('accepted') is not True:reason='runtime outcome not accepted'
            elif row.get('outcome',{}).get('oracle',{}).get('level')!='exact' or row.get('outcome',{}).get('oracle',{}).get('accepted') is not True:reason='requires accepted exact oracle'
            elif row.get('trace_admission',{}).get('admitted') is not True:reason='trace not admitted'
            elif ir.get('license')!='project-generated' or 'constructed-world-oracle' not in ir.get('gold_sources',[]):reason='unreviewed license or oracle'
            elif row.get('split') not in {'train','test'}:reason='missing explicit split'
            elif not row.get('source_groups'):reason='missing stable source groups'
            elif not all(g.startswith('authored-bounded-decisions-v1:') for g in row['source_groups']) and not (
                    source_review and selector_matches(row,allow_selector)):
                reason='non-fixture source groups require exact source review'
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
            bad=[n for n in dependencies if len(producers[n])!=1 or producers[n][0] not in rows or rows[producers[n][0]]['split']!=row['split']]
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
    report={'schema':'natlang.reviewed-recurrence-cohort/1',
      'admission':('reviewed-constructed-world-and-explicit-source-review-runtime-positive-graph-closed'
                   if source_review else 'reviewed-constructed-world-runtime-positive-source-and-graph-closed'),
      'reviewed_families':sorted(REVIEWED),'counts':dict(collections.Counter(r['split'] for r in rows.values())),
      'explicitly_reviewed_targets':sorted(r['id'] for r in rows.values()
        if source_review and selector_matches(r,allow_selector) and
        r.get('task',{}).get('program_ir',{}).get('curriculum',{}).get('family') not in REVIEWED),
      'source_groups':{s:sorted(g for g,v in groups.items() if s in v) for s in ['train','test']},
      'unique_source_groups':len(groups),
      'unique_authored_fixtures':sum(g.startswith('authored-bounded-decisions-v1:') for g in groups),
      'held_targets':len(rejected),'recurrence':audit,
      'inputs':{str(path):sha(path) for path in a.records+a.pieces},
      'source_review':({'schema':source_review['schema'],'sha256':sha(a.source_review),
                        'allow':source_review['allow'],'hold':source_review['hold']}
                       if source_review else None),
      'quality_scope':'exact authored-world decisions and runtime copying; not human/external benchmark labels; permutations and replicas are not independent facts',
      'evaluation_policy':'test is held out from optimization; task-execution diagnostics required in addition to NLL'}
    (a.out/'admission.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['counts']))
if __name__=='__main__':main()
