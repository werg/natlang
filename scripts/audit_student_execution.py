#!/usr/bin/env python3
"""Summarize successes, regressions and failure evidence without relabeling gold."""
import argparse,collections,json,pathlib,hashlib

def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def audit(roots):
    history=collections.defaultdict(list);pins={}
    for root in roots:
        for file in pathlib.Path(root).rglob('case-results.jsonl'):
            rows=read(file)
            if not (file.parent/'report.json').exists(): continue
            pins[str(file.resolve())]=hashlib.sha256(file.read_bytes()).hexdigest()
            for row in rows:
                row={**row,'evidence_file':str(file.resolve())}
                history[row['id']].append(row)
    cases=[];families=collections.defaultdict(collections.Counter)
    for identity,rows in sorted(history.items()):
        rows.sort(key=lambda r:(r['step'],r.get('finished_at','')))
        latest=rows[-1]; observations=[]
        for row in rows:
            observations.append({'step':row['step'],'disposition':row['disposition'],'turns':row.get('request_turns'),
                'value':row.get('value'),'failed_checks':[k for k,v in row.get('checks',{}).items() if v is False]})
        statuses=[r['disposition'] for r in rows];successes=statuses.count('complete_success')
        group='always_success' if successes==len(rows) else 'never_success' if not successes else 'mixed'
        failures=[k for k,v in latest.get('checks',{}).items() if v is False]
        category=('retain_success' if latest['disposition']=='complete_success' else
                  'unscored_policy_hold' if latest['disposition']=='policy_held' else
                  'resource_or_incomplete_review' if latest['disposition'] in ('resource_failure','infrastructure_failure','incomplete_task') else
                  'effects_and_answer' if any(k in failures for k in ('files','effects','file_return_consistency')) else
                  'semantic_or_computation')
        families[latest.get('family','unknown')][latest['disposition']]+=1
        cases.append({'id':identity,'family':latest.get('family'),'source_groups':latest.get('source_groups'),
            'pattern':group,'successes':successes,'evaluations':len(rows),'latest':latest,'history':observations,
            'review_category':category,'training_use':'family-level cohort guidance only; protected cases and answers never enter training'})
    return {'schema':'natlang.student_execution_review/1','input_sha256':pins,'cases':cases,
            'families':dict(families),'counts':dict(collections.Counter(c['pattern'] for c in cases)),
            'preference_labels_created':0,'training_rows_created':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+');p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    d=audit(a.roots);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
    print(json.dumps({'cases':len(d['cases']),'counts':d['counts'],'families':d['families']}))
