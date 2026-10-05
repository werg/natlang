#!/usr/bin/env python3
"""Compare completed candidate executions on the exact same protected task packet."""
import argparse,collections,datetime,hashlib,json,pathlib,statistics

def load(path):
    path=pathlib.Path(path);report=json.loads(path.read_text())
    if report.get('status')!='completed' or report.get('training_publication') is not False:
        raise ValueError('only completed review-only evaluations may be compared')
    rows=[json.loads(line) for line in pathlib.Path(report['case_results']).read_text().splitlines()]
    keys=[(r['id'],r['program_ir_sha256']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate case identity')
    return report,{key:row for key,row in zip(keys,rows)},hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(report,rows):
    eligible=[r for r in rows.values() if r['disposition']!='policy_held']
    passed=sum(r['disposition']=='complete_success' for r in eligible)
    durations=[(datetime.datetime.fromisoformat(r['finished_at'].replace('Z','+00:00'))-datetime.datetime.fromisoformat(r['started_at'].replace('Z','+00:00'))).total_seconds() for r in eligible]
    return {'model':report['model_id'],'eligible':len(eligible),'passed':passed,
            'pass_rate':passed/len(eligible) if eligible else None,
            'counts':dict(collections.Counter(r['disposition'] for r in rows.values())),
            'median_case_seconds':statistics.median(durations) if durations else None,
            'sum_case_seconds':sum(durations),'candidate_identity':report['candidate_identity']}

def compare(left,right):
    a,ar,asha=load(left);b,br,bsha=load(right)
    for key in ['packet_manifest_sha256','cases_ir_sha256','gold_reference_sha256','runtime_manifest_sha256','system_prompt_sha256','tool_surface_sha256','limits']:
        if a[key]!=b[key]:raise ValueError('comparison protocol differs: '+key)
    if ar.keys()!=br.keys():raise ValueError('comparison case identities differ')
    pairs=[];counts=collections.Counter()
    for key,x in ar.items():
        y=br[key]
        if (x['disposition']=='policy_held')!=(y['disposition']=='policy_held'):raise ValueError('policy eligibility differs')
        if x['disposition']=='policy_held':continue
        goodx=x['disposition']=='complete_success';goody=y['disposition']=='complete_success'
        label='both_pass' if goodx and goody else 'left_only' if goodx else 'right_only' if goody else 'both_fail'
        counts[label]+=1
        pairs.append({'id':x['id'],'source':x['source'],'left':x['disposition'],'right':y['disposition'],'pair':label})
    return {'schema':'natlang.candidate_comparison/1','training_publication':False,
            'left_report_sha256':asha,'right_report_sha256':bsha,
            'packet_manifest_sha256':a['packet_manifest_sha256'],'left':summarize(a,ar),'right':summarize(b,br),
            'paired_counts':dict(counts),'cases':pairs,
            'interpretation':'Small fixed task set; all eligible cases included. Native stacks, precision and hardware paths may differ. Case time includes tools and model-generated work, not raw decode speed.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--left',required=True);ap.add_argument('--right',required=True);ap.add_argument('--output',required=True,type=pathlib.Path);args=ap.parse_args()
    result=compare(args.left,args.right);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({side:result[side] for side in ['left','right','paired_counts']},indent=2))
