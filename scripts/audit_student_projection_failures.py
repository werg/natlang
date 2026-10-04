#!/usr/bin/env python3
"""Stream projection receipts into evidence-backed, heuristic failure clusters.

Search admission is NOT ordinary student accuracy. No answers, positive SFT rows,
DPO labels, or 'too hard' decisions are emitted by this analysis.
"""
import argparse, collections, hashlib, json, pathlib

def classify(candidate):
    tags=set();error=(candidate.get('error') or {}).get('message','')
    if 'prefix observation' in error or 'student_score' in error:tags.add('infrastructure_or_replay')
    if 'incomplete_projection' in error:tags.add('generation_truncated')
    if 'request_budget' in error:tags.add('request_budget')
    row=candidate.get('row') or {};outcome=row.get('outcome') or {};checks=outcome.get('checks') or {}
    turns=candidate.get('turns') or [];actions=[];feedback=[];delegations=0
    for t in turns:
        response=t.get('response') or {};calls=response.get('calls') or []
        actions.extend(json.dumps(c,sort_keys=True,ensure_ascii=False) for c in calls)
        for call in calls:
            if isinstance(call,list) and len(call)>1 and call[0]=='eval':
                code=str(call[1].get('code',''));delegations+=int('nl`' in code or 'nl<' in code)
        for message in (t.get('request') or {}).get('messages',[])[-1:]:
            if message.get('role')=='tool':feedback.append(str(message.get('content','')))
    text='\n'.join(feedback).lower()
    if any(x in text for x in ('typeerror:','referenceerror:','no such','not a function','unknown-field','type-mismatch','not-writable','expected a sentence')):tags.add('api_or_type_contract')
    if any(x in text for x in ('nothing else from this eval was kept','cannot move an entry into itself','file not found:')):tags.add('state_recovery')
    if checks.get('files') is False or checks.get('effects') is False or checks.get('file_return_consistency') is False:tags.add('effects_or_file_state')
    if checks.get('honest_stop') is False:tags.add('honest_stop')
    if 'returns no value' in text or ('omit value' in text and 'reason' in text) or 'before returning a value' in text:tags.add('finish_protocol')
    if 'budget exhausted' in str(outcome.get('detail','')) or checks.get('expected_status') is False:tags.add('incomplete_execution')
    if any(n>1 for n in collections.Counter(actions).values()):tags.add('repeated_action')
    if delegations>1:tags.add('delegation_review') # Not itself a fault: review redundant vs necessary calls.
    if checks.get('answer') is False:tags.add('answer_mismatch_needs_review')
    if not tags:tags.add('unclassified_needs_review')
    return sorted(tags)

def audit(root):
    root=pathlib.Path(root);counts=collections.Counter();families=collections.defaultdict(collections.Counter);cases=[];pins={};completed=candidates=turns=0
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        initial=directory/'initial.json'
        if not initial.exists():continue
        start=json.loads(initial.read_text());row=start.get('row') or {};ir=row.get('task',{}).get('program_ir',{});family=ir.get('family','unknown')
        case={'program_id':ir.get('id'),'family':family,'receipt_directory':str(directory.resolve()),'failure_tags':{},'verified_proposals':0,'failed_proposals':0,'mh_accepted_moves':0}
        for receipt in sorted(directory.glob('search-*.json')):
            raw=receipt.read_bytes();pins[str(receipt.resolve())]=hashlib.sha256(raw).hexdigest();proposal=json.loads(raw);candidate=proposal.get('candidate')
            case['mh_accepted_moves']+=int(proposal.get('accepted',False))
            if not candidate:continue # State-independent cut outside current trace: legitimate self-loop.
            if candidate.get('admitted'):case['verified_proposals']+=1;continue
            case['failed_proposals']+=1
            for tag in classify(candidate):
                counts[tag]+=1;families[family][tag]+=1;case['failure_tags'][tag]=case['failure_tags'].get(tag,0)+1
        final=directory/'final.json'
        if final.exists():
            d=json.loads(final.read_text());completed+=1;candidates+=int(d['summary']['published_candidate']);turns+=len(d.get('turns',[]));case['candidate_created']=d['summary']['published_candidate']
        cases.append(case)
    return {'schema':'natlang.student_projection_failure_clusters/1','heuristic_labels_require_review':True,'input_sha256':pins,'counts':dict(counts),'families':{f:dict(v) for f,v in families.items()},'cases':cases,'collection_progress':{'completed':completed,'candidates':candidates,'candidate_turns':turns},'accuracy_estimate':None,'too_hard_labels_created':0,'training_rows_created':0,'preference_labels_created':0}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();report=audit(a.collection)
    with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'progress':report['collection_progress'],'clusters':report['counts']}))
