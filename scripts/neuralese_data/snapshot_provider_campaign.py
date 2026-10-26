"""Seal completed provider outputs while the other generation lanes keep running."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from natlang_neuralese.common.jsonio import write_json_durable
from scripts.coreweave.teacher_tranche_admission import file_binding
from scripts.neuralese_data.provider_campaign_source import SCHEMA, validated_dispatch_plan
from scripts.run_reviewed_luna_dispatcher import digest, _verify_ledger_identity, _verify_ledger_events


def snapshot(campaign, output, *, limit=64, previous=()):
    campaign, output = Path(campaign).resolve(strict=True), Path(output).resolve()
    if type(limit) is not int or limit < 1 or output.exists():
        raise ValueError('provider snapshot needs a positive limit and fresh output')
    plan_path = campaign/'plan.json'
    plan, identity = validated_dispatch_plan(plan_path, digest(plan_path))
    raw = Path(plan['claim_ledger']).read_bytes()
    raw = raw[:raw.rfind(b'\n')+1]  # A concurrent append may have an unfinished last line.
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    _verify_ledger_identity(events, identity)
    _verify_ledger_events(events, identity)
    claims = {event['claim_id']:event for event in events if event['event']=='claim'}
    done = {event['claim_id']:event for event in events if event['event']=='terminal'}
    excluded = set()
    for path in previous:
        earlier = json.loads(Path(path).read_text())
        if earlier.get('schema') != SCHEMA or earlier.get('plan') != file_binding(plan_path):
            raise ValueError('previous provider snapshot belongs to another plan')
        excluded.update(entry['claim_id'] for entry in earlier['results'])
    selected, rows, rejected = [], [], []
    for claim_id, terminal in sorted(done.items(),key=lambda item:claims[item[0]]['index']):
        if claim_id in excluded:
            continue
        claim = claims[claim_id]
        if terminal.get('exit_code') != 0 or not terminal.get('output_accounting',{}).get('complete'):
            rejected.append({'claim_id':claim_id,'reason':'not an exact completed collector output'})
            continue
        queue_path = Path(claim['queue'])
        if digest(queue_path) != claim['queue_sha256']:
            raise ValueError('completed provider queue changed')
        queue = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
        if len(queue) != 1:
            raise ValueError('one provider claim must own one exact case')
        result = Path(queue[0]['output']); manifest = Path(str(result)+'.manifest.json')
        actual = [line for line in result.read_bytes().splitlines() if line.strip()]
        if len(actual) != 1:
            raise ValueError('completed provider output must contain one full trajectory')
        row = json.loads(actual[0])
        if row.get('outcome',{}).get('accepted') is not True:
            rejected.append({'claim_id':claim_id,'reason':'task outcome rejected',
                             'result':file_binding(result)})
            continue
        selected.append({'claim_id':claim_id,'queue':file_binding(queue_path),
                         'result':file_binding(result),'manifest':file_binding(manifest)})
        rows.append(actual[0]+b'\n')
        if len(selected) >= limit:
            break
    if not selected:
        return {'selected':0,'training_admission':False}
    output.mkdir(parents=True)
    source = output/'native-trajectories.jsonl'; source.write_bytes(b''.join(rows))
    ledger = output/'closed-claim-ledger.jsonl'; ledger.write_bytes(raw)
    declaration = {'schema':SCHEMA,'training_admission':False,'training_exposure':False,
        'plan':file_binding(plan_path),'source':file_binding(source),
        'claim_ledger':file_binding(ledger),'live_claim_ledger_origin':plan['claim_ledger'],
        'provider':plan['provider'],'model':plan['model_id'],'results':selected,
        'rejected_completed_outputs':rejected,
        'scope':'Exact closed-output snapshot only. Full source/decision/protection conversion and corpus admission remain separate.'}
    write_json_durable(output/'provider-source.json',declaration)
    return {'selected':len(selected),'manifest':str(output/'provider-source.json'),
            'source':str(source),'training_admission':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--limit',type=int,default=64)
    parser.add_argument('--previous',action='append',type=Path,default=[])
    args = parser.parse_args()
    print(json.dumps(snapshot(args.campaign,args.out,limit=args.limit,previous=args.previous)))


if __name__=='__main__':main()
