"""Seal completed provider outputs while the other generation lanes keep running."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from natlang_neuralese.common.jsonio import write_json_durable
from scripts.coreweave.teacher_tranche_admission import file_binding
from scripts.neuralese_data.provider_campaign_source import (SCHEMA, validated_dispatch_plan,
    ledger_dispatch_identity)
from scripts.neuralese_data.provider_judged_regrade import project as project_judged_outcome
from scripts.run_reviewed_luna_dispatcher import digest, _verify_ledger_identity, _verify_ledger_events


def snapshot(campaign, output, *, limit=64, previous=(), plan_path=None,
             oracle_regrades=(), source_worlds=None):
    campaign, output = Path(campaign).resolve(strict=True), Path(output).resolve()
    if type(limit) is not int or limit < 1 or output.exists():
        raise ValueError('provider snapshot needs a positive limit and fresh output')
    plan_path = (Path(plan_path) if plan_path is not None else campaign/'plan.json').resolve(strict=True)
    plan, identity = validated_dispatch_plan(plan_path, digest(plan_path))
    raw = Path(plan['claim_ledger']).read_bytes()
    raw = raw[:raw.rfind(b'\n')+1]  # A concurrent append may have an unfinished last line.
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ledger_identity = ledger_dispatch_identity(identity)
    _verify_ledger_identity(events, ledger_identity)
    _verify_ledger_events(events, ledger_identity)
    claims = {event['claim_id']:event for event in events if event['event']=='claim'}
    done = {event['claim_id']:event for event in events if event['event']=='terminal'}
    excluded = set()
    for path in previous:
        earlier = json.loads(Path(path).read_text())
        if earlier.get('schema') != SCHEMA or earlier.get('plan') != file_binding(plan_path):
            raise ValueError('previous provider snapshot belongs to another plan')
        excluded.update(entry['claim_id'] for entry in earlier['results'])
    selected, rows, rejected, regrade_bindings, regrade_files = [], [], [], [], {}
    regrades = {}
    for item in oracle_regrades:
        path = Path(item).resolve(strict=True)
        value = json.loads(path.read_text())
        case_id = value.get('case_id')
        if not isinstance(case_id, str) or case_id in regrades:
            raise ValueError('oracle regrades need unique exact original result IDs')
        regrades[case_id] = path
    if regrades and source_worlds is None:
        raise ValueError('oracle regrades require their exact source world fixture file')
    source_worlds = Path(source_worlds).resolve(strict=True) if source_worlds is not None else None
    programs = {index: json.loads(line) for index, line in enumerate(
        Path(plan['source']).read_text().splitlines()) if line.strip()}
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
        row_regrade = None
        if row.get('outcome',{}).get('accepted') is not True:
            regrade_path = regrades.get(row.get('id'))
            if regrade_path is None:
                rejected.append({'claim_id':claim_id,'reason':'task outcome rejected',
                                 'result':file_binding(result)})
                continue
            program = programs.get(claim['index'])
            if not isinstance(program, dict):
                raise ValueError('regraded Luna result has no exact selected plan task')
            projected, projection = project_judged_outcome(row, program, source_worlds, regrade_path,
                plan_path=plan_path, plan_sha256=digest(plan_path), claim_id=claim_id,
                claim_index=claim['index'],
                queue_path=queue_path, result_path=result, result_manifest_path=manifest,
                runtime=Path(plan['runtime']))
            proof_data = (json.dumps(projection, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
            proof_name = hashlib.sha256(row['id'].encode()).hexdigest()[:20] + '.json'
            proof_path = output / 'regrades' / proof_name
            row_regrade = {'schema': projection['schema'], 'projection': projection,
                'receipt': {'path': str(proof_path), 'sha256': hashlib.sha256(proof_data).hexdigest(),
                            'bytes': len(proof_data)}}
            projected['source_regrade'] = row_regrade
            regrade_files[proof_path] = proof_data
            regrade_bindings.append({'case_id': row['id'], 'proof': file_binding(regrade_path),
                                     'projection': row_regrade['receipt']})
            row = projected
        selected_entry = {'claim_id':claim_id,'queue':file_binding(queue_path),
                          'result':file_binding(result),'manifest':file_binding(manifest)}
        if row_regrade is not None:
            selected_entry['source_regrade'] = row_regrade['receipt']
        selected.append(selected_entry)
        rows.append((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
        if len(selected) >= limit:
            break
    if not selected:
        return {'selected':0,'training_admission':False}
    output.mkdir(parents=True)
    for path, data in regrade_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    source = output/'native-trajectories.jsonl'; source.write_bytes(b''.join(rows))
    ledger = output/'closed-claim-ledger.jsonl'; ledger.write_bytes(raw)
    declaration = {'schema':SCHEMA,'training_admission':False,'training_exposure':False,
        'plan':file_binding(plan_path),'source':file_binding(source),
        'claim_ledger':file_binding(ledger),'live_claim_ledger_origin':plan['claim_ledger'],
        'provider':plan['provider'],'model':plan['model_id'],'results':selected,
        'oracle_regrades':regrade_bindings,
        'rejected_completed_outputs':rejected,
        'scope':'Exact closed-output snapshot only. Full source/decision/protection conversion and corpus admission remain separate.'}
    write_json_durable(output/'provider-source.json',declaration)
    return {'selected':len(selected),'manifest':str(output/'provider-source.json'),
            'source':str(source),'training_admission':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--plan', type=Path, help='exact reviewed plan when stored outside campaign/plan.json')
    parser.add_argument('--limit',type=int,default=64)
    parser.add_argument('--previous',action='append',type=Path,default=[])
    parser.add_argument('--oracle-regrade',action='append',type=Path,default=[],
        help='exact independent judged-file qualification for a matching raw rejected result')
    parser.add_argument('--source-worlds',type=Path,
        help='exact source-world JSONL bound by the judged file qualification')
    args = parser.parse_args()
    print(json.dumps(snapshot(args.campaign,args.out,limit=args.limit,previous=args.previous,
        plan_path=args.plan,oracle_regrades=args.oracle_regrade,source_worlds=args.source_worlds)))


if __name__=='__main__':main()
