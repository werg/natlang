"""One bounded four-role retention cycle using declared live custody paths.

The caller supplies exact roots and accepted-owner epoch files in a JSON config.
Default is dry. Apply remains an explicit operation and every object is fenced,
rechecked and durably recorded by the shared guard.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from .dependency_references import reference_paths
from .retention_bookkeeping import own_bookkeeping,recheck_own_bookkeeping
from .retention_evidence import authenticated_owner_epochs
from .retention_guard import guarded_full,guarded_pair
from .retention_live import live_facts
from .retention_presence import resolve_object
from .retention_select import candidates
from .retention_manual import normalize_manual
from .retention_batch import prepare_reference_evidence


def rotate_cycle(items, cycle):
    if not items:return []
    offset=cycle%len(items);return items[offset:]+items[:offset]


def run_cycle(config, *, api, get, write_durable, apply=False):
    root=Path(config['state_root']);ops=Path(config['operations_root']);roots=[Path(x) for x in config['reference_roots']];retired=root/'standalone-retired-cloud-objects.json';lock=root/'standalone-cloud-retention.lock';fallback=config['fallback_role'];qualifications=[Path(x) for x in config['qualified_local_parents']]
    if not qualifications:raise ValueError('qualified local fallback is required')
    def receipt_path(role):return ops/('coreweave-production-'+role+'-private-export-receipt-20261014.json')
    def read(path):return json.loads(Path(path).read_text())
    def bindings(paths):return [{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]
    def present(role,digest):return resolve_object(api,read(receipt_path(role)),role,digest,get=get) is not None
    def remove(role,digest):
        obj=resolve_object(api,read(receipt_path(role)),role,digest,get=get)
        if obj is None:raise ValueError('candidate disappeared before deletion')
        obj.delete()
    def current_units():
        result=subprocess.run(['systemctl','--user','list-units','--state=active','--plain','--no-legend'],check=True,text=True,capture_output=True)
        return [line.split()[0].removesuffix('.service') for line in result.stdout.splitlines() if line.split()]
    role_configs=config['roles']
    if sorted(x['rank'] for x in role_configs)!=[1,2,3,4]:raise ValueError('exactly four roles are required')
    preferred=config.get('priority_ranks',[1,2,3,4])
    if sorted(preferred)!=[1,2,3,4]:raise ValueError('invalid rank priority')
    cycle=config.get('cycle_index',int(time.time()//300));max_attempts=config.get('max_attempts',4);per_rank=config.get('max_attempts_per_rank',1);max_seconds=config.get('max_start_seconds',240)
    if type(cycle) is not int or cycle<0 or type(max_attempts) is not int or max_attempts<1 or type(per_rank) is not int or per_rank<1 or not isinstance(max_seconds,(int,float)) or max_seconds<=0:raise ValueError('invalid retention cycle budget')
    preferred=rotate_cycle(preferred,cycle)
    role_configs=sorted(role_configs,key=lambda x:preferred.index(x['rank']))
    outcomes=[];attempts=0;started=time.monotonic();plans=[]
    for role_config in role_configs:
        rank=role_config['rank'];journal=Path(role_config['journal']);owner_bindings=bindings(role_config['owner_epochs']);owners=authenticated_owner_epochs(owner_bindings)
        def read_context(rank=rank,journal=journal,owner_bindings=owner_bindings,role_config=role_config):
            manifest=read(config['active_roles']);current_name=next(x['ownership'] for x in manifest['roles'] if x['rank']==rank);current=read(root/current_name)
            exports=read(journal)['exports'];manual=root/('standalone-peer%d-manual-cloud-journal.json'%rank);manual_exports=read(manual)['exports'] if manual.exists() else []
            exports+=manual_exports
            epoch_owners=authenticated_owner_epochs(owner_bindings)
            intents={x['role']:read(x['intent']) for x in exports}
            for e in manual_exports:intents[e['role']]=normalize_manual(e,intents[e['role']],epoch_owners)
            owner_state=read(config['owner_state']);units=current_units()
            wanted={fallback,*intents}
            wanted.update((e.get('control') or {}).get('role') for e in exports if (e.get('control') or {}).get('role'))
            receipts=[read(receipt_path(role)) for role in wanted if receipt_path(role).exists()]
            protected=set()
            for item in manifest['roles']:protected.add(read(root/item['ownership'])['parent_sha256'])
            for path in root.glob('standalone-peer*-manual-cloud-journal.json'):
                entries=read(path)['exports'];verified_manual=[]
                if entries:protected.add(read(max(entries,key=lambda e:e['step'])['intent'])['sha256'])
                for e in entries:
                    pin=read(e['intent']);rp=receipt_path(e['role']);r=read(rp) if rp.exists() else None;identity=(r or {}).get('identity') or {};proof=(r or {}).get('independent_verification') or {}
                    settled=bool(r and r.get('private') and identity=={'sha256':pin['sha256'],'bytes':pin['bytes']} and proof.get('sha256')==pin['sha256'] and proof.get('bytes')==pin['bytes'])
                    if not settled or e.get('unit') in units or (owner_state.get('pending') or {}).get('role')==e['role']:protected.add(pin['sha256'])
                    if settled:verified_manual.append((e,pin))
                if verified_manual:protected.add(max(verified_manual,key=lambda x:x[0]['step'])[1]['sha256'])
            for path in root.glob('*intent.json'):
                item=read(path)
                for key in ('historical_parent_sha256','parent_sha256'):
                    if item.get(key):protected.add(item[key])
            qualified_shas=set()
            for path in qualifications:
                q=read(path)
                if q.get('passed') is not True or Path(q['checkpoint']).stat().st_size!=q['bytes']:raise ValueError('local qualified parent unavailable')
                from natlang_neuralese.common.hashing import sha256_file_hex_cached
                if sha256_file_hex_cached(q['checkpoint'],trust_archive_receipt=False)!=q['sha256']:raise ValueError('local qualified parent changed')
                protected.add(q['sha256']);qualified_shas.add(q['sha256'])
            fallback_receipt=next((r for r in receipts if r.get('role')==fallback),None)
            if fallback_receipt is None or fallback_receipt.get('identity',{}).get('sha256') not in qualified_shas:raise ValueError('fallback receipt is not a qualified local parent')
            transitions={key:read(path) for key,path in role_config.get('transitions',{}).items()}
            return exports,intents,epoch_owners,current,owner_state,units,receipts,protected,transitions
        context=read_context();exports,intents,owners,current,owner_state,units,receipts,protected,transitions=context
        retired_state=read(retired) if retired.exists() else {'retired_roles':[]}
        selected=candidates(exports=exports,intents=intents,owners=owners,current_owner=current,owner_state=owner_state,receipts=receipts,fallback_role=fallback,active_units=units,transitions=transitions,protected_sha256s=protected)
        selected=[x for x in selected if x['role'] not in retired_state.get('retired_roles',[]) and (config.get('candidate_allowlist') is None or x['role'] in config['candidate_allowlist'])]
        selected=rotate_cycle(selected,cycle)
        for candidate in selected[:per_rank]:
            if attempts>=max_attempts or time.monotonic()-started>=max_seconds:break
            plans.append((candidate,rank,journal,exports,intents,receipts,read_context));attempts+=1
        if attempts>=max_attempts or time.monotonic()-started>=max_seconds:break
    if not plans:return outcomes
    digests={candidate['sha256'] for candidate,*_ in plans}
    digests.update(candidate['control_sha256'] for candidate,*_ in plans if candidate['control_required'])
    evidence=prepare_reference_evidence(roots,digests,reference_paths)
    for candidate,rank,journal,exports,intents,receipts,read_context in plans:
        if time.monotonic()-started>=max_seconds:break
        try:
            entry=next(x for x in exports if x['role']==candidate['role']);intent=read(entry['intent']);receipt=next(x for x in receipts if x['role']==candidate['role'])
            phase=ops/('coreweave-production-'+candidate['role']+'-private-upload-phase-20261014.json');cache_root=ops/'private-export-cache';cache_refs=reference_paths([candidate['sha256']],[cache_root])[candidate['sha256']]
            own_caches=[p for p in cache_refs if (value:=read(p)).get('identity')==receipt.get('identity') and value.get('receipt')==receipt]
            reviews=[Path(x) for x in config.get('reviews',[]) if candidate['sha256'] in Path(x).read_text()]
            own=own_bookkeeping(role=candidate['role'],digest=candidate['sha256'],receipt=receipt,receipt_path=receipt_path(candidate['role']),phase=phase if phase.exists() else None,reviews=reviews,retirement_state=retired if retired.exists() else None)
            for path in own_caches:own.update(own_bookkeeping(role=candidate['role'],digest=candidate['sha256'],receipt=receipt,cache=path))
            control=entry.get('control') or {}
            if candidate['control_required']:
                control_receipt=next(x for x in receipts if x['role']==candidate['control_role'])
                control_phase=ops/('coreweave-production-'+candidate['control_role']+'-private-upload-phase-20261014.json')
                control_cache=reference_paths([candidate['control_sha256']],[cache_root])[candidate['control_sha256']]
                own_control_caches=[p for p in control_cache if (value:=read(p)).get('identity')==control_receipt.get('identity') and value.get('receipt')==control_receipt]
                own.update(own_bookkeeping(role=candidate['control_role'],digest=candidate['control_sha256'],receipt=control_receipt,receipt_path=receipt_path(candidate['control_role']),phase=control_phase if control_phase.exists() else None,watched_digests=(candidate['sha256'],)))
                for path in own_control_caches:own.update(own_bookkeeping(role=candidate['control_role'],digest=candidate['control_sha256'],receipt=control_receipt,cache=path,watched_digests=(candidate['sha256'],)))
            allowed=[str(Path(entry['intent'])),*own]
            manual_journal=root/('standalone-peer%d-manual-cloud-journal.json'%rank)
            is_manual=manual_journal.exists() and any(x['role']==candidate['role'] for x in read(manual_journal)['exports'])
            scope={'allowed_paths':allowed,'journal_path':manual_journal if is_manual else journal,'role':candidate['role'],'intent_path':entry['intent']}
            if candidate['control_required']:
                scope.update(control=control,control_receipt=receipt_path(candidate['control_role']),control_phase=Path(control['phase_path']))
            def read_live(value,phase_name):
                recheck_own_bookkeeping(own,role=value['role'],digest=value['sha256'])
                if read(entry['intent'])!=intent:raise ValueError('candidate intent changed')
                ex,ins,os_,cu,st,active,rs,pro,ts=read_context()
                value['operation_units']=[u for u in active if value['role'] in u or (value.get('control_role') and value['control_role'] in u)]
                live=live_facts(candidate=value,exports=ex,intents=ins,owners=os_,current_owner=cu,owner_state=st,receipts=rs,successor_role=value['successor_role'],fallback_role=fallback,active_units=active,transition=ts.get(cu['parent_sha256']))
                live['eligibility']['protected_sha256s'].update(pro);return live
            guard=guarded_pair if candidate['control_required'] else guarded_full
            outcomes.append(guard(candidate=candidate,roots=roots,scan=reference_paths,reference_scope=scope,read_live=read_live,api=api,lock_path=lock,state_path=retired,remove=remove,present=present,write_durable=write_durable,apply=apply,evidence=evidence))
        except (ValueError,RuntimeError,OSError,KeyError) as error:
            print(json.dumps({'role':candidate['role'],'status':'protected_or_unavailable','reason':str(error)}),flush=True)
            continue
        if len(outcomes)>=config.get('max_per_cycle',1):return outcomes
    return outcomes


def main(argv=None):
    import requests,wandb
    from natlang_neuralese.common.jsonio import write_json_durable
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--apply',action='store_true');args=parser.parse_args(argv)
    print(json.dumps(run_cycle(json.loads(args.config.read_text()),api=wandb.Api(timeout=15),get=requests.get,write_durable=write_json_durable,apply=args.apply)))


if __name__=='__main__':main()
