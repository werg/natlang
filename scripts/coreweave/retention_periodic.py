"""One bounded four-role retention cycle using declared live custody paths.

The caller supplies exact roots and accepted-owner epoch files in a JSON config.
Default is dry. Apply remains an explicit operation and every object is fenced,
rechecked and durably recorded by the shared guard.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
from .dependency_references import reference_paths
from .retention_bookkeeping import own_bookkeeping,recheck_own_bookkeeping
from .retention_evidence import authenticated_owner_epochs
from .retention_guard import guarded_full,guarded_pair
from .retention_live import live_facts
from .retention_presence import resolve_object
from .retention_select import candidates


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
    role_configs=sorted(role_configs,key=lambda x:preferred.index(x['rank']))
    outcomes=[]
    for role_config in role_configs:
        rank=role_config['rank'];journal=Path(role_config['journal']);owner_bindings=bindings(role_config['owner_epochs']);owners=authenticated_owner_epochs(owner_bindings)
        def read_context():
            manifest=read(config['active_roles']);current_name=next(x['ownership'] for x in manifest['roles'] if x['rank']==rank);current=read(root/current_name)
            exports=read(journal)['exports'];manual=root/('standalone-peer%d-manual-cloud-journal.json'%rank)
            if manual.exists():exports+=read(manual)['exports']
            intents={x['role']:read(x['intent']) for x in exports};owner_state=read(config['owner_state']);units=current_units()
            wanted={fallback,*intents}
            wanted.update((e.get('control') or {}).get('role') for e in exports if (e.get('control') or {}).get('role'))
            receipts=[read(receipt_path(role)) for role in wanted if receipt_path(role).exists()]
            protected=set()
            for item in manifest['roles']:protected.add(read(root/item['ownership'])['parent_sha256'])
            for path in root.glob('standalone-peer*-manual-cloud-journal.json'):
                for e in read(path)['exports']:protected.add(read(e['intent'])['sha256'])
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
            return exports,intents,authenticated_owner_epochs(owner_bindings),current,owner_state,units,receipts,protected,transitions
        context=read_context();exports,intents,owners,current,owner_state,units,receipts,protected,transitions=context
        retired_state=read(retired) if retired.exists() else {'retired_roles':[]}
        selected=candidates(exports=exports,intents=intents,owners=owners,current_owner=current,owner_state=owner_state,receipts=receipts,fallback_role=fallback,active_units=units,transitions=transitions,protected_sha256s=protected)
        for candidate in selected:
            if candidate['role'] in retired_state.get('retired_roles',[]):continue
            if config.get('candidate_allowlist') is not None and candidate['role'] not in config['candidate_allowlist']:continue
            try:
                entry=next(x for x in exports if x['role']==candidate['role']);intent=intents[candidate['role']];receipt=next(x for x in receipts if x['role']==candidate['role'])
                phase=ops/('coreweave-production-'+candidate['role']+'-private-upload-phase-20261014.json');cache_root=ops/'private-export-cache';cache_refs=reference_paths([candidate['sha256']],[cache_root])[candidate['sha256']]
                if len(cache_refs)>1:raise ValueError('ambiguous candidate export cache')
                reviews=[Path(x) for x in config.get('reviews',[]) if candidate['sha256'] in Path(x).read_text()]
                own=own_bookkeeping(role=candidate['role'],digest=candidate['sha256'],receipt=receipt,receipt_path=receipt_path(candidate['role']),phase=phase if phase.exists() else None,cache=next(iter(cache_refs),None),reviews=reviews,retirement_state=retired if retired.exists() else None)
                control=entry.get('control') or {}
                if candidate['control_required']:
                    control_receipt=next(x for x in receipts if x['role']==candidate['control_role'])
                    control_phase=ops/('coreweave-production-'+candidate['control_role']+'-private-upload-phase-20261014.json')
                    control_cache=reference_paths([candidate['control_sha256']],[cache_root])[candidate['control_sha256']]
                    if len(control_cache)>1:raise ValueError('ambiguous control export cache')
                    own.update(own_bookkeeping(role=candidate['control_role'],digest=candidate['control_sha256'],receipt=control_receipt,receipt_path=receipt_path(candidate['control_role']),phase=control_phase if control_phase.exists() else None,cache=next(iter(control_cache),None),watched_digests=(candidate['sha256'],)))
                allowed=[str(Path(entry['intent'])),*own]
                scope={'allowed_paths':allowed,'journal_path':journal,'role':candidate['role'],'intent_path':entry['intent']}
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
                outcomes.append(guard(candidate=candidate,roots=roots,scan=reference_paths,reference_scope=scope,read_live=read_live,api=api,lock_path=lock,state_path=retired,remove=remove,present=present,write_durable=write_durable,apply=apply))
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
