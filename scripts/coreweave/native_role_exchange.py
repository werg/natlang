#!/usr/bin/env python3
"""Relay exact existing-source-host role receipts; stage assets only for joint consumption.

No credentials enter a sandbox. Scheduling and mathematical readiness stay in
native_readiness/native_role_release. This service reuses SDKIO and its scoped
verified private ferry, with the existing explicit resource allowlist.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'/'neuralese'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_role_exchange_io import NativeRoleExchangeIO
from natlang_neuralese.common.hashing import sha256_hex,canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.rl import native_role_release as roles
from provider_bridge import checked_binding


def call(io,sid,expression):
    # The same declared runtime/CPU composition and ENV precede imports.
    code=io.dependency_code(sid)
    controls=(io.source_host_controls(sid)if hasattr(io,'source_host_controls')
        else io.config['sandboxes'][sid].get('role_controls'))
    if controls is not None:
        # Reuse the actual source-host controller loader. Imported role proofs
        # must never be published under a retired CPU-helper composition.
        if getattr(io,'local',None)is not None and sid==io.local['source_host_id']:
            path=io._local_path(controls['path']);raw=io._local_file(path)
            actual={'path':str(path),'sha256':__import__('hashlib').sha256(raw).hexdigest(),
                'bytes':len(raw)}
            if (actual['sha256']!=controls.get('sha256')
                    or('bytes'in controls and actual['bytes']!=controls['bytes'])):
                raise ValueError('local source-host controls changed their exact binding')
            controls=actual
        else:controls=checked_binding(controls)
        code+=('import pathlib,hashlib,runpy;'
            'p=pathlib.Path('+repr(controls['path'])+');raw=p.read_bytes();'
            'assert len(raw)=='+str(controls['bytes'])+' and hashlib.sha256(raw).hexdigest()=='+repr(controls['sha256'])+';'
            'role_controls=json.loads(raw);ref=role_controls["controller"];'
            'p=pathlib.Path(ref["path"]);raw=p.read_bytes();'
            'assert hashlib.sha256(raw).hexdigest()==ref["sha256"] and ("bytes"not in ref or len(raw)==ref["bytes"]);'
            'runpy.run_path(str(p),run_name="qualified_role_controller")["load_controls"](role_controls);')
    return io.remote(sid,code+'from natlang_neuralese.rl import native_role_release as roles;'
        +'print(json.dumps('+expression+'))',timeout=900)


def _read_bound(io,sid,reference):
    raw=io.read(sid,reference['path'])
    if isinstance(raw,str):raw=raw.encode()
    checked=checked_binding({**reference,'bytes':reference.get('bytes',len(raw))})
    if len(raw)!=checked['bytes']or sha256_hex(raw)!=checked['sha256']:
        raise ValueError('source-host role transfer changed exact bytes')
    return raw


def _arm_preparation(config,io,sid,plan,request):
    if not request.get('snapshot_preparation') or not plan.get('consumer_preparation_handoff'):return
    curriculum=config.get('preparation_curriculum')
    if not curriculum:raise ValueError('consumer allocation handoff needs its exact frozen curriculum')
    closure=io.dependency_closure(sid,{'semantic_config':curriculum})
    io.verify_assets(sid,closure['assets'])
    call(io,sid,'roles.arm_preparation_handoff('+repr(plan)+','+repr(request)+','+repr(curriculum)+')')


def _return_local_cycle_descendants(config,io,plans):
    """Transfer completed participant bundles back to the released local owners."""
    declaration=config.get('local_cycle_returns')
    if not declaration:return None
    if declaration.get('schema')!='natlang.native-local-cycle-return-transport/1':
        raise ValueError('unsupported local-cycle return transport declaration')
    consumer=declaration['consumer'];host=config['hosts'][consumer];sid=host['source_host_id']
    completion_path=declaration['completion_path']
    if io.read_optional(sid,completion_path)is None:return None
    completion=call(io,sid,'roles.binding('+repr(completion_path)+')')
    role_plans=declaration['role_plans']
    exported=call(io,sid,'roles.export_local_cycle_returns('+repr(host['plan']['path'])+','+
        repr(completion)+','+repr(role_plans)+',directory='+repr(declaration['directory'])+')')
    if not isinstance(exported,list):raise ValueError('local-cycle return exporter omitted its package inventory')
    if not exported:return None
    already=[]
    for item in exported:
        role=item.get('role');target=config['hosts'][role]['source_host_id']
        exchange=plans[role]['role_exchange'];participant=item.get('participant')
        pointer_path=exchange['directory']+'/'+participant+'-return-current.json'
        raw=io.read_optional(target,pointer_path)
        current=json.loads(raw)if raw is not None else None
        if (current and current.get('receipt')==item['receipt']
                and current.get('checkpoint',{}).get('sha256')==item['checkpoint']['sha256']):
            already.append(role)
    if len(already)==len(exported):return None
    completed=[]
    for item in exported:
        role=item.get('role');target=config['hosts'][role]['source_host_id']
        if item.get('role_plan')!=role_plans[role] or item.get('source_release') is None:
            raise ValueError('local-cycle return package changed its source role authority')
        mirrored=io.mirror_many(sid,target,item['files'])
        if mirrored is False:return {'status':'return-transfer-paused'}
        call(io,target,'roles.publish_local_cycle_return_pointer('+repr(role_plans[role]['path'])+','+
            repr(item['receipt']['path'])+')')
        completed.append({'role':role,'participant':item['participant'],'receipt':item['receipt'],
            'checkpoint':item['bundle']})
    return {'status':'local-cycle-descendants-returned','participants':completed}


def stage_preparation_files(config,io):
    staged=config.get('preparation_stage_files')
    if staged is None:return None
    if (staged.get('schema')!='natlang.native-role-preparation-files/1'
            or not hasattr(io,'stage_prepared_files')):
        raise ValueError('shared-stage file preparation needs the bound local/cloud source-host adapter')
    source_host=staged.get('source_host_id')
    if source_host!=(getattr(io,'local',None)or{}).get('source_host_id'):
        raise ValueError('prepared source inventory differs from the current local source owner')
    owner_hosts={host['source_host_id']for host in config.get('hosts',{}).values()}
    if any(row.get('target_host_id')not in owner_hosts for row in staged.get('files',[])):
        raise ValueError('prepared file inventory targets an owner outside this two-model exchange')
    consumer=config.get('local_cycle_returns',{}).get('consumer')
    if consumer not in config.get('hosts',{}):
        raise ValueError('prepared file receipt has no declared semantic consumer host')
    sid=config['hosts'][consumer]['source_host_id']
    inventory=canonical_json_sha256_hex({'schema':staged['schema'],'source_host_id':source_host,
        'files':staged['files']})
    if inventory!=staged.get('inventory_sha256')or not isinstance(staged.get('receipt_path'),str):
        raise ValueError('prepared file inventory differs from its frozen exact digest')
    raw=io.read_optional(sid,staged['receipt_path'])
    if raw is not None:
        if isinstance(raw,str):raw=raw.encode()
        receipt=json.loads(raw)
        if (receipt.get('schema')!='natlang.native-role-preparation-files-receipt/1'
                or receipt.get('inventory_sha256')!=inventory
                or receipt.get('files')!=staged['files']):
            raise ValueError('existing prepared-file receipt is for another source inventory')
        return receipt
    copied=io.stage_prepared_files(staged['files'],source_host_id=source_host)
    receipt={'schema':'natlang.native-role-preparation-files-receipt/1',
        'inventory_sha256':inventory,'files':staged['files'],'verified_copies':copied}
    io.write_atomic(sid,staged['receipt_path'],
        json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())
    return receipt


def tick(config,io):
    if config.get('schema')!='natlang.native-role-exchange-service/1':
        raise ValueError('unsupported source-host role exchange service')
    stage_preparation_files(config,io)
    hosts=config['hosts'];plans={};contract=None
    for role,host in hosts.items():
        allowed=(io.has_host(host['source_host_id'])if hasattr(io,'has_host')
            else host['source_host_id']in io.config['sandboxes'])
        if not allowed:
            raise ValueError('source host is outside the existing provider allowlist')
        raw=_read_bound(io,host['source_host_id'],host['plan']);plan=json.loads(raw)
        _,exchange=roles.ownership(plan)
        if (exchange is None or exchange['role']!=role
                or exchange['roles'][role]['source_host_id']!=host['source_host_id']
                or set(exchange['roles'])!=set(hosts)):
            raise ValueError('service hosts differ from the declared actual optimizer owners')
        actual=roles.contract(plan)
        if contract is not None and actual!=contract:raise ValueError('source hosts disagree on the joint contract')
        contract=actual;plans[role]=plan
    runtime_assets=config.get('task_eval_collector_assets')
    if (not isinstance(runtime_assets,dict) or set(runtime_assets)!=set(hosts)):
        raise ValueError('role relay requires each current owner TaskEval Python collector binding')
    for role,host in hosts.items():
        asset=runtime_assets[role]
        runtime=io.config['sandboxes'][host['source_host_id']]['runtime'] if host['source_host_id'] in io.config['sandboxes'] else io.local['runtime']
        expected_path=str(Path(runtime)/'scripts/run_bonsai_queue.py')
        if (asset.get('kind')!='task-source-file' or asset.get('path')!=expected_path
                or type(asset.get('bytes'))is not int or asset['bytes']<=0
                or not isinstance(asset.get('sha256'),str)):
            raise ValueError('TaskEval collector binding differs from the actual frozen host runtime')
        io.verify_assets(host['source_host_id'],[asset])
    if len({plan.get('snapshot_preparation',False) for plan in plans.values()})!=1:
        raise ValueError('source hosts disagree on snapshot preparation ownership')
    snapshots={};qualified_snapshots={}
    for role,host in hosts.items():
        sid=host['source_host_id'];exchange=plans[role]['role_exchange']
        # A final proof is sealed by the actual source host only after its raw
        # coordinator has exited. A live optimizer simply stays useful.
        call(io,sid,'roles.finalize('+repr(host['plan']['path'])+','+repr(host['raw_state'])+')')
        pointer=io.read_optional(sid,exchange['directory']+'/'+role+'-status.json')
        if pointer is None:continue
        reference=json.loads(pointer);raw=_read_bound(io,sid,reference)
        proof=json.loads(raw)
        if proof.get('role')!=role:raise ValueError('role relay source substituted another owner')
        snapshots[role]=raw.decode()
        if plans[role].get('snapshot_preparation'):
            pointer=io.read_optional(sid,exchange['directory']+'/'+role+'-eligible.json')
            if pointer is not None:
                raw=_read_bound(io,sid,json.loads(pointer));proof=json.loads(raw)
                if proof.get('role')!=role:raise ValueError('qualified snapshot substituted another owner')
                qualified_snapshots[role]=raw.decode()
    for target,host in hosts.items():
        for source,raw in snapshots.items():
            if source==target:continue
            dependency_inventory=roles.proof_dependencies(json.loads(raw))
            if dependency_inventory:
                transferred=io.mirror_many(hosts[source]['source_host_id'],host['source_host_id'],
                    dependency_inventory)
                if transferred is False:return {'status':'semantic-proof-transfer-paused','source':source,'target':target}
            call(io,host['source_host_id'],'roles.receive('+repr(raw)+','+repr(host['plan']['path'])+')')
        for source,raw in qualified_snapshots.items():
            if source!=target:
                dependency_inventory=roles.proof_dependencies(json.loads(raw))
                if dependency_inventory:
                    transferred=io.mirror_many(hosts[source]['source_host_id'],host['source_host_id'],
                        dependency_inventory)
                    if transferred is False:return {'status':'semantic-proof-transfer-paused','source':source,'target':target}
                call(io,host['source_host_id'],'roles.receive('+repr(raw)+','+repr(host['plan']['path'])+',snapshot_only=True)')
    # Exposure candidates are not readiness. Once both owners have published
    # their actual current candidates, seal a checkpoint-specific full-check
    # request and install the same immutable bytes on both source hosts.
    consumer=contract['semantic_consumer'];consumer_host=hosts[consumer]
    generation={}
    for role,role_plan in plans.items():
        declaration=roles.plan_model(role_plan,role)
        generation[declaration['handoff_requirement']['model_id']]=declaration.get(
            'foundation_qualification_generation')
    if len(generation)==2 and all(isinstance(value,str) and value for value in generation.values()):
        generation_id=canonical_json_sha256_hex({'schema':'natlang.local-cycle-foundation-generation/1',
            'roles':generation})
        request_ref=call(io,consumer_host['source_host_id'],'roles.foundation_request('+repr(plans[consumer])+','+
            repr(plans[consumer]['role_exchange']['directory']+'/foundation-qualification-requests')+
            ',generation='+repr(generation_id)+')')
        if request_ref is not None:
            raw=io.read(consumer_host['source_host_id'],request_ref['path'])
            if isinstance(raw,str):raw=raw.encode()
            checked=checked_binding({**request_ref,'bytes':request_ref.get('bytes',len(raw))})
            if len(raw)!=checked['bytes'] or sha256_hex(raw)!=checked['sha256']:
                raise ValueError('foundation qualification request changed during relay')
            request=json.loads(raw)
            for role,host in hosts.items():
                call(io,host['source_host_id'],'roles.publish_foundation_request('+repr(host['plan']['path'])+
                    ','+repr(request)+')')
    returned=_return_local_cycle_descendants(config,io,plans)
    if returned is not None:return returned
    consumer=contract['semantic_consumer'];host=hosts[consumer];sid=host['source_host_id']
    handoff_pointer=io.read_optional(sid,plans[consumer]['role_exchange']['directory']+'/post-vae-handoff.json')
    if handoff_pointer is not None:
        handoff=json.loads(_read_bound(io,sid,json.loads(handoff_pointer)))
        if config.get('preparation_curriculum') and not call(io,sid,
                'roles.handoff_matches_preparation_work('+repr(plans[consumer])+','+repr(handoff)+','+
                repr(config['preparation_curriculum'])+')'):
            for name,row in hosts.items():
                call(io,row['source_host_id'],'roles.cancel_handoff('+repr(plans[name])+','+repr(handoff)+')')
            handoff=None
    else:handoff=None
    if handoff is not None:
        lease=config.get('handoff_lease_seconds')
        interval=config.get('poll_seconds',15)
        if type(lease) not in (int,float) or not 2*interval<lease:
            raise ValueError('joint handoff requires its explicit renewable provider lease')
        closure=io.dependency_closure(sid,{'semantic_config':handoff['joint_controller']['curriculum']})
        io.verify_assets(sid,closure['assets'])
        offers={name:call(io,row['source_host_id'],'roles.handoff_offer('+repr(plans[name])+','+
            repr(row['raw_state'])+','+repr(handoff)+')') for name,row in hosts.items()}
        attempt=(call(io,sid,'roles.select_handoff_attempt('+repr(plans[consumer])+','+
            repr(handoff)+','+repr(offers)+',lease_seconds='+repr(lease)+')') if all(offers.values()) else None)
        expires=time.time()+lease
        for name,row in hosts.items():
            expression=('roles.renew_handoff('+repr(plans[name])+','+repr(handoff)+','+repr(offers)+
                ',expires_at='+repr(expires)+',attempt='+repr(attempt)+')' if attempt is not None else
                'roles.cancel_handoff('+repr(plans[name])+','+repr(handoff)+')')
            call(io,row['source_host_id'],expression)
    request_raw=io.read_optional(sid,plans[consumer]['role_exchange']['directory']+'/semantic-assets-request.json')
    if request_raw is None:
        returned=_return_local_cycle_descendants(config,io,plans)
        return returned or {'status':'waiting-for-semantic-consumption','roles':sorted(snapshots)}
    request=json.loads(request_raw)
    mode=request.get('snapshot_preparation',False)
    if mode and plans[consumer].get('consumer_preparation_handoff'):
        curriculum=config.get('preparation_curriculum')
        if not curriculum:raise ValueError('snapshot relay needs its actual declared work')
        if call(io,sid,'roles.preparation_work_complete('+repr(plans[consumer])+','+repr(curriculum)+')'):
            return {'status':'preparation-work-complete','roles':sorted(snapshots)}
    expected=call(io,sid,'roles.semantic_request('+repr(plans[consumer])+',snapshot='+repr(mode)+')')
    if request!=expected:return {'status':'waiting-for-current-releases','roles':sorted(snapshots)}
    usable=call(io,sid,'roles.ready_participants('+repr(plans[consumer])+',snapshot='+repr(mode)+')[0]')
    if len(usable)>=2:
        _arm_preparation(config,io,sid,plans[consumer],request)
        return {'status':'semantic-assets-ready','models':sorted(row['model_id']for row in usable),'transferred':[]}
    # Explicit snapshot preparation copies exact qualified historical weights;
    # ordinary joint consumption continues to require actual optimizer release.
    checkpoints={};staged=[]
    for model_id,participant in request['participants'].items():
        source=participant['source_host_id']
        source_role=next(role for role,row in hosts.items()if row['source_host_id']==source)
        source_plan=plans[source_role]
        kind='snapshot' if request.get('snapshot_preparation') else 'released'
        found=call(io,source,'roles.exchanged('+repr(source_plan)+',kind='+repr(kind)+')[0]')
        ready=found[model_id]
        if (ready['checkpoint']!=participant['checkpoint']
                or ready['source_role_release']['sha256']!=participant['role_release']['sha256']):
            raise ValueError('source role changed during semantic consumption')
        checkpoint=ready['checkpoint']
        if source!=sid:
            directory=source_plan['role_exchange']['directory']+'/semantic-checkpoints'
            checkpoint=call(io,source,'roles.pin_checkpoint('+repr(checkpoint)+','+repr(directory)+')')
            closure=io.dependency_closure(source,{'semantic_participants':[checkpoint]})
            metadata=call(io,source,'roles.consumed_metadata('+repr(ready)+')')
            needs_alias=(False if request.get('snapshot_preparation') else
                call(io,source,'roles.recipe_recovery_needed('+repr(ready)+')'))
            missing=call(io,sid,'roles.missing_metadata('+repr(metadata)+')')
            package=call(io,source,'roles.metadata_package('+repr(missing)+','+repr(directory+'/metadata')+')')
            io.mirror_many(source,sid,[*closure['files'],*(item['transport']for item in package)])
            call(io,sid,'roles.install_metadata('+repr(package)+')')
            io.verify_assets(sid,closure['assets'])
            if needs_alias:
                call(io,sid,'roles.install_checkpoint_alias('+repr(ready)+','+repr(checkpoint)+')')
            staged.append(model_id)
        checkpoints[model_id]=checkpoint
    call(io,sid,'roles.install_assets('+repr(host['plan']['path'])+','+repr(request)+','+repr(checkpoints)+')')
    _arm_preparation(config,io,sid,plans[consumer],request)
    return {'status':'semantic-assets-ready','models':sorted(checkpoints),'transferred':staged}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();config_path=Path(args.config).resolve(strict=True)
    config=json.loads(config_path.read_text());out=Path(config['out']);out.mkdir(parents=True,exist_ok=True)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (out/'service.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io=NativeRoleExchangeIO(config['provider'])
        while not stopped:
            try:result=tick(config,io)
            except Exception as error:result={'status':'repair-required','error_type':type(error).__name__}
            # Provider exceptions may contain capabilities. Preserve only bounded
            # error classes here; SDKIO retains its private scoped diagnostics.
            write_json_durable(out/'state.json',{'at':time.time(),**result})
            if args.once:return result
            interval=config.get('poll_seconds',15)
            if type(interval)not in(int,float)or not 0<interval<=60:raise ValueError('role relay poll must be positive and at most 60s')
            until=time.monotonic()+interval
            while not stopped and time.monotonic()<until:time.sleep(min(.2,max(0,until-time.monotonic())))


if __name__=='__main__':main()
