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
from provider_worker import SDKIO
from natlang_neuralese.common.hashing import sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.rl import native_role_release as roles
from provider_bridge import checked_binding


def call(io,sid,expression):
    # The same declared runtime/CPU composition and ENV precede imports.
    code=io.dependency_code(sid)
    return io.remote(sid,code+'from natlang_neuralese.rl import native_role_release as roles;'
        +'print(json.dumps('+expression+'))',timeout=900)


def _read_bound(io,sid,reference):
    raw=io.read(sid,reference['path'])
    if isinstance(raw,str):raw=raw.encode()
    checked=checked_binding({**reference,'bytes':reference.get('bytes',len(raw))})
    if len(raw)!=checked['bytes']or sha256_hex(raw)!=checked['sha256']:
        raise ValueError('source-host role transfer changed exact bytes')
    return raw


def tick(config,io):
    if config.get('schema')!='natlang.native-role-exchange-service/1':
        raise ValueError('unsupported source-host role exchange service')
    hosts=config['hosts'];plans={};contract=None
    for role,host in hosts.items():
        if host['source_host_id']not in io.config['sandboxes']:
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
            call(io,host['source_host_id'],'roles.receive('+repr(raw)+','+repr(host['plan']['path'])+')')
        for source,raw in qualified_snapshots.items():
            if source!=target:
                call(io,host['source_host_id'],'roles.receive('+repr(raw)+','+repr(host['plan']['path'])+',snapshot_only=True)')
    consumer=contract['semantic_consumer'];host=hosts[consumer];sid=host['source_host_id']
    request_raw=io.read_optional(sid,plans[consumer]['role_exchange']['directory']+'/semantic-assets-request.json')
    if request_raw is None:return {'status':'waiting-for-semantic-consumption','roles':sorted(snapshots)}
    request=json.loads(request_raw)
    expected=call(io,sid,'roles.semantic_request('+repr(plans[consumer])+')')
    if request!=expected:return {'status':'waiting-for-current-releases','roles':sorted(snapshots)}
    usable=call(io,sid,'roles.ready_participants('+repr(plans[consumer])+')[0]')
    if len(usable)>=2:return {'status':'semantic-assets-ready','models':sorted(row['model_id']for row in usable),'transferred':[]}
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
        io=SDKIO(config['provider'])
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
