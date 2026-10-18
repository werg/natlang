"""Continuous verified-export ingestion; policy in ASYNC_INGEST_SERVICE_DECOMPOSITION."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from ..common.jsonio import write_json_durable
from .native_async import poll_latest, verified_export, verify_request, selected_ingest_export
from .task_cadence import parse_task_eval, request_queue


def binding(path):
    path=Path(path)
    with path.open('rb')as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path.resolve()),sha256=digest,bytes=path.stat().st_size)


def _executes_request(argv, directory):
    module='natlang_neuralese.eval.task_cadence'
    return bool(argv and Path(argv[0]).name.startswith('python')and module in argv
        and argv.index(module)>0 and argv[argv.index(module)-1]=='-m'
        and '--async-request'in argv and argv.index('--async-request')+1<len(argv)
        and Path(argv[argv.index('--async-request')+1]).resolve()==Path(directory).resolve())


def running_request_processes(unit, request_dir, *, proc=Path('/proc'), cgroups=Path('/sys/fs/cgroup'), systemctl=subprocess.run):
    """Only an actual process in this active unit, with this exact request argv."""
    result=systemctl(['systemctl','--user','show',unit+'.service','-p','ActiveState','-p','ControlGroup'],
                          capture_output=True,text=True,check=True)
    values=dict(line.split('=',1)for line in result.stdout.splitlines()if '='in line)
    group=values.get('ControlGroup','')
    if values.get('ActiveState')!='active' or not group.startswith('/')or '..'in Path(group).parts:return []
    root=cgroups/group.lstrip('/')
    if not root.is_dir():return []
    boot=(proc/'sys/kernel/random/boot_id').read_text().strip();found=[]
    pids=set()
    for path in root.rglob('cgroup.procs'):
        try:pids.update(map(int,path.read_text().split()))
        except FileNotFoundError:continue
    for pid in sorted(pids):
        directory=proc/str(pid)
        try:
            before=(directory/'stat').read_text().rsplit(')',1)[1].split()
            argv=[x.decode()for x in(directory/'cmdline').read_bytes().split(b'\0')if x]
            after=(directory/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):continue
        if before[19]!=after[19]or after[0]in('Z','X'):continue
        if _executes_request(argv,request_dir):
            found.append(dict(pid=pid,start_ticks=after[19],ppid=int(after[1]),boot_id=boot,argv=argv))
    return found


def dispatch_evidence(state_path, *, minimum_step=0, architecture=None,
                      verify=verify_request, processes=running_request_processes):
    """Return exact observed execution evidence, or None while not dispatched."""
    state_path=Path(state_path)
    if not state_path.exists():return None
    state_ref=binding(state_path);state=json.loads(state_path.read_bytes());active=state.get('active')
    if not active:return None
    if type(active.get('step'))is not int:raise ValueError('dispatch dependency step is invalid')
    if active['step']<minimum_step:return None
    directory=Path(active['request']);request_path=directory/'request.json';dispatch_path=directory/'dispatch.json'
    if not request_path.exists()or not dispatch_path.exists():return None
    request_ref=binding(request_path);dispatch_ref=binding(dispatch_path)
    request=json.loads(request_path.read_bytes());dispatch=json.loads(dispatch_path.read_bytes())
    provenance=request.get('evaluation_provenance')or{}
    if (request.get('native_state')is not True or request.get('step')!=active['step']
            or request.get('stage_out')!=active['stage_out']
            or provenance.get('checkpoint',{}).get('sha256')!=active['checkpoint_sha256']
            or provenance.get('source_checkpoint_sha256')!=active['checkpoint_sha256']
            or (architecture is not None and provenance.get('model_config',{}).get('architecture')!=architecture)):
        raise ValueError('dispatch dependency differs from its active checkpoint/request')
    expected_weights=directory/'weights.pt'
    if (Path(request['weights']).resolve()!=expected_weights.resolve()
            or Path(provenance['checkpoint']['path']).resolve()!=expected_weights.resolve()
            or not expected_weights.is_file()or expected_weights.stat().st_size!=provenance['checkpoint']['bytes']):
        raise ValueError('dispatch dependency checkpoint custody differs')
    verify(request)
    from scripts.task_eval_dispatch import unit_name
    unit=unit_name(directory,request)
    if dispatch.get('unit')!=unit or dispatch.get('placement')!='dgx-ledger':
        raise ValueError('dispatch dependency unit differs from its exact native request')
    observed=processes(unit,directory)
    if not observed:return None
    if binding(state_path)!=state_ref or binding(request_path)!=request_ref or binding(dispatch_path)!=dispatch_ref:
        raise ValueError('dispatch dependency changed during execution observation')
    return dict(schema='natlang.native-async-dispatch-observation/1',at=time.time(),
                state=state_ref,request=request_ref,dispatch=dispatch_ref,unit=unit,
                checkpoint_sha256=active['checkpoint_sha256'],step=active['step'],stage_out=request['stage_out'],
                architecture=provenance['model_config']['architecture'],processes=observed)


def gate(status, requirement, *, observe=dispatch_evidence):
    if requirement is None:return True
    path=Path(status)/'after-dispatch-satisfied.json'
    if path.exists():
        old=json.loads(path.read_text())
        evidence=old.get('evidence')or{}
        if (old.get('requirement')!=requirement or old.get('historical')is not True
                or evidence.get('schema')!='natlang.native-async-dispatch-observation/1'
                or evidence.get('state',{}).get('path')!=str(Path(requirement['state']).resolve())
                or type(evidence.get('step'))is not int or evidence['step']<requirement['minimum_step']
                or evidence.get('architecture')!=requirement['architecture']or not evidence.get('processes')):
            raise ValueError('historical dispatch dependency proof differs')
        from scripts.task_eval_dispatch import unit_name
        expected_unit=unit_name(None,dict(stage_out=evidence['stage_out'],step=evidence['step'],
            native_state=True,evaluation_provenance=dict(model_config=dict(architecture=evidence['architecture']))))
        directory=Path(evidence['request']['path']).parent
        if evidence['unit']!=expected_unit or Path(evidence['dispatch']['path'])!=directory/'dispatch.json':
            raise ValueError('historical dispatch dependency unit differs')
        for process in evidence['processes']:
            argv=process['argv']
            if not _executes_request(argv,directory):
                raise ValueError('historical dispatch dependency process differs')
        return True
    evidence=observe(requirement['state'],minimum_step=requirement['minimum_step'],architecture=requirement['architecture'])
    if evidence is None:return False
    write_json_durable(path,dict(requirement=requirement,evidence=evidence,historical=True))
    return True


def poll(config, *, observe=dispatch_evidence, ingest=poll_latest, available=None):
    status=Path(config['status']);status.mkdir(parents=True,exist_ok=True)
    if not gate(status,config.get('after_dispatch'),observe=observe):
        return dict(event='native_eval_waiting_for_prior_dispatch',dependency=config['after_dispatch'])
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=True)
    existing=out/'native-async-state.json'
    prior=json.loads(existing.read_text())if existing.exists()else{}
    spec=parse_task_eval(json.loads(Path(config['spec']).read_text()))
    active=bool(prior.get('active'))
    payload_root_value=config.get('external_payload_root')
    payload_root=Path(payload_root_value).resolve() if payload_root_value else None
    if payload_root is not None and not payload_root.is_dir():
        raise ValueError('configured external payload root must already exist')
    if payload_root is not None and payload_root.stat().st_dev==out.stat().st_dev:
        raise ValueError('external payload root shares the logical evaluation filesystem')
    external_floor=config.get('external_min_free_bytes',20<<30)
    if type(external_floor)is not int or external_floor<0 or (payload_root is not None and external_floor<20<<30):
        raise ValueError('external payload free-space floor is invalid or below 20 GiB')
    fetch_command=list(config['fetch_command'])
    if payload_root is not None:
        if '--payload-root'in fetch_command:
            raise ValueError('external payload root must be declared once in ingester config')
        fetch_command.extend(['--payload-root',str(payload_root)])
    def admit(selected,destination):
        exact=verified_export(selected)
        if exact['role']!=config['role']:raise ValueError('native ingestion export role differs')
        if selected['last_verified_step']<config['minimum_step']:
            return dict(event='native_eval_waiting_for_minimum_step',step=selected['last_verified_step'])
        owned=out/'checkpoints'/exact['identity']['sha256']/'task-eval/async'/f"step-{selected['last_verified_step']:07d}"/'weights.pt'
        if payload_root is not None:
            legacy_pending=destination.with_name(destination.name+'.pending')
            legacy_control=legacy_pending.with_name(legacy_pending.name+'.aria2')
            if legacy_pending.exists() or legacy_control.exists():
                return dict(event='native_eval_external_migration_required',
                            logical_partial=str(legacy_pending),control=str(legacy_control),
                            external_payload_root=str(payload_root))
            payload=payload_root/exact['identity']['sha256']/('checkpoint-'+exact['identity']['sha256']+'.pt')
            payload_pending=payload.with_name(payload.name+'.pending')
            if payload.is_file() and payload.stat().st_size==exact['identity']['bytes']:
                retained_external=exact['identity']['bytes']
            else:
                retained_external=(min(exact['identity']['bytes'],payload_pending.stat().st_blocks*512)
                                   if payload_pending.is_file() else 0)
            external_free=(available or shutil.disk_usage)(payload_root).free
            external_needed=max(0,exact['identity']['bytes']-retained_external)+external_floor
            if external_free<external_needed:
                return dict(event='native_eval_external_disk_wait',available_bytes=external_free,
                            needed_bytes=external_needed,free_floor_bytes=external_floor)
            local_free=(available or shutil.disk_usage)(out).free
            if local_free<config['disk_reserve_bytes']:
                return dict(event='native_eval_disk_wait',available_bytes=local_free,
                            needed_bytes=config['disk_reserve_bytes'])
            return None
        retained=max((p.stat().st_size for p in(destination,destination.with_name(destination.name+'.pending'),owned)if p.is_file()),default=0)
        needed=max(0,exact['identity']['bytes']-retained)+config['disk_reserve_bytes']
        free=(available or shutil.disk_usage)(out).free
        if free<needed:return dict(event='native_eval_disk_wait',available_bytes=free,needed_bytes=needed)
        return None
    if not active:
        source=selected_ingest_export(config['export_state'],prior);receipt=verified_export(source)
        decision=admit(source,out/'incoming'/('checkpoint-'+receipt['identity']['sha256']+'.pt'))
        if decision is not None:return decision
    # Recheck the selected receipt under the ingester's transfer lease. The
    # exporter may advance after the inexpensive preliminary floor check.
    return ingest(config['export_state'],out=out,spec=spec,base=config['base'],queue=config['queue'],
                  fetch_command=fetch_command,device=config['device'],export_admission=admit)


def pending_event(config):
    """Inspect an existing request without checkpoint decoding or transfer."""
    from .task_cadence import TaskEval
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=True)
    with(out/'export-ingest.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=out/'native-async-state.json'
        prior=json.loads(path.read_text())if path.exists()else{}
        active=prior.get('active')
        if not active:
            if prior.get('last_completed_sha256')and not any(key in prior for key in('fetching_export','sealing_request')):
                selected=selected_ingest_export(config['export_state'],prior);receipt=verified_export(selected)
                if receipt['role']!=config['role']:raise ValueError('native ingestion export role differs')
                if selected['last_verified_step']<config['minimum_step']:
                    return dict(event='native_eval_waiting_for_minimum_step',step=selected['last_verified_step'])
                if receipt['identity']['sha256']==prior['last_completed_sha256']:
                    return dict(event='native_eval_up_to_date',step=selected['last_verified_step'])
            return None
        if active.get('publication_pending'):return None
        evaluator=TaskEval(active['spec'],Path(active['stage_out']),None,
                           prompt_rendering=active['spec']['prompt_rendering'])
        if evaluator.landed(active['step'])is None or Path(active['request']).exists():
            return dict(event='native_eval_pending',active_step=active['step'],eligible=None)
    return None


def worker_state(unit, *, execute=subprocess.run):
    result=execute(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState','-p','Result','-p','ExecMainStatus','-p','MainPID'],
                   capture_output=True,text=True)
    values=dict(line.split('=',1)for line in result.stdout.splitlines()if '='in line)
    if result.returncode and values.get('LoadState')!='not-found':
        raise RuntimeError('cannot inspect ingestion worker unit')
    if int(values.get('MainPID')or 0):
        proc=Path('/proc')/values['MainPID']
        try:
            before=(proc/'stat').read_text().rsplit(')',1)[1].split()
            argv=[value.decode()for value in(proc/'cmdline').read_bytes().split(b'\0')if value]
            after=(proc/'stat').read_text().rsplit(')',1)[1].split()
        except(FileNotFoundError,ProcessLookupError):return values
        if before[19]==after[19]and after[0]not in('Z','X'):values['argv']=argv
    from scripts.memory_ledger import STATE
    ledger_path=Path(STATE)
    ledger=json.loads(ledger_path.read_text())if ledger_path.exists()else{}
    queued=ledger.get('waiting',{}).get(unit,{})
    if queued.get('requeued'):
        values['queued']=True;values['queued_argv']=queued['spec']['command']
    claim=ledger.get('claims',{}).get(unit,{})
    if claim and time.time()-claim['admitted']<30:
        values['launch_pending']=True;values['claimed_argv']=claim['command']
    return values


def _publish_latest_result(config):
    """Expose durable feedback independently of the next checkpoint transfer."""
    source=Path(config['out'])/'latest-result.json'
    target=Path(config['status'])/'latest-result.json'
    if source.exists():
        value=json.loads(source.read_text())
        try:prior=json.loads(target.read_text())
        except (OSError,json.JSONDecodeError,UnicodeDecodeError):prior=None
        if prior!=value:
            write_json_durable(target,value)


def ledgered_poll(config, *, execute=subprocess.run, inspect=worker_state, sleep=time.sleep):
    """One separately admitted transfer/sealing poll, recoverable by its owner."""
    from ..common.hashing import canonical_json_sha256_hex
    status=Path(config['status']);status.mkdir(parents=True,exist_ok=True)
    options=config['transfer_ledger'];repo=Path(__file__).resolve().parents[4]
    configuration={key:value for key,value in config.items()if key!='transfer_ledger'}
    mechanism=binding(__file__)
    identity=canonical_json_sha256_hex(dict(configuration=configuration,options=options,mechanism=mechanism))
    intent_path=status/'ledger-poll.json';result_path=status/'ledger-poll-result.json'
    config_path=status/'ledger-poll-config.json'
    unit='native-eval-ingest-'+config['role']+'-'+hashlib.sha256(str(status.resolve()).encode()).hexdigest()[:12]+'.service'
    if intent_path.exists():
        intent=json.loads(intent_path.read_text())
        if binding(config_path)!=intent['config']:raise ValueError('ingestion worker configuration changed')
        document=json.loads(config_path.read_text())
        expected=canonical_json_sha256_hex(dict(configuration=configuration,options=options,mechanism=document['mechanism']))
        if intent['identity']!=expected or intent['unit']!=unit or document['configuration']!=configuration:
            raise ValueError('existing ingestion worker configuration changed; reconcile its exact intent')
    else:
        token=uuid.uuid4().hex
        write_json_durable(config_path,dict(configuration=configuration,result=str(result_path),token=token,
                                           mechanism=mechanism))
        intent=dict(identity=identity,unit=unit,config=binding(config_path),token=token)
        write_json_durable(intent_path,intent)
        document=json.loads(config_path.read_text())
    def completed():
        if not result_path.exists():return None
        result=json.loads(result_path.read_text())
        if (result.get('token')==intent['token']and result.get('configuration')==intent['config']
                and result.get('mechanism')==document['mechanism']):return result['event']
        return None
    expected_argv=[sys.executable,'-m','natlang_neuralese.eval.native_async_service',
                   '--ledger-poll-config',str(config_path),'--ledger-poll-sha256',intent['config']['sha256']]
    def observed():
        value=inspect(unit)
        for key in('argv','queued_argv','claimed_argv'):
            if value.get(key)is not None and value[key]!=expected_argv:
                raise ValueError('live ingestion worker argv differs from its journaled owner')
        return value
    state=observed()
    live=lambda value:bool(value.get('queued')or value.get('launch_pending')or value.get('ActiveState')in('active','activating','deactivating','reloading'))
    if not live(state)and completed()is None:
        if document['mechanism']!=mechanism:
            # No old worker is running, and no authentic success exists.
            # A fresh poll retries the durable protocol with the new code.
            intent_path.unlink()
            raise RuntimeError('unfinished ingestion worker mechanism changed; retry with current code')
        # Rejoin the ledger's exact resumable worker after preemption. Only a
        # terminal failure with no queued/relaunching owner gets a fresh poll.
        execute([sys.executable,str(repo/'scripts/memory_ledger.py'),'run','--unit',unit,
                 '--budget-gb',str(options['budget_gib']),'--reserve-gb',str(options['reserve_gib']),
                 '--priority','low','--resumable','--wait','86400','--workdir',str(repo),
                 '--env','PYTHONPATH='+str(repo/'training/neuralese')+os.pathsep+str(repo),
                 '--',*expected_argv],check=True)
        state=observed()
    while live(state):
        _publish_latest_result(config)
        sleep(2);state=observed()
    event=completed()
    # A failed unit must not turn an old or absent result into success.
    if state.get('Result')!='success'or state.get('ExecMainStatus')!='0'or event is None:
        intent_path.unlink()
        raise RuntimeError('admitted ingestion worker failed; durable transfer state retained')
    intent_path.unlink()
    return event


def worker_poll(path, sha256):
    path=Path(path)
    if binding(path)['sha256']!=sha256:raise ValueError('ingestion worker config hash differs')
    document=json.loads(path.read_text());configuration=document['configuration']
    if document['mechanism']!=binding(__file__):raise ValueError('ingestion worker mechanism changed before execution')
    with(Path(configuration['status'])/'ledger-worker.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        event=poll(configuration)
        if document['mechanism']!=binding(__file__):raise ValueError('ingestion worker mechanism changed during execution')
        write_json_durable(document['result'],dict(token=document['token'],event=event,
                            configuration=binding(path),mechanism=document['mechanism']))


def run(config, *, every=60, max_polls=None):
    status=Path(config['status']);status.mkdir(parents=True,exist_ok=True)
    with(status/'service.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json_durable(status/'started.json',dict(at=time.time(),pid=os.getpid(),configuration=config,mechanism=binding(__file__)))
        count=0
        while max_polls is None or count<max_polls:
            # A prior poll may have drained a result before blocking in the
            # next checkpoint transfer. Publish that durable result on restart.
            _publish_latest_result(config)
            try:
                if config.get('transfer_ledger'):
                    if(status/'ledger-poll.json').exists():event=ledgered_poll(config)
                    else:
                        event=pending_event(config)
                        if event is None:event=ledgered_poll(config)
                else:event=poll(config)
            except Exception as error:event=dict(event='native_eval_ingest_retry',error_type=type(error).__name__)
            _publish_latest_result(config)
            event=dict(at=time.time(),**event);write_json_durable(status/'status.json',event)
            print(json.dumps(event),flush=True);count+=1
            if max_polls is None or count<max_polls:time.sleep(every)


def main(argv=None):
    argv=list(sys.argv[1:]if argv is None else argv)
    if argv[:1]==['--ledger-poll-config']:
        worker=argparse.ArgumentParser()
        worker.add_argument('--ledger-poll-config',type=Path,required=True)
        worker.add_argument('--ledger-poll-sha256',required=True)
        args=worker.parse_args(argv);worker_poll(args.ledger_poll_config,args.ledger_poll_sha256);return
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('export-state','out','spec','base','status'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--queue',type=Path,default=None);parser.add_argument('--role',choices=('lfm','mellum'),required=True)
    parser.add_argument('--minimum-step',type=int,default=0);parser.add_argument('--disk-reserve-gib',type=int,default=2)
    parser.add_argument('--after-dispatch-state',type=Path);parser.add_argument('--after-dispatch-minimum-step',type=int,default=0)
    parser.add_argument('--after-dispatch-architecture',choices=('lfm2','lfm2_moe','maple','mellum'))
    parser.add_argument('--device',default='cuda');parser.add_argument('--every',type=float,default=60)
    parser.add_argument('--external-payload-root',type=Path)
    parser.add_argument('--external-min-free-gib',type=int,default=20)
    parser.add_argument('--transfer-budget-gib',type=float)
    parser.add_argument('--transfer-reserve-gib',type=float,default=1)
    parser.add_argument('--fetch-command',nargs=argparse.REMAINDER,required=True)
    args=parser.parse_args(argv)
    import math
    if args.transfer_budget_gib is not None and(not math.isfinite(args.transfer_budget_gib)or args.transfer_budget_gib<=0
            or not math.isfinite(args.transfer_reserve_gib)or args.transfer_reserve_gib<0):parser.error('invalid transfer ledger allocation')
    if (not args.fetch_command or args.every<=0 or min(args.minimum_step,args.after_dispatch_minimum_step,args.disk_reserve_gib,args.external_min_free_gib)<0
            or (args.external_payload_root is not None and args.external_min_free_gib<20)):
        parser.error('invalid native ingestion limits/fetch command or external free-space floor below 20 GiB')
    if bool(args.after_dispatch_state)!=bool(args.after_dispatch_architecture):parser.error('dispatch dependency needs state and architecture')
    config={key:str(getattr(args,key).resolve())for key in('export_state','out','spec','base','status')}
    config.update(role=args.role,minimum_step=args.minimum_step,disk_reserve_bytes=args.disk_reserve_gib*(1<<30),
                  queue=str((args.queue or request_queue()).resolve()),fetch_command=args.fetch_command,device=args.device,
                  after_dispatch=(dict(state=str(args.after_dispatch_state.resolve()),minimum_step=args.after_dispatch_minimum_step,architecture=args.after_dispatch_architecture)if args.after_dispatch_state else None))
    if args.external_payload_root is not None:
        config['external_payload_root']=str(args.external_payload_root.resolve())
        config['external_min_free_bytes']=args.external_min_free_gib*(1<<30)
    if args.transfer_budget_gib is not None:
        config['transfer_ledger']=dict(budget_gib=args.transfer_budget_gib,reserve_gib=args.transfer_reserve_gib)
    run(config,every=args.every)


if __name__=='__main__':main()
