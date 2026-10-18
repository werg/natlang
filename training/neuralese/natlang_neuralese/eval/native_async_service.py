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
import time

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
    source=selected_ingest_export(config['export_state'],prior);receipt=verified_export(source)
    if receipt['role']!=config['role']:raise ValueError('native ingestion export role differs')
    if source['last_verified_step']<config['minimum_step']:
        return dict(event='native_eval_waiting_for_minimum_step',step=source['last_verified_step'])
    spec=parse_task_eval(json.loads(Path(config['spec']).read_text()))
    active=bool(prior.get('active'))
    def admit(selected,destination):
        exact=verified_export(selected)
        if exact['role']!=config['role']:raise ValueError('native ingestion export role differs')
        if selected['last_verified_step']<config['minimum_step']:
            return dict(event='native_eval_waiting_for_minimum_step',step=selected['last_verified_step'])
        retained=max((p.stat().st_size for p in(destination,destination.with_name(destination.name+'.pending'))if p.is_file()),default=0)
        needed=max(0,exact['identity']['bytes']-retained)+config['disk_reserve_bytes']
        free=(available or shutil.disk_usage)(out).free
        if free<needed:return dict(event='native_eval_disk_wait',available_bytes=free,needed_bytes=needed)
        return None
    if not active:
        decision=admit(source,out/'incoming'/('checkpoint-'+receipt['identity']['sha256']+'.pt'))
        if decision is not None:return decision
    # Recheck the selected receipt under the ingester's transfer lease. The
    # exporter may advance after the inexpensive preliminary floor check.
    return ingest(config['export_state'],out=out,spec=spec,base=config['base'],queue=config['queue'],
                  fetch_command=config['fetch_command'],device=config['device'],export_admission=admit)


def run(config, *, every=60, max_polls=None):
    status=Path(config['status']);status.mkdir(parents=True,exist_ok=True)
    with(status/'service.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json_durable(status/'started.json',dict(at=time.time(),pid=os.getpid(),configuration=config,mechanism=binding(__file__)))
        count=0
        while max_polls is None or count<max_polls:
            try:event=poll(config)
            except Exception as error:event=dict(event='native_eval_ingest_retry',error_type=type(error).__name__)
            latest=Path(config['out'])/'latest-result.json'
            if latest.exists():write_json_durable(status/'latest-result.json',json.loads(latest.read_text()))
            event=dict(at=time.time(),**event);write_json_durable(status/'status.json',event)
            print(json.dumps(event),flush=True);count+=1
            if max_polls is None or count<max_polls:time.sleep(every)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('export-state','out','spec','base','status'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--queue',type=Path,default=None);parser.add_argument('--role',choices=('lfm','mellum'),required=True)
    parser.add_argument('--minimum-step',type=int,default=0);parser.add_argument('--disk-reserve-gib',type=int,default=2)
    parser.add_argument('--after-dispatch-state',type=Path);parser.add_argument('--after-dispatch-minimum-step',type=int,default=0)
    parser.add_argument('--after-dispatch-architecture',choices=('lfm2','lfm2_moe','maple','mellum'))
    parser.add_argument('--device',default='cuda');parser.add_argument('--every',type=float,default=60)
    parser.add_argument('--fetch-command',nargs=argparse.REMAINDER,required=True)
    args=parser.parse_args(argv)
    if not args.fetch_command or args.every<=0 or min(args.minimum_step,args.after_dispatch_minimum_step,args.disk_reserve_gib)<0:parser.error('invalid native ingestion limits/fetch command')
    if bool(args.after_dispatch_state)!=bool(args.after_dispatch_architecture):parser.error('dispatch dependency needs state and architecture')
    config={key:str(getattr(args,key).resolve())for key in('export_state','out','spec','base','status')}
    config.update(role=args.role,minimum_step=args.minimum_step,disk_reserve_bytes=args.disk_reserve_gib*(1<<30),
                  queue=str((args.queue or request_queue()).resolve()),fetch_command=args.fetch_command,device=args.device,
                  after_dispatch=(dict(state=str(args.after_dispatch_state.resolve()),minimum_step=args.after_dispatch_minimum_step,architecture=args.after_dispatch_architecture)if args.after_dispatch_state else None))
    run(config,every=args.every)


if __name__=='__main__':main()
