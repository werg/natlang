#!/usr/bin/env python3
"""Persistent post-exit owner for DiLoCo sandbox roles.

The bootstrap remains the provider entrypoint after its original child exits.
A controller publishes one atomic request.json; this process starts/stops and
reaps its own replacement children, retaining an auditable role journal.
"""
from __future__ import annotations
import hashlib, json, os, signal, subprocess, sys, time, uuid
from natlang_neuralese.common.jsonio import write_json_durable
from pathlib import Path


def _atomic(path: Path, value: dict) -> None:
    write_json_durable(path, value)


def main(directory: Path) -> None:
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    roles={};stopping=False
    path=directory/'state.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    generation=previous.get('generation',0)
    events=previous.get('events',[])
    seen={event['request_id'] for event in events if 'request_id' in event}
    owner_id=str(uuid.uuid4())
    ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    state={'schema':'natlang.diloco-role-supervisor/1','boot_id':boot,'supervisor_pid':os.getpid(),
           'supervisor_start_ticks':ticks,'owner_id':owner_id,'generation':generation,'roles':{},'events':events}
    state['events'].append({'event':'owner_started','owner_id':owner_id,'boot_id':boot,'pid':os.getpid(),'start_ticks':ticks})
    def save():
        state['generation']=generation
        state['roles']={role:{'pid':p.pid,'alive':p.poll() is None,'returncode':p.poll()} for role,p in roles.items()}
        _atomic(directory/'state.json',state)
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    save()
    while not stopping:
        for role,p in list(roles.items()):
            if p.poll() is not None and state['roles'].get(role,{}).get('returncode') is None:
                state['events'].append({'event':'exited','role':role,'pid':p.pid,'returncode':p.returncode});save()
        request=directory/'request.json'
        if request.exists():
            claimed=directory/'request.claimed.json'
            v=None
            try:
                os.replace(request,claimed);v=json.loads(claimed.read_text())
                if v.get('schema')!='natlang.diloco-role-request/1' or not isinstance(v.get('id'),str) or not v['id']:
                    raise ValueError('invalid request identity')
                if v['id'] in seen:raise ValueError('duplicate request')
                if v.get('owner_id')!=owner_id:raise ValueError('stale supervisor owner')
                if v.get('expected_generation')!=generation:raise ValueError('stale supervisor generation')
                role=v.get('role');action=v.get('action')
                if role not in ('coordinator','worker'):raise ValueError('invalid role')
                prior=roles.get(role)
                if action=='start':
                    if prior and prior.poll() is None:raise ValueError('role already alive')
                    argv=v.get('argv');env=v.get('env',{})
                    if not isinstance(argv,list) or not argv or not all(isinstance(x,str) and x for x in argv):raise ValueError('invalid argv')
                    if not isinstance(env,dict) or not all(isinstance(k,str) and isinstance(x,str) for k,x in env.items()):raise ValueError('invalid environment')
                    child_env={**os.environ,**env}
                    if role=='coordinator':child_env['CUDA_VISIBLE_DEVICES']=''
                    log=(directory/(role+'.log')).open('ab',buffering=0)
                    try:roles[role]=subprocess.Popen(argv,env=child_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    finally:log.close()
                elif action=='stop':
                    if not prior or prior.poll() is not None:raise ValueError('role not alive')
                    if v.get('expected_pid')!=prior.pid:raise ValueError('role ownership changed')
                    prior.terminate()
                else:raise ValueError('invalid action')
                seen.add(v['id']);generation+=1
                event={'event':action,'role':role,'request_id':v['id'],'pid':roles[role].pid}
                if action=='start':event['argv_sha256']=hashlib.sha256(json.dumps(argv,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                state['events'].append(event)
                save();_atomic(directory/'result.json',{'id':v['id'],'accepted':True,'generation':generation,'pid':roles[role].pid})
            except Exception as e:
                _atomic(directory/'result.json',{'id':v.get('id') if isinstance(v,dict) else None,'accepted':False,'error':str(e)[:500],'generation':generation})
            finally:claimed.unlink(missing_ok=True)
        time.sleep(.2)
    # Preserve final updates: wait for the learner to drain before stopping the coordinator.
    for role in ('worker','coordinator'):
        p=roles.get(role)
        if p and p.poll() is None:
            p.terminate();p.wait()
        if p:state['events'].append({'event':'shutdown','role':role,'pid':p.pid,'returncode':p.returncode})
    save()


if __name__=='__main__':main(Path(sys.argv[1]))
