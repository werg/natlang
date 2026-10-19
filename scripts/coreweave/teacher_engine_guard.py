"""Outside-service process custody and measured RAM protection for a bound engine."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
try:
    from .teacher_generation_guardian import binding,write,process,public_identity,alive
except ImportError:
    from teacher_generation_guardian import binding,write,process,public_identity,alive


def ram_sample(root=Path('/sys/fs/cgroup'), *, meminfo=Path('/proc/meminfo')):
    def number(name):return int((root/name).read_text().strip())
    def pairs(name):return {key:int(value)for key,value in (line.split()for line in(root/name).read_text().splitlines())}
    current,limit=number('memory.current'),number('memory.max')
    stat,events=pairs('memory.stat'),pairs('memory.events')
    required=('anon','kernel','shmem','file_dirty','file_writeback','unevictable')
    if any(key not in stat or stat[key]<0 for key in required)or current<0 or limit<=0:
        raise ValueError('invalid actual cgroup RAM accounting')
    # Conservative bound: unevictable may overlap anon/shmem. Clean mapped
    # source pages remain reclaimable; memory.current alone is not RSS/OOM.
    protected=min(current,sum(stat[key]for key in required))
    fields = [line.split() for line in Path(meminfo).read_text().splitlines() if line.startswith('MemAvailable:')]
    if len(fields) != 1 or len(fields[0]) != 3 or fields[0][2] != 'kB':
        raise ValueError('invalid actual guest MemAvailable accounting')
    guest_available = int(fields[0][1]) * 1024
    if guest_available < 0:
        raise ValueError('invalid actual guest MemAvailable accounting')
    cgroup_headroom = limit - protected
    return {'current':current,'limit':limit,'nonreclaimable_upper_bound':protected,
        'reclaimable_lower_bound':max(0,current-protected),'headroom_lower_bound':min(cgroup_headroom, guest_available),
        'cgroup_headroom_lower_bound':cgroup_headroom,'guest_available':guest_available,
        'stat':stat,'events':events}


def ram_threat(sample,reserve,initial_events):
    if type(reserve)is not int or reserve<=0 or reserve>=sample['limit']:
        raise ValueError('RAM reserve must be an explicit positive part of actual limit')
    return (sample['headroom_lower_bound']<reserve
        or any(sample['events'].get(key,0)>initial_events.get(key,0)for key in('oom','oom_kill','oom_group_kill')))


def session_process(pid):
    """Include an unreaped leader zombie so its session birth stays provable."""
    try:
        root=Path('/proc',str(pid));raw=(root/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid),'state':raw[0],'ppid':int(raw[1]),'pgid':int(raw[2]),
            'session':int(raw[3]),'start_ticks':raw[19],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except(FileNotFoundError,ProcessLookupError):return None


def same(a,b):
    return bool(a and b and all(a[key]==b[key]for key in('pid','start_ticks','boot_id')))


def session_members(owner, *, session_only=False):
    guard=owner['guard'];actual=session_process(guard['pid'])
    if not same(actual,guard):raise RuntimeError('engine guard identity changed')
    table={}
    for root in Path('/proc').iterdir():
        if root.name.isdigit():
            current=session_process(root.name)
            if current and current['state']not in('Z','X'):table[current['pid']]=current
    leader=session_process(owner['pid']);session_valid=leader is None or same(leader,owner)
    selected=set() if session_only else {guard['pid']}
    selected.update(pid for pid,current in table.items()if session_valid and current['session']==owner['pid']
        and current['boot_id']==owner['boot_id']and int(current['start_ticks'])>=int(owner['start_ticks']))
    while True:
        expanded=selected|{pid for pid,current in table.items()if current['ppid']in selected}
        if expanded==selected:break
        selected=expanded
    return [table[pid]for pid in selected if pid!=guard['pid']and pid in table]


def subreaper(enable):
    libc=ctypes.CDLL(None,use_errno=True);previous=ctypes.c_int()
    if libc.prctl(37,ctypes.byref(previous),0,0,0)!=0:raise OSError(ctypes.get_errno(),'get child subreaper')
    if libc.prctl(36,int(enable),0,0,0)!=0:raise OSError(ctypes.get_errno(),'set child subreaper')
    return bool(previous.value)


def signal_members(owner,members,signum):
    for expected in members:
        current=session_process(expected['pid'])
        if same(current,expected)and current['state']not in('Z','X'):
            try:os.kill(current['pid'],signum)
            except ProcessLookupError:pass


def cleanup_session(owner,child,grace,*,session_only=False):
    """Stop every live owned-session descendant even after its leader exits."""
    started=time.monotonic();observed={};sent=set();escalated=False
    while True:
        members=session_members(owner,session_only=session_only)
        for item in members:observed[(item['pid'],item['start_ticks'])]=item
        if not members:break
        if time.monotonic()-started>=grace:
            signal_members(owner,members,signal.SIGKILL);escalated=True
        else:
            fresh=[item for item in members if(item['pid'],item['start_ticks'])not in sent]
            signal_members(owner,fresh,signal.SIGTERM)
            sent.update((item['pid'],item['start_ticks'])for item in fresh)
        time.sleep(.02)
    code=child.wait();reaped=0
    if session_only:
        # Concurrent collectors share a parent. Reap only observed owned
        # descendants, never another worker's Popen child/status.
        for expected in observed.values():
            actual=session_process(expected['pid'])
            if expected['pid']==owner['pid'] or not same(actual,expected):continue
            try:
                pid,_=os.waitpid(expected['pid'],os.WNOHANG)
                reaped+=bool(pid)
            except ChildProcessError:pass
    else:
        while True:
            try:
                pid,_=os.waitpid(-1,os.WNOHANG)
                if not pid:break
                reaped+=1
            except ChildProcessError:break
    return {'group_gone':True,'reaped_orphans':reaped,'observed_members':list(observed.values()),
        'sigkill_required':escalated,'leader_returncode':code}


def run(spec_path,out,*,sample=ram_sample,pause=time.sleep):
    spec_path,out=Path(spec_path),Path(out);spec=json.loads(spec_path.read_text())
    if spec.get('schema')!='natlang.teacher-engine-process-guard/1':raise ValueError('unknown bound engine guard specification')
    argv=spec.get('argv');reserve=spec.get('ram_reserve_bytes');grace=spec.get('shutdown_grace_seconds',30.)
    if not isinstance(argv,list)or not argv or any(not isinstance(x,str)or not x for x in argv):raise ValueError('engine argv is not exact')
    if type(grace)not in(int,float)or not 0<grace<=30:raise ValueError('invalid explicit shutdown grace')
    for ref in spec['assets']:
        if binding(ref['path'])!=ref:raise ValueError('engine asset changed: '+ref['path'])
    first=sample()
    if ram_threat(first,reserve,first['events']):raise ValueError('engine startup lacks declared RAM reserve')
    out.mkdir(parents=True,exist_ok=True)
    with(out/'owner.lock').open('a')as lease:
        import fcntl
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(out/'intent.json').exists():raise ValueError('previous engine launch requires exact reconciliation')
        write(out/'intent.json',{'at':time.time(),'specification':binding(spec_path),'initial_ram':first,
            'scope':'bound engine process only; no generation/control/admission claim'})
        # This service is a dedicated parent, with no unrelated child custody.
        guard=session_process(os.getpid())
        if any((row:=session_process(p.name))and row['ppid']==guard['pid']for p in Path('/proc').iterdir()if p.name.isdigit()):
            raise ValueError('engine guard must not own unrelated existing children')
        was_subreaper=subreaper(True)
        env={**os.environ,**spec.get('environment',{})}
        received=[]
        def stop(signum,frame):received.append(signum)
        previous={sig:signal.signal(sig,stop)for sig in(signal.SIGTERM,signal.SIGINT)}
        child=None;owner=None;error=None;last=first;custody=None
        reason='engine_exit';peak={key:first[key]for key in('current','nonreclaimable_upper_bound')}
        try:
            with(out/'engine.log').open('ab',buffering=0)as log:
                child=subprocess.Popen(argv,cwd=spec['cwd'],env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owner=session_process(child.pid)
            if owner is None:raise RuntimeError('owned child lost before session identity publication')
            owner['guard']=guard
            if owner['session']!=owner['pid']or owner['pgid']!=owner['pid']:
                raise RuntimeError('owned child did not create its independent session')
            write(out/'spawn.json',{'at':time.time(),'identity':owner,'intent':binding(out/'intent.json'),'argv':argv})
            while child.poll()is None:
                if received:reason='guard_signal';break
                last=sample()
                for key in peak:peak[key]=max(peak[key],last[key])
                progress={'at':time.time(),'identity':owner,'ram':last,'peak':peak,'reserve_bytes':reserve}
                write(out/'progress.json',progress)
                if ram_threat(last,reserve,first['events']):
                    reason='observed_ram_threat';write(out/'ram-stop-intent.json',progress);break
                pause(1)
        except BaseException as caught:
            error=caught;reason='guard_error'
            write(out/'guard-error.json',{'at':time.time(),'type':type(caught).__name__,'error':str(caught),'identity':owner})
        finally:
            try:
                if child is not None and owner is not None:custody=cleanup_session(owner,child,grace)
            finally:
                for sig,handler in previous.items():signal.signal(sig,handler)
                subreaper(was_subreaper)
        code=child.returncode if child is not None else None
        terminal={'at':time.time(),'identity':owner,'returncode':code,'reason':reason,'signals':received,
            'peak':peak,'last_ram':last,'custody':custody,'log':binding(out/'engine.log')if(out/'engine.log').exists()else None,
            'optimizer_updates':0,'scope':'actual engine exit/resource telemetry, not task quality or generated data'}
        write(out/'terminal.json',terminal)
        if error is not None:raise error
        return code if reason=='engine_exit'else 75


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);return run(a.spec,a.out)

if __name__=='__main__':raise SystemExit(main())
