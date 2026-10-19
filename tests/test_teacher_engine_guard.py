import json
import functools
import inspect
import subprocess
import sys
from pathlib import Path
import pytest
from scripts.coreweave import teacher_engine_guard as guard


def isolated_guard_owner(test):
    """Give the production subreaper a dedicated process, not pytest's children."""
    @functools.wraps(test)
    def execute(*args, **kwargs):
        arguments = dict(inspect.signature(test).bind(*args, **kwargs).arguments)
        arguments['tmp_path'] = str(arguments['tmp_path'])
        code = ('import runpy,pathlib,json,sys; namespace=runpy.run_path(sys.argv[1]); '
                'arguments=json.loads(sys.argv[3]); '
                "arguments['tmp_path']=pathlib.Path(arguments['tmp_path']); "
                'namespace[sys.argv[2]].__wrapped__(**arguments)')
        result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).resolve()),
                                 test.__name__, json.dumps(arguments)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    return execute


def sample(current=95,anon=12,file=80):
    return {'current':current,'limit':100,'nonreclaimable_upper_bound':anon,
        'headroom_lower_bound':100-anon,'reclaimable_lower_bound':current-anon,
        'events':{'oom':0,'oom_kill':0},'stat':{'anon':anon,'file':file}}


def test_large_clean_mapped_cache_is_not_anonymous_oom():
    assert not guard.ram_threat(sample(),16,{'oom':0,'oom_kill':0})
    assert guard.ram_threat(sample(99,86,13),16,{'oom':0,'oom_kill':0})
    value=sample();value['events']['oom_kill']=1
    assert guard.ram_threat(value,16,{'oom':0,'oom_kill':0})


def test_actual_cgroup_accounting_keeps_nonreclaimable_and_cache_distinct(tmp_path):
    for name,value in {'memory.current':'95','memory.max':'100','memory.events':'oom 0\noom_kill 0\n',
        'memory.stat':'anon 10\nkernel 2\nshmem 3\nfile_dirty 1\nfile_writeback 0\nunevictable 4\nfile 80\n'}.items():
        (tmp_path/name).write_text(value)
    meminfo=tmp_path/'meminfo';meminfo.write_text('MemAvailable: 100 kB\n')
    value=guard.ram_sample(tmp_path,meminfo=meminfo)
    assert value['nonreclaimable_upper_bound']==20 and value['reclaimable_lower_bound']==75
    assert value['headroom_lower_bound']==80


@pytest.mark.parametrize('code',[0,7])
@isolated_guard_owner
def test_actual_child_terminal_custody_and_duplicate_launch_rejected(tmp_path,code):
    spec={'schema':'natlang.teacher-engine-process-guard/1','argv':[sys.executable,'-c',f'raise SystemExit({code})'],
        'cwd':str(tmp_path),'assets':[],'ram_reserve_bytes':16}
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec));out=tmp_path/'out'
    assert guard.run(path,out,sample=sample,pause=lambda seconds:None)==code
    receipt=json.loads((out/'terminal.json').read_text());assert receipt['returncode']==code
    assert receipt['reason']=='engine_exit' and receipt['optimizer_updates']==0
    with pytest.raises(ValueError,match='previous engine launch'):
        guard.run(path,out,sample=sample)


def test_low_memory_rejects_before_launch_or_intent(tmp_path):
    path=tmp_path/'spec.json';path.write_text(json.dumps({'schema':'natlang.teacher-engine-process-guard/1',
        'argv':[sys.executable,'-c','raise AssertionError()'],'cwd':str(tmp_path),'assets':[],'ram_reserve_bytes':16}))
    with pytest.raises(ValueError,match='lacks declared RAM reserve'):
        guard.run(path,tmp_path/'out',sample=lambda:sample(99,90,9))
    assert not(tmp_path/'out').exists()


@isolated_guard_owner
def test_leader_exit_still_drains_real_signal_ignoring_detached_grandchild(tmp_path):
    pidfile=tmp_path/'grandchild.pid'
    grandchild="import os,signal,time,pathlib;os.setsid();signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path("+repr(str(pidfile))+").write_text(str(os.getpid()));time.sleep(60)"
    parent="import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-c',"+repr(grandchild)+"]);p=pathlib.Path("+repr(str(pidfile))+");\nwhile not p.exists():time.sleep(.01)\n"
    path=tmp_path/'spec.json';path.write_text(json.dumps({'schema':'natlang.teacher-engine-process-guard/1',
        'argv':[sys.executable,'-c',parent],'cwd':str(tmp_path),'assets':[],'ram_reserve_bytes':16,'shutdown_grace_seconds':.1}))
    assert guard.run(path,tmp_path/'out',sample=sample,pause=lambda _:__import__('time').sleep(.01))==0
    receipt=json.loads((tmp_path/'out/terminal.json').read_text())
    assert receipt['custody']['group_gone']and receipt['custody']['sigkill_required']
    assert not Path('/proc',pidfile.read_text()).exists()


@isolated_guard_owner
def test_observer_exception_cleans_real_child_before_rethrow(tmp_path):
    path=tmp_path/'spec.json';path.write_text(json.dumps({'schema':'natlang.teacher-engine-process-guard/1',
        'argv':[sys.executable,'-c','import time;time.sleep(60)'],'cwd':str(tmp_path),'assets':[],
        'ram_reserve_bytes':16,'shutdown_grace_seconds':.1}))
    calls=[]
    def observe():
        calls.append(1)
        if len(calls)>1:raise RuntimeError('actual observer failure')
        return sample()
    with pytest.raises(RuntimeError,match='actual observer failure'):guard.run(path,tmp_path/'out',sample=observe)
    receipt=json.loads((tmp_path/'out/terminal.json').read_text());assert receipt['reason']=='guard_error'
    assert receipt['custody']['group_gone']and not Path('/proc',str(receipt['identity']['pid'])).exists()


def test_isolated_guard_does_not_claim_or_stop_the_test_hosts_other_child(tmp_path):
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    try:
        test_actual_child_terminal_custody_and_duplicate_launch_rejected(tmp_path, 0)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_real_guard_sigterm_forwards_and_reaps_signal_ignoring_child(tmp_path):
    import os,signal,subprocess,time
    spec=tmp_path/'spec.json';out=tmp_path/'out'
    spec.write_text(json.dumps({'schema':'natlang.teacher-engine-process-guard/1',
        'argv':[sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],
        'cwd':str(tmp_path),'assets':[],'ram_reserve_bytes':16,'shutdown_grace_seconds':.1}))
    code='from scripts.coreweave.teacher_engine_guard import run;import sys;raise SystemExit(run(sys.argv[1],sys.argv[2],sample=lambda:'+repr(sample())+'))'
    child=subprocess.Popen([sys.executable,'-c',code,str(spec),str(out)],cwd=Path(__file__).resolve().parents[1])
    try:
        end=time.monotonic()+10
        while not(out/'spawn.json').exists():
            assert child.poll()is None and time.monotonic()<end
            time.sleep(.01)
        time.sleep(.1);os.kill(child.pid,signal.SIGTERM)
        assert child.wait(timeout=5)==75
        terminal=json.loads((out/'terminal.json').read_text())
        assert terminal['reason']=='guard_signal'and terminal['signals']==[signal.SIGTERM]
        assert terminal['custody']['group_gone']and terminal['custody']['sigkill_required']
        assert not Path('/proc',str(terminal['identity']['pid'])).exists()
    finally:
        if child.poll()is None:child.terminate();child.wait(timeout=5)


@pytest.mark.parametrize('guest_kb,cgroup_limit,expected', [(3,10000,3072),(100,10000,9980),(0,10000,0)])
def test_actual_global_or_cgroup_headroom_both_bound_reserve(tmp_path,guest_kb,cgroup_limit,expected):
    for name,value in {'memory.current':'95','memory.max':str(cgroup_limit),'memory.events':'oom 0\noom_kill 0\n',
        'memory.stat':'anon 10\nkernel 2\nshmem 3\nfile_dirty 1\nfile_writeback 0\nunevictable 4\nfile 80\n'}.items():
        (tmp_path/name).write_text(value)
    meminfo=tmp_path/'meminfo';meminfo.write_text(f'MemAvailable: {guest_kb} kB\n')
    actual=guard.ram_sample(tmp_path,meminfo=meminfo)
    assert actual['headroom_lower_bound']==expected
    assert actual['cgroup_headroom_lower_bound']==cgroup_limit-20
    assert actual['guest_available']==guest_kb*1024
    assert guard.ram_threat(actual,4096,actual['events']) == (expected<4096)


@pytest.mark.parametrize('text',['MemTotal: 100 kB\n','MemAvailable: -1 kB\n','MemAvailable: 1 MB\n',
    'MemAvailable: 1 kB\nMemAvailable: 2 kB\n'])
def test_global_memory_accounting_must_be_present_and_valid(tmp_path,text):
    for name,value in {'memory.current':'95','memory.max':'100','memory.events':'oom 0\n',
        'memory.stat':'anon 10\nkernel 2\nshmem 3\nfile_dirty 1\nfile_writeback 0\nunevictable 4\n'}.items():
        (tmp_path/name).write_text(value)
    meminfo=tmp_path/'meminfo';meminfo.write_text(text)
    with pytest.raises(ValueError,match='guest MemAvailable'):guard.ram_sample(tmp_path,meminfo=meminfo)
