"""Data tools can advance independently of an already running learner/runtime."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/neuralese_data'),str(REPO/'training/neuralese'),str(REPO)]
import expert_iteration as ei


def driver(tmp_path,monkeypatch,extra=()):
    captured={}
    monkeypatch.setattr(sys,'argv',['expert_iteration.py','async-drive','--root',str(tmp_path/'loop'),
        '--name','aei','--start','/checkpoint/start.pt','--start-sha','a'*64,'--commit','b'*40,
        '--pin',str(tmp_path/'runtime'),'--sft-work',str(tmp_path/'work'),*map(str,extra)])
    monkeypatch.setattr(ei,'async_drive',lambda args:captured.setdefault('args',args) and 0)
    ei.main()
    return ei.AsyncRoundDriver(captured['args'])


@pytest.mark.parametrize('explicit',[False,True])
def test_async_readers_preserve_runtime_and_route_every_data_consumer(tmp_path,monkeypatch,explicit):
    readers=tmp_path/'readers';fills=tmp_path/'fill-reader'
    extra=['--data-repo',readers,'--fill-repo',fills] if explicit else []
    d=driver(tmp_path,monkeypatch,extra)
    data=readers if explicit else ei.REPO
    fill=fills if explicit else ei.REPO
    assert d.c.pin==tmp_path/'runtime' and d.c.a.commit=='b'*40
    assert d.c.data_repo()==data and d.c.fill_repo()==fill
    rd=d.root/'r0001'
    fillout=tmp_path/'filled';fillout.mkdir();(fillout/'fills.jsonl').write_text('')
    d.a.fill=str(fillout)
    monkeypatch.setattr(d,'failed_policies',lambda:[])
    _,lines,build=d.prepare_plan(rd,d.lineage_policies())
    for command in lines:
        if command[0]=='/usr/bin/env':
            assert command[1]=='NATLANG_REPO='+str(fill)
            assert str(fill/'scripts/neuralese_data/filled_reasoning.py') in command
        else:
            assert command[1]==str(data/'scripts/neuralese_data/sft_sources.py')
    replay=next(command for command in lines if 'replay' in command)
    assert replay[replay.index('--runtime-repo')+1]==str(d.c.pin)
    assert str(data/'scripts/neuralese_data/sft_sources.py') in build
    from natlang_neuralese.common import data_registry
    monkeypatch.setattr(data_registry,'role_paths',lambda role:[])
    actor={'checkpoint_sha256':'a'*64,'sglang':'http://actor','url':'http://front','export':'/export'}
    work=tmp_path/'collect';work.mkdir();(work/'actor.json').write_text(json.dumps(actor))
    launched=[]
    monkeypatch.setattr(d.c,'work_tokenizer',lambda:None)
    monkeypatch.setattr(d.c,'ledger',lambda *args,**kwargs:launched.append((args,kwargs)))
    d._fill_beside('aei-collect-v0000',work,d.controller.current)
    args,kwargs=launched[0]
    assert kwargs['repo']==fill and args[3][1]==str(fill/'scripts/neuralese_data/filled_reasoning.py')
    assert d.controller.current['checkpoint']=='/checkpoint/start.pt'
    assert d.reader_sources()['runtime']['path']==str(d.c.pin)


def test_explicit_runtime_paths_opt_out_of_new_controller_readers(tmp_path,monkeypatch):
    old=tmp_path/'runtime'
    d=driver(tmp_path,monkeypatch,['--data-repo',old,'--fill-repo',old])
    assert d.c.data_repo()==old and d.c.fill_repo()==old


def test_cycle_data_override_also_supplies_fill_unless_explicit(tmp_path):
    from types import SimpleNamespace
    c=object.__new__(ei.Cycle);c.pin=tmp_path/'runtime';c.a=SimpleNamespace(data_repo=str(tmp_path/'data'))
    assert c.fill_repo()==tmp_path/'data'
    c.a.fill_repo=str(tmp_path/'fill')
    assert c.fill_repo()==tmp_path/'fill'


def test_resume_does_not_overwrite_an_active_data_plan(tmp_path,monkeypatch):
    d=driver(tmp_path,monkeypatch,['--data-repo',tmp_path/'new-reader'])
    rd=d.root/'r0001';(rd/'stages').mkdir(parents=True);(rd/'stages/frontier.done').write_text('{}')
    old_script='set -e\nold-reader --immutable-plan\n';(rd/'data.sh').write_text(old_script)
    runs=d.collect_dir(d.controller.current);runs.mkdir(parents=True)
    (runs/'runs.jsonl').write_text(json.dumps({'started':0})+'\n')
    monkeypatch.setattr(ei,'record_collection',lambda *a,**kw:None)
    monkeypatch.setattr(d,'collections_of',lambda policies:[runs])
    monkeypatch.setattr(d,'prepare_plan',lambda *args:(['frontier'],[['new-reader']],['new-build']))
    monkeypatch.setattr(d.c,'present',lambda unit:unit=='aei-r0001-data')
    monkeypatch.setattr(d.c,'work_tokenizer',lambda:None)
    def ledger(unit,*args,**kwargs):
        assert (rd/'data.sh').read_text()==old_script
        if unit.endswith('-build'):
            assert kwargs['repo']==tmp_path/'new-reader'
            (rd/'mix').mkdir();(rd/'mix/chains.pkl').write_bytes(b'chains')
            (rd/'mix/build.json').write_text('{"admitted_episode_keys":[]}')
    monkeypatch.setattr(d.c,'ledger',ledger)
    monkeypatch.setattr(d,'ensure_teacher',lambda *args:None)
    d.prepare(rd,d.lineage_policies())
    assert (rd/'data.sh').read_text()==old_script
    assert (rd/'stages/data.done').exists()


def test_sigterm_and_normal_phase_resume_keep_independent_children_and_state(tmp_path,monkeypatch):
    d=driver(tmp_path,monkeypatch)
    rid=d.controller.begin_round(d.controller.current['sha256'])
    phase=d.controller.path.read_bytes()
    protected={name:tmp_path/name for name in ('optimizer','rng','checkpoint')}
    for name,path in protected.items():path.write_bytes(('actual-fixture-'+name).encode())
    digests={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in protected.items()}
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
    rd=d.root/rid;rd.mkdir(parents=True)
    (rd/'round.json').write_text('{}')
    args=tmp_path/'args.json';args.write_text(json.dumps(vars(d.a),default=str))
    code="""import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path[:0]=[sys.argv[1]+'/scripts/neuralese_data',sys.argv[1]+'/training/neuralese',sys.argv[1]]
import expert_iteration as ei
d=ei.AsyncRoundDriver(SimpleNamespace(**json.loads(Path(sys.argv[2]).read_text())))
d.c.pin_stage=lambda:None
d.held_loop=lambda:None
d.ensure_collection=lambda policy:None
d.ensure_remote_collection=lambda policy:None
d.prepare=lambda *args:None
d.health_report=lambda *args:None
d.prepare_ahead=lambda *args:None
def train(*args):
 Path(sys.argv[3]).write_text(str(os.getpid()))
 time.sleep(120)
d.train_sft=train
d.run()
"""
    owner=subprocess.Popen([sys.executable,'-c',code,str(REPO),str(args),str(tmp_path/'owner-ready')])
    try:
        limit=time.monotonic()+10
        while not (tmp_path/'owner-ready').exists() and time.monotonic()<limit:time.sleep(.01)
        assert (tmp_path/'owner-ready').exists()
        owner.send_signal(signal.SIGTERM);owner.wait(timeout=10)
        assert owner.returncode==-signal.SIGTERM and child.poll() is None
        resumed=ei.AsyncRoundDriver(d.a)
        assert resumed.controller.path.read_bytes()==phase
        assert resumed.controller.begin_round(resumed.controller.current['sha256'])==rid
        assert {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in protected.items()}==digests
        assert child.poll() is None
    finally:
        if owner.poll() is None:owner.kill();owner.wait()
        child.terminate();child.wait(timeout=10)


def test_minimum_loader_gate_checks_data_and_fill_not_learner_runtime(tmp_path):
    from types import SimpleNamespace
    repo=tmp_path/'reader';repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    git('init','-q');git('config','user.email','test@example.invalid');git('config','user.name','Reader Test')
    (repo/'source.py').write_text('old')
    git('add','source.py');git('commit','-qm','old reader');old=git('rev-parse','HEAD')
    (repo/'source.py').write_text('new');git('add','source.py');git('commit','-qm','new reader');new=git('rev-parse','HEAD')
    (repo/'.natlang-pin.json').write_text(json.dumps({'commit':new}))
    c=object.__new__(ei.Cycle);c.pin=tmp_path/'unchanged-runtime'
    c.a=SimpleNamespace(data_repo=str(repo),fill_repo=str(repo),minimum_data_reader_commit=new)
    c.require_data_readers()  # Runtime needs no new pin and is not queried.
    git('checkout','-q',old);(repo/'.natlang-pin.json').write_text(json.dumps({'commit':old}))
    with pytest.raises(subprocess.CalledProcessError):c.require_data_readers()
    git('checkout','-q',new);(repo/'.natlang-pin.json').write_text(json.dumps({'commit':new}))
    (repo/'source.py').write_text('unbound modification')
    with pytest.raises(subprocess.CalledProcessError):c.require_data_readers()


def test_owner_and_pop_eval_presence_keep_preempted_jobs_without_full_status_probe(tmp_path,monkeypatch):
    from scripts import memory_ledger
    import grpo
    from types import SimpleNamespace
    path=tmp_path/'ledger.json'
    path.write_text(json.dumps({'schema':'natlang.memory-ledger/1','claims':{},'waiting':{
        'owned.service':{'priority':'low','since':1,'requeued':True}}}))
    monkeypatch.setattr(memory_ledger,'STATE',str(path))
    monkeypatch.setattr(grpo.subprocess,'run',lambda args,**kwargs:SimpleNamespace(returncode=3))
    assert ei.ledger_queued('owned') and grpo.unit_active('owned')
    assert not ei.ledger_queued('other') and not grpo.unit_active('other')
