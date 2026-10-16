"""Trusted supervisor evidence: actual process tree, immutable pin and state."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pytest
import torch
from natlang_neuralese.rl.native_raw_handoff import (binding,capture_drain_intent,complete_drain,verify_handoff)
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.train.loop import capture_training_rng_state


@pytest.fixture
def actual_drain(tmp_path):
    argv=[sys.executable,'-c',"import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.buffer.read()'],stdin=subprocess.PIPE);print('ready',flush=True);sys.stdin.buffer.read();p.communicate()"]
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    assert process.stdout.readline().strip()=='ready'
    launch=tmp_path/'launch.json';write_json_durable(launch,{'at':time.time(),'pid':process.pid,'argv':argv})
    intent=capture_drain_intent(binding(launch),tmp_path/'intent.json')
    assert len(json.loads(Path(intent['path']).read_text())['children'])==1
    slot=tmp_path/'slot.pt'
    # Compact protocol state fixture. Actual optimizer/model-update continuation
    # is separately covered by native_stage tests; this does not claim quality.
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'fixture','base_inventory_sha256':'a'*64},
        'training':{'schema':TRAINING_SCHEMA,'optimizer':{'moments':torch.tensor([.4,.7])},
            'rng':capture_training_rng_state('cpu'),'sft_progress':{'full':{'earned':[1,2],'required':[2,3]}}}},slot)
    pin=tmp_path/'parent.pt';shutil.copyfile(slot,pin)
    terminal=tmp_path/'exit.json'
    try:
        yield process,intent,terminal,slot,pin
    finally:
        if process.poll() is None:process.communicate()


def drained(value):
    process,intent,terminal,slot,pin=value
    process.communicate();assert process.returncode==0
    write_json_durable(terminal,{'at':time.time(),'returncode':process.returncode})
    return complete_drain(intent,binding(terminal),binding(pin),pin.parent/'handoff.json',original_slot=binding(slot))


def test_actual_tree_must_drain_and_independent_pin_retains_complete_state(actual_drain):
    process,intent,terminal,slot,pin=actual_drain
    write_json_durable(terminal,{'at':time.time(),'returncode':0})
    with pytest.raises(ValueError,match='still alive'):
        complete_drain(intent,binding(terminal),binding(pin),pin.parent/'bad.json',original_slot=binding(slot))
    reference=drained(actual_drain)
    record,state=verify_handoff(reference)
    assert record['checkpoint']['sha256']==binding(slot)['sha256']
    assert state['training']['sft_progress']['full']['earned']==[1,2]
    assert torch.equal(state['training']['optimizer']['moments'],torch.tensor([.4,.7]))
    assert all(value is None or value['state'] in ('Z','X') for value in record['observations'])


@pytest.mark.parametrize('tamper',['alive','missing-child','returncode','clock','pin','checkpoint','rng','progress','launch'])
def test_drain_rejects_incomplete_or_changed_evidence(actual_drain,tamper):
    reference=drained(actual_drain);record=json.loads(Path(reference['path']).read_text())
    intent=json.loads(Path(record['intent']['path']).read_text())
    if tamper=='alive':record['observations'][0]=intent['owner']
    elif tamper=='missing-child':record['observations'].pop()
    elif tamper in ('returncode','clock'):
        path=Path(record['terminal']['path']);value=json.loads(path.read_text())
        if tamper=='returncode':value['returncode']=1
        else:value['at']=0
        write_json_durable(path,value);record['terminal']=binding(path)
    elif tamper=='pin':record['checkpoint_pin']['pin_inode']=record['checkpoint_pin']['original_inode']
    elif tamper in ('checkpoint','rng','progress'):
        path=Path(record['checkpoint']['path'])
        if tamper=='checkpoint':path.write_bytes(b'changed bytes')
        else:
            state=torch.load(path,weights_only=False)
            state['training'].pop('rng' if tamper=='rng' else 'sft_progress')
            torch.save(state,path);record['checkpoint']=binding(path)
            record['checkpoint_pin']['bytes']=path.stat().st_size
            record['checkpoint_pin']['original_slot']['sha256']=record['checkpoint']['sha256']
    else:
        path=Path(record['intent']['path']);intent['owner']['argv_hex']=b'other-owner\x00'.hex()
        write_json_durable(path,intent);record['intent']=binding(path)
    path=Path(reference['path']);write_json_durable(path,record)
    with pytest.raises((ValueError,RuntimeError)):verify_handoff(binding(path))


def test_final_resident_snapshot_survives_later_legitimate_owner_rewrite(actual_drain,tmp_path,monkeypatch):
    from test_native_resident import setup
    from natlang_neuralese.train import native_resident as resident
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    source=tmp_path/'source';source.mkdir()
    _,directory,manager,new_recipe=setup(source)
    resident.prepare(directory,new_recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    manager.poll(7);current=resident.current(directory)
    process,intent,terminal,slot,pin=actual_drain
    tree=json.loads(Path(intent['path']).read_text())
    initial=json.loads((directory/'initial.json').read_text());initial['owner']=tree['children'][0]
    write_json_durable(directory/'initial.json',initial)
    write_json_durable(directory/'pending.json',{'uncommitted':'no-credit'})
    saved=torch.load(slot,weights_only=False)
    saved['training'].update(global_step=7,source_receipts=[current['payload']['train']],
        options={'resident_adoption':{'generation':current['generation'],'history':current['history']}})
    torch.save(saved,slot);shutil.copyfile(slot,pin)
    process.communicate();write_json_durable(terminal,{'at':time.time(),'returncode':process.returncode})
    reference=complete_drain(intent,binding(terminal),binding(pin),tmp_path/'resident-handoff.json',
        original_slot=binding(slot),resident_directory=directory)
    record,_=verify_handoff(reference)
    assert record['resident']['current']['path']!=str(directory/'current.json')
    assert json.loads(Path(record['resident']['pending_uncredited']['path']).read_text())=={'uncommitted':'no-credit'}
    # Later ownership/source changes may not rewrite the historical evidence.
    write_json_durable(directory/'initial.json',{'owner':resident.process_owner()})
    write_json_durable(directory/'current.json',{'new-generation':'later'})
    verify_handoff(reference)


def test_original_control_launch_and_exit_can_advance_without_rewriting_handoff(actual_drain):
    _,intent,terminal,_,_=actual_drain
    source_launch=json.loads(Path(intent['path']).read_text())['source_launch']
    reference=drained(actual_drain)
    requirement={'trainer_launch':source_launch,'trainer_exit':str(terminal)}
    verify_handoff(reference,requirement=requirement)
    write_json_durable(Path(source_launch['path']),{'pid':123,'argv':['later-owner']})
    write_json_durable(terminal,{'at':time.time(),'returncode':1,'reason':'later-worker'})
    verify_handoff(reference,requirement=requirement)


@pytest.mark.parametrize('missing_children',[False,True])
def test_process_tree_includes_children_spawned_by_nonleader_thread(tmp_path,monkeypatch,missing_children):
    code="""import subprocess,sys,threading
ready=threading.Event();drain=threading.Event()
def worker():
    p=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.buffer.read()'],stdin=subprocess.PIPE)
    print('ready',flush=True);ready.set();drain.wait();p.communicate()
t=threading.Thread(target=worker);t.start();ready.wait();sys.stdin.buffer.read();drain.set();t.join()
"""
    argv=[sys.executable,'-c',code]
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='ready'
        if missing_children:
            from natlang_neuralese.rl import native_raw_handoff as handoff
            monkeypatch.setattr(handoff,'_thread_children',lambda pid:set())
        launch=tmp_path/'thread-launch.json';write_json_durable(launch,{'at':time.time(),'pid':process.pid,'argv':argv})
        intent=capture_drain_intent(binding(launch),tmp_path/'thread-intent.json')
        children=json.loads(Path(intent['path']).read_text())['children']
        assert len(children)==1 and children[0]['pid']!=process.pid
    finally:process.communicate()
    assert process.returncode==0


def test_proc_parent_census_survives_missing_children_files(actual_drain,tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_handoff as handoff
    process,original,_,_,_=actual_drain
    value=json.loads(Path(original['path']).read_text())
    monkeypatch.setattr(handoff,'_thread_children',lambda pid:set())
    reference=capture_drain_intent(value['source_launch'],tmp_path/'ppid-intent.json')
    captured=json.loads(Path(reference['path']).read_text())
    assert [child['pid'] for child in captured['children']]==[child['pid'] for child in value['children']]
    assert captured['process_tree']['links'][0]['observed_by']==['proc-ppid']
    assert captured['children'][0]['ppid']==process.pid


@pytest.mark.parametrize('race',['reparent','reuse','new-child','owner-reuse'])
def test_census_rejects_changed_parent_or_identity_without_claiming_unrelated_process(tmp_path,monkeypatch,race):
    from natlang_neuralese.rl import native_raw_handoff as handoff
    owner={'pid':101,'ppid':1,'boot_id':'boot','start_ticks':'10','state':'S','argv_hex':'6100'}
    child={'pid':102,'ppid':101,'boot_id':'boot','start_ticks':'11','state':'S','argv_hex':'6200'}
    unrelated={'pid':103,'ppid':1,'boot_id':'boot','start_ticks':'12','state':'S','argv_hex':'6300'}
    table={item['pid']:dict(item) for item in (owner,child,unrelated)};calls=0
    def scan():
        nonlocal calls
        calls+=1
        if race=='new-child' and calls>1:
            return {**table,104:{**child,'pid':104,'start_ticks':'14'}}
        return table
    def observe(pid):
        value=table.get(pid)
        if value is None:return None
        if pid==child['pid'] and race in ('reparent','reuse'):
            return {**value,**({'ppid':103} if race=='reparent' else {'start_ticks':'99'})}
        if pid==owner['pid'] and race=='owner-reuse':return {**value,'start_ticks':'99'}
        return value
    monkeypatch.setattr(handoff,'_process_table',scan)
    monkeypatch.setattr(handoff,'_thread_children',lambda pid:set())
    monkeypatch.setattr(handoff,'observe',observe)
    with pytest.raises(ValueError,match='census'):
        handoff._capture_process_tree(owner)


def test_census_never_admits_unrelated_process_or_stale_thread_pid(monkeypatch):
    from natlang_neuralese.rl import native_raw_handoff as handoff
    owner={'pid':101,'ppid':1,'boot_id':'boot','start_ticks':'10','state':'S','argv_hex':'6100'}
    unrelated={**owner,'pid':103,'start_ticks':'12'}
    table={item['pid']:item for item in (owner,unrelated)}
    monkeypatch.setattr(handoff,'_process_table',lambda:table)
    monkeypatch.setattr(handoff,'observe',lambda pid:table.get(pid))
    monkeypatch.setattr(handoff,'_thread_children',lambda pid:set())
    assert handoff._capture_process_tree(owner)==([],[])
    monkeypatch.setattr(handoff,'_thread_children',lambda pid:{103} if pid==101 else set())
    with pytest.raises(ValueError,match='descendant identity or parent'):
        handoff._capture_process_tree(owner)


@pytest.mark.parametrize('tamper',['parent','omit','origin','reuse'])
def test_census_parent_evidence_survives_archive_validation(actual_drain,tamper):
    from natlang_neuralese.rl import native_raw_handoff as handoff
    reference=drained(actual_drain);record=json.loads(Path(reference['path']).read_text())
    intent_path=Path(record['intent']['path']);intent=json.loads(intent_path.read_text())
    links=intent['process_tree']['links']
    if tamper=='parent':links[0]['child']['ppid']=999
    elif tamper=='omit':links.clear()
    elif tamper=='origin':links[0]['observed_by']=[]
    else:links[0]['parent']['start_ticks']='9999999999'
    write_json_durable(intent_path,intent);record['intent']=binding(intent_path)
    path=Path(reference['path']);write_json_durable(path,record)
    with pytest.raises(ValueError,match='census'):handoff.verify_handoff(binding(path))
