"""Real in-process owner evidence, without a synthetic trainer child."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from natlang_neuralese.rl import native_raw_handoff as raw, native_runtime_handoff as runtime
from natlang_neuralese.train.native_resident import binding


@pytest.fixture
def dispatch(tmp_path):
    parent=tmp_path/'continuation.json';parent.write_text(json.dumps({'parent':{'sha256':'a'*64}}))
    script=tmp_path/'owned-entry.py'
    command=[sys.executable,'-u','-m','natlang_neuralese.train.native_state_sft','--out',str(tmp_path/'learner')]
    script.write_text('from natlang_neuralese.rl.native_owned_dispatch import capture\n'
        'import json,sys\n'
        f'print(json.dumps(capture(__file__,{command!r},{binding(parent)!r},{str(tmp_path/"capture")!r},environment={{"PYTHONPATH":"/exact/runtime"}})),flush=True)\n'
        'sys.stdin.buffer.read()\n')
    child_env={**os.environ,'PYTHONPATH':str(Path(raw.__file__).resolve().parents[2])}
    child=subprocess.Popen([sys.executable,str(script)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=child_env)
    try:
        launch=json.loads(child.stdout.readline())
        intent=raw.capture_drain_intent(launch,tmp_path/'intent.json')
        captured=json.loads(Path(intent['path']).read_text())
        yield {'intent':intent,'captured':captured,'record':{'intent':intent},'command':command,'root':tmp_path}
    finally:
        child.communicate(timeout=15)


def test_actual_inprocess_owner_has_no_fake_child(dispatch):
    x=dispatch;parent,argv,refs=runtime._owned_launch_chain(x['record'])
    assert parent=={'parent':{'sha256':'a'*64}}
    assert argv==x['command']
    assert x['captured']['children']==[]
    assert set(refs)=={'intent','supervisor','continuation'}
    assert json.loads(Path(refs['intent']['path']).read_text())['env']=={'PYTHONPATH':'/exact/runtime'}


@pytest.mark.parametrize('tamper',['owner','argv','wrapper','module','schema','environment','snapshot'])
def test_dispatch_cannot_borrow_an_owner_or_command(dispatch,tamper):
    from natlang_neuralese.rl.native_owned_dispatch import verify
    x=dispatch;observed=json.loads(Path(x['captured']['launch']['path']).read_text())
    p=Path(observed['inprocess_dispatch']['path']);value=json.loads(p.read_text())
    snapshots=None
    if tamper=='owner':value['owner']['start_ticks']='0'
    elif tamper=='argv':value['owner']['argv_hex']='00'
    elif tamper=='wrapper':value['wrapper']=binding(x['root']/'continuation.json')
    elif tamper=='module':value['effective_argv'][3]='unknown.module'
    elif tamper=='schema':value['schema']='unknown'
    elif tamper=='environment':value['env']=None
    else:snapshots={'intent':{'path':str(p),'sha256':'0'*64}}
    if tamper!='snapshot':
        forged=x['root']/'changed.json';forged.write_text(json.dumps(value));observed['inprocess_dispatch']=binding(forged)
    with pytest.raises(ValueError):verify(x['captured'],observed,snapshots)


def test_dispatch_snapshots_survive_pruned_wrapper(dispatch):
    x=dispatch;_,_,refs=runtime._owned_launch_chain(x['record']);snapshots={}
    for name,reference in refs.items():
        p=x['root']/('saved-'+name);p.write_bytes(Path(reference['path']).read_bytes());snapshots[name]=binding(p)
    Path(refs['supervisor']['path']).unlink()
    assert runtime._owned_launch_chain(x['record'],snapshots)[1]==x['command']
