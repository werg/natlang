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


@pytest.fixture
def completing_dispatch(tmp_path):
    import time
    from natlang_neuralese.rl.native_owned_dispatch import capture_completion_contract
    parent=tmp_path/'parent.json';parent.write_text('{}')
    script=tmp_path/'owned-entry.py'
    command=[sys.executable,'-u','-m','natlang_neuralese.train.native_state_sft']
    script.write_text('from natlang_neuralese.rl.native_owned_dispatch import capture\n'
        'from natlang_neuralese.common.jsonio import write_json_durable\n'
        'from pathlib import Path\nimport json,sys,time\n'
        f'print(json.dumps(capture(__file__,{command!r},{binding(parent)!r},{str(tmp_path/"capture")!r},environment={{}})),flush=True)\n'
        'sys.stdin.buffer.read()\n'
        'write_json_durable(Path(__file__).parent/"entry-terminal.json",'
        '{"at":time.time(),"passed":True,"raw_completion_credit":0})\n')
    env={**os.environ,'PYTHONPATH':str(Path(raw.__file__).resolve().parents[2])}
    child=subprocess.Popen([sys.executable,str(script)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
    try:
        launch=json.loads(child.stdout.readline())
        terminal=tmp_path/'entry-terminal.json'
        contract=capture_completion_contract(launch,terminal,tmp_path/'completion.json')
        intent=raw.capture_drain_intent(launch,tmp_path/'drain.json')
        yield child,launch,contract,intent,terminal
    finally:
        if child.poll() is None:child.communicate(timeout=15)


def test_genuine_predeclared_return_survives_pruned_original_controls(completing_dispatch):
    from natlang_neuralese.rl.native_owned_dispatch import verify_completion_contract
    child,launch,contract,intent,terminal=completing_dispatch
    child.communicate(timeout=15)
    captured=json.loads(Path(intent['path']).read_text())
    value=json.loads(terminal.read_text())
    assert 'returncode' not in value
    receipt=verify_completion_contract(contract,captured,binding(terminal),value)
    original=json.loads(Path(launch['path']).read_text())
    dispatch=json.loads(Path(original['inprocess_dispatch']['path']).read_text())
    for ref in (launch,original['inprocess_dispatch'],dispatch['wrapper'],dispatch['continuation']):
        Path(ref['path']).unlink()
    assert verify_completion_contract(contract,captured,binding(terminal),value)==receipt


@pytest.mark.parametrize('tamper',['terminal-path','owner','launch','late','passed','credit','returncode','kind'])
def test_completion_rejects_borrowed_or_ambiguous_success(completing_dispatch,tamper):
    from natlang_neuralese.rl.native_owned_dispatch import verify_completion_contract
    child,launch,contract,intent,terminal=completing_dispatch
    child.communicate(timeout=15)
    captured=json.loads(Path(intent['path']).read_text());value=json.loads(terminal.read_text())
    path=Path(contract['path']);declaration=json.loads(path.read_text());reference=binding(terminal)
    if tamper=='terminal-path':reference={**reference,'path':str(terminal.with_name('other.json'))}
    elif tamper=='owner':declaration['owner']['start_ticks']='0'
    elif tamper=='launch':declaration['source_launch']['sha256']='0'*64
    elif tamper=='late':declaration['at']=value['at']+1
    elif tamper=='passed':value['passed']=False
    elif tamper=='credit':value['raw_completion_credit']=1
    elif tamper=='returncode':value['returncode']=0
    elif tamper=='kind':declaration['success_kind']='os-returncode'
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError):verify_completion_contract(binding(path),captured,reference,value)


def test_completion_contract_cannot_be_created_after_exit(completing_dispatch,tmp_path):
    from natlang_neuralese.rl.native_owned_dispatch import capture_completion_contract
    child,launch,contract,intent,terminal=completing_dispatch
    child.communicate(timeout=15)
    with pytest.raises(ValueError,match='live captured owner'):
        capture_completion_contract(launch,terminal,tmp_path/'late.json')


def test_full_typed_drain_returns_one_decoded_pair(completing_dispatch,tmp_path,monkeypatch):
    import torch,shutil
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
    from natlang_neuralese.train.loop import capture_training_rng_state
    child,launch,contract,intent,terminal=completing_dispatch
    slot=tmp_path/'slot.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'fixture','base_inventory_sha256':'a'*64},
        'training':{'schema':TRAINING_SCHEMA,'optimizer':{'value':torch.tensor([7.])},
            'rng':capture_training_rng_state('cpu'),'sft_progress':{'earned':[1,2]}}},slot)
    pin=tmp_path/'pin.pt';shutil.copyfile(slot,pin)
    child.communicate(timeout=15)
    original=torch.load;loads=[]
    def counted(*args,**kwargs):
        loads.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',counted)
    reference,loaded=raw.complete_drain(intent,binding(terminal),binding(pin),tmp_path/'handoff.json',
        original_slot=binding(slot),inprocess_completion=contract,return_verified=True)
    assert len(loads)==1
    assert loaded[0]==json.loads(Path(reference['path']).read_text())
    assert loaded[1]['training']['sft_progress']=={'earned':[1,2]}
    assert torch.equal(loaded[1]['training']['optimizer']['value'],torch.tensor([7.]))
    assert raw.verify_handoff_controls(reference)==loaded[0]
    assert len(loads)==1


def test_typed_drain_retires_exact_adoption_without_another_decode(completing_dispatch,tmp_path,monkeypatch):
    import shutil,torch
    from test_native_resident import setup
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
    from natlang_neuralese.train.loop import capture_training_rng_state
    child,launch,contract,intent,terminal=completing_dispatch
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    source=tmp_path/'source';source.mkdir()
    _,directory,manager,recipe=setup(source)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    manager.poll(7);state=resident.current(directory)
    initial=json.loads((directory/'initial.json').read_text())
    initial['owner']=json.loads(Path(intent['path']).read_text())['owner']
    (directory/'initial.json').write_text(json.dumps(initial))
    slot=tmp_path/'slot.pt';pin=tmp_path/'pin.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'fixture','base_inventory_sha256':'a'*64},
        'training':{'schema':TRAINING_SCHEMA,'optimizer':{'value':torch.tensor([7.])},
            'rng':capture_training_rng_state('cpu'),'sft_progress':{'earned':[1,2]},'global_step':7,
            'source_receipts':[state['payload']['train']],
            'options':{'resident_adoption':{'generation':state['generation'],'history':state['history']}}}},slot)
    shutil.copyfile(slot,pin);child.communicate(timeout=15)
    reference,loaded=raw.complete_drain(intent,binding(terminal),binding(pin),tmp_path/'handoff.json',
        original_slot=binding(slot),resident_directory=directory,inprocess_completion=contract,return_verified=True)
    # The marker alone never authorizes retiring a resident generation.
    with pytest.raises(ValueError,match='drained optimizer'):
        resident.retire(directory,binding(pin),binding(terminal))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('retirement must reuse the verified full state'))
    resident.retire(directory,loaded[0]['checkpoint'],loaded[0]['terminal'],handoff=reference,verified_handoff=loaded)
    assert resident.current(directory) is None
    assert (directory/'retired'/(state['generation']+'.json')).exists()


@pytest.fixture
def reviewed_normal_return(tmp_path):
    from natlang_neuralese.rl.native_owned_dispatch import capture_completion_contract
    parent=tmp_path/'parent.json';parent.write_text('{}')
    script=tmp_path/'recover-earned-owner.py'
    command=[sys.executable,'-u','-m','natlang_neuralese.train.native_state_sft']
    script.write_text('from natlang_neuralese.rl.native_owned_dispatch import capture\n'
        'from natlang_neuralese.train.native_resident import binding,process_owner\n'
        'from natlang_neuralese.common.jsonio import write_json_durable\n'
        'from pathlib import Path\nimport json,sys,time\n'
        'owner=process_owner(); root=Path(__file__).parent\n'
        'write_json_durable(root/"original-launch.json",{"owner":owner,"controller_source":binding(__file__)})\n'
        f'launch=capture(__file__,{command!r},{binding(parent)!r},{str(tmp_path/"capture")!r},environment={{}})\n'
        'print(json.dumps({"dispatch":launch,"marker_launch":binding(root/"original-launch.json")}),flush=True)\n'
        'sys.stdin.buffer.read()\n'
        'write_json_durable(root/"earned-terminal.json",{"at":time.time(),"owner":owner,'
        '"launch":binding(root/"original-launch.json"),"whole_recipe_completed":False})\n')
    env={**os.environ,'PYTHONPATH':str(Path(raw.__file__).resolve().parents[2])}
    child=subprocess.Popen([sys.executable,str(script)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
    try:
        identity=json.loads(child.stdout.readline());terminal=tmp_path/'earned-terminal.json'
        contract=capture_completion_contract(identity['dispatch'],terminal,tmp_path/'completion.json',
            normal_return_launch=identity['marker_launch'])
        intent=raw.capture_drain_intent(identity['dispatch'],tmp_path/'drain.json')
        yield child,identity,contract,intent,terminal
    finally:
        if child.poll() is None:child.communicate(timeout=15)


def test_reviewed_normal_return_preserves_owner_and_grants_no_completion(reviewed_normal_return):
    from natlang_neuralese.rl.native_owned_dispatch import verify_completion_contract
    child,identity,contract,intent,terminal=reviewed_normal_return;child.communicate(timeout=15)
    captured=json.loads(Path(intent['path']).read_text());value=json.loads(terminal.read_text())
    receipt=verify_completion_contract(contract,captured,binding(terminal),value)
    assert receipt['qualification_credit'] is False and value['whole_recipe_completed'] is False
    Path(identity['marker_launch']['path']).unlink()
    assert verify_completion_contract(contract,captured,binding(terminal),value)==receipt


@pytest.mark.parametrize('fault',['owner','launch','completed','credit','returncode','path','snapshot','qualification'])
def test_reviewed_normal_return_cannot_borrow_or_promote_work(reviewed_normal_return,fault):
    from natlang_neuralese.rl.native_owned_dispatch import verify_completion_contract
    child,identity,contract,intent,terminal=reviewed_normal_return;child.communicate(timeout=15)
    captured=json.loads(Path(intent['path']).read_text());value=json.loads(terminal.read_text());ref=binding(terminal)
    if fault=='owner':value['owner']['start_ticks']='0'
    elif fault=='launch':value['launch']['sha256']='0'*64
    elif fault=='completed':value['whole_recipe_completed']=True
    elif fault=='credit':value['raw_completion_credit']=1
    elif fault=='returncode':value['returncode']=0
    elif fault=='path':ref={**ref,'path':str(terminal.with_name('borrowed.json'))}
    else:
        p=Path(contract['path']);data=json.loads(p.read_text());
        if fault=='qualification':data['qualification_credit']=True
        else:data['controls']['normal_return_launch']['sha256']='0'*64
        p.write_text(json.dumps(data));contract=binding(p)
    with pytest.raises(ValueError):verify_completion_contract(contract,captured,ref,value)
