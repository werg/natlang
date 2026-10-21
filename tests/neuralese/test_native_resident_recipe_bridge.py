"""A real installed recipe handoff bridges stale initial metadata without fake history."""
import copy
import json
from pathlib import Path
import pytest
from test_native_later_runtime_handoff import later
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.rl import native_raw_rounds as raw


@pytest.fixture
def adopted(later):
    x=later;directory=x['directory'];plan=json.loads((directory.parent/'recipe-plan.json').read_text())
    transition=json.loads(Path(x['transition']['path']).read_text())
    old_config=json.loads(x['config'].read_text())
    old_recipe=json.loads(Path(json.loads(Path(x['handoff']['path']).read_text())['resident']['initial']['path']).read_text())['recipe']
    cert=Path(plan['recipe']['stages'][0]['parameters']['resident_geometry_qualification'])
    evidence=cert.with_name('evidence.json');write(evidence,{'passed':True,'scope':'tiny metadata fixture'})
    write(cert,{'schema':'natlang.native-resident-geometry-qualification/1','passed':True,'identity':None,
        'runtime_code_sha256':resident.digest(plan['code']),'limits':{'forward_tokens':1000000,'kv_slots':1000000},
        'evidence':resident.binding(evidence)})
    # Reproduce the actual old producer: validates installed successor plan but
    # retains the former recipe binding in initial authority/flattened history.
    manager=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=old_recipe,
        train=x['manager'].train,held=x['manager'].held,package=x['newpackage'],geometry_qualification=cert)
    recipe=copy.deepcopy(plan['recipe']);recipe['input_bindings'].update(
        train=resident.binding(x['paths']['new']),prior=resident.binding(x['paths']['old']))
    recipe['stages'][0]['inputs']['continued_native_graphs']='prior'
    path=directory.parent/'append.json';write(path,recipe)
    resident.prepare(directory,path,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),runtime_package=x['newpackage'])
    assert manager.poll(20) is not None
    state=resident.current(directory)
    assert state['payload']['prior_recipe']==old_recipe
    assert transition['recipe']['sha256'] not in [state['payload']['prior_recipe']['sha256'],*[r['recipe']['sha256'] for r in state['history']]]
    declaration={'handoff_requirement':x['declaration']['handoff_requirement'],'drained_recipe_handoff':x['transition']}
    return x,declaration,{'resident_current':resident.binding(directory/'current.json')}


def test_real_append_uses_verified_installed_plan_without_mutating_old_history(adopted):
    x,declaration,value=adopted;path=Path(value['resident_current']['path']);before=path.read_bytes();parent=x['pin'].read_bytes()
    result=raw._release_requirement(declaration,value)
    state=json.loads(before)
    assert result['recipe_sha256']==state['payload']['recipe']['sha256']
    assert result['recipe_lineage_sha256']==state['payload']['plan']['recipe_sha256']
    assert path.read_bytes()==before and x['pin'].read_bytes()==parent


@pytest.mark.parametrize('change',['no-transition','prior','predecessor','handoffs','prefix','missing-adoption','stage','history'])
def test_missing_recipe_bridge_rejects_forged_or_incomplete_authority(adopted,tmp_path,change):
    x,declaration,value=adopted
    if change=='no-transition':declaration.pop('drained_recipe_handoff')
    else:
        state=json.loads(Path(value['resident_current']['path']).read_text())
        if change=='prior':state['payload']['prior_recipe']['lineage_sha256']='0'*64
        elif change=='predecessor':state['payload']['plan']['resident_adoptions'][-1]['predecessor']='0'*64
        elif change=='handoffs':state['payload']['plan']['handoffs']=[]
        elif change=='prefix':state['payload']['plan']['resident_adoptions'].insert(0,{'forged':True})
        elif change=='missing-adoption':state['payload']['plan']['resident_adoptions']=[]
        elif change=='stage':state['payload']['plan']['resident_adoptions'][-1]['stage_id']='another-stage'
        else:state['history'][0]['recipe']['sha256']='0'*64
        # Rehash the actual payload/generation so current()'s basic integrity
        # check passes: the independent installed-plan join must still reject.
        state['generation']=resident.digest(state['payload'])
        state['history'][-1]['generation']=state['generation']
        target=tmp_path/'forged/current.json';target.parent.mkdir();write(target,state)
        value['resident_current']=resident.binding(target)
    with pytest.raises((ValueError,KeyError)):raw._release_requirement(declaration,value)


def test_preworker_configuration_failure_does_not_stop_independent_student(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.rl.native_raw_rounds import student_cycle as actual_cycle
    plan=tmp_path/'plan.json';out=tmp_path/'out';stop=SimpleNamespace(requested=False)
    rows=[{'model':name,'handoff_receipt':{'path':'protocol'},'handoff_requirement':{'model_id':name}}
          for name in ('lfm','mellum')]
    write(plan,{'schema':'natlang.native-raw-rl-bootstrap/1','models':rows})
    calls=[]
    def invalid(*a,**k):raise ValueError('exact LF recipe authority missing')
    monkeypatch.setattr(raw,'configure_model',invalid)
    def student(path,declaration,witness,directory,*,stop,owned_controller_fd=None):
        raw.verify_controller_fd(owned_controller_fd,Path(directory).parent/'controller.lock')
        name=declaration['model'];calls.append(name)
        if name=='lfm':return actual_cycle(path,declaration,witness,directory,stop=stop,owned_controller_fd=owned_controller_fd)
        assert not stop.requested
        directory.mkdir(parents=True,exist_ok=True);production=directory/'production';production.mkdir()
        parent=directory/'parent';parent.write_text('actual test publication')
        write(production/'state.json',{'checkpoint':str(parent)})
        write(directory/'config.json',{'out':str(production)})
        return {}
    monkeypatch.setattr(raw,'student_cycle',student)
    value=raw.run(plan,out,cycles=1,stop=stop,poll_seconds=.01)
    assert calls.count('lfm')==1 and calls.count('mellum')==1
    assert stop.requested is False and value['released'] is False
    assert value['configuration_errors']['lfm']['error_type']=='ValueError'
    assert 'mellum' in value['models'] and 'lfm' not in value['models']


def test_suspended_student_retries_corrected_declaration_without_restarting_sibling(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.rl.native_raw_rounds import student_cycle as actual_cycle
    plan=tmp_path/'plan.json';out=tmp_path/'out';stop=SimpleNamespace(requested=False)
    rows=[{'model':name,'handoff_receipt':{'path':'protocol'},'handoff_requirement':{'model_id':name}}
          for name in ('lfm','mellum')]
    write(plan,{'schema':'natlang.native-raw-rl-bootstrap/1','models':rows})
    from threading import Event
    second_mellum=Event();calls={'lfm':0,'mellum':0}
    def invalid(*args,**kwargs):
        assert second_mellum.wait(3)
        raise ValueError('old invalid authority')
    monkeypatch.setattr(raw,'configure_model',invalid)
    def student(path,declaration,witness,directory,*,stop,owned_controller_fd=None):
        raw.verify_controller_fd(owned_controller_fd,Path(directory).parent/'controller.lock')
        name=declaration['model'];calls[name]+=1
        if name=='lfm' and not declaration.get('repaired'):return actual_cycle(path,declaration,witness,directory,stop=stop,owned_controller_fd=owned_controller_fd)
        if name=='mellum' and calls[name]==1:
            replacement=json.loads(path.read_text());replacement['models'][0]['repaired']=True;write(path,replacement)
        if name=='mellum' and calls[name]==2:second_mellum.set()
        assert not stop.requested
        directory.mkdir(parents=True,exist_ok=True);production=directory/'production';production.mkdir(exist_ok=True)
        parent=directory/'parent';parent.write_text(name+str(calls[name]))
        write(production/'state.json',{'checkpoint':str(parent)})
        write(directory/'config.json',{'out':str(production)})
        return {}
    monkeypatch.setattr(raw,'student_cycle',student)
    value=raw.run(plan,out,cycles=2,stop=stop,poll_seconds=.01)
    assert not stop.requested and calls=={'lfm':3,'mellum':2}
    assert not value['configuration_errors'] and set(value['models'])==set(calls)
