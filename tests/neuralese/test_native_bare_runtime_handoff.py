"""Actual bare owner drains preserve the original coordinator and full optimizer."""
import copy,json,shutil,time
from pathlib import Path
import pytest
import torch
from test_native_later_runtime_handoff import later,_assert_next_update
from test_native_raw_handoff import actual_drain,drained
from natlang_neuralese.rl import native_runtime_handoff as runtime,native_raw_handoff as handoff,native_raw_rounds as raw
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.recipe import native_recipe_binding
from natlang_neuralese.common.jsonio import write_json_durable as write

@pytest.fixture
def bare(later,tmp_path,request):
    x=later;recipe=json.loads(Path(json.loads(Path(x['transition']['path']).read_text())['recipe']['path']).read_text())
    for stage in recipe['stages']:
        for key in ('resident_geometry_qualification','resident_data_directory'):stage['parameters'].pop(key,None)
    recipe_path=tmp_path/'actual-bare-recipe.json';write(recipe_path,recipe)
    config=json.loads(x['config'].read_text());rb=native_recipe_binding(recipe_path,config['recipe_lineage_sha256'])
    saved=copy.deepcopy(x['saved']);saved['training']['options'].update(recipe_binding=rb,stage_id='sft')
    capacity_request=tmp_path/'max-request.json';write(capacity_request,{'scope':'CPU protocol fixture; not model capacity'})
    qualification=tmp_path/'memory-qualification.json';summary={'passed':True,'request':resident.binding(capacity_request),'parent_checkpoint_sha256':'f'*64}
    write(qualification,{**summary,'checks':[{'passed':True,'runtime_code_sha256':resident.digest(resident.code_inventory(x['newpackage'])),'identity':{'model_config':saved['model_config']}}]})
    saved['training']['options']['memory_qualification']=summary
    if getattr(request,'param',None)=='carried':
        from natlang_neuralese.train.native_continuation import _TRANSIENT
        saved['training']['sft_progress']['sft']={key:copy.deepcopy(value)for key,value in saved['training'].items()if key not in _TRANSIENT}
        saved['training']['options']={**saved['training']['options'],'stage_id':'raw_policy'}
    owner=tmp_path/'bare-owner';owner.mkdir();gen=actual_drain.__wrapped__(owner)
    try:
        values=next(gen);process,intent,terminal,slot,pin=values
        torch.save(saved,slot);shutil.copyfile(slot,pin);reference=drained(values)
    finally:gen.close()
    state=json.loads((x['out']/'state.json').read_text());state['checkpoint']=str(pin);write(x['out']/'state.json',state)
    authority={'recipe':resident.binding(recipe_path),'package':str(x['newpackage']),'qualification':resident.binding(qualification)}
    proof=runtime.prepare_runtime_handoff(x['bootstrap'],reference,None,x['config'],tmp_path/'bare-runtime-proof.json',bare_direct_owner=authority)
    x.update(handoff=reference,pin=pin,saved=saved,proof=proof,bare=authority,slot=slot)
    x['declaration']['runtime_handoff']=proof;write(x['plan'],{'models':[x['declaration']]})
    return x


def test_bare_current_recipe_rejoins_without_resident_plan_or_cursor_reset(bare):
    x=bare;before={p:p.read_bytes()for p in (x['root']/'cycle.json',x['out']/'state.json',x['pin'])}
    original=json.loads(x['config'].read_text())
    proof=runtime.verify_runtime_handoff(x['proof'],bootstrap=x['bootstrap'])
    assert proof['recipe_handoff'] is None and proof['bare_direct_owner']
    assert json.loads(Path(x['handoff']['path']).read_text())['resident'] is None
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    config=json.loads(x['config'].read_text())
    assert config['bootstrap_binding']==original['bootstrap_binding'] and config['checkpoint']==original['checkpoint']
    assert config['production_recipe']==x['bare']['recipe']['path']
    requirement=raw._release_requirement(x['declaration'],{})
    assert requirement['recipe_sha256']==x['bare']['recipe']['sha256']
    assert requirement['recipe_lineage_sha256']==original['recipe_lineage_sha256']
    assert all(p.read_bytes()==value for p,value in before.items())
    assert runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],None,x['config'],Path(x['proof']['path']),bare_direct_owner=x['bare'])==x['proof']
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    assert all(p.read_bytes()==value for p,value in before.items())


@pytest.mark.parametrize('later',['lfm','mellum'],indirect=True)
def test_both_real_native_optimizers_keep_bitwise_next_update_after_bare_handoff(bare):
    _assert_next_update(bare)


@pytest.mark.parametrize('field',['recipe','code','model_config_sha256','qualification','lineage','mix'])
def test_bare_authority_rejects_changed_recipe_runtime_profile_and_qualification(bare,tmp_path,field):
    value=json.loads(Path(bare['proof']['path']).read_text());authority=value['bare_direct_owner']
    if field=='recipe':authority['recipe']=resident.binding(bare['config'])
    elif field=='qualification':
        bad=tmp_path/'failed.json';write(bad,{'passed':False});authority[field]=resident.binding(bad)
    elif field=='lineage':value[field]='0'*64
    elif field=='mix':value['recipe_handoff']=bare['transition']
    else:authority[field]={}if field=='code'else'0'*64
    forged=tmp_path/'forged.json';write(forged,value)
    with pytest.raises((ValueError,KeyError)):runtime.verify_runtime_handoff(resident.binding(forged),bootstrap=bare['bootstrap'])


def test_bare_qualification_snapshot_survives_real_operation_file_reuse(bare):
    x=bare;Path(x['bare']['qualification']['path']).write_text('{"next_operation":true}')
    assert runtime.verify_runtime_handoff(x['proof'],bootstrap=x['bootstrap'])['bare_direct_owner']


@pytest.mark.parametrize('bare',['carried'],indirect=True)
def test_bare_policy_owner_keeps_authentic_carried_production_stage(bare):
    assert bare['saved']['training']['options']['stage_id']=='raw_policy'
    before=bare['pin'].read_bytes()
    raw.configure_model(bare['plan'],bare['declaration'],bare['bootstrap'],bare['root'])
    assert bare['pin'].read_bytes()==before


@pytest.mark.parametrize('change',['stage','sources','missing-qualification','failed-qualification','wrong-qualified-profile'])
def test_bare_saved_stage_and_profile_evidence_cannot_be_invented(bare,change):
    saved=copy.deepcopy(bare['saved']);record,_=handoff.verify_handoff(bare['handoff'])
    authority=json.loads(Path(bare['proof']['path']).read_text())['bare_direct_owner']
    if change=='stage':saved['training']['options']['stage_id']='missing';saved['training']['sft_progress']={}
    elif change=='sources':saved['training']['source_receipts'][0]['sha256']='0'*64
    elif change=='missing-qualification':saved['training']['options'].pop('memory_qualification')
    elif change=='failed-qualification':saved['training']['options']['memory_qualification']['passed']=False
    else:
        saved['model_config']['native_ordinary_precision']='different-profile'
        authority['model_config_sha256']=resident.digest(saved['model_config'])
    with pytest.raises(ValueError):runtime._bare_authority(authority,record,saved,json.loads(bare['config'].read_text()))


def test_bootstrap_metadata_tracks_bare_package_qualification_and_current_recipe(bare,monkeypatch):
    from natlang_neuralese.rl import native_bootstrap_metadata as metadata
    from natlang_neuralese.common import runtime_inventory
    monkeypatch.setattr(runtime_inventory,'task_runtime_inventory',lambda path:{'fixture':'collector'})
    monkeypatch.setattr(raw,'_inputs',lambda declaration:{'cases':Path(declaration['cases']['path'])})
    raw.configure_model(bare['plan'],bare['declaration'],bare['bootstrap'],bare['root'])
    config=json.loads(bare['config'].read_text())
    value=metadata.evidence(bare['plan'],bare['declaration'],bare['bootstrap'],config)
    proof=json.loads(Path(bare['proof']['path']).read_text())
    assert proof['bare_direct_owner']['qualification']['path'] in value['files']
    assert value['packages']['later_bare_package']==resident.code_inventory(bare['newpackage'])
    assert value['current_recipe']['sha256']==bare['bare']['recipe']['sha256']
