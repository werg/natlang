"""Future role material is byte-bound and cannot manufacture a reviewed owner."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/coreweave'))
from prepare_native_actor_role import prepare
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.train.native_resident import code_inventory


def fixture(tmp_path,role):
    composition=tmp_path/'runtime';composition.mkdir()
    package=composition/'training/neuralese/natlang_neuralese';package.mkdir(parents=True)
    (package/'__init__.py').write_text('# exact source\n')
    def emit(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
        return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    refs={name:emit(composition/(name+'.json'),{'source':name}) for name in
          ('controller','binder','control_loader','control_manifest','role_mechanism_authorities')}
    material={**refs,'model':role,'runtime':str(composition),'actor_runtime':str(composition),
              'runtime_code_sha256':canonical_json_sha256_hex(code_inventory(package)),'controls':{}}
    material_ref=emit(composition/'material.json',material)
    inventory=code_inventory(package)
    files={str(path.relative_to(composition)):{'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
           for path in composition.rglob('*') if path.is_file()}
    emit(composition/'construction.json',{'schema':'natlang.native-actor-coordinator-composition/1',
        'inventory':inventory,'code':canonical_json_sha256_hex(inventory),'files':files,
        'role_materials':{role:material_ref}})
    # An honest older review can bind path+SHA without redundant byte counts.
    review={'passed':True,**{name:{key:refs[name][key] for key in ('path','sha256')}
                            for name in ('controller','binder')}}
    inputs={'entry_root':str(tmp_path/'future-owner-not-started'),
        'bootstrap':emit(tmp_path/'bootstrap.json',{'schema':'natlang.native-raw-rl-bootstrap/1',
            'owned_models':[role],'role_exchange':{'roles':{role:{}}}}),
        'root_review':emit(tmp_path/'review.json',review),
        'provider_readiness':emit(tmp_path/'provider.json',{'fixture':True}),
        'publication_configuration':{},'frozen_refs':[]}
    inputs_path=tmp_path/'inputs.json';emit(inputs_path,inputs)
    return composition,inputs_path,inputs


@pytest.mark.parametrize('role',['lfm','mellum'])
def test_future_binder_preserves_actual_inputs_and_two_field_review(tmp_path,role):
    composition,path,inputs=fixture(tmp_path,role)
    out=tmp_path/'binder-config.json'
    reference=prepare(composition,role,path,out)
    result=json.loads(out.read_text())
    assert result['actor_runtime']==str(composition) and result['model']==role
    assert set(result['controller'])=={'path','sha256'}
    assert result['entry_root']==inputs['entry_root'] and not Path(inputs['entry_root']).exists()
    assert json.loads(path.read_text())==inputs
    assert reference['sha256']==sha256_file_hex(out)
    assert prepare(composition,role,path,out)==reference


def test_future_binder_rejects_review_of_another_script_and_mutated_source(tmp_path):
    composition,path,inputs=fixture(tmp_path,'lfm')
    review_path=Path(inputs['root_review']['path']);review=json.loads(review_path.read_text())
    review['binder']['sha256']='0'*64;review_path.write_text(json.dumps(review))
    inputs['root_review']['sha256']=sha256_file_hex(review_path)
    inputs['root_review']['bytes']=review_path.stat().st_size;path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError,match='actual composed controller and binder'):
        prepare(composition,'lfm',path,tmp_path/'unused')
    (composition/'controller.json').write_text('tampered')
    with pytest.raises(ValueError,match='source file changed'):
        prepare(composition,'lfm',path,tmp_path/'unused')
