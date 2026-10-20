"""Historical admission must bind actual dependencies and every consumed artifact."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.neuralese_data import verify_completed_search_parent as v


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return v.binding(path)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
 tree=tmp_path/'historical';files={}
 for name in v.FILES:
  path=tree/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name);files[name]=v.binding(path)
 monkeypatch.setattr(v,'FILES',{key:value['sha256']for key,value in files.items()})
 config=tmp_path/'composition.json';write(config,dict(tree=str(tree),files=files));launch=tmp_path/'launch.json';write(launch,dict(source='exact'))
 nested=tmp_path/'groups/nested-output.jsonl';nested.parent.mkdir();nested.write_text('unchanged full trajectory\n')
 receipt=dict(schema='natlang.search-completed-parent-verification/1',passed=True,verifier=v.binding(Path(v.__file__).resolve()),shard='0040',launch=v.binding(launch),composition=v.binding(config),artifacts=[v.binding(nested)])
 return config,launch,nested,receipt


def test_exact_historical_composition_and_outputs(fixture):
 config,launch,_,receipt=fixture
 assert v.composition(config)[1]['files']
 assert v.verified_artifacts(receipt,shard='0040',launch_path=launch,composition_path=config)==receipt


@pytest.mark.parametrize('change',['bytes','hash','path','missing','extra'])
def test_dependency_tampering_rejected(fixture,change):
 config,_,_,_=fixture;value=json.loads(config.read_text());name=next(iter(value['files']));ref=value['files'][name]
 if change=='bytes':ref['bytes']+=1
 elif change=='hash':ref['sha256']='0'*64
 elif change=='path':ref['path']=str(Path(ref['path']).with_name('unqualified.py'))
 elif change=='missing':value['files'].pop(name)
 else:value['files']['unqualified.py']=ref
 config.write_text(json.dumps(value))
 with pytest.raises((ValueError,FileNotFoundError)):v.composition(config)


def test_actual_nested_output_changed_after_verification(fixture):
 config,launch,nested,receipt=fixture;nested.write_text('changed full trajectory\n')
 with pytest.raises(ValueError):v.verified_artifacts(receipt,shard='0040',launch_path=launch,composition_path=config)


@pytest.mark.parametrize('change',['shard','launch','duplicate','empty','not_passed'])
def test_receipt_custody_rejected(fixture,change):
 config,launch,_,receipt=fixture
 if change=='shard':receipt['shard']='0041'
 elif change=='launch':launch.write_text('changed source selection')
 elif change=='duplicate':receipt['artifacts']*=2
 elif change=='empty':receipt['artifacts']=[]
 else:receipt['passed']=False
 with pytest.raises(ValueError):v.verified_artifacts(receipt,shard='0040',launch_path=launch,composition_path=config)


def test_child_requires_fresh_import_process(fixture,monkeypatch):
 config,launch,_,_=fixture;import sys
 monkeypatch.setitem(sys.modules,'natlang_neuralese',SimpleNamespace())
 with pytest.raises(ValueError,match='fresh interpreter'):v.worker('0040',launch_path=launch,composition_path=config,output=config.parent/'output.json')


def renderer():
 return SimpleNamespace(SEARCH_REVIEWED_RESOURCE_PARENT_CHAIN={'known'},file_sha=lambda p:'source-sha',_resource_parent_code_binding=lambda parent,job:dict(renderer=parent),_render_job_binding_from_job=lambda job,source,code:dict(source=source,code=code,policy=job))


def test_exact_parent_job_equation_is_recomputed():
 r=renderer();job=('actual policy',);expected=dict(source='source-sha',code=dict(renderer='known'),policy=job)
 assert v.parent_job_binding(r,job,dict(renderer_code_sha256='known',render_job_binding=expected),Path('selected'))==expected
 with pytest.raises(ValueError,match='equation'):v.parent_job_binding(r,('different policy',),dict(renderer_code_sha256='known',render_job_binding=expected),Path('selected'))
 with pytest.raises(ValueError,match='unreviewed'):v.parent_job_binding(r,job,dict(renderer_code_sha256='unknown',render_job_binding=expected),Path('selected'))
