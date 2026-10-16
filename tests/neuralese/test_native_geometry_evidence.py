"""Completed source parses can be reused without weakening source admission."""
import copy
import json
from pathlib import Path
import pytest
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.common.jsonio import write_json_durable as write_json
from test_native_resident import setup


def receipt(tmp_path, source, prior):
    indices = list(range(len(prior), len(source)))
    cases = []
    for index, example in zip(indices, source.iter_indices(indices)):
        cases.append({'index': index, 'id': example.id, 'binding': source.entries[index]['binding'],
                      'forward_tokens': example.plan.token_count, 'kv_slots': len(example.plan.slot_sources)})
    value = {'passed': True, 'source': resident.binding(source.path),
             'prior_train': resident.binding(prior.path), 'old_train_rows': len(prior),
             'new_train_rows': len(source)-len(prior), 'indexes': resident._source_witness(source),
             'maxima': {key: max(cases, key=lambda case: case[key]) for key in ('forward_tokens','kv_slots')}}
    path = tmp_path/'completed-geometry.json';write_json(path,value)
    return path


def sources(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    source=NativeGraphSource(paths['new'],'tokenizer-sha',required_split='train')
    proof=receipt(tmp_path,source,manager.train)
    return source,proof,directory,manager,recipe


def test_reuse_exact_completed_train_geometry_and_existing_runtime_inventory(tmp_path,monkeypatch):
    source,proof,directory,manager,recipe=sources(tmp_path,monkeypatch)
    # A train-only proof must eliminate the full training parse. Its maximum
    # row is independently rechecked, while the unchanged held set needs none.
    monkeypatch.setattr(resident,'new_geometry',lambda *a:pytest.fail('unexpected complete source parse'))
    result=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
        runtime_package=Path(resident.__file__).parents[1],geometry_evidence={'train':resident.binding(proof)})
    assert manager.poll(9) is not None
    assert resident.current(directory)['payload']['geometry_evidence']['train']==resident.binding(proof)
    assert resident.bound(result).is_file()


@pytest.mark.parametrize('change',['missing-row','binding','priority','tokenizer','policy','count','maximum','total-slots','source-sha'])
def test_corrupted_or_noncovering_geometry_cannot_publish_request(tmp_path,monkeypatch,change):
    source,proof,directory,manager,recipe=sources(tmp_path,monkeypatch)
    value=json.loads(proof.read_text())
    if change=='missing-row':value['old_train_rows']+=1
    elif change=='binding':value['maxima']['forward_tokens']['binding']='bad'
    elif change=='priority':
        source.entries[-1]=dict(source.entries[-1],training_priority=99)
    elif change=='tokenizer':value['indexes'][0]['key']['tokenizer']='other'
    elif change=='policy':value['indexes'][0]['key']['policy_sha256']='bad'
    elif change=='count':value['new_train_rows']+=1
    elif change=='maximum':value['maxima']['forward_tokens']['forward_tokens']=1
    elif change=='total-slots':value['maxima']['kv_slots']['kv_slots']+=1
    else:value['source']['sha256']='0'*64
    write_json(proof,value)
    with pytest.raises((ValueError,KeyError)):
        resident.geometry_from_evidence(source,2,resident.binding(proof),package=Path(resident.__file__).parents[1])
    assert not(directory/'pending.json').exists()


def test_legacy_training_geometry_never_certifies_held_rows(tmp_path,monkeypatch):
    source,proof,directory,manager,recipe=sources(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='training geometry only'):
        resident.geometry_from_evidence(manager.held,0,resident.binding(proof),package=Path(resident.__file__).parents[1])


def test_prepared_request_covers_both_exact_split_suffixes(tmp_path,monkeypatch):
    source,proof,directory,manager,recipe=sources(tmp_path,monkeypatch)
    request=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    maximum=resident.geometry_from_evidence(source,2,request,package=Path(resident.__file__).parents[1])
    assert maximum['forward_tokens']>=source.entries[-1]['forward_tokens']
    value=json.loads(resident.bound(request).read_text());value['payload']['entries_sha256']='bad'
    value['generation']=resident.digest(value['payload']);path=tmp_path/'bad-request.json';write_json(path,value)
    with pytest.raises(ValueError,match='metadata'):
        resident.geometry_from_evidence(source,2,resident.binding(path),package=Path(resident.__file__).parents[1])


def test_runtime_package_override_must_equal_running_plan(tmp_path,monkeypatch):
    source,proof,directory,manager,recipe=sources(tmp_path,monkeypatch)
    other=tmp_path/'other';other.mkdir();(other/'changed.py').write_text('changed')
    with pytest.raises(ValueError,match='runtime differs'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),runtime_package=other)
    assert not(directory/'pending.json').exists()


def test_external_cpu_preparer_keeps_actual_package_inventory_and_old_consumer(tmp_path,monkeypatch):
    import os
    import subprocess
    import sys
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path,role_ids=(6,7,8,9))
    source=NativeGraphSource(paths['new'],'tokenizer-sha',required_split='train',role_ids=(6,7,8,9))
    proof=receipt(tmp_path,source,manager.train)
    external=tmp_path/'outside-package.py';external.write_bytes(Path(resident.__file__).read_bytes())
    preparer=tmp_path/'preparer-binding.json';write_json(preparer,resident.binding(external))
    evidence=tmp_path/'evidence-map.json';write_json(evidence,{'train':resident.binding(proof)})
    package=Path(resident.__file__).parents[1]
    before=resident.code_inventory(package)
    repo=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(repo/'scripts/coreweave/prepare_resident_data.py'),
        '--package',str(package),'--preparer',str(preparer),'--directory',str(directory),
        '--recipe',str(recipe),'--stage-id','sft','--tokenizer-fingerprint','tokenizer-sha',
        '--role-ids','6','7','8','9','--geometry-evidence',str(evidence)],
        capture_output=True,text=True,env=os.environ,timeout=60)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout.splitlines()[-1])
    assert output['runtime_code_sha256']==resident.digest(before)
    assert resident.code_inventory(package)==before
    assert manager.poll(11) is not None


def test_proof_maximum_may_conservatively_include_a_known_old_case(tmp_path,monkeypatch):
    source,proof,directory,manager,recipe=sources(tmp_path,monkeypatch)
    value=json.loads(proof.read_text())
    example=next(source.iter_indices([0]));entry=source.entries[0]
    old={'index':0,'id':example.id,'binding':entry['binding'],
         'forward_tokens':example.plan.token_count,'kv_slots':len(example.plan.slot_sources)}
    assert all(old[key]>=value['maxima'][key][key] for key in ('forward_tokens','kv_slots'))
    value['maxima']={key:old for key in ('forward_tokens','kv_slots')};write_json(proof,value)
    assert resident.geometry_from_evidence(source,2,resident.binding(proof),package=Path(resident.__file__).parents[1])=={key:old[key] for key in ('forward_tokens','kv_slots')}
