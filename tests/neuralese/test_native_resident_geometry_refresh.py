import copy
import json
from pathlib import Path

import pytest

from test_native_resident import setup
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.native_resident_geometry_refresh import refresh


def queued(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    request=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    certificate=json.loads(resident.bound(manager.initial['geometry_qualification']).read_text())
    evidence=tmp_path/'new-evidence.json';resident.write_json(evidence,{'passed':True,'scope':'same tiny fixture identity'})
    certificate['evidence']=resident.binding(evidence)
    new_certificate=tmp_path/'new-capacity.json';resident.write_json(new_certificate,certificate)
    # Reproduce the actual failed-stage startup followed by correct-stage
    # ResidentAdoption initialization, which installs the actual new authority.
    initial=copy.deepcopy(manager.initial);initial['stage_id']=None
    resident.write_json(directory/'initial.json',initial)
    new_manager=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=manager.recipe_binding,
        train=manager.train,held=manager.held,geometry_qualification=new_certificate,
        geometry_identity=manager.initial['geometry_identity'])
    return paths,directory,new_manager,request,new_certificate


def test_refresh_preserves_all_payload_and_real_poll_adopts_after_old_rejection(tmp_path,monkeypatch):
    paths,directory,manager,old,new_certificate=queued(tmp_path,monkeypatch)
    assert manager.poll(3) is None
    assert old['sha256'] in manager.rejected
    original=resident.bound(old).read_bytes();old_payload=json.loads(original)['payload']
    initial_bytes=(directory/'initial.json').read_bytes();plan_bytes=(directory.parent/'recipe-plan.json').read_bytes()
    from natlang_neuralese.train.native_source_preparation import NativeSourcePreparation
    with monkeypatch.context() as patch:
        patch.setattr(NativeSourcePreparation,'load',lambda *a,**kw:pytest.fail('refresh must not load row inventories'))
        patch.setattr(resident,'new_geometry',lambda *a,**kw:pytest.fail('refresh must not parse graphs'))
        result=refresh(directory,old)
    new=json.loads(resident.bound(result['request']).read_text())
    assert new['payload']=={**old_payload,'geometry_qualification':resident.binding(new_certificate)}
    assert new['generation']==resident.digest(new['payload'])
    assert resident.bound(old).read_bytes()==original
    assert (directory/'initial.json').read_bytes()==initial_bytes
    assert (directory.parent/'recipe-plan.json').read_bytes()==plan_bytes
    assert not (directory/'current.json').exists()
    assert refresh(directory,old)==result
    assert manager.poll(4) is not None
    committed=resident.current(directory)
    assert committed['generation']==new['generation']
    assert manager.train.ordered_ids()==['a','b','c']
    assert 'exposure_counts' not in committed


def test_prepare_only_retains_pending_and_actual_learner_authority(tmp_path,monkeypatch):
    _,directory,_,request,_=queued(tmp_path,monkeypatch)
    pointer=(directory/'pending.json').read_bytes()
    result=refresh(directory,request,publish=False)
    assert result['published'] is False and result['request']!=request
    assert (directory/'pending.json').read_bytes()==pointer
    assert refresh(directory,request)['request']==result['request']


def test_completed_refresh_retains_historical_proof_after_actual_owner_restart(tmp_path,monkeypatch):
    _,directory,manager,request,_=queued(tmp_path,monkeypatch)
    initial=json.loads((directory/'initial.json').read_text())
    initial['owner']={**resident.process_owner(),'pid':99999999,'start_ticks':'99999999'}
    assert not resident.owner_alive(initial['owner'])
    resident.write_json(directory/'initial.json',initial)
    result=refresh(directory,request)
    proof_path=resident.bound(result['receipt']);proof_bytes=proof_path.read_bytes()
    proof=json.loads(proof_bytes)
    snapshot=resident.bound(proof['initial']);snapshot_bytes=snapshot.read_bytes()
    assert snapshot_bytes==(directory/'initial.json').read_bytes()
    # The real constructor updates the owner while preserving the authority
    # contract. The past proof must remain verifiable and byte-identical.
    resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=manager.recipe_binding,
        train=manager.train,held=manager.held,
        geometry_qualification=resident.bound(manager.initial['geometry_qualification']),
        geometry_identity=manager.initial['geometry_identity'])
    assert (directory/'initial.json').read_bytes()!=snapshot_bytes
    assert resident.bound(proof['initial']).read_bytes()==snapshot_bytes
    assert refresh(directory,request)==result
    assert proof_path.read_bytes()==proof_bytes


@pytest.mark.parametrize('bad',['identity','authority','limits','source','index','pending','initial-race'])
def test_refresh_rejects_changed_authority_or_proofs_without_publishing(tmp_path,monkeypatch,bad):
    paths,directory,manager,request,new_certificate=queued(tmp_path,monkeypatch)
    pointer=(directory/'pending.json').read_bytes()
    if bad=='identity':
        initial=json.loads((directory/'initial.json').read_text());initial['geometry_identity']={'changed':True}
        resident.write_json(directory/'initial.json',initial)
    elif bad=='authority':
        initial=json.loads((directory/'initial.json').read_text());initial['recipe']['lineage_sha256']='changed'
        resident.write_json(directory/'initial.json',initial)
    elif bad=='limits':
        certificate=json.loads(new_certificate.read_text());certificate['limits']={'forward_tokens':1,'kv_slots':1}
        resident.write_json(new_certificate,certificate)
        initial=json.loads((directory/'initial.json').read_text());initial['geometry_qualification']=resident.binding(new_certificate)
        resident.write_json(directory/'initial.json',initial)
    elif bad=='source':paths['new'].write_text('changed actual source bytes')
    elif bad=='index':
        payload=json.loads(resident.bound(request).read_text())['payload']
        Path(payload['indexes'][0]['cache']['path']).write_bytes(b'changed index bytes')
    elif bad=='pending':
        another=tmp_path/'other-request.json';resident.write_json(another,{'another':'obligation'})
        resident.write_json(directory/'pending.json',resident.binding(another));pointer=(directory/'pending.json').read_bytes()
    elif bad=='initial-race':
        original=resident._check_indexes
        def changed(indexes):
            original(indexes)
            initial=json.loads((directory/'initial.json').read_text());initial['owner']={'changed':'owner'}
            resident.write_json(directory/'initial.json',initial)
        monkeypatch.setattr(resident,'_check_indexes',changed)
    with pytest.raises(ValueError):refresh(directory,request)
    assert (directory/'pending.json').read_bytes()==pointer
    assert not (directory/'current.json').exists()
