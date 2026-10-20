"""Source-host candidate/ownership protocols, distinct from CUDA qualification."""
import copy
import json
from pathlib import Path
import pytest
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_readiness import validate_foundation_qualification_request
from test_foundation_qualification_scheduling import candidate_fixture


def test_source_host_candidates_seal_one_request_per_declared_generation(tmp_path):
    from natlang_neuralese.rl import native_role_release as roles
    declarations=[];values={}
    for name in ('lfm','mellum'):
        path,receipt,requirement=candidate_fixture(tmp_path,name)
        declarations.append({'model':name,'handoff_requirement':requirement})
        values[name]={'foundation_candidate':{'path':str(path),'sha256':sha256_file_hex(path)},
            'checkpoint':receipt['checkpoint']}
    plan={'models':declarations,'owned_models':['lfm'],'role_exchange':{
        'schema':roles.EXCHANGE,'role':'lfm','directory':str(tmp_path/'exchange'),
        'semantic_consumer':'mellum','roles':{name:{'owned_models':[name],'source_host_id':name+'-fixture'}
            for name in ('lfm','mellum')}}}
    state=tmp_path/'state.json'
    def advertise(name):
        current=copy.deepcopy(plan);current['owned_models']=[name];current['role_exchange']['role']=name
        state.write_text(json.dumps({'models':{name:values[name]},'owner':{'protocol':'source-host fixture'}}))
        return roles.withdraw(current,state)
    advertise('lfm')
    assert roles.foundation_request(plan,tmp_path/'requests',generation='initial') is None
    advertise('mellum')
    reference=roles.foundation_request(plan,tmp_path/'requests',generation='initial')
    request=json.loads(Path(reference['path']).read_text())
    assert set(request['candidates'])=={'lfm','mellum'}
    assert request['readiness_granted'] is False
    assert reference==roles.foundation_request(plan,tmp_path/'requests',generation='initial')
    assert reference['sha256']!=roles.foundation_request(plan,tmp_path/'requests',generation='repair-after-feedback')['sha256']
    checked=validate_foundation_qualification_request(reference['path'],'lfm',request['candidates']['lfm']['communication_foundation'])
    assert checked['sha256']==reference['sha256']


def test_snapshot_adoption_requires_current_real_releases_not_stale_eligibility(tmp_path,monkeypatch):
    from test_native_role_release import fixture
    from natlang_neuralese.rl import native_role_release as roles
    from natlang_neuralese.train.native_semantic_stage import _snapshot_adoption
    rows=fixture(tmp_path,monkeypatch)
    consumer=rows['mellum'];releases=[];bindings={}
    for name,row in rows.items():
        reference=roles.publish(row['plan_path'],row['state_path'],kind='released')
        releases.append(reference)
        if name!='mellum':roles.receive(Path(reference['path']).read_text(),consumer['plan_path'])
        receipt=row['receipt']
        bindings[receipt['model_id']]={'checkpoint':receipt['checkpoint'],'native_identity':receipt['native_identity']}
    config={'snapshot_adoption':{'schema':'natlang.native-snapshot-adoption/1',
        'plan':{'path':str(consumer['plan_path']),'sha256':sha256_file_hex(consumer['plan_path'])},
        'releases':releases}}
    assert _snapshot_adoption(config,bindings)==config['snapshot_adoption']
    # The old release remains an authentic immutable historical proof. It may
    # no longer adopt weights once the owning source host resumes useful work.
    reference=roles.withdraw(rows['lfm']['plan'],rows['lfm']['state_path'])
    roles.receive(Path(reference['path']).read_text(),consumer['plan_path'])
    with pytest.raises(ValueError,match='retired optimizer release'):_snapshot_adoption(config,bindings)


def test_qualified_snapshot_survives_pending_status_without_rolling_back_owner(tmp_path,monkeypatch):
    from test_native_role_release import fixture
    from natlang_neuralese.rl import native_role_release as roles
    rows=fixture(tmp_path,monkeypatch);source=rows['lfm'];target=rows['mellum']
    qualified=roles.publish(source['plan_path'],source['state_path'],kind='eligible')
    pending=roles.withdraw(source['plan'],source['state_path'])
    roles.receive(Path(pending['path']).read_text(),target['plan_path'])
    before=roles._status(target['plan']['role_exchange'],'lfm')[0]
    roles.receive(Path(qualified['path']).read_text(),target['plan_path'],snapshot_only=True)
    assert roles._status(target['plan']['role_exchange'],'lfm')[0]==before
    assert roles.exchanged(target['plan'],kind='eligible')[0]=={}
    snapshot,_=roles.exchanged(target['plan'],kind='snapshot')
    assert source['receipt']['model_id'] in snapshot
    # This remains ineligible for actual joint consumption/optimizer adoption.
    assert roles.aggregate(target['plan'])[0]==[]


def test_live_qualified_snapshot_pins_bytes_before_rolling_source_changes(tmp_path,monkeypatch):
    from test_native_role_release import fixture
    from natlang_neuralese.rl import native_role_release as roles
    from natlang_neuralese.train.native_semantic_stage import _qualified_snapshot
    rows=fixture(tmp_path,monkeypatch);source=rows['lfm']
    source['plan']['snapshot_preparation']=True
    source['plan_path'].write_text(json.dumps(source['plan']))
    source['state']['plan_sha256']=sha256_file_hex(source['plan_path'])
    value=source['state']['models']['lfm'];config_path=Path(value['config']['path'])
    config_path.write_text(json.dumps({'runtime_binding':{'plan_sha256':sha256_file_hex(source['plan_path'])}}))
    value['config']=roles.binding(config_path)
    source['state_path'].write_text(json.dumps(source['state']))
    def pin(reference,directory):
        # Real byte-pinning for the protocol fixture; native_bundle has separate
        # actual Torch/composite/sharded full-state roundtrip tests.
        target=Path(directory)/reference['sha256'];target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(reference['path']).read_bytes())
        return roles.binding(target)
    monkeypatch.setattr(roles,'pin_checkpoint',pin)
    roles.publish(source['plan_path'],source['state_path'],kind='eligible')
    qualified,_=roles.exchanged(source['plan'],kind='snapshot')
    name=source['receipt']['model_id'];binding=qualified[name]
    original=Path(source['receipt']['checkpoint']['path']);old=original.read_bytes()
    original.write_bytes(b'actual later rolling SGD snapshot in this protocol fixture')
    assert Path(binding['checkpoint']['path']).read_bytes()==old
    assert _qualified_snapshot(name,binding)['native_identity']==binding['native_identity']
    assert roles.consumed_metadata(binding)==[roles.binding(Path(binding['source_role_release']['path']))]


def test_snapshot_metadata_and_recipe_recovery_work_after_entire_source_host_is_removed(tmp_path,monkeypatch):
    """Only transported bytes survive; source paths cannot satisfy a missing closure.

    Runtime/task attestations and checkpoint decoding are protocol fixtures. The
    actual role verifier, recipe recovery and optimizer construction run here.
    """
    import shutil
    from test_native_role_release import fixture
    from test_native_semantic import setup_pair
    from natlang_neuralese.rl import native_role_release as roles
    from natlang_neuralese.train import native_semantic_stage as stage
    origin=tmp_path/'source-host';origin.mkdir()
    source=fixture(origin,monkeypatch)['lfm']
    reference=roles.publish(source['plan_path'],source['state_path'],kind='eligible')
    _,qualified=roles.validate(reference,source['plan'],kind='eligible')
    name=source['receipt']['model_id'];binding=qualified[name]
    binding['source_role_plan']=source['plan']
    # This is the exact relay metadata inventory, not an independently chosen
    # list of extra recipe/cohort files that could mask a transport omission.
    references=roles.consumed_metadata(binding)
    assert references==[roles.binding(reference['path'])]
    package=roles.metadata_package(references,tmp_path/'transport/metadata')
    checkpoint=tmp_path/'receiver/checkpoint';checkpoint.parent.mkdir()
    checkpoint.write_bytes(Path(binding['checkpoint']['path']).read_bytes())
    proof=checkpoint.parent/'role-proof.json'
    proof.write_bytes(Path(package[0]['transport']['path']).read_bytes())
    binding.update(checkpoint={**binding['checkpoint'],**roles.binding(checkpoint)},
        source_role_release=roles.binding(proof))
    shutil.rmtree(origin)
    assert not origin.exists() and not Path(binding['recipe']['path']).exists()
    assert not Path(binding['readiness']['path']).exists()
    assert stage._qualified_snapshot(name,binding)['native_identity']==binding['native_identity']
    assert stage._source_readiness(name,binding)['recipe']==binding['recipe']
    # Exercise the actual participant loader's old-checkpoint missing-recipe
    # recovery branch. Only tiny model decoding/identity are protocol fixtures;
    # no original recipe file is available or silently aliased on the receiver.
    participant=setup_pair()[0]['lfm']
    config={'architecture':'lfm2'}
    monkeypatch.setattr(stage,'load_native_checkpoint',lambda *args,**kwargs:
        (participant.model,participant.tokenizer,None,config,{'training':{}}))
    monkeypatch.setattr(stage,'native_identity',lambda *args,**kwargs:binding['native_identity'])
    loaded=stage.load_participant(name,binding,{'backbone_lr':1e-4},'cpu')
    assert loaded.recipe_binding==binding['recipe']
    assert loaded.optimizer is not None
    assert roles.consumed_metadata(binding)==[roles.binding(proof)]


def test_post_vae_offers_require_live_owned_per_row_candidates_and_failure_cancels(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_role_release as roles
    from natlang_neuralese.train.native_resident import process_owner
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from test_native_role_scheduler_deployment import prepared_snapshot
    from natlang_neuralese.train.native_readiness import snapshot_successor
    metadata=tmp_path/'metadata';metadata.mkdir()
    config,_,_=prepared_snapshot(metadata)
    with monkeypatch.context() as local:
        local.setattr(roles,'mechanism',lambda:{key:'a'*64 for key in roles.MECHANISM_FILES})
        local.setattr(roles,'ready_participants',lambda *a,**kw:([],[]))
        waiting=snapshot_successor(config,Path(config['out']))
    request=json.loads(Path(waiting['awaiting_descendant_handoff']['path']).read_text())
    declarations=[];values={};states={};plans={}
    for name in ('lfm','mellum'):
        path,receipt,requirement=candidate_fixture(tmp_path,name)
        declarations.append({'model':name,'handoff_requirement':requirement,
            'foundation_qualification_generation':'qualified-generation'})
        values[name]={'foundation_candidate':roles.binding(path),'checkpoint':receipt['checkpoint']}
    assignments={name:{'owned_models':[name],'source_host_id':name+'-existing'}for name in values}
    for name in values:
        plan={'models':declarations,'owned_models':[name],'snapshot_preparation':True,
            'role_exchange':{'schema':roles.EXCHANGE,'role':name,'directory':str(tmp_path/'exchange'),
                'semantic_consumer':'mellum','roles':assignments}}
        state=tmp_path/(name+'-state.json');state.write_text(json.dumps({'models':{name:values[name]},
            'owner':process_owner(),'released':False}))
        roles.withdraw(plan,state);plans[name]=plan;states[name]=state
    request['contract']=roles.contract(plans['mellum']);request['mechanism']=roles.mechanism()
    request_hash=canonical_json_sha256_hex(request)
    offers={name:roles.handoff_offer(plan,states[name],request)for name,plan in plans.items()}
    assert all(offers.values())
    import time
    initial=roles.foundation_request(plans['lfm'],tmp_path/'requests',generation='qualified-generation')
    for plan in plans.values():roles.renew_handoff(plan,request,offers,expires_at=time.time()+180)
    selected=roles.foundation_request(plans['lfm'],tmp_path/'requests',generation='qualified-generation')
    assert selected['sha256']!=initial['sha256']
    assert selected==roles.foundation_request(plans['lfm'],tmp_path/'requests',generation='qualified-generation')
    actual=json.loads(Path(selected['path']).read_text())
    assert actual['generation']==canonical_json_sha256_hex({'foundation_generation':'qualified-generation','handoff':request_hash})
    # A genuine failed qualification attempt ends this rendezvous; it cannot
    # be relabeled passed or reset by an ordinary progress counter.
    value=json.loads(states['lfm'].read_text());value['models']['lfm']['foundation_qualification_attempt']={
        'request':selected,'status':'failed'}
    states['lfm'].write_text(json.dumps(value))
    assert roles.handoff_offer(plans['lfm'],states['lfm'],request)is None
    for plan in plans.values():roles.cancel_handoff(plan,request)
    assert all(not roles.handoff_active(plan)for plan in plans.values())
    value['owner']['start_ticks']='0';states['lfm'].write_text(json.dumps(value))
    assert roles.handoff_offer(plans['lfm'],states['lfm'],request)is None
