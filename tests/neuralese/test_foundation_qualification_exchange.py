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
