"""Source-host protocol fixtures run the real full readiness validators.

Tiny checkpoint loading and original bootstrap process proofs are protocol
fixtures; they do not assert numerical model qualification.
"""
import copy
import json
from pathlib import Path

import pytest
from natlang_neuralese.rl import native_role_release as roles,native_raw_rounds as raw
from natlang_neuralese.train.native_readiness import ready_participants
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.hashing import sha256_file_hex


def fixture(tmp_path,monkeypatch,names=('lfm','mellum')):
    from test_native_readiness import ready
    from natlang_neuralese.train.native_resident import process_owner
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
    monkeypatch.setattr(raw,'verify_handoff',lambda *a,**kw:({},{}))
    monkeypatch.setattr(CheckpointPolicy,'load',lambda path:{'schema':CHECKPOINT_SCHEMA,
        'model_config':{'architecture':Path(path).parent.name.split(':')[0],'base_inventory_sha256':'a'*64},
        'training':{'schema':TRAINING_SCHEMA,'optimizer':{'protocol':True},'rng':{'protocol':True}}})
    entries=[];receipts={}
    for name in names:
        path,receipt,requirement=ready(tmp_path,name+':'+'a'*64)
        receipts[name]=(path,receipt,requirement)
        entries.append({'model':name,'handoff_requirement':requirement})
    assignment={name:{'owned_models':[name],'source_host_id':name+'-actual-sid'}for name in names}
    result={}
    for name in names:
        root=tmp_path/(name+'-source-host');root.mkdir()
        plan={'schema':'natlang.native-raw-rl-bootstrap/1','models':copy.deepcopy(entries),'owned_models':[name],
            'role_exchange':{'schema':roles.EXCHANGE,'role':name,'roles':assignment,
                'semantic_consumer':'mellum','directory':str(root/'exchange')}}
        plan_path=root/'plan.json';write_json_durable(plan_path,plan)
        path,receipt,requirement=receipts[name]
        config_path=root/'config.json';write_json_durable(config_path,{'runtime_binding':{'plan_sha256':sha256_file_hex(plan_path)}})
        production=root/'production.json';write_json_durable(production,{'checkpoint':receipt['checkpoint']['path']})
        value={'readiness':raw.binding(path),'checkpoint':raw.binding(receipt['checkpoint']['path']),
            'handoff':{'path':'protocol-fixture','sha256':'a'*64},
            'config':raw.binding(config_path),'production_state':raw.binding(production)}
        state_path=root/'state.json';state={'schema':raw.SCHEMA,'plan_sha256':sha256_file_hex(plan_path),
            'owner':{**process_owner(),'start_ticks':'0'},'released':True,'owned_models':[name],'models':{name:value}}
        write_json_durable(state_path,state)
        result[name]={'plan':plan,'plan_path':plan_path,'state':state,'state_path':state_path,
            'config':{'out':str(root/'shared'),'raw_rl_out':str(root),'raw_rl_bootstrap':raw.binding(plan_path),
                'participants':[row['handoff_requirement']for row in entries]},'receipt':receipt}
    return result


def exchange(source,target,kind='released'):
    reference=roles.publish(source['plan_path'],source['state_path'],kind=kind)
    assert reference
    return roles.receive(Path(reference['path']).read_text(),target['plan_path'])


def test_default_owner_unchanged_partial_owner_requires_exact_partition():
    plan={'models':[{'model':'lfm'},{'model':'mellum'}]}
    assert roles.ownership(plan)==({'lfm','mellum'},None)
    with pytest.raises(ValueError,match='partial ownership'):
        roles.ownership({**plan,'owned_models':['lfm']})


@pytest.mark.parametrize('bad',['duplicate','missing','wrong-owner','consumer','empty-host'])
def test_role_partition_fail_closed(tmp_path,monkeypatch,bad):
    rows=fixture(tmp_path,monkeypatch);plan=rows['lfm']['plan'];assignment=plan['role_exchange']['roles']
    if bad=='duplicate':assignment['mellum']['owned_models']=['lfm']
    elif bad=='missing':assignment.pop('mellum')
    elif bad=='wrong-owner':plan['owned_models']=['mellum']
    elif bad=='consumer':plan['role_exchange']['semantic_consumer']='unassigned'
    else:assignment['lfm']['source_host_id']=''
    with pytest.raises(ValueError):roles.ownership(plan)


def test_live_source_can_attest_eligibility_but_cannot_release_optimizer(tmp_path,monkeypatch):
    from natlang_neuralese.train.native_resident import process_owner
    rows=fixture(tmp_path,monkeypatch);row=rows['lfm'];row['state']['owner']=process_owner()
    row['state']['released']=False;write_json_durable(row['state_path'],row['state'])
    assert roles.publish(row['plan_path'],row['state_path'],kind='released')is None
    reference=roles.publish(row['plan_path'],row['state_path'],kind='eligible')
    assert roles.validate(reference,row['plan'],kind='eligible')[0]=='lfm'
    assert not roles.aggregate(row['plan'])[0]
    row['state']['released']=True;write_json_durable(row['state_path'],row['state'])
    assert roles.publish(row['plan_path'],row['state_path'],kind='released')is None


def test_two_source_releases_still_need_real_local_assets_before_semantic_launch(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);left,right=rows['lfm'],rows['mellum']
    exchange(left,right);exchange(right,left)
    ready,pending=ready_participants(right['config'])
    assert not ready and len(pending)==2 and all(row['semantic_assets_required']for row in pending)
    request=roles.semantic_request(right['plan']);assert request
    assets={row['receipt']['model_id']:raw.binding(row['receipt']['checkpoint']['path'])for row in rows.values()}
    roles.install_assets(right['plan_path'],request,assets)
    ready,pending=ready_participants(right['config'])
    assert len(ready)==2 and not pending
    assert {row['checkpoint']['path']for row in ready}=={row['receipt']['checkpoint']['path']for row in rows.values()}
    assert not ready_participants(left['config'])[0]  # Single declared joint optimizer owner.
    assert roles.local_released(right['config'])


def test_remote_evidence_never_reads_remote_checkpoint_or_other_source_paths(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);left,right=rows['lfm'],rows['mellum']
    reference=exchange(left,right)
    # Transported inline receipts retain genuine original paths as provenance.
    # They remain verifiable after the remote source filesystem is unavailable.
    for path in (tmp_path/('lfm:'+'a'*64)).iterdir():path.unlink()
    role,models=roles.validate(reference,right['plan'],kind='released')
    assert role=='lfm' and len(models)==1
    assert not roles.aggregate(right['plan'])[0]


@pytest.mark.parametrize('damage',['backbone','profile','cohort','task','checkpoint','no-communication'])
def test_source_host_full_validator_rejects_unqualified_or_substituted_evidence(tmp_path,monkeypatch,damage):
    rows=fixture(tmp_path,monkeypatch);row=rows['lfm'];receipt=row['receipt']
    if damage in('backbone','profile'):
        path=Path(receipt['runtime_certificate']['path']);value=json.loads(path.read_text())
        if damage=='backbone':value['complete_gradient_qualification']['comparisons'].pop('backbone.weight')
        else:value['qualification_profile']['fp8']={'kind':'unqualified'}
        write_json_durable(path,value);receipt['runtime_certificate']=raw.binding(path)
    elif damage=='cohort':receipt['model_id']='other'
    elif damage=='task':
        path=Path(receipt['task_execution_receipt']['path']);value=json.loads(path.read_text());value['eligible']=False
        write_json_durable(path,value);receipt['task_execution_receipt']=raw.binding(path)
    elif damage=='checkpoint':Path(receipt['checkpoint']['path']).write_bytes(b'not-the-qualified-state')
    else:receipt['communication_exposure']['graph_visits']=0
    receipt_path=Path(row['state']['models']['lfm']['readiness']['path']);write_json_durable(receipt_path,receipt)
    row['state']['models']['lfm']['readiness']=raw.binding(receipt_path);write_json_durable(row['state_path'],row['state'])
    with pytest.raises((ValueError,KeyError)):
        roles.publish(row['plan_path'],row['state_path'],kind='released')


def test_import_rejects_rollback_equivocation_and_pending_invalidates_old_release(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);left,right=rows['lfm'],rows['mellum']
    first=roles.publish(left['plan_path'],left['state_path'],kind='released')
    raw_first=Path(first['path']).read_text();roles.receive(raw_first,right['plan_path'])
    proof=json.loads(raw_first);proof['source_host']['hostname']='different-source-attestation'
    with pytest.raises(ValueError,match='equivocation'):roles.receive(json.dumps(proof),right['plan_path'])
    left['state']['released']=False;write_json_durable(left['state_path'],left['state'])
    later=roles.withdraw(left['plan'],left['state_path']);roles.receive(Path(later['path']).read_text(),right['plan_path'])
    with pytest.raises(ValueError,match='rollback'):roles.receive(raw_first,right['plan_path'])
    found,pending=roles.exchanged(right['plan'],kind='released')
    assert not found and pending


def test_superseded_assets_or_false_content_cannot_be_installed(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);left,right=rows['lfm'],rows['mellum'];exchange(left,right);exchange(right,left)
    request=roles.semantic_request(right['plan'])
    assets={row['receipt']['model_id']:raw.binding(row['receipt']['checkpoint']['path'])for row in rows.values()}
    bad=copy.deepcopy(request);next(iter(bad['participants'].values()))['checkpoint']['sha256']='f'*64
    with pytest.raises(ValueError,match='superseded'):roles.install_assets(right['plan_path'],bad,assets)
    first=next(iter(assets));changed=tmp_path/'wrong-content';changed.write_text('different bytes')
    assets[first]={**assets[first],'path':str(changed)}
    with pytest.raises(ValueError):roles.install_assets(right['plan_path'],request,assets)


def test_any_two_genuine_roles_start_without_pending_future_third(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch,('lfm','mellum','future'))
    exchange(rows['lfm'],rows['mellum']);roles.publish(rows['mellum']['plan_path'],rows['mellum']['state_path'],kind='released')
    assert roles.eligible(rows['mellum']['plan'])
    request=roles.semantic_request(rows['mellum']['plan']);assert len(request['participants'])==2
    assets={row['receipt']['model_id']:raw.binding(row['receipt']['checkpoint']['path'])for name,row in rows.items()if name!='future'}
    assert roles.install_assets(rows['mellum']['plan_path'],request,assets)


def test_immutable_publication_does_not_clobber_racing_object(tmp_path,monkeypatch):
    actual=roles.os.link;path=tmp_path/'object.json'
    def raced(source,target):
        Path(target).write_text('other owner');return actual(source,target)
    monkeypatch.setattr(roles.os,'link',raced)
    with pytest.raises(ValueError,match='immutable'):roles._immutable(path,{'a':1})
    assert path.read_text()=='other owner'


def _service(monkeypatch=None):
    import importlib.util
    path=Path(__file__).parents[2]/'scripts/coreweave/native_role_exchange.py'
    spec=importlib.util.spec_from_file_location('role_exchange_service',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if monkeypatch is not None:
        # Only the local protocol fixture maps its temporary namespace. The
        # real binding shape/SHA/size verifier and production namespace stay strict.
        checked=module.checked_binding
        monkeypatch.setattr(module,'checked_binding',lambda ref:{**checked({**ref,'path':'/work/native-production/protocol-fixture'}),'path':ref['path']})
    return module


class FileProtocolIO:
    """Exercise actual remote source helpers using isolated protocol files."""
    def __init__(self,rows):
        self.config={'sandboxes':{name+'-actual-sid':{}for name in rows}}
        self.transfers=[];self.closures=[]
    def dependency_code(self,sid):return 'import json;'
    def read(self,sid,path):return Path(path).read_bytes()
    def read_optional(self,sid,path):return Path(path).read_bytes()if Path(path).exists()else None
    def remote(self,sid,code,**kwargs):
        from contextlib import redirect_stdout
        from io import StringIO
        output=StringIO()
        with redirect_stdout(output):exec(code,{})
        return json.loads(output.getvalue())
    def dependency_closure(self,sid,request):
        self.closures.append((sid,request))
        return {'files':request['semantic_participants'],'assets':[]}
    def mirror_many(self,source,target,items):self.transfers.append((source,target,items))
    def verify_assets(self,target,assets):assert assets==[]


def test_actual_role_relay_waits_for_consumption_then_transfers_only_remote_participant(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);module=_service(monkeypatch);io=FileProtocolIO(rows)
    config={'schema':'natlang.native-role-exchange-service/1','hosts':{
        name:{'source_host_id':name+'-actual-sid','plan':raw.binding(row['plan_path']),
              'raw_state':str(row['state_path'])}for name,row in rows.items()}}
    first=module.tick(config,io)
    assert first['status']=='waiting-for-semantic-consumption' and not io.transfers
    assert not ready_participants(rows['mellum']['config'])[0]
    # Bundle byte-copy mechanics are checked separately with a real Torch file.
    # These JSON checkpoint fixtures only attest the source-host protocol.
    pins=[]
    def pin(reference,directory):pins.append(reference);return reference
    monkeypatch.setattr(roles,'pin_checkpoint',pin)
    monkeypatch.setattr(roles,'recipe_recovery_needed',lambda ready:False)
    second=module.tick(config,io)
    assert second['status']=='semantic-assets-ready'
    assert second['transferred']==[rows['lfm']['receipt']['model_id']]
    assert len(io.transfers)==len(pins)==1
    assert io.transfers[0][:2]==('lfm-actual-sid','mellum-actual-sid')
    assert len(ready_participants(rows['mellum']['config'])[0])==2
    assert module.tick(config,io)['status']=='semantic-assets-ready'
    assert len(io.transfers)==1  # Exact replay does not repeat the full ferry.
    assert io.closures[0][1]=={'semantic_participants':[pins[0]]}


def test_service_never_seals_a_live_raw_owner(tmp_path,monkeypatch):
    from natlang_neuralese.train.native_resident import process_owner
    rows=fixture(tmp_path,monkeypatch);module=_service(monkeypatch);io=FileProtocolIO(rows)
    for row in rows.values():
        row['state']['owner']=process_owner();write_json_durable(row['state_path'],row['state'])
    config={'schema':'natlang.native-role-exchange-service/1','hosts':{
        name:{'source_host_id':name+'-actual-sid','plan':raw.binding(row['plan_path']),
              'raw_state':str(row['state_path'])}for name,row in rows.items()}}
    assert module.tick(config,io)['status']=='waiting-for-semantic-consumption'
    assert not io.transfers
    assert all(not(Path(row['plan']['role_exchange']['directory'])/(name+'-status.json')).exists()
               for name,row in rows.items())


def test_real_portable_pin_has_independent_inode_and_preserves_exact_tensors(tmp_path):
    import torch
    from natlang_neuralese.model.native_bundle import verify_bundle
    source=tmp_path/'source.pt';torch.save({'ordinary':torch.tensor([1.,2.,3.])},source)
    reference=raw.binding(source);target=roles.pin_checkpoint(reference,tmp_path/'pins')
    assert target['sha256']==reference['sha256']
    assert Path(target['path']).stat().st_ino!=source.stat().st_ino
    assert torch.equal(torch.load(target['path'],weights_only=True)['ordinary'],torch.tensor([1.,2.,3.]))
    source.write_bytes(b'new mutable output')
    assert roles.binding(target['path'])==target
    assert verify_bundle(tmp_path/'pins'/reference['sha256'])['checkpoint']['sha256']==reference['sha256']


def test_consumed_metadata_preserves_inherited_recipe_and_original_process_guards(tmp_path,monkeypatch):
    from natlang_neuralese.train.recipe import native_recipe_binding
    rows=fixture(tmp_path,monkeypatch);row=rows['lfm'];root=tmp_path/'recipes';root.mkdir()
    parent=root/'parent.json';write_json_durable(parent,{'schema':'natlang.neuralese-training-recipe/1',
        'id':'parent','stages':[{'id':'graph','kind':'native_state_sft','parameters':{'epochs':1}}]})
    child=root/'child.json';write_json_durable(child,{'schema':'natlang.neuralese-training-recipe/1',
        'id':'child','extends':'parent','overrides':{'stages':{'graph':{'parameters':{'epochs':2}}}}})
    reference=native_recipe_binding(child,'a'*64)
    assert reference['sha256']!=reference['file_sha256']
    ready_path=Path(row['state']['models']['lfm']['readiness']['path'])
    receipt=json.loads(ready_path.read_text());receipt['recipe']=reference;write_json_durable(ready_path,receipt)
    launch=tmp_path/'launch.json';write_json_durable(launch,{'argv':[str(child)],'at':1})
    terminal=tmp_path/'terminal.json';write_json_durable(terminal,{'returncode':0,'at':2})
    adoption=tmp_path/'resident-data/current.json';adoption.parent.mkdir();write_json_durable(adoption,{'bound':'source-host proof fixture'})
    selected={'recipe':reference,'readiness':raw.binding(ready_path),'readiness_requirement':{
        'trainer_launch':raw.binding(launch),'trainer_exit':str(terminal),'resident_adoption':raw.binding(adoption)}}
    metadata=roles.consumed_metadata(selected);paths={value['path']for value in metadata}
    assert {str(parent),str(child),str(launch),str(terminal),str(adoption)}<=paths
    assert next(value for value in metadata if value['path']==str(child))['sha256']==reference['file_sha256']
    assert selected['recipe']==reference
    package=roles.metadata_package(metadata,tmp_path/'immutable-transport')
    assert all(item['source']['sha256']==item['transport']['sha256']for item in package)
    for item in metadata:Path(item['path']).unlink()
    assert roles.missing_metadata(metadata)==metadata
    roles.install_metadata(package)
    assert not roles.missing_metadata(metadata)
    parent.write_text('different live authority')
    with pytest.raises(ValueError):roles.install_metadata(package)
    assert parent.read_text()=='different live authority'


def test_remote_v2_task_evidence_validates_without_original_score_file(tmp_path,monkeypatch):
    from test_native_task_execution_receipt import score_fixture
    from natlang_neuralese.eval.native_state import task_execution_receipt
    rows=fixture(tmp_path,monkeypatch);row=rows['lfm'];score_root=tmp_path/'scores';score_root.mkdir()
    summary,score,score_path=score_fixture(score_root)
    receipt=row['receipt'];task=task_execution_receipt(summary,receipt['native_identity'],
        checkpoint_sha256=receipt['checkpoint']['sha256'])
    assert task['passed']
    task_path=Path(receipt['task_execution_receipt']['path']);write_json_durable(task_path,task)
    receipt['task_execution_receipt']=raw.binding(task_path)
    ready_path=Path(row['state']['models']['lfm']['readiness']['path']);write_json_durable(ready_path,receipt)
    row['state']['models']['lfm']['readiness']=raw.binding(ready_path);write_json_durable(row['state_path'],row['state'])
    reference=exchange(row,rows['mellum']);score_path.unlink()
    assert roles.validate(reference,rows['mellum']['plan'],kind='released')[0]=='lfm'


def test_genuine_semantic_load_after_transport_recovers_original_legacy_recipe_and_optimizer(tmp_path):
    import torch
    from test_native_full_updates import source
    from test_native_readiness import ready
    from test_native_continuation import progress
    from test_native_shared_checkpoint import equal_tree
    from natlang_neuralese.model.native_model import load_native_model,model_state,native_identity,CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,TRAINING_SCHEMA
    from natlang_neuralese.train.native_readiness import validate_ready
    from natlang_neuralese.train.native_semantic_stage import load_participant
    base=tmp_path/'base';dtype=source(base,'lfm')
    model,tokenizer,markers,config=load_native_model(base,dtype=dtype,rank=4)
    optimizer=make_native_optimizer(model,lr=.001,in_backward=False)[0]
    sum(parameter.float().square().sum()for parameter in model.parameters()if parameter.requires_grad).backward()
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    training=progress();training.update(schema=TRAINING_SCHEMA,optimizer=optimizer.state_dict(),
        options={'stage_id':'graph','epochs':1},fp8=None)
    parent=tmp_path/'source-host/legacy.pt';parent.parent.mkdir()
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),'training':training},parent)
    identity=native_identity(model,config,checkpoint_sha256=sha256_file_hex(parent))
    evidence=tmp_path/'evidence';evidence.mkdir();path,receipt,requirement=ready(evidence,'lfm')
    receipt.update(checkpoint={'schema':CHECKPOINT_SCHEMA,**raw.binding(parent)},native_identity=identity)
    for field in('runtime_certificate','task_execution_receipt'):
        file=Path(receipt[field]['path']);value=json.loads(file.read_text());value['native_identity']=identity
        if field=='runtime_certificate':value['complete_gradient_qualification']['native_identity']=identity
        write_json_durable(file,value);receipt[field]=raw.binding(file)
    report_path=Path(receipt['training_receipt']['path']);report=json.loads(report_path.read_text())
    report['checkpoint_sha256']=identity['checkpoint_sha256'];write_json_durable(report_path,report)
    receipt['training_receipt']=raw.binding(report_path)
    launch=tmp_path/'trainer-launch.json';write_json_durable(launch,{'at':1,'argv':['trainer',receipt['recipe']['path']]})
    terminal=tmp_path/'trainer-exit.json';write_json_durable(terminal,{'at':2,'returncode':0})
    requirement.update(trainer_exit=str(terminal),trainer_launch=raw.binding(launch),trainer_recipe=receipt['recipe']['path'])
    write_json_durable(path,receipt);selected=validate_ready(path,requirement)
    assert roles.recipe_recovery_needed(selected)
    pinned=roles.pin_checkpoint(selected['checkpoint'],tmp_path/'consumer/pins')
    selected={**selected,'checkpoint':{**selected['checkpoint'],'path':pinned['path']}}
    parent.unlink()  # Original source-host absolute path is absent on the consumer.
    semantic={'backbone_lr':.001,'models':{'lfm':{'base':str(base),'fp8_training':'off'}}}
    with pytest.raises(FileNotFoundError):load_participant('lfm',selected,semantic,'cpu')
    original={**selected,'checkpoint':receipt['checkpoint']}
    roles.install_checkpoint_alias(original,pinned)
    loaded=load_participant('lfm',selected,semantic,'cpu')
    assert loaded.recipe_binding==receipt['recipe']
    equal_tree(loaded.optimizer.state_dict(),optimizer.state_dict())
    assert loaded.sft_progress['graph']['exposure_counts']==training['exposure_counts']
    terminal.unlink()
    with pytest.raises(ValueError,match='qualified original readiness'):load_participant('lfm',selected,semantic,'cpu')


def test_partial_owner_keeps_eligible_student_useful_while_peer_missing(tmp_path,monkeypatch):
    from types import SimpleNamespace
    rows=fixture(tmp_path,monkeypatch);row=rows['mellum'];plan=copy.deepcopy(row['plan'])
    for declaration in plan['models']:
        declaration['handoff_receipt']={'path':'original-bootstrap-protocol','sha256':'a'*64}
    write_json_durable(row['plan_path'],plan)
    calls=[]
    def cycle(plan_path,model,witness,directory,stop):
        calls.append(model['model']);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        production=directory/'production';production.mkdir(exist_ok=True)
        checkpoint=row['receipt']['checkpoint']['path']
        write_json_durable(production/'state.json',{'checkpoint':checkpoint})
        write_json_durable(directory/'config.json',{'out':str(production),'checkpoint':checkpoint,
            'runtime_binding':{'plan_sha256':sha256_file_hex(row['plan_path'])}})
        return {'readiness':raw.binding(Path(row['state']['models']['mellum']['readiness']['path']))}
    monkeypatch.setattr(raw,'student_cycle',cycle)
    result=raw.run(row['plan_path'],tmp_path/'raw-loop',cycles=3,stop=SimpleNamespace(requested=False),poll_seconds=.01)
    assert calls==['mellum']*3 and result['released']is False
    assert set(result['models'])=={'mellum'} and result['owned_models']==['mellum']
    reference,proof=roles._status(plan['role_exchange'],'mellum')
    assert proof['kind']=='eligible' and not proof['optimizer_released']
    assert roles.validate(reference,plan,kind='eligible')[0]=='mellum'


def test_new_valid_release_refreshes_old_assets_without_reusing_stale_authority(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);left,right=rows['lfm'],rows['mellum']
    exchange(left,right);exchange(right,left)
    request=roles.semantic_request(right['plan'])
    assets={row['receipt']['model_id']:roles.binding(row['receipt']['checkpoint']['path'])for row in rows.values()}
    roles.install_assets(right['plan_path'],request,assets)
    assert len(roles.ready_participants(right['plan'])[0])==2
    roles.withdraw(left['plan'],left['state_path'])
    exchange(left,right)
    current=roles.semantic_request(right['plan']);assert current!=request
    ready,pending=roles.ready_participants(right['plan'])
    assert not ready and all(item.get('semantic_assets_required')for item in pending)
    roles.install_assets(right['plan_path'],current,assets)
    assert len(roles.ready_participants(right['plan'])[0])==2


def _bind_role_mechanisms(rows, authorities):
    for row in rows.values():
        for name, value in row['plan']['role_exchange']['roles'].items():
            value['verification_mechanism']=dict(authorities[name])
        write_json_durable(row['plan_path'],row['plan'])
        row['config']['raw_rl_bootstrap']=raw.binding(row['plan_path'])
        row['state']['plan_sha256']=sha256_file_hex(row['plan_path'])
        name=row['plan']['role_exchange']['role']
        path=Path(row['state']['models'][name]['config']['path'])
        write_json_durable(path,{'runtime_binding':{'plan_sha256':sha256_file_hex(row['plan_path'])}})
        row['state']['models'][name]['config']=raw.binding(path)
        write_json_durable(row['state_path'],row['state'])


def test_exact_role_authority_exchanges_distinct_reviewed_mechanisms_with_real_validators(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);left,right=rows['lfm'],rows['mellum']
    actual=roles.mechanism();peer={**actual,'eval/native_gradients.py':'d'*64}
    _bind_role_mechanisms(rows,{'lfm':actual,'mellum':peer})
    proof=roles.publish(left['plan_path'],left['state_path'],kind='released')
    monkeypatch.setattr(roles,'mechanism',lambda:peer)
    received=roles.receive(Path(proof['path']).read_text(),right['plan_path'])
    assert roles.validate(received,right['plan'],kind='released')[0]=='lfm'
    # The peer's own declaration is also checked before publishing a genuine proof.
    assert roles.publish(right['plan_path'],right['state_path'],kind='released')
    assert roles.contract(left['plan'])==roles.contract(right['plan'])


@pytest.mark.parametrize('damage',['missing-role','missing-file','extra-file','invalid-sha'])
def test_role_mechanism_authority_requires_complete_exact_maps(tmp_path,monkeypatch,damage):
    rows=fixture(tmp_path,monkeypatch);actual=roles.mechanism()
    _bind_role_mechanisms(rows,{'lfm':actual,'mellum':actual})
    assignment=rows['lfm']['plan']['role_exchange']['roles']
    if damage=='missing-role':assignment['mellum'].pop('verification_mechanism')
    elif damage=='missing-file':assignment['lfm']['verification_mechanism'].pop('eval/native_gradients.py')
    elif damage=='extra-file':assignment['lfm']['verification_mechanism']['unreviewed']='a'*64
    else:assignment['lfm']['verification_mechanism']['eval/native_gradients.py']='*'
    with pytest.raises(ValueError,match='mechanism authority'):roles.ownership(rows['lfm']['plan'])


def test_role_authority_rejects_local_substitution_pending_and_release(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);actual=roles.mechanism()
    _bind_role_mechanisms(rows,{'lfm':actual,'mellum':actual})
    monkeypatch.setattr(roles,'mechanism',lambda:{**actual,'eval/native_gradients.py':'e'*64})
    with pytest.raises(ValueError,match='actual source-host mechanism'):
        roles.publish(rows['lfm']['plan_path'],rows['lfm']['state_path'],kind='released')
    with pytest.raises(ValueError,match='actual source-host mechanism'):
        roles.withdraw(rows['lfm']['plan'],rows['lfm']['state_path'])


def test_role_authority_rejects_transported_substitution_or_changed_contract(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);left,right=rows['lfm'],rows['mellum'];actual=roles.mechanism()
    _bind_role_mechanisms(rows,{'lfm':actual,'mellum':actual})
    reference=roles.publish(left['plan_path'],left['state_path'],kind='released')
    proof=json.loads(Path(reference['path']).read_bytes());proof['mechanism']['eval/native_gradients.py']='e'*64
    with pytest.raises(ValueError,match='authority'):roles.receive(json.dumps(proof),right['plan_path'])
    changed=copy.deepcopy(right['plan']);changed['role_exchange']['roles']['lfm']['verification_mechanism']['eval/native_gradients.py']='e'*64
    with pytest.raises(ValueError,match='mechanism or release scope'):
        roles.validate(reference,changed,kind='released')


def test_legacy_role_plan_keeps_exact_local_mechanism_requirement(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);reference=roles.publish(rows['lfm']['plan_path'],rows['lfm']['state_path'],kind='released');actual=roles.mechanism()
    monkeypatch.setattr(roles,'mechanism',lambda:{**actual,'eval/native_gradients.py':'d'*64})
    with pytest.raises(ValueError,match='authority'):
        roles.receive(Path(reference['path']).read_text(),rows['mellum']['plan_path'])


def test_declared_mechanism_records_bound_external_control_source_not_installed_origin(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_handoff
    installed=roles.mechanism();path=tmp_path/'reviewed-raw-handoff-control.py';path.write_text('# reviewed external control\n')
    reference={**raw.binding(path),'bytes':path.stat().st_size}
    monkeypatch.setattr(native_raw_handoff,'__native_external_control_source__',reference,raising=False)
    actual=roles.mechanism()
    assert actual['rl/native_raw_handoff.py']==reference['sha256']
    assert {k:v for k,v in actual.items()if k!='rl/native_raw_handoff.py'}=={k:v for k,v in installed.items()if k!='rl/native_raw_handoff.py'}
    path.write_text('# substituted control\n')
    with pytest.raises(ValueError,match='source binding changed'):roles.mechanism()


def test_peer_rejects_unvisited_foundation_row_despite_complete_aggregate(tmp_path,monkeypatch):
    """Exercise the actual transported verifier, independent of source-host flags."""
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_hex
    rows=fixture(tmp_path,monkeypatch);left,right=rows['lfm'],rows['mellum']
    ref=roles.publish(left['plan_path'],left['state_path'],kind='released')
    proof=json.loads(Path(ref['path']).read_text())
    plan=copy.deepcopy(right['plan'])
    requirement=next(x for x in plan['models'] if x['model']=='lfm')['handoff_requirement']
    requirement.update(readiness_boundary='communication-foundation',foundation_epochs=1.,
                       foundation_sources={'path':'/remote/foundation.json','sha256':'f'*64})
    source_plan=roles._evidence(proof['plan'])
    next(x for x in source_plan['models'] if x['model']=='lfm')['handoff_requirement']=copy.deepcopy(requirement)
    def replace(item,value):
        text=json.dumps(value);raw=text.encode();item['text']=text;item['source']['sha256']=sha256_hex(raw)
        if 'bytes' in item['source']:item['source']['bytes']=len(raw)
        return copy.deepcopy(item['source'])
    replace(proof['plan'],source_plan)
    proof['contract']=roles.contract(plan)
    item=proof['models']['lfm'];checked=item['ready']
    checked['readiness_requirement']=copy.deepcopy(requirement)
    row_values={key:{'binding':key+'-binding','required_visits':1,'graph_visits':1,'admitted_to_training':True}
                for key in ('producer-consumer-a','producer-consumer-b')}
    cohort={key:{name:row[name] for name in ('binding','required_visits')}for key,row in row_values.items()}
    foundation={'schema':'natlang.native-communication-foundation/1','epochs':1.,'rows':row_values,
        'sources':{'sha256':'f'*64},'cohort_sha256':canonical_json_sha256_hex(cohort),
        'graphs':2,'required_graph_visits':2,'graph_visits':2,'complete':True}
    obligations={'schema':'natlang.native-sft-progress/1','stages':{'remaining-ce':{'complete':False}}}
    def seal():
        report=roles._evidence(item['report']);report.update(communication_foundation=foundation,mandatory_sft=obligations)
        report_ref=replace(item['report'],report)
        receipt=roles._evidence(item['readiness']);receipt.update(communication_foundation=foundation,
            mandatory_sft=obligations,training_receipt=report_ref)
        receipt_ref=replace(item['readiness'],receipt)
        checked.update(readiness=receipt_ref,communication_foundation=copy.deepcopy(foundation))
        state=roles._evidence(proof['raw_state']);state['plan_sha256']=proof['plan']['source']['sha256']
        state['models']['lfm']['readiness']=receipt_ref;replace(proof['raw_state'],state)
        path=tmp_path/'transported-foundation-proof.json';write_json_durable(path,proof)
        return raw.binding(path)
    valid=seal()
    # Peer validation never needs the remote graph/checkpoint/source paths.
    assert roles.validate(valid,plan,kind='released')[0]=='lfm'
    foundation['rows']['producer-consumer-a']['graph_visits']=2
    foundation['rows']['producer-consumer-b']['graph_visits']=0
    with pytest.raises(ValueError,match='completed exact exposure for every row'):
        roles.validate(seal(),plan,kind='released')
