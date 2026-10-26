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


def install_actual_checkpoint(row,path):
    previous=row['receipt']['checkpoint'];replacement=raw.binding(path)
    def replace(value):
        if isinstance(value,dict):return {key:replace(item)for key,item in value.items()}
        if isinstance(value,list):return [replace(item)for item in value]
        if value==previous['sha256']:return replacement['sha256']
        if value==previous['path']:return replacement['path']
        return value
    receipt=replace(row['receipt'])
    for key in ('training_receipt','runtime_certificate','task_execution_receipt'):
        target=Path(receipt[key]['path']);body=replace(json.loads(target.read_text()))
        if key=='training_receipt':body.pop('foundation_qualification_schedule',None)
        write_json_durable(target,body);receipt[key]=roles.binding(target)
    value=row['state']['models'][row['plan']['owned_models'][0]]
    receipt_path=Path(value['readiness']['path']);write_json_durable(receipt_path,receipt)
    value.update(readiness=roles.binding(receipt_path),checkpoint=replacement)
    production=Path(value['production_state']['path']);write_json_durable(production,{'checkpoint':str(path)})
    value['production_state']=roles.binding(production);row['receipt']=receipt
    write_json_durable(row['state_path'],row['state'])


def advance_actual_mellum(row):
    import torch
    from test_native_semantic import setup_pair
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.train.native_semantic import parse_pair
    from natlang_neuralese.data.native_graph import graph_loss
    actors,episode,_=setup_pair();actor=actors['mellum']
    actor.model.port.hf.natlang_precision='bf16'
    optimizer=make_native_optimizer(actor.model,lr=1e-3,in_backward=False)[0]
    parameters={key:value.detach().clone()for key,value in actor.model.named_parameters()}
    graph=parse_pair(episode,actors)['mellum'][0]
    loss,_=graph_loss(actor.model,graph,checkpoint_layers=False);loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert any(not torch.equal(parameters[key],value)for key,value in actor.model.named_parameters())
    checkpoint=Path(row['receipt']['checkpoint']['path']).parent/'operational-descendant.pt'
    torch.save({'model':actor.model.state_dict(),'optimizer':optimizer.state_dict(),
        'rng':torch.get_rng_state(),'earned_updates':1},checkpoint)
    row['state']=json.loads(row['state_path'].read_text())
    install_actual_checkpoint(row,checkpoint)
    return checkpoint


def exchange(source,target,kind='released'):
    reference=roles.publish(source['plan_path'],source['state_path'],kind=kind)
    assert reference
    return roles.receive(Path(reference['path']).read_text(),target['plan_path'])


def test_default_owner_unchanged_partial_owner_requires_exact_partition():
    plan={'models':[{'model':'lfm'},{'model':'mellum'}]}
    assert roles.ownership(plan)==({'lfm','mellum'},None)
    with pytest.raises(ValueError,match='partial ownership'):
        roles.ownership({**plan,'owned_models':['lfm']})


def test_current_full_checkpoint_proof_cache_is_bound_to_exact_bytes(tmp_path,monkeypatch):
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    path=tmp_path/'checkpoint.pt';path.write_bytes(b'first full-state bytes')
    calls=[]
    monkeypatch.setattr(CheckpointPolicy,'load',lambda selected:(calls.append(str(selected))or{
        'schema':'natlang.neuralese-native-checkpoint/1','model_config':{'architecture':'lfm2',
            'base_inventory_sha256':'a'*64},'training':{'optimizer':{'real':True},'rng':{'real':True},
                'source_receipts':[{}],'options':{'native_snapshot_ancestors':['b'*64]}}}))
    first=roles.binding(path)
    assert roles._local_cycle_checkpoint_proof(first)==roles._local_cycle_checkpoint_proof(first)
    assert len(calls)==1
    path.write_bytes(b'second exact full-state bytes')
    second=roles.binding(path)
    assert roles._local_cycle_checkpoint_proof(second)['model_config']['architecture']=='lfm2'
    assert len(calls)==2


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
        self.config={'sandboxes':{}}
        self.collector_assets={}
        for name,row in rows.items():
            runtime=Path(row['plan_path']).parent/'runtime';script=runtime/'scripts/run_bonsai_queue.py'
            script.parent.mkdir(parents=True,exist_ok=True);script.write_text('# fixture collector\n')
            self.config['sandboxes'][name+'-actual-sid']={'runtime':str(runtime)}
            self.collector_assets[name]={'path':str(script),'kind':'task-source-file',
                'sha256':sha256_file_hex(script),'bytes':script.stat().st_size}
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
    def verify_assets(self,target,assets):
        for asset in assets:
            assert asset['kind']=='task-source-file'
            path=Path(asset['path'])
            assert path.is_file() and sha256_file_hex(path)==asset['sha256']


def test_actual_role_relay_waits_for_consumption_then_transfers_only_remote_participant(tmp_path,monkeypatch):
    rows=fixture(tmp_path,monkeypatch);module=_service(monkeypatch);io=FileProtocolIO(rows)
    config={'schema':'natlang.native-role-exchange-service/1','hosts':{
        name:{'source_host_id':name+'-actual-sid','plan':raw.binding(row['plan_path']),
              'raw_state':str(row['state_path'])}for name,row in rows.items()},
        'task_eval_collector_assets':io.collector_assets}
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
              'raw_state':str(row['state_path'])}for name,row in rows.items()},
        'task_eval_collector_assets':io.collector_assets}
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
        identity={'optimizer_options':{'lr':.001,'weight_decay':.1,'latent_ns':'quintic',
            'learned_row_lr':None,'interface_row_lr':None,'native_optimizer_migration':None},
            'fp8':{'kind':'off','recipe':'delayed','amax_history_len':16}},
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


def test_local_cycle_shared_descendant_requires_exact_deployment_updated_participant(tmp_path):
    from natlang_neuralese.rl.native_role_release import _validate_current_deployment_participant
    from natlang_neuralese.train.native_resident import binding
    checkpoint=tmp_path/'current.pt';checkpoint.write_bytes(b'current full-state descendant')
    actual=binding(checkpoint)
    deployment={'updated_participants':{'lfm':actual}}
    assert _validate_current_deployment_participant(deployment,'lfm',actual)==actual
    with pytest.raises(ValueError,match='omitted its current participant descendant'):
        _validate_current_deployment_participant({'updated_participants':{}},'lfm',actual)
    with pytest.raises(ValueError,match='not the current checkpoint descendant'):
        _validate_current_deployment_participant({'updated_participants':{'lfm':{
            **actual,'sha256':'0'*64}}},'lfm',actual)


def test_partial_owner_keeps_eligible_student_useful_while_peer_missing(tmp_path,monkeypatch):
    from types import SimpleNamespace
    rows=fixture(tmp_path,monkeypatch);row=rows['mellum'];plan=copy.deepcopy(row['plan'])
    for declaration in plan['models']:
        declaration['handoff_receipt']={'path':'original-bootstrap-protocol','sha256':'a'*64}
    write_json_durable(row['plan_path'],plan)
    calls=[]
    def cycle(plan_path,model,witness,directory,stop,owned_controller_fd=None):
        raw.verify_controller_fd(owned_controller_fd,Path(directory).parent/'controller.lock')
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


def test_source_host_relay_loads_actual_bound_publishing_controls(tmp_path,monkeypatch):
    service=_service(monkeypatch)
    marker=tmp_path/'loaded.txt'
    controller=tmp_path/'controller.py'
    controller.write_text('from pathlib import Path\ndef load_controls(spec):\n    Path(spec["marker"]).write_text("actual-load")\n')
    spec=tmp_path/'controls.json'
    write_json_durable(spec,{'controller':{**raw.binding(controller),'bytes':controller.stat().st_size},'marker':str(marker)})
    class IO:
        config={'sandboxes':{'actual':{'role_controls':{**raw.binding(spec),'bytes':spec.stat().st_size}}}}
        def dependency_code(self,sid):return 'import json;'
        def remote(self,sid,code,**kwargs):
            from contextlib import redirect_stdout
            from io import StringIO
            stream=StringIO()
            with redirect_stdout(stream):exec(code,{})
            return json.loads(stream.getvalue())
    assert service.call(IO(),'actual','{"passed":True}')=={'passed':True}
    assert marker.read_text()=='actual-load'
    marker.unlink();controller.write_text('raise AssertionError("untrusted controller must not execute")\n')
    with pytest.raises(AssertionError):service.call(IO(),'actual','{"passed":True}')
    assert not marker.exists()


def test_historical_eligibility_cannot_release_and_fresh_descendant_pause_needs_both_lease_providers(tmp_path,monkeypatch):
    from test_native_role_scheduler_deployment import prepared_snapshot,attempt_for
    from natlang_neuralese.train.native_readiness import snapshot_successor
    from natlang_neuralese.train.native_resident import process_owner
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    import time
    metadata=tmp_path/'metadata';metadata.mkdir()
    config,_,_=prepared_snapshot(metadata)
    with monkeypatch.context() as local:
        local.setattr(roles,'mechanism',lambda:{key:'a'*64 for key in roles.MECHANISM_FILES})
        local.setattr(roles,'ready_participants',lambda *a,**kw:([],[]))
        waiting=snapshot_successor(config,Path(config['out']))
    request=json.loads(Path(waiting['awaiting_descendant_handoff']['path']).read_text())
    rows=fixture(tmp_path,monkeypatch)
    # Real tiny native actor/optimizer states are advanced below. The surrounding
    # runtime/task certificates remain explicit protocol fixtures, not CUDA or
    # behavioral qualification; their ordinary strict validators still execute.
    import torch
    from test_native_semantic import setup_pair
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.data.native_graph import graph_loss
    from natlang_neuralese.train.native_semantic import parse_pair
    actors,episode,_=setup_pair()
    actors['mellum'].model.port.hf.natlang_precision='bf16'
    optimizers={name:make_native_optimizer(entry.model,lr=1e-3,in_backward=False)[0]for name,entry in actors.items()}
    install_checkpoint=install_actual_checkpoint
    for name,row in rows.items():
        path=Path(row['receipt']['checkpoint']['path']).parent/'native-step-0.pt'
        torch.save({'model':actors[name].model.state_dict(),'optimizer':optimizers[name].state_dict(),
            'rng':torch.get_rng_state(),'earned_updates':0},path)
        install_checkpoint(row,path)
    monkeypatch.setattr(roles,'pin_checkpoint',lambda reference,directory:roles.binding(reference['path']))
    for name,row in rows.items():
        row['plan']['snapshot_preparation']=True
        for declaration in row['plan']['models']:declaration['foundation_qualification_generation']='same-actual-repair'
        write_json_durable(row['plan_path'],row['plan'])
        row['state'].update(owner=process_owner(),released=False,plan_sha256=sha256_file_hex(row['plan_path']))
        value=row['state']['models'][name];path=Path(value['config']['path'])
        write_json_durable(path,{'runtime_binding':{'plan_sha256':sha256_file_hex(row['plan_path'])}})
        value['config']=raw.binding(path);write_json_durable(row['state_path'],row['state'])
    request['contract']=roles.contract(rows['mellum']['plan']);request['mechanism']=roles.mechanism()
    offers={name:{'role':name,'contract':roles.contract(row['plan']),'mechanism':roles.mechanism(),
        'owner':process_owner(),'released':False,'request_sha256':canonical_json_sha256_hex(request),
        'checkpoints':{name:row['state']['models'][name]['checkpoint']},
        'qualification_generations':{name:'same-actual-repair'},'provider_inputs':{name:{}}}for name,row in rows.items()}
    attempt=attempt_for(rows['mellum']['plan'],request,offers)
    for row in rows.values():roles.renew_handoff(row['plan'],request,offers,expires_at=time.time()+180,attempt=attempt)
    exchange(rows['lfm'],rows['mellum'],kind='eligible');exchange(rows['mellum'],rows['lfm'],kind='eligible')
    assert not roles.eligible(rows['mellum']['plan']) and not roles.handoff_hold(rows['lfm']['plan'])
    for name,row in rows.items():
        generation=canonical_json_sha256_hex({'foundation_generation':'same-actual-repair',
            'handoff':attempt['source']['sha256']})
        path=tmp_path/(name+'-actual-request.json');write_json_durable(path,{'generation':generation})
        report_path=Path(row['receipt']['training_receipt']['path'])
        report=json.loads(report_path.read_text());report['foundation_qualification_schedule']={
            'request':roles.binding(path),'requested':True,'readiness_granted':True}
        write_json_durable(report_path,report)
        row['receipt']['training_receipt']=roles.binding(report_path)
        receipt_path=Path(row['state']['models'][name]['readiness']['path'])
        write_json_durable(receipt_path,row['receipt'])
        row['state']['models'][name]['readiness']=roles.binding(receipt_path)
        row['state']['models'][name]['foundation_qualification_attempt']={'request':roles.binding(path),
            'status':'passed','checkpoint':row['state']['models'][name]['checkpoint'],'report':roles.binding(report_path)}
        write_json_durable(row['state_path'],row['state'])
    exchange(rows['lfm'],rows['mellum'],kind='eligible')
    assert roles.handoff_hold(rows['lfm']['plan'])
    assert not roles.eligible(rows['lfm']['plan'])  # Peer still has only historical eligibility.
    exchange(rows['mellum'],rows['lfm'],kind='eligible')
    assert roles.eligible(rows['lfm']['plan']) and roles.handoff_hold(rows['mellum']['plan'])
    # A stale relay's natural lease expiry releases both pauses automatically.
    # No current proof or optimizer/source checkpoint is rewritten by expiry.
    expired=time.time()+181
    monkeypatch.setattr(roles.time,'time',lambda:expired)
    assert not roles.handoff_hold(rows['lfm']['plan']) and not roles.eligible(rows['lfm']['plan'])

    # Expiry then real useful SGD advances exactly the same tiny
    # actor and its optimizer. That must allow a NEW authenticated rendezvous.
    old_attempt=attempt
    first=rows['lfm'];roles.withdraw(first['plan'],first['state_path'])
    before={name:value.detach().clone()for name,value in actors['lfm'].model.named_parameters()}
    examples=parse_pair(episode,actors)
    loss,_=graph_loss(actors['lfm'].model,examples['lfm'][0],checkpoint_layers=False)
    loss.backward();optimizers['lfm'].step();optimizers['lfm'].zero_grad(set_to_none=True)
    assert any(not torch.equal(before[name],value)for name,value in actors['lfm'].model.named_parameters())
    path=Path(first['receipt']['checkpoint']['path']).parent/'native-step-1.pt'
    torch.save({'model':actors['lfm'].model.state_dict(),'optimizer':optimizers['lfm'].state_dict(),
        'rng':torch.get_rng_state(),'earned_updates':1},path)
    install_checkpoint(first,path)
    roles.withdraw(first['plan'],first['state_path'])
    peer_reference,_=roles._status(rows['mellum']['plan']['role_exchange'],'mellum')
    offers['lfm'].update(checkpoints={'lfm':first['state']['models']['lfm']['checkpoint']},
        active_attempt=None,current_qualification=None)
    offers['mellum'].update(current_qualification=peer_reference,active_attempt=None)
    attempt=attempt_for(rows['mellum']['plan'],request,offers)
    assert attempt['source']['sha256']!=old_attempt['source']['sha256']
    document=roles._evidence(attempt)
    assert document['ordinal']==2 and document['superseded_attempt']==old_attempt['source']
    for row in rows.values():roles.renew_handoff(row['plan'],request,offers,expires_at=time.time()+180,attempt=attempt)
    # Unchanged current peer proof may be carried exactly; advanced LF cannot.
    assert roles.handoff_hold(rows['mellum']['plan']) and not roles.handoff_hold(first['plan'])
    generation=canonical_json_sha256_hex({'foundation_generation':'same-actual-repair',
        'handoff':attempt['source']['sha256']})
    request_path=tmp_path/'lfm-fresh-rendezvous-request.json';write_json_durable(request_path,{'generation':generation})
    report_path=Path(first['receipt']['training_receipt']['path']);report=json.loads(report_path.read_text())
    report['foundation_qualification_schedule']={'request':roles.binding(request_path),'requested':True,'readiness_granted':True}
    write_json_durable(report_path,report);first['receipt']['training_receipt']=roles.binding(report_path)
    value=first['state']['models']['lfm'];receipt_path=Path(value['readiness']['path'])
    write_json_durable(receipt_path,first['receipt']);value['readiness']=roles.binding(receipt_path)
    value['foundation_qualification_attempt']={'request':roles.binding(request_path),'status':'passed',
        'checkpoint':value['checkpoint'],'report':roles.binding(report_path)}
    write_json_durable(first['state_path'],first['state'])
    exchange(first,rows['mellum'],kind='eligible')
    assert roles.handoff_hold(first['plan']) and roles.eligible(rows['mellum']['plan'])
    actual=torch.load(path,weights_only=False,map_location='cpu')
    assert actual['earned_updates']==1 and actual['optimizer']
    with pytest.raises(ValueError,match='roll back'):
        roles.renew_handoff(first['plan'],request,offers,expires_at=time.time()+180,attempt=old_attempt)
    assert roles.handoff_active(first['plan'])['attempt_sha256']==attempt['source']['sha256']


@pytest.mark.parametrize('repair_required',[True,False])
def test_consumer_allocation_handoff_requires_staged_snapshots_and_dead_owner_then_reopens_raw(tmp_path,monkeypatch,repair_required):
    from natlang_neuralese.train.native_resident import process_owner
    from natlang_neuralese.train import native_readiness
    from test_native_role_scheduler_deployment import source
    from natlang_neuralese.train.native_deployment import binding
    rows=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(roles,'pin_checkpoint',lambda ref,directory:roles.binding(ref['path']))
    for name,row in rows.items():
        row['plan'].update(snapshot_preparation=True,consumer_preparation_handoff=True)
        for declaration in row['plan']['models']:declaration['foundation_qualification_generation']='repair-1'
        write_json_durable(row['plan_path'],row['plan'])
        row['state'].update(owner=process_owner(),released=False,plan_sha256=sha256_file_hex(row['plan_path']))
        value=row['state']['models'][name];config_path=Path(value['config']['path'])
        write_json_durable(config_path,{'runtime_binding':{'plan_sha256':sha256_file_hex(row['plan_path'])}})
        value['config']=raw.binding(config_path);write_json_durable(row['state_path'],row['state'])
    exchange(rows['lfm'],rows['mellum'],kind='eligible');exchange(rows['mellum'],rows['lfm'],kind='eligible')
    consumer=rows['mellum'];plan=consumer['plan']
    consumer['config']['raw_rl_bootstrap']=raw.binding(consumer['plan_path'])
    _,_,curriculum,_,_,_=source(tmp_path/'semantic-inputs')
    config=json.loads(curriculum.read_text());config['snapshot_preparation']=True
    from natlang_neuralese.train.native_semantic_stage import validate_config
    config=validate_config(config)
    write_json_durable(curriculum,config)
    request=roles.semantic_request(plan,snapshot=True)
    with pytest.raises(ValueError,match='assets must exist'):
        roles.arm_preparation_handoff(plan,request,binding(curriculum))
    ready,_=roles.exchanged(plan,kind='snapshot')
    roles.install_assets(consumer['plan_path'],request,{name:value['checkpoint']for name,value in ready.items()})
    reference=roles.arm_preparation_handoff(plan,request,binding(curriculum))
    assert reference and roles.eligible(plan)
    assert not roles.eligible(rows['lfm']['plan'])  # Peer never pauses for preparation.
    consumer['config'].update(curriculum=str(curriculum),manage_raw_bootstrap=False)
    ready,pending=native_readiness.ready_participants(consumer['config'])
    assert not ready and 'allocation release' in pending[0]['reason']
    with pytest.raises(ValueError,match='live RAW owner'):
        roles.cancel_preparation_handoff(consumer['config'],reason='cannot signal actual live owner')
    # Model-free protocol fixture: actual RAW loop alone may set released, and
    # publish(released) still refuses while its genuine process is alive.
    consumer['state']['released']=True
    write_json_durable(consumer['state_path'],consumer['state'])
    assert roles.publish(consumer['plan_path'],consumer['state_path'],kind='released')is None
    consumer['state']['owner']['start_ticks']='0'
    write_json_durable(consumer['state_path'],consumer['state'])
    assert roles.publish(consumer['plan_path'],consumer['state_path'],kind='released')
    ready,pending=native_readiness.ready_participants(consumer['config'])
    assert len(ready)==2 and not pending
    before=copy.deepcopy(consumer['state']['models'])
    cancellation=roles.cancel_preparation_handoff(consumer['config'],reason='owned VAE child drained',repair_required=repair_required)
    assert cancellation and not cancellation['optimizer_reset']
    after=json.loads(consumer['state_path'].read_text())
    assert after['models']==before and after['released']is False
    assert not roles.preparation_handoff(plan) and native_readiness._manage_raw_bootstrap(consumer['config'])
    if repair_required:
        assert roles.arm_preparation_handoff(plan,request,binding(curriculum))is None
    assert not roles.eligible(plan)
    # Simulate process loss after cancellation intent but before cursor rewrite.
    original=json.loads(Path(cancellation['released_state']['path']).read_text())
    write_json_durable(consumer['state_path'],original)
    assert roles.preparation_raw_recovery(consumer['config'])
    assert json.loads(consumer['state_path'].read_text())['released']is False
    # Neither full checkpoint bytes nor the optimizer/exposure-bearing cursor changed.
    assert json.loads(consumer['state_path'].read_text())['models']==before
    if not repair_required:
        # Actual whole-graph native SGD between operational cancellation and
        # the next ownership attempt; no fake runtime/task numerical claim.
        import torch
        checkpoint=advance_actual_mellum(consumer);roles.withdraw(plan,consumer['state_path'])
        successor=roles.arm_preparation_handoff(plan,request,binding(curriculum))
        assert successor and successor['sha256']!=reference['sha256']
        assert roles.preparation_handoff(plan)['value']['ownership_attempt']['ordinal']==2
        assert not roles.eligible(plan)  # Pending advanced weights need actual fresh proof.
        with monkeypatch.context()as local:
            local.setattr(roles,'foundation_candidates',lambda plan:({'lfm':{},'mellum':{}},{}))
            request_ref=roles.foundation_request(plan,tmp_path/'resumed-checks',generation='repair-1')
        assert request_ref
        generation=json.loads(Path(request_ref['path']).read_text())['generation']
        from natlang_neuralese.common.hashing import canonical_json_sha256_hex
        assert generation==canonical_json_sha256_hex({'foundation_generation':'repair-1','preparation':successor['sha256']})
        assert roles.publish(consumer['plan_path'],consumer['state_path'],kind='eligible')
        assert roles.semantic_request(plan,snapshot=True)==request  # Resume the SAME saved VAE parents.
        assert roles.eligible(plan)
        saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
        assert saved['earned_updates']==1 and saved['optimizer']
    # A changed repair generation invalidates the old request without blocking
    # useful RAW recovery before the fresh relay request has been staged.
    for declaration in plan['models']:declaration['foundation_qualification_generation']='repair-2'
    write_json_durable(consumer['plan_path'],plan)
    consumer['config']['raw_rl_bootstrap']=raw.binding(consumer['plan_path'])
    assert roles.preparation_handoff(plan)is None
    if repair_required:
        # Completion-format fixture: numerical VAE/optimizer execution is tested
        # separately by the real shared semantic-stage integration test.
        from natlang_neuralese.common.hashing import canonical_json_sha256_hex
        active=roles.arm_preparation_handoff(plan,request,binding(curriculum))
        assert active
        full=json.loads(curriculum.read_text());full['snapshot_preparation']=False
        full['models']={row['handoff_requirement']['model_id']:{'task_eval':{
            **full['models'][row['model']+':base']['task_eval']}}for row in plan['models']}
        joint=tmp_path/'actual-joint.json';write_json_durable(joint,full)
        checkpoint=tmp_path/'vae.pt';checkpoint.write_text('protocol VAE checkpoint')
        adapter=tmp_path/'vae-adapter.json';write_json_durable(adapter,{
            'schema':'natlang.native-semantic-adapter/1','checkpoint':binding(checkpoint)})
        complete=tmp_path/'frozen-complete.json';write_json_durable(complete,{
            'schema':'natlang.native-shared-stage-completion/1','completed':False,
            'production_ready':False,'preparation_complete':True,
            'snapshot_preparation':{'participant_optimizer_updates':0,'participants':{
                name:{'checkpoint_sha256':value['checkpoint']['sha256']}
                for name,value in request['participants'].items()}},
            'participants_signature':'fixture','curriculum_signature':canonical_json_sha256_hex(config),
            'curriculum':binding(curriculum),'checkpoint':binding(checkpoint),'adapter':binding(adapter)})
        post=roles.post_vae_request(plan,binding(complete),{'curriculum':binding(joint)})
        handoff=json.loads(Path(post['path']).read_text())
        assert roles.handoff_matches_preparation_work(plan,handoff,binding(curriculum))
        assert roles.preparation_handoff(plan)is None
        assert roles.arm_preparation_handoff(plan,request,binding(curriculum))is None
        assert not roles.eligible(plan)  # No bypass of actual post-VAE lease/proof matching.
        assert (Path(plan['role_exchange']['directory'])/'preparation-consumed'/(active['sha256']+'.json')).is_file()
        # Actual model progress/new valid current proof cannot create more of
        # the same declared VAE work. Explicit curriculum work can use it.
        advance_actual_mellum(consumer)
        consumer['state']['plan_sha256']=sha256_file_hex(consumer['plan_path'])
        value=consumer['state']['models']['mellum'];config_path=Path(value['config']['path'])
        write_json_durable(config_path,{'runtime_binding':{'plan_sha256':sha256_file_hex(consumer['plan_path'])}})
        value['config']=raw.binding(config_path);write_json_durable(consumer['state_path'],consumer['state'])
        assert roles.publish(consumer['plan_path'],consumer['state_path'],kind='eligible')
        fresh=roles.semantic_request(plan,snapshot=True)
        assert fresh!=request
        assert roles.preparation_work_complete(plan,binding(curriculum))
        assert roles.arm_preparation_handoff(plan,fresh,binding(curriculum))is None
        relocated=tmp_path/'same-curriculum-new-path.json';write_json_durable(relocated,config)
        assert roles.preparation_work_complete(plan,binding(relocated))
        adjusted=tmp_path/'adjusted-frozen.json';adjusted_config=copy.deepcopy(config)
        adjusted_config['snapshot_work_generation']='explicit-next-adaptation'
        write_json_durable(adjusted,adjusted_config)
        roles.select_preparation_work(plan,binding(adjusted))
        assert not roles.handoff_matches_preparation_work(plan,handoff,binding(adjusted))
        assert not roles.preparation_work_complete(plan,binding(adjusted))
        assert roles.semantic_request(plan,snapshot=True)==fresh
        selected,_=roles.exchanged(plan,kind='snapshot')
        roles.install_assets(consumer['plan_path'],fresh,{name:row['checkpoint']for name,row in selected.items()})
        successor=roles.arm_preparation_handoff(plan,fresh,binding(adjusted))
        assert successor and roles.preparation_handoff(plan)['value']['snapshots']==fresh
