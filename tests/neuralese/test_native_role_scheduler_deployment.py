"""Metadata scheduling checks; no claim of full production CUDA readiness."""
import copy
import json
from pathlib import Path
import sys
import pytest
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.train.native_deployment import binding, prepare_role_scheduler
from natlang_neuralese.rl import native_role_release as roles


def attempt_for(plan,request,offers):
    consumer=copy.deepcopy(plan)
    role=consumer['role_exchange']['semantic_consumer']
    consumer['role_exchange']['role']=role
    consumer['owned_models']=consumer['role_exchange']['roles'][role]['owned_models']
    return roles.select_handoff_attempt(consumer,request,offers,lease_seconds=180)


def source(tmp_path):
    runtime=tmp_path/'runtime';package=runtime/'training/neuralese/natlang_neuralese'
    (package/'train').mkdir(parents=True);(package/'train/native_readiness.py').write_text('# actual fixture module\n')
    controller=tmp_path/'controller.py';controller.write_text('def load_controls(spec): return spec\n')
    from natlang_neuralese.data.native_pair_schema import PAIR_SCHEMA
    pairs=tmp_path/'pairs.jsonl';held=tmp_path/'held.jsonl'
    for path,split in ((pairs,'train'),(held,'test')):
        path.write_text(json.dumps({'schema':PAIR_SCHEMA,'id':split+':pair','split':split,
            'source_groups':[split+':group'],'models':{'lfm:base':{},'mellum:base':{}}})+'\n')
    semantic=tmp_path/'semantic.json';write(semantic,{'schema':'natlang.native-semantic-training/1',
        'backbone_lr':1e-5,'vae_lr':1e-4,'latent_width':8,'hidden_width':16,
        'checkpoint_every':1,'eval_every':1,
        'pairs':binding(pairs),'held_pairs':binding(held),'task_eval':{'packet':binding(held)},
        'policy_schedule':{'id':'policy','epochs':1,'every_semantic_updates':1,'updates_per_refresh':1},
        'models':{name+':base':{'task_eval':{'packet':binding(held),'cases':[0],
            'reference_successes':floor,'reference_provenance':label}}for name,floor,label in
            [('lfm',0,'explicit owner historical zero floor fixture'),
             ('mellum',1,'separately bound baseline fixture')]},
        'phases':[{'id':kind,'kind':kind,'epochs':1}for kind in ('vae','joint','distill')]})
    assignment={name:{'owned_models':[name],'source_host_id':name+'-existing',
        'verification_mechanism':{key:'a'*64 for key in roles.MECHANISM_FILES}}for name in ('lfm','mellum')}
    models=[{'model':name,'production_recipe':{'path':str(tmp_path/(name+'-recipe.json')),'sha256':'b'*64},
        'handoff_requirement':{'model_id':name+':base','stage_ids':['full'],
            'readiness_boundary':'communication-foundation','foundation_epochs':1,
            'foundation_sources':binding(pairs),'forward_atol':1e-4,'gradient_atol':1e-5},
        'foundation_readiness_policy':'on-shared-request','foundation_qualification_generation':'real-repair-generation',
        'retained_optimizer_binding':{'exact':name},'unfinished_recipe':['sft','dpo','policy']}
        for name in ('lfm','mellum')]
    for row in models:
        recipe=tmp_path/(row['model']+'-recipe.json');write(recipe,{'fixture':'metadata only'})
        row.update(production_recipe=binding(recipe),native_participant=row['model']+':base',
            cases=binding(pairs),seed_native_graphs=binding(pairs),held_native_graphs=binding(held))
    bootstrap=tmp_path/'rl-bootstrap.json';write(bootstrap,{'schema':'natlang.native-rl-bootstrap/1','models':models})
    plans={};controls={};states={}
    for name in assignment:
        path=tmp_path/(name+'-plan.json');write(path,{'schema':'natlang.native-raw-rl-bootstrap/1',
            'models':models,'owned_models':[name],'role_exchange':{'schema':roles.EXCHANGE,'role':name,
                'roles':assignment,'directory':str(tmp_path/(name+'-exchange')),'semantic_consumer':'mellum'}})
        plans[name]=binding(path)
        spec=tmp_path/(name+'-controls.json');write(spec,{'model':name,'controller':binding(controller),'numerical_package':str(package)})
        controls[name]=binding(spec);states[name]=str(tmp_path/name/'raw/state.json')
    provider=tmp_path/'provider.json';write(provider,{'coordinator_id':'mellum-existing',
        'sandboxes':{name+'-existing':{'runtime':str(runtime)}for name in assignment}})
    return plans,controls,semantic,provider,states,runtime


def test_prepares_exact_external_owner_configs_without_claiming_adoption(tmp_path):
    plans,controls,curriculum,provider,states,runtime=source(tmp_path)
    before={name:Path(ref['path']).read_bytes()for name,ref in plans.items()}
    result=prepare_role_scheduler(plans,controls,curriculum,provider,tmp_path/'prepared',raw_states=states,runtime=runtime,
        rl_bootstrap=binding(tmp_path/'rl-bootstrap.json'),handoff_lease_seconds=180)
    assert result['owner_changes']==result['checkpoint_decodes']==result['model_allocations']==0
    assert result['readiness_granted'] is result['activated'] is False
    for record in result['source_plan_successions']:
        old=json.loads(before[record['role']]);new=json.loads(Path(record['successor']['path']).read_text())
        assert new.pop('snapshot_preparation') is True
        assert new.pop('consumer_preparation_handoff') is True
        assert old==new and record['applied'] is False
    config=json.loads(Path(result['controller']['path']).read_text())
    assert config['manage_raw_bootstrap'] is False
    assert config['command'][-3:]==['{out}','--device','cuda']
    assert json.loads(Path(config['curriculum']).read_text())['snapshot_preparation'] is True
    relay=json.loads(Path(result['relay']['path']).read_text())
    for name,host in relay['hosts'].items():
        assert host['raw_state']==states[name]
        assert relay['provider']['sandboxes'][host['source_host_id']]['role_controls']['sha256']==controls[name]['sha256']
    assert before=={name:Path(ref['path']).read_bytes()for name,ref in plans.items()}
    with pytest.raises(FileExistsError):
        prepare_role_scheduler(plans,controls,curriculum,provider,tmp_path/'prepared',raw_states=states,runtime=runtime,
        rl_bootstrap=binding(tmp_path/'rl-bootstrap.json'),handoff_lease_seconds=180)


def test_role_scheduler_supports_one_explicit_local_source_and_one_existing_cloud_owner(tmp_path):
    plans,controls,curriculum,provider,states,runtime=source(tmp_path)
    provider_value=json.loads(provider.read_text());provider_value['sandboxes'].pop('lfm-existing')
    provider_value['local_source_host']={'source_host_id':'lfm-local','runtime':str(runtime),
        'python':sys.executable,'read_roots':[str(tmp_path)],'write_roots':[str(tmp_path)]}
    provider_value['coordinator_id']='mellum-existing';write(provider,provider_value)
    for role,reference in list(plans.items()):
        path=Path(reference['path']);plan=json.loads(path.read_text())
        plan['role_exchange']['roles']['lfm']['source_host_id']='lfm-local'
        write(path,plan);plans[role]=binding(path)
    result=prepare_role_scheduler(plans,controls,curriculum,provider,tmp_path/'mixed-host-prepared',
        raw_states=states,runtime=runtime,rl_bootstrap=binding(tmp_path/'rl-bootstrap.json'),
        handoff_lease_seconds=180)
    relay=json.loads(Path(result['relay']['path']).read_text())
    assert relay['hosts']['lfm']['source_host_id']=='lfm-local'
    local_controls=relay['provider']['local_source_host']['role_controls']
    assert json.loads(Path(local_controls['path']).read_text())['model']=='lfm'
    cloud_controls=relay['provider']['sandboxes']['mellum-existing']['role_controls']
    assert json.loads(Path(cloud_controls['path']).read_text())['model']=='mellum'


@pytest.mark.parametrize('fault',['peer-authority','cohort','old-readiness','extra-owner','provider-runtime','controls-role'])
def test_source_ownership_and_exact_contract_must_be_coherent(tmp_path,fault):
    plans,controls,curriculum,provider,states,runtime=source(tmp_path)
    if fault in ('peer-authority','cohort','old-readiness','extra-owner'):
        p=Path(plans['lfm']['path']);v=json.loads(p.read_text())
        if fault=='peer-authority':v['role_exchange']['roles']['mellum']['verification_mechanism'][roles.MECHANISM_FILES[0]]='c'*64
        elif fault=='cohort':v['models'][0]['handoff_requirement']['foundation_epochs']=2
        elif fault=='old-readiness':v['models'][0].pop('foundation_readiness_policy')
        else:v['owned_models']=['lfm','mellum']
        write(p,v);plans['lfm']=binding(p)
    elif fault=='provider-runtime':
        v=json.loads(provider.read_text());v['sandboxes']['lfm-existing']['runtime']='other';write(provider,v)
    else:
        p=Path(controls['lfm']['path']);v=json.loads(p.read_text());v['model']='mellum';write(p,v);controls['lfm']=binding(p)
    with pytest.raises(ValueError):
        prepare_role_scheduler(plans,controls,curriculum,provider,tmp_path/'bad',raw_states=states,runtime=runtime,
        rl_bootstrap=binding(tmp_path/'rl-bootstrap.json'),handoff_lease_seconds=180)
    assert not (tmp_path/'bad').exists()


def prepared_snapshot(tmp_path):
    """Receipt-format fixture, explicitly no numerical/model qualification."""
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    plans,controls,curriculum,provider,states,runtime=source(tmp_path)
    result=prepare_role_scheduler(plans,controls,curriculum,provider,tmp_path/'prepared',
        raw_states=states,runtime=runtime,rl_bootstrap=binding(tmp_path/'rl-bootstrap.json'),
        handoff_lease_seconds=180)
    config=json.loads(Path(result['controller']['path']).read_text())
    directory=Path(config['semantic_out']);directory.mkdir(parents=True)
    models=[{'model_id':name+':base','native_identity':{'checkpoint_sha256':('c' if name=='lfm' else 'd')*64,
        'tokenizer_fingerprint':('e' if name=='lfm' else 'f')*64,
        'embedding_weight_sha256':('1' if name=='lfm' else '2')*64,'architecture':name}}
        for name in plans]
    manifest={'models':models,'signature':canonical_json_sha256_hex(models)}
    write(directory/'participants.json',manifest)
    checkpoint=directory/'checkpoint.json';write(checkpoint,{'fixture':'no model/no readiness'})
    snapshot=json.loads(Path(config['curriculum']).read_text())
    exported=directory/'completed-curriculum.json';write(exported,snapshot)
    resolved=copy.deepcopy(snapshot);resolved['snapshot_preparation']=False
    resolved_path=directory/'resolved-curriculum.json'
    from natlang_neuralese.eval.native_state import task_execution_receipt
    from natlang_neuralese.eval.task_cadence import task_eval_policy_identity,task_eval_runtime_identity
    from natlang_neuralese.eval.task_packet import TASK_OUTCOMES
    baselines={}
    for row in models:
        model_id=row['model_id'];identity=row['native_identity']
        spec=snapshot['models'][model_id]['task_eval'];case_indices=spec['cases']
        run=directory/('baseline-'+model_id.replace(':','-'));run.mkdir()
        counts={outcome:0 for outcome in TASK_OUTCOMES};counts['invalid_action']=len(case_indices)
        score={'schema':'natlang.in-run-task-eval/1','step':0,'run':str(run),'cases':len(case_indices),
            'case_indices':case_indices,'packet_sha256':spec['packet']['sha256'],
            'outcome_counts':counts,'complete':True,'eligible':True,'task_successes':0,
            'cases_detail':[{'id':f'{model_id}:{index}','outcome':'invalid_action'}for index in case_indices]}
        score_path=run/'task-score.json';write(score_path,score)
        receipt=task_execution_receipt({key:value for key,value in score.items()if key!='cases_detail'},identity,
            checkpoint_sha256=identity['checkpoint_sha256'])
        policy_sha=task_eval_policy_identity(spec)
        runtime_identity=task_eval_runtime_identity(spec)
        receipt.update(source_baseline=True,baseline_comparison_floor=0,task_eval_policy_sha256=policy_sha,
            task_runtime_identity=runtime_identity)
        receipt_path=run/'native-task-execution-receipt.json';write(receipt_path,receipt)
        baseline={'native_identity':identity,'task_eval_policy_sha256':policy_sha,
            'task_runtime_identity':runtime_identity,'task_successes':0,
            'task_execution_receipt':binding(receipt_path),'score':binding(score_path)}
        baselines[model_id]=baseline
        resolved_spec=resolved['models'][model_id]['task_eval']
        resolved_spec['reference_successes']=0
        resolved_spec['reference_provenance']={'kind':'native-source-task-baseline/1',
            'source_identity':identity,'task_eval_policy_sha256':policy_sha,
            'task_execution_receipt':baseline['task_execution_receipt'],
            'task_runtime_identity':runtime_identity,'score':baseline['score'],
            'superseded_declared_reference':{'reference_successes':spec.get('reference_successes'),
                'reference_provenance':spec.get('reference_provenance')}}
    write(resolved_path,resolved)
    completion_task_baselines=baselines
    adapter=directory/'semantic-adapter.json';write(adapter,{'schema':'natlang.native-semantic-adapter/1',
        'checkpoint':binding(checkpoint)})
    completion=directory/'shared-stage-completion.json';write(completion,{
        'schema':'natlang.native-shared-stage-completion/1','completed':False,'production_ready':False,
        'preparation_complete':True,'snapshot_preparation':{'participant_optimizer_updates':0,
            'participants':{row['model_id']:row['native_identity']for row in models}},
        'participants_signature':manifest['signature'],'curriculum_signature':canonical_json_sha256_hex(snapshot),
        'checkpoint':binding(checkpoint),'curriculum':binding(exported),'adapter':binding(adapter),
        'source_task_baselines':completion_task_baselines,'resolved_curriculum':binding(resolved_path)})
    return config,completion,plans


def test_preparation_requests_real_descendants_then_uses_existing_adoption_resume(tmp_path,monkeypatch):
    from natlang_neuralese.train.native_readiness import snapshot_successor
    config,completion,plans=prepared_snapshot(tmp_path)
    monkeypatch.setattr(roles,'mechanism',lambda:{key:'a'*64 for key in roles.MECHANISM_FILES})
    monkeypatch.setattr(roles,'ready_participants',lambda *a,**kw:([],[{'reason':'SGD still owned'}]))
    waiting=snapshot_successor(config,Path(config['out']))
    assert waiting['awaiting_descendant_handoff']
    assert waiting['command']==config['command'] and 'rl_bootstrap'not in waiting
    plan=json.loads(Path(config['raw_rl_bootstrap']['path']).read_text())
    request=json.loads(Path(waiting['awaiting_descendant_handoff']['path']).read_text())
    assert roles.validate_handoff_request(plan,request)==request
    handoff_curriculum=roles._evidence(request['joint_curriculum'])
    assert handoff_curriculum['models']['lfm:base']['task_eval']['reference_successes']==0
    assert handoff_curriculum['models']['mellum:base']['task_eval']['reference_successes']==0
    # The independent role protocol tests validate real releases. This assertion
    # measures exact executable resume plumbing, not a synthetic release claim.
    descendants=[{'model_id':name+':base','source_role_release':binding(Path(ref['path']))}for name,ref in plans.items()]
    modes=[]
    def available(plan,*,snapshot):modes.append(snapshot);return descendants,[]
    monkeypatch.setattr(roles,'ready_participants',available)
    adopted=snapshot_successor(config,Path(config['out']))
    assert modes==[False] and adopted['snapshot_adoption_controller']
    assert adopted['command'][-3:]==['--resume',str(completion.parent/'checkpoint.json'),'--refresh-participants']
    assert adopted['semantic_out']==str(Path(config['out'])/'joint')
    assert adopted['rl_bootstrap']==config['snapshot_successor']['rl_bootstrap']
    full=json.loads(Path(adopted['curriculum']).read_text())
    assert full['snapshot_preparation'] is False
    assert full['models']['lfm:base']['task_eval']['reference_successes']==0
    assert full['snapshot_adoption']['releases']==[row['source_role_release']for row in descendants]
    assert full['phases']==json.loads(Path(config['snapshot_successor']['curriculum']['path']).read_text())['phases']
    assert completion.is_file()  # No overwrite/adoption of historical optimizer state.


@pytest.mark.parametrize('fault',['contract','consumer','mechanism','joint-work','preparation','curriculum-binding'])
def test_handoff_request_rejects_false_or_misattributed_work(tmp_path,monkeypatch,fault):
    from natlang_neuralese.train.native_readiness import snapshot_successor
    config,completion,plans=prepared_snapshot(tmp_path)
    monkeypatch.setattr(roles,'mechanism',lambda:{key:'a'*64 for key in roles.MECHANISM_FILES})
    monkeypatch.setattr(roles,'ready_participants',lambda *a,**kw:([],[]))
    waiting=snapshot_successor(config,Path(config['out']))
    plan=json.loads(Path(config['raw_rl_bootstrap']['path']).read_text())
    request=json.loads(Path(waiting['awaiting_descendant_handoff']['path']).read_text())
    if fault=='contract':request['contract']['models']['lfm']['foundation_epochs']=0
    elif fault=='consumer':request['role']='lfm'
    elif fault=='mechanism':request['mechanism'][roles.MECHANISM_FILES[0]]='b'*64
    elif fault=='joint-work':
        content=roles._evidence(request['joint_curriculum']);content.pop('task_eval')
        path=tmp_path/'bad-joint.json';write(path,content)
        request['joint_curriculum']=roles._inline(binding(path));request['joint_controller']['curriculum']=binding(path)
    elif fault=='preparation':
        content=roles._evidence(request['preparation']);content['snapshot_preparation']['participant_optimizer_updates']=1
        path=tmp_path/'bad-preparation.json';write(path,content);request['preparation']=roles._inline(binding(path))
    else:request['joint_controller']['curriculum']['sha256']='b'*64
    with pytest.raises(ValueError):roles.validate_handoff_request(plan,request)


def test_two_provider_handoff_lease_expiry_cancels_hold_without_any_training_credit(tmp_path,monkeypatch):
    import time
    from natlang_neuralese.train.native_readiness import snapshot_successor
    from natlang_neuralese.train.native_resident import process_owner
    config,completion,plans=prepared_snapshot(tmp_path)
    monkeypatch.setattr(roles,'mechanism',lambda:{key:'a'*64 for key in roles.MECHANISM_FILES})
    monkeypatch.setattr(roles,'ready_participants',lambda *a,**kw:([],[]))
    waiting=snapshot_successor(config,Path(config['out']))
    plan=json.loads(Path(config['raw_rl_bootstrap']['path']).read_text())
    request=json.loads(Path(waiting['awaiting_descendant_handoff']['path']).read_text())
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    offers={name:{'role':name,'contract':roles.contract(plan),'mechanism':roles.mechanism(),
        'owner':process_owner(),'released':False,'request_sha256':canonical_json_sha256_hex(request),
        'checkpoints':{name:binding(Path(plans[name]['path']))},
        'qualification_generations':{name:'repair'},'provider_inputs':{name:{}}}for name in plans}
    attempt=attempt_for(plan,request,offers)
    roles.renew_handoff(plan,request,offers,expires_at=time.time()+180,attempt=attempt)
    assert roles.handoff_active(plan)
    assert not roles.handoff_hold(plan)  # Arm/lease is NOT actual current qualification.
    with pytest.raises(ValueError):roles.renew_handoff(plan,request,{'mellum':offers['mellum']},expires_at=time.time()+180,attempt=attempt)
    roles.cancel_handoff(plan,request)
    assert roles.handoff_active(plan)is None and not roles.handoff_hold(plan)
    # Baseline evidence is revalidated on every renewal; allow that exact
    # read-only verification to finish before testing expiry.
    roles.renew_handoff(plan,request,offers,expires_at=time.time()+60,attempt=attempt)
    later=time.time()+300
    monkeypatch.setattr(roles.time,'time',lambda:later)
    assert roles.handoff_active(plan)is None and not roles.handoff_hold(plan)


def test_existing_controller_automatically_advances_preparation_to_released_descendant_work(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_readiness
    config,completion,plans=prepared_snapshot(tmp_path)
    monkeypatch.setattr(roles,'mechanism',lambda:{key:'a'*64 for key in roles.MECHANISM_FILES})
    monkeypatch.setattr(roles,'ready_participants',lambda *a,**kw:([],[]))
    path=tmp_path/'controller-live.json';write(path,config)
    waiting=native_readiness.run(path,once=True)
    assert waiting['ready_models']==0 and waiting['child_pid']is None
    assert any('descendant' in row['reason']for row in waiting['pending'])
    descendants=[{'model_id':name+':base','source_role_release':ref}for name,ref in plans.items()]
    calls=[]
    def released(plan,*,snapshot):calls.append(snapshot);return descendants,[]
    monkeypatch.setattr(roles,'ready_participants',released)
    monkeypatch.setattr(roles,'local_released',lambda config:True)
    active=native_readiness.run(path,once=True)
    assert active['ready_models']==2 and active['child_pid']is None and calls==[False,False]
    assert Path(active['participants']).parent==Path(config['out'])/'joint'
    participants=json.loads(Path(active['participants']).read_text())
    assert participants['models']==descendants
    assert completion.is_file()  # Actual work is launched only outside inspection mode.


def test_actual_relay_preflights_joint_assets_before_it_can_arm_any_owner(tmp_path,monkeypatch):
    from test_native_role_release import _service,FileProtocolIO
    from natlang_neuralese.train.native_readiness import snapshot_successor
    from natlang_neuralese.train.native_resident import process_owner
    config,_,plans=prepared_snapshot(tmp_path)
    monkeypatch.setattr(roles,'mechanism',lambda:{key:'a'*64 for key in roles.MECHANISM_FILES})
    with monkeypatch.context() as local:
        local.setattr(roles,'ready_participants',lambda *a,**kw:([],[]))
        waiting=snapshot_successor(config,Path(config['out']))
    relay=json.loads((Path(config['out']).parent/'relay.json').read_text())
    for host in relay['hosts'].values():
        path=Path(host['raw_state']);path.parent.mkdir(parents=True)
        write(path,{'owner':process_owner(),'released':False,'models':{}})
    io=FileProtocolIO(plans);io.config=relay['provider']
    from natlang_neuralese.common.hashing import sha256_file_hex
    collector_assets={}
    for role,host in relay['hosts'].items():
        runtime=Path(relay['provider']['sandboxes'][host['source_host_id']]['runtime'])
        collector=runtime/'scripts/run_bonsai_queue.py'
        collector.parent.mkdir(parents=True,exist_ok=True)
        collector.write_text('# fixture collector bound to the prepared runtime\n')
        collector_assets[role]={'kind':'task-source-file','path':str(collector),
            'bytes':collector.stat().st_size,'sha256':sha256_file_hex(collector)}
    relay['task_eval_collector_assets']=collector_assets
    checked=[]
    def preflight(sid,request):checked.append((sid,request));return {'files':[],'assets':[]}
    monkeypatch.setattr(io,'dependency_closure',preflight)
    module=_service(monkeypatch)
    result=module.tick(relay,io)
    request=json.loads(Path(waiting['awaiting_descendant_handoff']['path']).read_text())
    assert checked==[('mellum-existing',{'semantic_config':request['joint_controller']['curriculum']})]
    assert result['status']=='waiting-for-semantic-consumption'
    for reference in plans.values():
        plan=json.loads(Path(reference['path']).read_text())
        assert not roles.handoff_active(plan)  # Empty candidates are not ready.
    def missing(sid,request):raise ValueError('required joint asset missing')
    monkeypatch.setattr(io,'dependency_closure',missing)
    with pytest.raises(ValueError,match='joint asset missing'):module.tick(relay,io)
    assert all(not (Path(json.loads(Path(ref['path']).read_text())['role_exchange']['directory'])/
        'joint-handoff-live.json').exists()for ref in plans.values())


def test_joint_task_specs_preserve_distinct_declared_floors_and_reject_global_substitution(tmp_path):
    from natlang_neuralese.train.native_deployment import validate_joint_task_specs
    plans,controls,path,provider,states,runtime=source(tmp_path)
    value=json.loads(path.read_text());before=copy.deepcopy(value)
    assert validate_joint_task_specs(value,['lfm:base','mellum:base'])==before
    assert value['models']['lfm:base']['task_eval']['reference_successes']==0
    assert value['models']['mellum:base']['task_eval']['reference_successes']==1
    value['models']['lfm:base'].pop('task_eval')
    value['task_eval']=before['models']['mellum:base']['task_eval']
    with pytest.raises(ValueError,match='explicit participant'):validate_joint_task_specs(value,['lfm:base','mellum:base'])
    value=copy.deepcopy(before);value['models']['mellum:base']['task_eval']['reference_successes']=None
    with pytest.raises(ValueError,match='undeclared'):validate_joint_task_specs(value,['lfm:base','mellum:base'])


def test_prepare_only_task_policy_allows_only_explicit_unmeasured_baseline(tmp_path):
    from natlang_neuralese.train.native_deployment import validate_joint_task_specs
    _,_,path,_,_,_=source(tmp_path);value=json.loads(path.read_text())
    value['models']['mellum:base']['task_eval'].update(reference_successes=None,min_successes=0)
    assert validate_joint_task_specs(value,['lfm:base','mellum:base'],
        allow_pending_baseline=True)==value
    with pytest.raises(ValueError,match='undeclared'):
        validate_joint_task_specs(value,['lfm:base','mellum:base'])
    del value['models']['mellum:base']['task_eval']['min_successes']
    with pytest.raises(ValueError,match='undeclared'):
        validate_joint_task_specs(value,['lfm:base','mellum:base'],allow_pending_baseline=True)


def test_preparation_relay_checks_complete_semantic_closure_before_requesting_consumer_release(tmp_path,monkeypatch):
    from test_native_role_release import _service
    module=_service()
    calls=[]
    class IO:
        broken=True
        def dependency_closure(self,sid,value):
            calls.append(('closure',sid,value));return {'assets':['both-model-bases-and-tokenizers']}
        def verify_assets(self,sid,assets):
            calls.append(('assets',sid,assets))
            if self.broken:raise ValueError('missing actual peer base')
    io=IO();monkeypatch.setattr(module,'call',lambda *args:calls.append(('arm',args[-1])))
    config={'preparation_curriculum':{'path':str(tmp_path/'frozen.json'),'sha256':'a'*64}}
    plan={'consumer_preparation_handoff':True};request={'snapshot_preparation':True}
    with pytest.raises(ValueError,match='missing actual peer base'):
        module._arm_preparation(config,io,'existing-consumer',plan,request)
    assert [row[0]for row in calls]==['closure','assets']
    io.broken=False;calls.clear()
    module._arm_preparation(config,io,'existing-consumer',plan,request)
    assert [row[0]for row in calls]==['closure','assets','arm']


def test_explicit_new_work_builder_resumes_actual_bound_frozen_checkpoint(tmp_path):
    config,completion,_=prepared_snapshot(tmp_path/'prior')
    plans,controls,curriculum,provider,states,runtime=source(tmp_path/'next')
    adjusted=json.loads(curriculum.read_text());adjusted['snapshot_work_generation']='next-adaptation'
    adjusted['adaptive_refresh']={'updates_per_phase':1}
    write(curriculum,adjusted)
    result=prepare_role_scheduler(plans,controls,curriculum,provider,tmp_path/'next-prepared',raw_states=states,
        runtime=runtime,rl_bootstrap=binding(tmp_path/'next/rl-bootstrap.json'),handoff_lease_seconds=180,
        snapshot_parent=binding(completion))
    successor=json.loads(Path(result['controller']['path']).read_text())
    proof=json.loads(completion.read_text())
    command=successor['command']
    assert command[command.index('--resume')+1]==proof['checkpoint']['path']
    assert '--refresh-participants'in command
    assert result['inputs']['snapshot_parent']==binding(completion)
    assert result['model_allocations']==result['checkpoint_decodes']==0


def test_snapshot_owns_full_vae_epoch_and_keeps_later_adaptive_quanta(tmp_path):
    plans,controls,curriculum,provider,states,runtime=source(tmp_path)
    original=json.loads(curriculum.read_text());original['adaptive_refresh']={'updates_per_phase':16}
    original['checkpoint_seconds']=3600.
    write(curriculum,original)
    result=prepare_role_scheduler(plans,controls,curriculum,provider,tmp_path/'prepared',
        raw_states=states,runtime=runtime,rl_bootstrap=binding(tmp_path/'rl-bootstrap.json'),handoff_lease_seconds=180)
    controller=json.loads(Path(result['controller']['path']).read_text())
    snapshot=json.loads(Path(controller['curriculum']).read_text())
    full=json.loads(Path(controller['snapshot_successor']['curriculum']['path']).read_text())
    assert 'adaptive_refresh'not in snapshot
    assert full==original and full['adaptive_refresh']['updates_per_phase']==16
    assert snapshot=={**{k:v for k,v in full.items()if k!='adaptive_refresh'},
        'snapshot_preparation':True,'inherit_optimizer':True}
    from natlang_neuralese.train import native_deployment
    assert result['operational_inputs']==[binding(native_deployment.__file__)]
