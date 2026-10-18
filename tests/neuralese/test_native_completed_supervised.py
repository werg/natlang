"""Real serialized optimizer/RNG and sealed bridge protocol; no quality claim."""
import copy,json,os
from pathlib import Path
import pytest
import torch
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_completed_supervised as adapt,native_bridge as bridge
from natlang_neuralese.train.recipe import native_recipe_binding
from natlang_neuralese.train.loop import capture_training_rng_state
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from test_native_state_sft import graph


def fixture(tmp_path):
    root=tmp_path/'rounds'/'lfm';root.mkdir(parents=True);train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    train.write_text(json.dumps(graph('old'))+'\n');h=graph('held');h['split']='test';held.write_text(json.dumps(h)+'\n')
    recipe={'schema':'natlang.neuralese-training-recipe/1','id':'lineage','identity_qualification':'native_state_runtime_qualified',
        'input_bindings':{'train':adapt.binding(train),'held':adapt.binding(held)},
        'stages':[{'id':name,'kind':'native_state_sft','requires':[],
            'inputs':{'native_graphs':'train','held_native_graphs':'held'},
            'parameters':{'epochs':1,'learning_rate':3e-6,'checkpoint_every':2,'eval_every':1}}
            for name in('dpo','full')]}
    recipe_path=tmp_path/'recipe.json';write(recipe_path,recipe);rb=native_recipe_binding(recipe_path,'e'*64)
    parent=tmp_path/'parent.pt';parent.write_bytes(b'actual protocol parent')
    config={'native_state':True,'native_actor_mode':'raw-state','out':str(root/'production'),
        'supervised_stage_quantum_updates':{'dpo':2,'full':1000},'checkpoint':str(parent),'production_recipe':str(recipe_path),'production_stage':'full',
        'raw_model_id':'fixture:'+('a'*64),'recipe_lineage_sha256':'e'*64,
        'bootstrap_binding':{'handoff':{'path':'historical-bootstrap','sha256':'b'*64}},
        'seed_command':['python','-m','natlang_neuralese.train.native_state_sft','{seed_trainer_args}'],
        'trainer_parameters':{'recipe_binding':rb},'seed_trainer_parameters':{'recipe_binding':rb}}
    config_path=root/'config.json';write(config_path,config)
    (root/'production').mkdir();write(root/'production/state.json',{'phase':'collect','checkpoint':str(parent),'round':2,'trained_rounds':2})
    write(root/'cycle.json',{'phase':'sft','cycle':3,'next_stage':0,'current_checkpoint':adapt.binding(parent)})
    job=root/'cycles/000003/supervised/000';job.mkdir(parents=True)
    continuation={**config,'mandatory_sft_only':True,'production_stage':'dpo','out':str(job),
        'trainer_parameters':{'stage_id':'dpo'}};write(job/'config.json',continuation)
    request=bridge.write_request(job/'bridge',kind='round',out=job,config=job/'config.json')
    learner=job/'learner';learner.mkdir();checkpoint=learner/'checkpoint.pt'
    parameter=torch.nn.Parameter(torch.tensor([.3,.7]));optimizer=torch.optim.AdamW([parameter],lr=.01)
    for _ in range(2):optimizer.zero_grad();parameter.square().sum().backward();optimizer.step()
    training={'schema':TRAINING_SCHEMA,'global_step':4,'step':2,'optimizer':optimizer.state_dict(),
        'rng':capture_training_rng_state('cpu'),'options':{'stage_id':'dpo','recipe_binding':json.dumps(rb),
            'continuation':{'checkpoint_sha256':bridge.binding(parent)['sha256']}},
        'source_row_bindings':{'old':'exact-binding'},'source_row_order':['old'],'exposure_counts':{'old':2},
        'identity':{'objective':None},'sft_progress':{'full':{'options':{'stage_id':'full'},'exposure_counts':{'old':1}}}}
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'fixture','base_inventory_sha256':'a'*64},
        'training':training,'model':{'weight':parameter.detach()}},checkpoint)
    cp=bridge.binding(checkpoint);boundary={'kind':'semantic-refresh','reached':True,'new_optimizer_updates':2,'checkpoint_sha256':cp['sha256']}
    receipt={'schema':'natlang.native-training-readiness/1','stage_id':'dpo','checkpoint':cp,
        'continuation':{'checkpoint_sha256':bridge.binding(parent)['sha256']},'continuation_boundary':boundary,
        'mandatory_sft':{'stages':{'dpo':{'graph_visits':2},'full':{'graph_visits':1}}}}
    write(learner/'native-training-receipt.json',receipt)
    write(job/'state.json',{'checkpoint':str(parent),'phase':'awaiting_semantic_refresh','pending_native_checkpoint':str(checkpoint)})
    req=bridge.read_request(request);result=request.with_suffix('.result.json')
    write(result,{'schema':bridge.RESULT_SCHEMA,'kind':'round','request_id':req['request_id'],
        'request_sha256':bridge.binding(request)['sha256'],'status':'completed','state':bridge.binding(job/'state.json'),
        'checkpoint':cp,'artifacts':bridge.inventory(job,exclude=[result])})
    plan=tmp_path/'global.json';write(plan,{'models':[{'model':'lfm','production_recipe':adapt.binding(recipe_path),
        'handoff_requirement':{'model_id':config['raw_model_id'],'recipe_sha256':rb['sha256'],
            'recipe_lineage_sha256':rb['lineage_sha256'],'stage_ids':['full']}}]})
    config['runtime_binding']={'plan_sha256':adapt.binding(plan)['sha256'],'handoff':config['bootstrap_binding']['handoff']};config['supervised_stage_quantum_control']={'plan':adapt.binding(plan),'updates':config['supervised_stage_quantum_updates']};write(config_path,config)
    controller={'pid':99999999,'start_ticks':'0','boot_id':'gone'}
    return root,config_path,request,result,controller,plan,recipe_path


def test_completed_stage_accounted_once_and_same_decode_reused(tmp_path,monkeypatch):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path)
    original=torch.load;loads=[]
    def observed(*args,**kwargs):loads.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',observed)
    receipt,pair=adapt.consume(config,bridge.binding(request),bridge.binding(result),tmp_path/'consumed.json',controller=controller)
    assert len(loads)==1 and receipt['new_optimizer_updates']==2 and pair[1]['training']['global_step']==4
    assert json.loads((root/'cycle.json').read_text())['next_stage']==1
    assert json.loads((root/'production/state.json').read_text())['trained_rounds']==2
    before=bridge.binding(pair[0]['path']);moments=copy.deepcopy(pair[1]['training']['optimizer'])
    package=tmp_path/'package';package.mkdir();(package/'identity.py').write_text('VALUE=1\n')
    new=json.loads(recipe.read_text());new['stages'][1]['parameters']['learning_rate']=1e-5
    future=tmp_path/'future.json';write(future,new)
    proof=adapt.prepare(config,plan,'lfm',adapt.binding(future),adapt.binding(tmp_path/'consumed.json'),pair,tmp_path/'adaptation/proof.json',package=package)
    assert len(loads)==1 and bridge.binding(pair[0]['path'])==before
    actual=adapt.verify(proof);assert actual['new_sources_earned_visits']==0
    assert pair[1]['training']['optimizer']['param_groups']==moments['param_groups']
    # Exact consumption retry does not double-advance the cursor.
    adapt.consume(config,bridge.binding(request),bridge.binding(result),tmp_path/'consumed.json',controller=controller)
    assert json.loads((root/'cycle.json').read_text())['next_stage']==1


@pytest.mark.parametrize('mutation',['wrong-parent','wrong-stage','not-completed','live-controller'])
def test_bad_completed_authority_does_not_change_controls(tmp_path,mutation):
    root,config,request,result,controller,*_=fixture(tmp_path)
    if mutation=='wrong-parent':
        p=root/'cycle.json';x=json.loads(p.read_text());x['current_checkpoint']['sha256']='f'*64;write(p,x)
    if mutation=='wrong-stage':
        p=root/'cycle.json';x=json.loads(p.read_text());x['next_stage']=1;write(p,x)
    if mutation=='not-completed':
        x=json.loads(result.read_text());x['status']='failed';write(result,x)
    if mutation=='live-controller':
        from natlang_neuralese.rl.native_raw_handoff import observe
        controller=observe(os.getpid())
    before=[p.read_bytes()for p in (config,root/'cycle.json',root/'production/state.json')]
    with pytest.raises(ValueError):adapt.consume(config,bridge.binding(request),bridge.binding(result),tmp_path/'consumed.json',controller=controller)
    assert before==[p.read_bytes()for p in (config,root/'cycle.json',root/'production/state.json')]


def prepared(tmp_path,new_lr=1e-5,old_declaration_overrides=None,runtime_package=None,bootstrap_handoff=None):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path)
    if bootstrap_handoff:
        configured=json.loads(config.read_text());configured['bootstrap_binding']['handoff']=bootstrap_handoff
        configured['runtime_binding']['handoff']=bootstrap_handoff;write(config,configured)
    if old_declaration_overrides:
        global_plan=json.loads(plan.read_text());global_plan['models'][0].update(old_declaration_overrides);write(plan,global_plan)
        old_config=json.loads(config.read_text());old_config['runtime_binding']['plan_sha256']=adapt.binding(plan)['sha256']
        old_config['supervised_stage_quantum_control']['plan']=adapt.binding(plan)
        if 'runtime_handoff'in old_declaration_overrides:old_config['runtime_handoff']=old_declaration_overrides['runtime_handoff']
        write(config,old_config)
    consumption=tmp_path/'consumed.json'
    _,pair=adapt.consume(config,bridge.binding(request),bridge.binding(result),consumption,controller=controller)
    if runtime_package is None:
        package=tmp_path/'package';package.mkdir();(package/'identity.py').write_text('VALUE=1\n')
    else:package=Path(runtime_package)
    new=json.loads(recipe.read_text());new['stages'][1]['parameters']['learning_rate']=new_lr
    future=tmp_path/'future.json';write(future,new)
    proof=adapt.prepare(config,plan,'lfm',adapt.binding(future),adapt.binding(consumption),pair,tmp_path/'adaptation/proof.json',package=package)
    return root,config,plan,proof


@pytest.mark.parametrize('new_lr',[1e-5,1e-6])
def test_configuration_changes_future_lr_only_and_does_not_repeat_completed_stage(tmp_path,monkeypatch,new_lr):
    root,config,plan,proof=prepared(tmp_path,new_lr)
    from natlang_neuralese.rl import native_bootstrap_metadata
    monkeypatch.setattr(native_bootstrap_metadata,'save',lambda *args:None)
    value=adapt.verify(proof);p=json.loads(plan.read_text());p['models'][0].update(
        production_recipe=value['recipe'],completed_supervised_adaptation=proof);write(plan,p)
    before=(root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes()
    old=json.loads(config.read_text())
    adapt.configure(plan,p['models'][0],old['bootstrap_binding']['handoff'],config)
    future=json.loads(config.read_text())
    assert future['seed_trainer_parameters']['resident_data_directory']==str(Path(value['recipe']['path']).parent/'resident-data')
    assert future['resident_data_directory']==future['seed_trainer_parameters']['resident_data_directory']
    assert future['seed_trainer_parameters']['learning_rate']==new_lr
    assert future['supervised_stage_quantum_control']['plan']==adapt.binding(plan)
    assert before==((root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes())
    assert json.loads((root/'cycle.json').read_text())['next_stage']==1
    adapt.configure(plan,p['models'][0],old['bootstrap_binding']['handoff'],config)


@pytest.mark.parametrize('mutation',['future-command','started-stage','future-credit','other-role'])
def test_adaptation_tampering_rejects_without_changing_current_controls(tmp_path,mutation,monkeypatch):
    root,config,plan,proof=prepared(tmp_path)
    value=adapt.verify(proof);before=(config.read_bytes(),(root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes())
    if mutation=='future-command':
        path=Path(value['future_config']['path']);x=json.loads(path.read_text());x['seed_command']=['unreviewed'];write(path,x);value['future_config']=adapt.binding(path)
    elif mutation=='started-stage':value['started_stage_ids']=['dpo']
    elif mutation=='future-credit':value['new_sources_earned_visits']=1
    else:
        p=json.loads(plan.read_text());p['models'][0].update(production_recipe=value['recipe'],completed_supervised_adaptation=proof);p['unexpected_foreign_change']=True;write(plan,p)
        with pytest.raises(ValueError,match='global plan'):adapt.configure(plan,p['models'][0],{'path':'historical-bootstrap','sha256':'b'*64},config)
        return
    path=Path(proof['path']);write(path,value)
    with pytest.raises(ValueError):adapt.verify(adapt.binding(path))
    assert before==(config.read_bytes(),(root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes())


def test_interrupted_publication_recovers_exact_candidate_without_double_credit(tmp_path,monkeypatch):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path)
    original=adapt.write_bytes_durable;failed=[]
    def interrupted(path,data):
        if Path(path)==root/'cycle.json'and not failed:failed.append(True);raise OSError('actual interruption after state publication')
        return original(path,data)
    monkeypatch.setattr(adapt,'write_bytes_durable',interrupted)
    with pytest.raises(OSError):adapt.consume(config,bridge.binding(request),bridge.binding(result),tmp_path/'consumed.json',controller=controller)
    assert json.loads((root/'cycle.json').read_text())['next_stage']==0
    monkeypatch.setattr(adapt,'write_bytes_durable',original)
    receipt,_=adapt.consume(config,bridge.binding(request),bridge.binding(result),tmp_path/'consumed.json',controller=controller)
    assert receipt['new_optimizer_updates']==2 and json.loads((root/'cycle.json').read_text())['next_stage']==1


def test_wrong_declared_quantum_stops_before_publication(tmp_path):
    root,config,request,result,controller,*_=fixture(tmp_path)
    value=json.loads(config.read_text());value['supervised_stage_quantum_updates']['dpo']=250;write(config,value)
    before=(root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes()
    with pytest.raises(ValueError,match='finite update quantum'):
        adapt.consume(config,bridge.binding(request),bridge.binding(result),tmp_path/'consumed.json',controller=controller)
    assert before==((root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes())


def test_config_written_before_metadata_save_is_recoverable(tmp_path,monkeypatch):
    root,config,plan,proof=prepared(tmp_path)
    from natlang_neuralese.rl import native_bootstrap_metadata
    saved=[]
    def fail_once(*args):
        saved.append(True)
        if len(saved)==1:raise OSError('metadata write interrupted')
    monkeypatch.setattr(native_bootstrap_metadata,'save',fail_once)
    value=adapt.verify(proof);p=json.loads(plan.read_text());p['models'][0].update(production_recipe=value['recipe'],completed_supervised_adaptation=proof);write(plan,p)
    original=json.loads(config.read_text());bootstrap=original['bootstrap_binding']['handoff']
    with pytest.raises(OSError):adapt.configure(plan,p['models'][0],bootstrap,config)
    applied=config.read_bytes();adapt.configure(plan,p['models'][0],bootstrap,config)
    assert config.read_bytes()==applied and len(saved)==2


def test_prior_closed_withdrawal_requires_exact_proof_and_all_frame_absence(tmp_path):
    path=tmp_path/'prior.json';row={'id':'rejected','binding':'exact','source_sha256':'a'*64}
    write(path,{'schema':'natlang.native-graph-withdrawal/1','rows':[row]});reference=adapt.binding(path)
    old={'recipe':{'stages':[{'id':'full'}]},'stage_inputs':{'full':{'withdrawn_native_graphs':reference}}}
    training={'options':{'withdrawal_history':[{'proof_sha256':reference['sha256'],'rows':[row]}]},'source_row_bindings':{}}
    assert adapt._closed_withdrawals(training,{},old)==[{'proof':reference,'rows':1}]
    with pytest.raises(ValueError,match='reintroduced'):adapt._closed_withdrawals(training,{'full':{'source_row_bindings':{'rejected':'exact'}}},old)
    altered=copy.deepcopy(training);altered['options']['withdrawal_history'][0]['rows'][0]['binding']='wrong'
    with pytest.raises(ValueError,match='exact closed prior'):adapt._closed_withdrawals(altered,{},old)


def release_declaration(proof):
    value=adapt.verify(proof)
    old=json.loads(Path(value['old_global_plan']['path']).read_text())['models'][0]
    return {**old,'production_recipe':value['recipe'],'completed_supervised_adaptation':proof},value


def test_release_uses_verified_future_recipe_without_new_checkpoint_decode_or_readiness(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw
    from natlang_neuralese.train.recipe import recipe_identity_sha256
    _,_,_,proof=prepared(tmp_path)
    declaration,value=release_declaration(proof)
    def no_decode(*args,**kwargs):raise AssertionError('release adaptation metadata must not decode historical tensors')
    monkeypatch.setattr(torch,'load',no_decode)
    requirement=raw._release_requirement(declaration,{})
    assert requirement['recipe_sha256']==recipe_identity_sha256(Path(value['recipe']['path']))
    assert requirement['recipe_sha256']!=declaration['handoff_requirement']['recipe_sha256']
    assert requirement['recipe_lineage_sha256']==value['lineage']
    assert requirement['model_id']==declaration['handoff_requirement']['model_id']
    assert requirement['stage_ids']==['full']
    assert value['new_sources_earned_visits']==0 and value['whole_recipe_completed']is False
    assert 'readiness'not in requirement


@pytest.mark.parametrize('mutation',['role','recipe','runtime','foundation','credit','lineage'])
def test_release_adaptation_cannot_rebind_owned_authority_or_grant_credit(tmp_path,mutation):
    from natlang_neuralese.rl import native_raw_rounds as raw
    _,_,_,proof=prepared(tmp_path);declaration,value=release_declaration(proof)
    if mutation=='role':declaration['model']='mellum'
    elif mutation=='recipe':declaration['production_recipe']=value['old_plan']
    elif mutation=='runtime':declaration['runtime_handoff']={'path':'invented','sha256':'a'*64}
    elif mutation=='foundation':declaration['handoff_requirement']['stage_ids']=[]
    else:
        value['new_sources_earned_visits'if mutation=='credit'else'lineage']=1 if mutation=='credit'else'f'*64
        write(Path(proof['path']),value);declaration['completed_supervised_adaptation']=adapt.binding(proof['path'])
    with pytest.raises((ValueError,FileNotFoundError)):
        raw._release_requirement(declaration,{})


def test_release_preserves_runtime_authority_before_retired_resident_early_return(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw,native_runtime_handoff as runtime
    from natlang_neuralese.train.recipe import recipe_identity_sha256
    historical=tmp_path/'historical-runtime.json';write(historical,{'bootstrap':{'path':'old-bootstrap','sha256':'b'*64}})
    _,_,_,proof=prepared(tmp_path,old_declaration_overrides={'runtime_handoff':adapt.binding(historical)})
    declaration,value=release_declaration(proof)
    retired={'path':'retired-generation','sha256':'c'*64};cursor=tmp_path/'retired-cursor.json';write(cursor,{'resident_current':retired})
    observed=[]
    def verified(reference,*,bootstrap):
        observed.append((reference,bootstrap))
        return {'recipe_handoff':None,'handoff':{'path':'drained','sha256':'d'*64},
            'snapshots':{'cursor':adapt.binding(cursor)}}
    monkeypatch.setattr(runtime,'verify_runtime_handoff',verified)
    requirement=raw._release_requirement(declaration,{'resident_current':retired})
    assert observed==[(adapt.binding(historical),{'path':'old-bootstrap','sha256':'b'*64})]
    assert requirement['recipe_sha256']==recipe_identity_sha256(Path(value['recipe']['path']))
    assert requirement['recipe_lineage_sha256']==value['lineage']
    def rejected(*args,**kwargs):raise ValueError('original runtime authority rejected')
    monkeypatch.setattr(runtime,'verify_runtime_handoff',rejected)
    with pytest.raises(ValueError,match='original runtime authority rejected'):
        raw._release_requirement(declaration,{'resident_current':retired})


def test_release_follows_real_fresh_resident_append_after_completed_adaptation(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.recipe import native_recipe_binding
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    runtime=tmp_path/'ts-host'
    for name in ('dist/teacher/cli.js','prelude.js','package.json','../package.json','../package-lock.json',
                 '../training/source-reviews/holds.jsonl','../training/source-reviews/family-holds.json','../training/source-reviews/datasets.json'):
        path=runtime/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    cases=tmp_path/'cases.jsonl';cases.write_text('{"id":"owned-train-case","split":"train"}\n')
    historical=tmp_path/'historical';historical.mkdir();refs={}
    for name in ('launch','terminal','checkpoint'):
        path=historical/name;path.write_bytes(b'historical fixture artifact');refs[name]=adapt.binding(path)
    write(historical/'intent.json',{'launch':refs['launch']})
    write(historical/'handoff.json',{'intent':adapt.binding(historical/'intent.json'),
        'terminal':refs['terminal'],'checkpoint':refs['checkpoint']})
    handoff=adapt.binding(historical/'handoff.json')
    from natlang_neuralese.data import native_sources
    root,config,global_plan,proof=prepared(tmp_path,runtime_package=Path(native_sources.__file__).parents[1],
        old_declaration_overrides={'runtime':str(runtime),'cases':adapt.binding(cases)},bootstrap_handoff=handoff)
    declaration,value=release_declaration(proof)
    from natlang_neuralese.rl import native_bootstrap_metadata
    from natlang_neuralese.rl import native_raw_handoff
    # Only the historical full handoff proof is outside this serialized
    # completed-operation fixture; the metadata evidence/cache remain real.
    monkeypatch.setattr(native_raw_handoff,'verify_handoff_metadata',lambda *args,**kwargs:None)
    write(global_plan,{'models':[declaration]})
    old=json.loads(config.read_text());before=[p.read_bytes()for p in(root/'cycle.json',root/'production/state.json')]
    def no_second_load(*args,**kwargs):raise AssertionError('resident setup must not decode the checkpoint')
    monkeypatch.setattr(torch,'load',no_second_load)
    adapt.configure(global_plan,declaration,old['bootstrap_binding']['handoff'],config)
    assert (root/'bootstrap-verification.json').is_file()
    actual_config=json.loads(config.read_text());directory=Path(actual_config['resident_data_directory'])
    installed=json.loads((directory.parent/'recipe-plan.json').read_text())
    assert adapt.binding(directory.parent/'recipe-plan.json')['sha256']==value['lineage_handoff']['sha256']
    assert before==[p.read_bytes()for p in(root/'cycle.json',root/'production/state.json')]
    inputs=installed['stage_inputs']['full']
    train=NativeGraphSource(inputs['native_graphs']['path'],'tokenizer-sha',required_split='train')
    held=NativeGraphSource(inputs['held_native_graphs']['path'],'tokenizer-sha',required_split='test')
    evidence=tmp_path/'geometry-evidence.json';write(evidence,{'passed':True,'scope':'tiny CPU fixture'})
    capacity=tmp_path/'capacity.json';write(capacity,{'schema':'natlang.native-resident-geometry-qualification/1',
        'passed':True,'identity':None,'runtime_code_sha256':resident.digest(installed['code']),
        'limits':{'forward_tokens':1000000,'kv_slots':1000000},'evidence':resident.binding(evidence)})
    manager=resident.ResidentAdoption(directory,stage_id='full',
        recipe_binding=native_recipe_binding(value['recipe']['path'],value['lineage']),
        train=train,held=held,package=value['package'],geometry_qualification=capacity)
    appended=tmp_path/'appended.jsonl';appended.write_text(Path(inputs['native_graphs']['path']).read_text()+json.dumps(graph('new'))+'\n')
    recipe=json.loads(Path(value['recipe']['path']).read_text());recipe['input_bindings']['appended']=adapt.binding(appended)
    recipe['stages'][1]['inputs'].update(native_graphs='appended',continued_native_graphs='train')
    new_recipe=tmp_path/'append-recipe.json';write(new_recipe,recipe)
    resident.prepare(directory,new_recipe,'full',tokenizer_fingerprint='tokenizer-sha',role_ids=(),runtime_package=value['package'])
    assert manager.poll(5)is not None
    state=resident.current(directory);assert state['payload']['plan']['recipe_sha256']==value['lineage']
    # Use the actual supervised publisher and real cache invalidation, rather
    # than manually pretending the original bootstrap receipt still matches.
    job=tmp_path/'published-supervised';(job/'learner').mkdir(parents=True)
    checkpoint=job/'learner/checkpoint.pt';checkpoint.write_bytes(Path(value['checkpoint']['path']).read_bytes())
    write(checkpoint.parent/'native-training-receipt.json',{'checkpoint':adapt.binding(checkpoint),
        'recipe':state['payload']['recipe'],'resident_adoption':{'directory':str(directory),'generation':state['generation']}})
    cursor=json.loads((root/'cycle.json').read_text())
    recipe_reference={'path':state['payload']['recipe']['path'],'sha256':state['payload']['recipe']['file_sha256']}
    raw._publish_supervised_result(actual_config,job,actual_config,cursor,{
        'pending_native_checkpoint':str(checkpoint),'resident_current':adapt.binding(directory/'current.json'),
        'resident_recipe':{'source':recipe_reference,'copy':recipe_reference}})
    write(root/'cycle.json',cursor)
    current_config=json.loads(config.read_text())
    assert not native_bootstrap_metadata.unchanged(root/'bootstrap-verification.json',global_plan,declaration,handoff,current_config)
    preserved=[p.read_bytes()for p in(config,root/'cycle.json',directory.parent/'recipe-plan.json')]
    assert raw.configure_model(global_plan,declaration,handoff,root)==config
    assert native_bootstrap_metadata.unchanged(root/'bootstrap-verification.json',global_plan,declaration,handoff,current_config)
    assert raw.configure_model(global_plan,declaration,handoff,root)==config
    assert preserved==[p.read_bytes()for p in(config,root/'cycle.json',directory.parent/'recipe-plan.json')]
    for bad in ('initial-anchor','projected-plan','current-source','invocation','cursor-snapshot'):
        paths=(config,root/'cycle.json',directory/'initial.json',directory.parent/'recipe-plan.json')
        previous={path:path.read_bytes()for path in paths}
        if bad=='initial-anchor':
            x=json.loads((directory/'initial.json').read_text());x['plan_sha256']='0'*64;write(directory/'initial.json',x)
        elif bad=='projected-plan':
            x=json.loads(paths[-1].read_text());x['code']={};write(paths[-1],x)
        elif bad in ('current-source','invocation'):
            x=json.loads(config.read_text())
            if bad=='current-source':x['seed_native_graphs']=inputs['native_graphs']['path']
            else:x['seed_command']=['unreviewed-command']
            write(config,x)
        else:
            wrong=tmp_path/'other-resident-snapshot';wrong.mkdir(exist_ok=True)
            other=copy.deepcopy(state);other['at_global_step']+=1;write(wrong/'current.json',other)
            x=json.loads((root/'cycle.json').read_text());x['resident_current']=adapt.binding(wrong/'current.json');write(root/'cycle.json',x)
        mutated={path:path.read_bytes()for path in paths}
        with pytest.raises(ValueError):raw.configure_model(global_plan,declaration,handoff,root)
        assert mutated=={path:path.read_bytes()for path in paths}
        for path,data in previous.items():path.write_bytes(data)
    second_train=tmp_path/'appended-again.jsonl';second_train.write_text(appended.read_text()+json.dumps(graph('new-again'))+'\n')
    second_recipe=copy.deepcopy(recipe);second_recipe['input_bindings']['appended-again']=adapt.binding(second_train)
    second_recipe['stages'][1]['inputs'].update(native_graphs='appended-again',continued_native_graphs='appended')
    second_path=tmp_path/'append-again-recipe.json';write(second_path,second_recipe)
    resident.prepare(directory,second_path,'full',tokenizer_fingerprint='tokenizer-sha',role_ids=(),runtime_package=value['package'])
    assert manager.poll(6)is not None
    state=resident.current(directory);assert len(state['history'])==2
    second_job=tmp_path/'published-supervised-again';(second_job/'learner').mkdir(parents=True)
    second_checkpoint=second_job/'learner/checkpoint.pt';second_checkpoint.write_bytes(checkpoint.read_bytes())
    write(second_checkpoint.parent/'native-training-receipt.json',{'checkpoint':adapt.binding(second_checkpoint),
        'recipe':state['payload']['recipe'],'resident_adoption':{'directory':str(directory),'generation':state['generation']}})
    current_config=json.loads(config.read_text());cursor=json.loads((root/'cycle.json').read_text())
    recipe_reference={'path':state['payload']['recipe']['path'],'sha256':state['payload']['recipe']['file_sha256']}
    raw._publish_supervised_result(current_config,second_job,current_config,cursor,{
        'pending_native_checkpoint':str(second_checkpoint),'resident_current':adapt.binding(directory/'current.json'),
        'resident_recipe':{'source':recipe_reference,'copy':recipe_reference}})
    write(root/'cycle.json',cursor)
    assert not native_bootstrap_metadata.unchanged(root/'bootstrap-verification.json',global_plan,declaration,handoff,current_config)
    preserved=[p.read_bytes()for p in(config,root/'cycle.json',directory.parent/'recipe-plan.json')]
    assert raw.configure_model(global_plan,declaration,handoff,root)==config
    assert native_bootstrap_metadata.unchanged(root/'bootstrap-verification.json',global_plan,declaration,handoff,current_config)
    assert raw.configure_model(global_plan,declaration,handoff,root)==config
    assert preserved==[p.read_bytes()for p in(config,root/'cycle.json',directory.parent/'recipe-plan.json')]
    actual={'resident_current':adapt.binding(directory/'current.json')}
    requirement=raw._release_requirement(declaration,actual)
    assert requirement['recipe_sha256']==state['payload']['recipe']['sha256']
    assert requirement['recipe_lineage_sha256']==value['lineage']and requirement['stage_ids']==['full']
    assert 'exposure_counts'not in state
    bad_ready=tmp_path/'bad-readiness.json';write(bad_ready,{'resident_adoption':{'generation':'forged','history':state['history']}})
    with pytest.raises(ValueError,match='readiness omitted'):
        raw._release_requirement(declaration,{**actual,'readiness':adapt.binding(bad_ready)})


@pytest.mark.parametrize('bad_target',['unrelated','raw-future-plan','missing-after-adoption'])
def test_resident_setup_never_overwrites_unknown_plan_or_rebuilds_history(tmp_path,monkeypatch,bad_target):
    _,config,global_plan,proof=prepared(tmp_path)
    declaration,value=release_declaration(proof);write(global_plan,{'models':[declaration]})
    future=json.loads(Path(value['future_config']['path']).read_text());directory=Path(future['resident_data_directory'])
    target=directory.parent/'recipe-plan.json'
    if bad_target=='unrelated':target.write_bytes(b'unknown owner bytes')
    elif bad_target=='raw-future-plan':target.write_bytes(Path(value['future_plan']['path']).read_bytes())
    else:directory.mkdir();write(directory/'initial.json',{'history':'already owned'})
    prior_target=target.read_bytes()if target.exists()else None;before=config.read_bytes()
    with pytest.raises(ValueError):adapt.configure(global_plan,declaration,json.loads(before)['bootstrap_binding']['handoff'],config)
    assert config.read_bytes()==before
    assert (target.read_bytes()if target.exists()else None)==prior_target


def test_cached_configuration_repairs_missing_unstarted_plan_without_decode(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_bootstrap_metadata,native_raw_rounds as raw
    cases=tmp_path/'train-cases.jsonl';cases.write_text(json.dumps({'id':'admitted-train-case','split':'train'})+'\n')
    _,config,global_plan,proof=prepared(tmp_path,old_declaration_overrides={'cases':adapt.binding(cases)})
    declaration,value=release_declaration(proof);write(global_plan,{'models':[declaration]})
    monkeypatch.setattr(native_bootstrap_metadata,'save',lambda *args:None)
    bootstrap=json.loads(config.read_text())['bootstrap_binding']['handoff']
    adapt.configure(global_plan,declaration,bootstrap,config)
    before=config.read_bytes();future=json.loads(before);target=Path(future['resident_data_directory']).parent/'recipe-plan.json'
    target.unlink()
    monkeypatch.setattr(native_bootstrap_metadata,'unchanged',lambda *args:True)
    def no_load(*args,**kwargs):raise AssertionError('cached resident setup cannot decode historical state')
    monkeypatch.setattr(torch,'load',no_load)
    assert raw.configure_model(global_plan,declaration,bootstrap,config.parent)==config
    assert config.read_bytes()==before and adapt.binding(target)['sha256']==value['lineage_handoff']['sha256']
    target.write_bytes(b'unknown successor')
    with pytest.raises(ValueError):raw.configure_model(global_plan,declaration,bootstrap,config.parent)
    assert target.read_bytes()==b'unknown successor'and config.read_bytes()==before
