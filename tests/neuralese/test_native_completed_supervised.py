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
    plan=tmp_path/'global.json';write(plan,{'models':[{'model':'lfm','production_recipe':adapt.binding(recipe_path)}]})
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


def prepared(tmp_path,new_lr=1e-5):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path)
    consumption=tmp_path/'consumed.json'
    _,pair=adapt.consume(config,bridge.binding(request),bridge.binding(result),consumption,controller=controller)
    package=tmp_path/'package';package.mkdir();(package/'identity.py').write_text('VALUE=1\n')
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
