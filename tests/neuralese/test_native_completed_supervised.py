"""Real serialized optimizer/RNG and sealed bridge protocol; no quality claim."""
import copy,json,os,signal
from pathlib import Path
import pytest
import torch
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_completed_supervised as adapt,native_bridge as bridge
from natlang_neuralese.train.recipe import native_recipe_binding
from natlang_neuralese.train.loop import capture_training_rng_state,StopSignal
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from test_native_state_sft import graph


@pytest.mark.parametrize('fault', [None, 'code', 'device', 'recipe', 'flat-inventory',
    'missing-stage', 'held', 'data-dependency', 'proof', 'reader-bytes', 'old-reader-bytes'])
def test_completed_plan_replay_checks_real_classification_reader_relocation(tmp_path, monkeypatch, fault):
    from test_native_classification_recipe_handoff import prepared
    from natlang_neuralese.train import native_classification_continuation as continuation
    from scripts.neuralese_data import continue_native_source_indexes as adapter
    _, save, _, _ = prepared(tmp_path, monkeypatch)
    roles = save(); role = 'source_classification_migration'; readers = []
    for module in (continuation, adapter):
        path = tmp_path / 'historical' / Path(module.__file__).name
        path.parent.mkdir(exist_ok=True); path.write_bytes(Path(module.__file__).read_bytes())
        readers.append(adapt.binding(path))
    data = {k: roles['native_graphs'][k] for k in ('path', 'sha256')}
    roles[role]['dependencies'] = [data, *readers]
    def plan(inputs):
        return {'schema': 'natlang.neuralese-recipe-plan/1', 'code': {'frozen': 'exact'},
            'recipe': {'id': 'same'}, 'device': 'cuda', 'stage_inputs': {'full': inputs},
            'inputs': {item['path']: item['sha256'] for ref in inputs.values()
                       for item in (ref, *ref.get('dependencies', []))}}
    old = plan(roles); current = copy.deepcopy(roles)
    current[role]['dependencies'] = [data, *[adapt.binding(m.__file__) for m in (continuation, adapter)]]
    new = plan(current)
    if fault in ('code', 'device', 'recipe'): new[fault] = {'changed': True}
    elif fault == 'flat-inventory': new['inputs']['invented'] = 'a' * 64
    elif fault == 'missing-stage': new['stage_inputs'].clear()
    elif fault == 'held': current['held_native_graphs']['sha256'] = 'a' * 64
    elif fault == 'data-dependency': current[role]['dependencies'].pop(0)
    elif fault == 'proof': current[role]['sha256'] = 'a' * 64
    elif fault == 'reader-bytes':
        path = tmp_path / Path(continuation.__file__).name; path.write_text('# changed reader\n')
        current[role]['dependencies'][1] = adapt.binding(path)
    elif fault == 'old-reader-bytes': Path(readers[0]['path']).write_text('# changed history\n')
    if fault is None:
        assert adapt._same_replayed_recipe_plan(old, new)
    else:
        try: result = adapt._same_replayed_recipe_plan(old, new)
        except ValueError: result = False
        assert result is False


@pytest.mark.parametrize('fault', [None, 'held', 'proof', 'data-dependency', 'reader-bytes'])
def test_resident_release_replays_frozen_classification_readers(tmp_path, monkeypatch, fault):
    from test_native_classification_recipe_handoff import prepared
    from natlang_neuralese.train import recipe as recipes, native_resident as resident
    from natlang_neuralese.train import native_classification_continuation as continuation
    from natlang_neuralese.rl import native_raw_rounds as raw
    from scripts.neuralese_data import continue_native_source_indexes as adapter
    _, save, _, _ = prepared(tmp_path, monkeypatch)
    live = save(); role = 'source_classification_migration'
    data = {k: live['native_graphs'][k] for k in ('path', 'sha256')}
    live[role]['dependencies'] = [data, *[adapt.binding(m.__file__) for m in (continuation, adapter)]]
    frozen = copy.deepcopy(live)
    for index, module in enumerate((continuation, adapter), 1):
        path = tmp_path / 'old-runtime' / Path(module.__file__).name
        path.parent.mkdir(exist_ok=True); path.write_bytes(Path(module.__file__).read_bytes())
        frozen[role]['dependencies'][index] = adapt.binding(path)
    definition = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'frozen',
        'input_bindings': {v['binding']: {k: v[k] for k in ('path', 'sha256')} for v in live.values()},
        'stages': [{'id': 'full', 'kind': 'native_state_sft', 'requires': [],
                    'inputs': {k: v['binding'] for k, v in live.items()}, 'parameters': {'epochs': 1}}]}
    path = tmp_path / 'recipe.json'; write(path, definition)
    lineage = recipes.recipe_identity_sha256(path); rb = recipes.native_recipe_binding(path, lineage)
    actual_recipe = recipes.resolve_recipe_data(path)
    plan = {'recipe': actual_recipe, 'recipe_sha256': lineage, 'stage_inputs': {'full': frozen}}
    if fault == 'held': frozen['held_native_graphs']['sha256'] = 'a' * 64
    elif fault == 'proof': frozen[role]['sha256'] = 'a' * 64
    elif fault == 'data-dependency': frozen[role]['dependencies'].pop(0)
    elif fault == 'reader-bytes':
        old = Path(frozen[role]['dependencies'][1]['path']); old.write_text('# different historical reader\n')
        frozen[role]['dependencies'][1] = adapt.binding(old)
    payload = {'stage_id': 'full', 'recipe': rb, 'prior_recipe': rb, 'plan': plan}
    directory = tmp_path / 'resident'; directory.mkdir()
    write(directory / 'current.json', {'schema': resident.SCHEMA, 'payload': payload,
                                      'generation': resident.digest(payload), 'history': []})
    # Resolve the exact same immutable proof with today's same-byte readers,
    # reproducing the runtime-dependent path expansion in the real failure.
    original = recipes.resolve_stage_inputs
    def resolve(recipe, stage, supplied):
        values = original(recipe, stage, supplied)
        assert {k: {n: v[n] for n in ('path', 'sha256')} for k, v in values.items()} == \
               {k: {n: v[n] for n in ('path', 'sha256')} for k, v in live.items()}
        return copy.deepcopy(live)
    monkeypatch.setattr(recipes, 'resolve_stage_inputs', resolve)
    requirement = {'recipe_sha256': rb['sha256'], 'recipe_lineage_sha256': lineage, 'stage_ids': ['full']}
    if fault is None:
        result = raw._resident_release_requirement(requirement, {'resident_current': resident.binding(directory / 'current.json')})
        assert result['recipe_sha256'] == rb['sha256']
    else:
        with pytest.raises(ValueError):
            raw._resident_release_requirement(requirement, {'resident_current': resident.binding(directory / 'current.json')})


@pytest.mark.parametrize('bad_bytes',[-1,True,0,'12',None])
def test_scheduling_artifact_identity_validates_optional_bytes(tmp_path,bad_bytes):
    path=tmp_path/'global-plan.json';write(path,{'models':[]})
    plain=adapt.binding(path);complete=bridge.binding(path)
    assert adapt._artifact_binding(complete)==adapt._artifact_binding(plain)
    with pytest.raises(ValueError,match='byte count'):
        adapt._artifact_binding({**complete,'bytes':bad_bytes})


def fixture(tmp_path,quantum=2,recipe_with_bytes=False,opaque_inputs=False,resident_append=False):
    root=tmp_path/'rounds'/'lfm';root.mkdir(parents=True);train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    train.write_text(json.dumps(graph('old'))+'\n');h=graph('held');h['split']='test';held.write_text(json.dumps(h)+'\n')
    recipe={'schema':'natlang.neuralese-training-recipe/1','id':'lineage','identity_qualification':'native_state_runtime_qualified',
        'input_bindings':{'train':adapt.binding(train),'held':adapt.binding(held)},
        'stages':[{'id':name,'kind':'native_state_sft','requires':[],
            'inputs':{'native_graphs':'train','held_native_graphs':'held'},
            'parameters':{'epochs':1,'learning_rate':3e-6,'checkpoint_every':2,'eval_every':1}}
            for name in('dpo','full')]}
    if opaque_inputs:
        historical={'path':'dpo/retired-original.json','sha256':'a'*64}
        objective=tmp_path/'objective.json'
        write(objective,{'schema':'natlang.native-offline-pairwise-objective/1',
            'rows':{'row':{'source_preference':{'source_file':historical}}}})
        # A bound receipt is an opaque input, not a request to replay generation.
        teacher=tmp_path/'teacher.json';write(teacher,{'schema':'natlang.verified-teacher-clean-action-repair/1',
            'source':historical,'ordinary_partition':{'resident_current':historical}})
        recipe['input_bindings'].update(objective=adapt.binding(objective),teacher=adapt.binding(teacher))
        recipe['stages'][0]['kind']='native_offline_preference'
        recipe['stages'][0]['inputs']['objective_spec']='objective'
        # This role is declared by the shared SFT handler and binds bytes only.
        recipe['input_bindings'].pop('teacher')
        body=json.loads(objective.read_text());body['teacher_receipt']=adapt.binding(teacher);write(objective,body)
        recipe['input_bindings']['objective']=adapt.binding(objective)
    recipe_path=tmp_path/'recipe.json';write(recipe_path,recipe);rb=native_recipe_binding(recipe_path,'e'*64)
    declared_rb=rb;active_recipe=recipe_path;resident_cursor=None
    if resident_append:
        from natlang_neuralese.data import native_sources
        from natlang_neuralese.train import native_resident as resident
        from natlang_neuralese.rl.native_raw_handoff import _recipe_plan
        package=Path(native_sources.__file__).parents[1]
        directory=tmp_path/'resident-entry/resident-data';directory.mkdir(parents=True)
        installed=_recipe_plan(recipe_path,'cuda');installed.update(
            code=resident.code_inventory(package),recipe_sha256=rb['lineage_sha256'])
        write(directory.parent/'recipe-plan.json',installed)
        train_source=native_sources.NativeGraphSource(train,'tokenizer-sha',required_split='train')
        held_source=native_sources.NativeGraphSource(held,'tokenizer-sha',required_split='test')
        evidence=tmp_path/'geometry-evidence.json';write(evidence,{'passed':True,'scope':'tiny CPU fixture'})
        capacity=tmp_path/'capacity.json';write(capacity,{
            'schema':'natlang.native-resident-geometry-qualification/1','passed':True,
            'identity':None,'runtime_code_sha256':resident.digest(installed['code']),
            'limits':{'forward_tokens':1000000,'kv_slots':1000000},'evidence':resident.binding(evidence)})
        manager=resident.ResidentAdoption(directory,stage_id='full',recipe_binding=rb,
            train=train_source,held=held_source,package=package,geometry_qualification=capacity)
        appended=tmp_path/'appended.jsonl';appended.write_text(train.read_text()+json.dumps(graph('new'))+'\n')
        successor=copy.deepcopy(recipe);successor['input_bindings']['appended']=adapt.binding(appended)
        successor['stages'][1]['inputs'].update(native_graphs='appended',continued_native_graphs='train')
        active_recipe=tmp_path/'adopted-recipe.json';write(active_recipe,successor)
        resident.prepare(directory,active_recipe,'full',tokenizer_fingerprint='tokenizer-sha',
            role_ids=(),runtime_package=package)
        assert manager.poll(2)is not None
        state=resident.current(directory);rb=state['payload']['recipe']
        snapshot=tmp_path/'accounted-resident/current.json';snapshot.parent.mkdir();write(snapshot,state)
        resident_cursor=adapt.binding(snapshot)
    parent=tmp_path/'parent.pt';parent.write_bytes(b'actual protocol parent')
    bootstrap=tmp_path/'historical-bootstrap.json';write(bootstrap,{'retained':'original bootstrap provenance'})
    config={'native_state':True,'native_actor_mode':'raw-state','out':str(root/'production'),
        'supervised_stage_quantum_updates':{'dpo':quantum,'full':1000},'checkpoint':str(parent),'production_recipe':str(active_recipe),'production_stage':'full',
        'raw_model_id':'fixture:'+('a'*64),'recipe_lineage_sha256':'e'*64,
        'bootstrap_binding':{'handoff':adapt.binding(bootstrap)},
        'seed_command':['python','-m','natlang_neuralese.train.native_state_sft','{seed_trainer_args}'],
        'trainer_parameters':{'recipe_binding':rb},'seed_trainer_parameters':{'recipe_binding':rb}}
    config_path=root/'config.json';write(config_path,config)
    (root/'production').mkdir();write(root/'production/state.json',{'phase':'collect','checkpoint':str(parent),'round':2,'trained_rounds':2})
    write(root/'cycle.json',{'phase':'sft','cycle':3,'next_stage':0,'current_checkpoint':adapt.binding(parent),
        **({'resident_current':resident_cursor}if resident_cursor else{})})
    job=root/'cycles/000003/supervised/000';job.mkdir(parents=True)
    continuation={**config,'mandatory_sft_only':True,'production_stage':'dpo','out':str(job),
        'trainer_parameters':{'stage_id':'dpo','refresh_after_updates':quantum}};write(job/'config.json',continuation)
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
    plan=tmp_path/'global.json';write(plan,{'models':[{'model':'lfm','production_recipe':(bridge.binding(recipe_path)if recipe_with_bytes else adapt.binding(recipe_path)),
        'handoff_requirement':{'model_id':config['raw_model_id'],'recipe_sha256':declared_rb['sha256'],
            'recipe_lineage_sha256':declared_rb['lineage_sha256'],'stage_ids':['full']}}]})
    config['runtime_binding']={'plan_sha256':adapt.binding(plan)['sha256'],'handoff':config['bootstrap_binding']['handoff']};config['supervised_stage_quantum_control']={'plan':adapt.binding(plan),'updates':config['supervised_stage_quantum_updates']};write(config_path,config)
    controller={'pid':99999999,'start_ticks':'0','boot_id':'gone'}
    return root,config_path,request,result,controller,plan,recipe_path


@pytest.mark.parametrize('missing_witness',[False,True])
def test_completed_adaptation_after_previously_accounted_resident_append(tmp_path,monkeypatch,missing_witness):
    from natlang_neuralese.data import native_sources
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    root,config,request,result,controller,plan,declared_recipe=fixture(tmp_path,resident_append=True)
    frozen_global=plan.read_bytes();configuration=json.loads(config.read_text())
    assert configuration['production_recipe']!=str(declared_recipe)
    if missing_witness:
        cursor=json.loads((root/'cycle.json').read_text());cursor.pop('resident_current');write(root/'cycle.json',cursor)
    consumed,pair=adapt.consume(config,bridge.binding(request),bridge.binding(result),
        tmp_path/'consumed.json',controller=controller)
    before=tuple(p.read_bytes()for p in(config,root/'cycle.json',root/'production/state.json'))
    checkpoint=bridge.binding(pair[0]['path']);moments=copy.deepcopy(pair[1]['training']['optimizer'])
    package=Path(native_sources.__file__).parents[1]
    def prepare():
        return adapt.prepare(config,plan,'lfm',adapt.binding(configuration['production_recipe']),
            adapt.binding(tmp_path/'consumed.json'),pair,tmp_path/'adaptation/proof.json',package=package)
    if missing_witness:
        with pytest.raises(ValueError,match='without accounted resident history'):prepare()
    else:
        proof=prepare();value=adapt.verify(proof)
        assert value['global_step']==4 and value['new_sources_earned_visits']==0
        assert value['old_global_plan']['sha256']==adapt.binding(plan)['sha256']
        assert value['started_stage_ids']==['dpo','full']
    assert before==tuple(p.read_bytes()for p in(config,root/'cycle.json',root/'production/state.json'))
    assert checkpoint==bridge.binding(pair[0]['path'])and plan.read_bytes()==frozen_global
    from natlang_neuralese.train.checkpoint_offload import _same_values
    _same_values(moments,pair[1]['training']['optimizer'])


@pytest.mark.parametrize('recipe_with_bytes',[False,True])
def test_completed_stage_accounted_once_and_same_decode_reused(tmp_path,monkeypatch,recipe_with_bytes):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path,recipe_with_bytes=recipe_with_bytes)
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
    future_ref=bridge.binding(future)if recipe_with_bytes else adapt.binding(future)
    proof=adapt.prepare(config,plan,'lfm',future_ref,adapt.binding(tmp_path/'consumed.json'),pair,tmp_path/'adaptation/proof.json',package=package)
    assert len(loads)==1 and bridge.binding(pair[0]['path'])==before
    actual=adapt.verify(proof);assert actual['new_sources_earned_visits']==0
    assert actual['recipe']==future_ref
    declaration={**json.loads(plan.read_text())['models'][0],
        'production_recipe':adapt.binding(future)if recipe_with_bytes else bridge.binding(future),
        'completed_supervised_adaptation':proof}
    assert adapt.verify_declaration(proof,declaration)==actual
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


def prepared(tmp_path,new_lr=1e-5,old_declaration_overrides=None,runtime_package=None,bootstrap_handoff=None,opaque_inputs=False,future_declaration_overrides=None):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path,opaque_inputs=opaque_inputs)
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
    target=None
    if future_declaration_overrides is not None:
        updated=json.loads(plan.read_text());updated['models'][0].update(
            production_recipe=adapt.binding(future),actor_runtime=str(package.resolve().parents[2]),
            **future_declaration_overrides)
        target=tmp_path/'future-global.json';write(target,updated)
    proof=adapt.prepare(config,plan,'lfm',adapt.binding(future),adapt.binding(consumption),pair,tmp_path/'adaptation/proof.json',package=package,
        future_global_plan=adapt.binding(target)if target else None)
    return root,config,plan,proof


def test_raw_configuration_reuses_actual_owned_controller_lease(tmp_path,monkeypatch):
    import fcntl
    from natlang_neuralese.rl import native_raw_rounds as raw,native_bootstrap_metadata
    root,config,plan,proof=prepared(tmp_path)
    value=adapt.verify(proof);document=json.loads(plan.read_text())
    document['models'][0].update(production_recipe=value['recipe'],completed_supervised_adaptation=proof)
    write(plan,document);declaration=document['models'][0]
    prior=json.loads(config.read_text());bootstrap=prior['bootstrap_binding']['handoff']
    monkeypatch.setattr(raw,'_task_cases',lambda declaration:None)
    monkeypatch.setattr(native_bootstrap_metadata,'unchanged',lambda *args:False)
    monkeypatch.setattr(native_bootstrap_metadata,'save',lambda *args:None)
    before=(root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes()
    with (root.parent/'controller.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # The completed verifier must reject a competing open description.
        with pytest.raises(BlockingIOError):
            raw.configure_model(plan,declaration,bootstrap,root)
        assert raw.configure_model(plan,declaration,bootstrap,root,
            owned_controller_fd=lease.fileno())==config
        raw.verify_controller_fd(lease.fileno(),root.parent/'controller.lock')
    assert before==((root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes())
    assert json.loads(config.read_text())['completed_supervised_adaptation']==proof


@pytest.mark.parametrize('new_lr',[1e-5,1e-6])
@pytest.mark.parametrize('separate_configured_plan',[False,True])
def test_configuration_changes_future_lr_only_and_does_not_repeat_completed_stage(tmp_path,monkeypatch,new_lr,separate_configured_plan):
    root,config,plan,proof=prepared(tmp_path,new_lr)
    from natlang_neuralese.rl import native_bootstrap_metadata
    monkeypatch.setattr(native_bootstrap_metadata,'save',lambda *args:None)
    value=adapt.verify(proof);p=json.loads(plan.read_text());p['models'][0].update(
        production_recipe=value['recipe'],completed_supervised_adaptation=proof)
    configured_plan=tmp_path/'configured-global.json'if separate_configured_plan else plan
    write(configured_plan,p)
    before=(root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes()
    old=json.loads(config.read_text())
    adapt.configure(configured_plan,p['models'][0],old['bootstrap_binding']['handoff'],config)
    future=json.loads(config.read_text())
    assert future['seed_trainer_parameters']['resident_data_directory']==str(Path(value['recipe']['path']).parent/'resident-data')
    assert future['resident_data_directory']==future['seed_trainer_parameters']['resident_data_directory']
    assert future['seed_trainer_parameters']['learning_rate']==new_lr
    assert future['supervised_stage_quantum_control']['plan']==adapt.binding(configured_plan)
    assert before==((root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes())
    assert json.loads((root/'cycle.json').read_text())['next_stage']==1
    adapt.configure(configured_plan,p['models'][0],old['bootstrap_binding']['handoff'],config)


def test_raw_dispatch_applies_verified_completed_successor_to_runtime_handoff(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw,native_bootstrap_metadata
    historical=tmp_path/'historical-runtime.json';write(historical,{'retained':'original authority'})
    root,config,plan,proof=prepared(tmp_path,runtime_package=Path(adapt.__file__).parents[1],
        old_declaration_overrides={'runtime_handoff':adapt.binding(historical)},
        future_declaration_overrides={'runtime_handoff':None})
    value=adapt.verify(proof);future=json.loads(Path(value['future_global_plan']['path']).read_text())
    declaration=future['models'][0];declaration['completed_supervised_adaptation']=proof
    current=tmp_path/'current-global.json';write(current,future)
    original=json.loads(config.read_text());before=(root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes()
    # Task admission has separate coverage; this serialized fixture isolates dispatch.
    monkeypatch.setattr(raw,'_task_cases',lambda declaration:None)
    monkeypatch.setattr(native_bootstrap_metadata,'save',lambda *args:None)
    assert raw.configure_model(current,declaration,original['bootstrap_binding']['handoff'],root)==config
    applied=json.loads(config.read_text())
    assert applied['completed_supervised_adaptation']==proof and not applied.get('runtime_handoff')
    assert applied['production_recipe']==value['recipe']['path']
    assert before==((root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes())


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
    # A later completed operation still belongs to the frozen declaration;
    # its exact admitted recipe is authenticated by the accounted cursor.
    current_config=json.loads(config.read_text());cursor=json.loads((root/'cycle.json').read_text())
    adapt._verify_controller_recipe(declaration,current_config,cursor,value['lineage'])
    with pytest.raises(ValueError,match='without accounted resident history'):
        adapt._verify_controller_recipe(declaration,current_config,{},value['lineage'])
    with pytest.raises(ValueError,match='accounted resident recipe'):
        adapt._verify_controller_recipe(declaration,current_config,cursor,'f'*64)
    alias=tmp_path/'same-bytes-unaccounted-recipe.json'
    alias.write_bytes(Path(current_config['production_recipe']).read_bytes())
    with pytest.raises(ValueError,match='accounted resident recipe'):
        adapt._verify_controller_recipe(declaration,{**current_config,'production_recipe':str(alias)},cursor,value['lineage'])
    corrupted=copy.deepcopy(state);corrupted['history'][0]['recipe']=corrupted['history'][-1]['recipe']
    wrong=tmp_path/'broken-accounted-history/current.json';wrong.parent.mkdir();write(wrong,corrupted)
    with pytest.raises(ValueError):
        adapt._verify_controller_recipe(declaration,current_config,{'resident_current':adapt.binding(wrong)},value['lineage'])
    unrelated=copy.deepcopy(declaration);unrelated['production_recipe']=adapt.binding(alias)
    with pytest.raises(ValueError):
        adapt._verify_controller_recipe(unrelated,current_config,cursor,value['lineage'])


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


def test_qualified_early_refresh_uses_one_preloaded_saved_object_under_owned_lease(tmp_path,monkeypatch):
    import fcntl
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    root,config,request,result,controller,plan,recipe=fixture(tmp_path,quantum=3)
    checkpoint=Path(json.loads(result.read_text())['checkpoint']['path'])
    receipt=checkpoint.parent/'native-training-receipt.json';value=json.loads(receipt.read_text())
    stop=StopSignal().install((signal.SIGUSR2,))
    try:signal.raise_signal(signal.SIGUSR2)
    finally:stop.restore()
    assert stop.requested
    value['continuation_boundary']['requested_signals']=stop.received;write(receipt,value)
    result_value=json.loads(result.read_text());result_value['artifacts']=bridge.inventory(checkpoint.parent.parent,exclude=[result]);write(result,result_value)
    pair=(bridge.binding(checkpoint),CheckpointPolicy.load(checkpoint))
    def forbidden(*a,**k):raise AssertionError('an owned completed parent may not be decoded again')
    monkeypatch.setattr(CheckpointPolicy,'load',staticmethod(forbidden))
    lease=(root.parent/'controller.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        consumed,returned=adapt.consume(config,bridge.binding(request),bridge.binding(result),tmp_path/'consumed.json',controller=controller,verified_checkpoint=pair,owned_controller_fd=lease.fileno())
        assert returned[1]is pair[1]and consumed['new_optimizer_updates']==2
        assert json.loads((root/'cycle.json').read_text())['next_stage']==1
        with (root.parent/'controller.lock').open('a')as probe:
            with pytest.raises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:lease.close()


def test_explicit_old_to_new_runtime_keeps_exact_optimizer_rng_and_requires_new_qualification(tmp_path):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path)
    consumed_path=tmp_path/'consumed.json'
    _,pair=adapt.consume(config,bridge.binding(request),bridge.binding(result),consumed_path,controller=controller)
    old=tmp_path/'old-package';old.mkdir();(old/'identity.py').write_text('VALUE=1\n')
    new=tmp_path/'new-package';new.mkdir();(new/'identity.py').write_text('VALUE=2\n')
    future=tmp_path/'future.json';write(future,json.loads(recipe.read_text()))
    optimizer=copy.deepcopy(pair[1]['training']['optimizer']);rng=pair[1]['training']['rng']['torch_cpu'].clone() if 'torch_cpu'in pair[1]['training']['rng']else None
    proof=adapt.prepare(config,plan,'lfm',adapt.binding(future),adapt.binding(consumed_path),pair,tmp_path/'adaptation/proof.json',package=new,old_package=old)
    checked=adapt.verify(proof)
    assert checked['old_code']==adapt.code_inventory(old)and checked['code']==adapt.code_inventory(new)
    assert checked['fresh_runtime_qualification_required']is True
    assert pair[1]['training']['optimizer']['param_groups']==optimizer['param_groups']
    if rng is not None:assert torch.equal(pair[1]['training']['rng']['torch_cpu'],rng)
    assert checked['whole_recipe_completed']is False and checked['new_sources_earned_visits']==0


@pytest.mark.parametrize('future_recipe_bytes',[False,True])
@pytest.mark.parametrize('future_plan_bytes',[False,True])
@pytest.mark.parametrize('separate_configured_plan',[False,True])
def test_runtime_configuration_requires_exact_future_global_and_rebinds_factory_pair(tmp_path,monkeypatch,future_recipe_bytes,future_plan_bytes,separate_configured_plan):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path)
    consumed_path=tmp_path/'consumed.json'
    _,pair=adapt.consume(config,bridge.binding(request),bridge.binding(result),consumed_path,controller=controller)
    old=tmp_path/'old-runtime/training/neuralese/natlang_neuralese';old.mkdir(parents=True);(old/'identity.py').write_text('VALUE=1\n')
    package=tmp_path/'new-runtime/training/neuralese/natlang_neuralese';package.mkdir(parents=True);(package/'identity.py').write_text('VALUE=2\n')
    future=tmp_path/'future.json';write(future,json.loads(recipe.read_text()))
    target=json.loads(plan.read_text());row=target['models'][0];row.update(actor_runtime=str(package.parents[2]),production_recipe=(bridge.binding(future)if future_recipe_bytes else adapt.binding(future)),
        commands={'seed_command':['env','PYTHONPATH='+str(package.parent),'python','-m','natlang_neuralese.train.native_state_sft','{seed_trainer_args}']})
    target_path=tmp_path/'future-global.json';write(target_path,target)
    from natlang_neuralese.rl import production,native_bootstrap_metadata
    actor_root=Path(package).parents[2]
    actor_entry=actor_root/'scripts/neuralese_data/serve_direct_checkpoint.py'
    actor_entry.parent.mkdir(parents=True,exist_ok=True);actor_entry.write_text('# selected actor runtime')
    actor_environment={'NATLANG_REPO':str(actor_root),
        'PYTHONPATH':str(actor_root/'training/neuralese')+':'+str(actor_root)}
    actor_command=['env','NATLANG_REPO='+str(actor_root),'PYTHONPATH='+actor_environment['PYTHONPATH'],
        'python',str(actor_entry),'--native-state','--preserved-arg','yes']
    monkeypatch.setattr(production,'actor_runtime_configuration',lambda repository:{'actor_command':actor_command.copy(),
        'actor_environment':actor_environment,'native_actor_python_runtime':{'exact':'new-runtime'}})
    monkeypatch.setattr(native_bootstrap_metadata,'save',lambda *a:None)
    proof=adapt.prepare(config,plan,'lfm',adapt.binding(future),adapt.binding(consumed_path),pair,tmp_path/'adaptation/proof.json',package=package,old_package=old,future_global_plan=(bridge.binding(target_path)if future_plan_bytes else adapt.binding(target_path)))
    checked=adapt.verify(proof);assert checked['fresh_runtime_qualification_required']
    target['models'][0]['completed_supervised_adaptation']=proof
    configured_plan=tmp_path/'configured-global.json'if separate_configured_plan else plan
    write(configured_plan,target)
    before=(root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes()
    adapt.configure(configured_plan,target['models'][0],json.loads(config.read_text())['bootstrap_binding']['handoff'],config)
    configured=json.loads(config.read_text())
    assert configured['actor_command']==actor_command
    assert configured['actor_environment']['NATLANG_REPO']==str(package.parents[2])
    assert configured['native_actor_python_runtime']=={'exact':'new-runtime'}
    assert configured['seed_command'][1]=='PYTHONPATH='+str(package.parent)
    assert before==((root/'cycle.json').read_bytes(),(root/'production/state.json').read_bytes())


def applied_publication(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    root,config_path,request,result,controller,plan,recipe=fixture(tmp_path)
    paths={'config':config_path,'state':root/'production/state.json','cursor':root/'cycle.json'}
    directory=tmp_path/'publication';directory.mkdir()
    before={key:adapt.binding(path)for key,path in paths.items()}
    for key,path in paths.items():adapt.write_bytes_durable(directory/(key+'-before.json'),path.read_bytes())
    config=json.loads(config_path.read_text());cursor=json.loads(paths['cursor'].read_text());state=json.loads(paths['state'].read_text())
    request_value=bridge.read_request(request);continuation=json.loads(Path(request_value['config']['path']).read_text());result_value=json.loads(result.read_text());operation=json.loads(Path(result_value['state']['path']).read_text())
    checkpoint=result_value['checkpoint'];pair=(checkpoint,CheckpointPolicy.load(checkpoint['path']));effects={}
    def capture_json(path,value):effects[Path(path)]=json.dumps(value,indent=2,sort_keys=True).encode()
    with monkeypatch.context()as patch:
        patch.setattr(raw,'_publish_checkpoint',lambda cfg,cp:capture_json(paths['state'],raw._published_checkpoint_state(state,cp)))
        patch.setattr(raw,'write_json_durable',capture_json)
        raw._publish_supervised_result(continuation,Path(request_value['out']),config,cursor,operation)
    cursor.update(next_stage=cursor['next_stage']+1,current_checkpoint=adapt.binding(checkpoint['path']))
    capture_json(paths['cursor'],cursor)
    if paths['config']not in effects:capture_json(paths['config'],config)
    candidates={}
    for key,path in paths.items():
        target=directory/(key+'-candidate.json');adapt.write_bytes_durable(target,effects[path]);candidates[key]=adapt.binding(target);adapt.write_bytes_durable(path,effects[path])
    intent={'schema':'natlang.completed-worker-normal-publication-commit/1','request':adapt.binding(request),'result':adapt.binding(result),'checkpoint':checkpoint,
        'before':before,'candidates':candidates,'global_step':4,'actual_request_updates':2,
        'full_decodes':0,'new_optimizer_updates':0,'new_exposure':0,'whole_recipe_completed':False,'normal_shared_publisher_reused':True}
    write(directory/'intent.json',intent)
    publication={**intent,'passed':True,'intent':adapt.binding(directory/'intent.json'),'after':{key:adapt.binding(path)for key,path in paths.items()}}
    write(directory/'result.json',publication)
    return root,config_path,controller,pair,adapt.binding(directory/'result.json'),plan,recipe


def test_already_applied_publication_reuses_real_publisher_and_never_advances_again(tmp_path,monkeypatch):
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    root,config,controller,pair,publication,plan,recipe=applied_publication(tmp_path,monkeypatch)
    before=tuple(path.read_bytes()for path in(config,root/'cycle.json',root/'production/state.json'))
    def forbidden(*a,**k):raise AssertionError('registering real publication cannot decode another checkpoint')
    monkeypatch.setattr(CheckpointPolicy,'load',staticmethod(forbidden))
    record,returned=adapt.register_publication(config,publication,tmp_path/'registered.json',controller=controller,verified_checkpoint=pair)
    assert returned[1]is pair[1]and record['cursor_advances_from_registration']==0
    assert before==tuple(path.read_bytes()for path in(config,root/'cycle.json',root/'production/state.json'))
    adapt.register_publication(config,publication,tmp_path/'registered.json',controller=controller,verified_checkpoint=pair)
    package=tmp_path/'package';package.mkdir();(package/'identity.py').write_text('VALUE=1\n')
    proof=adapt.prepare(config,plan,'lfm',adapt.binding(recipe),adapt.binding(tmp_path/'registered.json'),pair,tmp_path/'adaptation/proof.json',package=package)
    assert adapt.verify(proof)['new_sources_earned_visits']==0


@pytest.mark.parametrize('mutation',['counter','successor','candidate'])
def test_bad_publication_reconciliation_cannot_touch_current_controls(tmp_path,monkeypatch,mutation):
    root,config,controller,pair,publication,_,_=applied_publication(tmp_path,monkeypatch)
    published=json.loads(Path(publication['path']).read_text())
    if mutation=='counter':
        intent_path=Path(published['intent']['path']);intent=json.loads(intent_path.read_text());intent['actual_request_updates']=3;write(intent_path,intent)
        published.update(actual_request_updates=3,intent=adapt.binding(intent_path));write(Path(publication['path']),published);publication=adapt.binding(publication['path'])
    elif mutation=='successor':write(root/'cycle.json',{'unknown':'newer owner'})
    else:Path(published['candidates']['state']['path']).write_text('{}')
    before=tuple(path.read_bytes()for path in(config,root/'cycle.json',root/'production/state.json'))
    with pytest.raises(ValueError):adapt.register_publication(config,publication,tmp_path/'registered.json',controller=controller,verified_checkpoint=pair)
    assert before==tuple(path.read_bytes()for path in(config,root/'cycle.json',root/'production/state.json'))
    assert not(tmp_path/'registered.json').exists()


def publisher_fixture(tmp_path):
    source=tmp_path/'external-control.py';source.write_text('CONTROL=1\n')
    loader=tmp_path/'loader.py';loader.write_text('LOADER=1\n')
    authority=tmp_path/'authority.json';write(authority,{'actual':'both role maps'})
    base=tmp_path/'base.json';write(base,{'schema':'natlang.external-owned-controller-closure/1',
        'modules':[{'module':'rl.native_completed_supervised','source':adapt.binding(source)}]})
    manifest=tmp_path/'publisher.json';write(manifest,{'schema':'natlang.source-host-publishing-control-closure/1',
        'base_loader':adapt.binding(loader),'base_manifest':adapt.binding(base),
        'role_authorities':adapt.binding(authority),'controls':{'actual':adapt.binding(source)}})
    return manifest,source,authority,base,loader


def test_dependency_closure_keeps_nested_publisher_controls_and_exact_packages(tmp_path):
    manifest,source,authority,base,loader=publisher_fixture(tmp_path)
    root,config,plan,proof=prepared(tmp_path,old_declaration_overrides={'publisher_manifest':adapt.binding(manifest)})
    refs=adapt.dependencies(proof)
    for path in(source,manifest,authority,base,loader,Path(proof['path'])):
        assert refs[str(path)]['sha256']==adapt.binding(path)['sha256']
    value=adapt.verify(proof)
    for package,code in(('package','code'),('old_package','old_code')):
        for relative in value[code]:assert str(Path(value[package])/relative)in refs
    consumed=json.loads(Path(value['consumption']['path']).read_text())
    for item in consumed['snapshots'].values():assert item['path']in refs
    source.write_text('CONTROL=2\n')
    with pytest.raises(ValueError):adapt.dependencies(proof)


def test_dependency_closure_rejects_missing_future_authority(tmp_path):
    manifest,source,authority,base,loader=publisher_fixture(tmp_path)
    _,_,_,proof=prepared(tmp_path,old_declaration_overrides={'publisher_manifest':adapt.binding(manifest)})
    authority.unlink()
    with pytest.raises((ValueError,FileNotFoundError)):adapt.dependencies(proof)


def test_superseded_publisher_is_history_but_selected_future_controls_remain_strict(tmp_path,monkeypatch):
    from natlang_neuralese.rl import production
    monkeypatch.setattr(production,'actor_runtime_configuration',lambda repository:{})
    old=tmp_path/'old-controls';old.mkdir();new=tmp_path/'new-controls';new.mkdir()
    previous,old_source,*_=publisher_fixture(old)
    current,new_source,*_=publisher_fixture(new)
    _,_,_,proof=prepared(tmp_path,old_declaration_overrides={'publisher_manifest':adapt.binding(previous)},
        future_declaration_overrides={'publisher_manifest':adapt.binding(current)})
    previous.unlink();old_source.unlink()
    refs=adapt.dependencies(proof)
    assert str(current)in refs and str(new_source)in refs
    assert str(previous)not in refs and str(old_source)not in refs
    new_source.write_text('CONTROL=wrong\n')
    with pytest.raises(ValueError):adapt.dependencies(proof)


@pytest.mark.parametrize('missing',['producer','dependency_producer','operational'])
def test_completed_current_publication_selection_is_a_strict_dependency(tmp_path,missing):
    files={}
    for name in ('producer','dependency_producer','operational'):
        path=tmp_path/(name+'.py');path.write_text(name+'=1\n');files[name]=bridge.binding(path)
    selection={'schema':'natlang.native-checkpoint-publication-selection/1',
        'producer':files['producer'],'dependency_producer':files['dependency_producer'],
        'namespace':str(tmp_path),'readonly_input_roots':[]}
    _,_,_,proof=prepared(tmp_path,old_declaration_overrides={
        'native_checkpoint_publication':selection,'operational_inputs':[files['operational']]})
    refs=adapt.dependencies(proof)
    assert all(item['path']in refs for item in files.values())
    Path(files[missing]['path']).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):adapt.dependencies(proof)


@pytest.mark.parametrize('mutation',[None,'objective','train','held','snapshot','recipe'])
def test_actual_completed_replay_uses_typed_inputs_not_opaque_history(tmp_path,mutation):
    _,_,_,proof=prepared(tmp_path,opaque_inputs=True)
    value=adapt.verify(proof)
    if mutation:
        selected={'objective':tmp_path/'objective.json','train':tmp_path/'train.jsonl',
            'held':tmp_path/'held.jsonl','snapshot':Path(json.loads(Path(value['consumption']['path']).read_text())['snapshots']['cursor']['path']),
            'recipe':Path(value['recipe']['path'])}[mutation]
        selected.write_text('{}')
        with pytest.raises((ValueError,FileNotFoundError,KeyError)):adapt.dependencies(proof)
    else:
        refs=adapt.dependencies(proof)
        assert str(tmp_path/'objective.json')in refs
        assert str(tmp_path/'teacher.json')not in refs
        assert not any('retired-original'in path for path in refs)
        # Historical operation parent is not a checkpoint that replay decodes.
        assert str(tmp_path/'parent.pt')not in refs
        assert value['checkpoint']['path']in refs


@pytest.mark.parametrize('mutation',['signal-string','wrong-signal','wrong-kind','false-reached','integer-reached','wrong-target','boolean-updates'])
def test_early_refresh_requires_actual_typed_request_and_exact_declared_target(mutation):
    config={'supervised_stage_quantum_updates':{'full':1000}}
    receipt={'continuation_boundary':{'kind':'semantic-refresh','reached':True,'requested_signals':[{'signal':'SIGUSR2','unix':1792488423.992293}],'new_optimizer_updates':749}}
    continuation={'trainer_parameters':{'refresh_after_updates':1000}}
    if mutation=='signal-string':receipt['continuation_boundary']['requested_signals']='SIGUSR2'
    elif mutation=='wrong-signal':receipt['continuation_boundary']['requested_signals']=[{'signal':'SIGTERM','unix':1792488423.992293}]
    elif mutation=='wrong-kind':receipt['continuation_boundary']['kind']='operational-stop'
    elif mutation=='false-reached':receipt['continuation_boundary']['reached']=False
    elif mutation=='integer-reached':receipt['continuation_boundary']['reached']=1
    elif mutation=='wrong-target':continuation['trainer_parameters']['refresh_after_updates']=999
    else:receipt['continuation_boundary']['new_optimizer_updates']=True
    with pytest.raises(ValueError):adapt._qualified_refresh_updates(config,{'id':'full'},receipt,continuation)


@pytest.mark.parametrize('mutation', ['bytes', 'bool-bytes', 'negative-bytes', 'hash', 'path', 'extra'])
def test_completed_recipe_optional_bytes_rejects_tampering_with_real_proof(tmp_path, mutation):
    root, config, plan, proof = prepared(tmp_path)
    value = adapt.verify(proof)
    recipe = Path(value['recipe']['path'])
    reference = bridge.binding(recipe)
    if mutation == 'bytes': reference['bytes'] += 1
    elif mutation == 'bool-bytes': reference['bytes'] = True
    elif mutation == 'negative-bytes': reference['bytes'] = -1
    elif mutation == 'hash': reference['sha256'] = '0' * 64
    elif mutation == 'path':
        other = tmp_path / 'same-bytes-other-recipe.json'
        other.write_bytes(recipe.read_bytes())
        reference = bridge.binding(other)
    else: reference['unchecked'] = 'not an allowed recipe binding field'
    value['recipe'] = reference
    write(Path(proof['path']), value)
    before = tuple(p.read_bytes() for p in (config, root/'cycle.json', root/'production/state.json'))
    with pytest.raises(ValueError):
        adapt.verify(adapt.binding(Path(proof['path'])))
    assert before == tuple(p.read_bytes() for p in (config, root/'cycle.json', root/'production/state.json'))


@pytest.mark.parametrize('records', [None, [], ['SIGUSR2'], 'SIGUSR2',
    [{'signal':'SIGUSR2'}], [{'unix':1.0}], [{'signal':'SIGUSR2','unix':True}],
    [{'signal':'SIGUSR2','unix':float('nan')}], [{'signal':'SIGUSR2','unix':float('inf')}],
    [{'signal':'SIGUSR2','unix':float('-inf')}], [{'signal':'SIGUSR2','unix':'1.0'}],
    [{'signal':'SIGUSR2','unix':10**1000}], [{'signal':'UNKNOWN','unix':1.0}],
    [{'signal':'SIGUSR2','unix':1.0,'extra':True}],
    [{'signal':'SIGUSR2','unix':1.0}, {'signal':'SIGTERM','unix':False}]])
def test_early_refresh_rejects_malformed_actual_producer_records(records):
    config={'supervised_stage_quantum_updates':{'full':1000}}
    receipt={'continuation_boundary':{'kind':'semantic-refresh','reached':True,
        'requested_signals':records,'new_optimizer_updates':611}}
    with pytest.raises(ValueError):
        adapt._qualified_refresh_updates(config,{'id':'full'},receipt,
            {'trainer_parameters':{'refresh_after_updates':1000}})


def test_early_refresh_accepts_actual_stop_signal_records_without_rewriting_receipt():
    stop=StopSignal().install((signal.SIGUSR2,))
    try:signal.raise_signal(signal.SIGUSR2)
    finally:stop.restore()
    assert stop.requested and stop.received[0]['signal']=='SIGUSR2'
    receipt={'continuation_boundary':{'kind':'semantic-refresh','reached':True,
        'requested_signals':stop.received,'new_optimizer_updates':611}}
    original=copy.deepcopy(receipt)
    assert adapt._qualified_refresh_updates({'supervised_quantum_updates':1000},
        {'id':'full'},receipt,{'trainer_parameters':{'refresh_after_updates':1000}})==611
    assert receipt==original


@pytest.mark.parametrize('changed_cursor', [False, True])
def test_completed_recipe_replacement_retains_only_its_exact_historical_resident_cursor(tmp_path, monkeypatch, changed_cursor):
    from natlang_neuralese.data import native_sources
    from natlang_neuralese.rl import native_raw_rounds as raw
    from natlang_neuralese.train.recipe import recipe_identity_sha256
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    root, config, request, result, controller, plan, _ = fixture(tmp_path, resident_append=True)
    _, pair = adapt.consume(config, bridge.binding(request), bridge.binding(result),
        tmp_path / 'consumed.json', controller=controller)
    before = json.loads((root / 'cycle.json').read_text())['resident_current']
    definition = json.loads(Path(json.loads(config.read_text())['production_recipe']).read_text())
    definition['stages'][1]['parameters']['learning_rate'] = 1e-5
    future = tmp_path / 'changed-recipe.json'; write(future, definition)
    proof = adapt.prepare(config, plan, 'lfm', adapt.binding(future),
        adapt.binding(tmp_path / 'consumed.json'), pair, tmp_path / 'adaptation/proof.json',
        package=Path(native_sources.__file__).parents[1])
    declaration, value = release_declaration(proof)
    original = before
    if changed_cursor:
        alias = tmp_path / 'unaccounted-resident.json'
        alias.write_bytes(Path(before['path']).read_bytes())
        before = adapt.binding(alias)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('release must not decode tensors'))
    if changed_cursor:
        with pytest.raises(ValueError, match='raw release'):
            raw._release_requirement(declaration, {'resident_current': before})
    else:
        answer = raw._release_requirement(declaration, {'resident_current': before})
        assert answer['recipe_sha256'] == recipe_identity_sha256(future)
        assert answer['recipe_lineage_sha256'] == value['lineage']
        assert answer['recipe_sha256'] != json.loads(Path(original['path']).read_text())['payload']['recipe']['sha256']
        assert 'readiness' not in answer


@pytest.mark.parametrize('changed_recipe', [False, True])
def test_completed_continuation_retires_old_recipe_claims_without_changing_earned_state(tmp_path, changed_recipe, monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw
    from natlang_neuralese.train.recipe import recipe_identity_sha256
    root, config_path, request, result, controller, plan, recipe = fixture(tmp_path)
    config = json.loads(config_path.read_text())
    # Only claim invalidation is under test. This bound historical metadata is
    # never passed off as a qualified foundation or exposed as new training.
    receipt = tmp_path / 'historical-claim.json'
    write(receipt, {'recipe': {'sha256': recipe_identity_sha256(recipe)}})
    reference = adapt.binding(receipt)
    cursor = {'cycle': 4, 'phase': 'sft', 'next_stage': 0,
              'foundation_candidate': reference, 'readiness': reference,
              'foundation_qualified': True,
              'foundation_qualification_attempt': {'request': {'sha256': 'a' * 64}, 'status': 'passed'}}
    before_cursor = copy.deepcopy(cursor)
    checkpoint_files = {str(p): bridge.binding(p) for p in root.rglob('*.pt')}
    if changed_recipe:
        definition = json.loads(recipe.read_text())
        definition['stages'][1]['parameters']['learning_rate'] = 1e-5
        future = tmp_path / 'future.json'; write(future, definition)
        config['production_recipe'] = str(future)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('claim invalidation cannot decode or rewrite optimizer state'))
    assert raw._retire_recipe_readiness(config, cursor) is changed_recipe
    if changed_recipe:
        assert not any(k in cursor for k in ('foundation_candidate', 'readiness', 'foundation_qualified', 'foundation_qualification_attempt'))
        assert cursor['superseded_recipe_readiness'][0]['receipts'] == {
            'foundation_candidate': reference, 'readiness': reference}
        assert cursor['foundation_qualification_history']['a' * 64] == before_cursor['foundation_qualification_attempt']
        assert (cursor['cycle'], cursor['phase'], cursor['next_stage']) == (4, 'sft', 0)
        assert raw._retire_recipe_readiness(config, cursor) is False
    else:
        assert cursor == before_cursor
    assert checkpoint_files == {str(p): bridge.binding(p) for p in root.rglob('*.pt')}
    assert adapt.binding(receipt) == reference
