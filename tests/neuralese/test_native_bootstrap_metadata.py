"""No persistent tensor cache: exact unchanged bootstrap avoids model unpickling."""
import json
from pathlib import Path
import pytest
from test_native_raw_handoff import actual_drain,drained
from natlang_neuralese.rl import native_raw_rounds as raw,production
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.common.jsonio import write_json_durable as write


def test_unchanged_config_rechecks_sources_and_ownership_without_deserializing_parent(actual_drain,tmp_path,monkeypatch):
    handoff=drained(actual_drain);record=json.loads(Path(handoff['path']).read_text())
    intent=json.loads(Path(record['intent']['path']).read_text())
    runtime=tmp_path/'ts-host';(runtime/'dist/teacher').mkdir(parents=True)
    for p,value in [('dist/teacher/cli.js','// runtime'),('prelude.js','// prelude'),('package.json','{}'),
        ('../package.json','{}'),('../package-lock.json','{}'),('../training/source-reviews/holds.jsonl',''),
        ('../training/source-reviews/family-holds.json','{}'),('../training/source-reviews/datasets.json','{}')]:
        path=runtime/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    refs={}
    for name in ('train','held','recipe','cases'):
        p=tmp_path/name;p.write_text(json.dumps({'id':name,'split':'train'})+'\n');refs[name]=raw.binding(p)
    declaration={'model':'fixture','production_stage':'sft','runtime':str(runtime),'base':str(tmp_path/'base'),'port':8097,
        'cases':refs['cases'],'production_recipe':refs['recipe'],'seed_native_graphs':refs['train'],'held_native_graphs':refs['held'],
        'handoff_requirement':{'model_id':'fixture:'+'a'*64,'recipe_sha256':refs['recipe']['sha256'],
            'trainer_launch':intent['source_launch'],'trainer_exit':record['source_terminal']['path']}}
    plan=tmp_path/'plan.json';write(plan,{'models':[declaration]})
    def configure(args):
        return {'out':str(args.out),'checkpoint':str(args.checkpoint),'production_recipe':str(args.production_recipe),
            'production_stage':args.production_stage,
            'seed_native_graphs':str(args.seed_native_graphs),'held_native_graphs':str(args.held_native_graphs),
            'trainer_parameters':{},'seed_trainer_parameters':{}}
    monkeypatch.setattr(production,'configure_native',configure)
    actual=CheckpointPolicy.load;calls=[]
    def load(path,*args,**kwargs):calls.append(path);return actual(path,*args,**kwargs)
    monkeypatch.setattr(CheckpointPolicy,'load',load)
    out=tmp_path/'configured'
    path=raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==1 and (out/'bootstrap-verification.json').is_file()
    assert raw.configure_model(plan,declaration,handoff,out)==path
    assert len(calls)==1
    # Host configuration tamper invalidates reuse and re-runs the full verifier.
    value=json.loads(path.read_text());value['operational_note']='changed'
    value['trainer_parameters']['fp8_training']='triton-experts'
    value['seed_trainer_parameters']={}
    value['seed_command']=['python','seed','{trainer_args}']
    write(path,value)
    raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==2
    repaired=json.loads(path.read_text())
    assert repaired['seed_command']==['python','seed','{seed_trainer_args}']
    assert repaired['seed_trainer_parameters']['fp8_training']=='triton-experts'
    assert repaired['seed_trainer_parameters']['refresh_at_checkpoint'] is False
    seed=Path(refs['train']['path']);original=seed.read_bytes();seed.write_text('changed graph bytes')
    with pytest.raises(ValueError,match='changed'):raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==3  # Changed graph bytes trigger the complete original verifier.
    seed.write_bytes(original)
    raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==3
    Path(refs['cases']['path']).write_text('{"id":"changed","split":"test"}\n')
    with pytest.raises(ValueError,match='changed'):raw.configure_model(plan,declaration,handoff,out)
    assert len(calls)==3  # No tensor read is required to reject changed task bytes.


def test_hot_task_plan_keeps_latest_committed_resident_sources(actual_drain,tmp_path,monkeypatch):
    from test_native_resident import setup
    from natlang_neuralese.train import native_resident as resident
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    root=tmp_path/'source';root.mkdir()
    paths,resident_dir,manager,new_recipe=setup(root)
    old_recipe=Path(manager.recipe_binding['path'])
    resident.prepare(resident_dir,new_recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert manager.poll(7) is not None
    adopted=resident.current(resident_dir)
    snapshot=tmp_path/'immutable-resident';snapshot.mkdir()
    (snapshot/'current.json').write_bytes((resident_dir/'current.json').read_bytes())
    handoff=drained(actual_drain);record=json.loads(Path(handoff['path']).read_text())
    intent=json.loads(Path(record['intent']['path']).read_text())
    runtime=tmp_path/'runtime';(runtime/'dist/teacher').mkdir(parents=True)
    (runtime/'dist/teacher/cli.js').write_text('// test-only collector')
    cases=tmp_path/'cases';cases.write_text('{"id":"one","split":"train"}\n')
    declaration={'model':'fixture','production_stage':'sft','runtime':str(runtime),'base':str(tmp_path/'base'),'port':8097,
        'cases':raw.binding(cases),'production_recipe':raw.binding(old_recipe),
        'seed_native_graphs':raw.binding(paths['old']),'held_native_graphs':raw.binding(paths['held']),
        'handoff_requirement':{'model_id':'fixture:'+'a'*64,'recipe_sha256':manager.recipe_binding['sha256'],
            'stage_ids':['sft'],'trainer_launch':intent['source_launch'],'trainer_exit':record['source_terminal']['path']}}
    seen=[]
    def configure(args):
        seen.append(args)
        return {'out':str(args.out),'checkpoint':str(args.checkpoint),'production_recipe':str(args.production_recipe),
            'production_stage':args.production_stage,
            'seed_native_graphs':str(args.seed_native_graphs),'held_native_graphs':str(args.held_native_graphs),
            'trainer_parameters':{},'seed_trainer_parameters':{}}
    monkeypatch.setattr(production,'configure_native',configure)
    plan=tmp_path/'plan.json';write(plan,{'models':[declaration]})
    out=tmp_path/'configured';config_path=raw.configure_model(plan,declaration,handoff,out)
    assert seen[-1].seed_native_graphs==paths['old']
    write(out/'cycle.json',{'phase':'rl','cycle':1,'resident_current':raw.binding(snapshot/'current.json')})
    second=tmp_path/'next-cases';second.write_text('{"id":"second","split":"train"}\n')
    declaration={**declaration,'cases':raw.binding(second)};write(plan,{'models':[declaration]})
    raw.configure_model(plan,declaration,handoff,out)
    updated=json.loads(config_path.read_text())
    assert updated['production_recipe']==adopted['payload']['recipe']['path']
    assert updated['seed_native_graphs']==adopted['payload']['train']['path']
    assert seen[-1].cases==second and seen[-1].seed_native_graphs==paths['new']
    assert updated['recipe_lineage_sha256']==manager.recipe_binding['lineage_sha256']


def test_seed_quantum_emits_full_backend_rates_and_explicit_objective_overrides(monkeypatch):
    from natlang_neuralese.train.recipe import stage_parameter_args
    parameters={'learning_rate':3e-6,'checkpoint_layers':True,'fp8_training':'triton-experts',
        'fp8_recipe':'current','native_linear_precision':'bf16x6','checkpoint_input_offload':True}
    config={'production_stage':'native_full_sft',
        'trainer_parameters':parameters.copy(),'seed_trainer_parameters':{'learning_rate':0.0003},
        'seed_command':['python','-m','natlang_neuralese.train.native_state_sft','{trainer_args}'],
        'learner_command':['python','learner','{trainer_args}']}
    original=dict(parameters)
    raw._seed_quantum(config,{'resident':{'directory':'/actual/resident'}})
    actual=[]
    monkeypatch.setattr(production.subprocess,'run',lambda argv,**kw:actual.append(argv))
    production.run_command(config['seed_command'],{
        'trainer_args':stage_parameter_args(config['trainer_parameters']),
        'seed_trainer_args':stage_parameter_args(config['seed_trainer_parameters'])})
    argv=actual[0]
    for key,value in {**parameters,'learning_rate':0.0003,'resident_data_directory':'/actual/resident'}.items():
        flag='--'+key.replace('_','-')
        assert flag in argv
        if value is not True:assert argv[argv.index(flag)+1]==str(value)
    assert '--no-refresh-at-checkpoint' in argv
    assert argv[argv.index('--stage-id')+1]=='native_full_sft'
    assert config['trainer_parameters']==original
    assert config['learner_command']==['python','learner','{trainer_args}']
    before=json.dumps(config,sort_keys=True);raw._seed_quantum(config,{'resident':{'directory':'/actual/resident'}})
    assert json.dumps(config,sort_keys=True)==before


def test_seed_emitted_stage_initializes_actual_resident_and_mandatory_keeps_own_stage(tmp_path,monkeypatch):
    from test_native_resident import setup
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.train.native_state_sft import build_parser
    from natlang_neuralese.train.recipe import stage_parameter_args
    from natlang_neuralese.rl.native_supervised import continuation_config
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,_=setup(tmp_path)
    config={'production_stage':'sft','production_recipe':manager.recipe_binding['path'],
        'recipe_lineage_sha256':manager.recipe_binding['lineage_sha256'],
        'trainer_parameters':{'learning_rate':3e-6,'checkpoint_layers':True},
        'seed_trainer_parameters':{'stage_id':'obsolete-unrelated-stage'},
        'seed_command':['python','-m','natlang_neuralese.train.native_state_sft',
            '--native-graphs',str(paths['old']),'--held-native-graphs',str(paths['held']),
            '--out',str(tmp_path/'seed'),'{trainer_args}']}
    raw._seed_quantum(config,{'resident':{'directory':str(directory)}})
    emitted=[]
    monkeypatch.setattr(production.subprocess,'run',lambda argv,**kw:emitted.append(argv))
    production.run_command(config['seed_command'],{
        'seed_trainer_args':stage_parameter_args(config['seed_trainer_parameters'])})
    args=build_parser().parse_args(emitted[0][3:])
    assert args.stage_id=='sft' and not args.refresh_at_checkpoint
    resumed=resident.ResidentAdoption(args.resident_data_directory,stage_id=args.stage_id,
        recipe_binding=manager.recipe_binding,train=manager.train,held=manager.held,
        geometry_qualification=directory.parent.parent/'geometry-capacity.json')
    assert resumed.initial['stage_id']=='sft'
    assert resumed.train is manager.train and resumed.held is manager.held
    assert config['trainer_parameters']=={'learning_rate':3e-6,'checkpoint_layers':True}
    recipe=json.loads(Path(config['production_recipe']).read_text())
    other={**recipe['stages'][0],'id':'other-mandatory-stage'}
    mandatory,_=continuation_config(config,other,checkpoint='same-current-parent',adapter=None,
        out=tmp_path/'mandatory')
    # Mandatory execution expands trainer_parameters into BOTH placeholders;
    # the ordinary seed's stage must not override the selected mandatory stage.
    mandatory_args=build_parser().parse_args(['--native-graphs',str(paths['old']),
        '--held-native-graphs',str(paths['held']),'--out',str(tmp_path/'mandatory'),
        *stage_parameter_args(mandatory['trainer_parameters'])])
    assert mandatory_args.stage_id=='other-mandatory-stage'
    assert mandatory['seed_trainer_parameters']['stage_id']=='sft'


@pytest.mark.parametrize('field',['rows_per_round','supervised_quantum_updates'])
@pytest.mark.parametrize('value',[0,-1,True,1.5,'4',None])
def test_raw_scheduling_controls_reject_invalid_before_handoff(field,value,tmp_path,monkeypatch):
    monkeypatch.setattr(raw,'verify_handoff',lambda *a,**kw:pytest.fail('invalid control must fail before checkpoint load'))
    with pytest.raises(ValueError,match=field+' must be a positive integer'):
        raw.configure_model(tmp_path/'plan',{field:value},{},tmp_path/'out')


def test_supervised_quantum_changes_only_checkpoint_boundary_in_real_emitted_args(tmp_path,monkeypatch):
    from natlang_neuralese.train.recipe import stage_parameter_args
    plan=tmp_path/'plan';write(plan,{'supervised_quantum_updates':16,'rows_per_round':4})
    original={'learning_rate':3e-6,'epochs':3,'checkpoint_every':250,'eval_every':250,
        'held_eval_tokens':8192,'held_eval_graphs':8,'native_linear_precision':'bf16x6',
        'fp8_training':'triton-experts','fp8_recipe':'current','checkpoint_layers':True}
    config={'production_stage':'native_full_sft','trainer_parameters':original.copy(),'seed_trainer_parameters':{},
        'seed_command':['python','seed','{trainer_args}'],'rows_per_round':32}
    declaration={'rows_per_round':4,'supervised_quantum_updates':16}
    raw._scheduling_controls(config,declaration,plan,{},pending=True)
    assert config['rows_per_round']==32  # Current selection remains the old packet.
    mandatory=original.copy();raw._supervised_quantum(config,mandatory)
    assert mandatory=={**original,'checkpoint_every':16,'eval_every':16,'refresh_at_checkpoint':True,'refresh_after_updates':16}
    actual=[];monkeypatch.setattr(production.subprocess,'run',lambda argv,**kw:actual.append(argv))
    production.run_command(config['seed_command'],{'seed_trainer_args':stage_parameter_args(config['seed_trainer_parameters'])})
    assert actual[0][actual[0].index('--checkpoint-every')+1]=='16'
    assert actual[0][actual[0].index('--epochs')+1]=='3'
    assert actual[0][actual[0].index('--eval-every')+1]=='16'
    assert actual[0][actual[0].index('--held-eval-tokens')+1]=='8192'
    assert '--refresh-at-checkpoint' in actual[0]
    assert actual[0][actual[0].index('--refresh-after-updates')+1]=='16'
    assert config['trainer_parameters']==original
    raw._scheduling_controls(config,declaration,plan,{},pending=False)
    assert config['rows_per_round']==4


def test_quantum_emitted_full_settings_pass_actual_trainer_early_validation(tmp_path,monkeypatch):
    import random,torch
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train.recipe import stage_parameter_args
    parameters={'checkpoint_every':250,'eval_every':250,'held_eval_tokens':8192,'held_eval_graphs':8,
        'learning_rate':3e-6,'native_linear_precision':'bf16x6','fp8_training':'triton-experts',
        'fp8_recipe':'current','checkpoint_layers':True,'checkpoint_input_offload':True}
    raw._supervised_quantum({'supervised_quantum_updates':16},parameters)
    class ValidatedBeforeAnyModelLoad(Exception):pass
    def stop(*args,**kw):raise ValidatedBeforeAnyModelLoad()
    monkeypatch.setattr(trainer,'CheckpointPolicy',stop)
    argv=['--native-graphs','unused','--held-native-graphs','unused','--out',str(tmp_path/'out'),
          *stage_parameter_args(parameters)]
    state=random.getstate()
    try:
        with torch.random.fork_rng(),pytest.raises(ValidatedBeforeAnyModelLoad):trainer.main(argv)
    finally:random.setstate(state)
    incompatible={'eval_every':10}
    with pytest.raises(ValueError,match='shorter held evaluation cadence'):
        raw._supervised_quantum({'supervised_quantum_updates':16},incompatible)


def test_raw_storage_save_does_not_force_yield_or_change_real_trainer_arguments():
    from natlang_neuralese.train.native_state_sft import build_parser
    from natlang_neuralese.train.recipe import stage_parameter_args
    original={'stage_id':'ordinary','epochs':3,'checkpoint_every':1000,
        'checkpoint_seconds':3600,'refresh_at_checkpoint':True,'learning_rate':1e-5}
    parameters=dict(original)
    raw._supervised_quantum({'production_stage':'ordinary'},parameters)
    assert parameters=={**original,'refresh_at_checkpoint':False}
    args=build_parser().parse_args(['--native-graphs','train','--held-native-graphs','held',
        '--out','learner',*stage_parameter_args(parameters)])
    assert not args.refresh_at_checkpoint and args.refresh_after_updates is None
    assert args.checkpoint_every==1000 and args.checkpoint_seconds==3600
    assert args.epochs==3 and args.learning_rate==1e-5
    # An independently explicit relative boundary remains available even when
    # no controller override is supplied; it never turns a timed save into a yield.
    parameters['refresh_after_updates']=750
    raw._supervised_quantum({'production_stage':'ordinary'},parameters)
    args=build_parser().parse_args(['--native-graphs','train','--held-native-graphs','held',
        '--out','learner',*stage_parameter_args(parameters)])
    assert args.refresh_after_updates==750 and not args.refresh_at_checkpoint


def test_explicit_actor_runtime_reconciles_known_role_binder_command_only(tmp_path):
    from natlang_neuralese.rl.native_rounds import configure_provider
    actor=tmp_path/'new-actor';actor.mkdir()
    entry=actor/'scripts/neuralese_data/serve_native_checkpoint.py';entry.parent.mkdir(parents=True);entry.write_text('# actor')
    package=actor/'training/neuralese/natlang_neuralese';package.mkdir(parents=True);(package/'__init__.py').write_text('# package')
    assets=actor/'training/prompt-pieces';assets.mkdir(parents=True)
    for asset in ('pieces.json','rendering-contract-fixtures.json'):(assets/asset).write_text('{}')
    environment={'NATLANG_REPO':str(actor),'PYTHONPATH':str(actor/'training/neuralese')+':'+str(actor)}
    old=['/usr/bin/env','NATLANG_REPO=/old-learner','PYTHONPATH=/old-learner/training/neuralese',
         '/usr/bin/python','/old-learner/scripts/neuralese_data/serve_native_checkpoint.py','--port','8097']
    declaration={'actor_runtime':str(actor),'commands':{'actor_command':old.copy()}}
    config={'actor_environment':environment,'learner_command':['unchanged-learner']}
    actual=configure_provider(config,declaration,{}, {},tmp_path/'config.json',write=False)
    assert actual['actor_command'][1:3]==['NATLANG_REPO='+str(actor),'PYTHONPATH='+environment['PYTHONPATH']]
    assert actual['actor_command'][4]==str(actor/'scripts/neuralese_data/serve_native_checkpoint.py')
    assert actual['learner_command']==['unchanged-learner']
    # The operator's original immutable declaration remains unchanged.
    assert declaration['commands']['actor_command']==old
    with pytest.raises(ValueError,match='custom actor command'):
        configure_provider({'actor_environment':environment},
            {'actor_runtime':str(actor),'commands':{'actor_command':['custom-actor']}}, {}, {},tmp_path/'unused',write=False)
    rebound=configure_provider({'actor_environment':{**environment,'NATLANG_REPO':'/wrong'},'actor_command':old.copy()},
        {'actor_runtime':str(actor)}, {}, {},tmp_path/'rebound.json',write=False)
    assert rebound['actor_environment']==environment
    assert rebound['actor_command'][4]==str(entry)


@pytest.mark.parametrize('command',[
    ['custom-actor','/old/scripts/neuralese_data/serve_native_checkpoint.py'],
    ['/bin/sh','-c','/old/scripts/neuralese_data/serve_native_checkpoint.py'],
    ['/usr/bin/env','NATLANG_REPO=/old','custom-actor','/old/scripts/neuralese_data/serve_native_checkpoint.py'],
    ['/usr/bin/python','-c','/old/scripts/neuralese_data/serve_native_checkpoint.py'],
])
def test_actor_runtime_never_reinterprets_custom_prefix_with_native_path(tmp_path,command):
    from natlang_neuralese.rl.native_rounds import configure_provider
    entry=tmp_path/'scripts/neuralese_data/serve_native_checkpoint.py';entry.parent.mkdir(parents=True);entry.write_text('# actor')
    package=tmp_path/'training/neuralese/natlang_neuralese';package.mkdir(parents=True);(package/'__init__.py').write_text('# package')
    assets=tmp_path/'training/prompt-pieces';assets.mkdir(parents=True)
    for asset in ('pieces.json','rendering-contract-fixtures.json'):(assets/asset).write_text('{}')
    with pytest.raises(ValueError,match='custom actor command'):
        configure_provider({}, {'actor_runtime':str(tmp_path),'commands':{'actor_command':command}},
                           {},{},tmp_path/'unused',write=False)


@pytest.mark.parametrize('prefix',[['/usr/bin/python'],['/usr/bin/python','-u'],
    ['/usr/bin/env','OMP_NUM_THREADS=2','/usr/bin/python','-B','-u']])
def test_actor_runtime_accepts_declared_python_launcher_shapes(tmp_path,prefix):
    from natlang_neuralese.rl.native_rounds import configure_provider
    entry=tmp_path/'scripts/neuralese_data/serve_native_checkpoint.py';entry.parent.mkdir(parents=True);entry.write_text('# actor')
    package=tmp_path/'training/neuralese/natlang_neuralese';package.mkdir(parents=True);(package/'__init__.py').write_text('# package')
    assets=tmp_path/'training/prompt-pieces';assets.mkdir(parents=True)
    for asset in ('pieces.json','rendering-contract-fixtures.json'):(assets/asset).write_text('{}')
    environment={'NATLANG_REPO':str(tmp_path),'PYTHONPATH':str(tmp_path/'training/neuralese')}
    command=prefix+['/old/scripts/neuralese_data/serve_native_checkpoint.py','--port','8097']
    result=configure_provider({'actor_environment':environment},
        {'actor_runtime':str(tmp_path),'commands':{'actor_command':command}}, {},{},tmp_path/'unused',write=False)
    assert result['actor_command'][-3:]==[str(tmp_path/'scripts/neuralese_data/serve_native_checkpoint.py'),'--port','8097']


@pytest.mark.parametrize('model',['lfm','mellum'])
def test_actor_runtime_retargets_default_factory_command_and_rejects_stale_identity(tmp_path,model):
    from natlang_neuralese.rl.production import (actor_runtime_configuration,
        bind_actor_runtime_command,validate_actor_runtime_configuration)
    from natlang_neuralese.rl.native_rounds import configure_provider
    roots=[]
    for name in ('r22','r26'):
        root=tmp_path/f'{model}-{name}'
        entry=root/'scripts/neuralese_data/serve_native_checkpoint.py';entry.parent.mkdir(parents=True);entry.write_text('# '+name)
        package=root/'training/neuralese/natlang_neuralese';package.mkdir(parents=True);(package/'__init__.py').write_text('# '+name)
        assets=root/'training/prompt-pieces';assets.mkdir(parents=True)
        for asset in ('pieces.json','rendering-contract-fixtures.json'):(assets/asset).write_text('{}')
        roots.append(root)
    old,new=roots
    command=['/usr/bin/env','NATLANG_REPO='+str(old),'PYTHONPATH='+str(old/'training/neuralese')+':'+str(old),
        '/usr/bin/python','-u',str(old/'scripts/neuralese_data/serve_native_checkpoint.py'),
        '--port','8123','--provider-arg','preserve-me']
    config={'model':model,'actor_command':command,**actor_runtime_configuration(old)}
    rebound=configure_provider(config,{'actor_runtime':str(new)}, {},{},tmp_path/f'{model}.json',write=False)
    assert rebound['actor_command']==['/usr/bin/env','NATLANG_REPO='+str(new),
        'PYTHONPATH='+str(new/'training/neuralese')+':'+str(new),'/usr/bin/python','-u',
        str(new/'scripts/neuralese_data/serve_native_checkpoint.py'),'--port','8123','--provider-arg','preserve-me']
    assert rebound['actor_environment']==actor_runtime_configuration(new)['actor_environment']
    assert rebound['native_actor_python_runtime']==actor_runtime_configuration(new)['native_actor_python_runtime']
    validate_actor_runtime_configuration(rebound)
    stale={**rebound,'actor_command':command}
    with pytest.raises(ValueError,match='different checkouts'):
        validate_actor_runtime_configuration(stale)
    with pytest.raises(ValueError,match='native-checkpoint entry point'):
        bind_actor_runtime_command(['/usr/bin/python','-c','custom actor'],new)


@pytest.mark.parametrize('control',[None,[],{'plan':{'path':'old','sha256':'a'*64}}])
def test_completed_stale_or_malformed_scheduling_control_invalidates_cache_before_reuse(tmp_path,control):
    from natlang_neuralese.rl import native_bootstrap_metadata as metadata
    plan=tmp_path/'plan.json';write(plan,{'models':[]})
    config={'rl_scheduling_control':control}
    assert metadata.unchanged(tmp_path/'unused-cache.json',plan,
        {'completed_supervised_adaptation':{'path':'unused-proof','sha256':'b'*64}},None,config)is False
