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
    assert repaired['seed_trainer_parameters']['refresh_at_checkpoint'] is True
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
    config={'trainer_parameters':parameters.copy(),'seed_trainer_parameters':{'learning_rate':0.0003},
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
    assert '--refresh-at-checkpoint' in argv
    assert config['trainer_parameters']==original
    assert config['learner_command']==['python','learner','{trainer_args}']
    before=json.dumps(config,sort_keys=True);raw._seed_quantum(config,{'resident':{'directory':'/actual/resident'}})
    assert json.dumps(config,sort_keys=True)==before


@pytest.mark.parametrize('field',['rows_per_round','supervised_quantum_updates'])
@pytest.mark.parametrize('value',[0,-1,True,1.5,'4',None])
def test_raw_scheduling_controls_reject_invalid_before_handoff(field,value,tmp_path,monkeypatch):
    monkeypatch.setattr(raw,'verify_handoff',lambda *a,**kw:pytest.fail('invalid control must fail before checkpoint load'))
    with pytest.raises(ValueError,match=field+' must be a positive integer'):
        raw.configure_model(tmp_path/'plan',{field:value},{},tmp_path/'out')


def test_supervised_quantum_changes_only_checkpoint_boundary_in_real_emitted_args(tmp_path,monkeypatch):
    from natlang_neuralese.train.recipe import stage_parameter_args
    plan=tmp_path/'plan';write(plan,{'supervised_quantum_updates':16,'rows_per_round':4})
    original={'learning_rate':3e-6,'epochs':3,'checkpoint_every':250,'native_linear_precision':'bf16x6'}
    config={'trainer_parameters':original.copy(),'seed_trainer_parameters':{},
        'seed_command':['python','seed','{trainer_args}'],'rows_per_round':32}
    declaration={'rows_per_round':4,'supervised_quantum_updates':16}
    raw._scheduling_controls(config,declaration,plan,{},pending=True)
    assert config['rows_per_round']==32  # Current selection remains the old packet.
    mandatory=original.copy();raw._supervised_quantum(config,mandatory)
    assert mandatory=={**original,'checkpoint_every':16,'refresh_at_checkpoint':True}
    actual=[];monkeypatch.setattr(production.subprocess,'run',lambda argv,**kw:actual.append(argv))
    production.run_command(config['seed_command'],{'seed_trainer_args':stage_parameter_args(config['seed_trainer_parameters'])})
    assert actual[0][actual[0].index('--checkpoint-every')+1]=='16'
    assert actual[0][actual[0].index('--epochs')+1]=='3'
    assert '--refresh-at-checkpoint' in actual[0]
    assert config['trainer_parameters']==original
    raw._scheduling_controls(config,declaration,plan,{},pending=False)
    assert config['rows_per_round']==4
