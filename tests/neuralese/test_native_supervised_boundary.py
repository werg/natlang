"""A declared successor changes only the next unsealed operation."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from natlang_neuralese.rl import native_supervised_boundary as m,native_bootstrap_metadata
from natlang_neuralese.train import native_resident as res
from natlang_neuralese.common.jsonio import write_json_durable as write


def fixture(tmp_path,monkeypatch,*,append=False):
    from test_native_resident import setup
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,newsource=setup(tmp_path)
    oldplan=json.loads((directory.parent/'recipe-plan.json').read_text())
    oldrecipe=Path(manager.recipe_binding['path'])
    source=None
    if append:
        source=res.prepare(directory,newsource,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),publish=False)
        base=json.loads(newsource.read_text())
    else:base=json.loads(oldrecipe.read_text())
    updated=copy.deepcopy(base);updated['stages'][0]['parameters']['checkpoint_seconds']=900
    successor=tmp_path/'operational-recipe.json';write(successor,updated)
    root=tmp_path/'rounds';root.mkdir();production=root/'production';production.mkdir()
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'exact opaque full parent')
    config={'out':str(production),'production_recipe':str(oldrecipe),'production_stage':'sft',
        'resident_data_directory':str(directory),'recipe_lineage_sha256':oldplan['recipe_sha256'],
        'raw_model_id':'lfm:base','checkpoint':str(checkpoint),'seed_command':['trainer','{trainer_args}'],
        'native_actor_mode':'raw-state','seed_native_graphs':str(paths['old']),
        'held_native_graphs':str(paths['held'])}
    write(root/'config.json',config);cursor={'phase':'sft','cycle':4,'next_stage':1,'mandatory_debt':{'ordinary':8},'readiness':{'historical':True},'foundation_qualified':True}
    write(root/'cycle.json',cursor);write(production/'state.json',{'phase':'collect','checkpoint':str(checkpoint),'round':5})
    request={'schema':m.SCHEMA,'model_id':'lfm:base','parent_recipe':res.binding(oldrecipe),
        'recipe':res.binding(successor),'package':{'path':str(Path(res.__file__).parents[1]),'code':oldplan['code']}}
    if source:request['source_preparation']=source
    request_path=tmp_path/'continuation.json';write(request_path,request)
    declaration={'model':'lfm','production_stage':'sft','supervised_boundary_continuation':res.binding(request_path)}
    plan_path=tmp_path/'raw-plan.json';write(plan_path,{'models':[declaration]})
    monkeypatch.setattr(native_bootstrap_metadata,'save',lambda *args:None)
    return SimpleNamespace(**locals())


@pytest.mark.parametrize('append',[False,True])
def test_actual_prepared_source_and_operational_commit_preserve_parent_debt(tmp_path,monkeypatch,append):
    f=fixture(tmp_path,monkeypatch,append=append);before=Path(f.config['checkpoint']).read_bytes()
    config,cursor=m.apply(f.plan_path,f.declaration,{'path':'original-bootstrap'},f.config,f.cursor,f.root)
    assert config['production_recipe']==str(f.successor)
    assert config['seed_trainer_parameters']['checkpoint_seconds']==900
    assert cursor['next_stage']==1 and cursor['mandatory_debt']=={'ordinary':8}
    assert 'readiness'not in cursor and 'foundation_qualified'not in cursor
    assert Path(f.config['checkpoint']).read_bytes()==before
    assert not (Path(config['resident_data_directory'])/'initial.json').exists()
    plan=json.loads((Path(config['resident_data_directory']).parent/'recipe-plan.json').read_text())
    assert plan['recipe_sha256']==f.oldplan['recipe_sha256']
    assert plan['stage_inputs']['sft']['native_graphs']['sha256']==res.sha(f.paths['new']if append else f.paths['old'])
    assert m.recover_configuration(f.root/'config.json',f.plan_path,f.declaration,{'path':'original-bootstrap'})
    # A later earned cursor is never overwritten by committed recovery.
    cursor['next_stage']=2;write(f.root/'cycle.json',cursor)
    assert m.recover_configuration(f.root/'config.json',f.plan_path,f.declaration,{'path':'original-bootstrap'})
    assert json.loads((f.root/'cycle.json').read_text())['next_stage']==2


@pytest.mark.parametrize('change',['math','objective','source','stage'])
def test_undeclared_changes_rejected_before_config_commit(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);value=json.loads(f.successor.read_text())
    if change=='math':value['stages'][0]['parameters']['native_linear_precision']='bf16x6'
    elif change=='objective':value['stages'][0]['parameters']['epochs']=2.
    elif change=='source':value['input_bindings']['train']=res.binding(f.paths['new'])
    else:value['stages'][0]['id']='different'
    write(f.successor,value);f.request['recipe']=res.binding(f.successor);write(f.request_path,f.request)
    f.declaration['supervised_boundary_continuation']=res.binding(f.request_path);write(f.plan_path,{'models':[f.declaration]})
    with pytest.raises(ValueError,match='preserve'):
        m.apply(f.plan_path,f.declaration,{},f.config,f.cursor,f.root)
    assert json.loads((f.root/'config.json').read_text())==f.config


def test_crash_after_config_commit_recovers_same_intent_without_parent_decode(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);original=m.write_json_durable
    def fail_cursor(path,value):
        if Path(path)==f.root/'cycle.json':raise OSError('crash after config commit')
        return original(path,value)
    with monkeypatch.context()as patch:
        patch.setattr(m,'write_json_durable',fail_cursor)
        with pytest.raises(OSError,match='crash'):
            m.apply(f.plan_path,f.declaration,{},f.config,f.cursor,f.root)
    assert json.loads((f.root/'config.json').read_text())['production_recipe']==str(f.successor)
    assert m.recover_configuration(f.root/'config.json',f.plan_path,f.declaration,{})
    actual=json.loads((f.root/'cycle.json').read_text());assert actual['mandatory_debt']=={'ordinary':8}
    assert actual['supervised_boundary_continuation']['training_credit']==0


def test_sealed_operation_refuses_boundary_commit(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);write(f.production/'state.json',{'phase':'updating_mandatory_sft'})
    with pytest.raises(ValueError,match='sealed'):
        m.apply(f.plan_path,f.declaration,{},f.config,f.cursor,f.root)
    assert json.loads((f.root/'config.json').read_text())==f.config


def test_effective_default_task_policy_requires_native_transport(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);old=json.loads(f.successor.read_text());new=copy.deepcopy(old)
    new['stage_parameter_defaults']={'native_state_sft':{'task_eval':{'packet':{'path':'unused','sha256':'0'*64},'neuralese_server':False}}}
    with pytest.raises(ValueError,match='neuralese_server'):
        m._validate(old,new)


def test_live_declared_successor_is_read_between_actual_stage_commands(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw,native_rounds
    from natlang_neuralese.train.recipe import load_recipe,resolve_stage_inputs
    f=fixture(tmp_path,monkeypatch)
    old=json.loads(f.oldrecipe.read_text());warm=copy.deepcopy(old['stages'][0]);warm['id']='warm';warm['parameters']['checkpoint_seconds']=3600
    old['stages'].insert(0,warm);write(f.oldrecipe,old)
    plan=copy.deepcopy(f.oldplan);plan['recipe']=load_recipe(f.oldrecipe)
    plan['stage_inputs']={row['id']:resolve_stage_inputs(plan['recipe'],row,{})for row in old['stages']}
    write(f.directory.parent/'recipe-plan.json',plan)
    new=copy.deepcopy(old)
    for row in new['stages']:row['parameters']['checkpoint_seconds']=900
    write(f.successor,new);f.request.update(parent_recipe=res.binding(f.oldrecipe),recipe=res.binding(f.successor));write(f.request_path,f.request)
    declaration={k:v for k,v in f.declaration.items()if k!='supervised_boundary_continuation'};write(f.plan_path,{'models':[declaration]})
    write(f.root/'cycle.json',{'phase':'sft','cycle':4,'next_stage':0,'mandatory_debt':{'ordinary':8}})
    monkeypatch.setattr(raw,'configure_model',lambda *a,**k:f.root/'config.json')
    monkeypatch.setattr(raw,'_progress',lambda cp:{'stages':{}})
    commands=[]
    def job(path,config,out,stop):
        commands.append(copy.deepcopy(config['trainer_parameters']))
        directory=Path(config['out'])/'learner';directory.mkdir(parents=True,exist_ok=True)
        checkpoint=directory/'checkpoint.pt';checkpoint.write_bytes(b'actual-stage-output')
        write(directory/'native-training-receipt.json',{'checkpoint':res.binding(checkpoint),'foundation_ready':False,'completed':False})
        if config['production_stage']=='warm':
            write(f.plan_path,{'models':[{**declaration,'supervised_boundary_continuation':res.binding(f.request_path)}]})
        return {'pending_native_checkpoint':str(checkpoint)}
    monkeypatch.setattr(native_rounds,'round_job',job)
    result=raw.student_cycle(f.plan_path,declaration,{},f.root,stop=SimpleNamespace(requested=False))
    assert [(row['stage_id'],row.get('checkpoint_seconds'))for row in commands]==[('warm',3600),('sft',900)]
    assert result['mandatory_debt']=={'ordinary':8}and result['cycle']==5 and result['phase']=='sft'
    assert result['foundation_readiness_repair']['readiness_granted']is False


def test_fixed_pointer_selects_recipe_without_global_plan_rewrite(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);pointer=tmp_path/'current-continuation.json'
    declaration={k:v for k,v in f.declaration.items()if k!='supervised_boundary_continuation'}
    declaration['supervised_boundary_continuation_pointer']=str(pointer);write(f.plan_path,{'models':[declaration]});before=f.plan_path.read_bytes()
    assert m.apply(f.plan_path,declaration,{},f.config,f.cursor,f.root)==(f.config,f.cursor)
    write(pointer,res.binding(f.request_path))
    config,cursor=m.apply(f.plan_path,declaration,{},f.config,f.cursor,f.root)
    assert config['production_recipe']==str(f.successor)and f.plan_path.read_bytes()==before
    assert cursor['supervised_boundary_continuation']['training_credit']==0


@pytest.mark.parametrize('role',['lfm','other'])
def test_explicit_publication_role_python_runtime_and_roots(tmp_path,monkeypatch,role):
    from natlang_neuralese.model.native_checkpoint_control import publication_selection
    f=fixture(tmp_path,monkeypatch);package=Path(res.__file__).parents[1]
    f.request['publication']={'native_checkpoint_publication':publication_selection(namespace=str(tmp_path),readonly_input_roots=[str(package)]),
        'native_checkpoint_publication_owner_pointer':str(tmp_path/'publication/current-owner.json'),
        'native_checkpoint_publication_role':role,'native_checkpoint_publication_runtime_root':str(package.parents[2]),
        'native_checkpoint_publication_reserve_bytes':1024}
    write(f.request_path,f.request);f.declaration['supervised_boundary_continuation']=res.binding(f.request_path);write(f.plan_path,{'models':[f.declaration]})
    if role=='other':
        with pytest.raises(ValueError,match='role/Python runtime'):
            m.apply(f.plan_path,f.declaration,{},f.config,f.cursor,f.root)
        assert json.loads((f.root/'config.json').read_text())==f.config
    else:
        config,cursor=m.apply(f.plan_path,f.declaration,{},f.config,f.cursor,f.root)
        assert config['native_checkpoint_publication_runtime_root']==str(package.parents[2])
        assert config['native_checkpoint_publication_owner_pointer'].endswith('current-owner.json')
        assert config['seed_trainer_parameters']['checkpoint_publication']is True


def test_restart_consumes_exact_sealed_request_without_rewriting_advanced_state(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw,native_rounds,native_bridge
    job=tmp_path/'job';job.mkdir();parent=tmp_path/'parent';parent.write_bytes(b'parent')
    original={'out':str(job),'checkpoint':str(parent),'round_command':['real-provider'],
              'trainer_parameters':{'checkpoint_seconds':3600},'raw_foundation_qualification':False}
    write(job/'config.json',original)
    request=native_bridge.write_request(job/'bridge',kind='round',out=job,config=job/'config.json')
    before=request.read_bytes();advanced={'phase':'awaiting_semantic_refresh','pending_native_checkpoint':'earned'}
    write(job/'state.json',advanced)
    observed=[]
    def bridge(command,path,stop):
        observed.append((command,Path(path).read_bytes()))
        return {'state':native_bridge.binding(job/'state.json')}
    monkeypatch.setattr(native_rounds,'bridge_result',bridge)
    monkeypatch.setattr(native_rounds,'round_job',lambda *a:pytest.fail('sealed request must not be regenerated'))
    monkeypatch.setattr(raw,'_publish_supervised_result',lambda continuation,j,c,u,r:observed.append((continuation,r)))
    candidate=copy.deepcopy(original);candidate['trainer_parameters']['checkpoint_seconds']=900
    raw._supervised_job(candidate,job,{}, {},SimpleNamespace(requested=False))
    assert observed==[(['real-provider'],before),(original,advanced)]
    assert json.loads((job/'config.json').read_text())==original
    assert request.read_bytes()==before
    write(job/'config.json',candidate)
    with pytest.raises(ValueError,match='configuration changed'):raw._sealed_supervised_request(job)


def test_config_owned_selector_does_not_require_bootstrap_plan_mutation(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);pointer=tmp_path/'selector.json';write(pointer,res.binding(f.request_path))
    declaration={k:v for k,v in f.declaration.items()if k!='supervised_boundary_continuation'}
    write(f.plan_path,{'models':[declaration]});before=f.plan_path.read_bytes()
    f.config['supervised_boundary_continuation_pointer']=str(pointer);write(f.root/'config.json',f.config)
    config,cursor=m.apply(f.plan_path,declaration,{},f.config,f.cursor,f.root)
    assert config['production_recipe']==str(f.successor)
    assert f.plan_path.read_bytes()==before and cursor['mandatory_debt']=={'ordinary':8}


def test_sealed_request_selection_retains_failed_attempt_and_selects_earned_successor(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_raw_rounds as raw,native_bridge
    job=tmp_path/'job';job.mkdir();parent=tmp_path/'parent';parent.write_bytes(b'parent')
    original={'out':str(job),'checkpoint':str(parent)};write(job/'config.json',original)
    failed=native_bridge.write_request(job/'bridge',kind='round',out=job,config=job/'config.json')
    write(failed.with_suffix('.result.json'),{'schema':native_bridge.RESULT_SCHEMA,'kind':'round','request_sha256':res.sha(failed),'request_id':json.loads(failed.read_text())['request_id'],'status':'failed','artifacts':[]})
    successor=native_bridge.write_request(job/'bridge',kind='round',out=job,config=job/'config.json')
    assert failed!=successor and raw._sealed_supervised_request(job)==(successor,original)
    assert failed.exists()and failed.with_suffix('.result.json').exists()
    other=native_bridge.write_request(job/'bridge',kind='round',out=job,config=job/'config.json')
    with pytest.raises(ValueError,match='ambiguous'):raw._sealed_supervised_request(job)
    assert raw._sealed_supervised_request(job,res.binding(successor))==(successor,original)
    with pytest.raises(ValueError):raw._sealed_supervised_request(job,res.binding(parent))
