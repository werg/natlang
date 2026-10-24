"""Actual owner drains authorize metadata changes without rewinding a raw cursor.

Small protocol checkpoint fixtures test authority/durability, not model quality;
actual native_stage optimizer/RNG update coverage is exercised alongside these.
"""
import copy
import json
import shutil
import time
from pathlib import Path
import pytest
import torch
from test_native_raw_handoff import actual_drain,drained
from test_native_resident import setup
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_runtime_handoff as runtime
from natlang_neuralese.rl import native_raw_handoff as handoff,native_raw_rounds as raw
from natlang_neuralese.train import native_resident as resident


def _step(model,optimizer):
    from natlang_neuralese.model.shared_state import Invocation,compile_graph
    optimizer.zero_grad(set_to_none=True)
    plan=compile_graph([Invocation('answer',(4,5,6))])
    model.logits(model(plan)['h_final']).float().square().mean().backward()
    optimizer.step();optimizer.zero_grad(set_to_none=True)


def _actual_state(saved,model,optimizer,config):
    from natlang_neuralese.model.native_model import model_state
    from natlang_neuralese.train.loop import capture_training_rng_state
    saved.update(model_state(model));saved['model_config']=config
    saved['training']['optimizer']=optimizer.state_dict()
    saved['training']['rng']=capture_training_rng_state('cpu')


@pytest.fixture
def later(tmp_path,monkeypatch,request):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    family=getattr(request,'param',None)
    model=optimizer=model_config=None
    if isinstance(family,str):
        from test_native_full_updates import source
        from natlang_neuralese.model.native_model import load_native_model
        from natlang_neuralese.train.native_state_sft import make_native_optimizer
        dtype=source(tmp_path/'base',family)
        model,_,_,model_config=load_native_model(tmp_path/'base',dtype=dtype,rank=4)
        optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=False)
        _step(model,optimizer)
    first=tmp_path/'bootstrap';first.mkdir();gen=actual_drain.__wrapped__(first)
    try:
        firstvalue=next(gen)
        if model is not None:
            firstsaved=torch.load(firstvalue[3],weights_only=False)
            _actual_state(firstsaved,model,optimizer,model_config)
            torch.save(firstsaved,firstvalue[3]);shutil.copyfile(firstvalue[3],firstvalue[4])
        bootstrap=drained(firstvalue)
    finally:gen.close()
    source=tmp_path/'source';source.mkdir();paths,directory,manager,_=setup(source,fingerprint=model_config['tokenizer_fingerprint'] if model_config else 'tokenizer-sha')
    package=Path(resident.__file__).parents[1]
    shutil.copytree(package,directory.parent/'runtime/natlang_neuralese',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    newpackage=tmp_path/'reviewed/natlang_neuralese'
    shutil.copytree(package,newpackage,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    with (newpackage/'__init__.py').open('a') as stream:stream.write('\n# reviewed control-only successor\n')
    (directory.parent/'sft').mkdir()
    root=tmp_path/'rounds/lfm';out=root/'production';out.mkdir(parents=True)
    partial=getattr(request,'param',False) is True
    owner=tmp_path/'owner';owner.mkdir();gen=actual_drain.__wrapped__(owner)
    try:
        process,intent,terminal,slot,pin=next(gen)
        initial=json.loads((directory/'initial.json').read_text());initial['owner']=json.loads(Path(intent['path']).read_text())['children'][0]
        write(directory/'initial.json',initial)
        saved=torch.load(slot,weights_only=False)
        if model is not None:
            _step(model,optimizer)
            _actual_state(saved,model,optimizer,model_config)
        if partial:
            slot=root/'operations/round-7/learner/checkpoint.pt';slot.parent.mkdir(parents=True)
        saved['training'].update(global_step=19,exposure_counts={'a':3,'b':2},source_receipts=[resident.binding(paths['old']),resident.binding(paths['held'])],
            options={'recipe_binding':json.dumps(initial['recipe']),'stage_id':'sft'})
        from natlang_neuralese.train.native_continuation import _TRANSIENT
        saved['training']['sft_progress']['sft']={k:copy.deepcopy(v) for k,v in saved['training'].items() if k not in _TRANSIENT}
        if partial:saved['training']['options']['continuation']={'checkpoint_sha256':json.loads(Path(bootstrap['path']).read_text())['checkpoint']['sha256']}
        torch.save(saved,slot);shutil.copyfile(slot,pin)
        process.communicate();write(terminal,{'at':time.time(),'returncode':0})
        current=handoff.complete_drain(intent,resident.binding(terminal),resident.binding(pin),tmp_path/'later-drain.json',
            original_slot=resident.binding(slot),resident_directory=directory)
    finally:gen.close()
    cert=source/'new-capacity.json';write(cert,{'passed':True,'scope':'control fixture only'})
    recipe=copy.deepcopy(json.loads((directory.parent/'recipe-plan.json').read_text())['recipe'])
    recipe['stages'][0]['parameters']['resident_geometry_qualification']=str(cert)
    successor=source/'successor.json';write(successor,recipe)
    transition=handoff.prepare_drained_recipe_handoff(current,successor,tmp_path/'transition.json',package=newpackage)
    config={'out':str(out),'checkpoint':bootstrap['path'],'bootstrap_binding':{'plan_sha256':'original','handoff':bootstrap},
        'raw_model_id':(model_config['architecture']+':'+model_config['base_inventory_sha256']) if model_config else 'fixture:'+'a'*64,'recipe_lineage_sha256':initial['recipe']['lineage_sha256'],
        'production_recipe':initial['recipe']['path'],'production_stage':'sft','trainer_parameters':{},'seed_trainer_parameters':{},
        'seed_command':['python','-m','natlang_neuralese.train.native_state_sft','{seed_trainer_args}'],
        'cases':'sealed-cases','runtime':'sealed-runtime','actor_command':['sealed-actor'],
        'before_collect':['sealed-before'],'task_source':{'sealed':True},'source_admission':{'sealed':True},
        'native_state':True,'native_actor_mode':'raw-state','seed_native_graphs':str(paths['old']),
        'held_native_graphs':str(paths['held']),'unknown':{'retained':[1,2,3]}}
    configpath=root/'config.json';write(configpath,config)
    write(root/'cycle.json',{'phase':'sft','pending_rl_round':7,'unknown_cursor':{'debt':[4,5]}})
    # Latest already-published optimizer is the actual later drained full parent.
    published=json.loads(Path(bootstrap['path']).read_text())['checkpoint']['path'] if partial else str(pin)
    write(out/'state.json',{'checkpoint':published,'round':7,'trained_rounds':6,'selection':{'all':True}})
    cases=tmp_path/'cases.jsonl';cases.write_text('{"id":"private","split":"train"}\n')
    collector=tmp_path/'collector';(collector/'dist/teacher').mkdir(parents=True);(collector/'dist/teacher/cli.js').write_text('/* fixture */')
    declaration={'model':'lfm','production_stage':'sft','handoff_receipt':bootstrap,
        'cases':resident.binding(cases),'runtime':str(collector),
        'handoff_requirement':{'model_id':config['raw_model_id'],'recipe_sha256':initial['recipe']['sha256'],'stage_ids':['sft']},
        'commands':{'seed_command':['env','PYTHONPATH='+str(newpackage.parent),*config['seed_command']]}}
    operation=None
    if partial:
        from natlang_neuralese.rl import native_bridge as bridge
        job=slot.parent.parent;jobstate=job/'state.json'
        alias=job/'checkpoint-alias.pt';alias.symlink_to(slot)
        write(jobstate,{'checkpoint':published,'phase':'learner_pending','partial_checkpoint':str(alias)})
        jobconfig=tmp_path/'operation-config.json';write(jobconfig,{'checkpoint':published})
        req=bridge.write_request(root/'requests',kind='round',out=job,config=jobconfig,state=jobstate,device='cpu')
        reply=req.with_suffix('.result.json')
        write(reply,{'schema':bridge.RESULT_SCHEMA,'kind':'round','request_sha256':resident.binding(req)['sha256'],
            'request_id':bridge.read_request(req)['request_id'],'status':'stopped','state':resident.binding(jobstate),
            'checkpoint':resident.binding(slot),'artifacts':bridge.inventory(job)})
        operation={'request':resident.binding(req),'result':resident.binding(reply)}
    proof=runtime.prepare_runtime_handoff(bootstrap,current,transition,configpath,tmp_path/'runtime-proof.json',operation=operation)
    declaration['runtime_handoff']=proof
    globalplan=tmp_path/'bootstrap-plan.json';write(globalplan,{'models':[declaration]})
    return dict(bootstrap=bootstrap,handoff=current,transition=transition,config=configpath,root=root,out=out,
                declaration=declaration,plan=globalplan,proof=proof,pin=pin,saved=saved,newpackage=newpackage,operation=operation,slot=slot,model=model,optimizer=optimizer,model_config=model_config,directory=directory,paths=paths,manager=manager)


def test_later_actual_owner_rebinds_recipe_without_rebootstrap_or_cursor_reset(later):
    x=later;old=json.loads(x['config'].read_text())
    unchanged={p:p.read_bytes() for p in (x['root']/'cycle.json',x['out']/'state.json',x['pin'])}
    checked=runtime.verify_runtime_handoff(x['proof'],bootstrap=x['bootstrap'])
    assert checked['checkpoint']['sha256']==resident.binding(x['pin'])['sha256']
    result=raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    value=json.loads(result.read_text())
    for key in ('bootstrap_binding','checkpoint','cases','runtime','actor_command','before_collect','task_source','source_admission','unknown'):
        assert value[key]==old[key]
    assert all(p.read_bytes()==rawbytes for p,rawbytes in unchanged.items())
    transition=json.loads(Path(x['transition']['path']).read_text())
    assert value['production_recipe']==transition['recipe']['path']
    assert value['trainer_parameters']['recipe_binding']['sha256']==transition['recipe']['sha256']
    assert value['trainer_parameters']['recipe_binding']['lineage_sha256']==old['recipe_lineage_sha256']
    assert value['seed_trainer_parameters']['stage_id']=='sft'
    assert value['runtime_binding']['handoff']==old['bootstrap_binding']['handoff']
    assert raw._release_requirement(x['declaration'],{})['recipe_sha256']==transition['recipe']['sha256']
    assert raw._release_requirement(x['declaration'],{})['recipe_lineage_sha256']==old['recipe_lineage_sha256']
    assert runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],Path(x['proof']['path']))==x['proof']
    assert raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])==result
    assert all(p.read_bytes()==rawbytes for p,rawbytes in unchanged.items())
    restored=torch.load(x['pin'],weights_only=False)
    assert restored['training']['exposure_counts']=={'a':3,'b':2}
    assert torch.equal(restored['training']['optimizer']['moments'],x['saved']['training']['optimizer']['moments'])
    assert torch.equal(restored['training']['rng']['torch_rng'],x['saved']['training']['rng']['torch_rng'])


@pytest.mark.parametrize('pending',[True,False])
def test_actor_command_environment_and_runtime_move_only_at_free_boundary(later,tmp_path,pending):
    from natlang_neuralese.rl.production import actor_runtime_configuration
    x=later
    roots=[]
    for name in ('old-actor','new-actor'):
        root=tmp_path/name
        entry=root/'scripts/neuralese_data/serve_native_checkpoint.py'
        entry.parent.mkdir(parents=True);entry.write_text('# '+name+'\n')
        package=root/'training/neuralese/natlang_neuralese';package.mkdir(parents=True)
        (package/'__init__.py').write_text('# '+name+'\n')
        assets=root/'training/prompt-pieces';assets.mkdir(parents=True)
        for asset in ('pieces.json','rendering-contract-fixtures.json'):(assets/asset).write_text('{}')
        roots.append(root)
    old_actor,new_actor=roots
    config=json.loads(x['config'].read_text());config.update(actor_runtime_configuration(old_actor))
    write(x['config'],config)
    cursor=json.loads((x['root']/'cycle.json').read_text())
    if not pending:
        cursor['phase']='collect';cursor.pop('pending_rl_round',None)
        write(x['root']/'cycle.json',cursor)
    state=(x['out']/'state.json').read_bytes();cursor_bytes=(x['root']/'cycle.json').read_bytes()
    proof=runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],
        x['config'],tmp_path/'actor-runtime-handoff.json')
    declaration=copy.deepcopy(x['declaration']);declaration.update(runtime_handoff=proof,actor_runtime=str(new_actor))
    declaration['commands']['actor_command']=['env','NATLANG_REPO='+str(old_actor),
        'PYTHONPATH=stale-imports','python',str(old_actor/'scripts/neuralese_data/serve_native_checkpoint.py')]
    original=copy.deepcopy(declaration);write(x['plan'],{'models':[declaration]})
    runtime.configure_runtime_handoff(x['plan'],declaration,x['bootstrap'],x['config'])
    actual=json.loads(x['config'].read_text())
    assert declaration==original and (x['out']/'state.json').read_bytes()==state
    assert (x['root']/'cycle.json').read_bytes()==cursor_bytes
    expected=actor_runtime_configuration(old_actor if pending else new_actor)
    for key,value in expected.items():assert actual[key]==value
    if pending:
        assert actual['actor_command']==config['actor_command']
        assert actual['cases']==config['cases'] and actual['runtime']==config['runtime']
    else:
        assert actual['actor_command'][-1]==str(new_actor/'scripts/neuralese_data/serve_native_checkpoint.py')
        assert 'NATLANG_REPO='+str(new_actor) in actual['actor_command']
        assert 'PYTHONPATH='+expected['actor_environment']['PYTHONPATH'] in actual['actor_command']


def test_actor_runtime_factory_requires_real_entry_and_python_inventory(tmp_path):
    from natlang_neuralese.rl.production import actor_runtime_configuration
    with pytest.raises(ValueError,match='serving entry'):
        actor_runtime_configuration(tmp_path)
    entry=tmp_path/'scripts/neuralese_data/serve_native_checkpoint.py'
    entry.parent.mkdir(parents=True);entry.write_text('# real entry\n')
    with pytest.raises(ValueError,match='no source inventory'):
        actor_runtime_configuration(tmp_path)


@pytest.mark.parametrize('change',['config','cursor','state','bootstrap','old-package','drop'])
def test_later_authority_rejects_changed_control_or_forged_origin(later,change):
    x=later
    if change in ('config','cursor','state'):
        path={'config':x['config'],'cursor':x['root']/'cycle.json','state':x['out']/'state.json'}[change]
        value=json.loads(path.read_text());value['unauthorized']='change';write(path,value)
    elif change=='bootstrap':x['bootstrap']=x['handoff']
    elif change=='old-package':x['declaration']['commands']['seed_command']=json.loads(x['config'].read_text())['seed_command']
    else:
        raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
        x['declaration'].pop('runtime_handoff')
    with pytest.raises(ValueError):raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])


def test_unpublished_partial_requires_existing_verified_native_operation(later,tmp_path):
    x=later;state=x['out']/'state.json';value=json.loads(state.read_text());value['checkpoint']=json.loads(Path(x['bootstrap']['path']).read_text())['checkpoint']['path'];write(state,value)
    with pytest.raises(ValueError,match='verified partial required'):
        runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],tmp_path/'unproven.json')


@pytest.mark.parametrize('later',[True],indirect=True)
def test_verified_partial_survives_same_operation_slot_and_state_reuse(later):
    x=later
    proof=runtime.verify_runtime_handoff(x['proof'],bootstrap=x['bootstrap'])
    assert proof['operation']['original']==x['operation']
    state=x['slot'].parent.parent/'state.json'
    state.write_text('{"later_operation_state":true}\n')
    x['slot'].write_bytes(b'later legitimate mutable slot')
    alias=x['slot'].parent.parent/'checkpoint-alias.pt'
    alias.unlink();alias.symlink_to(x['pin'])
    assert runtime.verify_runtime_handoff(x['proof'],bootstrap=x['bootstrap'])==proof
    before=(x['out']/'state.json').read_bytes()
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    assert (x['out']/'state.json').read_bytes()==before


def test_retry_after_config_commit_metadata_failure_preserves_original_cursor(later,monkeypatch):
    x=later
    from natlang_neuralese.rl import native_bootstrap_metadata as metadata
    before={p:p.read_bytes() for p in (x['root']/'cycle.json',x['out']/'state.json',x['pin'])}
    with monkeypatch.context() as patch:
        patch.setattr(metadata,'save',lambda *a,**kw:(_ for _ in ()).throw(OSError('interrupted verification publication')))
        with pytest.raises(OSError,match='verification publication'):
            raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    assert json.loads(x['config'].read_text())['runtime_handoff']==x['proof']
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    assert all(p.read_bytes()==value for p,value in before.items())
    assert runtime.main(['--bootstrap',x['bootstrap']['path'],'--verify',x['proof']['path']])==0


def test_unwitnessed_policy_reversion_is_not_a_legitimate_later_adoption(later):
    x=later
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    value=json.loads(x['config'].read_text())
    value['production_recipe']=json.loads(Path(x['proof']['path']).read_text())['snapshots']['config']['path']
    write(x['config'],value)
    with pytest.raises(ValueError,match='resident adoption history'):
        raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])


def test_prepare_holds_existing_owner_lease_and_cannot_overwrite_cursor(later,tmp_path):
    import fcntl
    x=later
    with (x['root'].parent/'controller.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],tmp_path/'while-owned.json')
    assert not (tmp_path/'while-owned.json').exists()


@pytest.mark.parametrize('later',['lfm','mellum'],indirect=True)
def test_actual_both_port_full_optimizer_continues_after_later_runtime_metadata(later):
    _assert_next_update(later)


def _assert_next_update(x):
    from natlang_neuralese.model.native_model import load_native_checkpoint
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.train.loop import restore_training_rng_state
    before=x['pin'].read_bytes()
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    assert x['pin'].read_bytes()==before
    restore_training_rng_state(x['saved']['training']['rng'],'cpu')
    _step(x['model'],x['optimizer'])
    expected={name:p.detach().clone() for name,p in x['model'].named_parameters()}
    restored,_,_,_,saved=load_native_checkpoint(x['pin'],dtype=next(x['model'].port.parameters()).dtype)
    optimizer,_=make_native_optimizer(restored,lr=3e-6,in_backward=False)
    optimizer.load_state_dict(saved['training']['optimizer'])
    restore_training_rng_state(saved['training']['rng'],'cpu')
    _step(restored,optimizer)
    assert all(torch.equal(p,expected[name]) for name,p in restored.named_parameters())
    assert saved['training']['exposure_counts']=={'a':3,'b':2}
    # Named optimizer tensors and scalar continuation state equal uninterrupted work.
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(i,j) for i,j in zip(a,b))
        return a==b
    assert equal(optimizer.state_dict(),x['optimizer'].state_dict())


@pytest.mark.parametrize('field',['lineage','model_id','checkpoint'])
def test_forged_later_proof_cannot_relabel_lineage_student_or_parent(later,tmp_path,field):
    x=later;proof=json.loads(Path(x['proof']['path']).read_text())
    if field=='checkpoint':proof[field]=json.loads(Path(x['bootstrap']['path']).read_text())['checkpoint']
    else:proof[field]='0'*64
    path=tmp_path/'forged.json';write(path,proof)
    with pytest.raises(ValueError):runtime.verify_runtime_handoff(resident.binding(path),bootstrap=x['bootstrap'])


@pytest.mark.parametrize('later',[None,'lfm','mellum'],indirect=True)
def test_actual_later_resident_adoption_does_not_replace_runtime_authority(later):
    x=later
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    config=json.loads(x['config'].read_text());directory=x['directory']
    plan=json.loads((directory.parent/'recipe-plan.json').read_text())
    cert=Path(plan['recipe']['stages'][0]['parameters']['resident_geometry_qualification'])
    evidence=cert.with_name('evidence.json');write(evidence,{'passed':True,'scope':'tiny CPU admission fixture'})
    write(cert,{'schema':'natlang.native-resident-geometry-qualification/1','passed':True,'identity':None,
        'runtime_code_sha256':resident.digest(plan['code']),'limits':{'forward_tokens':1000000,'kv_slots':1000000},
        'evidence':resident.binding(evidence)})
    from natlang_neuralese.train.recipe import native_recipe_binding
    binding=native_recipe_binding(Path(config['production_recipe']),config['recipe_lineage_sha256'])
    manager=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=binding,train=x['manager'].train,
        held=x['manager'].held,package=x['newpackage'],geometry_qualification=cert)
    recipe=copy.deepcopy(plan['recipe']);recipe['input_bindings'].update(
        train=resident.binding(x['paths']['new']),prior=resident.binding(x['paths']['old']))
    recipe['stages'][0]['inputs']['continued_native_graphs']='prior'
    path=directory.parent/'appended-recipe.json';write(path,recipe)
    resident.prepare(directory,path,'sft',tokenizer_fingerprint=(x['model_config'] or {}).get('tokenizer_fingerprint','tokenizer-sha'),role_ids=(),runtime_package=x['newpackage'])
    assert manager.poll(20) is not None
    adopted=resident.current(directory)
    cursor=json.loads((x['root']/'cycle.json').read_text());cursor['resident_current']=resident.binding(directory/'current.json');write(x['root']/'cycle.json',cursor)
    config['production_recipe']=str(path)
    adopted_inputs=adopted['payload']['plan']['stage_inputs']['sft']
    config['seed_native_graphs']=adopted_inputs['native_graphs']['path']
    config['held_native_graphs']=adopted_inputs['held_native_graphs']['path']
    from natlang_neuralese.rl.native_supervised import bind_recipe_lineage
    bind_recipe_lineage(config,adopted['payload']['plan']['recipe_sha256']);write(x['config'],config)
    before=(x['root']/'cycle.json').read_bytes()
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    result=json.loads(x['config'].read_text())
    assert result['runtime_handoff']==x['proof']
    assert result['production_recipe']==str(path)
    assert (x['root']/'cycle.json').read_bytes()==before
    if x['model'] is not None:_assert_next_update(x)


@pytest.mark.parametrize('tamper',[False,True])
@pytest.mark.parametrize('objective',['native-policy','native-offline-pairwise'])
def test_current_policy_owner_uses_carried_source_plan_without_impersonating_old_owner(later,tmp_path,tamper,objective):
    x=later;directory=tmp_path/'policy-owner';directory.mkdir();gen=actual_drain.__wrapped__(directory)
    try:
        process,intent,terminal,slot,pin=next(gen)
        saved=copy.deepcopy(x['saved'])
        saved['training'].update(global_step=20,identity={'objective':{'kind':objective}},
            source_receipts=[resident.binding(x['paths']['new']),resident.binding(x['paths']['held'])])
        saved['training']['options']={**saved['training']['options'],'stage_id':'policy'}
        saved['training']['optimizer']['moments'].add_(.125)
        if tamper:saved['training']['sft_progress']['sft']['exposure_counts']['a']=0
        torch.save(saved,slot);shutil.copyfile(slot,pin)
        current=drained((process,intent,terminal,slot,pin))
    finally:gen.close()
    state=json.loads((x['out']/'state.json').read_text());state['checkpoint']=str(pin);write(x['out']/'state.json',state)
    out=tmp_path/'policy-runtime-proof.json'
    if tamper:
        with pytest.raises(ValueError,match='historical source-plan frame'):
            runtime.prepare_runtime_handoff(x['bootstrap'],current,x['transition'],x['config'],out)
        return
    proof=runtime.prepare_runtime_handoff(x['bootstrap'],current,x['transition'],x['config'],out)
    x['declaration']['runtime_handoff']=proof;write(x['plan'],{'models':[x['declaration']]})
    before=pin.read_bytes()
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    assert pin.read_bytes()==before
    assert json.loads(x['config'].read_text())['runtime_handoff']==proof
    assert json.loads((x['out']/'state.json').read_text())['checkpoint']==str(pin)
    assert runtime.verify_runtime_handoff(proof,bootstrap=x['bootstrap'])['checkpoint']==resident.binding(pin)


def test_direct_owner_joins_runtime_without_releasing_lease_or_changing_debt(later,tmp_path):
    import fcntl
    x=later;cursor=x['root']/'cycle.json';state=x['out']/'state.json'
    before=(cursor.read_bytes(),state.read_bytes())
    lock=x['root'].parent/'controller.lock'
    with lock.open('a')as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],
            tmp_path/'owned-runtime-proof.json',owned_controller_fd=held.fileno())
        declaration={**x['declaration'],'runtime_handoff':proof}
        runtime.configure_runtime_handoff(x['plan'],declaration,x['bootstrap'],x['config'],owned_controller_fd=held.fileno())
        assert (cursor.read_bytes(),state.read_bytes())==before
        with lock.open('a')as competing:
            with pytest.raises(BlockingIOError):fcntl.flock(competing,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Existing-proof retry must still authenticate any supplied descriptor.
        with pytest.raises(ValueError,match='lease'):
            runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],
                Path(proof['path']),owned_controller_fd=-1)
    with lock.open('a')as released:fcntl.flock(released,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(x['config'].read_text())
    assert config['runtime_handoff']==proof and config['task_source']=={'sealed':True}


@pytest.mark.parametrize('mode',['unheld','shared','wrong-path','invalid'])
def test_runtime_join_rejects_unowned_descriptor(later,tmp_path,mode):
    import fcntl
    x=later;lock=x['root'].parent/'controller.lock';path=tmp_path/'other.lock'if mode=='wrong-path'else lock
    with path.open('a')as lease:
        if mode=='shared':fcntl.flock(lease,fcntl.LOCK_SH|fcntl.LOCK_NB)
        elif mode=='wrong-path':fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fd=-1 if mode=='invalid'else lease.fileno()
        with pytest.raises(ValueError,match='lease'):
            runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],
                tmp_path/'rejected-owned-proof.json',owned_controller_fd=fd)
        with pytest.raises(ValueError,match='lease'):
            runtime.configure_runtime_handoff(x['plan'],x['declaration'],x['bootstrap'],x['config'],owned_controller_fd=fd)
    assert not(tmp_path/'rejected-owned-proof.json').exists()


def test_borrowed_controller_lease_still_requires_production_lock(later,tmp_path):
    import fcntl
    x=later
    with(x['root'].parent/'controller.lock').open('a')as owner,(x['out']/'controller.lock').open('a')as producer:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(producer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],
                tmp_path/'blocked-production-proof.json',owned_controller_fd=owner.fileno())
        with pytest.raises(BlockingIOError):
            runtime.configure_runtime_handoff(x['plan'],x['declaration'],x['bootstrap'],x['config'],owned_controller_fd=owner.fileno())
    assert not(tmp_path/'blocked-production-proof.json').exists()
