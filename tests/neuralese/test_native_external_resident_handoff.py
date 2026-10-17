"""Real captured resident owners on external packages rejoin without fake installs.

Tiny protocol fixtures are not production capacity/exposure qualifications. The
parameterized cases also check the actual next update for both native optimizers.
"""
import copy,json,shutil,subprocess,sys,time
from pathlib import Path
import pytest
import torch
from test_native_owned_operation_reconciliation import owned
from test_native_ordinary_runtime_handoff import ordinary
from test_native_bare_runtime_handoff import bare
from test_native_later_runtime_handoff import later,_assert_next_update
from natlang_neuralese.rl import native_raw_handoff as handoff,native_runtime_handoff as runtime,native_raw_rounds as raw
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.recipe import native_recipe_binding
from natlang_neuralese.common.jsonio import write_json_durable as write


@pytest.fixture
def external(ordinary,tmp_path):
    x=ordinary;N=tmp_path/'external-resident';N.mkdir();(N/'sft').mkdir();out=N/'learner';out.mkdir()
    recipe=copy.deepcopy(json.loads(Path(x['bare']['recipe']['path']).read_text()));recipe_path=N/'initial-recipe.json';write(recipe_path,recipe)
    config=json.loads(x['config'].read_text());lineage=config['recipe_lineage_sha256'];rb=native_recipe_binding(recipe_path,lineage)
    plan=handoff._recipe_plan(recipe_path,'cpu');plan.update(recipe_sha256=lineage,code=resident.code_inventory(x['newpackage']));write(N/'recipe-plan.json',plan)
    directory=N/'resident-data';old=x['manager'].train;held=x['manager'].held
    certificate=json.loads(x['capacity'].read_text())
    manager=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=rb,train=old,held=held,
        package=x['newpackage'],geometry_qualification=x['capacity'],geometry_identity=certificate['identity'])
    argv=[sys.executable,'-c','import sys;sys.stdin.buffer.read()',
        '--out',str(out),'--stage-id','sft','--recipe-binding',json.dumps(rb),'--task-eval','',
        '--resident-data-directory',str(directory)]
    parent=resident.binding(x['pin']);argv_path=N/'argv.json';write(argv_path,argv)
    continuation=N/'continuation.json';write(continuation,{'parent':parent,'recipe':rb,'certificate':resident.binding(x['capacity'])})
    script=N/'supervisor.py';script.write_text("import subprocess,sys,json;p=subprocess.Popen(json.loads(sys.argv[1]),stdin=subprocess.PIPE);print('ready',flush=True);sys.stdin.buffer.read();p.communicate()")
    launch_intent=N/'launch-intent.json';write(launch_intent,{'continuation':resident.binding(continuation),'argv_binding':resident.binding(argv_path),
        'supervisor':resident.binding(script),'env':{'PYTHONPATH':str(x['newpackage'].parent)}})
    supervisor=[sys.executable,str(script),json.dumps(argv)];process=subprocess.Popen(supervisor,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='ready';owner=handoff.observe(process.pid)
        original=N/'launch.json';write(original,{'owner':owner,'argv':argv,'intent':resident.binding(launch_intent)})
        spawn=N/'spawn.json';write(spawn,{'identity':owner,'intent':resident.binding(launch_intent),'script':resident.binding(script)})
        observed=N/'observed.json';write(observed,{'at':time.time(),'pid':owner['pid'],'argv':supervisor,'identity':owner,
            'original_launch':resident.binding(original),'spawn':resident.binding(spawn)})
        intent=handoff.capture_drain_intent(resident.binding(observed),N/'drain-intent.json')
        captured=json.loads(Path(intent['path']).read_text());child=next(c for c in captured['children']if handoff.process_argv(c)==argv)
        initial=json.loads((directory/'initial.json').read_text());initial['owner']=child;write(directory/'initial.json',initial)
        # Genuine resident publication and adoption, while its captured owner lives.
        successor=copy.deepcopy(recipe);successor['input_bindings']['prior']=successor['input_bindings']['train']
        successor['input_bindings']['train']=resident.binding(x['paths']['new']);successor['stages'][0]['inputs']['continued_native_graphs']='prior'
        successor_path=N/'adopted-recipe.json';write(successor_path,successor)
        resident.prepare(directory,successor_path,'sft',tokenizer_fingerprint=old.tokenizer_fingerprint,role_ids=old.role_ids,runtime_package=x['newpackage'])
        assert manager.poll(21) is not None;state=resident.current(directory)
        saved=copy.deepcopy(x['saved']);saved.update(step=22);saved['training'].update(step=22,global_step=22,
            source_receipts=[resident.binding(x['paths']['new']),resident.binding(x['paths']['held'])])
        saved['training']['options'].update(recipe_binding=state['payload']['recipe'],resident_data_directory=str(directory),
            resident_adoption={'generation':state['generation'],'history':state['history']},continuation={'checkpoint_sha256':parent['sha256']})
        from natlang_neuralese.train.native_continuation import _TRANSIENT
        saved['training']['sft_progress']['sft']={k:copy.deepcopy(v)for k,v in saved['training'].items()if k not in _TRANSIENT}
        slot=out/'checkpoint.pt';pin=N/'immutable-full.pt';torch.save(saved,slot);shutil.copyfile(slot,pin)
        process.communicate();assert process.returncode==0;terminal=N/'exit.json';write(terminal,{'at':time.time(),'returncode':0})
        drained=handoff.complete_drain(intent,resident.binding(terminal),resident.binding(pin),N/'handoff.json',
            original_slot=resident.binding(slot),resident_directory=directory)
    finally:
        if process.poll() is None:process.communicate()
    resident.retire(directory,resident.binding(pin),resident.binding(terminal))
    transition=handoff.prepare_drained_recipe_handoff(drained,successor_path,N/'external-transition.json',
        package=x['newpackage'],external_package=x['newpackage'])
    receipt=out/'native-training-receipt.json';write(receipt,{'schema':'natlang.native-training-readiness/1',
        'model_id':config['raw_model_id'],'stage_id':'sft','recipe':state['payload']['recipe'],
        'checkpoint':resident.binding(slot),'native_identity':{'checkpoint_sha256':resident.binding(pin)['sha256']},
        'continuation':{'checkpoint_sha256':parent['sha256']},'continuation_boundary':{
            'new_optimizer_updates':3,'checkpoint_sha256':resident.binding(pin)['sha256']},'completed':False})
    central=json.loads((x['out']/'state.json').read_text());central.update(checkpoint=parent['path'],phase='collect');write(x['out']/'state.json',central)
    cursor=json.loads((x['root']/'cycle.json').read_text());cursor.update(phase='rl',next_stage=0,pending_rl_round=None);write(x['root']/'cycle.json',cursor)
    x.update(saved=saved,pin=pin,slot=slot,handoff=drained,directory=directory,transition=transition,
        operations=[{'handoff':drained,'receipt':resident.binding(receipt)}],recipe=resident.binding(successor_path),
        reconciliation=N/'reconciliation.json',external_root=N,initial_recipe=rb)
    return x


def test_external_resident_runtime_is_never_copied_or_renamed(external):
    x=external;assert not(x['external_root']/'runtime').exists()
    value=handoff.verify_drained_recipe_handoff(x['transition'],handoff=x['handoff'])
    assert value['schema']==handoff.EXTERNAL_RECIPE_HANDOFF
    assert Path(value['old_package'])==x['newpackage'] and value['old_code']==value['new_code']
    assert not list(x['external_root'].glob('runtime-superseded-*'))
    assert handoff.prepare_drained_recipe_handoff(x['handoff'],Path(x['recipe']['path']),Path(x['transition']['path']),
        package=x['newpackage'],external_package=x['newpackage'])==x['transition']


def _rejoin(x):
    owned=raw.reconcile_owned_operations(x['config'],x['operations'],x['recipe'],x['reconciliation'],recipe_handoff=x['transition'])
    x['proof']=runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],x['external_root']/'runtime-authority.json',owned_operations=owned)
    x['declaration']['runtime_handoff']=x['proof'];write(x['plan'],{'models':[x['declaration']]})
    return owned


def test_adopted_recipe_rejoins_earned_cursor_from_original_launch_recipe(external):
    x=external;before=x['pin'].read_bytes()
    owned=_rejoin(x)
    cursor=json.loads((x['root']/'cycle.json').read_text());assert cursor['phase']=='sft'and cursor['next_stage']==1
    assert json.loads(Path(owned['path']).read_text())['whole_recipe_completed']is False
    proof=runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],x['external_root']/'runtime-authority.json',owned_operations=owned)
    assert runtime.verify_runtime_handoff(proof,bootstrap=x['bootstrap'])['owned_operations']==owned
    assert x['pin'].read_bytes()==before


@pytest.mark.parametrize('tamper',['external-path','launch-env','schema','history-recipe','history-source'])
def test_external_resident_cannot_borrow_a_runtime_or_rewrite_adoption(external,tamper,tmp_path):
    x=external;value=json.loads(Path(x['transition']['path']).read_text())
    if tamper=='external-path':value['external_runtime']['package']=str(tmp_path)
    elif tamper=='schema':value['schema']=handoff.RECIPE_HANDOFF
    elif tamper=='launch-env':
        control=value['external_runtime']['launch_control'];p=tmp_path/'forged-intent.json';v=json.loads(Path(control['intent']['path']).read_text());v['env']['PYTHONPATH']=str(tmp_path);write(p,v);control['intent']=resident.binding(p)
    else:
        record=json.loads(Path(x['handoff']['path']).read_text());current=json.loads(Path(record['resident']['current']['path']).read_text())
        if tamper=='history-recipe':current['history'][0]['recipe']=x['initial_recipe']
        else:current['history'][0]['train']=current['payload']['held']
        p=tmp_path/'changed-current.json';write(p,current);record['resident']['current']=resident.binding(p)
        intent=json.loads(Path(record['intent']['path']).read_text());child=next(c for c in intent['children']if '--recipe-binding'in handoff.process_argv(c))
        with pytest.raises(ValueError):raw._owned_recipe_invocation(handoff.process_argv(child),child,record,current['payload']['recipe'],'sft',json.loads(x['config'].read_text())['recipe_lineage_sha256'])
        return
    forged=tmp_path/'forged-transition.json';write(forged,value)
    with pytest.raises((ValueError,FileNotFoundError)):handoff.verify_drained_recipe_handoff(resident.binding(forged),handoff=x['handoff'])


@pytest.mark.parametrize('later',['lfm','mellum'],indirect=True)
def test_both_actual_optimizer_next_updates_survive_external_resident_authority(external):
    _rejoin(external)
    _assert_next_update(external)


def test_external_launch_snapshots_survive_pruned_original_control_scripts(external):
    x=external;record,_=handoff.verify_handoff(x['handoff'])
    _,_,references=runtime._owned_launch_chain(record)
    for reference in references.values():Path(reference['path']).unlink()
    assert handoff.verify_drained_recipe_handoff(x['transition'],handoff=x['handoff'])['external_runtime']


def test_external_qualified_package_cannot_change_after_drained_authority(external):
    x=external
    with(x['newpackage']/'__init__.py').open('a')as f:f.write('\n# wrong external runtime\n')
    with pytest.raises(ValueError):handoff.verify_drained_recipe_handoff(x['transition'],handoff=x['handoff'])


def test_external_launch_controls_are_in_bootstrap_cache_closure(external,monkeypatch,tmp_path):
    from natlang_neuralese.rl import native_bootstrap_metadata as metadata
    from natlang_neuralese.common import runtime_inventory
    x=external;_rejoin(x)
    bootstrap=json.loads(Path(x['bootstrap']['path']).read_text());captured=json.loads(Path(bootstrap['intent']['path']).read_text())
    x['declaration']['handoff_requirement'].update(trainer_launch=captured.get('source_launch',captured['launch']),
        trainer_exit=bootstrap.get('source_terminal',bootstrap['terminal'])['path'])
    write(x['plan'],{'models':[x['declaration']]})
    monkeypatch.setattr(runtime_inventory,'task_runtime_inventory',lambda path:{'fixture':'complete collector inventory'})
    monkeypatch.setattr(raw,'_inputs',lambda declaration:{'cases':Path(declaration['cases']['path'])})
    raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'])
    config=json.loads(x['config'].read_text());value=metadata.evidence(x['plan'],x['declaration'],x['bootstrap'],config)
    transition=json.loads(Path(x['transition']['path']).read_text())
    controls=transition['external_runtime']['launch_control']
    assert all(reference['path']in value['files']for reference in controls.values())
    receipt=tmp_path/'metadata.json';metadata.save(receipt,x['plan'],x['declaration'],x['bootstrap'],config)
    assert metadata.unchanged(receipt,x['plan'],x['declaration'],x['bootstrap'],config)
    Path(controls['supervisor']['path']).write_text('# changed authority snapshot\n')
    with pytest.raises(ValueError):metadata.unchanged(receipt,x['plan'],x['declaration'],x['bootstrap'],config)
