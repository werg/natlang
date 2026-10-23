"""Exact pre-update failure authority and one-use same-stage retry boundary."""
import copy
import fcntl
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_bridge as bridge
from natlang_neuralese.rl import native_failed_supervised_retry as retry
from natlang_neuralese.rl.native_raw_rounds import cycle_work_directory
from natlang_neuralese.train.native_resident import binding

_ACTUAL_MEMORY_FAILURE = 'memory qualification request differs from actual restored model/runtime/packing'


def _recipe(tmp_path):
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    train.write_text('{"id":"train"}\n');held.write_text('{"id":"held","split":"test"}\n')
    value={'schema':'natlang.neuralese-training-recipe/1','id':'retry-fixture',
        'identity_qualification':'native_state_runtime_qualified',
        'input_bindings':{'train':binding(train),'held':binding(held)},
        'stages':[{'id':stage,'kind':'native_state_sft','requires':[],
            'inputs':{'native_graphs':'train','held_native_graphs':'held'},'parameters':{'epochs':1}}
            for stage in ('dpo','full')]}
    path=tmp_path/'recipe.json';write(path,value);return path


def _as_stale_resident_plan_failure(x):
    config=json.loads(x['config'].read_text())
    train=Path(config['production_recipe']).parent/'train.jsonl'
    held=Path(config['production_recipe']).parent/'held.jsonl'
    resident=Path(config['production_recipe']).parent/'resident-data'
    resident.mkdir(exist_ok=True)
    old_train=train.with_name('old-train.jsonl');old_train.write_text('{"id":"old-train"}\n')
    old_held=held.with_name('old-held.jsonl');old_held.write_text('{"id":"old-held","split":"test"}\n')
    config.update({'resident_data_directory':str(resident),'seed_native_graphs':binding(train),
        'held_native_graphs':binding(held)})
    plan={'stage_inputs':{'full':{'native_graphs':binding(old_train),
        'held_native_graphs':binding(old_held)}}}
    plan_path=Path(config['production_recipe']).parent/'recipe-plan.json';write(plan_path,plan)
    from natlang_neuralese.train.native_resident import digest as resident_digest
    write(resident/'initial.json',{'stage_id':'full','plan_sha256':resident_digest(plan),
        'owner':{'pid':99999999}})
    write(x['config'],config)
    invocation={**config,'out':str(x['job']),'mandatory_sft_only':True}
    invocation_path=x['job']/'config.json';write(invocation_path,invocation)
    request_path=bridge.write_request(x['job']/'bridge-stale-plan',kind='round',out=x['job'],
        config=invocation_path,state=Path(config['out'])/'state.json')
    request=bridge.read_request(request_path);result_path=x['job']/'failed-stale-plan.result.json'
    log_path=Path(x['failure_log']['path']).parent.parent/request['request_id']/'worker.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    log_path.write_text('Traceback (most recent call last):\n'
        '  File "/runtime/training/neuralese/natlang_neuralese/train/native_state_sft.py", line 1704, in _main\n'
        '    resident = ResidentAdoption(args.resident_data_directory)\n'
        '  File "/runtime/training/neuralese/natlang_neuralese/train/native_resident.py", line 1577, in __init__\n'
        "    raise ValueError('"+retry.RESIDENT_PLAN_FAILURE+"')\n"
        'ValueError: '+retry.RESIDENT_PLAN_FAILURE+'\n')
    write(result_path,{'schema':bridge.RESULT_SCHEMA,'kind':'round','request_id':request['request_id'],
        'request_sha256':bridge.sha256_file_hex_cached(request_path),'status':'failed',
        'error':"CalledProcessError: Command ['env', 'python'] returned non-zero exit status 1.",
        'artifacts':bridge.inventory(x['job'],exclude=[result_path])})
    x.update({'request':request_path,'result':result_path,
        'failure_log':{**binding(log_path),'bytes':log_path.stat().st_size},
        'resident_plan':plan_path,'resident_plan_value':plan})
    return x


@pytest.fixture(params=['lfm','mellum'])
def failed_operation(tmp_path,request):
    model=request.param
    root=tmp_path/'rounds'/model;production=root/'production';production.mkdir(parents=True)
    parent=tmp_path/'checkpoint.pt';parent.write_bytes(b'verified full parent')
    checkpoint=bridge.binding(parent)
    recipe=_recipe(tmp_path)
    state={'phase':'collect','checkpoint':str(parent),'round':4,'trained_rounds':4}
    write(production/'state.json',state)
    cursor={'phase':'sft','cycle':4,'next_stage':1,'raw_retry_generation':0,
        'current_checkpoint':checkpoint,'earned_dpo_exposure':{'updates':250,'rows':100}}
    write(root/'cycle.json',cursor)
    current={'out':str(production),'checkpoint':str(parent),'native_actor_mode':'raw-state',
        'production_recipe':str(recipe),'production_stage':'full','raw_model_id':model+':'+'a'*64,
        'mandatory_sft_only':True,'trainer_parameters':{'stage_id':'full','memory_qualification':str(tmp_path/'old-capacity.json')}}
    write(tmp_path/'old-capacity.json',{'schema':'natlang.native-memory-qualification-request/1'})
    write(root/'config.json',current)
    job=cycle_work_directory(root,cursor)/'supervised'/'001';job.mkdir(parents=True)
    invocation={**current,'out':str(job),'mandatory_sft_only':True,
        'trainer_parameters':{'stage_id':'full','memory_qualification':str(tmp_path/'old-capacity.json')}}
    invocation_path=job/'config.json';write(invocation_path,invocation)
    request_path=bridge.write_request(job/'bridge',kind='round',out=job,config=invocation_path,
        state=production/'state.json')
    request=bridge.read_request(request_path)
    result_path=job/'failed.result.json'
    failure_log_path=tmp_path/request['request_id']/'worker.log';failure_log_path.parent.mkdir()
    failure_log_path.write_text("Traceback (most recent call last):\n  File \"native_memory.py\", line 744, in run_prepared\n"
        "  File \"native_memory.py\", line 427, in _run_one_prepared\n"
        "    raise ValueError('"+_ACTUAL_MEMORY_FAILURE+"')\nValueError: "+_ACTUAL_MEMORY_FAILURE+"\n")
    write(result_path,{'schema':bridge.RESULT_SCHEMA,'kind':'round','request_id':request['request_id'],
        'request_sha256':bridge.sha256_file_hex_cached(request_path),'status':'failed',
        'error':"CalledProcessError: Command ['env', 'python'] returned non-zero exit status 1.",
        'artifacts':bridge.inventory(job,exclude=[result_path])})
    saved={'model_config':{'architecture':model,'base_inventory_sha256':'a'*64},
        'training':{'global_step':31716,'schema':'fixture','sft_progress':{
            'dpo':{'exposure_counts':{'row':250}},'full':{'exposure_counts':{'row':0}}}}}
    return {'root':root,'config':root/'config.json','cursor':cursor,'state':state,
        'checkpoint':checkpoint,'saved':saved,'request':request_path,'result':result_path,
        'failure_log':{**binding(failure_log_path),'bytes':failure_log_path.stat().st_size},
        'job':job,'invocation':invocation}


def test_exact_known_preupdate_failure_has_zero_checkpoint_authority(failed_operation):
    x=failed_operation
    config=json.loads(x['config'].read_text())
    result=retry._verify_failure(binding(x['request']),binding(x['result']),x['failure_log'],
        config=config,cursor=x['cursor'],
        state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
    assert result[3]['id']=='full'
    assert result[1]['status']=='failed' and result[1].get('checkpoint') is None
    baseline_result=json.loads(x['result'].read_text());baseline_request=x['request'].read_bytes()
    for change in ('status','error','checkpoint','parent','traceback'):
        x['request'].write_bytes(baseline_request)
        value=copy.deepcopy(baseline_result)
        if change=='status':value['status']='completed'
        elif change=='error':value['error']='ValueError: another failure'
        elif change=='checkpoint':value['checkpoint']=x['checkpoint']
        elif change=='traceback':
            old_log=x['failure_log'];log_path=Path(old_log['path']);original=log_path.read_bytes()
            log_path.write_bytes(b'ValueError: unrelated failure\n')
            with pytest.raises(ValueError):
                retry._verify_failure(binding(x['request']),binding(x['result']),old_log,
                    config=config,cursor=x['cursor'],state=x['state'],
                    verified_checkpoint=(x['checkpoint'],x['saved']))
            log_path.write_bytes(original)
            continue
        else:
            request=bridge.read_request(x['request']);request['parent_checkpoint']={**request['parent_checkpoint'],'sha256':'b'*64}
            # Preserve the sealed request/result hash relationship while testing
            # a valid but wrong parent identity.
            request.pop('request_id',None)
            from natlang_neuralese.common.hashing import canonical_json_sha256_hex
            request['request_id']=canonical_json_sha256_hex(request)
            write(x['request'],request)
            value['request_id']=request['request_id'];value['request_sha256']=bridge.sha256_file_hex_cached(x['request'])
        write(x['result'],value)
        with pytest.raises(ValueError):
            retry._verify_failure(binding(x['request']),binding(x['result']),x['failure_log'],
                config=config,cursor=x['cursor'],
                state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
    x['request'].write_bytes(baseline_request)


def test_exact_stale_resident_plan_failure_is_zero_credit_and_source_bound(failed_operation):
    x=_as_stale_resident_plan_failure(failed_operation);config=json.loads(x['config'].read_text())
    verified=retry._verify_failure(binding(x['request']),binding(x['result']),x['failure_log'],
        config=config,cursor=x['cursor'],state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
    assert verified[3]['id']=='full' and verified[1].get('checkpoint') is None
    assert verified[6]['kind']==retry.RESIDENT_PLAN_FAILURE_KIND
    assert verified[6]['stale_resident_plan']['planned_sources']!=verified[6]['stale_resident_plan']['active_sources']
    request=bridge.read_request(x['request'])
    request['parent_checkpoint']={**request['parent_checkpoint'],'sha256':'b'*64}
    request.pop('request_id',None)
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    request['request_id']=canonical_json_sha256_hex(request);write(x['request'],request)
    result=json.loads(x['result'].read_text());result.update(request_id=request['request_id'],
        request_sha256=bridge.sha256_file_hex_cached(x['request']));write(x['result'],result)
    with pytest.raises(ValueError):
        retry._verify_failure(binding(x['request']),binding(x['result']),x['failure_log'],
            config=config,cursor=x['cursor'],state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))


def test_metadata_only_retry_reuses_exact_passed_memory_request(tmp_path):
    passed=tmp_path/'passed-memory.json';write(passed,{'schema':'natlang.native-memory-qualification-request/1'})
    same=binding(passed)
    retry._validate_corrected_capacity_binding(retry.RESIDENT_PLAN_FAILURE_KIND,same,same)
    altered=tmp_path/'altered-memory.json';write(altered,{'schema':'natlang.native-memory-qualification-request/1','packing':'changed'})
    with pytest.raises(ValueError,match='previously passed memory qualification'):
        retry._validate_corrected_capacity_binding(retry.RESIDENT_PLAN_FAILURE_KIND,same,binding(altered))
    with pytest.raises(ValueError,match='distinct versioned request'):
        retry._validate_corrected_capacity_binding(retry.MEMORY_FAILURE_KIND,same,same)
    corrected=tmp_path/'corrected-memory.json';write(corrected,{'schema':'natlang.native-memory-qualification-request/1'})
    retry._validate_corrected_capacity_binding(retry.MEMORY_FAILURE_KIND,same,binding(corrected))


def test_stale_resident_plan_rejects_wrong_traceback_source_and_credit_artifacts(failed_operation):
    x=_as_stale_resident_plan_failure(failed_operation);config=json.loads(x['config'].read_text())
    log_path=Path(x['failure_log']['path']);original=log_path.read_bytes()
    result_path=x['result'];original_result=result_path.read_bytes()
    try:
        log_path.write_text('Traceback (most recent call last):\nValueError: '+retry.RESIDENT_PLAN_FAILURE+'\n')
        with pytest.raises(ValueError,match='exact failed pre-update'):
            retry._verify_failure(binding(x['request']),binding(result_path),{**binding(log_path),'bytes':log_path.stat().st_size},
                config=config,cursor=x['cursor'],state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
        log_path.write_bytes(original)
        # The traceback is exact, but the resident plan no longer demonstrates
        # a stale source binding once it is changed to the active sources.
        plan=x['resident_plan_value'];plan['stage_inputs']['full']['native_graphs']=binding(config['seed_native_graphs']['path'])
        plan['stage_inputs']['full']['held_native_graphs']=binding(config['held_native_graphs']['path'])
        write(x['resident_plan'],plan)
        from natlang_neuralese.train.native_resident import digest as resident_digest
        initial=x['resident_plan'].parent/'resident-data'/'initial.json'
        init=json.loads(initial.read_text());init['plan_sha256']=resident_digest(plan);write(initial,init)
        with pytest.raises(ValueError,match='not explained by a stale'):
            retry._verify_failure(binding(x['request']),binding(result_path),x['failure_log'],
                config=config,cursor=x['cursor'],state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
        write(x['resident_plan'],x['resident_plan_value'])
        init['plan_sha256']=resident_digest(x['resident_plan_value']);write(initial,init)
        forged=json.loads(original_result);forged['artifacts'].append({'path':str(x['job']/'checkpoint.pt'),
            'sha256':'c'*64,'bytes':1});write(result_path,forged)
        with pytest.raises(ValueError,match='exact failed pre-update'):
            retry._verify_failure(binding(x['request']),binding(result_path),x['failure_log'],
                config=config,cursor=x['cursor'],state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
    finally:
        log_path.write_bytes(original);write(x['resident_plan'],x['resident_plan_value'])
        result_path.write_bytes(original_result)


@pytest.mark.parametrize('bad_log', [
    "Traceback (most recent call last):\n  File \"native_memory.py\", line 744, in run_prepared\n"
    "  File \"native_memory.py\", line 427, in _run_one_prepared\n"
    "    raise ValueError('ValueError: "+_ACTUAL_MEMORY_FAILURE+"')\nValueError: ValueError: "+_ACTUAL_MEMORY_FAILURE+"\n",
    "Traceback (most recent call last):\n  File \"native_memory.py\", line 744, in run_prepared\n"
    "  File \"native_memory.py\", line 427, in _run_one_prepared\n"
    "    raise ValueError('different failure')\nValueError: "+_ACTUAL_MEMORY_FAILURE+"\n",
])
def test_failure_log_requires_exact_single_prefix_and_source_message(failed_operation,bad_log):
    x=failed_operation;config=json.loads(x['config'].read_text())
    log_path=Path(x['failure_log']['path']);original=log_path.read_bytes()
    log_path.write_text(bad_log)
    try:
        with pytest.raises(ValueError,match='exact failed pre-update'):
            retry._verify_failure(binding(x['request']),binding(x['result']),
                {**binding(log_path),'bytes':log_path.stat().st_size},config=config,cursor=x['cursor'],
                state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
    finally:
        log_path.write_bytes(original)


def test_corrected_capacity_identity_requires_ffn_chunk_and_exact_runtime(tmp_path):
    from natlang_neuralese.train.native_resident import code_inventory, digest as resident_digest
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    import natlang_neuralese
    package=Path(natlang_neuralese.__file__).parent
    model={'architecture':'fixture','base_inventory_sha256':'a'*64}
    parameters={'batch_graphs':1,'batch_tokens':64,'checkpoint_layers':0,'chunk_tokens':64,
        'ffn_chunk_tokens':4096,'checkpoint_input_offload':False,'share_causal_prefixes':False}
    identity={**parameters,'model_config':model}
    value={'schema':'natlang.native-memory-qualification-request/1','identity':identity,
        'runtime_code_sha256':canonical_json_sha256_hex(code_inventory(package))}
    path=tmp_path/'capacity-v2.json';write(path,value)
    verified=({'path':str(tmp_path/'checkpoint.pt'),'sha256':'a'*64}, {'model_config':model})
    assert retry._corrected_request(binding(path),config={'trainer_parameters':parameters,
        'production_stage':'full'},verified_checkpoint=verified,package=package,stage_id='full')==value
    value['identity'].pop('ffn_chunk_tokens');write(path,value)
    with pytest.raises(ValueError,match='packing'):
        retry._corrected_request(binding(path),config={'trainer_parameters':parameters,
            'production_stage':'full'},verified_checkpoint=verified,package=package,stage_id='full')


def test_preference_capacity_descriptor_is_bound_to_the_exact_future_inputs(tmp_path):
    source=tmp_path/'source.jsonl';source.write_text('{"id":"pair"}\n')
    objective=tmp_path/'objective.json';write(objective,{'schema':'natlang.native-offline-pairwise-objective/1',
        'kind':'native-offline-pairwise','beta':0.1,'margin':0.0})
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    descriptor={'schema':'natlang.native-preference-capacity-case/1','source':binding(source),
        'objective':{'kind':'native-offline-pairwise','sha256':sha256_file_hex_cached(objective),
            'beta':0.1,'margin':0.0},
        'case':{'id':'pair','graph_sha256':'a'*64,'split':'train',
            'geometry':{'forward_tokens':64,'kv_slots':2}},
        'numerical_reference':{'id':'reference','graph_sha256':'b'*64,'split':'train',
            'geometry':{'forward_tokens':32,'kv_slots':1}}}
    path=tmp_path/'capacity-case.json';write(path,descriptor)
    config={'trainer_parameters':{'preference_beta':0.1,'preference_margin':0.0},
        'seed_native_graphs':binding(source),'objective_spec':str(objective),'production_stage':'full'}
    assert retry._corrected_request(binding(path),config=config,verified_checkpoint=({},{}),
        package=tmp_path,stage_id='full')==descriptor
    other=tmp_path/'other';other.write_text('different source\n')
    for key,change in (('source',lambda value:value.update(source=binding(other))),
                       ('objective',lambda value:value['objective'].update(sha256='d'*64)),
                       ('case',lambda value:value['case'].update(split='test'))):
        bad=copy.deepcopy(descriptor);change(bad);write(path,bad)
        with pytest.raises(ValueError):
            retry._corrected_request(binding(path),config=config,verified_checkpoint=({},{}),
                package=tmp_path,stage_id='full')


def test_retry_installs_durable_future_plan_through_shared_resident_verifier(tmp_path):
    import natlang_neuralese
    from natlang_neuralese.train.native_resident import code_inventory,digest as resident_digest
    from natlang_neuralese.train.recipe import lineage_handoff,recipe_identity_sha256
    from natlang_neuralese.rl.native_raw_handoff import _recipe_plan
    package=Path(natlang_neuralese.__file__).parent
    recipe=_recipe(tmp_path);lineage=recipe_identity_sha256(recipe);code=code_inventory(package)
    old=_recipe_plan(recipe,'cuda');old.update(code=code,recipe_sha256=lineage)
    future=_recipe_plan(recipe,'cuda');future['code']=code
    proof=lineage_handoff(old,future,package,package,tmp_path/'lineage.json',
        out=tmp_path,dry_run=True,started_stage_ids=[])
    lineage_path=tmp_path/'lineage-proof.json';future_path=tmp_path/'future-plan.json'
    write(lineage_path,proof);write(future_path,future)
    resident_dir=recipe.parent/'resident-data'
    resident_dir.mkdir()
    config={'production_recipe':str(recipe),'production_stage':'full',
        'resident_data_directory':str(resident_dir),
        'seed_trainer_parameters':{'resident_data_directory':str(resident_dir)},
        'seed_native_graphs':binding(tmp_path/'train.jsonl'),
        'held_native_graphs':binding(tmp_path/'held.jsonl')}
    old_train=tmp_path/'old-train.jsonl';old_train.write_text('{"id":"prior-train"}\n')
    old_held=tmp_path/'old-held.jsonl';old_held.write_text('{"id":"prior-held","split":"test"}\n')
    stale_plan={'stage_inputs':{'full':{'native_graphs':binding(old_train),
        'held_native_graphs':binding(old_held)}}}
    target=recipe.parent/'recipe-plan.json';write(target,stale_plan)
    write(resident_dir/'initial.json',{'stage_id':'full','plan_sha256':resident_digest(stale_plan),
        'owner':{'pid':99999999}})
    stale=retry._stale_resident_plan(config,'full')
    plan_snapshot=tmp_path/'stale-plan.json';plan_snapshot.write_bytes(target.read_bytes())
    initial_snapshot=tmp_path/'stale-initial.json';initial_snapshot.write_bytes((resident_dir/'initial.json').read_bytes())
    stale={**stale,'snapshot':{**binding(plan_snapshot),'bytes':plan_snapshot.stat().st_size},
        'initial_snapshot':{**binding(initial_snapshot),'bytes':initial_snapshot.stat().st_size}}
    request={'recipe':binding(recipe),'lineage_handoff':binding(lineage_path),
        'future_plan':binding(future_path),'package':str(package),'lineage_handoff_sha256':retry._digest(proof),
        'failure_kind':retry.RESIDENT_PLAN_FAILURE_KIND,'stage_id':'full','stale_resident_plan':stale}
    installed=retry._install_retry_resident_plan(request,config)
    target=recipe.parent/'recipe-plan.json'
    assert installed==binding(target)
    assert json.loads(target.read_text())==proof
    assert binding(resident_dir/'initial.json')['sha256']==binding(initial_snapshot)['sha256']
    assert binding(tmp_path/'train.jsonl')['sha256']==proof['stage_inputs']['full']['native_graphs']['sha256']


def test_no_credit_retry_consumes_only_exact_fresh_stage_boundary(tmp_path):
    root=tmp_path/'rounds';root.mkdir();receipt_path=tmp_path/'retry.json'
    checkpoint={'path':str(tmp_path/'checkpoint.pt'),'sha256':'c'*64}
    request={'schema':retry.SCHEMA,'next_stage':1,'stage_id':'full'}
    request_sha=retry._digest(request)
    intent_path=tmp_path/'intent.json';write(intent_path,{'schema':retry.SCHEMA,'failed_stage_index':1,
        'request':request,'training_credit':0,'whole_recipe_completed':False,'readiness_granted':False})
    write(receipt_path,{'schema':retry.SCHEMA,'intent':binding(intent_path),'request_sha256':request_sha,
        'stage_id':'full','checkpoint':checkpoint,'training_credit':0,
        'whole_recipe_completed':False,'readiness_granted':False})
    reference=binding(receipt_path)
    cursor={'phase':'sft','cycle':2,'next_stage':1,'raw_retry_generation':1,
        'current_checkpoint':checkpoint,'earned_dpo_exposure':{'updates':250,'rows':100},
        'failed_supervised_retry':{'receipt_path':str(receipt_path),
            'request_sha256':request_sha}}
    before=copy.deepcopy(cursor);state={'phase':'collect'}
    first=retry.consume_boundary(reference,{'failed_supervised_retry':cursor['failed_supervised_retry']},
        cursor,state,root)
    assert first is not cursor and 'failed_supervised_retry_consumed' not in cursor
    assert first['failed_supervised_retry_consumed']['training_credit']==0
    assert first['next_stage']==1 and first['current_checkpoint']==checkpoint
    assert first['earned_dpo_exposure']==before['earned_dpo_exposure']
    assert retry.consume_boundary(reference,{'failed_supervised_retry':cursor['failed_supervised_retry']},first,state,root)==first
    completed=copy.deepcopy(first);identity=completed.pop('failed_supervised_retry_consumed')
    completed.pop('failed_supervised_retry');completed['failed_supervised_retry_history']=[identity]
    completed['next_stage']=0 # later normal cycle reset must not reopen the old receipt
    assert retry.consume_boundary(reference,{'failed_supervised_retry':cursor['failed_supervised_retry']},
        completed,state,root)==completed
    bad=copy.deepcopy(before);bad['next_stage']=0
    with pytest.raises(ValueError):retry.consume_boundary(reference,{},bad,state,root)
    bad=copy.deepcopy(before);bad['current_checkpoint']={'path':checkpoint['path'],'sha256':'d'*64}
    with pytest.raises(ValueError):retry.consume_boundary(reference,{},bad,state,root)
    badstate={'phase':'learner_pending'}
    with pytest.raises(ValueError):retry.consume_boundary(reference,{},before,badstate,root)


def test_configure_replays_exact_candidate_with_inherited_lease_and_recovers_partial_cas(
        tmp_path,monkeypatch,failed_operation):
    x=_as_stale_resident_plan_failure(failed_operation);root=x['root'];production=root/'production'
    config_path=x['config'];cursor_path=root/'cycle.json';state_path=production/'state.json'
    original_config=json.loads(config_path.read_text());recipe=Path(original_config['production_recipe'])
    plan=tmp_path/'plan.json'
    write(plan,{'plan':'bound'})
    import natlang_neuralese
    from natlang_neuralese.train.native_resident import code_inventory
    from natlang_neuralese.train.recipe import lineage_handoff,recipe_identity_sha256
    from natlang_neuralese.rl.native_raw_handoff import _recipe_plan
    package=Path(natlang_neuralese.__file__).parent
    lineage=recipe_identity_sha256(recipe)
    original_config['recipe_lineage_sha256']=lineage
    original_config['seed_trainer_parameters']={'resident_data_directory':original_config['resident_data_directory']}
    original_cursor=json.loads(cursor_path.read_text());original_cursor['raw_retry_generation']=0
    original_state=json.loads(state_path.read_text())
    predecessor_resident=tmp_path/'predecessor-current.json';write(predecessor_resident,{'historical':'committed'})
    original_cursor['resident_current']=binding(predecessor_resident)
    write(config_path,original_config);write(cursor_path,original_cursor);write(state_path,original_state)
    (root.parent/'controller.lock').touch();(production/'controller.lock').touch()
    candidate_config={**original_config,'production_recipe':str(recipe),'trainer_parameters':{'memory_qualification':'new'},
        'failed_supervised_retry':{'receipt_path':str(tmp_path/'retry.json')}}
    candidate_cursor={**original_cursor,'raw_retry_generation':1,
        'failed_supervised_retry':{'receipt_path':str(tmp_path/'retry.json')}}
    candidates={};snapshots={};before={}
    for name,path,value in (('config',config_path,candidate_config),('cursor',cursor_path,candidate_cursor),('state',state_path,original_state)):
        before[name]=binding(path)
        snapshot=tmp_path/f'original-{name}.json';snapshot.write_bytes(path.read_bytes());snapshots[name]=binding(snapshot)
        candidate=tmp_path/f'candidate-{name}.json';write(candidate,value);candidates[name]=binding(candidate)
    failed_request=binding(x['request']);failed_result=binding(x['result']);corrected={}
    controller={'pid':99999999};census=[{'pid':99999998}]
    source_log=x['failure_log'];source_log_path=Path(source_log['path'])
    failure_log={'source':source_log,'snapshot':binding(source_log_path)}
    tx_log_path=tmp_path/'retry-control'/'failure-worker.log';tx_log_path.parent.mkdir()
    tx_log_path.write_bytes(source_log_path.read_bytes());tx_log=binding(tx_log_path)
    stale=retry._stale_resident_plan(original_config,'full')
    stale_snapshot_path=tmp_path/'retry-control'/'stale-resident-plan.json'
    stale_snapshot_path.write_bytes(Path(stale['path']).read_bytes())
    initial_snapshot_path=tmp_path/'retry-control'/'stale-resident-initial.json'
    initial_snapshot_path.write_bytes(Path(stale['initial_path']).read_bytes())
    stale={**stale,'snapshot':{**binding(stale_snapshot_path),'bytes':stale_snapshot_path.stat().st_size}}
    stale['initial_snapshot']={**binding(initial_snapshot_path),'bytes':initial_snapshot_path.stat().st_size}
    old_plan=_recipe_plan(recipe,'cuda');old_plan.update(code=code_inventory(package),recipe_sha256=lineage)
    future_value=_recipe_plan(recipe,'cuda');future_value['code']=code_inventory(package)
    lineage_value=lineage_handoff(old_plan,future_value,package,package,tmp_path/'lineage.json',
        out=tmp_path,dry_run=True,started_stage_ids=['dpo','full'])
    lineage_path=tmp_path/'lineage-handoff.json';future_path=tmp_path/'future-plan.json'
    write(lineage_path,lineage_value);write(future_path,future_value)
    request={'schema':retry.SCHEMA,'config':binding(config_path),'plan':binding(plan),'recipe':binding(recipe),
        'future_global_plan':None,'role':'mellum','package':str(package.resolve()),'failed_request':failed_request,
        'failed_result':failed_result,'failure_log':failure_log,'corrected_request':corrected,
        'failure_kind':retry.RESIDENT_PLAN_FAILURE_KIND,'stale_resident_plan':stale,
        'lineage_handoff':binding(lineage_path),'future_plan':binding(future_path),
        'started_stage_ids':['dpo','full'],
        'failure_log_snapshot':tx_log,'controller':controller,'owner_census':census,
        'parent':original_cursor['current_checkpoint'],'stage_id':'full','next_stage':1,
        'lineage_handoff_sha256':retry._digest(lineage_value),
        'training_credit':0}
    intent={'schema':retry.SCHEMA,'request':request,'before':before,'snapshots':snapshots,'candidates':candidates,
        'failure_log_source':source_log,'failure_log_snapshot':tx_log,
        'whole_recipe_completed':False,'training_credit':0,'readiness_granted':False,'failed_stage_index':1}
    intent_path=tmp_path/'intent.json';write(intent_path,intent)
    receipt_path=tmp_path/'retry.json';receipt={'schema':retry.SCHEMA,'intent':binding(intent_path),
        'request_sha256':retry._digest(request),'stage_id':'full','checkpoint':request['parent'],
        'training_credit':0,'whole_recipe_completed':False,'readiness_granted':False}
    write(receipt_path,receipt);reference=binding(receipt_path)
    declaration={'failed_supervised_retry':reference,'production_recipe':binding(recipe)}
    declaration['handoff_requirement']={'recipe_sha256':'old-recipe','recipe_lineage_sha256':lineage,
        'stage_ids':['full'],'model_id':'fixture'}
    from natlang_neuralese.rl import native_completed_supervised as completed
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.rl import native_bootstrap_metadata as metadata
    monkeypatch.setattr(completed,'_no_owner',lambda owner:None)
    monkeypatch.setattr(resident,'owner_alive',lambda owner:False)
    monkeypatch.setattr(metadata,'save',lambda *args,**kwargs:None)
    authority=retry.verify_release(reference,declaration)
    assert authority['recipe_sha256']==lineage
    assert authority['prior_resident_current']==binding(predecessor_resident)
    declaration['failed_supervised_retry']=str(receipt_path) # production plan rows use path strings
    from natlang_neuralese.rl.native_raw_rounds import _release_requirement
    released=_release_requirement(declaration,{'resident_current':binding(predecessor_resident)})
    assert released['recipe_sha256']==lineage and released['recipe_lineage_sha256']==lineage
    lease=(root.parent/'controller.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        # A partial config write is the only known successor; remaining files
        # are recovered from their exact immutable candidates.
        from natlang_neuralese.common.jsonio import write_bytes_durable
        write_bytes_durable(config_path,Path(candidates['config']['path']).read_bytes())
        retry.configure(reference,plan,declaration,{'binding':'fixture'},config_path,owned_controller_fd=lease.fileno())
        assert binding(config_path)['sha256']==candidates['config']['sha256']
        assert binding(cursor_path)['sha256']==candidates['cursor']['sha256']
        assert binding(state_path)['sha256']==candidates['state']['sha256']
        assert json.loads((recipe.parent/'recipe-plan.json').read_text())==lineage_value
        release=retry.verify_release(reference,declaration)
        assert release['recipe_lineage_sha256']==lineage
        dependencies=retry.dependencies(reference)
        assert dependencies[snapshots['config']['path']]==snapshots['config']
        assert config_path.as_posix() not in dependencies
        retry.configure(reference,plan,declaration,{'binding':'fixture'},config_path,owned_controller_fd=lease.fileno())
    finally:
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    # An unrelated successor is rejected before any further writes.
    write(cursor_path,{'successor':'unrecognized'})
    with pytest.raises(ValueError,match='unknown controller successor'):
        retry.configure(reference,plan,declaration,{'binding':'fixture'},config_path)
    # Restore exact controls, then model a successful retry followed by normal
    # recipe adoption before the next coordinator configure call.
    completed_cursor=copy.deepcopy(candidate_cursor);completed_cursor.pop('failed_supervised_retry')
    completed_cursor['failed_supervised_retry_history']=[
        {'receipt':reference,'stage_id':'full','checkpoint':request['parent'],
         'training_credit':0,'readiness_granted':False}]
    completed_cursor.pop('failed_supervised_retry_consumed',None)
    write(cursor_path,completed_cursor)
    evolved=tmp_path/'evolved-recipe.json';write(evolved,{'recipe':'resident append'})
    adopted=json.loads(config_path.read_text());adopted['production_recipe']=str(evolved)
    adopted['resident_data_directory']=str(tmp_path/'resident-data');write(config_path,adopted)
    retry.configure(reference,plan,declaration,{'binding':'fixture'},config_path)
    assert 'failed_supervised_retry' not in json.loads(config_path.read_text())


def test_prepare_rejects_live_captured_owner_before_writing_retry_intent(tmp_path):
    from natlang_neuralese.train.native_resident import process_owner
    production=tmp_path/'owner'/'production';production.mkdir(parents=True)
    config_path=production.parent/'config.json'
    write(config_path,{'out':str(production)})
    write(tmp_path/'plan.json',{'plan':'fixture'})
    before=config_path.read_bytes();output=tmp_path/'retry.json'
    with pytest.raises(ValueError,match='owners to have exited'):
        retry.prepare(config_path,tmp_path/'plan.json','mellum',{}, {}, {}, {}, {},output,
            package=tmp_path,future_global_plan=None,controller={'pid':99999999},
            owner_census=[process_owner()],verified_checkpoint=({},{}))
    assert config_path.read_bytes()==before and not output.exists()


def test_raw_configure_dispatches_first_retry_before_runtime_handoff_without_marker(
        tmp_path,monkeypatch,failed_operation):
    from natlang_neuralese.rl import native_raw_rounds as raw
    from natlang_neuralese.rl import native_runtime_handoff as runtime
    x=failed_operation;config_path=x['config'];root=x['root']
    config=json.loads(config_path.read_text())
    runtime_path=tmp_path/'prior-runtime-handoff.json';write(runtime_path,{'schema':'fixture'})
    prior_handoff=binding(runtime_path);config['runtime_handoff']=prior_handoff;write(config_path,config)
    plan=tmp_path/'plan.json';write(plan,{'plan':'fixture'})
    request={'schema':retry.SCHEMA,'config':binding(config_path)}
    intent_path=tmp_path/'intent.json';write(intent_path,{'schema':retry.SCHEMA,'request':request})
    receipt_path=tmp_path/'retry.json';write(receipt_path,{'schema':retry.SCHEMA,
        'intent':binding(intent_path),'request_sha256':retry._digest(request)})
    reference=binding(receipt_path)
    declaration={'cases':'fixture','failed_supervised_retry':reference,'runtime_handoff':prior_handoff}
    monkeypatch.setattr(raw,'_task_cases',lambda declaration:tmp_path/'cases.jsonl')
    calls=[]
    monkeypatch.setattr(retry,'configure',lambda *args,**kwargs:calls.append((args,kwargs)) or config_path)
    monkeypatch.setattr(runtime,'configure_runtime_handoff',
        lambda *args,**kwargs:pytest.fail('first retry was misrouted to runtime-handoff reconfigure'))
    assert 'failed_supervised_retry' not in config
    result=raw.configure_model(plan,declaration,{'binding':'bootstrap'},root)
    assert result==config_path and len(calls)==1
    assert calls[0][0][0]==reference and calls[0][0][1]==plan
