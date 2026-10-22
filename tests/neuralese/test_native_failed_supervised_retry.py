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


def _recipe(tmp_path):
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    train.write_text('{"id":"train"}\n');held.write_text('{"id":"held","split":"test"}\n')
    value={'schema':'natlang.neuralese-training-recipe/1','id':'retry-fixture',
        'input_bindings':{'train':binding(train),'held':binding(held)},
        'stages':[{'id':stage,'kind':'native_state_sft','requires':[],
            'inputs':{'native_graphs':'train','held_native_graphs':'held'},'parameters':{'epochs':1}}
            for stage in ('dpo','full')]}
    path=tmp_path/'recipe.json';write(path,value);return path


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
    write(result_path,{'schema':bridge.RESULT_SCHEMA,'kind':'round','request_id':request['request_id'],
        'request_sha256':bridge.sha256_file_hex_cached(request_path),'status':'failed',
        'error':retry.FAILURE,'artifacts':bridge.inventory(job,exclude=[result_path])})
    saved={'model_config':{'architecture':model,'base_inventory_sha256':'a'*64},
        'training':{'global_step':31716,'schema':'fixture','sft_progress':{
            'dpo':{'exposure_counts':{'row':250}},'full':{'exposure_counts':{'row':0}}}}}
    return {'root':root,'config':root/'config.json','cursor':cursor,'state':state,
        'checkpoint':checkpoint,'saved':saved,'request':request_path,'result':result_path,
        'job':job,'invocation':invocation}


def test_exact_known_preupdate_failure_has_zero_checkpoint_authority(failed_operation):
    x=failed_operation
    config=json.loads(x['config'].read_text())
    result=retry._verify_failure(binding(x['request']),binding(x['result']),config=config,cursor=x['cursor'],
        state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
    assert result[3]['id']=='full'
    assert result[1]['status']=='failed' and result[1].get('checkpoint') is None
    baseline_result=json.loads(x['result'].read_text());baseline_request=x['request'].read_bytes()
    for change in ('status','error','checkpoint','parent'):
        x['request'].write_bytes(baseline_request)
        value=copy.deepcopy(baseline_result)
        if change=='status':value['status']='completed'
        elif change=='error':value['error']='ValueError: another failure'
        elif change=='checkpoint':value['checkpoint']=x['checkpoint']
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
            retry._verify_failure(binding(x['request']),binding(x['result']),config=config,cursor=x['cursor'],
                state=x['state'],verified_checkpoint=(x['checkpoint'],x['saved']))
    x['request'].write_bytes(baseline_request)


def test_corrected_capacity_identity_requires_ffn_chunk_and_exact_runtime(tmp_path):
    from natlang_neuralese.train.native_resident import code_inventory
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
    assert first['failed_supervised_retry_consumed']['training_credit']==0
    assert first['next_stage']==1 and first['current_checkpoint']==checkpoint
    assert first['earned_dpo_exposure']==before['earned_dpo_exposure']
    assert retry.consume_boundary(reference,{'failed_supervised_retry':cursor['failed_supervised_retry']},first,state,root)==first
    bad=copy.deepcopy(before);bad['next_stage']=0
    with pytest.raises(ValueError):retry.consume_boundary(reference,{},bad,state,root)
    bad=copy.deepcopy(before);bad['current_checkpoint']={'path':checkpoint['path'],'sha256':'d'*64}
    with pytest.raises(ValueError):retry.consume_boundary(reference,{},bad,state,root)
    badstate={'phase':'learner_pending'}
    with pytest.raises(ValueError):retry.consume_boundary(reference,{},before,badstate,root)


def test_configure_replays_exact_candidate_with_inherited_lease_and_recovers_partial_cas(tmp_path,monkeypatch):
    root=tmp_path/'owner';production=root/'production';production.mkdir(parents=True)
    config_path=root/'config.json';cursor_path=root/'cycle.json';state_path=production/'state.json'
    recipe=tmp_path/'recipe.json';plan=tmp_path/'plan.json'
    write(recipe,{'recipe':'bound'});write(plan,{'plan':'bound'})
    original_config={'out':str(production),'production_recipe':str(recipe),'raw_model_id':'fixture:a'*8,
        'trainer_parameters':{'memory_qualification':'old'},'runtime_binding':{'old':True}}
    original_cursor={'phase':'sft','next_stage':1,'raw_retry_generation':3,'current_checkpoint':{'path':'p','sha256':'a'*64}}
    original_state={'phase':'collect','checkpoint':'p','training':{'global_step':10}}
    write(config_path,original_config);write(cursor_path,original_cursor);write(state_path,original_state)
    (tmp_path/'controller.lock').touch();(production/'controller.lock').touch()
    candidate_config={**original_config,'production_recipe':str(recipe),'trainer_parameters':{'memory_qualification':'new'},
        'failed_supervised_retry':{'receipt_path':str(tmp_path/'retry.json')}}
    candidate_cursor={**original_cursor,'raw_retry_generation':4,
        'failed_supervised_retry':{'receipt_path':str(tmp_path/'retry.json')}}
    candidates={};snapshots={};before={}
    for name,path,value in (('config',config_path,candidate_config),('cursor',cursor_path,candidate_cursor),('state',state_path,original_state)):
        before[name]=binding(path)
        snapshot=tmp_path/f'original-{name}.json';snapshot.write_bytes(path.read_bytes());snapshots[name]=binding(snapshot)
        candidate=tmp_path/f'candidate-{name}.json';write(candidate,value);candidates[name]=binding(candidate)
    failed_request={};failed_result={};corrected={};package=tmp_path/'package';package.mkdir()
    controller={'pid':99999999};census=[{'pid':99999998}]
    request={'schema':retry.SCHEMA,'config':binding(config_path),'plan':binding(plan),'recipe':binding(recipe),
        'future_global_plan':None,'role':'mellum','package':str(package.resolve()),'failed_request':failed_request,
        'failed_result':failed_result,'corrected_request':corrected,'controller':controller,'owner_census':census,
        'parent':original_cursor['current_checkpoint'],'stage_id':'full','next_stage':1,'lineage_handoff_sha256':'d'*64,
        'training_credit':0}
    intent={'schema':retry.SCHEMA,'request':request,'before':before,'snapshots':snapshots,'candidates':candidates,
        'whole_recipe_completed':False,'training_credit':0,'readiness_granted':False,'failed_stage_index':1}
    intent_path=tmp_path/'intent.json';write(intent_path,intent)
    receipt_path=tmp_path/'retry.json';receipt={'schema':retry.SCHEMA,'intent':binding(intent_path),
        'request_sha256':retry._digest(request),'stage_id':'full','checkpoint':request['parent'],
        'training_credit':0,'whole_recipe_completed':False,'readiness_granted':False}
    write(receipt_path,receipt);reference=binding(receipt_path)
    declaration={'failed_supervised_retry':reference,'production_recipe':binding(recipe)}
    from natlang_neuralese.rl import native_completed_supervised as completed
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.rl import native_bootstrap_metadata as metadata
    monkeypatch.setattr(completed,'_no_owner',lambda owner:None)
    monkeypatch.setattr(resident,'owner_alive',lambda owner:False)
    monkeypatch.setattr(metadata,'save',lambda *args,**kwargs:None)
    lease=(tmp_path/'controller.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        # A partial config write is the only known successor; remaining files
        # are recovered from their exact immutable candidates.
        from natlang_neuralese.common.jsonio import write_bytes_durable
        write_bytes_durable(config_path,Path(candidates['config']['path']).read_bytes())
        retry.configure(reference,plan,declaration,{'binding':'fixture'},config_path,owned_controller_fd=lease.fileno())
        assert binding(config_path)['sha256']==candidates['config']['sha256']
        assert binding(cursor_path)['sha256']==candidates['cursor']['sha256']
        assert binding(state_path)['sha256']==candidates['state']['sha256']
        retry.configure(reference,plan,declaration,{'binding':'fixture'},config_path,owned_controller_fd=lease.fileno())
    finally:
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    # An unrelated successor is rejected before any further writes.
    write(cursor_path,{'successor':'unrecognized'})
    with pytest.raises(ValueError,match='unknown controller successor'):
        retry.configure(reference,plan,declaration,{'binding':'fixture'},config_path)


def test_prepare_rejects_live_captured_owner_before_writing_retry_intent(tmp_path):
    from natlang_neuralese.train.native_resident import process_owner
    production=tmp_path/'owner'/'production';production.mkdir(parents=True)
    config_path=production.parent/'config.json'
    write(config_path,{'out':str(production)})
    write(tmp_path/'plan.json',{'plan':'fixture'})
    before=config_path.read_bytes();output=tmp_path/'retry.json'
    with pytest.raises(ValueError,match='owners to have exited'):
        retry.prepare(config_path,tmp_path/'plan.json','mellum',{}, {}, {}, {},output,
            package=tmp_path,future_global_plan=None,controller={'pid':99999999},
            owner_census=[process_owner()],verified_checkpoint=({},{}))
    assert config_path.read_bytes()==before and not output.exists()
