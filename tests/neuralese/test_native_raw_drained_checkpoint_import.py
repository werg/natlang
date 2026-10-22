"""A verified drained full parent resumes the existing RAW debt safely."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_raw_handoff as handoff_module
from natlang_neuralese.rl import native_raw_rounds as raw
from natlang_neuralese.train import native_resident as resident


@pytest.fixture(params=[('mellum21:base-a','initial'),('lfm2:base-b','prefix')])
def case(request,tmp_path,monkeypatch):
    model_name,recipe_position=request.param
    root=tmp_path/'owner';production=root/'production';production.mkdir(parents=True)
    train=root/'train.jsonl';train.write_text('{"split":"train"}\n')
    held=root/'held.jsonl';held.write_text('{"split":"held"}\n')
    delta=root/'delta.jsonl';delta.write_text('{"split":"train","id":"delta"}\n')
    initial_train=root/'initial-train.json';write(initial_train,{'schema':'natlang.native-graph-sources/1','sources':[resident.binding(train)]})
    initial_held=root/'initial-held.json';write(initial_held,{'schema':'natlang.native-graph-sources/1','sources':[resident.binding(held)]})
    prefix_train=root/'prefix-train.json';write(prefix_train,{'schema':'natlang.native-graph-sources/1','sources':[resident.binding(initial_train),resident.binding(delta)]})
    prefix_held=root/'prefix-held.json';write(prefix_held,{'schema':'natlang.native-graph-sources/1','sources':[resident.binding(initial_held)]})
    final_train=root/'final-train.json';write(final_train,{'schema':'natlang.native-graph-sources/1','sources':[resident.binding(prefix_train)]})
    final_held=root/'final-held.json';write(final_held,{'schema':'natlang.native-graph-sources/1','sources':[resident.binding(prefix_held)]})
    initial_recipe=root/'initial-recipe.json';write(initial_recipe,{'schema':'fixture-recipe','version':'initial'})
    prefix_recipe=root/'prefix-recipe.json';write(prefix_recipe,{'schema':'fixture-recipe','version':'prefix'})
    final_recipe=root/'final-recipe.json';write(final_recipe,{'schema':'fixture-recipe','version':'final'})
    config_recipe=initial_recipe if recipe_position=='initial' else prefix_recipe
    config_train=train
    config_held=held
    config_path=root/'config.json';config={'out':str(production),'raw_model_id':model_name,
        'recipe_lineage_sha256':'old-lineage','recipe_lineage':{'opaque':'preserved'},
        'production_recipe':str(config_recipe),'production_stage':'stage-a',
        'seed_native_graphs':str(config_train),'held_native_graphs':str(config_held)}
    write(config_path,config)
    old=root/'old.pt';old.write_bytes(b'old-state')
    new=root/'drained-full.pt';new.write_bytes(b'full-model-optimizer-rng-exposure')
    oldref=resident.binding(old);newref={**resident.binding(new),'bytes':new.stat().st_size}
    state_path=production/'state.json';cursor_path=root/'cycle.json'
    state={'schema':'natlang.rl-production-state/1','checkpoint':str(old),'phase':'collect',
        'trained_task_ids':['a'],'trained_supervised_tokens':77,'optimizer_receipt':{'step':30291},
        'exposure':{'stage-a':{'updates':11}},'readiness':{'stale':True}}
    cursor={'cycle':11,'phase':'sft','next_stage':1,'current_checkpoint':oldref,
        'task_cases':{'pending':['held-out? no, train-c']},'pending_collection':{'offset':16},
        'pending_rl_round':None,'readiness':{'stale':True},'foundation_qualified':True}
    write(state_path,state);write(cursor_path,cursor)
    (root/'cycles'/'000011').mkdir(parents=True)
    intent_path=root/'drain-intent.json';write(intent_path,{'schema':'natlang.native-raw-optimizer-drain/2',
        'owner':{'pid':123,'boot_id':'boot','start_ticks':'123','argv_hex':'00'},'children':[]})
    hpath=root/'handoff.json';write(hpath,{'test':'handoff'})
    record={'schema':'natlang.native-raw-optimizer-handoff/1','checkpoint':newref,
        'intent':resident.binding(intent_path)}
    saved={'training':{'global_step':31400,'optimizer':{'moments':'retained'},
            'rng':{'torch':'retained'},'sft_progress':{'stage-a':{'exposure':77}}},
        'model_config':{'architecture':model_name.split(':')[0],'base_inventory_sha256':model_name.split(':')[1]}}
    transition_path=root/'new-plan.json';write(transition_path,{'recipe_sha256':'recipe-lineage',
        'stage_inputs':{'stage-a':{'native_graphs':resident.binding(final_train),'held_native_graphs':resident.binding(final_held)}}})
    rh_path=root/'recipe-handoff.json';write(rh_path,{'schema':'natlang.native-raw-drained-recipe-handoff/4'})
    old_plan_path=root/'old-plan.json';write(old_plan_path,{'recipe_sha256':'old-lineage','code':{},'device':'cpu'})
    initial_path=root/'resident-initial.json';write(initial_path,{'recipe':resident.binding(initial_recipe)})
    initial_ref={'recipe':resident.binding(initial_recipe)}
    history=[{'recipe':{'path':str(prefix_recipe.resolve()),'file_sha256':resident.sha(prefix_recipe)},
        'train':resident.binding(prefix_train),'held':resident.binding(prefix_held)}]
    current_path=root/'resident-current.json';write(current_path,{'payload':{
        'prior_recipe':resident.binding(initial_recipe),
        'recipe':{'path':str(final_recipe.resolve()),'file_sha256':resident.sha(final_recipe)},
        'train':resident.binding(final_train),'held':resident.binding(final_held)},'history':history})
    record['resident']={'initial':resident.binding(initial_path),'current':resident.binding(current_path)}
    transition={'schema':handoff_module.RECOVERED_RECIPE_HANDOFF,'new_plan':resident.binding(transition_path),
        'old_plan':resident.binding(old_plan_path),
        'old_package':str(root/'historical-package'),
        'external_runtime':{'executed_runtime':{'package':'/frozen/executed'}}}
    monkeypatch.setattr(raw,'verify_handoff',lambda ref:(record,saved))
    monkeypatch.setattr(handoff_module,'verify_handoff_controls',lambda ref:record)
    monkeypatch.setattr(handoff_module,'verify_drained_recipe_handoff',lambda ref,**kwargs:transition)
    historical_inputs={str(initial_recipe.resolve()):{
        'native_graphs':resident.binding(train),'held_native_graphs':resident.binding(held)},
        str(prefix_recipe.resolve()):{'native_graphs':resident.binding(train),
            'held_native_graphs':resident.binding(held)}}
    monkeypatch.setattr(handoff_module,'_package_recipe_plans',lambda package,code,refs,device:[{'stage_inputs':{
        'stage-a':historical_inputs[refs[0]['path']]}}])
    monkeypatch.setattr('natlang_neuralese.train.native_resident.owner_alive',lambda owner:False)
    monkeypatch.setattr('natlang_neuralese.rl.native_raw_policy.require_raw_training',lambda loaded:loaded['training'])
    return {'root':root,'config':config_path,'state':state_path,'cursor':cursor_path,
        'handoff':resident.binding(hpath),'recipe_handoff':resident.binding(rh_path),
        'new':newref,'old_state':state,'old_cursor':cursor,'receipt':root/'import.json',
        'train':train,'held':held,'config_held':config_held}


def call(case):
    return raw.import_drained_checkpoint(case['config'],case['cursor'],case['handoff'],
        case['recipe_handoff'],case['receipt'],recovery_reason='actual optimizer drained at H8; resume mandatory recipe with no completion credit')


def test_import_preserves_full_recipe_debt_and_reserves_unused_operation_namespace(case):
    call(case)
    state=json.loads(case['state'].read_text());cursor=json.loads(case['cursor'].read_text())
    assert state['checkpoint']==case['new']['path'] and state['actor_policy_pending_load']==case['new']['sha256']
    for key,value in case['old_state'].items():
        if key not in ('checkpoint','readiness','actor_policy_pending_load'):assert state[key]==value
    assert cursor['current_checkpoint']==case['new']
    assert cursor['phase']=='sft' and cursor['next_stage']==0
    assert cursor['raw_retry_generation']==1
    assert cursor['task_cases']==case['old_cursor']['task_cases']
    assert cursor['pending_collection']==case['old_cursor']['pending_collection']
    assert not cursor.get('readiness') and 'foundation_qualified' not in cursor
    receipt=json.loads(case['receipt'].read_text())
    assert receipt['whole_recipe_completed'] is False and receipt['training_credit']==0
    assert receipt['raw_completion_credit']==0 and receipt['readiness_granted'] is False
    assert call(case)==resident.binding(case['receipt'])


@pytest.mark.parametrize('mutation',['pending-rl','active-operation','cursor-mismatch'])
def test_import_rejects_live_or_inconsistent_work(case,mutation):
    if mutation=='pending-rl':
        value=json.loads(case['cursor'].read_text());value['pending_rl_round']=12;write(case['cursor'],value)
    elif mutation=='active-operation':
        value=json.loads(case['state'].read_text());value['pending_native_checkpoint']='pending.pt';write(case['state'],value)
    else:
        value=json.loads(case['cursor'].read_text());value['current_checkpoint']=case['new'];write(case['cursor'],value)
    with pytest.raises(ValueError):call(case)


def test_interrupted_two_file_publication_resumes_from_durable_intent(case,monkeypatch):
    original=raw.write_bytes_durable;failed={'done':False}
    def interrupt(path,data):
        if Path(path)==case['cursor'] and not failed['done']:
            failed['done']=True
            raise OSError('injected interruption before cursor publication')
        return original(path,data)
    monkeypatch.setattr(raw,'write_bytes_durable',interrupt)
    with pytest.raises(OSError,match='injected interruption'):call(case)
    assert json.loads(case['state'].read_text())['checkpoint']==case['new']['path']
    monkeypatch.setattr(raw,'write_bytes_durable',original)
    call(case)
    assert json.loads(case['cursor'].read_text())['current_checkpoint']==case['new']


def test_import_rejects_unknown_successor_after_snapshot(case,monkeypatch):
    original=raw.write_json_durable;changed={'done':False}
    def race(path,value):
        original(path,value)
        if Path(path).name=='intent.json' and not changed['done']:
            changed['done']=True
            successor=json.loads(case['cursor'].read_text());successor['foreign_successor']=True
            write(case['cursor'],successor)
    monkeypatch.setattr(raw,'write_json_durable',race)
    with pytest.raises(ValueError,match='concurrent controller successor'):
        call(case)


@pytest.mark.parametrize('mutation',['live-owner','wrong-model','changed-held-source'])
def test_import_rejects_owner_model_or_recipe_source_mismatch(case,mutation,monkeypatch):
    if mutation=='live-owner':
        monkeypatch.setattr('natlang_neuralese.train.native_resident.owner_alive',lambda owner:True)
    elif mutation=='wrong-model':
        value=json.loads(case['config'].read_text());value['raw_model_id']='another-model:another-base';write(case['config'],value)
    else:
        case['config_held'].write_text('{"split":"held","changed":true}\n')
    with pytest.raises(ValueError):call(case)

