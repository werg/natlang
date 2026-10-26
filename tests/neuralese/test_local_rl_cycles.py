import json
from pathlib import Path
import pytest
from natlang_neuralese.rl import local_cycles as lc


def test_round_publication_before_cursor_commit_is_not_repeated():
    assert lc.round_target(4,5)
    assert not lc.round_target(5,5)
    with pytest.raises(ValueError,match='unknown controller successor'):
        lc.round_target(6,5)


def test_legacy_completed_round_markers_normalize_only_with_current_report_and_rollouts(tmp_path):
    from natlang_neuralese.rl.production import normalize_legacy_completed_update
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.common.jsonio import write_json_durable
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'exact current full checkpoint')
    round_dir=tmp_path/'round-000010';round_dir.mkdir()
    rollouts=round_dir/'rollouts.jsonl';rollouts.write_text('{"sealed":true}\n')
    report=round_dir/'learner'/'report.json';report.parent.mkdir()
    digest=sha256_file_hex(checkpoint)
    write_json_durable(report,{'completed':True,'numerics_ok':True,'checkpoint_sha256':digest,
        'progress_numerics':{'passed':True}})
    state_path=tmp_path/'state.json'
    state={'phase':'collect','round':11,'trained_rounds':11,'checkpoint':str(checkpoint),
        'active_update_round':str(round_dir),'update_mode':'actor-learner',
        'update_rollouts':str(rollouts),'update_stages':['policy'],
        'last_learner_progress':{'completed':True,'checkpoint_sha256':digest,
            'full_report':{'path':str(report),'sha256':sha256_file_hex(report),'bytes':report.stat().st_size}}}
    write_json_durable(state_path,state)
    assert normalize_legacy_completed_update(state,state_path) is True
    assert not any(key.startswith(('active_update_','partial_update_'))for key in state)
    audit=state['last_completed_update']
    assert audit['legacy_marker_recovery'] is True and audit['round']==10
    assert audit['checkpoint']['sha256']==digest


def test_legacy_completed_round_accepts_only_reported_current_sft_descendant(tmp_path):
    from natlang_neuralese.rl.production import normalize_legacy_completed_update
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.common.jsonio import write_json_durable
    parent=tmp_path/'round-parent.pt';parent.write_bytes(b'completed round checkpoint')
    child=tmp_path/'sft-child.pt';child.write_bytes(b'published SFT descendant')
    round_dir=tmp_path/'round-000010';(round_dir/'learner').mkdir(parents=True)
    rollouts=round_dir/'rollouts.jsonl';rollouts.write_text('sealed\n')
    parent_sha=sha256_file_hex(parent);child_sha=sha256_file_hex(child)
    old_report=round_dir/'learner/report.json'
    write_json_durable(old_report,{'completed':True,'numerics_ok':True,'checkpoint_sha256':parent_sha,
        'progress_numerics':{'passed':True}})
    latest_report=tmp_path/'cycle-11/sft/report.json';latest_report.parent.mkdir(parents=True)
    write_json_durable(latest_report,{'completed':False,'numerics_ok':True,'checkpoint_sha256':child_sha,
        'continuation':{'checkpoint_sha256':parent_sha},'progress_numerics':{'passed':True}})
    state_path=tmp_path/'state.json'
    state={'phase':'collect','round':11,'trained_rounds':11,'checkpoint':str(child),
        'active_update_round':str(round_dir),'update_mode':'grpo','update_rollouts':str(rollouts),
        'last_learner_progress':{'completed':False,'checkpoint_sha256':child_sha,
            'full_report':{'path':str(latest_report),'sha256':sha256_file_hex(latest_report),
                'bytes':latest_report.stat().st_size}}}
    write_json_durable(state_path,state)
    assert normalize_legacy_completed_update(state,state_path) is True
    assert state['last_completed_update']['checkpoint']['sha256']==child_sha
    assert state['last_completed_update']['completion_report']['sha256']==sha256_file_hex(old_report)


@pytest.mark.parametrize('mutation',[
    {'phase':'updating'}, {'pending_native_checkpoint':'/tmp/uncommitted.pt'},
])
def test_legacy_round_marker_refuses_partial_or_noncollection_state(tmp_path,mutation):
    from natlang_neuralese.rl.production import normalize_legacy_completed_update
    state_path=tmp_path/'state.json';state_path.write_text('{}')
    state={'phase':'collect','round':11,'trained_rounds':11,'active_update_round':str(tmp_path/'round-000010'),**mutation}
    with pytest.raises(ValueError,match='clean completed collection boundary'):
        normalize_legacy_completed_update(state,state_path)


def test_supervised_publication_retries_exact_child_but_rejects_foreign_successor(tmp_path,monkeypatch):
    state=tmp_path/'state.json';parent=tmp_path/'parent.pt';child=tmp_path/'child.pt';foreign=tmp_path/'foreign.pt'
    for p in (parent,child,foreign):p.write_bytes(p.name.encode())
    config={'out':str(tmp_path)};calls=[]
    def publish(config,checkpoint):
        calls.append(checkpoint);state.write_text(json.dumps({'checkpoint':checkpoint['path']}))
    monkeypatch.setattr(lc,'_publish_checkpoint',publish)
    state.write_text(json.dumps({'checkpoint':str(parent)}))
    lc.publish_supervised(config,lc.binding(parent),lc.binding(child))
    lc.publish_supervised(config,lc.binding(parent),lc.binding(child))
    assert calls==[lc.binding(child)]
    state.write_text(json.dumps({'checkpoint':str(foreign)}))
    with pytest.raises(ValueError,match='unknown policy successor'):
        lc.publish_supervised(config,lc.binding(parent),lc.binding(child))


def test_failed_worker_can_yield_to_useful_work_only_without_saved_child(tmp_path):
    import subprocess
    parent=tmp_path/'parent.pt';parent.write_bytes(b'earned-full-state')
    recipe=tmp_path/'recipe.json';recipe.write_text('{}')
    (tmp_path/'state.json').write_text(json.dumps({'checkpoint':str(parent)}))
    config={'out':str(tmp_path),'production_recipe':str(recipe)}
    stage={'id':'dpo','parameters':{'epochs':1}}
    operation=tmp_path/'operation';operation.mkdir();cursor={'stage_index':1}
    error=subprocess.CalledProcessError(1,['actual-worker'])
    failure=lc.record_unstarted_stage_failure(cursor,stage,config,operation,lc.binding(parent),error,enabled=True)
    assert cursor['stage_index']==2 and not failure['completed']
    assert failure['signature']==lc.stage_failure_signature(stage,config)
    changed={**stage,'parameters':{'epochs':2}}
    assert failure['signature']!=lc.stage_failure_signature(changed,config)
    (operation/'learner').mkdir();(operation/'learner/checkpoint.pt').write_bytes(b'newer-earned-state')
    before=json.loads(json.dumps(cursor))
    with pytest.raises(subprocess.CalledProcessError):
        lc.record_unstarted_stage_failure(cursor,stage,config,operation,lc.binding(parent),error,enabled=True)
    assert cursor==before


def test_cycle_reloads_recipe_after_rl_and_before_each_unreserved_stage(tmp_path,monkeypatch):
    production=tmp_path/'production';production.mkdir()
    parent=tmp_path/'parent.pt';parent.write_bytes(b'earned-parent')
    state_path=production/'state.json';state_path.write_text(json.dumps({'round':0,'checkpoint':str(parent)}))
    config_path=tmp_path/'config.json';plan_path=tmp_path/'plan.json';cycle_out=tmp_path/'cycles'
    recipes=[]
    for index in range(3):
        path=tmp_path/f'recipe-{index}.json'
        path.write_text(json.dumps({'stages':[
            {'id':'sft','kind':'native_state_sft','parameters':{'data_revision':index}},
            {'id':'preference','kind':'native_offline_preference','parameters':{'data_revision':index}}]}))
        recipes.append(path)
    config={'out':str(production),'native_actor_mode':'raw-state','production_recipe':str(recipes[0]),'model':'lfm'}
    plan={'out':str(cycle_out),'optimizer_owner_lock':str(tmp_path/'owner.lock'),
          'rl_config':str(config_path),'supervised_quantum_updates':50,
          'supervised_stage_quantum_updates':{'sft':500,'preference':100}}
    config_path.write_text(json.dumps(config));plan_path.write_text(json.dumps(plan));seen=[]
    monkeypatch.setattr(lc,'resolve_recipe_data',lambda p:json.loads(Path(p).read_text()))
    def continuation(config,stage,*,checkpoint,adapter,out):
        seen.append((stage['id'],stage['parameters']['data_revision'],checkpoint))
        return {'out':str(out),'stage_id':stage['id'],'trainer_parameters':{'stage_id':stage['id']}},None
    monkeypatch.setattr(lc,'continuation_config',continuation)
    quanta=[]
    def quantum(config,parameters):quanta.append(config['supervised_stage_quantum_updates'][parameters['stage_id']])
    monkeypatch.setattr(lc,'_supervised_quantum',quantum)
    def worker(path,**kwargs):
        path=Path(path)
        if path==config_path:
            # Data and cadence become ready while the existing RL round runs.
            config['production_recipe']=str(recipes[1]);config_path.write_text(json.dumps(config))
            plan['supervised_stage_quantum_updates']['sft']=100;plan_path.write_text(json.dumps(plan))
            state=json.loads(state_path.read_text());state['round']=1;state_path.write_text(json.dumps(state));return state
        declaration=json.loads(path.read_text());directory=path.parent/'learner';directory.mkdir()
        child=directory/'checkpoint.pt';child.write_bytes(declaration['stage_id'].encode())
        if declaration['stage_id']=='sft':
            # Another ready append arrives before preference is reserved.
            config['production_recipe']=str(recipes[2]);config_path.write_text(json.dumps(config))
            plan['supervised_stage_quantum_updates']['preference']=25;plan_path.write_text(json.dumps(plan))
        return {'pending_native_checkpoint':str(child)}
    monkeypatch.setattr(lc.production,'run',worker)
    def publish(config,checkpoint):
        state=json.loads(state_path.read_text());state['checkpoint']=checkpoint['path'];state_path.write_text(json.dumps(state))
    monkeypatch.setattr(lc,'_publish_checkpoint',publish)
    lc.run(plan_path,cycles=1)
    assert [(stage,revision)for stage,revision,_ in seen]==[('sft',1),('preference',2)]
    assert quanta==[100,25]
    assert seen[0][2]==str(parent)
    assert seen[1][2]==str(cycle_out/'cycle-000000/sft/learner/checkpoint.pt')
    cursor=json.loads((cycle_out/'cursor.json').read_text())
    assert cursor['cycle']==1 and cursor['phase']=='rl'and cursor['rl_round_target']==2


def test_reload_rejects_owner_runtime_and_pending_stage_replacement(tmp_path,monkeypatch):
    plan_path=tmp_path/'plan.json';config_path=tmp_path/'config.json';recipe_path=tmp_path/'recipe.json'
    stages=[{'id':name,'kind':'native_state_sft'}for name in('sft','recovery')]
    recipe_path.write_text(json.dumps({'stages':stages}))
    plan={'out':str(tmp_path/'cycles'),'optimizer_owner_lock':str(tmp_path/'owner.lock'),'rl_config':str(config_path)}
    config={'out':str(tmp_path/'production'),'native_actor_mode':'raw-state','model':'lfm',
            'production_recipe':str(recipe_path),'native_actor_python_runtime':{'sha256':'current'}}
    plan_path.write_text(json.dumps(plan));config_path.write_text(json.dumps(config))
    monkeypatch.setattr(lc,'resolve_recipe_data',lambda p:json.loads(Path(p).read_text()))
    lc.reload_schedule(plan_path,plan,config,stages,cursor={'cycle':0,'stage_index':0})
    for change in ({'out':str(tmp_path/'foreign')},{'native_actor_python_runtime':{'sha256':'other'}},
                   {'native_checkpoint_publication_owner_pointer':str(tmp_path/'foreign-owner.json')},
                   {'native_checkpoint_publication_runtime_root':str(tmp_path/'foreign-runtime')},
                   {'native_checkpoint_publication_role':'mellum'},
                   {'native_participant':'mellum'},
                   {'native_checkpoint_publication':{'namespace':str(tmp_path/'foreign-publication')}}):
        config_path.write_text(json.dumps({**config,**change}))
        with pytest.raises(ValueError,match='ownership'):
            lc.reload_schedule(plan_path,plan,config)
    config_path.write_text(json.dumps(config))
    roots={'native_checkpoint_publication':{'readonly_input_roots':['/original']}}
    config_path.write_text(json.dumps({**config,'native_checkpoint_publication':{
        'readonly_input_roots':['/new','/original']}}))
    lc.reload_schedule(plan_path,plan,{**config,**roots})
    config_path.write_text(json.dumps({**config,'native_checkpoint_publication':{
        'readonly_input_roots':['/replacement']}}))
    with pytest.raises(ValueError,match='ownership'):
        lc.reload_schedule(plan_path,plan,{**config,**roots})
    config_path.write_text(json.dumps(config))
    directory=tmp_path/'cycles/cycle-000000/sft';directory.mkdir(parents=True)
    (directory/'config.json').write_text('{"actual":"frozen-operation"}')
    recipe_path.write_text(json.dumps({'stages':[stages[1]]}))
    with pytest.raises(ValueError,match='pending operation'):
        lc.reload_schedule(plan_path,plan,config,stages,cursor={'cycle':0,'stage_index':0})
    with pytest.raises(ValueError,match='visited stage prefix'):
        lc.reload_schedule(plan_path,plan,config,stages,cursor={'cycle':0,'stage_index':1})


def test_semantic_return_recovers_interrupted_pointer_commit_without_losing_debts(tmp_path,monkeypatch):
    from natlang_neuralese.rl import local_cycle_semantic_return as returned
    from natlang_neuralese.rl import native_role_release as roles
    from natlang_neuralese.common.jsonio import write_json_durable

    exchange_dir=tmp_path/'exchange';exchange_dir.mkdir()
    role_plan=tmp_path/'role-plan.json';role_plan.write_text('{}')
    role_release={'path':str(exchange_dir/'lfm-proof.json'),'sha256':'a'*64}
    (exchange_dir/'lfm-status.json').write_text(json.dumps(role_release))
    monkeypatch.setattr(roles,'ownership',lambda plan:( {'lfm'},
        {'role':'lfm','directory':str(exchange_dir),'roles':{'lfm':{} }}))

    parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
    child=tmp_path/'returned.pt';child.write_bytes(b'returned full-state child')
    adapter=tmp_path/'semantic-adapter.json';adapter.write_text('{}')
    receipt_dir=tmp_path/'returns';receipt_dir.mkdir()
    receipt_path=receipt_dir/'lfm-return.json'
    receipt_path.write_text(json.dumps({'role_plan':__import__(
        'natlang_neuralese.rl.native_raw_handoff',fromlist=['binding']).binding(role_plan)}))
    plan={'semantic_return':{'return_directory':str(receipt_dir),'role_plan':str(role_plan),'role':'lfm'}}
    config={'native_participant':'lfm'}
    state_path=tmp_path/'production-state.json';cursor_path=tmp_path/'cursor.json'
    state={'schema':'natlang.rl-production-state/1','round':9,'offset':7,
        'checkpoint':str(parent),'trained_task_ids':['earned-task'],'trained_supervised_tokens':123}
    cursor={'cycle':3,'phase':'supervised','stage_index':2,'rl_round_target':10,
        'failed_stages':{'old-stage':{'completed':False}}}
    write_json_durable(state_path,state);write_json_durable(cursor_path,cursor)
    checkpoint={'path':str(child.resolve()),'sha256':__import__('hashlib').sha256(child.read_bytes()).hexdigest()}
    semantic={'path':str(adapter.resolve()),'sha256':__import__('hashlib').sha256(adapter.read_bytes()).hexdigest()}
    from natlang_neuralese.train.native_resident import binding as file_binding
    pointer={'schema':'natlang.native-local-cycle-return-pointer/1','participant':'lfm',
        'receipt':file_binding(receipt_path),'checkpoint':checkpoint,'previous':None,'ordinal':1,'training_credit':0}
    write_json_durable(receipt_dir/'lfm-return-current.json',pointer)
    monkeypatch.setattr(returned,'validate_return',lambda *args,**kwargs:{
        'checkpoint':checkpoint,'semantic_adapter':semantic,'parent':{'path':str(parent),
            'sha256':__import__('hashlib').sha256(parent.read_bytes()).hexdigest()},
        'completion':{'path':'completion.json','sha256':'b'*64},
        'semantic_refresh_config':{'path':'refresh.json','sha256':'c'*64},'role':'lfm'})
    original=returned.write_json_durable
    def interrupted(path,value):
        if Path(path)==cursor_path:
            raise OSError('simulated crash between durable owner-state and cursor writes')
        return original(path,value)
    monkeypatch.setattr(returned,'write_json_durable',interrupted)
    with pytest.raises(OSError,match='simulated crash'):
        returned.adopt_at_boundary(config,plan,cursor,state_path=state_path,cursor_path=cursor_path)
    committed=json.loads(state_path.read_text())
    assert committed['checkpoint']==checkpoint['path'] and committed['semantic_adapter']==semantic['path']
    assert committed['round']==9 and committed['trained_task_ids']==['earned-task']

    monkeypatch.setattr(returned,'write_json_durable',original)
    resumed=returned.adopt_at_boundary(config,plan,cursor,state_path=state_path,cursor_path=cursor_path)
    assert resumed['training_credit']==0
    adopted_cursor=json.loads(cursor_path.read_text())
    assert adopted_cursor['checkpoint']==checkpoint and adopted_cursor['semantic_adapter']==semantic
    assert adopted_cursor['cycle']==3 and adopted_cursor['stage_index']==2
    assert adopted_cursor['failed_stages']==cursor['failed_stages']
    assert returned.adopt_at_boundary(config,plan,adopted_cursor,state_path=state_path,cursor_path=cursor_path) is None


def test_return_checkpoint_reader_normalizes_original_full_native_parent(tmp_path):
    import torch
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.rl.local_cycle_semantic_return import _checkpoint_state
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy

    rng={'torch_cpu':torch.get_rng_state().clone()}
    optimizer={'param_groups':[{'param_names':['native.test'],'params':[0]}],
        'state':{0:{'step':torch.tensor(4.),'exp_avg':torch.tensor([1.]),
            'exp_avg_sq':torch.tensor([2.])}}}
    state={'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'lfm',
            'base_inventory_sha256':'a'*64},'model_state':{},
        'training':{'optimizer':optimizer,'rng':rng,'source_receipts':[{'sha256':'b'*64}],
            'options':{'native_snapshot_ancestors':['c'*64]},'global_step':4,
            'sft_progress':{'schema':'natlang.native-sft-progress/1','stages':[]}}}
    policy=CheckpointPolicy(tmp_path,every_steps=1)
    policy.save(state)
    participant,training=_checkpoint_state(policy.slot,'lfm')
    assert participant['optimizer']['state'][0]['step'].item()==4
    assert participant['global_step']==4
    assert training['source_rng']['torch_cpu'].equal(rng['torch_cpu'])
    assert training['training_options']['native_snapshot_ancestors']==['c'*64]
    assert training['sft_progress']==state['training']['sft_progress']


def test_role_snapshot_plan_is_only_exact_two_flag_successor(tmp_path):
    from natlang_neuralese.rl.native_role_release import _local_cycle_role_plan_predecessor
    from natlang_neuralese.rl.native_raw_handoff import binding

    old={'schema':'natlang.native-source-host-plan/1','snapshot_preparation':False,
        'consumer_preparation_handoff':False,'role_exchange':{'role':'lfm'},
        'local_cycle_source':{'cycles_plan':str(tmp_path/'cycles.json')}}
    old_path=tmp_path/'old.json';old_path.write_text(json.dumps(old))
    new={**old,'snapshot_preparation':True,'consumer_preparation_handoff':True}
    new_path=tmp_path/'new.json';new_path.write_text(json.dumps(new))
    cycle={'local_cycle_role_plan':binding(old_path)}
    assert _local_cycle_role_plan_predecessor(cycle,old['local_cycle_source'],new_path)==binding(old_path)
    changed={**new,'role_exchange':{'role':'mellum'}}
    changed_path=tmp_path/'changed.json';changed_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='exact pinned plan'):
        _local_cycle_role_plan_predecessor(cycle,old['local_cycle_source'],changed_path)


def test_returned_shared_route_bootstraps_a_bare_cycle_config(tmp_path):
    from natlang_neuralese.rl.local_cycles import bind_state_semantic_route
    from natlang_neuralese.common.hashing import sha256_file_hex

    curriculum=tmp_path/'curriculum.json';curriculum.write_text('{"schema":"fixture"}\n')
    state={'native_participant':'lfm','semantic_refresh_config':{
        'path':str(curriculum.resolve()),'sha256':sha256_file_hex(curriculum)}}
    resolved=bind_state_semantic_route({'native_participant':None,'semantic_refresh_config':None},state)
    assert resolved['native_participant']=='lfm'
    assert resolved['semantic_refresh_config']==str(curriculum.resolve())
    with pytest.raises(ValueError,match='changed its adopted semantic refresh curriculum'):
        bind_state_semantic_route({'native_participant':None,'semantic_refresh_config':'/tmp/other.json'},state)
