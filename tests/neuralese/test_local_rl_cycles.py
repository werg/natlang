import json
from pathlib import Path
import pytest
from natlang_neuralese.rl import local_cycles as lc


def test_round_publication_before_cursor_commit_is_not_repeated():
    assert lc.round_target(4,5)
    assert not lc.round_target(5,5)
    with pytest.raises(ValueError,match='unknown controller successor'):
        lc.round_target(6,5)


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
