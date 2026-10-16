"""Protocol routing never sends a semantic batch to the raw-only learner."""
import copy
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.rl.native_semantic_route import has_semantic_inputs,bind_policy_source,policy_curriculum,accept_policy_deployment


def fixture(tmp_path):
    parent=tmp_path/'parent.pt';parent.write_bytes(b'bound routing parent; this is a routing fixture, not numerical qualification')
    digest=sha256_file_hex(parent)
    identity={'checkpoint_sha256':digest,'architecture':'lfm2','tokenizer_fingerprint':'2'*64,'embedding_weight_sha256':'3'*64}
    adapter=tmp_path/'adapter.json';adapter.write_text(json.dumps({'schema':'natlang.native-semantic-adapter/1',
        'checkpoint':{'path':str(parent),'sha256':digest},'models':{'student':{'native_identity':identity}},
        'tokenizers':{'student':{'path':'tokenizer'}}}))
    roles=tmp_path/'tokenizer';roles.mkdir()
    rows=[{'task_id':'train','sample':0,'split':'train','policy':{'checkpoint_sha256':digest},'reward':{'quality':1},
        'turns':[{'invocation_id':'write','sampling':{'writes':[{'state_id':'wire'}]}},
            {'invocation_id':'read','sampling':{'prompt':[{'kind':'native_state','id':'observed',
                'source_state_id':'wire','semantic_adaptation':{},'source_occurrence':'write-node'}]}}],
        'execution_graph':[{'node':'write#turn1','kind':'model_turn','call_id':'write'},
            {'node':'read#turn1','kind':'model_turn','call_id':'read'},
            {'node':'write-node','kind':'block_write','turn':'write#turn1','block':'wire'},
            {'node':'read-node','kind':'block_read','turn':'read#turn1','block':'wire','inputs':[{'node':'write-node','block':'wire'}]}]}]
    rollouts=tmp_path/'rollouts.jsonl';rollouts.write_text(json.dumps(rows[0])+'\n')
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'policy_id':digest,'blocks':{}}))
    config={'native_state':True,'native_participant':'student','semantic_adapter':str(adapter)}
    values={'rollouts':str(rollouts),'block_bundle':str(bundle),'semantic_adapter':str(adapter),'policy_id':digest}
    return config,values,rows,parent,identity


def test_pending_source_retains_whole_batch_and_curriculum_rebinds_only_actual_sources(tmp_path):
    config,values,rows,parent,_=fixture(tmp_path)
    ordinary=copy.deepcopy(rows[0]);ordinary.update(task_id='ordinary',sample=1)
    ordinary['turns']=[{'invocation_id':'ordinary','sampling':{}}]
    ordinary['execution_graph']=[{'node':'ordinary#turn1','kind':'model_turn','call_id':'ordinary'}]
    rows.append(ordinary);Path(values['rollouts']).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert has_semantic_inputs(rows)
    source=bind_policy_source(config,values,rows,'rft')
    assert source['rollouts']['sha256']==sha256_file_hex(values['rollouts'])
    assert len(Path(source['rollouts']['path']).read_text().splitlines())==2
    curriculum=tmp_path/'curriculum.json';curriculum.write_text(json.dumps({'phases':[{'id':'joint'}],
        'policy_schedule':{'id':'policy','epochs':2,'every_semantic_updates':1,'updates_per_refresh':16}}))
    bound=policy_curriculum(curriculum,[source],tmp_path/'actual.json')
    actual=json.loads(bound.read_text())['policy_training']
    assert actual['mode']=='mixed' and actual['sources']==[source]
    assert json.loads(curriculum.read_text()).get('policy_training') is None


def test_previous_deployment_does_not_clear_new_pending_batch_and_unaccepted_new_parent_rejects(tmp_path):
    config,values,rows,parent,_=fixture(tmp_path);source=bind_policy_source(config,values,rows,'rft')
    state={'checkpoint':str(parent),'pending_semantic_source':source,'round':0}
    assert not accept_policy_deployment(config,state,{'checkpoint':{'path':str(parent)}})
    assert state['pending_semantic_source']==source
    with pytest.raises(ValueError,match='omitted a pending whole semantic batch'):
        accept_policy_deployment(config,state,{'checkpoint':{'path':'different-parent'}})


def test_actual_admission_progress_earns_only_visited_episodes_and_is_idempotent(tmp_path):
    config,values,rows,parent,_=fixture(tmp_path);source=bind_policy_source(config,values,rows,'rft')
    progress={'policy_training_phase':'policy','policy_source_history':[source],
        'policy_obligations':{'complete':False,'remaining_episode_visits':1}}
    completion=tmp_path/'completion.json';completion.write_text(json.dumps({'policy_training_progress':progress,
        'exposure':{'policy':{canonical_json_sha256_hex(rows[0]):{'episode':1}}}}))
    deployment={'policy_training_progress':progress,'checkpoint':{'path':'new-parent','sha256':'4'*64},
        'shared_stage_completion':{'path':str(completion),'sha256':sha256_file_hex(completion)}}
    state={'checkpoint':str(parent),'round':0,'trained_rounds':0,'pending_semantic_source':source,'pending_semantic_next_offset':3}
    assert accept_policy_deployment(config,state,deployment)
    assert state['trained_task_ids']==['train'] and state['trained_rounds']==1 and state['round']==1
    assert state['offset']==3 and not accept_policy_deployment(config,state,deployment)


def test_production_routes_all_success_semantic_batch_and_restart_never_recollects_or_runs_raw_learner(tmp_path,monkeypatch):
    from natlang_neuralese.rl import production
    config,values,rows,parent,identity=fixture(tmp_path)
    second=copy.deepcopy(rows[0]);second['sample']=1;rows.append(second)
    adapter_path=Path(values['semantic_adapter']);adapter=json.loads(adapter_path.read_text())
    adapter['models']['student']['checkpoint']={**adapter['checkpoint'],'participant':'student'}
    adapter_path.write_text(json.dumps(adapter))
    cases=tmp_path/'cases.jsonl';cases.write_text(json.dumps({'id':'train','split':'train'})+'\n')
    config.update(schema='natlang.rl-production-config/1',id='routing',out=str(tmp_path/'controller'),
        checkpoint=str(parent),cases=str(cases),actor_command=['fixture-native-actor'],
        endpoints=['fixture-native-endpoint'],rows_per_round=1,group=2,workers=1,
        learner_command=['raw-only-must-not-run'],seed_command=['seed-must-not-run'])
    path=tmp_path/'production.json';path.write_text(json.dumps(config))
    calls=[]
    def collect(*args,**kwargs):
        calls.append(kwargs)
        assert kwargs['expected_native']['native_identity']==identity
        out=Path(kwargs['out']);out.mkdir(parents=True,exist_ok=True)
        (out/'rollouts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return {'scored':2,'successes':2,'groups_with_spread':False,'groups':[],
                'block_bundle':values['block_bundle']}
    monkeypatch.setattr(production,'collect_with_actor',collect)
    monkeypatch.setattr(production,'run_command',lambda *args,**kwargs:pytest.fail('raw learner was invoked'))
    first=production.run(path,rounds=1,defer_native_refresh=True)
    assert first['phase']=='awaiting_semantic_refresh' and first['round']==0
    assert first['pending_semantic_source']['mode']=='rft' and len(calls)==1
    second=production.run(path,rounds=1,defer_native_refresh=True)
    assert second['pending_semantic_source']==first['pending_semantic_source'] and len(calls)==1
    assert 'pending_native_checkpoint' not in second
