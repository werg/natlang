"""Real graph arithmetic; fixture rewards are not environment-success claims."""
import copy,json,os,random
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.eval.native_memory import policy_capacity_request,qualify_policy_objective
from natlang_neuralese.rl.native_replay import NativePolicyObjective,prepare_objective
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update,pending_exposure_batches
from natlang_neuralese.common.jsonio import write_json_durable
from test_native_optimizer_checkpointing import assert_exact
from natlang_neuralese.train.loop import capture_training_rng_state


def actual(tmp_path,family,*,device='cpu',fp8=False,recipe='current'):
    from test_native_serving_adapters import engine_fixture
    from test_native_rl_replay import actor_rollout
    from test_rl_episode_graph import attach_execution_graph
    from unittest.mock import patch
    with patch.dict(os.environ,{'NATLANG_NATIVE_ADAPTER_TEST_DEVICE':device}):engine=engine_fixture(tmp_path,family)
    model,roles,identity,_,block,turns=actor_rollout(family,loaded=(engine.shared_model,engine.role_tokens,engine.native_identity))
    episodes=[]
    for sample,quality in [(0,1),(1,0)]:
        episodes.append({'task_id':'capacity-train','sample':sample,'split':'train','group':'real-prompt',
            'reward':{'quality':quality},'policy':{'checkpoint_sha256':identity['checkpoint_sha256']},
            'turns':[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],
                'sampling':turn['sampling'],'replay':{'raw_calls_retained':True}}for turn in turns]})
    attach_execution_graph(episodes)
    graphs,spec=prepare_objective(episodes,identity=identity,roles=roles,fetch_block=lambda key:block,mode='grpo')
    source=tmp_path/'graphs.jsonl';source.write_text(''.join(json.dumps(row)+'\n'for row in graphs))
    path=tmp_path/'objective.json';write_json_durable(path,spec)
    objective=NativePolicyObjective(path);train=NativeGraphSource(source,identity['tokenizer_fingerprint'],required_split='train',role_ids=roles.ids)
    optimizer,named=make_native_optimizer(model,lr=3e-6,in_backward=True)
    backend=None
    if fp8:
        from natlang_neuralese.train.fp8_backend import make_backend
        backend=make_backend('triton-experts',model.port.hf,device=device,recipe=recipe)
        model.port.activation_checkpoint=backend.checkpoint
    config=engine.model_config
    args=SimpleNamespace(out=tmp_path/'learner',policy_capacity_request=None,objective_spec=path,
        batch_graphs=2,batch_tokens=32768,chunk_tokens=3,checkpoint_layers=True,
        checkpoint_input_offload=False,checkpoint_input_offload_directory=None,
        device=device,fp8_training='triton-experts'if fp8 else'off',fp8_recipe=recipe,fp8_amax_history_len=16)
    args.out.mkdir()
    _,batches=pending_exposure_batches(train.entries,{},epochs=1,batch_graphs=2,batch_tokens=32768)
    return engine,model,optimizer,objective,train,args,config,backend,batches


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('cotangent_backend',['plain','cpu_file'])
def test_policy_max_backward_restores_optimizer_rng_all_parameters_and_reuses_proof(tmp_path,family,cotangent_backend):
    values=actual(tmp_path,family,device=os.environ.get('NATLANG_POLICY_CAPACITY_DEVICE','cpu'))
    engine,model,opt,objective,train,args,config,backend,batches=values
    args.cotangent_accumulator_backend=cotangent_backend
    try:
        # Seed a real optimizer history before the read-only boundary.
        native_training_update(model,opt,list(train.iter_indices([0])),checkpoint_layers=True,chunk_tokens=3,objective=objective)
        opt.zero_grad(set_to_none=True)
        parameters={name:p.detach().clone()for name,p in opt._native_named_parameters}
        state=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone();prng=random.getstate()
        all_rng=capture_training_rng_state(args.device)
        ref=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
            checkpoint_sha256='a'*64,pending_batches=batches)
        proof=json.loads(Path(ref['path']).read_text())
        assert proof['passed']and proof['optimizer_updates']==proof['earned_exposure']==0
        assert proof['capacity']['production_packing'] is True
        assert proof['capacity']['cotangent_mode']==('plain' if getattr(args,'cotangent_accumulator_backend',None)=='plain' else 'vjp')
        assert any(len(row['example_ids'])==2 for row in proof['capacity']['cases'])
        assert any(any(name.startswith('backbone.')for name in row['gradients'])for row in proof['capacity']['cases'])
        assert_exact(state,opt.state_dict())
        for name,p in opt._native_named_parameters:assert torch.equal(parameters[name],p)
        assert torch.equal(rng,torch.get_rng_state())and random.getstate()==prng
        assert_exact(all_rng,capture_training_rng_state(args.device))
        assert all(p.grad is None for _,p in opt._native_named_parameters)
        again=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
            checkpoint_sha256='b'*64,pending_batches=batches,prior_proof=ref)
        assert again['authenticated_resume_checkpoint_sha256']=='b'*64
        assert json.loads(Path(ref['path']).read_text())['parent_checkpoint_sha256']=='a'*64
        from natlang_neuralese.train.native_packed_capacity import policy_capacity_baseline
        from natlang_neuralese.train.native_resident import binding
        carried=policy_capacity_baseline(again,checkpoint_sha256='b'*64,objective_binding=objective.binding(),
            source_binding=binding(train.path),cotangent_mode='plain' if cotangent_backend=='plain' else 'vjp',
            retained_gradients=getattr(args,'accumulate_batches',1)>1)
        assert carried['proof_parent_checkpoint_sha256']=='a'*64 and carried['continuation_checkpoint_sha256']=='b'*64
        saved_bytes=Path(ref['path']).read_bytes();Path(ref['path']).unlink()
        portable=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
            checkpoint_sha256='b'*64,pending_batches=batches,prior_proof=ref)
        assert portable['authenticated_resume_checkpoint_sha256']=='b'*64
        assert policy_capacity_baseline(portable,checkpoint_sha256='b'*64,objective_binding=objective.binding(),
            source_binding=binding(train.path),cotangent_mode='plain' if cotangent_backend=='plain' else 'vjp',
            retained_gradients=getattr(args,'accumulate_batches',1)>1)['proof_parent_checkpoint_sha256']=='a'*64
        bad=copy.deepcopy(ref);bad['proof']['passed']=False
        with pytest.raises(ValueError,match='payload changed'):
            qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
                checkpoint_sha256='b'*64,pending_batches=batches,prior_proof=bad)
        Path(ref['path']).write_bytes(saved_bytes)
        with pytest.raises(ValueError,match='authenticated saved continuation'):
            qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
                checkpoint_sha256='c'*64,pending_batches=batches)
        # The same loaded model still makes a genuine policy optimizer update.
        native_training_update(model,opt,list(train.iter_indices([0])),checkpoint_layers=True,chunk_tokens=3,objective=objective)
        assert any(not torch.equal(parameters[name],p)for name,p in opt._native_named_parameters)
    finally:engine.stop();engine.store.close()


def test_policy_request_tamper_and_unexecuted_fp8_fail_before_update(tmp_path,monkeypatch):
    engine,model,opt,objective,train,args,config,backend,batches=actual(tmp_path,'lfm')
    try:
        request=policy_capacity_request(train,objective);assert request['max_example_indices']
        request['entries'][0]['kv_slots']+=1
        write_json_durable(args.out/'policy-capacity-request.json',request)
        with pytest.raises(ValueError,match='actual admitted graph/objective'):
            qualify_policy_objective(model,opt,train,objective,args,config,backend=None,
                checkpoint_sha256='a'*64,pending_batches=batches)
        (args.out/'policy-capacity-request.json').unlink()
        args.fp8_training='triton-experts'
        with pytest.raises(ValueError,match='actual configured FP8'):
            qualify_policy_objective(model,opt,train,objective,args,config,backend=None,
                checkpoint_sha256='a'*64,pending_batches=batches)
    finally:engine.stop();engine.store.close()


@pytest.mark.parametrize('family',['lfm'])
def test_cotangent_storage_transition_uses_actual_policy_capacity_before_update(tmp_path,family):
    from natlang_neuralese.train.native_state_sft import _cotangent_transition_capacity_request
    from natlang_neuralese.train.native_resident import binding
    engine,model,opt,objective,train,args,config,backend,batches=actual(tmp_path,family)
    try:
        args.memory_qualification=None
        with pytest.raises(ValueError,match='fresh exact-parent capacity'):
            _cotangent_transition_capacity_request(args)
        path=args.out/'rl-policy-capacity-request.json'
        write_json_durable(path,policy_capacity_request(train,objective))
        args.policy_capacity_request=json.dumps(binding(path))
        assert _cotangent_transition_capacity_request(args)==args.policy_capacity_request
        before={name:p.detach().clone()for name,p in opt._native_named_parameters}
        history=copy.deepcopy(opt.state_dict());rng=capture_training_rng_state(args.device)
        proof=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
            checkpoint_sha256='a'*64,pending_batches=batches)
        assert proof['proof']['passed'] and proof['proof']['optimizer_updates']==0
        assert_exact(history,opt.state_dict());assert_exact(rng,capture_training_rng_state(args.device))
        assert all(torch.equal(before[name],p)for name,p in opt._native_named_parameters)
        native_training_update(model,opt,list(train.iter_indices([0])),checkpoint_layers=True,
            chunk_tokens=args.chunk_tokens,objective=objective)
        assert any(not torch.equal(before[name],p)for name,p in opt._native_named_parameters)
        value=json.loads(path.read_text());value['objective']['sha256']='f'*64
        write_json_durable(path,value);args.policy_capacity_request=json.dumps(binding(path))
        with pytest.raises(ValueError,match='actual objective'):
            _cotangent_transition_capacity_request(args)
        write_json_durable(path,{'schema':'natlang.native-memory-qualification-request/1'})
        args.policy_capacity_request=json.dumps(binding(path))
        with pytest.raises(ValueError,match='actual objective'):
            _cotangent_transition_capacity_request(args)
    finally:engine.stop();engine.store.close()


def test_same_actor_complete_episode_logprob_gate_fails_before_optimizer_update(tmp_path,monkeypatch):
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    engine,model,opt,objective,train,args,config,backend,batches=actual(tmp_path,'lfm')
    try:
        objective.bind_initial_checkpoint(objective.identity['checkpoint_sha256'])
        example=next(train.iter_indices([0]));receipt=tmp_path/'policy-parity.json'
        monkeypatch.setenv('NATLANG_POLICY_LOGPROB_RECEIPT',str(receipt))
        before={name:p.detach().clone() for name,p in opt._native_named_parameters}
        packed_graph_loss(model,[example],checkpoint_layers=True,chunk_tokens=3,objective=objective)
        report=json.loads(receipt.read_text())
        assert report['passed'] and report['all_episode_actions']
        assert report['action_count']==len(objective.rows[example.id]['actions'])
        assert all(torch.equal(before[name],p) for name,p in opt._native_named_parameters)
        receipt.unlink();objective._behavior_logprob_probe_done=False
        objective.rows[example.id]['actions'][-1]['behavior_logp']+=0.1
        with pytest.raises(ValueError,match='parity failed before SGD'):
            packed_graph_loss(model,[example],checkpoint_layers=True,chunk_tokens=3,objective=objective)
        assert not json.loads(receipt.read_text())['passed']
        assert all(torch.equal(before[name],p) for name,p in opt._native_named_parameters)
        # A genuinely updated rolling checkpoint retains the original behavior
        # probabilities for importance correction; it must not be relabeled as
        # the original actor or rejected for legitimate policy change.
        receipt.unlink();objective._behavior_logprob_probe_done=False
        objective.bind_initial_checkpoint('f'*64)
        packed_graph_loss(model,[example],checkpoint_layers=True,chunk_tokens=3,objective=objective)
        assert not receipt.exists()
    finally:engine.stop();engine.store.close()


def test_policy_capacity_prioritizes_real_same_actor_episode():
    from natlang_neuralese.eval.native_memory import policy_parity_first_group
    obj=NativePolicyObjective.__new__(NativePolicyObjective)
    obj.identity={'checkpoint_sha256':'a'*64};obj.evaluation_identity=obj.identity.copy()
    obj.adapter_binding=obj.evaluation_adapter_binding=None;obj.behavior_policy=None
    obj.bind_initial_checkpoint('a'*64)
    obj.rows={'stale':{'actor_checkpoint_sha256':'b'*64,'actions':[{}]},
              'same':{'actions':[{}]},'empty':{'actions':[]}}
    expected={'entries':[{'id':'stale','index':0},{'id':'empty','index':1},{'id':'same','index':2}]}
    assert policy_parity_first_group(expected,obj)==(2,)
    obj._behavior_logprob_probe_done=True
    assert policy_parity_first_group(expected,obj) is None


def test_engine_sampled_episode_is_guarded_not_exact_gated(tmp_path,monkeypatch):
    """SGLang/llama.cpp exports of the same weights: per-episode mismatch guard instead of the exact native gate."""
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    from natlang_neuralese.rl.native_replay import DEFAULT_ENGINE_GUARD, engine_mismatch
    engine,model,opt,objective,train,args,config,backend,batches=actual(tmp_path,'lfm')
    try:
        objective.bind_initial_checkpoint(objective.identity['checkpoint_sha256'])
        example=next(train.iter_indices([0]))
        log=tmp_path/'engine-guard.jsonl';monkeypatch.setenv('NATLANG_ENGINE_GUARD_LOG',str(log))
        for row in objective.rows.values():  # every episode was sampled on SGLang
            for action in row['actions']:
                action['actor_export']='exact:sglang-hf-bfloat16';action['behavior_logp']+=0.01
        actions=objective.rows[example.id]['actions']
        objective.engine_guard=dict(DEFAULT_ENGINE_GUARD)
        # A small engine gap trains (the exact 1e-4 gate would have failed on these tokens).
        packed_graph_loss(model,[example],checkpoint_layers=True,chunk_tokens=3,objective=objective)
        record=json.loads(log.read_text().splitlines()[-1])
        assert record['accepted'] and record['tokens']==len(actions) and abs(record['mean_abs']-0.01)<1e-3
        assert not getattr(objective,'_behavior_logprob_probe_done',False)
        # A large gap rejects the episode: no loss terms, logged.
        actions[-1]['behavior_logp']-=3.
        packed_graph_loss(model,[example],checkpoint_layers=True,chunk_tokens=3,objective=objective)
        record=json.loads(log.read_text().splitlines()[-1])
        assert not record['accepted'] and record['max_abs']>2.9
    finally:engine.stop();engine.store.close()
    assert engine_mismatch([0.,-1.],[0.,-1.],DEFAULT_ENGINE_GUARD)['mean_k3']==0.


def test_prepare_objective_binds_engine_guard_for_exported_actor_captures(tmp_path):
    from natlang_neuralese.rl.native_replay import DEFAULT_ENGINE_GUARD
    from test_native_serving_adapters import engine_fixture
    from test_native_rl_replay import actor_rollout
    from test_rl_episode_graph import attach_execution_graph
    engine=engine_fixture(tmp_path,'lfm')
    try:
        model,roles,identity,_,block,turns=actor_rollout('lfm',loaded=(engine.shared_model,engine.role_tokens,engine.native_identity))
        export={'kind':'exact','format':'sglang-hf-bfloat16','sha256':'e'*64}
        episodes=[{'task_id':'t','sample':sample,'split':'train','group':'g','reward':{'quality':quality},
                   'policy':{'checkpoint_sha256':identity['checkpoint_sha256']},
                   'turns':[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],
                             'sampling':{**turn['sampling'],'actor_export':export},'replay':{'raw_calls_retained':True}}
                            for turn in turns]} for sample,quality in [(0,1),(1,0)]]
        attach_execution_graph(episodes)
        _,spec=prepare_objective(episodes,identity=identity,roles=roles,fetch_block=lambda key:block,mode='grpo')
        assert spec['actor_engine_guard']==DEFAULT_ENGINE_GUARD
        assert all(a['actor_export']=='exact:sglang-hf-bfloat16' for row in spec['rows'].values() for a in row['actions'])
    finally:engine.stop();engine.store.close()


def test_tis_cap_mode_is_configurable_without_changing_default_bytes():
    from natlang_neuralese.rl.policy_math import importance_weight, policy_loss_declaration
    r=torch.tensor([.5,2.5,9.])
    assert importance_weight(r,'tis').tolist()==[.5,2.,2.]
    assert importance_weight(r,'tis-c3').tolist()==[.5,2.5,3.]
    assert policy_loss_declaration({'schema':'score-centered/1'})['importance']=='tis'
    assert policy_loss_declaration({'schema':'score-centered/1','importance':'tis-c3'})['importance']=='tis-c3'
    with pytest.raises(ValueError):policy_loss_declaration({'schema':'score-centered/1','importance':'tis-c0.5'})


def test_lfm_continuations_adopt_sglang_actor_through_recorded_transition(tmp_path):
    from natlang_neuralese.rl import production
    path=tmp_path/'state.json';state={'round':3}
    config={'model':'lfm','actor_command':['python','serve.py','--port','8200','--engine','native']}
    adopted=production.adopt_actor_engine_defaults(config,state,path)
    assert adopted['actor_command'][-4:]==['--engine','sglang','--sglang-projection',production.DEFAULT_SGLANG_PROJECTION]
    record=json.loads(path.read_text())['actor_engine_transitions']
    assert len(record)==1 and record[0]['from']==config['actor_command'] and record[0]['round']==3
    # Idempotent; sealed selections, active updates, opt-outs and coefficient actors keep their actor.
    assert production.adopt_actor_engine_defaults(config,state,path)==adopted
    assert len(json.loads(path.read_text())['actor_engine_transitions'])==1
    for kept,kwargs in (({},{'sealed_selection':True}),({'actor_engine_keep_native':True},{}),
                        ({'coefficient_policy':'p.json'},{})):
        assert production.adopt_actor_engine_defaults({**config,**kept},state,path,**kwargs)['actor_command']==config['actor_command']
    assert production.adopt_actor_engine_defaults(config,{'active_update_round':'r'},path)==config
    assert production.adopt_actor_engine_defaults(adopted,state,path)==adopted


def test_engine_scored_decisions_are_guarded_not_exact_gated(monkeypatch):
    """SGLang-scored decide options join the per-episode engine guard; only native decisions take the exact probe."""
    from types import SimpleNamespace
    from natlang_neuralese.rl import decision_policy
    from natlang_neuralese.rl.native_replay import DEFAULT_ENGINE_GUARD, NativePolicyObjective, native_sampled_actions
    native={'kind':'decision','behavior_logps':[-.1,-2.4],'options':[[{'query':0}],[{'query':1}]]}
    engine={**native,'actor_export':'exact:sglang-hf-bfloat16'}
    assert native_sampled_actions([native,engine])==[native]
    monkeypatch.setattr(decision_policy,'decision_log_policy',
                        lambda model,hidden,action,queries=None:torch.tensor([-.1,-2.4])+torch.tensor([.02,-.02]))
    owner=SimpleNamespace(engine_guard=dict(DEFAULT_ENGINE_GUARD),engine_guard_records=[])
    record=NativePolicyObjective._engine_guard(owner,None,torch.zeros(2,1),'g',[],True,decisions=[engine],
                                               physical_rows=[0,1])
    assert record['accepted'] and record['decisions']==1 and record['tokens']==2 and abs(record['max_abs']-.02)<1e-6
    far={**engine,'behavior_logps':[-2.4,-.1]}
    assert not NativePolicyObjective._engine_guard(owner,None,torch.zeros(2,1),'g',[],True,decisions=[far])['accepted']
