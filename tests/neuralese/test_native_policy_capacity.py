"""Real graph arithmetic; fixture rewards are not environment-success claims."""
import copy,json,os,random
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.eval.native_memory import policy_capacity_request,qualify_policy_objective,capacity_identity
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


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_policy_max_backward_restores_optimizer_rng_all_parameters_and_reuses_proof(tmp_path,family):
    values=actual(tmp_path,family,device=os.environ.get('NATLANG_POLICY_CAPACITY_DEVICE','cpu'))
    engine,model,opt,objective,train,args,config,backend,batches=values
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
        assert any(len(row['example_ids'])==2 for row in proof['capacity']['cases'])
        assert any(any(name.startswith('backbone.')for name in row['gradients'])for row in proof['capacity']['cases'])
        assert_exact(state,opt.state_dict())
        for name,p in opt._native_named_parameters:assert torch.equal(parameters[name],p)
        assert torch.equal(rng,torch.get_rng_state())and random.getstate()==prng
        assert_exact(all_rng,capture_training_rng_state(args.device))
        assert all(p.grad is None for _,p in opt._native_named_parameters)
        again=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
            checkpoint_sha256='b'*64,pending_batches=batches,prior_proof=ref)
        assert again==ref and json.loads(Path(ref['path']).read_text())['parent_checkpoint_sha256']=='a'*64
        saved_bytes=Path(ref['path']).read_bytes();Path(ref['path']).unlink()
        portable=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
            checkpoint_sha256='b'*64,pending_batches=batches,prior_proof=ref)
        assert portable==ref
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


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual executed expert FP8')
@pytest.mark.parametrize('recipe',['current','current-bf16-dx'])
def test_mellum_actual_policy_capacity_executes_fp8_without_optimizer_update(tmp_path,recipe):
    engine,model,opt,objective,train,args,config,backend,batches=actual(tmp_path,'mellum',device='cuda',fp8=True,recipe=recipe)
    try:
        before=copy.deepcopy(opt.state_dict())
        parameters={name:p.detach().clone()for name,p in opt._native_named_parameters}
        rng=capture_training_rng_state(args.device)
        ref=qualify_policy_objective(model,opt,train,objective,args,config,backend=backend,
            checkpoint_sha256='a'*64,pending_batches=batches)
        proof=json.loads(Path(ref['path']).read_text());counts=proof['capacity']['executed_projections']
        assert proof['passed']and counts['calls']>0 and counts['rows']>0
        assert_exact(before,opt.state_dict())
        assert_exact(rng,capture_training_rng_state(args.device))
        for name,p in opt._native_named_parameters:assert torch.equal(parameters[name],p)
        native_training_update(model,opt,list(train.iter_indices([0])),backend=backend,
            checkpoint_layers=True,chunk_tokens=3,objective=objective)
        assert any(not torch.equal(parameters[name],p)for name,p in opt._native_named_parameters)
    finally:engine.stop();engine.store.close()
