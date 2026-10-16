"""Real adapted actor sampling, one-pass producer gradients and durable replay."""
import copy
import json

import pytest
import torch

from test_native_serving_adapters import engine_fixture, nonzero_adapter
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.model.native_adapters import run_native, effective_identity, resolve_adapters, create_adapter, target_parameters
from natlang_neuralese.rl.native_replay import compile_replay, replay_logps, prepare_objective
from natlang_neuralese.rl.graph_replay import save_bundle, restore_bundle
from natlang_neuralese.serve.native_grad import native_graph_gradient
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.serve.store import TensorStore, encode_native_state
from natlang_neuralese.model.native_artifact import encode_native_tensor_artifact


def post_service(engine, path, body):
    import threading
    import urllib.request
    from natlang_neuralese.serve.http import serve
    server=serve(engine,'127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}'+path,
            data=json.dumps(body).encode(),headers={'content-type':'application/json'})
        with urllib.request.urlopen(request,timeout=30) as response:return json.loads(response.read())
    finally:
        server.shutdown();server.server_close();thread.join()


def capture_actor(engine, entries, monkeypatch):
    roles = engine.role_tokens
    selected = iter([roles.produce_open, 5, 6, roles.produce_close, 2])
    # Select deterministic actions, but retain their ACTUAL normalized behavior
    # logp. This exercises normal Engine inputs/caches/captures/termination.
    def token(seq):
        value = next(selected)
        return value, float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[value])
    monkeypatch.setattr(engine, '_native_token', token)
    writer = engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4 t5'}],
        adapters=entries, temperature=.8, seed=21, max_tokens=8, rl_capture=True))
    block_id = writer['native_state']['blocks'][0]['id']
    selected = iter([7, 2])
    reader = engine.generate(GenerationRequest(messages=[{'role':'user','content':[
        {'type':'text','text':'t6 '}, {'type':'native_state','id':block_id}]}],
        adapters=entries, temperature=.8, seed=22, max_tokens=8, rl_capture=True))
    assert reader['native_state']['imports'][0]['mode'] == 'exact-native-state'
    turns = [{'graph_turn':'writer','sampling':writer['x_natlang_rl_capture'],
              'written_nodes':{block_id:'writer-occurrence'}},
             {'graph_turn':'reader','sampling':reader['x_natlang_rl_capture'],
              'read_bindings':{block_id:'writer-occurrence'}}]
    return block_id, turns


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_fp32_actual_adapted_actor_capture_replay_and_consumer_producer_coefficient_gradients(tmp_path,monkeypatch,family):
    torch.manual_seed(1701)
    engine=engine_fixture(tmp_path,family,service_template=True)
    # Exact FP32 diagnostic on both ports. Mellum's production loader correctly
    # requires BF16. Promote this diagnostic copy explicitly and derive a new
    # identity; never mislabel it as the persisted BF16 checkpoint.
    if family=='mellum':
        from natlang_neuralese.common.hashing import canonical_json_sha256_hex
        from natlang_neuralese.model.native_model import native_identity
        engine.shared_model.float()
        engine.checkpoint_sha256=canonical_json_sha256_hex({'diagnostic_parent':engine.checkpoint_sha256,
            'transform':'exact-fp32-promotion/1'})
        engine.native_identity=native_identity(engine.shared_model,engine.model_config,
            checkpoint_sha256=engine.checkpoint_sha256)
        engine.embedding_weight_sha256=engine.native_identity['embedding_weight_sha256']
    # V directly participates in the exported state even in the one-attention-
    # layer fixture. A final-layer Q adapter would affect only consumer queries.
    from dataclasses import replace
    name=next(name for name in target_parameters(engine.shared_model) if name.endswith('v_proj.weight'))
    created=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    adapter=engine.store.put_native_tensor_artifact(replace(created.artifact,
        tensors={name:torch.full_like(created.artifact.tensors[name],.04)}))
    entries=[{'id':adapter.id,'scale':.75}]
    identity=effective_identity(engine,resolve_adapters(engine,entries))
    before={name:value.detach().clone() for name,value in engine.shared_model.named_parameters()}
    block_id,turns=capture_actor(engine,entries,monkeypatch)
    for turn in turns:
        assert turn['sampling']['schema']=='natlang.native-rl-sampling/3'
        assert turn['sampling']['policy']==identity
        assert turn['sampling']['adapter_binding']['base_native_identity']==engine.native_identity
    block=engine.store.get_native_state(block_id).block
    replay=compile_replay(turns,identity=identity,roles=engine.role_tokens,fetch_block=lambda ident:engine.store.get_native_state(ident).block)
    assert not replay.external_blocks and len(replay.plan.invocations)==2
    coefficient=adapter.artifact.tensors[name].clone().requires_grad_(True)
    def execute():
        logps,behavior=replay_logps(engine.shared_model,replay,chunk_tokens=2)
        torch.testing.assert_close(logps.detach(),behavior,rtol=2e-5,atol=3e-6)
        reader=[i for i,item in enumerate(replay.actions) if item['turn']=='reader']
        # Consumer-only objective: gradient includes the replayed writer span.
        grad=torch.autograd.grad(-logps[reader].mean(),coefficient)[0]
        assert grad.norm()>0 and torch.isfinite(grad).all()
        return grad
    gradient=run_native(engine,execute,entries,leaves={adapter.id:{name:coefficient}})
    with pytest.raises(ValueError,match='exact effective actor scope'):
        replay_logps(engine.shared_model,replay)
    # Compare against explicit cached graph execution with differentiable writer
    # state. A consumer-only graph with the archived (detached) producer must
    # omit a real, measurable part of the coefficient gradient.
    from natlang_neuralese.rl.native_replay import action_logps
    def reference():
        result=engine.shared_model(replay.plan,reference=True,retain_states=False)
        logps,_=action_logps(engine.shared_model,result['h_final'][0],replay.actions,chunk_tokens=2)
        reader=[i for i,item in enumerate(replay.actions) if item['turn']=='reader']
        return torch.autograd.grad(-logps[reader].mean(),coefficient)[0]
    expected=run_native(engine,reference,entries,leaves={adapter.id:{name:coefficient}})
    torch.testing.assert_close(gradient,expected,rtol=2e-5,atol=3e-6)
    detached_reader=copy.deepcopy(turns[1]);detached_reader.pop('read_bindings')
    from natlang_neuralese.serve.native_grad import _leaf_block
    detached_block,_=_leaf_block(block)
    detached=compile_replay([detached_reader],identity=identity,roles=engine.role_tokens,
        fetch_block=lambda _:detached_block)
    def detached_gradient():
        logps,_=replay_logps(engine.shared_model,detached,chunk_tokens=2)
        return torch.autograd.grad(-logps.mean(),coefficient)[0]
    detached_grad=run_native(engine,detached_gradient,entries,leaves={adapter.id:{name:coefficient}})
    assert (gradient-detached_grad).norm()>gradient.norm()*1e-5
    assert all(torch.equal(value,before[name]) for name,value in engine.shared_model.named_parameters())
    # Wrong ordered scales and missing explicit adapter schema cannot be admitted.
    wrong=copy.deepcopy(turns);wrong[0]['sampling']['adapter_binding']['adapters'][0]['scale_hex']=(.5).hex()
    with pytest.raises(ValueError,match='effective identity'):
        compile_replay(wrong,identity=identity,roles=engine.role_tokens,fetch_block=lambda _:block)
    wrong=copy.deepcopy(turns);wrong[0]['sampling']['schema']='natlang.native-rl-sampling/2'
    with pytest.raises(ValueError,match='explicit adapter-bound'):
        compile_replay(wrong,identity=identity,roles=engine.role_tokens,fetch_block=lambda _:block)
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_adapted_policy_bundle_restart_gradient_service_and_optimizer(tmp_path,monkeypatch,family):
    torch.manual_seed(1701)
    engine=engine_fixture(tmp_path,family,service_template=True)
    adapter,name=nonzero_adapter(engine)
    peer,_=nonzero_adapter(engine,'tiny')
    entries=[{'id':adapter.id,'scale':.75},{'id':peer.id,'scale':-.25},{'id':adapter.id,'scale':.1}]
    identity=effective_identity(engine,resolve_adapters(engine,entries))
    block_id,turns=capture_actor(engine,entries,monkeypatch)
    rows=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],
           'sampling':turn['sampling'],'replay':{'raw_calls_retained':True}} for turn in turns]
    episodes=[{'task_id':'train-adapter-task','split':'train','sample':sample,'group':'group',
        'reward':{'quality':reward},'policy':{'checkpoint_sha256':identity['checkpoint_sha256']},
        'turns':copy.deepcopy(rows)} for sample,reward in enumerate((1,0))]
    def fetch(ident):
        if ident.startswith('nnt1_'):
            return encode_native_tensor_artifact(engine.store.get_native_tensor_artifact(ident).artifact)
        return encode_native_state(engine.store.get_native_state(ident).block)
    manifest=save_bundle(episodes,tmp_path/'bundle',policy_id=identity['checkpoint_sha256'],fetch=fetch)
    basis=adapter.artifact.source_artifact_id
    assert set(manifest['blocks'])=={adapter.id,basis,block_id,peer.id,peer.artifact.source_artifact_id}
    bundle=tmp_path/'bundle/manifest.json'
    graphs,spec=prepare_objective(episodes,identity=identity,roles=engine.role_tokens,
        fetch_block=lambda ident:engine.store.get_native_state(ident).block,block_bundle=bundle)
    spec_path=tmp_path/'objective.json';spec_path.write_text(json.dumps(spec))
    assert spec['adapter_binding']==turns[0]['sampling']['adapter_binding']
    # Simulate actor store death. Durable archive is sufficient on a new store.
    engine.store.close();engine.store=TensorStore(directory=tmp_path/'restored')
    restore_bundle(engine,bundle,policy_id=identity['checkpoint_sha256'])
    engine.store.close();engine.store=TensorStore(directory=tmp_path/'restored')
    body={'schema':'natlang.native-gradient-request/1','graph':graphs[0],'adapter_gradient_precision':'fp32',
          'adapters':entries,'native_identity':engine.native_identity,
          'objective':{'kind':'native-policy','path':str(spec_path),'sha256':sha256_file_hex_cached(spec_path)},
          'target':{'kind':'adapter','id':adapter.id,'names':[name]}}
    reply=post_service(engine,'/v1/neuralese/grad',body)
    artifact=engine.store.get_native_tensor_artifact(reply['gradient']['id']).artifact
    assert artifact.kind=='adapter-gradient' and artifact.tensors[name].norm()>0
    assert artifact.provenance['effective_native_identity']==identity
    from natlang_neuralese.serve.native_optim import native_optimizer_step
    updated=post_service(engine,'/v1/neuralese/optim',{'schema':'natlang.native-optimizer-request/1',
        'native_identity':engine.native_identity,'gradient_id':reply['gradient']['id'],'optimizer':{'name':'adam','learning_rate':.01}})
    assert updated['adapter']['id']!=adapter.id
    expected_binding=copy.deepcopy(spec['adapter_binding'])
    for entry in expected_binding['adapters']:
        if entry['id']==adapter.id:entry['id']=updated['adapter']['id']
    assert updated['adapter_binding']==expected_binding
    assert updated['effective_native_identity']['checkpoint_sha256']!=identity['checkpoint_sha256']
    wrong=copy.deepcopy(body);wrong['adapters']=[{'id':adapter.id,'scale':.5}]
    with pytest.raises(Exception,match='another exact actor'):
        native_graph_gradient(engine,wrong,'test-owner')
    entry=manifest['blocks'][basis];path=bundle.parent/entry['path'];path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='bytes changed'):
        restore_bundle(engine,bundle,policy_id=identity['checkpoint_sha256'])
    engine.store.close()


def test_bf16_mellum_coefficient_adjoint_gap_is_accounted_by_weight_cotangent(tmp_path,monkeypatch):
    """Exact FP64 chain-rule oracle; this is NOT a BF16 parity certificate."""
    from natlang_neuralese.rl.native_replay import action_logps
    torch.manual_seed(1701)
    engine=engine_fixture(tmp_path,'mellum',service_template=True)
    adapter,name=nonzero_adapter(engine)
    entries=[{'id':adapter.id,'scale':.75}]
    identity=effective_identity(engine,resolve_adapters(engine,entries))
    _,turns=capture_actor(engine,entries,monkeypatch)
    replay=compile_replay(turns,identity=identity,roles=engine.role_tokens,
        fetch_block=lambda ident:engine.store.get_native_state(ident).block)
    coefficient=adapter.artifact.tensors[name].clone().requires_grad_(True)
    reader=[i for i,item in enumerate(replay.actions) if item['turn']=='reader']
    def execute(reference):
        cotangents=[]
        handle=target_parameters(engine.shared_model)[name].register_hook(lambda grad:cotangents.append(grad.detach().clone()))
        try:
            result=engine.shared_model(replay.plan,reference=reference,retain_states=False)
            logps,_=action_logps(engine.shared_model,result['h_final'][0],replay.actions,chunk_tokens=2)
            grad=torch.autograd.grad(-logps[reader].mean(),coefficient)[0]
        finally:
            handle.remove()
        assert len(cotangents)==1 and cotangents[0].dtype==torch.bfloat16
        return grad.detach(),cotangents[0]
    packed=run_native(engine,lambda:execute(False),entries,leaves={adapter.id:{name:coefficient}})
    cached=run_native(engine,lambda:execute(True),entries,leaves={adapter.id:{name:coefficient}})
    basis=engine.store.get_native_tensor_artifact(adapter.artifact.source_artifact_id).artifact.tensors
    left,right=basis['left.'+name].double(),basis['right.'+name].double()
    def oracle(cotangent):return ((left.T@cotangent.double()@right)*.75).reshape(-1)
    # Every observed coefficient discrepancy is explained by the actual BF16
    # dL/dW supplied by the backbone. No adapter identity or replay-coordinate
    # discrepancy is hidden under an absolute tolerance on tiny gradients.
    for grad,cotangent in (packed,cached):
        expected=oracle(cotangent)
        assert (grad.double()-expected).norm()/expected.norm()<2e-6
    predicted=oracle(packed[1].double()-cached[1].double())
    observed=(packed[0]-cached[0]).double()
    assert (observed-predicted).norm()/max(observed.norm(),torch.tensor(1e-12))<2e-4
    report={'schema':'natlang.native-adapter-adjoint-localization/1','qualification':False,
        'port':'mellum','dtype':'BF16','device':str(engine.device),
        'gradient_relative_l2':float(observed.norm()/cached[0].norm()),
        'weight_cotangent_relative_l2':float((packed[1].float()-cached[1].float()).norm()/cached[1].float().norm()),
        'oracle_residual_relative_l2':float((observed-predicted).norm()/max(observed.norm(),torch.tensor(1e-12)))}
    print(json.dumps(report,sort_keys=True))
    engine.store.close()


def test_capture_rejects_adapter_mutation_between_actual_scheduler_rounds(tmp_path):
    from concurrent.futures import Future
    from natlang_neuralese.serve.engine import Sequence
    from natlang_neuralese.serve.chat import RequestError
    torch.manual_seed(1701)
    engine=engine_fixture(tmp_path,'lfm',service_template=True)
    adapter,_=nonzero_adapter(engine)
    entries=[{'id':adapter.id,'scale':.75}]
    request=GenerationRequest(messages=[{'role':'user','content':'t4 t5'}],
        adapters=entries,temperature=.8,seed=21,rl_capture=True)
    seq=Sequence(request,Future(),prompt_plan=engine._prompt_plan(request.messages,request.tools))
    with torch.inference_mode():engine._round([seq])
    assert seq.native_rl_policy is not None and not seq.future.done()
    entries[0]['scale']=.5
    with torch.inference_mode():engine._round([seq])
    with pytest.raises(RequestError,match='neuralese-native-actor-changed'):
        seq.future.result()
    engine.store.close()
