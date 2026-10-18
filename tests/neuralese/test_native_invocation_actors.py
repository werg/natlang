"""Exact one-pass distinct-actor native graphs, not per-node production replay."""
from dataclasses import replace
import pytest
import torch
from test_native_serving_adapters import engine_fixture
from natlang_neuralese.model.native_adapters import create_adapter,target_parameters,run_native
from natlang_neuralese.model.native_actor_graph import run_invocation_actors
from natlang_neuralese.model.shared_state import Invocation,ImportSpan,compile_graph


def adapters(engine):
    names=list(target_parameters(engine.shared_model))
    values=[]
    for kind in ('xs','tiny'):
        adapter=create_adapter(engine,{'kind':kind,'rank':2,'u':3,'parameter_names':names})
        tensors={name:torch.full_like(value,.04 if kind=='xs' else -.03) for name,value in adapter.artifact.tensors.items()}
        values.append(engine.store.put_native_tensor_artifact(replace(adapter.artifact,tensors=tensors)))
    return values


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('checkpoint',[False,True])
def test_layer_major_distinct_actor_graph_matches_cached_and_source_adjoints(tmp_path,monkeypatch,family,checkpoint,backend_recipe=None):
    torch.manual_seed(1711)
    engine=engine_fixture(tmp_path,family)
    model=engine.shared_model
    backend=None
    if backend_recipe:
        from natlang_neuralese.train.fp8_backend import make_backend
        backend=make_backend('triton-experts',model.port.hf,device=engine.device,recipe=backend_recipe)
    source,target=adapters(engine)
    with torch.no_grad(): model.adapter.up.weight.normal_(std=.01)
    roles=engine.role_tokens
    plan=compile_graph([Invocation('producer',(4,roles.produce_open,5,6,roles.produce_close,7),producing=((2,4),)),
        Invocation('reader',(8,roles.consume_open,roles.consume_close,9,10),imports=(ImportSpan(2,'producer',2,4),)),
        Invocation('peer',(11,12,13))])
    bindings={'producer':[source.id],'reader':[target.id],'peer':[source.id]}
    original=dict(model.named_parameters());original_buffers=dict(model.named_buffers())
    baseline=model(plan)['h_final'].detach()
    leaves={actor.id:{name:value.clone().requires_grad_(True) for name,value in actor.artifact.tensors.items()}
            for actor in (source,target)}
    selected=[(actor.id,name,value) for actor in (source,target) for name,value in leaves[actor.id].items()]
    model.port.checkpoint_layers=checkpoint
    import natlang_neuralese.model.shared_state as shared
    real_inputs=shared._layer_inputs
    visits=[]
    def layer_inputs(port,layer,h,attention):
        visits.append((id(layer),h.shape[1]))
        return real_inputs(port,layer,h,attention)
    monkeypatch.setattr(shared,'_layer_inputs',layer_inputs)
    def execute(reference):
        visits.clear()
        result=model(plan,reference=reference)
        if not reference:
            for layer in model.port.layers:
                rows=[count for ident,count in visits if ident==id(layer)]
                # Keep invocation order even when producer and peer share an
                # actor: bucket coalescing changes shared-base FP32 fan-in
                # after prior microbatches. Every physical row still runs once.
                assert rows==[len(node.token_ids) for node in plan.invocations],rows
                assert sum(rows)==plan.token_count
        from natlang_neuralese.model.shared_state import capture_block
        written=capture_block(result,'producer',2,4,text='t5 t6')
        receipt=result['invocation_actors']
        assert written.checkpoint_sha256==receipt['invocations']['producer']['native_identity']['checkpoint_sha256']
        assert written.adapter_binding==receipt['invocations']['producer']['adapter_binding']
        with pytest.raises(ValueError,match='actual invocation actor'):
            capture_block(result,'producer',2,4,text='t5 t6',**receipt['invocations']['reader']['native_identity'])
        logits=model.logits(result['h_final'])
        loss=logits[:,plan.offsets[1]+3:plan.offsets[1]+5].float().square().mean()
        grads=torch.autograd.grad(loss,[item[2] for item in selected],allow_unused=True)
        return result['h_final'].detach(),logits.detach(),loss.detach(),grads
    def incremental(bank):
        from natlang_neuralese.model.shared_state import concatenate_captures,splice_block
        from natlang_neuralese.model.state_block import NativeStateBlock
        from natlang_neuralese.model.native_adapters import execution_identity,execution_binding,SCHEMA
        outputs=[]
        payload=[]
        def writer():
            cache=None
            for index,token in enumerate(plan.invocations[0].token_ids):
                result=engine._native_forward([token],cache,producing=2<=index<4)
                cache=result['cache'];outputs.append(result['h_final'])
                if 2<=index<4:payload.append(result)
            identity=execution_identity(engine)
            return NativeStateBlock(text='t5 t6',**identity,
                adapter_binding={'schema':SCHEMA,'base_native_identity':engine.native_identity,'adapters':execution_binding(engine)},
                **concatenate_captures(payload))
        block=bank.execute('producer',writer)
        def reader():
            cache=None
            for index,token in enumerate(plan.invocations[1].token_ids):
                if index==2:
                    cache=splice_block(model,cache,block,checkpoint_sha256=block.checkpoint_sha256,
                        tokenizer_fingerprint=block.tokenizer_fingerprint,embedding_weight_sha256=block.embedding_weight_sha256)
                result=engine._native_forward([token],cache)
                cache=result['cache'];outputs.append(result['h_final'])
        bank.execute('reader',reader)
        def peer():
            cache=None
            for token in plan.invocations[2].token_ids:
                result=engine._native_forward([token],cache)
                cache=result['cache'];outputs.append(result['h_final'])
        bank.execute('peer',peer)
        hidden=torch.cat(outputs,dim=1);logits=model.logits(hidden)
        loss=logits[:,plan.offsets[1]+3:plan.offsets[1]+5].float().square().mean()
        grads=torch.autograd.grad(loss,[item[2] for item in selected],allow_unused=True)
        return hidden.detach(),logits.detach(),loss.detach(),grads
    def run(callback):
        from contextlib import nullcontext
        before=backend.row_counts['calls'] if backend else 0
        with backend.autocast() if backend else nullcontext():
            result=run_invocation_actors(engine,[node.name for node in plan.invocations],bindings,
                callback,leaves=leaves,gradient_precision='fp32')
        if backend:assert backend.row_counts['calls']>before
        return result
    actual=run(lambda bank:execute(False))
    expected=run(lambda bank:execute(True))
    for left,right in zip(actual[:3],expected[:3]): torch.testing.assert_close(left,right,atol=3e-6,rtol=2e-5)
    for (actor,name,_),left,right in zip(selected,actual[3],expected[3]):
        assert (left is None)==(right is None),(actor,name)
        if left is not None:
            torch.testing.assert_close(left,right,atol=3e-6,rtol=2e-5,msg=lambda message:f'{family}:{name}: {message}')
            if right.norm()>1e-8: assert (left-right).norm()/right.norm()<2e-5,(family,name)
    cached=run(incremental)
    for left,right in zip(actual[:3],cached[:3]):torch.testing.assert_close(left,right,atol=3e-6,rtol=2e-5)
    for (_,name,_),left,right in zip(selected,actual[3],cached[3]):
        assert (left is None)==(right is None),name
        if left is not None:
            torch.testing.assert_close(left,right,atol=3e-6,rtol=2e-5,msg=lambda message:f'actual Engine {family}:{name}: {message}')
            if right.norm()>1e-8:assert (left-right).norm()/right.norm()<2e-5,(family,name)
    source_grads=[grad for (actor,_,_),grad in zip(selected,actual[3]) if actor==source.id and grad is not None]
    assert sum(float(value.norm()) for value in source_grads)>0,'reader loss must reach the producer actor'
    assert not torch.equal(actual[0],baseline)
    assert all(dict(model.named_parameters())[name] is value for name,value in original.items())
    assert all(dict(model.named_buffers())[name] is value for name,value in original_buffers.items())
    assert torch.equal(model(plan)['h_final'],baseline)
    with pytest.raises(ValueError,match='outside any active'):
        run_native(engine,lambda:run_invocation_actors(engine,['x'],{'x':[target.id]},lambda bank:None),[source.id])
    bank=run_invocation_actors(engine,[node.name for node in plan.invocations],bindings,lambda value:value)
    with pytest.raises(ValueError,match='scope expired'):bank.execute('producer',lambda:None)
    def fail(bank):
        def nested():
            with pytest.raises(ValueError,match='cannot nest'):
                bank.execute('reader',lambda:None)
            raise RuntimeError('actor callback failure')
        bank.execute('producer',nested)
    with pytest.raises(RuntimeError,match='actor callback failure'):
        run_invocation_actors(engine,[node.name for node in plan.invocations],bindings,fail)
    assert all(dict(model.named_parameters())[name] is value for name,value in original.items())
    assert all(dict(model.named_buffers())[name] is value for name,value in original_buffers.items())
    engine.store.close()


def test_same_actor_matrix_reuse_preserves_each_invocation_request_binding(tmp_path):
    from natlang_neuralese.model.native_adapters import execution_identity
    engine=engine_fixture(tmp_path,'lfm')
    source,_=adapters(engine)
    bindings={'writer':[source.id],'reader':[{'id':source.id,'scale':1.}], 'base':[], 'default':None}
    def check(bank):
        assert bank.actors['writer'][1] is bank.actors['reader'][1]
        assert bank.actors['base'][1] is bank.actors['default'][1]
        for name,entries in bindings.items():
            actual=bank.execute(name,lambda:run_native(engine,lambda:execution_identity(engine),entries))
            assert actual==bank.actors[name][0][1]
    run_invocation_actors(engine,list(bindings),bindings,check)
    engine.store.close()


def test_invocation_gradient_receipt_binds_all_actor_dependencies_and_rejects_forgery(tmp_path):
    import copy
    from natlang_neuralese.model.native_actor_graph import validate_invocation_actor_receipt
    from natlang_neuralese.model.native_artifact import NativeTensorArtifact,native_tensor_dependencies
    engine=engine_fixture(tmp_path,'lfm')
    source,target=adapters(engine)
    receipt=run_invocation_actors(engine,['writer','reader'],{'writer':[source.id],'reader':[target.id]},lambda bank:bank.receipt())
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    gradient=NativeTensorArtifact('adapter-gradient',engine.native_identity,
        {name:torch.zeros_like(value) for name,value in source.artifact.tensors.items()},source.id,
        {'invocation_actors':receipt,'actor_graph_sha256':canonical_json_sha256_hex(receipt)}).validate()
    assert set(native_tensor_dependencies(gradient))=={source.id,target.id}
    with pytest.raises(ValueError,match='graph digest'):
        replace(gradient,provenance={**gradient.provenance,'actor_graph_sha256':'0'*64}).validate()
    import json,subprocess
    from pathlib import Path
    from natlang_neuralese.model.native_artifact import encode_native_tensor_artifact,native_tensor_artifact_id
    records=[source.artifact,target.artifact,gradient,
        engine.store.get_native_tensor_artifact(source.artifact.source_artifact_id).artifact,
        engine.store.get_native_tensor_artifact(target.artifact.source_artifact_id).artifact]
    files=[]
    for index,artifact in enumerate(records):
        path=tmp_path/f'actor-{index}.safetensors';path.write_bytes(encode_native_tensor_artifact(artifact));files.append(str(path))
    result=subprocess.run(['node',str(Path(__file__).resolve().parents[2]/'ts-host/test/fixtures/native-invocation-actor-wire.mjs'),
        json.dumps(files),native_tensor_artifact_id(gradient)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr

    for path in ('source','base','binding','keys'):
        forged=copy.deepcopy(receipt)
        if path=='source':forged['invocations']['writer']['native_identity']=engine.native_identity
        elif path=='base':forged['base_native_identity']={**engine.native_identity,'checkpoint_sha256':'0'*64}
        elif path=='binding':forged['invocations']['writer']['adapter_binding']=forged['invocations']['reader']['adapter_binding']
        else:forged['invocations']['writer']['extra']='ignored'
        with pytest.raises(ValueError):validate_invocation_actor_receipt(forged,engine.native_identity)
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_typed_invocation_gradient_optimizer_service_and_durable_restart(tmp_path,family):
    import json,subprocess,threading
    from pathlib import Path
    from natlang_neuralese.serve.http import serve
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,target=adapters(engine)
    roles=engine.role_tokens
    graph={'schema':'natlang.neuralese-native-graph/1','id':'distinct-native-actor-service','split':'train',
        'source_groups':['native-actor-service-fixture'],'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
        'nodes':[{'name':'producer','token_ids':[4,roles.produce_open,5,6,roles.produce_close,7],
                  'producing':[[2,4]],'target_weights':[0]*6},
                 {'name':'reader','token_ids':[8,roles.consume_open,roles.consume_close,9,10],
                  'imports':[{'at':2,'source':'producer','start':2,'stop':4}], 'target_weights':[0,0,0,1,1]},
                 {'name':'peer','token_ids':[11,12,13],'target_weights':[0]*3}]}
    fixture=tmp_path/'actor-service.json';fixture.write_text(json.dumps({'identity':engine.native_identity,'source':source.id,
        'target':target.id,'target_names':[next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))],'graph':graph,'bindings':{'producer':[source.id],'reader':[target.id],'peer':[source.id]}}))
    # The model-parameter gradient API temporarily unfreezes only its exact
    # named BF16 target. Its carrier scope must be created after that flag
    # change and shared across the producer/reader/peer actor replacements.
    # Run this while the fixture adapters are still live; the Node roundtrip
    # below intentionally updates and collects their original IDs.
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    parameter_name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    parameter=dict(engine.shared_model.named_parameters())[parameter_name]
    was_trainable,old_grad=parameter.requires_grad,parameter.grad
    model_gradient=native_graph_gradient(engine,{
        'schema':'natlang.native-gradient-request/1','graph':graph,
        'adapters':[source.id], 'invocation_adapters':{'producer':[source.id],
            'reader':[target.id], 'peer':[source.id]},
        'adapter_gradient_precision':'fp32',
        'target':{'kind':'model-parameters','names':[parameter_name]}},None)
    gradient=engine.store.get_native_tensor_artifact(model_gradient['gradient']['id']).artifact
    assert gradient.tensors[parameter_name].dtype==parameter.dtype
    assert gradient.tensors[parameter_name].norm()>0
    assert parameter.requires_grad is was_trainable and parameter.grad is old_grad
    engine.start();server=serve(engine,'127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        result=subprocess.run(['node',str(Path(__file__).resolve().parents[2]/'ts-host/test/fixtures/native-invocation-actor-service.mjs'),
            f'http://127.0.0.1:{server.server_address[1]}',str(tmp_path/'archive'),str(fixture)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    finally:
        server.shutdown();server.server_close();engine.stop();engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_model_parameter_vjp_uses_one_scope_across_distinct_invocation_actors(tmp_path,family):
    """The public parameter VJP crosses actor replacements without nesting scopes."""
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,target=adapters(engine)
    roles=engine.role_tokens
    graph={'schema':'natlang.neuralese-native-graph/1','id':'cotangent-actor-vjp',
        'split':'train','source_groups':['cotangent-actor-vjp-fixture'],
        'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
        'nodes':[{'name':'producer','token_ids':[4,roles.produce_open,5,6,roles.produce_close,7],
                  'producing':[[2,4]],'target_weights':[0]*6},
                 {'name':'reader','token_ids':[8,roles.consume_open,roles.consume_close,9,10],
                  'imports':[{'at':2,'source':'producer','start':2,'stop':4}],
                  'target_weights':[0,0,0,1,1]},
                 {'name':'peer','token_ids':[11,12,13],'target_weights':[0]*3}]}
    parameter_name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    parameter=dict(engine.shared_model.named_parameters())[parameter_name]
    was_trainable,old_grad=parameter.requires_grad,parameter.grad
    result=native_graph_gradient(engine,{'schema':'natlang.native-gradient-request/1',
        'graph':graph,'adapters':[source.id],
        'invocation_adapters':{'producer':[source.id],'reader':[target.id],'peer':[source.id]},
        'adapter_gradient_precision':'fp32',
        'target':{'kind':'model-parameters','names':[parameter_name]}},None)
    gradient=engine.store.get_native_tensor_artifact(result['gradient']['id']).artifact
    assert gradient.tensors[parameter_name].dtype==parameter.dtype
    assert gradient.tensors[parameter_name].norm()>0
    assert parameter.requires_grad is was_trainable and parameter.grad is old_grad
    engine.store.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual grouped expert FP8 execution')
@pytest.mark.parametrize('recipe',['current','current-bf16-dx'])
def test_executed_fp8_distinct_actor_graph_matches_actual_cached_adjoints(tmp_path,monkeypatch,recipe):
    monkeypatch.setenv('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cuda')
    return test_layer_major_distinct_actor_graph_matches_cached_and_source_adjoints(tmp_path,monkeypatch,'mellum',True,backend_recipe=recipe)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_distinct_actor_sampling_receipts_replay_behavior_and_source_gradient(tmp_path,monkeypatch,family):
    import copy
    from test_native_adapted_rl import capture_actor
    from natlang_neuralese.rl.native_replay import capture_invocation_actors,compile_replay,replay_logps
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,target=adapters(engine)
    _,turns=capture_actor(engine,[source.id],monkeypatch,reader_entries=[target.id],explicit_state_mode=True)
    actors=capture_invocation_actors(turns,engine.native_identity)
    assert actors['invocations']['writer']['native_identity']!=actors['invocations']['reader']['native_identity']
    fetch=lambda ident:engine.store.get_native_state(ident).block
    replay=compile_replay(turns,identity=engine.native_identity,roles=engine.role_tokens,fetch_block=fetch,invocation_actors=actors)
    assert not replay.external_blocks and replay.adapter_binding is None
    assert replay.invocation_actors==actors
    name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    leaf=source.artifact.tensors[name].clone().requires_grad_(True)
    bindings={'writer':[source.id],'reader':[target.id]}
    def execute(bank):
        actual,behavior=replay_logps(engine.shared_model,replay,chunk_tokens=2)
        torch.testing.assert_close(actual.detach(),behavior,atol=3e-6,rtol=2e-5)
        selected=[index for index,action in enumerate(replay.actions) if action['turn']=='reader']
        gradient=torch.autograd.grad(-actual[selected].mean(),leaf)[0]
        assert gradient.norm()>0
    run_invocation_actors(engine,list(bindings),bindings,execute,leaves={source.id:{**source.artifact.tensors,name:leaf}},gradient_precision='fp32')
    with pytest.raises(ValueError,match='per-invocation actor scope'):replay_logps(engine.shared_model,replay)
    forged=copy.deepcopy(actors);forged['invocations']['reader']=forged['invocations']['writer']
    with pytest.raises(ValueError,match='captured behavior'):
        compile_replay(turns,identity=engine.native_identity,roles=engine.role_tokens,fetch_block=fetch,invocation_actors=forged)
    engine.store.close()
