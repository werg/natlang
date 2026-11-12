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
    for seed in (0,1):
        adapter=create_adapter(engine,{'kind':'xs','rank':2,'seed':seed,'parameter_names':names})
        tensors={name:torch.full_like(value,.04 if seed==0 else -.03) for name,value in adapter.artifact.tensors.items()}
        values.append(engine.store.put_native_tensor_artifact(replace(adapter.artifact,tensors=tensors)))
    return values


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


@pytest.mark.parametrize('family',['lfm'])
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


@pytest.mark.parametrize('family',['lfm'])
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


@pytest.mark.parametrize('family',['lfm'])
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
