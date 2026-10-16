"""Actual heterogeneous behavior admission and durable current-policy updates."""
import copy
import json
from dataclasses import replace

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_native_invocation_actors import adapters
from test_native_adapted_rl import capture_actor, post_service
from natlang_neuralese.rl.native_replay import invocation_behavior_policy, prepare_objective
from natlang_neuralese.rl.graph_replay import save_bundle, restore_bundle
from natlang_neuralese.model.native_artifact import encode_native_tensor_artifact
from natlang_neuralese.serve.store import TensorStore, encode_native_state
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached


def actual_engine_gradient(engine,graph,row,bindings,adapter_id,name):
    """Independent per-token Engine/cache oracle; production remains one graph."""
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.model.native_actor_graph import run_invocation_actors
    from natlang_neuralese.model.shared_state import concatenate_captures,splice_block
    from natlang_neuralese.model.state_block import NativeStateBlock
    from natlang_neuralese.rl.native_replay import action_logps
    from natlang_neuralese.rl.full_policy import clipped_token_loss
    example=parse_graph(graph,tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint'])
    coefficient=engine.store.get_native_tensor_artifact(adapter_id).artifact.tensors[name].clone().requires_grad_(True)
    def execute(bank):
        outputs=[];saved={}
        for node in example.plan.invocations:
            def invocation():
                cache=None
                for index,token in enumerate(node.token_ids):
                    for span in node.imports:
                        if span.at!=index:continue
                        if span.source in example.external_states:
                            block=engine.store.get_native_state(example.external_states[span.source]['id']).block
                        else:
                            source=example.plan.invocations[[item.name for item in example.plan.invocations].index(span.source)]
                            actor=bank.receipt()['invocations'][span.source]
                            capture=concatenate_captures([saved[(span.source,position)] for position in range(span.start,span.stop)])
                            block=NativeStateBlock(text=engine.tokenizer.decode(list(source.token_ids[span.start:span.stop])),
                                **actor['native_identity'],adapter_binding=actor['adapter_binding'],**capture)
                        cache=splice_block(engine.shared_model,cache,block,checkpoint_sha256=block.checkpoint_sha256,
                            tokenizer_fingerprint=block.tokenizer_fingerprint,embedding_weight_sha256=block.embedding_weight_sha256)
                    output=engine._native_forward([token],cache,producing=any(start<=index<stop for start,stop in node.producing))
                    cache=output['cache'];saved[(node.name,index)]=output;outputs.append(output['h_final'])
            bank.execute(node.name,invocation)
        hidden=torch.cat(outputs,dim=1)[0]
        current=action_logps(engine.shared_model,hidden,row['actions'])[0]
        old=current.new_tensor([action['behavior_logp'] for action in row['actions']])
        loss=clipped_token_loss(current,old,row['advantage'],clip_low=.2,clip_high=.28).mean()
        return loss.detach(),torch.autograd.grad(loss,coefficient)[0],current.detach(),old
    return run_invocation_actors(engine,list(bindings),bindings,execute,
        leaves={adapter_id:{**engine.store.get_native_tensor_artifact(adapter_id).artifact.tensors,name:coefficient}},gradient_precision='fp32')


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('external',[False,True])
def test_distinct_policy_two_actual_http_updates_and_restart(tmp_path,monkeypatch,family,external):
    torch.manual_seed(1731)
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,target=adapters(engine)
    block_id,turns=capture_actor(engine,[source.id],monkeypatch,reader_entries=[target.id],explicit_state_mode=True)
    rows=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],
           'sampling':turn['sampling'],'replay':{'raw_calls_retained':True}} for turn in (turns[1:] if external else turns)]
    episodes=[{'task_id':'heterogeneous-actor-train','split':'train','sample':sample,'group':'same-task',
        'reward':{'quality':reward},'policy':{'checkpoint_sha256':engine.native_identity['checkpoint_sha256']},
        'turns':copy.deepcopy(rows)} for sample,reward in enumerate((1,0))]
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    fetch_block=lambda ident:engine.store.get_native_state(ident).block
    policy=invocation_behavior_policy(episodes,identity=engine.native_identity,roles=engine.role_tokens,fetch_block=fetch_block)
    owner=canonical_json_sha256_hex(policy)
    for change in ({'split':'test'},{'training_admission':'held'},{'reward':{'quality':.5}},
                   {'policy':{'checkpoint_sha256':'0'*64}}):
        with pytest.raises(ValueError,match='scored train-only'):
            invocation_behavior_policy([{**episodes[0],**change}],identity=engine.native_identity,
                roles=engine.role_tokens,fetch_block=fetch_block)

    def fetch(ident):
        return (encode_native_tensor_artifact(engine.store.get_native_tensor_artifact(ident).artifact)
                if ident.startswith('nnt1_') else encode_native_state(fetch_block(ident)))
    manifest=save_bundle(episodes,tmp_path/'bundle',policy_id=owner,fetch=fetch)
    assert set(manifest['blocks'])=={source.id,target.id,source.artifact.source_artifact_id,target.artifact.source_artifact_id,block_id}
    bundle=tmp_path/'bundle/manifest.json'
    graphs,spec=prepare_objective(episodes,identity=engine.native_identity,roles=engine.role_tokens,
        fetch_block=fetch_block,block_bundle=bundle,behavior_policy=policy)
    assert spec['schema']=='natlang.native-policy-objective/2' and 'policy' not in spec and 'adapter_binding' not in spec
    path=tmp_path/'objective.json';path.write_text(json.dumps(spec));original=path.read_bytes()
    engine.store.close();engine.store=TensorStore(directory=tmp_path/'restart')
    restore_bundle(engine,bundle,policy_id=owner)
    graph=graphs[0]
    bindings={node['name']:[source.id if node['name'].startswith('writer#') else target.id] for node in graph['nodes']}
    trained=target if external else source
    name=next(name for name in trained.artifact.tensors if name.endswith('v_proj.weight'))
    reference=torch.nn.Parameter(trained.artifact.tensors[name].clone())
    optimizer=torch.optim.Adam([reference],lr=.01)
    current_id,state_id=trained.id,None
    gradients=[]
    gradient_ids=[]
    for update in range(2):
        body={'schema':'natlang.native-gradient-request/1','native_identity':engine.native_identity,'graph':graph,
              'invocation_adapters':bindings,'adapter_gradient_precision':'fp32',
              'objective':{'kind':'native-policy','path':str(path),'sha256':sha256_file_hex_cached(path)},
              'target':{'kind':'adapter','id':current_id,'names':[name]}}
        if external:
            from urllib.error import HTTPError
            with pytest.raises(HTTPError):post_service(engine,'/v1/neuralese/grad',body)
            body['native_state_modes']={block_id:'same-base-adapter-state'}
        reply=post_service(engine,'/v1/neuralese/grad',body)
        artifact=engine.store.get_native_tensor_artifact(reply['gradient']['id']).artifact
        assert artifact.provenance['behavior_policy']==policy
        assert artifact.provenance['behavior_policy_sha256']==owner
        assert 'behavior_native_identity' not in artifact.provenance and 'effective_native_identity' not in artifact.provenance
        assert artifact.tensors[name].norm()>0
        expected_loss,expected_grad,current_logps,behavior_logps=actual_engine_gradient(engine,graph,spec['rows'][graph['id']],bindings,current_id,name)
        torch.testing.assert_close(expected_loss,expected_loss.new_tensor(reply['loss']),atol=3e-6,rtol=2e-5)
        torch.testing.assert_close(artifact.tensors[name].to(expected_grad),expected_grad,atol=3e-6,rtol=2e-5)
        assert (artifact.tensors[name].to(expected_grad)-expected_grad).norm()/expected_grad.norm()<2e-5
        if update==0:torch.testing.assert_close(current_logps,behavior_logps,atol=3e-6,rtol=2e-5)
        gradients.append(artifact.tensors[name]);gradient_ids.append(reply['gradient']['id'])
        if external:
            contracts=artifact.provenance['external_consumption_receipts']
            assert len(contracts)==1
            receipt=next(iter(contracts.values()))
            captured=next(iter(spec['rows'][graph['id']]['external_invocations'].values()))[0]['receipt']
            assert receipt['target_position_start']==captured['target_position_start']
            assert receipt['source_native_identity']==captured['source_native_identity']
            assert receipt['target_native_identity']==artifact.provenance['invocation_actors']['invocations'][next(iter(artifact.provenance['invocation_actors']['invocations']))]['native_identity']
        # A different coefficient is permitted, a changed basis/scaled program
        # is not the current policy of these immutable behavior samples.
        if update==1:
            from urllib.error import HTTPError
            forged=copy.deepcopy(body)
            changed=next(node for node,ids in bindings.items() if current_id in ids)
            forged['invocation_adapters'][changed]=[{'id':current_id,'scale':2.}]
            with pytest.raises(HTTPError) as error:post_service(engine,'/v1/neuralese/grad',forged)
            assert b'ordered behavior adapter scale' in error.value.read()

        reference.grad=artifact.tensors[name].clone();optimizer.step();optimizer.zero_grad()
        optim={'schema':'natlang.native-optimizer-request/1','native_identity':engine.native_identity,
               'gradient_id':reply['gradient']['id'],'optimizer':{'name':'adam','learning_rate':.01}}
        if state_id:optim['optimizer_state_id']=state_id
        result=post_service(engine,'/v1/neuralese/optim',optim)
        updated=result['adapter']['id']
        assert torch.equal(engine.store.get_native_tensor_artifact(updated).artifact.tensors[name],reference)
        receipt=result['invocation_actors']
        for raw,packed in result['invocation_source_names'].items():
            actor=receipt['invocations'][packed]
            expected=updated if (external or raw.startswith('writer#')) else target.id
            assert actor['adapter_binding']['adapters'][0]['id']==expected
        state_id=result['optimizer_state_id']
        state=engine.store.get_native_tensor_artifact(state_id).artifact
        assert state.provenance['invocation_actors']==receipt
        bindings={node:[updated if ident==current_id else ident for ident in ids] for node,ids in bindings.items()}
        current_id=updated
        engine.store.close();engine.store=TensorStore(directory=tmp_path/'restart')
        assert path.read_bytes()==original
    assert not torch.equal(*gradients)
    from natlang_neuralese.model.native_artifact import native_tensor_dependencies
    last=engine.store.get_native_tensor_artifact(gradient_ids[-1]).artifact
    dependencies=native_tensor_dependencies(last)
    assert source.id in dependencies and target.id in dependencies
    assert current_id not in dependencies  # latest gradient predates final update
    with pytest.raises(ValueError,match='descriptor digest'):
        replace(last,provenance={**last.provenance,'behavior_policy_sha256':'0'*64}).validate()
    from pathlib import Path
    import subprocess
    pending=[*gradient_ids,state_id];saved=set();files=[]
    while pending:
        ident=pending.pop()
        if ident in saved:continue
        saved.add(ident)
        artifact=engine.store.get_native_tensor_artifact(ident).artifact
        pending.extend(native_tensor_dependencies(artifact))
        file=tmp_path/f'{ident}.bin';file.write_bytes(encode_native_tensor_artifact(artifact));files.append(str(file))
    result=subprocess.run(['node',str(Path(__file__).resolve().parents[2]/'ts-host/test/fixtures/native-invocation-policy-wire.mjs'),
        json.dumps(files),gradient_ids[-1],state_id],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    if (family=='lfm' and not external) or (family=='mellum' and external):
        import threading
        from natlang_neuralese.serve.http import serve
        original_bindings={node['name']:[source.id if node['name'].startswith('writer#') else target.id] for node in graph['nodes']}
        fixture=tmp_path/'typed-policy.json';fixture.write_text(json.dumps({'identity':engine.native_identity,
            'parent':trained.id,'target_names':[name],'graph':graph,'bindings':original_bindings,
            'objective':{'kind':'native-policy','path':str(path),'sha256':sha256_file_hex_cached(path)},
            'behavior_policy_sha256':owner,'native_state_modes':{block_id:'same-base-adapter-state'} if external else {}}))
        server=serve(engine,'127.0.0.1',0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            result=subprocess.run(['node',str(Path(__file__).resolve().parents[2]/'ts-host/test/fixtures/native-invocation-policy-service.mjs'),
                f'http://127.0.0.1:{server.server_address[1]}',str(tmp_path/'typed-archive'),str(fixture)],capture_output=True,text=True)
            assert result.returncode==0,result.stdout+'\n'+result.stderr
        finally:server.shutdown();server.server_close();thread.join()
    engine.store.close()
