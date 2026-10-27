"""Immutable native KV optimization retains exact producer and update history."""
from dataclasses import replace
import copy
import pytest
import torch
from test_native_serving_adapters import engine_fixture
from natlang_neuralese.model.native_artifact import NativeTensorArtifact,encode_native_tensor_artifact,decode_native_tensor_artifact
from natlang_neuralese.model.native_derivative_contract import derivation_dependencies
from natlang_neuralese.model.native_state_derivation import optimize_state,verify_state_derivation,state_values
from natlang_neuralese.serve.store import encode_native_state,decode_native_state,TensorStore


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('algorithm',['sgd','adam'])
def test_actual_state_updates_restore_immutable_origin_and_named_moments(tmp_path,family,algorithm):
    engine=engine_fixture(tmp_path,family,service_template=True)
    parent_id=engine.native_write_text('t5 t6')['id']
    parent=engine.store.get_native_state(parent_id).block
    name=next(iter(state_values(parent)))
    options={'name':algorithm,'learning_rate':.01,'momentum':.4,'eps':1e-6}
    ids=[parent_id];history_id=None
    original={key:value.clone() for key,value in parent.tensors().items()}
    try:
        for step in (1,2):
            record=engine.store.get_native_state(ids[-1])
            gradient=engine.store.put_native_tensor_artifact(NativeTensorArtifact('state-gradient',engine.native_identity,
                {name:torch.full_like(state_values(record.block)[name],.03)},record.id,{'scope':'test-real-state-update'}))
            result,live,moments=optimize_state(engine,gradient,options,history_id,'state-owner')
            ids.append(result['native_state_id']);history_id=result['optimizer_state_id']
            updated=engine.store.get_native_state(ids[-1]).block
            assert updated.derivation['step']==step and updated.derivation['origin_state_ids']==[parent_id]
            assert updated.derivation['parent_state_id']==ids[-2]
            verify_state_derivation(engine.store,updated)
            assert torch.equal(updated.tensors()[name],live.tensors()[name].detach().cpu())
            for key,value in updated.tensors().items():
                if key!=name:assert torch.equal(value,original[key])
            assert gradient.id in derivation_dependencies(updated.derivation)
        # Full dependency closure retains origins, both gradients/updates and the
        # prior moments even when only the latest state/history remain roots.
        engine.store.collect({ids[-1],history_id},owner='state-owner')
        assert all(engine.store.get_native_state(ident) is not None for ident in ids)
        saved=engine.store.get_native_state(ids[-1]).block
        decoded,ident=decode_native_state(encode_native_state(saved))
        assert ident==ids[-1] and decoded.derivation==saved.derivation
        verify_state_derivation(engine.store,decoded)
        # Real generation consumes updated KV/hybrid values and reports their
        # origin/update contract, rather than silently reencoding source text.
        from concurrent.futures import Future
        from natlang_neuralese.serve.engine import GenerationRequest,Sequence
        from natlang_neuralese.model.shared_state import Invocation,ImportSpan,compile_graph
        seq=Sequence(GenerationRequest(messages=[]),Future())
        seq.cache=engine._native_forward([4],None)['cache'];seq.native_own_position=1
        engine._native_consume(seq,ids[-1])
        actual=engine._native_forward([7],seq.cache)['h_final'][:,-1]
        plan=compile_graph([Invocation('read',(4,engine.role_tokens.consume_open,engine.role_tokens.consume_close,7),
            imports=(ImportSpan(2,'prior',0,len(saved.token_ids)),))],external_sizes={'prior':len(saved.token_ids)})
        expected=engine.shared_model(plan,external_blocks={'prior':saved},external_identity={'prior':engine.native_identity})['h_final'][:,-1]
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        assert seq.native_imports[0]['mode']=='exact-native-state' and seq.native_imports[0]['state_derivation']==saved.derivation
        wrong=copy.deepcopy(saved.derivation);wrong['origin_state_ids']=[ids[-2]]
        with pytest.raises(ValueError,match='original producer'):engine.store.put_native_state(replace(saved,derivation=wrong))
        tensors=state_values(saved);tensors[name]=tensors[name]+1
        from natlang_neuralese.model.native_state_derivation import replace_state_values
        with pytest.raises(ValueError,match='numeric update'):engine.store.put_native_state(replace_state_values(saved,tensors))
        # Matching hashes/provenance are insufficient: forge BOTH the numeric
        # update and selected state, keeping their bytes self-consistent.
        actual_update=engine.store.get_native_tensor_artifact(saved.derivation['update_artifact_id']).artifact
        forged_update=engine.store.put_native_tensor_artifact(replace(actual_update,
            tensors={key:value+1 for key,value in actual_update.tensors.items()}))
        forged_derivation={**saved.derivation,'update_artifact_id':forged_update.id}
        with pytest.raises(ValueError,match='exact shared optimizer arithmetic'):
            engine.store.put_native_state(replace_state_values(saved,forged_update.artifact.tensors,derivation=forged_derivation))
        prior=engine.store.get_native_tensor_artifact(saved.derivation['optimizer_state_parent_id'])
        if algorithm=='adam':
            forged_moments={**prior.artifact.tensors,'m.'+name:prior.artifact.tensors['m.'+name]+.25}
        else:
            forged_moments={**prior.artifact.tensors,'momentum.'+name:prior.artifact.tensors['momentum.'+name]+.25}
        forged_history=engine.store.put_native_tensor_artifact(replace(prior.artifact,tensors=forged_moments))
        with pytest.raises(ValueError,match='history differs from the exact shared optimizer'):
            optimize_state(engine,gradient,options,forged_history.id)
        wrong_runtime={**saved.derivation['arithmetic'],'machine':'unqualified-other-cpu'}
        runtime_update=engine.store.put_native_tensor_artifact(replace(actual_update,
            provenance={**actual_update.provenance,'arithmetic':wrong_runtime}))
        with pytest.raises(ValueError,match='bound CPU runtime'):
            engine.store.put_native_state(replace(saved,derivation={**saved.derivation,
                'arithmetic':wrong_runtime,'update_artifact_id':runtime_update.id}))
        with pytest.raises(ValueError,match='prior state'):optimize_state(engine,gradient,options,'nnt1_'+'a'*52)
        with pytest.raises(ValueError,match='finite numeric settings'):optimize_state(engine,gradient,{},None)
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_inference_snapshot_matches_optimizer_state_forward_and_reader_gradients(tmp_path,family):
    """Serving snapshots omit only optimizer ancestry; training and whole-graph state remain original."""
    from concurrent.futures import Future
    from natlang_neuralese.model.native_artifact import NativeTensorArtifact
    from natlang_neuralese.model.native_state_derivation import optimize_state, state_values, verify_state_derivation
    from natlang_neuralese.model.shared_state import Invocation, ImportSpan, compile_graph
    from natlang_neuralese.serve.engine import GenerationRequest, Sequence
    engine=engine_fixture(tmp_path,family,service_template=True)
    try:
        parent_id=engine.native_write_text('t5 t6')['id']
        parent=engine.store.get_native_state(parent_id).block
        name=next(iter(state_values(parent)))
        gradient=engine.store.put_native_tensor_artifact(NativeTensorArtifact('state-gradient',engine.native_identity,
            {name:torch.full_like(state_values(parent)[name],.03)},parent_id,{'scope':'inference-snapshot-parity'}))
        result,_,_=optimize_state(engine,gradient,{'name':'adam','learning_rate':.01,'eps':1e-6},None,'state-owner')
        source_id=result['native_state_id'];source=engine.store.get_native_state(source_id).block
        verify_state_derivation(engine.store,source)
        snapshot_record=engine.store.put_native_state(replace(source,derivation=None))
        snapshot_id=snapshot_record.id;snapshot=snapshot_record.block
        assert snapshot_id!=source_id and snapshot.derivation is None
        assert source.derivation['parent_state_id']==parent_id and source.derivation['gradient_id']==gradient.id
        assert source.derivation['origin_state_ids']==[parent_id]
        assert set(snapshot.tensors())==set(source.tensors())
        for key,value in source.tensors().items():assert torch.equal(snapshot.tensors()[key],value),key
        verify_state_derivation(engine.store,source)

        # Exercise the real LFM/Mellum engine import and next-token forward for both IDs.
        def consume(state_id):
            sequence=Sequence(GenerationRequest(messages=[]),Future())
            sequence.cache=engine._native_forward([4],None)['cache'];sequence.native_own_position=1
            engine._native_consume(sequence,state_id)
            return sequence.logits.detach().clone()
        torch.testing.assert_close(consume(source_id),consume(snapshot_id),rtol=0,atol=0)

        # The same consumer graph and current learnable model receive the same exact state at the same positions.
        engine.shared_model.requires_grad_(True)
        plan=compile_graph([Invocation('read',(4,engine.role_tokens.consume_open,engine.role_tokens.consume_close,7),
            imports=(ImportSpan(2,'prior',0,len(source.token_ids)),))],external_sizes={'prior':len(source.token_ids)})
        gradients=[];losses=[]
        for block in (source,snapshot):
            engine.shared_model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                output=engine.shared_model(plan,external_blocks={'prior':block},external_identity={'prior':engine.native_identity})
                loss=output['h_final'][:,-1].square().sum();loss.backward()
            losses.append(loss.detach().clone())
            gradients.append({name:param.grad.detach().clone() for name,param in engine.shared_model.named_parameters()
                if param.grad is not None})
        torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
        assert gradients[0].keys()==gradients[1].keys()
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=0,atol=0)
        # Snapshot identity never substitutes into a producer/consumer training export.
        assert source.derivation['gradient_id']==gradient.id
        assert engine.store.get_native_state(source_id).block.derivation==source.derivation
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('objective_kind',['ce','query'])
def test_python_ts_program_state_archive_nz_browser_and_restore(tmp_path,family,objective_kind):
    import base64,json,os,subprocess
    from pathlib import Path
    from test_native_derivative_program import program
    from natlang_neuralese.serve.native_derivative import execute_derivative_program
    from test_native_serving_adapters import nonzero_adapter
    engine=engine_fixture(tmp_path,family,service_template=True)
    adapter,_=nonzero_adapter(engine)
    source=program(engine,adapter,'sgd')
    ident=engine.native_write_text('t5 t6')['id']
    source['inputs']={'a':{'kind':'native-state','id':ident}}
    if objective_kind=='query':
        from natlang_neuralese.model.native_readout import snapshot_readout
        readout_id=snapshot_readout(engine).id
    for objective in [source['outer'],*(node for node in source['nodes'] if node['kind']=='gradient')]:
        objective['request']=copy.deepcopy(objective['request']);objective['request'].pop('adapters')
        graph=objective['request']['graph'];graph['external_states']={'prior':{'id':ident,'token_count':2}}
        graph['nodes']=graph['nodes'][1:];graph['nodes'][0]['imports']=[{'source':'prior','start':0,'stop':2,'at':2}]
        if objective_kind=='query':
            from test_native_query_objective import query_spec
            node=graph['nodes'][0]
            objective['request']['objective']=query_spec(graph,[{'kind':'nll','reduction':'mean','readout':readout_id,'actions':[
                {'query':index-1,'token_id':token} for index,(token,weight) in
                enumerate(zip(node['token_ids'],node['target_weights'])) if weight]}])
    try:
        result=execute_derivative_program(engine,source)
        artifacts={}
        for (key,) in engine.store._db.execute('SELECT id FROM blocks'):
            record=engine.store.get_native_state(key) if key.startswith('nzs1_') else engine.store.get_native_tensor_artifact(key)
            body=encode_native_state(record.block) if key.startswith('nzs1_') else encode_native_tensor_artifact(record.artifact)
            artifacts[key]=base64.b64encode(body).decode()
        spec=tmp_path/'bundle.json';spec.write_text(json.dumps({'root':result['outputs']['a2'],'artifacts':artifacts}))
        output=tmp_path/'returned.json';repo=Path(__file__).resolve().parents[2]
        process=subprocess.run(['node',str(repo/'ts-host/test/fixtures/native-derivative-archive.mjs'),str(spec),str(output),str(tmp_path/'node-archive')],capture_output=True,text=True)
        assert process.returncode==0,process.stdout+'\n'+process.stderr
        restored=TensorStore(tmp_path/'restored')
        # Python store enforces the same exact dependency-first state contract.
        pending={key:base64.b64decode(body) for key,body in json.loads(output.read_text()).items()}
        while pending:
            progress=False
            for key,body in list(pending.items()):
                if key.startswith('nnt1_'):
                    from natlang_neuralese.model.native_artifact import native_tensor_dependencies
                    artifact,decoded=decode_native_tensor_artifact(body)
                    if any(dependency in pending for dependency in native_tensor_dependencies(artifact)):continue
                    restored.put_native_tensor_artifact(artifact)
                else:
                    block,decoded=decode_native_state(body)
                    if block.derivation and any(dependency in pending for dependency in derivation_dependencies(block.derivation)):continue
                    restored.put_native_state(block)
                assert decoded==key;del pending[key];progress=True
            assert progress
        verify_state_derivation(restored,restored.get_native_state(result['outputs']['a2']).block)
        restored.close()
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_http_typed_state_updates_and_restart(tmp_path,family):
    import json,subprocess,threading
    from pathlib import Path
    from natlang_neuralese.serve.http import serve
    engine=engine_fixture(tmp_path,family,service_template=True)
    ident=engine.native_write_text('t5 t6')['id'];roles=engine.role_tokens
    source={'id':ident,'identity':engine.native_identity,'graph':{
        'schema':'natlang.neuralese-native-graph/1','id':'actual-state-service','split':'train','source_groups':['state-service-training-only'],
        'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],'nodes':[
            {'name':'reader','token_ids':[4,roles.consume_open,roles.consume_close,7,8],
             'imports':[{'source':'prior','start':0,'stop':2,'at':2}],'target_weights':[0,0,0,1,1]}]}}
    binding=tmp_path/'service.json';binding.write_text(json.dumps(source))
    engine.start();server=serve(engine,'127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        repo=Path(__file__).resolve().parents[2]
        result=subprocess.run(['node',str(repo/'ts-host/test/fixtures/native-state-optimizer-service.mjs'),
            f'http://127.0.0.1:{server.server_address[1]}',str(tmp_path/'archive'),str(binding)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        returned=json.loads(result.stdout);assert returned['restored'] is True
        updated=engine.store.get_native_state(returned['id']).block
        assert updated.derivation['step']==2
    finally:server.shutdown();server.server_close();engine.stop();engine.store.close()


@pytest.mark.parametrize('compact,training',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('target_name',['lfm','mellum'])
def test_optimized_state_semantic_encoding_decoding_preserves_actual_source(tmp_path,compact,training,target_name):
    from types import SimpleNamespace
    from test_native_semantic import setup_pair
    from natlang_neuralese.train.native_semantic import parse_pair,producer
    from natlang_neuralese.model.semantic_channel import SemanticChannel
    participants,row,vae=setup_pair()
    import os
    device=os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu')
    vae.to(device)
    for participant in participants.values():participant.model.to(device)
    source_name='mellum' if target_name=='lfm' else 'lfm'
    source=participants[source_name];target=participants[target_name]
    channel=SemanticChannel(vae,{name:item.identity for name,item in participants.items()},
        {name:item.tokenizer for name,item in participants.items()},adapter_sha256='a'*64)
    store=TensorStore(tmp_path/'states')
    example,view=parse_pair(row,participants)[source_name]
    with torch.no_grad():block=producer(source,example,view,supervise=False)[0]
    original=store.put_native_state(block)
    name=next(iter(state_values(block)))
    engine=SimpleNamespace(store=store,native_identity=source.identity,device=next(source.model.parameters()).device)
    gradient=store.put_native_tensor_artifact(NativeTensorArtifact('state-gradient',source.identity,
        {name:torch.full_like(state_values(block)[name],.02)},original.id))
    try:
        updated,_,_=optimize_state(engine,gradient,{'name':'sgd','learning_rate':.1},None)
        current=store.get_native_state(updated['native_state_id'])
        with torch.no_grad():
            if compact:
                with pytest.raises(ValueError,match='exact source state ID'):channel.message(current.block)
                current=store.put_native_state(channel.message(current.block,source_state_id=current.id))
                assert current.block.derivation['kind']=='semantic-encode'
        execution=None;decoder=channel;target_identity=target.identity
        if training:
            bindings={name:{**item.identity,'checkpoint_sha256':'d'*64} for name,item in participants.items()}
            decoder=SemanticChannel(copy.deepcopy(vae),bindings,channel.tokenizers,adapter_sha256='c'*64)
            with torch.no_grad():next(decoder.vae.shared_decoder.parameters()).add_(.01)
            target_identity=bindings[target_name]
            execution={'schema':'natlang.native-semantic-policy-execution/1','parent_checkpoint_sha256':'d'*64,
                'parent_adapter_sha256':'c'*64,'behavior_adapter_sha256':'a'*64,'parent_step':2,'step':3}
        decoded,receipt=decoder.adapt(current.block,source_state_id=current.id,target_name=target_name,target_model=target.model,
            target_tokenizer=target.tokenizer,target_identity=target_identity,
            **({'source_channel':channel,'training_execution':execution} if training else {}))
        if training:
            assert decoded.derivation['source_adapter_sha256']=='a'*64
            assert decoded.derivation['adapter_sha256']=='c'*64
            assert decoded.derivation['training_execution']==execution
            assert receipt['source_native_identity']==source.identity
            from natlang_neuralese.model.shared_state import Invocation,ImportSpan,compile_graph
            _,_,consume_open,consume_close=target.model.marker_ids
            reader=compile_graph([Invocation('reader',(4,consume_open,consume_close,7),imports=(
                ImportSpan(2,'external',0,len(decoded.token_ids)),))],external_sizes={'external':len(decoded.token_ids)})
            observed=target.model(reader,external_blocks={'external':decoded},external_identity=target_identity)
            target.model.logits(observed['h_final'][0,-1]).float().log_softmax(-1)[8].neg().backward()
            assert any(value.grad is not None and value.grad.norm()>0 for value in decoder.vae.shared_decoder.parameters())
        actual=store.put_native_state(decoded)
        assert actual.block.derivation['kind']=='semantic-adapt'
        assert actual.block.derivation['parent_state_id']==current.id
        assert actual.block.derivation['origin_state_ids']==[original.id]
        verify_state_derivation(store,actual.block)
        if training:
            for mutate in ('source-adapter','decoder-adapter','step','parent','missing','source-identity'):
                wrong=copy.deepcopy(actual.block.derivation)
                if mutate=='source-adapter':wrong['source_adapter_sha256']='b'*64
                elif mutate=='decoder-adapter':wrong['adapter_sha256']='b'*64
                elif mutate=='step':wrong['training_execution']['step']=1
                elif mutate=='parent':wrong['training_execution']['parent_checkpoint_sha256']='b'*64
                elif mutate=='missing':wrong.pop('training_execution')
                else:wrong['source_native_identity']=target_identity
                with pytest.raises(ValueError):store.put_native_state(replace(actual.block,derivation=wrong))
        wrong=copy.deepcopy(actual.block.derivation);wrong['source_native_identity']=target.identity
        with pytest.raises(ValueError,match='source, target'):store.put_native_state(replace(actual.block,derivation=wrong))
        store.collect({actual.id})
        assert store.get_native_state(original.id) is not None
        if compact:
            # The raw optimizer parent, semantic code and actual decoded target
            # survive every archive together, without relabeling the producer.
            import base64,json,subprocess
            from pathlib import Path
            artifacts={}
            for (key,) in store._db.execute('SELECT id FROM blocks'):
                record=store.get_native_state(key) if key.startswith('nzs1_') else store.get_native_tensor_artifact(key)
                wire=encode_native_state(record.block) if key.startswith('nzs1_') else encode_native_tensor_artifact(record.artifact)
                artifacts[key]=base64.b64encode(wire).decode()
            spec=tmp_path/'semantic-bundle.json';spec.write_text(json.dumps({'root':actual.id,'artifacts':artifacts}))
            returned=tmp_path/'semantic-returned.json';repo=Path(__file__).resolve().parents[2]
            result=subprocess.run(['node',str(repo/'ts-host/test/fixtures/native-derivative-archive.mjs'),
                str(spec),str(returned),str(tmp_path/'semantic-node')],capture_output=True,text=True)
            assert result.returncode==0,result.stdout+'\n'+result.stderr
            restored,ident=decode_native_state(base64.b64decode(json.loads(returned.read_text())[actual.id]))
            assert ident==actual.id and restored.derivation==actual.block.derivation
    finally:store.close()
