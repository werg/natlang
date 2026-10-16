"""Explicit exact-base raw-state transport; different actor is never relabelled."""
from dataclasses import replace
from concurrent.futures import Future
import pytest
import torch
from test_native_serving_adapters import engine_fixture, nonzero_adapter
from natlang_neuralese.model.native_adapters import run_native, adapter_state_input_receipt
from natlang_neuralese.model.shared_state import Invocation, ImportSpan, compile_graph
from natlang_neuralese.serve.engine import GenerationRequest, Sequence


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_explicit_adapter_state_cached_graph_and_source_contract(tmp_path,family):
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,_=nonzero_adapter(engine)
    target,_=nonzero_adapter(engine,'tiny')
    record=engine.native_write_text('t5 t6',adapters=[source.id])
    ident=record['id']; block=engine.store.get_native_state(ident).block
    modes={ident:'same-base-adapter-state'}
    before={name:value for name,value in engine.shared_model.named_parameters()}
    def execute():
        receipt=adapter_state_input_receipt(engine,block,ident,target_position_start=2)
        assert receipt['source_native_identity'] != receipt['target_native_identity']
        seq=Sequence(GenerationRequest(messages=[],adapters=[target.id],native_state_modes=modes,rl_capture=True),Future())
        own=[4,engine.role_tokens.consume_open,engine.role_tokens.consume_close,7]
        first=engine._native_forward([4],None);seq.cache=first['cache'];seq.native_own_position=1
        engine._native_consume(seq,ident)
        current=engine._native_forward([7],seq.cache)
        plan=compile_graph([Invocation('consumer',tuple(own),imports=(ImportSpan(2,'external',0,len(block.token_ids)),))],
            external_sizes={'external':len(block.token_ids)})
        result=engine.shared_model(plan,external_blocks={'external':block},external_identity={'external':receipt['source_native_identity']})
        torch.testing.assert_close(current['h_final'][0,-1],result['h_final'][0,-1],atol=3e-6 if family=='lfm' else .005,rtol=2e-5 if family=='lfm' else .005)
        assert seq.native_imports[0]['adapter_state_consumption']==receipt
        assert seq.native_rl_prompt[1]['adapter_state_consumption']==receipt
        assert block.checkpoint_sha256==receipt['source_native_identity']['checkpoint_sha256']
        malformed=replace(block,layers=(*block.layers[:-1],{**block.layers[-1],
            'raw_k':block.layers[-1]['raw_k'][:,:1], 'v':block.layers[-1]['v'][:,:1]})) if block.layers[-1]['kind']=='attention' else replace(block,
            layers=(*block.layers[:-1],{'kind':'finite_convolution','gated_inputs':block.layers[-1]['gated_inputs'][:,:1]}))
        with pytest.raises(ValueError,match='head layout|convolution channels'):
            adapter_state_input_receipt(engine,malformed,ident,target_position_start=2)
        wrong=replace(block,adapter_binding=None,checkpoint_sha256='1'*64)
        with pytest.raises(ValueError,match='exact same base'):
            adapter_state_input_receipt(engine,wrong,ident,target_position_start=2)
    run_native(engine,execute,[target.id])
    assert all(dict(engine.shared_model.named_parameters())[key] is value for key,value in before.items())
    # The source artifact alone is insufficient: the exact coefficient closure
    # must be restored even though its tensor dimensions match.
    original=engine.store.get_native_tensor_artifact
    engine.store.get_native_tensor_artifact=lambda value: None if value==source.id else original(value)
    with pytest.raises(ValueError,match='restore its coefficients'):
        run_native(engine,lambda:adapter_state_input_receipt(engine,block,ident,target_position_start=2),[target.id])
    engine.store.get_native_tensor_artifact=original
    bad=Sequence(GenerationRequest(messages=[],native_state_modes={ident:'typo'}),Future(),prompt_plan=[('native_state',ident)])
    with pytest.raises(ValueError,match='native_state_modes'):
        engine._native_prefill(bad)
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_adapter_state_receipt_forgery_and_per_input_identity_inventory(tmp_path,family):
    import copy
    from natlang_neuralese.model.native_adapters import validate_adapter_state_receipt, execution_binding, SCHEMA
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,_=nonzero_adapter(engine);target,_=nonzero_adapter(engine,'tiny')
    ident=engine.native_write_text('t5 t6',adapters=[source.id])['id']
    block=engine.store.get_native_state(ident).block
    def execute():
        receipt=adapter_state_input_receipt(engine,block,ident,target_position_start=1)
        binding={'schema':SCHEMA,'base_native_identity':engine.native_identity,'adapters':execution_binding(engine)}
        validate_adapter_state_receipt(block,receipt,target_identity=engine.execution_native_identity,target_binding=binding)
        for field,value in [('source_state_id','nzs1_'+'a'*52),('target_position_start',-1),
                ('tokens',len(block.token_ids)+1),('source_adapter_binding',None),
                ('target_adapter_binding',None),('base_native_identity',{})]:
            wrong=copy.deepcopy(receipt);wrong[field]=value
            with pytest.raises(ValueError,match='receipt differs'):
                validate_adapter_state_receipt(block,wrong,target_identity=engine.execution_native_identity,target_binding=binding)
        plan=compile_graph([Invocation('consumer',(4,7),imports=(ImportSpan(1,'state',0,len(block.token_ids)),))],
            external_sizes={'state':len(block.token_ids)})
        for identities in ({'wrong':receipt['source_native_identity']},{'state':{}},
                {'state':receipt['target_native_identity']}):
            with pytest.raises(ValueError):
                engine.shared_model(plan,external_blocks={'state':block},external_identity=identities)
    run_native(engine,execute,[target.id])
    engine.store.close()


def test_gradient_contract_retains_both_actor_dependencies(tmp_path):
    from natlang_neuralese.model.native_artifact import NativeTensorArtifact,native_tensor_dependencies
    engine=engine_fixture(tmp_path,'lfm',service_template=True)
    source,name=nonzero_adapter(engine);target,_=nonzero_adapter(engine,'tiny')
    ident=engine.native_write_text('t5',adapters=[source.id])['id']
    block=engine.store.get_native_state(ident).block
    contract=run_native(engine,lambda:adapter_state_input_receipt(engine,block,ident,target_position_start=1),[target.id])
    gradient=NativeTensorArtifact('adapter-gradient',engine.native_identity,
        {name:torch.zeros_like(target.artifact.tensors[name])},target.id,
        {'external_adapter_state_contracts':{'input':contract}})
    assert set(native_tensor_dependencies(gradient))=={source.id,target.id}
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_explicit_mode_contextual_capture_and_score_service_retains_both_actors(tmp_path,family):
    from test_native_adapted_rl import post_service
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,_=nonzero_adapter(engine);target,_=nonzero_adapter(engine,'tiny')
    source_id=engine.native_write_text('t5 t6',adapters=[source.id])['id']
    placeholder=engine.native_write_text('t4')['id']
    modes={source_id:'same-base-adapter-state'}
    messages=[{'role':'user','content':[{'type':'text','text':'t4 '},
        {'type':'native_state','id':source_id},{'type':'text','text':' t5 '},
        {'type':'native_state','id':placeholder}]}]
    result=post_service(engine,'/v1/neuralese/native-states/init-body',{
        'messages':messages,'placeholder':placeholder,'text':'t7 t8','adapters':[target.id],
        'native_state_modes':modes})
    imported=result['source_capture']['imports'][0]
    receipt=imported['adapter_state_consumption']
    assert imported['mode']=='same-base-adapter-state' and imported['observed_native_state_id']==source_id
    assert receipt['target_native_identity']==result['source_capture']['policy']
    assert receipt['source_native_identity']!=receipt['target_native_identity']
    assert receipt['target_native_identity']['checkpoint_sha256']==result['native_state']['checkpoint_sha256']
    segments=[segment for segment in result['source_capture']['segments'] if segment['kind']=='native_state']
    assert segments[0]['adapter_state_consumption']==receipt
    decision=post_service(engine,'/v1/neuralese/decide',{'messages':messages,'options':['t7','t8'],
        'adapters':[target.id],'native_state_modes':modes})
    assert len(decision['log_probs'])==2
    consumed=decision['native_state_imports'][0]['adapter_state_consumption']
    assert consumed['source_native_identity']==receipt['source_native_identity']
    assert consumed['target_native_identity']==receipt['target_native_identity']
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_typed_contextual_adapter_state_client_and_archive(tmp_path,family):
    import json,subprocess,threading
    from pathlib import Path
    from natlang_neuralese.serve.http import serve
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,_=nonzero_adapter(engine);target,_=nonzero_adapter(engine,'tiny')
    source_id=engine.native_write_text('t5 t6',adapters=[source.id])['id']
    placeholder=engine.native_write_text('t4')['id']
    body={'messages':[{'role':'user','content':[{'type':'text','text':'t4 '},
        {'type':'native_state','id':source_id},{'type':'text','text':' t5 '},
        {'type':'native_state','id':placeholder}]}], 'placeholder':placeholder,'text':'t7 t8',
        'adapters':[target.id],'native_state_modes':{source_id:'same-base-adapter-state'}}
    fixture=tmp_path/'typed-context.json';fixture.write_text(json.dumps(body))
    engine.start();server=serve(engine,'127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        repo=Path(__file__).resolve().parents[2]
        result=subprocess.run(['node',str(repo/'ts-host/test/fixtures/native-adapter-state-client.mjs'),
            f'http://127.0.0.1:{server.server_address[1]}',str(fixture)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    finally:
        server.shutdown();server.server_close();engine.stop();engine.store.close()
