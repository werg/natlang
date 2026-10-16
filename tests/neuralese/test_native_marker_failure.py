"""Malformed model control draws remain real, non-executable captured turns."""
import json
import pytest
import torch
from test_native_serving_adapters import engine_fixture
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.serve.chat import RequestError


def choose(engine,monkeypatch,ids):
    values=iter(ids)
    def token(seq):
        value=next(values)
        return value,(None if seq.request.temperature<=0 else float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[value]))
    monkeypatch.setattr(engine,'_native_token',token)


def sequence(engine,kind):
    r=engine.role_tokens
    return {'close':([r.produce_close],8,'close',False),
            'consume':([r.consume_open],8,'consume-marker',False),
            'consume-close':([r.consume_close],8,'consume-marker',False),
            'consume-opened':([r.produce_open,5,r.consume_close],8,'consume-marker',False),
            'nested-limit':([r.produce_open,5,r.produce_open],3,'nested',True),
            'empty-length-open':([r.produce_open],1,'open',False),
            'nested':([r.produce_open,5,r.produce_open],8,'nested',True),
            'stop-open':([r.produce_open,5,2],8,'open',False),
            'length-open':([r.produce_open,5],2,'open',False),
            'complete-then-close':([r.produce_open,5,r.produce_close,r.produce_close],8,'close',False)}[kind]


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('kind',['close','consume','consume-close','consume-opened','nested','nested-limit','stop-open','length-open','empty-length-open','complete-then-close'])
def test_actual_bad_draws_and_fed_inputs_survive_without_requested_rl_capture(tmp_path,monkeypatch,family,kind):
    engine=engine_fixture(tmp_path,family,service_template=True)
    ids,limit,code,fed=sequence(engine,kind);choose(engine,monkeypatch,ids)
    try:
        reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4 t5'}],
            temperature=.8,max_tokens=limit,native_protocol_recovery=True))
        capture=reply['x_natlang_rl_capture'];failure=reply['x_natlang_native_failure']
        assert capture['schema']=='natlang.native-rl-sampling/4'
        assert capture['policy']==engine.native_identity
        assert capture['protocol_failure']==failure
        assert failure['code']=='neuralese-native-state-'+code
        assert [sample['token_id'] for sample in capture['samples']]==ids
        assert all(torch.isfinite(torch.tensor(sample['logp'])) for sample in capture['samples'])
        assert failure['fed'] is fed
        assert failure['sample_index']==(None if kind in ('length-open','empty-length-open') else len(ids)-1)
        assert failure['trigger']==('length' if kind in ('length-open','empty-length-open') else 'sample')
        fed_ids=[sample['token_id'] for sample in capture['generated_inputs']]
        assert fed_ids==(ids if fed or kind in ('length-open','empty-length-open') else ids[:-1])
        assert failure['generated_input_count']==len(fed_ids)
        assert reply['choices'][0]['finish_reason']=='error'
        assert not reply['choices'][0]['message'].get('tool_calls')
        assert reply['choices'][0]['message']['content']==engine.tokenizer.decode(ids,skip_special_tokens=False)
        assert reply['usage']['completion_tokens']==len(ids)
        assert len(capture['writes'])==int(kind=='complete-then-close')
        assert len(reply['native_state']['blocks'])==int(kind=='complete-then-close')
        for write in capture['writes']:
            assert engine.store.get_native_state(write['state_id']).block.token_ids.tolist()==[5]
        # The trace is scalar/token metadata, never a second KV copy per step.
        json.dumps(capture)
    finally:engine.store.close()


def test_old_clients_keep_fail_closed_and_valid_public_reply_stays_unchanged(tmp_path,monkeypatch):
    engine=engine_fixture(tmp_path,'lfm',service_template=True)
    try:
        choose(engine,monkeypatch,[engine.role_tokens.produce_close])
        with pytest.raises(RequestError,match='without an open'):
            engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4'}],temperature=.8,max_tokens=4))
        choose(engine,monkeypatch,[5,2])
        reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4'}],temperature=.8,max_tokens=4,
            native_protocol_recovery=True))
        assert 'x_natlang_native_failure' not in reply and 'x_natlang_rl_capture' not in reply
    finally:engine.store.close()


def replay_engine(tmp_path,family):
    engine=engine_fixture(tmp_path,family,service_template=True,context=8192)
    if family=='mellum':
        from natlang_neuralese.model.native_model import native_identity
        from natlang_neuralese.common.hashing import canonical_json_sha256_hex
        engine.shared_model.float()
        engine.checkpoint_sha256=canonical_json_sha256_hex({'diagnostic_parent':engine.checkpoint_sha256,
            'transform':'exact-fp32-promotion/1'})
        engine.native_identity=native_identity(engine.shared_model,engine.model_config,
            checkpoint_sha256=engine.checkpoint_sha256)
        engine.embedding_weight_sha256=engine.native_identity['embedding_weight_sha256']
    return engine


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('kind',['close','consume','consume-close','consume-opened','nested','nested-limit','stop-open','length-open','empty-length-open','complete-then-close'])
def test_every_failed_draw_replays_actual_cached_action_logps(tmp_path,monkeypatch,family,kind):
    from natlang_neuralese.rl.native_replay import compile_replay,replay_logps
    engine=replay_engine(tmp_path,family)
    ids,limit,_,_=sequence(engine,kind);choose(engine,monkeypatch,ids)
    try:
        reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4 t5'}],
            temperature=.8,max_tokens=limit,native_protocol_recovery=True))
        capture=reply['x_natlang_rl_capture']
        engine.shared_model.requires_grad_(True)
        replay=compile_replay([{'graph_turn':'failed','sampling':capture}],identity=engine.native_identity,
            roles=engine.role_tokens,fetch_block=lambda ident:engine.store.get_native_state(ident).block)
        logps,behavior=replay_logps(engine.shared_model,replay,chunk_tokens=2)
        assert len(logps)==len(ids)
        torch.testing.assert_close(logps.detach(),behavior,atol=3e-6,rtol=2e-5)
        assert replay.protocol_failures=={'failed':reply['x_natlang_native_failure']}
        assert len(replay.written_sources)==int(kind=='complete-then-close')
        gradients=torch.autograd.grad(-logps.mean(),tuple(p for p in engine.shared_model.parameters() if p.requires_grad),allow_unused=True)
        assert any(value is not None and value.norm()>0 for value in gradients)
        assert all(value is None or torch.isfinite(value).all() for value in gradients)
    finally:engine.store.close()


@pytest.mark.parametrize('tamper',['code','draw','fed','positions','trigger','stop','schema','missing'])
def test_forged_failure_law_cannot_admit_malformed_graph(tmp_path,monkeypatch,tamper):
    import copy
    from natlang_neuralese.rl.native_replay import compile_replay
    engine=replay_engine(tmp_path,'lfm');choose(engine,monkeypatch,[engine.role_tokens.produce_open,5,2])
    try:
        reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4'}],temperature=.8,
            max_tokens=8,native_protocol_recovery=True))
        capture=copy.deepcopy(reply['x_natlang_rl_capture']);failure=capture['protocol_failure']
        if tamper=='code':failure['code']='neuralese-native-state-close';failure['message']=failure['code']+': forged'
        elif tamper=='draw':failure['sample_index']=0
        elif tamper=='fed':failure['fed']=True
        elif tamper=='positions':failure['own_input_count']+=1
        elif tamper=='trigger':failure['trigger']='length';failure['sample_index']=None
        elif tamper=='stop':failure['stop_token_ids']=[9999]
        elif tamper=='schema':capture['schema']='natlang.native-rl-sampling/2'
        else:capture.pop('protocol_failure')
        with pytest.raises(ValueError,match='native'):
            compile_replay([{'graph_turn':'failed','sampling':capture}],identity=engine.native_identity,
                roles=engine.role_tokens,fetch_block=lambda _:pytest.fail('unfinished span must not fetch a fabricated artifact'))
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_http_runtime_repair_retains_failed_turn_and_consumer_graph(tmp_path,monkeypatch,family):
    import subprocess,threading
    from pathlib import Path
    from natlang_neuralese.serve.http import serve
    from natlang_neuralese.serve.native_recorded_graph import authenticated_turns
    from natlang_neuralese.rl.native_replay import compile_replay,replay_logps
    engine=replay_engine(tmp_path,family);roles=engine.role_tokens
    requests={}
    patterns=([roles.produce_open,5,roles.produce_close,roles.produce_close],
              [roles.produce_open,6,roles.produce_close,2],[7,2])
    def token(seq):
        ident=seq.request.request_id
        if ident not in requests:requests[ident]=iter(patterns[len(requests)])
        value=next(requests[ident])
        return value,(None if seq.request.temperature<=0 else float(seq.logits[0].float().div(seq.request.temperature).log_softmax(-1)[value]))
    monkeypatch.setattr(engine,'_native_token',token)
    engine.start();server=serve(engine,'127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        repo=Path(__file__).resolve().parents[2];archive=tmp_path/'archive'
        result=subprocess.run(['node',str(repo/'ts-host/test/fixtures/native-marker-recovery.mjs'),
            f'http://127.0.0.1:{server.server_address[1]}',str(archive)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        assert json.loads(result.stdout)['recovered']
        recording=json.loads((archive/'recording.json').read_text())
        prepared=authenticated_turns(recording['turns'],recording['graph'])
        # Production rollout namespaces differ from public recorder node names;
        # preserve the exact occurrence bijection and every actual event.
        import copy
        from natlang_neuralese.rl.episode_graph import rollout_turn_names
        from natlang_neuralese.rl.native_replay import prepare_objective
        indexed={event['node']:event for event in recording['graph']}
        turns=[{'invocation_id':indexed[turn['native']['graph']['graph_turn']]['call_id'],
                'messages':turn['messages'],'target':turn['reply'],'sampling':copy.deepcopy(turn['native']['sampling']),
                'replay':{'raw_calls_retained':True}} for turn in recording['turns']]
        names=rollout_turn_names(turns)
        renames={turn['graph_turn']:name for turn,name in zip(prepared,names)}
        def renamed(value):
            if isinstance(value,str):return renames.get(value,value)
            if isinstance(value,list):return [renamed(item) for item in value]
            if isinstance(value,dict):return {key:renamed(item) for key,item in value.items()}
            return value
        episode={'task_id':'private-marker-repair-fixture','sample':0,'split':'train','group':'private',
            'reward':{'quality':1},'policy':engine.native_identity,'turns':turns,
            'execution_graph':renamed(recording['graph'])}
        graphs,spec=prepare_objective([episode],identity=engine.native_identity,roles=roles,
            fetch_block=lambda ident:engine.store.get_native_state(ident).block,mode='rft')
        assert len(graphs)==1
        objective=next(iter(spec['rows'].values()))
        assert objective['episode_turns']==names
        assert len(objective['actions'])==sum(map(len,patterns))
        assert set(objective['protocol_failures'])=={names[0]}
        from natlang_neuralese.train.native_state_sft import make_native_optimizer
        engine.shared_model.requires_grad_(True)
        optimizer,named=make_native_optimizer(engine.shared_model,lr=3e-5,in_backward=False)
        replay=compile_replay(prepared,identity=engine.native_identity,roles=roles,
            fetch_block=lambda ident:engine.store.get_native_state(ident).block)
        actual,behavior=replay_logps(engine.shared_model,replay,chunk_tokens=2)
        torch.testing.assert_close(actual.detach(),behavior,atol=3e-6,rtol=2e-5)
        assert len(actual)==sum(len(pattern) for pattern in patterns)
        assert len(replay.protocol_failures)==1
        assert len(replay.written_sources)==2
        producer_gradient=torch.autograd.grad(-actual[-2:].mean(),engine.shared_model.adapter.up.weight,retain_graph=True)[0]
        assert producer_gradient.norm()>0
        before=engine.shared_model.adapter.up.weight.detach().clone()
        (-actual.mean()).backward();optimizer.step()
        assert not torch.equal(before,engine.shared_model.adapter.up.weight)
        from natlang_neuralese.rl.graph_replay import save_bundle,restore_bundle
        from natlang_neuralese.serve.store import encode_native_state,TensorStore
        from types import SimpleNamespace
        bundle=tmp_path/'bundle'
        manifest=save_bundle(recording,bundle,policy_id='actual-failure-repair',
            fetch=lambda ident:encode_native_state(engine.store.get_native_state(ident).block))
        fresh=TensorStore(directory=tmp_path/'restored')
        try:
            restore_bundle(SimpleNamespace(store=fresh,native_state_enabled=True),bundle/'manifest.json',policy_id='actual-failure-repair')
            assert len(manifest['blocks'])==2
            assert all(fresh.get_native_state(ident).id==ident for ident in manifest['blocks'])
        finally:fresh.close()
        from natlang_neuralese.train.loop import capture_training_rng_state
        saved={'optimizer':optimizer.state_dict(),'rng':capture_training_rng_state(engine.device)}
        torch.save(saved,tmp_path/'optimizer.pt')
        restored=torch.load(tmp_path/'optimizer.pt',weights_only=False)
        assert restored['optimizer'].keys()==saved['optimizer'].keys()
        assert torch.equal(restored['rng']['torch_rng'],saved['rng']['torch_rng'])
    finally:
        server.shutdown();server.server_close();engine.stop();engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_greedy_failed_turn_is_structural_only_and_adapted_failure_binds_actor(tmp_path,monkeypatch,family):
    from test_native_serving_adapters import nonzero_adapter
    from natlang_neuralese.rl.native_replay import compile_replay
    engine=replay_engine(tmp_path,family)
    try:
        choose(engine,monkeypatch,[engine.role_tokens.produce_close])
        reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4'}],
            max_tokens=4,native_protocol_recovery=True))
        capture=reply['x_natlang_rl_capture']
        assert capture['schema']=='natlang.native-structural-capture/2'
        assert capture['samples'][0]['logp'] is None
        arguments=dict(identity=engine.native_identity,roles=engine.role_tokens,fetch_block=lambda _:None)
        assert not compile_replay([{'graph_turn':'failed','sampling':capture}],structural=True,**arguments).actions
        with pytest.raises(ValueError,match='schema'):
            compile_replay([{'graph_turn':'failed','sampling':capture}],**arguments)
        adapter,_=nonzero_adapter(engine)
        choose(engine,monkeypatch,[engine.role_tokens.produce_close])
        reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4'}],temperature=.8,
            max_tokens=4,native_protocol_recovery=True,adapters=[{'id':adapter.id,'scale':.75}]))
        capture=reply['x_natlang_rl_capture']
        assert capture['adapter_binding']['adapters']==[{'id':adapter.id,'scale_hex':float(.75).hex()}]
        assert capture['policy']!=engine.native_identity
        assert compile_replay([{'graph_turn':'failed','sampling':capture}],identity=capture['policy'],
            roles=engine.role_tokens,fetch_block=lambda _:None).adapter_binding==capture['adapter_binding']
    finally:engine.store.close()


def test_inflight_completed_failure_dependencies_survive_foreign_collection(tmp_path,monkeypatch):
    engine=replay_engine(tmp_path,'lfm');roles=engine.role_tokens
    values=iter([roles.produce_open,5,roles.produce_close,roles.produce_close])
    def token(seq):
        value=next(values)
        if seq.native_rl_writes:
            ident=seq.native_rl_writes[0]['state_id']
            assert engine.store.holders(ident)=={'request-owner'}
            assert ident not in engine.store.collect(set(),owner='other-owner')
        return value,float(seq.logits[0].float().div(.8).log_softmax(-1)[value])
    monkeypatch.setattr(engine,'_native_token',token)
    try:
        reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4'}],temperature=.8,
            max_tokens=8,native_protocol_recovery=True,native_store_owner='request-owner'))
        ident=reply['x_natlang_rl_capture']['writes'][0]['state_id']
        assert engine.store.get_native_state(ident)
        assert ident in engine.store.collect(set(),owner='request-owner')
    finally:engine.store.close()


def test_normal_trace_metadata_cost_has_no_added_tensor_artifacts(tmp_path,monkeypatch):
    import time,statistics
    import natlang_neuralese.serve.engine as module
    engine=replay_engine(tmp_path,'lfm');reports=[]
    original_response=engine._native_response
    def response(seq):
        capture={'prompt':seq.native_rl_prompt,'samples':seq.rl_samples,'inputs':seq.native_rl_inputs,'writes':seq.native_rl_writes}
        reports.append({'metadata_bytes':len(json.dumps(capture).encode()),'samples':len(seq.rl_samples),
                        'retained_trace_tensor_bytes':0,'new_artifacts':len(seq.blocks)})
        return original_response(seq)
    monkeypatch.setattr(engine,'_native_response',response)
    elapsed={}
    try:
        for traced in (False,True):
            monkeypatch.setattr(module,'capture_enabled',lambda seq,enabled=traced:enabled or seq.request.rl_capture)
            durations=[]
            for repeat in range(4):
                choose(engine,monkeypatch,[5]*64+[2]);start=time.perf_counter()
                reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4 t5'}],temperature=.8,
                    max_tokens=80,native_protocol_recovery=traced))
                durations.append(time.perf_counter()-start)
                assert 'x_natlang_rl_capture' not in reply
            elapsed[str(traced)]=statistics.median(durations[1:])
        assert reports[-1]['samples']==65 and reports[-1]['new_artifacts']==0
        assert reports[-1]['metadata_bytes']<20000
        print(json.dumps({'normal_path_seconds':elapsed,'trace':reports[-1],
                         'scope':'tiny CPU real engine, paired medians, no production throughput claim'}))
    finally:engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_view_chunk_failure_keeps_template_inputs_and_prior_completed_chunk(tmp_path,monkeypatch,family):
    import threading,urllib.request
    from natlang_neuralese.serve.http import serve
    from natlang_neuralese.rl.native_replay import compile_replay,replay_logps
    engine=replay_engine(tmp_path,family);engine.max_block=8
    original_apply=engine.tokenizer.apply_chat_template
    def apply(messages,*args,**kwargs):
        if messages and messages[-1].get('tool_calls'):
            opening=original_apply([{'role':'user','content':'x'}],tools=kwargs.get('tools'),tokenize=False,add_generation_prompt=True)
            return opening+'<|tool_call_start|>[return_result(status="success", value="natlangValue7f3a9c")]<|tool_call_end|>'
        return original_apply(messages,*args,**kwargs)
    monkeypatch.setattr(engine.tokenizer,'apply_chat_template',apply)
    sequences={};patterns=([5,engine.role_tokens.produce_close],[6,2])
    def token(seq):
        if seq.request.request_id not in sequences:sequences[seq.request.request_id]=iter(patterns[len(sequences)])
        value=next(sequences[seq.request.request_id])
        return value,float(seq.logits[0].float().log_softmax(-1)[value])
    monkeypatch.setattr(engine,'_native_token',token)
    engine.start();server=serve(engine,'127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        body={'value':' '.join(['t5']*8),'window':3,'system':'t4','x_natlang_native_protocol_recovery':True}
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/neuralese/view',
            data=json.dumps(body).encode(),headers={'content-type':'application/json','x-natlang-owner':'view-owner'})
        try:
            response=urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            raise AssertionError(error.read().decode()) from error
        with response:
            assert response.status==200
            result=json.loads(response.read())
        assert 'id' not in result and len(result['source_turns'])==2
        first,last=result['source_turns']
        assert len(first['response']['x_natlang_rl_capture']['writes'])==1
        assert not last['response']['x_natlang_rl_capture']['writes']
        assert result['x_natlang_native_failure']['code']=='neuralese-native-state-open'
        turns=[{'graph_turn':str(index),'sampling':turn['response']['x_natlang_rl_capture']} for index,turn in enumerate(result['source_turns'])]
        replay=compile_replay(turns,identity=engine.native_identity,roles=engine.role_tokens,
            fetch_block=lambda ident:engine.store.get_native_state(ident).block)
        actual,expected=replay_logps(engine.shared_model,replay)
        torch.testing.assert_close(actual.detach(),expected,atol=3e-6,rtol=2e-5)
        assert len(actual)==4  # First template ends at its actual forced stop; no sampled EOS is invented.
        ident=first['response']['x_natlang_rl_capture']['writes'][0]['state_id']
        assert engine.store.holders(ident)=={'view-owner'}
    finally:server.shutdown();server.server_close();engine.stop();engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_compact_failure_archives_one_real_raw_capture_not_per_decode_step(tmp_path,monkeypatch,family):
    from natlang_neuralese.model.semantic_vae import SemanticStateVAE,StateLayout
    from natlang_neuralese.model.semantic_channel import SemanticChannel
    first=tmp_path/'first';first.mkdir();second=tmp_path/'second';second.mkdir()
    engine=replay_engine(first,family);other='mellum' if family=='lfm' else 'lfm'
    peer=replay_engine(second,other)
    try:
        source=engine.store.get_native_state(engine.native_write_text('t5')['id']).block
        target=peer.store.get_native_state(peer.native_write_text('t5')['id']).block
        # Actual two-port codec plumbing; random fixture weights make no semantic-quality claim.
        vae=SemanticStateVAE({family:StateLayout.from_block(source),other:StateLayout.from_block(target)},
            latent_width=4,hidden_width=16)
        engine.semantic_channel=SemanticChannel(vae,{family:engine.native_identity,other:peer.native_identity},
            {family:engine.tokenizer,other:peer.tokenizer},adapter_sha256='a'*64)
        stored=[];original=engine.store.put_native_state
        def put(block,owner=None):
            stored.append((block.metadata()['schema'],sum(value.numel()*value.element_size() for value in block.tensors().values())))
            return original(block,owner=owner)
        monkeypatch.setattr(engine.store,'put_native_state',put)
        choose(engine,monkeypatch,[engine.role_tokens.produce_open,5,engine.role_tokens.produce_close,engine.role_tokens.produce_close])
        reply=engine.generate(GenerationRequest(messages=[{'role':'user','content':'t4'}],temperature=.8,max_tokens=8,
            native_protocol_recovery=True,native_store_owner='compact-owner'))
        write=reply['x_natlang_rl_capture']['writes'][0]
        assert write['raw_state_id']!=write['state_id']
        assert [schema for schema,_ in stored]==['neuralese.semantic-state/1','neuralese.native-state/1']
        assert all(engine.store.holders(write[key])=={'compact-owner'} for key in ('state_id','raw_state_id'))
        assert engine.store.get_native_state(write['raw_state_id']).block.token_ids.tolist()==[5]
        print(json.dumps({'family':family,'compact_trace_stored_tensor_bytes':stored,
            'scope':'one completed one-token producer followed by unfed malformed close; no per-decode tensor copies'}))
    finally:engine.store.close();peer.store.close()
