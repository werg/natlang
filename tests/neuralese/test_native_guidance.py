"""Actual native cache/conv rollback and the shared guided sampling law.

The rejection policy and pre-rejection action choices are explicit control
fixtures, not evidence of learned task quality. Logits, masks, states, gradients,
and post-rollback stochastic draws are the actual two model implementations.
"""
from concurrent.futures import Future
from types import SimpleNamespace
import pytest
import torch

from natlang_neuralese.common.sampling import masked_log_distribution
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.engine import GenerationRequest, Sequence
from natlang_neuralese.serve.guidance import Settings
from natlang_neuralese.serve.native_guidance import render
from natlang_neuralese.rl.native_replay import compile_replay, replay_logps
from test_native_full_updates import source


LEXICAL=[(4,'[bad('),(5,'[good(')]
THINK=[(6,'<think>'),(7,'</think>')]


def actor(tmp_path,family,*,lexical=False,words=LEXICAL):
    dtype=source(tmp_path/'base',family)
    if lexical:
        import json
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(tmp_path/'base')
        state=json.loads(tokenizer.backend_tokenizer.to_str())
        vocab=state['model']['vocab']
        for token,text in words:
            del vocab[f't{token}'];vocab[text]=token
        tokenizer._tokenizer=Tokenizer.from_str(json.dumps(state))
        tokenizer.save_pretrained(tmp_path/'base')
    model,_,_,config=load_native_model(tmp_path/'base',dtype=dtype,rank=4)
    path=tmp_path/'actor.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model)},path)
    return load_native_engine(str(path),dtype=dtype)


class RejectOnce:
    def __init__(self,seq,engine,*,block):
        self.settings=SimpleNamespace(retries=4);self.rejections=[];self.rejected=False
        self.seq,self.engine,self.block=seq,engine,block
    def check(self,text):
        if not self.rejected and len(self.seq.items)==3:
            self.rejected=True
            _,starts,_=render(self.engine,self.seq)
            return 'controlled rejection',starts[0 if self.block else 1]
    def rewind(self,offset):pass
    def accept(self,offset):pass


def chosen(engine,seq,token):
    law=masked_log_distribution(seq.logits[0],seq.request.temperature,
                                banned=sorted(seq.bans.get(seq.native_own_position,())))
    assert torch.isfinite(law[token])
    return token,float(law[token])


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('reject_block',[False,True])
def test_actual_hybrid_cache_rollback_discards_rejected_effects_and_replays_law(tmp_path,monkeypatch,family,reject_block):
    engine=actor(tmp_path,family)
    imported=engine.native_write_text('t12')['id']
    prompt=[('tokens',[4,9]),('native_state',imported),('tokens',[11])]
    deltas=[]
    seq=Sequence(GenerationRequest(messages=[],temperature=.8,seed=729,rl_capture=True,
        guidance=Settings(syntax=False),on_delta=deltas.append,max_tokens=30),Future(),
        prompt_plan=prompt)
    with torch.no_grad():
        engine._native_prefill(seq)
        seq.guide=RejectOnce(seq,engine,block=reject_block)
        roles=engine.role_tokens
        script=iter([roles.produce_open,5,6,roles.produce_close,7,8])
        real_sampler=engine._native_token
        monkeypatch.setattr(engine,'_native_token',lambda s:chosen(engine,s,next(script)))
        for _ in range(6):engine._native_round([seq])
        assert not seq.future.done(),seq.future.exception() if seq.future.done() else ''
        assert seq.guide.rejections and deltas==[]
        assert seq.items==([] if reject_block else [seq.blocks[0]])
        assert len(seq.native_rl_writes)==(0 if reject_block else 1)
        assert not seq.native_capture_open and not seq.native_capture_ids and not seq.native_capture_chunks
        rejected=roles.produce_open if reject_block else 7
        assert seq.bans[seq.native_own_position]=={rejected}
        # Fresh explicit cached reference, including the actual producing role.
        fresh=Sequence(GenerationRequest(messages=[]),Future(),prompt_plan=prompt)
        engine._native_prefill(fresh)
        reference={'cache':fresh.cache,'h_final':None}
        if not reject_block:
            for token,role in [(roles.produce_open,False),(5,True),(6,True),(roles.produce_close,False)]:
                reference=engine._native_forward([token],reference['cache'],producing=role)
        expected_logits=fresh.logits if reject_block else engine._native_logits(reference['h_final'])
        # Another cache shape may round differently on some BLAS paths: FP32 roundoff, far below a state error.
        torch.testing.assert_close(seq.logits,expected_logits,rtol=1e-5,atol=1e-6)
        next_actual=engine._native_forward([10],seq.cache)
        next_reference=engine._native_forward([10],reference['cache'])
        torch.testing.assert_close(next_actual['h_final'],next_reference['h_final'],rtol=1e-5,atol=1e-6)
        # Actual stochastic continuation, same request RNG (rejections do not rewind it).
        seq.bans[seq.native_own_position].update([roles.produce_open,roles.produce_close,roles.consume_open,roles.consume_close])
        rng=torch.Generator().set_state(seq.rng.get_state())
        distribution=masked_log_distribution(seq.logits[0],.8,banned=sorted(seq.bans[seq.native_own_position]))
        expected=int(torch.multinomial(distribution.exp().cpu(),1,generator=rng))
        monkeypatch.setattr(engine,'_native_token',real_sampler)
        token,logp=engine._native_token(seq)
        assert token==expected and logp==float(distribution[token]) and token!=rejected
        # Replay the same stochastic draw in the normal scheduler.
        seq.rng.set_state(rng.get_state())  # start from a fresh documented next draw
        engine._native_round([seq])
        assert not seq.future.done() or seq.future.exception() is None
        if not seq.future.done():engine._finish(seq,'length')
        response=seq.future.result()
        # Finishing releases sequence-owned block objects; compare streamed
        # committed states against the response's durable metadata snapshot.
        response_block_ids=[r['id'] for r in response['native_state']['blocks']]
        assert all(delta.get('native_state',{}).get('id') in response_block_ids
                   for delta in deltas if 'native_state' in delta)
        if reject_block:assert not seq.blocks and not response['native_state']['blocks']
        capture=response['x_natlang_rl_capture']
        assert capture['sampling_law']['guidance'] is True
        replay=compile_replay([{'graph_turn':'guided','sampling':capture}],identity=engine.native_identity,
            roles=roles,fetch_block=lambda key:engine.store.get_native_state(key).block)
    engine.shared_model.requires_grad_(True)
    logps,behavior=replay_logps(engine.shared_model,replay)
    # BF16 reference uses the same graph/step kernels; retain existing numerical qualification tolerance.
    torch.testing.assert_close(logps.detach(),behavior,rtol=2e-5,atol=2e-6)
    (-logps[-1]).backward()
    if not reject_block:
        assert engine.shared_model.adapter.up.weight.grad.abs().sum()>0
    assert any(value.grad is not None and value.grad.abs().sum()>0 for value in engine.shared_model.parameters())


def test_sparse_and_allowed_masks_are_one_full_vocabulary_gradient_law():
    logits=torch.randn(3,11,requires_grad=True)
    a=masked_log_distribution(logits,.7,allowed=[0,2,4,6,8,10])
    b=masked_log_distribution(logits,.7,banned=[1,3,5,7,9])
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a[:,2].sum(),logits,retain_graph=True)[0]
    gb=torch.autograd.grad(b[:,2].sum(),logits)[0]
    assert torch.equal(ga,gb) and ga[:,[1,3,5,7,9]].count_nonzero()==0


@pytest.mark.parametrize('family',['lfm'])
def test_real_stochastic_rejection_keeps_rng_and_records_only_accepted_action(tmp_path,family):
    engine=actor(tmp_path,family)
    seq=Sequence(GenerationRequest(messages=[],temperature=.8,seed=417,rl_capture=True,
        guidance=Settings(syntax=False),max_tokens=20),Future(),prompt_plan=[('tokens',[4,9])])
    with torch.no_grad():
        engine._native_prefill(seq)
        class FirstOnly:
            settings=SimpleNamespace(retries=4)
            rejections=[]
            fired=False
            def check(self,text):
                if not self.fired:
                    self.fired=True
                    return 'controlled first draw rejection',0
            def rewind(self,offset):pass
        seq.guide=FirstOnly()
        own=seq.native_own_position
        roles=engine.role_tokens
        structural={roles.produce_open,roles.produce_close,roles.consume_open,roles.consume_close,*engine.stop_ids}
        seq.bans[own]=set(structural)
        rng=torch.Generator().set_state(seq.rng.get_state())
        first_law=masked_log_distribution(seq.logits[0],.8,banned=sorted(structural))
        first=int(torch.multinomial(first_law.exp().cpu(),1,generator=rng))
        engine._native_round([seq])
        assert not seq.future.done() and not seq.items and not seq.rl_samples
        assert torch.equal(seq.rng.get_state(),rng.get_state())
        assert seq.bans[own]==structural|{first}
        second_law=masked_log_distribution(seq.logits[0],.8,banned=sorted(structural|{first}))
        second=int(torch.multinomial(second_law.exp().cpu(),1,generator=rng))
        engine._native_round([seq])
        assert seq.items==[second] and second!=first
        assert torch.equal(seq.rng.get_state(),rng.get_state())
        engine._finish(seq,'length')
        capture=seq.future.result()['x_natlang_rl_capture']
        assert len(capture['samples'])==1
        assert capture['samples'][0]['token_id']==second
        assert capture['samples'][0]['banned_token_ids']==sorted(structural|{first})
        assert capture['samples'][0]['logp']==float(second_law[second])
        replay=compile_replay([{'graph_turn':'stochastic','sampling':capture}],identity=engine.native_identity,
            roles=roles,fetch_block=lambda _:pytest.fail('unexpected external state'))
    logps,behavior=replay_logps(engine.shared_model,replay)
    torch.testing.assert_close(logps,behavior,rtol=2e-5,atol=2e-6)


def test_actual_guide_tool_policy_uses_native_rollback_and_teaches_forced_conflict(tmp_path,monkeypatch):
    engine=actor(tmp_path,'lfm',lexical=True)
    deltas=[]
    request=GenerationRequest(messages=[],temperature=.8,rl_capture=True,
        guidance=Settings(tools=['good'],syntax=False),on_delta=deltas.append)
    seq=Sequence(request,Future(),prompt_plan=[('tokens',[9])])
    with torch.no_grad():
        engine._native_prefill(seq)
        monkeypatch.setattr(engine,'_native_token',lambda s:chosen(engine,s,4))
        engine._native_round([seq])
        assert not seq.future.done() and seq.items==[]
        assert seq.guide.rejections==[{'reason':'unknown-tool','offset':1}]
        assert seq.bans[seq.native_own_position]=={4} and deltas==[]
        monkeypatch.setattr(engine,'_native_token',lambda s:chosen(engine,s,5))
        engine._native_round([seq])
        engine._finish(seq,'length')
        assert ''.join(delta.get('text','') for delta in deltas)=='[good('
        assert seq.future.result()['x_natlang_guidance']['rejections'][0]['reason']=='unknown-tool'
        forced=Sequence(request,Future(),prompt_plan=[('tokens',[9])])
        engine._native_prefill(forced)
        forced.native_forced_tokens=[4]
        engine._native_round([forced])
        assert forced.future.done()
        error=forced.future.exception()
        assert error.code=='neuralese-guidance-template-conflict'
        assert 'compatible call template' in str(error)


@pytest.mark.parametrize('opened',[True,False])
def test_open_reasoning_streams_before_the_guided_reply_is_final(tmp_path,monkeypatch,opened):
    """Fork a6401bbd3: reasoning the generation prompt opened streams as it is generated, a rejection after it never
    reaches the stream, and the rest of the reply is held back until it is final. Without open reasoning the whole
    reply is held back as before."""
    engine=actor(tmp_path,'lfm',lexical=True,words=LEXICAL+THINK)
    deltas=[]
    request=GenerationRequest(messages=[],temperature=.8,rl_capture=True,
        guidance=Settings(tools=['good'],syntax=False),on_delta=deltas.append,max_tokens=20)
    seq=Sequence(request,Future(),prompt_plan=[('tokens',[9]+([6] if opened else []))])
    script=iter([10,11,7,4,5])
    with torch.no_grad():
        engine._native_prefill(seq)
        assert seq.guide.settings.reasoning_open is opened
        monkeypatch.setattr(engine,'_native_token',lambda s:chosen(engine,s,next(script)))
        for _ in range(2):engine._native_round([seq])
        assert deltas==([{'reasoning':'t10'},{'reasoning':' t11'}] if opened else [])
        engine._native_round([seq])  # the closing tag is held back with the reply
        engine._native_round([seq])  # `[bad(` is rejected and rolled back
        assert seq.guide.rejections==[{'reason':'unknown-tool','offset':len('t10 t11 </think> [')}]
        assert seq.bans[seq.native_own_position]=={4}
        assert deltas==([{'reasoning':'t10'},{'reasoning':' t11'}] if opened else [])
        engine._native_round([seq])
        engine._finish(seq,'length')
    assert not any('[bad(' in str(delta) for delta in deltas)
    reasoning=''.join(d.get('reasoning','') for d in deltas)
    text=''.join(d.get('text','') for d in deltas)
    if opened:
        assert reasoning.strip()=='t10 t11' and '</think>' not in reasoning+text and text.strip()=='[good('
    else:
        assert not reasoning and text.replace(' ','')=='t10t11</think>[good('
    assert seq.future.result()['x_natlang_guidance']['rejections'][0]['reason']=='unknown-tool'


def test_streamed_guided_reply_sends_open_reasoning_as_reasoning_content(tmp_path,monkeypatch):
    """The server's stream: the open reasoning arrives as `reasoning_content` deltas (as the managed server sends it),
    the rest of the reply as content after it is final, without the closing tag."""
    import json,threading,urllib.request
    from natlang_neuralese.serve.http import serve
    engine=actor(tmp_path,'lfm',lexical=True,words=LEXICAL+THINK)
    engine._prompt_plan=lambda messages,tools:[('tokens',[9,6])]
    script=iter([10,11,7,4,5,sorted(engine.stop_ids)[0]])
    monkeypatch.setattr(engine,'_native_token',lambda s:chosen(engine,s,next(script)))
    engine.start()
    server=serve(engine,'127.0.0.1',0)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        body={'messages':[{'role':'user','content':'t9'}],'temperature':.8,'max_tokens':12,'stream':True,
              'guidance':{'tools':['good'],'syntax':False}}
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions',
            data=json.dumps(body).encode(),headers={'content-type':'application/json'})
        with urllib.request.urlopen(request,timeout=60) as response:
            chunks=[json.loads(line[5:]) for line in response.read().decode().splitlines()
                    if line.startswith('data:') and line.strip()!='data: [DONE]']
    finally:
        server.shutdown();engine.stop()
    deltas=[chunk['choices'][0]['delta'] for chunk in chunks if 'choices' in chunk]
    reasoning=[d['reasoning_content'] for d in deltas if 'reasoning_content' in d]
    content=''.join(d.get('content','') for d in deltas if isinstance(d.get('content'),str))
    assert reasoning[:2]==['t10',' t11'] and ''.join(reasoning).strip()=='t10 t11'
    assert content.strip()=='[good(' and '</think>' not in content
    # reasoning deltas precede every content delta
    kinds=[('reasoning_content' in d) for d in deltas if 'reasoning_content' in d or d.get('content')]
    assert kinds==sorted(kinds,reverse=True)
    assert chunks[-1]['x_natlang_guidance']['rejections'][0]['reason']=='unknown-tool'


def _stream_chunks(engine,body):
    import json,threading,urllib.request
    from natlang_neuralese.serve.http import serve
    server=serve(engine,'127.0.0.1',0)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        request=urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions',
            data=json.dumps({**body,'stream':True}).encode(),headers={'content-type':'application/json'})
        with urllib.request.urlopen(request,timeout=60) as response:
            return [json.loads(line[5:]) for line in response.read().decode().splitlines()
                    if line.startswith('data:') and line.strip()!='data: [DONE]']
    finally:
        server.shutdown()


def _split(chunks):
    deltas=[chunk['choices'][0]['delta'] for chunk in chunks if 'choices' in chunk]
    return ([d['reasoning_content'] for d in deltas if 'reasoning_content' in d],
            ''.join(d.get('content','') for d in deltas if isinstance(d.get('content'),str)))


@pytest.mark.parametrize('opened',[True,False])
def test_unguided_stream_sends_open_reasoning_as_reasoning_content(tmp_path,monkeypatch,opened):
    """An unguided streamed reply continuing the reasoning its prompt opened streams it as `reasoning_content` and the
    rest as content without the closing tag (the managed server's shape); without open reasoning all text is content."""
    engine=actor(tmp_path,'lfm',lexical=True,words=LEXICAL+THINK)
    engine._prompt_plan=lambda messages,tools:[('tokens',[9]+([6] if opened else []))]
    script=iter([10,11,7,12,sorted(engine.stop_ids)[0]])
    monkeypatch.setattr(engine,'_native_token',lambda s:chosen(engine,s,next(script)))
    engine.start()
    try:
        chunks=_stream_chunks(engine,{'messages':[{'role':'user','content':'t9'}],'temperature':.8,'max_tokens':12})
    finally:
        engine.stop()
    reasoning,content=_split(chunks)
    if opened:
        # (the fixture's word-level tokenizer decodes each streamed token without its separating space)
        assert reasoning[0]=='t10' and ''.join(reasoning).replace(' ','')=='t10t11'
        assert content.strip()=='t12' and '</think>' not in content
    else:
        assert not reasoning and content.replace(' ','')=='t10t11</think>t12'
    assert chunks[-1]['x_natlang_message']['content'].strip()=='t12'


def test_stream_holds_back_a_split_closing_tag_and_ends_inside_reasoning():
    """The HTTP stream's split of announced reasoning: a closing tag split across deltas never leaks into either
    channel, and a reply that ends inside its reasoning sends all of it as reasoning."""
    import threading
    from concurrent.futures import Future
    def fake(items,message):
        def submit(request):
            future=Future()
            def run():
                for item in items:request.on_delta(item)
                future.set_result({'choices':[{'message':message,'finish_reason':'stop'}],'usage':{}})
            threading.Thread(target=run).start()
            return future
        return SimpleNamespace(model_name='m',tool_call_format='xml',submit=submit)
    body={'messages':[{'role':'user','content':'q'}],'temperature':.8}
    split=[{'reasoning_open':True},{'text':'a '},{'text':'b</thi'},{'text':'nk>\n\nAns'},{'text':'wer.'}]
    chunks=_stream_chunks(fake(split,{'role':'assistant','content':'Answer.','reasoning_content':'a b'}),body)
    reasoning,content=_split(chunks)
    assert reasoning==['a ','b'] and content=='\n\nAnswer.'
    unfinished=[{'reasoning_open':True},{'text':'x </th'}]
    chunks=_stream_chunks(fake(unfinished,{'role':'assistant','content':'','reasoning_content':'x </th'}),body)
    reasoning,content=_split(chunks)
    assert ''.join(reasoning)=='x </th' and content==''
