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


def actor(tmp_path,family,*,lexical=False):
    dtype=source(tmp_path/'base',family)
    if lexical:
        import json
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(tmp_path/'base')
        state=json.loads(tokenizer.backend_tokenizer.to_str())
        vocab=state['model']['vocab']
        for token,text in [(4,'[bad('),(5,'[good(')]:
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


@pytest.mark.parametrize('family',['lfm','mellum'])
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
        torch.testing.assert_close(seq.logits,expected_logits,rtol=0,atol=0)
        next_actual=engine._native_forward([10],seq.cache)
        next_reference=engine._native_forward([10],reference['cache'])
        torch.testing.assert_close(next_actual['h_final'],next_reference['h_final'],rtol=0,atol=0)
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
        assert all(delta.get('native_state',{}).get('id') in [r.id for r in seq.blocks] for delta in deltas if 'native_state' in delta)
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


@pytest.mark.parametrize('family',['lfm','mellum'])
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
