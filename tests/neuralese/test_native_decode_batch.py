import copy
import pytest
import torch
from natlang_neuralese.model.shared_state import SharedStateModel,native_step
from natlang_neuralese.model.native_decode_batch import native_step_batch
from test_shared_state import tiny


def model(family):
    if family=='qwen35':
        from test_qwen35_native import tiny_port
        return SharedStateModel(tiny_port(),(58,59,60,61),rank=4)
    return SharedStateModel(tiny(family),(58,59,60,61),rank=4)


@pytest.mark.parametrize('family',['lfm','mellum','qwen','qwen35','lfm3'])
@pytest.mark.parametrize('grad',[False,True])
def test_unequal_cached_lanes_mixed_producing_match_independent_states_and_gradients(family,grad):
    a=model(family);b=copy.deepcopy(a)
    for m in(a,b):m.adapter.up.weight.data.fill_(.03)
    with torch.set_grad_enabled(grad):
        ca=[];cb=[]
        for tokens in [[2,3],[4,5,6,7],[]]:
            ids=torch.tensor([tokens],dtype=torch.long)
            ca.append(native_step(a,ids)['cache']);cb.append(native_step(b,ids)['cache'])
        actual=[];expected=[]
        for turn in range(3):
            ids=torch.tensor([[8+turn],[11+turn],[14+turn]])
            flags=[True,False,turn%2==0]
            outputs=native_step_batch(a,ids,ca,producing=flags)
            singles=[native_step(b,ids[i:i+1],cb[i],producing=flags[i])for i in range(3)]
            for output,single in zip(outputs,singles):
                torch.testing.assert_close(output['h_final'],single['h_final'],atol=3e-6,rtol=3e-5)
                torch.testing.assert_close(output['embeddings'],single['embeddings'])
                assert output['cache'].lengths==single['cache'].lengths
                for x,y in zip(output['layers'],single['layers']):
                    assert x.keys()==y.keys()
                    for name in x:
                        if name!='kind':torch.testing.assert_close(x[name],y[name],atol=3e-6,rtol=3e-5)
                for x,y in zip(output['cache'].states,single['cache'].states):
                    for name in ('k','v','window','conv','recurrent'):
                        if hasattr(x,name):torch.testing.assert_close(getattr(x,name),getattr(y,name),atol=3e-6,rtol=3e-5)
            ca=[o['cache']for o in outputs];cb=[o['cache']for o in singles]
            actual.extend(o['h_final']for o in outputs);expected.extend(o['h_final']for o in singles)
        if grad:
            sum(v.square().sum()for v in actual).backward();sum(v.square().sum()for v in expected).backward()
            for (name,p),(name2,q)in zip(a.named_parameters(),b.named_parameters()):
                assert name==name2
                if p.grad is None:assert q.grad is None
                else:torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=5e-4)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_batched_engine_keeps_complete_producer_import_and_rl_capture(tmp_path,family):
    from concurrent.futures import Future
    from natlang_neuralese.serve.engine import Sequence,GenerationRequest
    from test_native_serving_adapters import engine_fixture
    from test_native_rl_replay import actor_rollout
    engine=engine_fixture(tmp_path,family)
    (tmp_path/'reference').mkdir()
    reference=engine_fixture(tmp_path/'reference',family)
    # Bind both executions to the same actual model/identity for exact block IDs.
    reference.shared_model=copy.deepcopy(engine.shared_model);reference.backbone=reference.shared_model.port
    reference.role_tokens=engine.role_tokens;reference.model_config=engine.model_config
    reference.native_identity=engine.native_identity;reference.checkpoint_sha256=engine.checkpoint_sha256
    reference.embedding_weight_sha256=engine.embedding_weight_sha256
    role=engine.role_tokens
    _,_,_,block_id,block,_=actor_rollout(family,loaded=(engine.shared_model,role,engine.native_identity))
    for e in(engine,reference):e.store.put_native_state(block)
    try:
        sequences=[]
        for e in(engine,reference):
            seqs=[]
            for i,content in enumerate(['first user prompt',[
                {'type':'text','text':'consume'}, {'type':'neuralese_state_ref','id':block_id}]]):
                seq=Sequence(GenerationRequest(messages=[{'role':'user','content':content}],
                    max_tokens=10,temperature=.8,seed=21+i,rl_capture=True,request_id=str(i)),Future())
                seq.prompt_plan=e._prompt_plan(seq.request.messages,seq.request.tools)
                seqs.append(seq)
            scripts={str(i):iter([role.produce_open,5+i,6+i,role.produce_close,9+i,next(iter(e.stop_ids))])for i in range(2)}
            def token(seq,scripts=scripts):
                value=next(scripts[seq.request.request_id]);logp=seq.logits.float().div(.8).log_softmax(-1)[0,value]
                return value,float(logp)
            e._native_token=token
            sequences.append(seqs)
        with torch.no_grad():
            for _ in range(9):
                engine._native_round(sequences[0])
                for seq in sequences[1]:reference._native_run(lambda:reference._native_round_impl([seq]),seq.request.adapters)
        for seqs in sequences:
            for seq in seqs:
                if seq.future.done():seq.future.result()
        assert engine._native_batched_decode_rounds>=4
        for a,b in zip(*sequences):
            assert a.future.done() and b.future.done()
            a.future.result();b.future.result()
            assert a.native_own_position==b.native_own_position and len(a.items)==len(b.items)
            for x,y in zip(a.items,b.items):
                if isinstance(x,int):assert x==y
                else:
                    x,y=x.block,y.block
                    assert x.text==y.text
                    torch.testing.assert_close(x.token_ids,y.token_ids)
                    torch.testing.assert_close(x.embeddings,y.embeddings,atol=3e-6,rtol=3e-5)
                    for xl,yl in zip(x.layers,y.layers):
                        assert xl['kind']==yl['kind']
                        for key in xl:
                            if key!='kind':torch.testing.assert_close(xl[key],yl[key],atol=3e-6,rtol=3e-5)
            assert a.finish_reason==b.finish_reason=='stop'
            assert a.native_rl_inputs==b.native_rl_inputs
            assert len(a.native_rl_writes)==len(b.native_rl_writes)==1
            for x,y in zip(a.rl_samples,b.rl_samples):
                assert {k:v for k,v in x.items()if k!='logp'}=={k:v for k,v in y.items()if k!='logp'}
                assert x['logp']==pytest.approx(y['logp'],abs=2e-5)
            torch.testing.assert_close(a.logits,b.logits,atol=3e-6,rtol=3e-5)
    finally:engine.store.close();reference.store.close()


def test_engine_batches_only_equal_adapter_declarations_and_isolates_cancelled_request(tmp_path,monkeypatch):
    from concurrent.futures import Future
    from natlang_neuralese.serve.engine import Sequence,GenerationRequest
    from natlang_neuralese.model import native_decode_batch as batch
    from test_native_serving_adapters import engine_fixture,nonzero_adapter
    engine=engine_fixture(tmp_path,'lfm');adapter,_=nonzero_adapter(engine)
    seqs=[];calls=[];step=batch.native_step_batch
    def record(model,ids,caches,**kwargs):
        calls.append(len(caches));return step(model,ids,caches,**kwargs)
    monkeypatch.setattr(batch,'native_step_batch',record)
    for index in range(5):
        request=GenerationRequest(messages=[{'role':'user','content':'hello'}],max_tokens=2,
            adapters=None if index<2 else [{'id':adapter.id,'scale':.5}],cancelled=index==4)
        seqs.append(Sequence(request,Future(),prompt_plan=engine._prompt_plan(request.messages,None)))
    engine._native_token=lambda seq:(11,-.2)
    try:
        with torch.no_grad():
            engine._native_round(seqs);engine._native_round(seqs)
        assert calls==[2,2]
        assert seqs[-1].future.done() and seqs[-1].future.exception() is not None
        assert all(not seq.future.done() for seq in seqs[:-1])
        assert not torch.equal(seqs[0].logits,seqs[2].logits)
        assert engine.progress_metrics()['natlang_native_batched_decode_tokens_total']==4
    finally:engine.store.close()
