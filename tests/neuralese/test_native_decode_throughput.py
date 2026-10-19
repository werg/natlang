"""Same native math/state and per-call bounds after inference launch reduction."""
from types import SimpleNamespace
import pytest
import torch
from test_shared_state import tiny
from natlang_neuralese.model import shared_state as shared


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_single_query_geometry_is_exact_and_bounds_never_alias(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    shared._decode_block_geometry.cache_clear()
    for keys in [1,127,128,129,255,256,257]:
        for query in sorted({0,keys//2,keys-1}):
            for window in [None,1,3,128,129]:
                plan=shared.AttentionLayout((query,),keys)
                with torch.no_grad():actual=shared.graph_block_mask(plan,device,window)
                with torch.enable_grad():expected=shared.graph_block_mask(plan,device,window)
                assert actual.seq_lengths==expected.seq_lengths
                for key in ['kv_num_blocks','kv_indices','q_num_blocks','q_indices']:
                    assert torch.equal(getattr(actual,key),getattr(expected,key)),(keys,query,window,key)
                for a,b in zip(actual._natlang_bounds,expected._natlang_bounds):assert torch.equal(a,b)
    with torch.no_grad():
        first=shared.graph_block_mask(shared.AttentionLayout((129,),130),device)
        second=shared.graph_block_mask(shared.AttentionLayout((130,),131),device)
    assert first.kv_indices.data_ptr()==second.kv_indices.data_ptr()
    assert first._natlang_bounds[1].data_ptr()!=second._natlang_bounds[1].data_ptr()
    assert first._natlang_bounds[1].item()==129
    assert second._natlang_bounds[1].item()==130
    assert shared._decode_block_geometry.cache_info().currsize<=64


def _old_geometry(plan,device,window=None):
    with torch.enable_grad():return _ORIGINAL_MASK(plan,device,window)


_ORIGINAL_MASK=shared.graph_block_mask


@pytest.mark.parametrize('family',['lfm','mellum','qwen'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_decode_imported_native_state_and_producer_capture_match_prior_path(family,device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model=shared.SharedStateModel(tiny(family).to(device),(58,59,60,61),rank=4).eval()
    torch.nn.init.normal_(model.adapter.up.weight,std=.03)
    identity=dict(checkpoint_sha256='a'*64,tokenizer_fingerprint='b'*64,embedding_weight_sha256='c'*64)
    def run(old):
        ids=lambda values:torch.tensor([values],device=device)
        ordinary=lambda values:torch.zeros((1,len(values)),device=device,dtype=torch.bool) if old else False
        # Private producer prefix is NOT imported, only its actual exported span.
        private=shared.native_step(model,ids([11,12,13]))
        produced=shared.native_step(model,ids([2,3,4]),private['cache'],producing=True)
        block=SimpleNamespace(**{key:produced[key] for key in ['token_ids','embeddings','positions','layers']},
            **identity,architecture=model.port.config.model_type,validate=lambda:None)
        consumer=shared.native_step(model,ids([7,8]),producing=ordinary([7,8]))
        cache=shared.splice_block(model,consumer['cache'],block,**identity)
        results=[]
        for token in [5,6,9,10]:
            result=shared.native_step(model,ids([token]),cache,producing=ordinary([token]))
            cache=result['cache'];results.append(result)
        return produced,results
    with torch.inference_mode():
        actual=run(False)
        monkeypatch.setattr(shared,'graph_block_mask',_old_geometry)
        expected=run(True)
    for left,right in zip([actual[0],*actual[1]],[expected[0],*expected[1]]):
        assert torch.equal(left['h_final'],right['h_final'])
        assert left['cache'].lengths==right['cache'].lengths
        for l,r in zip(left['layers'],right['layers']):
            for key in l:
                if key!='kind':assert torch.equal(l[key],r[key]),(family,key)
        for l,r in zip(left['cache'].states,right['cache'].states):
            if hasattr(l,'window'):assert torch.equal(l.window,r.window)
            else:
                assert torch.equal(l.k,r.k)
                assert torch.equal(l.v,r.v)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_ordinary_training_keeps_explicit_zero_adapter_gradients(family):
    model=shared.SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    result=shared.native_step(model,torch.tensor([[3,4]]),producing=False)
    result['h_final'].square().sum().backward()
    for parameter in model.adapter.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad)==0



@pytest.mark.parametrize('family',['lfm','mellum'])
def test_unadapted_inference_skips_parameter_discovery_but_gradient_execution_keeps_it(family,monkeypatch):
    import threading
    from natlang_neuralese.model import native_adapters
    model=shared.SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    identity={'checkpoint_sha256':'a'*64}
    engine=SimpleNamespace(shared_model=model,native_identity=identity,native_execution_lock=threading.RLock())
    original=native_adapters.target_parameters
    visits=[]
    def parameters(value):visits.append(value);return original(value)
    monkeypatch.setattr(native_adapters,'target_parameters',parameters)
    def execute():
        assert native_adapters.execution_identity(engine)==identity
        return shared.native_step(model,torch.tensor([[3,4]]))['h_final']
    with torch.inference_mode():actual=native_adapters.run_native(engine,execute)
    assert not visits
    expected=native_adapters.run_native(engine,execute)
    assert visits==[model]
    assert torch.equal(actual,expected.detach())
    expected.square().sum().backward()
    assert model.adapter.up.weight.grad is not None
