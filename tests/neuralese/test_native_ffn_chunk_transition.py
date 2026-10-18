"""FFN execution uses the complete shared numerical transition verifier."""
import copy
import pytest
import torch
from natlang_neuralese.eval import native_attention_transition as transition
from test_native_attention_transition import fixture,Tokenizer,equal,inherited_cached_gap


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('chunk',[2,32])
def test_actual_ffn_boundary_complete_graph_cached_derivatives_preserve_real_parent(family,chunk):
    model,optimizer,named,roles,config=fixture(family)
    weights={name:p.detach().clone()for name,p in named};state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone();had=hasattr(model.port,'ffn_chunk_tokens');old=getattr(model.port,'ffn_chunk_tokens',None)
    report=transition.qualify_ffn_chunk_transition(model,Tokenizer(),roles,config,optimizer=optimizer,
        previous_chunk_tokens=0,candidate_chunk_tokens=chunk,checkpoint_sha256='a'*64,
        forward_atol=1e-4,gradient_atol=1e-5)
    assert report['schema']=='natlang.native-ffn-chunk-transition-complete/1'
    assert report['execution_setting']=='ffn_chunk_tokens'and report['transition_equivalence']['passed']
    assert report['cross_mode_passed']and report['source_state_preserved']and not report['readiness_qualified']
    assert len(report['case']['invocations'][0]['token_ids'])>chunk
    assert set(report['cross_mode']['graph']['comparisons'])=={'input_embeddings',*(name for name,_ in named)}
    assert hasattr(model.port,'ffn_chunk_tokens')==had
    if had:assert model.port.ffn_chunk_tokens==old
    equal(state,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())
    for name,p in named:assert torch.equal(p,weights[name])


def test_inherited_cached_debt_stays_failed_with_honest_ffn_equivalence():
    model,optimizer,named,roles,config=fixture('lfm');inherited_cached_gap(model)
    report=transition.qualify_ffn_chunk_transition(model,Tokenizer(),roles,config,optimizer=optimizer,
        previous_chunk_tokens=0,candidate_chunk_tokens=2,checkpoint_sha256='a'*64,
        forward_atol=1e-4,gradient_atol=1e-5)
    assert not report['passed']and not report['cached_reference_passed']
    assert report['transition_equivalence']['passed']and not report['readiness_qualified']


@pytest.mark.parametrize('value',[-1,True,1.5,'4'])
def test_invalid_ffn_transition_count_rejects_without_state_change(value):
    model,optimizer,named,roles,config=fixture('lfm');before=copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError,match='nonnegative'):
        transition.qualify_ffn_chunk_transition(model,Tokenizer(),roles,config,optimizer=optimizer,
            previous_chunk_tokens=0,candidate_chunk_tokens=value,checkpoint_sha256='a'*64,
            forward_atol=1e-4,gradient_atol=1e-5)
    equal(before,optimizer.state_dict())


def test_equivalence_rejects_unexercised_ffn_boundary():
    model,opt,_,roles,config=fixture('lfm')
    report=transition.qualify_ffn_chunk_transition(model,Tokenizer(),roles,config,optimizer=opt,
        previous_chunk_tokens=0,candidate_chunk_tokens=2,checkpoint_sha256='a'*64,
        forward_atol=1e-4,gradient_atol=1e-5)
    report['candidate_mode']=4096
    verdict=transition.transition_equivalence(report)
    assert not verdict['passed']and 'FFN reference did not cross the selected chunk boundary'in verdict['failures']
