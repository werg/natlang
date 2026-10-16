"""Per-import constants preserve native values and independent live uses."""
import os
import pytest
import torch

from test_shared_state import tiny
from natlang_neuralese.model.native_graph_plan import Invocation,ImportSpan,compile_graph,invocation_dict
from natlang_neuralese.model.shared_state import forward_graph,forward_reference
from natlang_neuralese.common.native_higher_order import reference_arithmetic


def stopped_plan(stop=True):
    return compile_graph([
        Invocation('source',(2,3,4,5),producing=((1,4),)),
        Invocation('constant',(6,7,8),imports=(ImportSpan(1,'source',1,4,stop_gradient=stop),),
                   logical_producing=((1,4),)),
        Invocation('live',(9,10,11),imports=(ImportSpan(1,'source',1,4),)),
        Invocation('relay',(12,13),imports=(ImportSpan(0,'constant',1,4,logical=True),))])


def test_stop_flag_schema_legacy_identity_and_transitive_logical_aliases():
    legacy=stopped_plan(False)
    assert legacy.slot_stops==()
    assert all('stop_gradient' not in edge for node in legacy.invocations for edge in invocation_dict(node)['imports'])
    plan=stopped_plan()
    flags=lambda index:plan.slot_stops[plan.slot_offsets[index]:plan.slot_offsets[index]+len(plan.sequences[index])]
    assert flags(0)==(False,)*4
    assert flags(1)==(False,True,True,True,False,False)
    assert flags(2)==(False,)*6
    assert flags(3)==(True,True,True,False,False)
    assert invocation_dict(plan.invocations[1])['imports'][0]['stop_gradient'] is True
    with pytest.raises(ValueError,match='explicit boolean'):
        compile_graph([Invocation('bad',(1,),imports=(ImportSpan(0,'external',0,1,stop_gradient=1),))],external_sizes={'external':1})


def test_packing_and_semantic_rewrite_preserve_explicit_constants():
    from natlang_neuralese.data.native_graph_schema import NativeGraphExample
    from natlang_neuralese.train.native_state_sft import pack_native_graphs
    from natlang_neuralese.train.native_semantic import consumer_example
    plan=stopped_plan()
    example=NativeGraphExample('stop','train',('stop',),'fixture',plan,(0.,)*plan.token_count)
    packed,*_=pack_native_graphs([example,example])
    assert sum(packed.slot_stops)==2*sum(plan.slot_stops)
    assert sum(edge.stop_gradient for node in packed.invocations for edge in node.imports)==2
    adapted,wire=consumer_example(example,{'export':{'node':'source','start':1,'stop':4},
        'consumer_nodes':['constant']},3)
    edge=adapted.plan.invocations[0].imports[0]
    assert edge.source==wire and edge.stop_gradient
    assert sum(adapted.plan.slot_stops)==3


def test_existing_row_and_source_hash_bytes_unchanged_without_explicit_stop():
    import hashlib,json
    from natlang_neuralese.data.native_graph_schema import NativeGraphExample
    from natlang_neuralese.data.native_sources import _row_bindings
    from natlang_neuralese.train.native_state_sft import _source_hash
    plan=stopped_plan(False)
    example=NativeGraphExample('legacy','train',('protected-source',),'fixture',plan,(0.,)*plan.token_count)
    # Frozen pre-stop serialization contract: notably logical=False WAS present,
    # while stop_gradient did not exist. Verify exact bytes, not approximate IDs.
    nodes=[{'name':node.name,'token_ids':node.token_ids,
        'imports':[{'at':edge.at,'source':edge.source,'start':edge.start,'stop':edge.stop,
                    'logical':edge.logical} for edge in node.imports],
        'producing':node.producing,'logical_producing':node.logical_producing} for node in plan.invocations]
    old={'id':example.id,'groups':example.source_groups,'split':example.split,
         'tokenizer':example.tokenizer_fingerprint,'targets':example.target_weights,'nodes':nodes}
    encoded=json.dumps(old,sort_keys=True,separators=(',',':')).encode()
    assert _source_hash([example])==hashlib.sha256(len(encoded).to_bytes(8,'big')+encoded).hexdigest()
    old.update(external_names=(),external_sizes=(),external_states={})
    assert _row_bindings([example])[example.id]==hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    flagged=NativeGraphExample(example.id,example.split,example.source_groups,
        example.tokenizer_fingerprint,stopped_plan(),example.target_weights)
    assert _source_hash([flagged])!=_source_hash([example])
    assert _row_bindings([flagged])!=_row_bindings([example])


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('checkpointed',[False,True])
def test_actual_native_stop_one_use_preserves_values_and_all_other_adjoints(family,checkpointed):
    device=os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu')
    port=tiny(family).to(device)
    port.checkpoint_layers=checkpointed
    plan=stopped_plan()
    inputs=torch.randn(1,plan.token_count,port.embedding_weight.shape[1],device=device,requires_grad=True)
    weight=torch.randn_like(inputs)
    actual=forward_graph(port,plan,embeddings=inputs)['h_final']
    unchanged=forward_graph(port,stopped_plan(False),embeddings=inputs)['h_final']
    assert torch.equal(actual,unchanged), 'stop is adjoint semantics only, never different native forward values'
    outputs={node.name:slice(offset,offset+len(node.token_ids)) for node,offset in zip(plan.invocations,plan.offsets)}
    for name in ('constant','relay'):
        current=forward_graph(port,plan,embeddings=inputs)['h_final']
        gradient=torch.autograd.grad((current[:,outputs[name]]*weight[:,outputs[name]]).sum(),inputs)[0]
        assert torch.count_nonzero(gradient[:,outputs['source']])==0
        assert gradient[:,outputs[name]].norm()>0
    current=forward_graph(port,plan,embeddings=inputs)['h_final']
    live=torch.autograd.grad((current[:,outputs['live']]*weight[:,outputs['live']]).sum(),inputs)[0]
    assert live[:,outputs['source']].norm()>0
    parameters=(inputs,*(parameter for parameter in port.parameters() if parameter.requires_grad))
    current=forward_graph(port,plan,embeddings=inputs)['h_final']
    got=torch.autograd.grad((current*weight).sum(),parameters,allow_unused=True)
    reference=forward_reference(port,plan,embeddings=inputs)['h_final']
    expected=torch.autograd.grad((reference*weight).sum(),parameters,allow_unused=True)
    torch.testing.assert_close(actual,reference,rtol=2e-5,atol=3e-6)
    for a,b in zip(got,expected):
        assert (a is None)==(b is None)
        if a is not None:
            torch.testing.assert_close(a,b,rtol=2e-4,atol=1e-5)
            if b.double().norm()>1e-5:assert (a-b).double().norm()/b.double().norm()<2e-4


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_reference_hessian_keeps_constant_import_source_coordinates_zero(family):
    device=os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu')
    port=tiny(family).to(device);port.checkpoint_layers=True
    plan=stopped_plan()
    inputs=torch.randn(1,plan.token_count,port.embedding_weight.shape[1],device=device,requires_grad=True)
    weight=torch.randn(1,2,inputs.shape[-1],device=device)
    with reference_arithmetic(query_chunk_tokens=2):
        output=forward_graph(port,plan,embeddings=inputs)['h_final'][:,-2:]
        loss=(output*weight).square().sum()
        first=torch.autograd.grad(loss,inputs,create_graph=True)[0]
        second=torch.autograd.grad(first.square().sum(),inputs)[0]
    assert torch.count_nonzero(first[:,:4])==0
    assert torch.count_nonzero(second[:,:4])==0
    assert second[:,-2:].norm()>0 and torch.isfinite(second).all()
