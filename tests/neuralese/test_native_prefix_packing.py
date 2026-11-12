"""Exact cached-prefix layout and complete ordinary SFT gradient preservation."""
from natlang_neuralese.data.native_graph_schema import NativeGraphExample
import copy
import pytest
import torch
from natlang_neuralese.model.shared_state import Invocation, ImportSpan, compile_graph, SharedStateModel
from natlang_neuralese.train.native_prefix_packing import pack_causal_prefixes
from natlang_neuralese.train.native_graph_loss import packed_graph_loss, pack_native_graphs
from test_shared_state import tiny


def examples():
    sequences = [(2,3,4,5,6,7),(2,3,4,5,8),(2,3,4,5),(2,3,4,5,6,7),(9,10,11)]
    return [NativeGraphExample(id=f'e{index}',split='train',source_groups=(f'e{index}',),
        tokenizer_fingerprint='test',plan=compile_graph([Invocation('answer',tokens)]),
        target_weights=tuple([0]+[(.05 if position%2 else 1)*(index+1)
                                  for position in range(1,len(tokens))]))
            for index,tokens in enumerate(sequences)]


def test_prefix_layout_retains_duplicates_and_internal_terminal_queries_without_adaptation():
    rows=examples();plan,maps=pack_causal_prefixes(rows)
    assert plan.token_count == 10
    assert not any(plan.producing)
    assert maps[0] == maps[3] and maps[2] == maps[0][:4]
    tokens=[token for node in plan.invocations for token in node.token_ids]
    for example,mapping in zip(rows,maps):
        assert tuple(tokens[index] for index in mapping) == example.plan.invocations[0].token_ids
        for logical,physical in enumerate(mapping):
            assert plan.own_positions[physical] == logical
            assert plan.history(physical) == mapping[:logical+1]


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('checkpoint_layers',[False,True])
def test_prefix_logits_loss_and_all_gradients_match_disjoint_packing(family,checkpoint_layers):
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    # The tiny Mellum loader freezes base weights; exercise real ordinary
    # derivatives rather than only sparse head/adapter parameters.
    for parameter in model.port.parameters():parameter.requires_grad_(True)
    reference=copy.deepcopy(model);rows=examples()
    shared,maps=pack_causal_prefixes(rows)
    disjoint,_,ownership,_=pack_native_graphs(rows)
    with torch.no_grad():
        actual=model(shared,retain_states=False)['h_final'][0]
        expected=reference(disjoint,retain_states=False)['h_final'][0]
        for mapping,(first,last) in zip(maps,ownership):
            start=disjoint.offsets[first];stop=disjoint.offsets[last-1]+len(disjoint.invocations[last-1].token_ids)
            torch.testing.assert_close(actual[list(mapping)],expected[start:stop],atol=2e-6,rtol=2e-5)
    loss,details=packed_graph_loss(model,rows,checkpoint_layers=checkpoint_layers,chunk_tokens=2,
                                  share_causal_prefixes=True)
    reference_loss,reference_details=packed_graph_loss(reference,rows,checkpoint_layers=checkpoint_layers,chunk_tokens=2)
    torch.testing.assert_close(loss,reference_loss,atol=1e-6,rtol=1e-6)
    assert [row['weight'] for row in details] == [row['weight'] for row in reference_details]
    assert [row['supervised_tokens'] for row in details] == [row['supervised_tokens'] for row in reference_details]
    loss.backward();reference_loss.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
               for parameter in model.port.parameters())
    for (name,parameter),(other_name,other) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other_name
        assert (parameter.grad is None)==(other.grad is None),name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad,other.grad,atol=3e-6,rtol=2e-4,msg=name)


def test_bf16_ordinary_sft_aliases_raw_prefixes_with_bounded_gradient_change():
    """Prefix arithmetic v4: BF16 ordinary SFT aliases exact raw prefixes physically.

    Shared rows receive the summed cotangent of all consumers, a deliberate
    rounding-order transition from per-occurrence BF16 cotangents. The scalar
    CE and supervision are unchanged; gradients agree within the transition bound.
    """
    from natlang_neuralese.train.native_prefix_packing import ordinary_sft_physical_sharing_enabled
    model=SharedStateModel(tiny('lfm').to(dtype=torch.bfloat16),(58,59,60,61),rank=4)
    assert ordinary_sft_physical_sharing_enabled(model, True)
    assert not ordinary_sft_physical_sharing_enabled(model, False)
    with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    for parameter in model.port.parameters():parameter.requires_grad_(True)
    reference=copy.deepcopy(model);rows=examples()
    shared,shared_details=packed_graph_loss(model,rows,share_causal_prefixes=True,
        checkpoint_layers=False,chunk_tokens=2)
    unshared,unshared_details=packed_graph_loss(reference,rows,share_causal_prefixes=False,
        checkpoint_layers=False,chunk_tokens=2)
    shared.backward();unshared.backward()
    assert shared_details[0]['packed_physical_tokens'] < shared_details[0]['original_physical_tokens']
    assert shared_details[0]['supervised_tokens'] == unshared_details[0]['supervised_tokens']
    assert shared_details[0]['readout_forward_rows'] == shared_details[0]['readout_original_rows']
    torch.testing.assert_close(shared.detach().float(),unshared.detach().float(),atol=2e-2,rtol=1e-2)
    actual=[];expected=[]
    for (name,parameter),(other_name,other) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other_name and (parameter.grad is None)==(other.grad is None),name
        if parameter.grad is not None:
            actual.append(parameter.grad.detach().double().flatten())
            expected.append(other.grad.detach().double().flatten())
    actual=torch.cat(actual);expected=torch.cat(expected)
    relative=float((actual-expected).norm()/expected.norm())
    cosine=float(actual@expected/(actual.norm()*expected.norm()))
    assert relative <= .1 and cosine >= .9997, (relative, cosine)


def test_authenticated_producing_or_imported_graphs_alias_only_initial_raw_context():
    rows=examples()
    rows[0].plan=compile_graph([Invocation('writer',(2,3,4),producing=((1,3),)),
        Invocation('reader',(5,6),imports=(ImportSpan(0,'writer',1,3),))])
    for selected in (rows, [*rows[1:],rows[0]]):
        plan,maps=pack_causal_prefixes(selected)
        for row,mapping in zip(selected,maps):
            for old,new in enumerate(mapping):
                assert plan.producing[new]==row.plan.producing[old]
                assert plan.history(new)==tuple(mapping[index]for index in row.plan.history(old))
    external=compile_graph([Invocation('reader',(5,6),imports=(ImportSpan(0,'external',0,2),))],
                           external_sizes={'external':2})
    rows[0].plan=external
    assert pack_causal_prefixes(rows) is None


@pytest.mark.parametrize('edge',[ImportSpan(0,'prefix',1,3,True),ImportSpan(1,'prefix',0,3,True),
                                 ImportSpan(0,'prefix',0,3,False)])
def test_raw_cache_requires_complete_positionally_identical_prefix(edge):
    with pytest.raises(ValueError,match='complete causal prefix'):
        compile_graph([Invocation('prefix',(2,3,4),cache_prefix=True),Invocation('tail',(5,),imports=(edge,))])


def test_raw_cache_export_never_activates_adapter_or_accepts_external_or_neuralese_parent():
    with pytest.raises(ValueError,match='unadapted'):
        compile_graph([Invocation('prefix',(2,3,4),producing=((0,3),),cache_prefix=True)])
    with pytest.raises(ValueError,match='raw complete prefix'):
        compile_graph([Invocation('prefix',(2,3,4),producing=((0,3),)),
            Invocation('tail',(5,),imports=(ImportSpan(0,'prefix',0,3),),cache_prefix=True)])
    with pytest.raises(ValueError,match='raw complete prefix'):
        compile_graph([Invocation('tail',(5,),imports=(ImportSpan(0,'external',0,3),),cache_prefix=True)],
                      external_sizes={'external':3})


def test_original_rl_graph_identity_does_not_gain_default_cache_field():
    from dataclasses import asdict
    from natlang_neuralese.model.native_graph_plan import invocation_dict
    node=Invocation('original',(2,3,4))
    old=asdict(node);old.pop('cache_prefix')
    assert invocation_dict(node)==old
    assert invocation_dict(Invocation('cached',(2,3),cache_prefix=True))['cache_prefix'] is True
