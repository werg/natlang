import json
import pytest
import torch
from scripts.neuralese_data.qualify_native_causal_prefixes import select_adjacent,gradient_reference,metrics,gradient_metrics
from natlang_neuralese.data.native_graph import SCHEMA


def test_selects_complete_adjacent_train_graphs_and_preserves_loss_weights(tmp_path):
    source=tmp_path/'train.jsonl'
    rows=[{'schema':SCHEMA,'id':str(i),'split':'train','source_groups':['train'],
        'tokenizer_fingerprint':'a'*64,'nodes':[{'name':'answer','token_ids':ids,
        'target_weights':[0]+[.05]*(len(ids)-2)+[1]}]}
        for i,ids in enumerate([[2,3,4,5],[2,3,4,6],[8,9,10]])]
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    items,plan,maps=select_adjacent(source,scan=3,max_tokens=8)
    assert [item[1].id for item in items]==['0','1']
    assert [item[2] for item in items]==[0,1]
    assert plan.token_count==5 and maps==((0,1,2,3),(0,1,2,4))
    assert [item[1].target_weights for item in items]==[(0,.05,.05,1)]*2
    rows[0]['split']='test'
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError,match='only admitted train'):select_adjacent(source)


def test_gradient_reference_sums_every_original_use_in_fp32():
    disjoint=torch.tensor([[[1.,2.],[3.,4.],[5.,6.],[7.,8.]]],dtype=torch.bfloat16)
    result=gradient_reference(disjoint,((0,1),(0,2)))
    assert result.dtype==torch.float32
    torch.testing.assert_close(result,torch.tensor([[[6.,8.],[3.,4.],[7.,8.]]]))
    assert metrics(result,result,atol=0,rtol=0)['passed']
    failed=metrics(result+1,result,atol=0,rtol=0)
    assert failed['passed'] is False and failed['mismatched_elements']==6


def test_tiny_gradient_does_not_qualify_only_because_absolute_error_is_small():
    expected=torch.tensor([1e-12,2e-12])
    wrong=gradient_metrics(-expected,expected,atol=1e-5,rtol=1e-4)
    assert wrong['elementwise_passed'] and not wrong['passed']
    assert wrong['relative_l2']==2 and wrong['cosine']<-.999
    assert gradient_metrics(expected,expected,atol=1e-5,rtol=1e-4)['passed']


def test_sparse_visibility_has_one_chronological_accumulation_order():
    from natlang_neuralese.model.shared_state import Invocation,compile_graph,graph_block_mask,query_workspace
    plan=compile_graph([Invocation('first',tuple(range(257))),Invocation('second',tuple(range(129)))])
    workspace,physical=query_workspace(plan)
    mask=graph_block_mask(plan,'cpu')
    assert mask.full_kv_num_blocks is None and mask.full_kv_indices is None
    for query_block,(count,indices) in enumerate(zip(mask.kv_num_blocks[0,0],mask.kv_indices[0,0])):
        keys=indices[:int(count)].tolist()
        assert keys==sorted(set(keys))
    # Every real query retains its original causal visibility. Dummy-query
    # geometry cannot reorder earlier full tiles ahead of the last partial one.
    for row,query in enumerate(physical):
        for key in range(len(plan.slot_sources)):
            allowed=bool(mask.mask_mod(torch.tensor(0),torch.tensor(0),torch.tensor(query),torch.tensor(key)))
            assert allowed==(workspace.starts[query]<=key<=workspace.ends[query])
        query_block=query//128
        blocks=set(mask.kv_indices[0,0,query_block,:int(mask.kv_num_blocks[0,0,query_block])].tolist())
        assert all(key//128 in blocks for key in plan.allowed[row])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_native_attention_prefix_cut_keeps_outputs_and_consumer_source_adjoint():
    from types import SimpleNamespace
    from natlang_neuralese.model.shared_state import Invocation,compile_graph,graph_attention
    from natlang_neuralese.train.native_prefix_packing import pack_causal_prefixes
    from natlang_neuralese.train.native_state_sft import pack_native_graphs
    torch.manual_seed(117)
    prefix=tuple(range(257))
    rows=[SimpleNamespace(id=str(i),plan=compile_graph([Invocation('answer',prefix+tuple(range(300+i*100,300+i*100+tail)))]),
        target_weights=(0,)*(257+tail)) for i,tail in enumerate((31,48))]
    shared,maps=pack_causal_prefixes(rows);disjoint=pack_native_graphs(rows)[0]
    flat=torch.tensor([r for mapping in maps for r in mapping],device='cuda')
    leaves=[(torch.randn(1,heads,shared.token_count,128,device='cuda',dtype=torch.bfloat16)*scale).to(dtype).requires_grad_()
        for heads,scale,dtype in ((8,3,torch.bfloat16),(1,1,torch.float32),(1,4,torch.float32))]
    original=[value.detach().index_select(2,flat).requires_grad_() for value in leaves]
    def execute(values,plan):
        q,k,v=values;slots=plan.tensor(plan.slot_sources,'cuda')
        return graph_attention(q,k[:,:,slots].to(q.dtype),v[:,:,slots].to(q.dtype),plan,None,128**-.5)
    actual=execute(leaves,shared);expected=execute(original,disjoint)
    assert torch.equal(actual.index_select(2,flat),expected)
    # A real consumer query backpropagates to its imported prefix. Isolating
    # one query avoids introducing unrelated BF16 loss fan-in rounding here.
    upstream=torch.randn_like(expected[:,:,-1])*.001
    ga=torch.autograd.grad((actual[:,:,maps[-1][-1]]*upstream).sum(),leaves)
    gb=torch.autograd.grad((expected[:,:,-1]*upstream).sum(),original)
    for actual_gradient,original_gradient in zip(ga,gb):
        summed=torch.zeros_like(actual_gradient,dtype=torch.float32)
        summed.index_add_(2,flat,original_gradient.float())
        assert gradient_metrics(actual_gradient,summed,atol=1e-7,rtol=1e-4)['passed']
