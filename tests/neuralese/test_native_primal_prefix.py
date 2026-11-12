"""Original occurrence VJPs while fixed native projections reuse primal rows."""
import copy
from contextlib import nullcontext
import pytest
import torch
from test_shared_state import tiny
from test_native_communication_prefix import episodes
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.model.native_gradient_partitions import primal_row_reuse
from natlang_neuralese.train.native_graph_loss import pack_native_graphs
from natlang_neuralese.train.native_prefix_packing import communication_primal_rows


def test_update_diagnostic_reports_exact_mismatches_and_bf16_adjacency():
    from diagnose_primal_update_equivalence import differences
    a=torch.tensor([1.,2.],dtype=torch.bfloat16)
    b=torch.tensor([1.,2.],dtype=torch.bfloat16)
    b[1]=torch.nextafter(b[1],torch.tensor(float('inf'),dtype=b.dtype))
    result=differences({'weight':a},{'weight':b})
    assert len(result)==1 and result[0]['count']==result[0]['adjacent']==1
    assert differences({'weight':a},{'weight':a.clone()})==[]


def test_primal_binding_is_exact_plan_scoped_and_validates_representatives():
    from natlang_neuralese.model.native_gradient_partitions import bound_primal_rows
    plan,_,_,_=pack_native_graphs(episodes())
    aliases=communication_primal_rows(episodes())
    other=copy.copy(plan)
    assert bound_primal_rows(plan) is None
    with primal_row_reuse(plan,aliases):
        assert bound_primal_rows(plan).rows==aliases
        assert bound_primal_rows(other) is None
    assert bound_primal_rows(plan) is None
    for invalid in [(0,),tuple([plan.token_count]*plan.token_count)]:
        with pytest.raises(ValueError,match='representatives'):
            with primal_row_reuse(plan,invalid):pass


@pytest.mark.parametrize('kind',['wrong-token','adapted','imported','wrong-position','wrong-chain'])
def test_primal_aliases_require_original_raw_causal_equivalence(kind):
    from dataclasses import replace
    from natlang_neuralese.model.native_graph_plan import Invocation,ImportSpan,compile_graph
    if kind=='adapted':
        plan=compile_graph([Invocation('a',(2,3),producing=((1,2),)),
            Invocation('b',(2,3),producing=((1,2),))])
        aliases=(0,1,0,1)
    elif kind=='imported':
        plan=compile_graph([Invocation('a',(2,3),producing=((1,2),)),
            Invocation('b',(2,3),imports=(ImportSpan(1,'a',1,2),))])
        aliases=(0,1,0,1)
    elif kind=='wrong-position':
        plan=compile_graph([Invocation('a',(2,3)),Invocation('b',(2,3))])
        plan=replace(plan,own_positions=(0,1,1,2))
        aliases=(0,1,0,1)
    elif kind=='wrong-chain':
        plan=compile_graph([Invocation('a',(2,3)),Invocation('b',(4,3))])
        aliases=(0,1,2,1)
    else:
        plan=compile_graph([Invocation('a',(2,3)),Invocation('b',(2,4))])
        aliases=(0,0,0,0)
    with pytest.raises(ValueError,match='primal representatives'):
        with primal_row_reuse(plan,aliases):pass
