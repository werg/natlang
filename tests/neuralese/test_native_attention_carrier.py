"""Exact storage arithmetic and partition-independent KV cotangent carrier."""
import pytest
import torch
from natlang_neuralese.model.shared_state import graph_attention, AttentionLayout


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native sparse CUDA kernel')
def test_attention_carrier_preserves_forward_and_unrounded_kv_gradient_sum():
    torch.manual_seed(1701)
    query=torch.randn(1,2,2,16,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    key=torch.randn(1,1,5,16,device='cuda',dtype=torch.bfloat16).float().requires_grad_(True)
    value=torch.randn_like(key).to(torch.bfloat16).float().requires_grad_(True)
    gradient=torch.randn_like(query)
    layout=AttentionLayout((3,4),5)
    plain=graph_attention(query,key.to(query.dtype),value.to(query.dtype),layout,None,.25)
    carried=graph_attention(query,key,value,layout,None,.25)
    assert torch.equal(plain,carried)
    dq,dk,dv=torch.autograd.grad(carried,(query,key,value),gradient,retain_graph=True)
    assert dk.dtype==dv.dtype==torch.float32
    baseline=torch.autograd.grad(plain,(query,key,value),gradient,retain_graph=True)
    assert torch.equal(dq,baseline[0])
    assert torch.equal(dk.to(torch.bfloat16).float(),baseline[1])
    assert torch.equal(dv.to(torch.bfloat16).float(),baseline[2])
    partials=[]
    for index in range(2):
        current=graph_attention(query[:,:,index:index+1],key,value,AttentionLayout((3+index,),5),None,.25)
        assert torch.equal(current,carried[:,:,index:index+1])
        partials.append(torch.autograd.grad(current,(key,value),gradient[:,:,index:index+1],retain_graph=True))
    for full,parts in zip((dk,dv),zip(*partials)):
        exact_sum=torch.stack([item.double() for item in parts]).sum(0)
        torch.testing.assert_close(full.double(),exact_sum,atol=2e-7,rtol=2e-5)
        assert (full.double()-exact_sum).norm()/exact_sum.norm()<2e-5


def test_checkpoint_actor_context_crosses_threads_and_expires_with_operation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from test_native_serving_adapters import engine_fixture,nonzero_adapter
    from natlang_neuralese.model.native_adapters import run_native,capture_native_execution,execution_identity,native_fp32_state_cotangents
    engine=engine_fixture(tmp_path,'lfm',service_template=True)
    adapter,_=nonzero_adapter(engine)
    def prepare():
        expected=dict(engine.execution_native_identity)
        callback=capture_native_execution(lambda:(execution_identity(engine),native_fp32_state_cotangents()))
        with ThreadPoolExecutor(max_workers=1) as pool:
            actual=pool.submit(callback).result()
        assert actual==(expected,True)
        assert engine.execution_native_identity==expected
        return callback
    callback=run_native(engine,prepare,[adapter.id],gradient_precision='fp32')
    with pytest.raises(ValueError,match='actor scope expired'):
        callback()
    assert engine.execution_native_identity==engine.native_identity
    engine.store.close()


@pytest.mark.parametrize('mutation',['order','dtype','destination','shape','conditional','nested-extra'])
def test_native_kv_allocation_signature_fails_closed_on_same_count_changes(mutation):
    import ast,inspect,textwrap
    from natlang_neuralese.common.native_attention_delta import verify_native_kv_gradient_allocations
    from torch._inductor.kernel.flex.flex_attention import flex_attention_backward
    source=inspect.getsource(inspect.unwrap(flex_attention_backward))
    verify_native_kv_gradient_allocations(source)
    tree=ast.parse(textwrap.dedent(source));function=tree.body[0]
    allocations=[node for node in function.body if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                 and isinstance(node.value.func,ast.Name) and node.value.func.id in {'FixedLayout','empty_strided'}]
    assert len(allocations)==3
    if mutation=='order':
        left,right=map(function.body.index,allocations[1:])
        function.body[left],function.body[right]=function.body[right],function.body[left]
    elif mutation=='dtype':
        next(option for option in allocations[2].value.keywords if option.arg=='dtype').value=ast.parse('query.get_dtype()',mode='eval').body
    elif mutation=='destination':allocations[1].targets[0].id='broadcasted_grad_value'
    elif mutation=='conditional':
        index=function.body.index(allocations[1])
        function.body[index]=ast.If(test=ast.Constant(False),body=[allocations[1]],orelse=[])
    elif mutation=='nested-extra':
        import copy
        function.body.append(ast.Expr(value=ast.Call(func=ast.Name(id='consume',ctx=ast.Load()),args=[copy.deepcopy(allocations[0].value)],keywords=[])))
    else:allocations[2].value.args[0]=ast.Name(id='query_size',ctx=ast.Load())
    with pytest.raises(RuntimeError,match='exact Flex dK/dQ/dV'):
        verify_native_kv_gradient_allocations(ast.unparse(ast.fix_missing_locations(tree)))


def test_nested_native_actor_precision_selects_the_actual_port(tmp_path):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.model.native_adapters import run_native,native_fp32_state_cotangents
    (tmp_path/'left').mkdir();(tmp_path/'right').mkdir()
    left=engine_fixture(tmp_path/'left','lfm');right=engine_fixture(tmp_path/'right','mellum')
    def nested():
        assert not native_fp32_state_cotangents(left.shared_model.port)
        assert native_fp32_state_cotangents(right.shared_model.port)
    run_native(left,lambda:run_native(right,nested,gradient_precision='fp32'),gradient_precision='storage')
    left.store.close();right.store.close()
