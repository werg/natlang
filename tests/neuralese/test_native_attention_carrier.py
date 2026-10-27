"""Exact storage arithmetic and partition-independent KV cotangent carrier."""
import pytest
import torch
from natlang_neuralese.model.shared_state import graph_attention, AttentionLayout


def test_cpu_cached_attention_preserves_empty_query_shape():
    q = torch.empty(1, 4, 0, 16)
    k = torch.randn(1, 2, 3, 16)
    result = graph_attention(q, k, k, AttentionLayout((), 3), 8, .25)
    assert result.shape == q.shape


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_cpu_windowed_single_query_matches_graph_and_crops_cached_keys(dtype, monkeypatch):
    from natlang_neuralese.model.shared_state import Invocation, compile_graph
    import torch.nn.functional as functional
    torch.manual_seed(51)
    plan=compile_graph([Invocation('reader',tuple(range(11)))])
    q=torch.randn(1,4,11,16,dtype=dtype,requires_grad=True)
    k=torch.randn(1,2,11,16,dtype=dtype,requires_grad=True)
    v=torch.randn(1,2,11,16,dtype=dtype,requires_grad=True)
    lengths=[]
    original=functional.scaled_dot_product_attention
    from natlang_neuralese.common import native_attention_carrier as carrier
    original_rows=carrier.cpu_flash_rows
    def record(query,key,value,**kwargs):
        lengths.append(key.shape[-2])
        return original(query,key,value,**kwargs)
    monkeypatch.setattr(functional,'scaled_dot_product_attention',record)
    def record_rows(query,key,value,**kwargs):
        lengths.extend(end-start for start,end in zip(kwargs['starts'],kwargs['ends']))
        return original_rows(query,key,value,**kwargs)
    monkeypatch.setattr(carrier,'cpu_flash_rows',record_rows)
    graph=graph_attention(q,k,v,plan,8,.25)
    cached=graph_attention(q[:,:,-1:],k,v,AttentionLayout((10,),11),8,.25)
    assert lengths[-2:]==[8,8]
    assert torch.equal(graph[:,:,-1:],cached)
    graph_grad=torch.autograd.grad(graph[:,:,-1:].float().sum(),(q,k,v),retain_graph=True)
    cached_grad=torch.autograd.grad(cached.float().sum(),(q,k,v))
    for actual,expected in zip(graph_grad,cached_grad):
        assert torch.equal(actual,expected)


@pytest.mark.parametrize('window',[None,4])
def test_cpu_flash_rows_matches_independent_prefix_attention_gradients(window):
    from natlang_neuralese.common.native_attention_carrier import cpu_flash_rows
    from torch.nn.functional import scaled_dot_product_attention
    torch.manual_seed(1702)
    q=torch.randn(1,4,17,8,requires_grad=True)
    k=torch.randn(1,2,17,8,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    mask=torch.rand(1,1,17,17)>.2
    ends=list(range(1,18))
    starts=[max(0,end-window) if window is not None else 0 for end in ends]
    reference=torch.cat([scaled_dot_product_attention(q[:,:,r:r+1],k[:,:,a:b],v[:,:,a:b],
        attn_mask=mask[...,r:r+1,a:b],scale=.25,enable_gqa=True)
        for r,(a,b) in enumerate(zip(starts,ends))],dim=2)
    actual=cpu_flash_rows(q,k,v,mask=mask,starts=starts,ends=ends,scale=.25)
    gradient=torch.randn_like(actual)
    expected_grad=torch.autograd.grad(reference,(q,k,v),gradient,retain_graph=True)
    actual_grad=torch.autograd.grad(actual,(q,k,v),gradient)
    assert torch.equal(actual,reference)
    for got,expected in zip(actual_grad,expected_grad): assert torch.equal(got,expected)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_attention_carrier_preserves_forward_and_unrounded_kv_gradient_sum(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('native sparse CUDA kernel')
    torch.manual_seed(1701)
    query=torch.randn(1,2,2,16,device=device,dtype=torch.bfloat16,requires_grad=True)
    key=torch.randn(1,1,5,16,device=device,dtype=torch.bfloat16).float().requires_grad_(True)
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
    if device=='cuda':
        assert torch.equal(dk.to(torch.bfloat16).float(),baseline[1])
        assert torch.equal(dv.to(torch.bfloat16).float(),baseline[2])
    else:
        # The CPU path uses cached one-query geometry. Native Flash rounds
        # each query's shared-head result, then autograd adds those BF16
        # contributions. Compare against independent raw native SDPA calls;
        # the carrier must retain unrounded cotangents through both fan-ins.
        from torch.nn.functional import scaled_dot_product_attention
        rounded_key=key.detach().to(query.dtype).requires_grad_()
        rounded_value=value.detach().to(query.dtype).requires_grad_()
        accumulated=[torch.zeros_like(rounded_key),torch.zeros_like(rounded_value)]
        for row in reversed(range(query.shape[2])):
            end=layout.query_positions[row]+1
            one=scaled_dot_product_attention(query[:,:,row:row+1],rounded_key[:,:,:end],
                rounded_value[:,:,:end],scale=.25,enable_gqa=True)
            gradients=torch.autograd.grad(one,(rounded_key,rounded_value),gradient[:,:,row:row+1],retain_graph=True)
            accumulated=[old+new for old,new in zip(accumulated,gradients)]
        for actual,rounded in zip(baseline[1:],accumulated): assert torch.equal(actual,rounded.float())
        heads=[]
        for head in range(query.shape[1]):
            one=graph_attention(query[:,head:head+1],key,value,layout,None,.25)
            gradients=torch.autograd.grad(one,(key,value),gradient[:,head:head+1],retain_graph=True)
            heads.append(gradients)
        for actual,parts in zip((dk,dv),zip(*heads)):
            torch.testing.assert_close(actual.double(),torch.stack([part.double() for part in parts]).sum(0),atol=2e-7,rtol=2e-5)
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


def test_cpu_flash_rows_honors_math_backend(monkeypatch):
    from natlang_neuralese.common.native_attention_carrier import cpu_flash_rows
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention
    torch.manual_seed(1703)
    q=torch.randn(1,4,5,8,requires_grad=True)
    k=torch.randn(1,2,5,8,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    mask=torch.ones(1,1,5,5,dtype=torch.bool)
    starts=[0]*5; ends=list(range(1,6))
    with sdpa_kernel(SDPBackend.MATH):
        reference=torch.cat([scaled_dot_product_attention(q[:,:,r:r+1],k[:,:,:end],v[:,:,:end],
            attn_mask=mask[...,r:r+1,:end],scale=.25,enable_gqa=True)
            for r,end in enumerate(ends)],dim=2)
        actual=cpu_flash_rows(q,k,v,mask=mask,starts=starts,ends=ends,scale=.25)
        gradient=torch.randn_like(actual)
        expected=torch.autograd.grad(reference,(q,k,v),gradient,retain_graph=True)
        got=torch.autograd.grad(actual,(q,k,v),gradient)
    assert torch.equal(actual,reference)
    for a,b in zip(got,expected):assert torch.equal(a,b)
