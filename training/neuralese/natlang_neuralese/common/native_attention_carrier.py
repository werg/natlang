"""Exact BF16 attention forward with FP32 shared KV cotangent outputs.

Only explicit FP32 gradient scopes use these carriers. The native sparse
forward and derivative arithmetic remain shared with ordinary graph attention;
the marked lowering prevents a BF16 store before cached fan-in.
"""
import torch


def _identity_joint(score, batch, head, query, key, gradient):
    return [gradient, None, None, None, None]


from torch.nn.attention.flex_attention import _identity
from ..model.shared_state import native_causal_mask
_GRAPHS = {name: torch.fx.symbolic_trace(function) for name,function in
           [('forward',_identity),('joint',_identity_joint),('mask',native_causal_mask)]}


@torch.compiler.allow_in_graph
def _native_backward_hop(query,key,value,output,lse,gradient,mask_parts,starts,ends,query_count,scale,options):
    # Dynamo cannot directly represent reusable FX score-graph arguments to
    # its backward HOP. This pure wrapper leaves all tensor inputs explicit;
    # AOTAutograd traces through it into the SAME native Flex backward HOP and
    # Inductor applies the verified sparse lowering. No eager fallback occurs.
    from torch._higher_order_ops.flex_attention import flex_attention_backward
    dq,dk,dv,_=flex_attention_backward(query,key,value,output,lse,gradient,None,
        _GRAPHS['forward'],_GRAPHS['joint'],(*mask_parts,_GRAPHS['mask']),scale,options,(),(starts,ends,query_count))
    return dq,dk.float(),dv.float()


def _backward(query,key,value,output,lse,gradient,mask_parts,starts,ends,query_count,scale,options):
    for tensor in (query,key,value):
        torch._dynamo.mark_static(tensor,-3)
        torch._dynamo.mark_static(tensor,-1)
    return _native_backward_hop(query,key,value,output,lse,gradient,mask_parts,starts,ends,query_count,scale,options)


_backward_kernels = {}


def _compiled_backward(query,key,value,output,gradient,scale,options):
    from ..model.shared_state import native_attention_layout,compile_native_attention_contract
    tensors=(query,key,value,output,gradient)
    contract=(tuple((tensor.dtype,native_attention_layout(tensor),tensor.requires_grad,tensor.is_inference()) for tensor in tensors),
        query.device,query.shape[0],query.shape[1],key.shape[1],query.shape[-1],value.shape[-1],
        torch.is_grad_enabled(),torch.is_inference_mode_enabled(),scale,tuple(sorted(options.items())))
    return compile_native_attention_contract(_backward_kernels,_backward,contract)


class _KVCarrierAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, block_mask, scale, options):
        from ..model.shared_state import _compiled_flex, native_flex_options
        if query.dtype != torch.bfloat16 or key.dtype != torch.float32 or value.dtype != torch.float32:
            raise ValueError('native attention KV carriers require BF16 query and FP32 key/value carriers')
        key, value = key.to(query.dtype), value.to(query.dtype)
        options = native_flex_options({**options, 'NATLANG_FP32_KV_GRADS': True})
        output, lse = _compiled_flex(query,key,value,scale)(query,key,value,
            block_mask=block_mask,scale=scale,kernel_options=options)
        ctx.save_for_backward(query,key,value,output,lse)
        ctx.block_mask,ctx.scale,ctx.options=block_mask,scale,options
        return output

    @staticmethod
    def backward(ctx, gradient):
        query,key,value,output,lse=ctx.saved_tensors
        if not hasattr(ctx.block_mask, '_natlang_bounds'):
            raise ValueError('FP32 KV cotangents require the exact native causal mask contract')
        starts,ends=ctx.block_mask._natlang_bounds
        kernel=_compiled_backward(query,key,value,output,gradient,ctx.scale,ctx.options)
        dq,dk,dv=kernel(query,key,value,output,lse,gradient,
            ctx.block_mask.as_tuple()[:-1],starts,ends,torch.tensor(starts.shape[0],device=starts.device),ctx.scale,ctx.options)
        return dq,dk,dv,None,None,None


def kv_carrier_attention(query,key,value,*,block_mask,scale,kernel_options):
    return _KVCarrierAttention.apply(query,key,value,block_mask,scale,kernel_options)


class _CPUKVCarrierAttention(torch.autograd.Function):
    """Host SDPA storage values with FP32 cotangents before shared fan-in."""
    @staticmethod
    def forward(ctx, query, key, value, mask, scale):
        additive = torch.zeros_like(mask,dtype=query.dtype).masked_fill(~mask,float('-inf')) if mask.dtype == torch.bool else mask
        output,lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            query,key.to(query.dtype),value.to(query.dtype),0.,False,attn_mask=additive,scale=scale)
        ctx.save_for_backward(query,key,value,additive,output,lse)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, gradient):
        query,key,value,mask,output,lse=ctx.saved_tensors
        dq,_,_=torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward.default(
            gradient,query,key.to(query.dtype),value.to(query.dtype),output,lse,0.,False,attn_mask=mask,scale=ctx.scale)
        repeat=query.shape[1]//key.shape[1]
        k=key.repeat_interleave(repeat,1);v=value.repeat_interleave(repeat,1)
        scores=(query.float() @ k.transpose(-2,-1))*ctx.scale+mask
        probability=torch.exp(scores-lse[...,None])
        delta=(gradient.float()*output.float()).sum(-1,keepdim=True)
        grad_probability=gradient.float() @ v.transpose(-2,-1)
        grad_score=(probability*(grad_probability-delta)).to(query.dtype).float()
        dk=(grad_score.transpose(-2,-1) @ query.float())*ctx.scale
        dv=probability.to(query.dtype).float().transpose(-2,-1) @ gradient.float()
        shape=(key.shape[0],key.shape[1],repeat,key.shape[2],key.shape[3])
        return dq,dk.reshape(shape).sum(2),dv.reshape(shape).sum(2),None,None


def cpu_kv_carrier_attention(query, key, value, *, mask, scale):
    if query.is_cuda or query.dtype != torch.bfloat16 or key.dtype != value.dtype or key.dtype != torch.float32:
        raise ValueError('host KV carriers require CPU BF16 query and FP32 key/value carriers')
    return _CPUKVCarrierAttention.apply(query, key, value, mask, scale)
