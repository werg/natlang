"""Explicit differentiable reference arithmetic; ordinary native kernels stay default.

This is the derivative of this declared mathematical execution, not a Hessian
of the quantized/fused first-order VJP. No detached fast result is substituted.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,field

import torch
import torch.nn.functional as F

SCHEMA='natlang.native-reference-arithmetic/1'
_ACTIVE=ContextVar('native_higher_order_arithmetic',default=None)


@dataclass
class ReferenceArithmetic:
    query_chunk_tokens:int=16
    counters:dict=field(default_factory=dict)
    live:bool=True
    lifetime:dict=field(default_factory=lambda:{'active':True})

    def binding(self):
        return {'schema':SCHEMA,'native_projection_product_dtype':'float64',
            'stock_torch_modules':'source-dtype/differentiable',
            'adapter_delta':'stored-base-plus-unrounded-float32',
            'optimizer_adam_zero_moment_tangent':'zero-v-boundary/finite-valid-history',
            'activation_storage':'source-native','attention':'exact-visible-softmax',
            'expert_dispatch':'source-top-k/differentiable-selected-weights',
            'query_chunk_tokens':self.query_chunk_tokens}

    def record(self,name):
        self.counters[name]=self.counters.get(name,0)+1


def active_reference_arithmetic():
    return _ACTIVE.get()


@contextmanager
def reference_arithmetic(model=None,*,query_chunk_tokens=16):
    if type(query_chunk_tokens) is not int or query_chunk_tokens<1:
        raise ValueError('native reference query chunk must be a positive integer')
    state=ReferenceArithmetic(query_chunk_tokens)
    missing=object()
    port=None if model is None else model.port
    old_checkpoint=missing if port is None else getattr(port,'checkpoint_layers',missing)
    token=_ACTIVE.set(state)
    try:
        yield state
    finally:
        state.live=False
        state.lifetime['active']=False
        _ACTIVE.reset(token)
        if port is not None:
            if old_checkpoint is missing:
                if hasattr(port,'checkpoint_layers'):delattr(port,'checkpoint_layers')
            else:port.checkpoint_layers=old_checkpoint


def capture_reference_execution(callback):
    state=active_reference_arithmetic()
    if state is None:return callback
    def execute(*args,**kwargs):
        if not state.live:
            raise RuntimeError('native reference arithmetic scope expired before checkpoint recomputation')
        token=_ACTIVE.set(state)
        try:return callback(*args,**kwargs)
        finally:_ACTIVE.reset(token)
    return execute


def reference_linear(x,weight,bias=None):
    """High precision products, ordinary differentiable casts to native storage."""
    state=active_reference_arithmetic()
    if state is None:raise RuntimeError('native reference linear needs its explicit arithmetic scope')
    state.record('linear')
    # A shared ordinary cast node preserves higher-order autograd while making
    # repeated low-precision parameter uses add in FP32 before the leaf cast.
    # The production carrier uses a first-order file-backed edge; reference
    # arithmetic must remain differentiable for nested derivative programs.
    from ..model.native_weight_carrier import active_cotangent_scope
    scope=active_cotangent_scope()
    if scope is not None:
        weight=scope.reference_parameter(weight)
        if bias is not None:bias=scope.reference_parameter(bias)
    return F.linear(x.double(),weight.double(),None if bias is None else bias.double()).to(x.dtype)


def reference_attention(q,k,v,plan,window,scale):
    """Bounded query tiles with the existing exact logical visibility bounds."""
    state=active_reference_arithmetic()
    if state is None:raise RuntimeError('native reference attention needs its explicit arithmetic scope')
    state.record('attention')
    starts,ends=plan.bounds(q.device,window)
    if q.shape[-2]!=len(starts) or k.shape[-2]!=len(plan.slot_sources):
        raise ValueError('native reference attention geometry differs from its exact plan')
    if q.shape[1]%k.shape[1] or k.shape[1]!=v.shape[1]:
        raise ValueError('native reference attention requires exact grouped-query head ratios')
    repeat=q.shape[1]//k.shape[1]
    pieces=[]
    for at in range(0,q.shape[-2],state.query_chunk_tokens):
        stop=min(at+state.query_chunk_tokens,q.shape[-2])
        lo=int(starts[at:stop].min());hi=int(ends[at:stop].max())+1
        if lo<0 or hi>k.shape[-2] or hi<=lo:
            raise ValueError('native reference attention has invalid nonempty causal bounds')
        keys=torch.arange(lo,hi,device=q.device)
        mask=(keys[None]>=starts[at:stop,None]) & (keys[None]<=ends[at:stop,None])
        kk=k[:,:,lo:hi].repeat_interleave(repeat,1).double()
        vv=v[:,:,lo:hi].repeat_interleave(repeat,1).double()
        scores=(q[:,:,at:stop].double() @ kk.transpose(-2,-1))*scale
        probability=scores.masked_fill(~mask[None,None],float('-inf')).softmax(-1)
        pieces.append((probability @ vv).to(q.dtype))
    return torch.cat(pieces,dim=2) if pieces else q.new_empty((*q.shape[:-1],v.shape[-1]))


@contextmanager
def adam_zero_moment_root():
    """Native Adam values with the finite zero-history composite derivative.

    Along a valid zero Adam history, m=v=0 and dv=0: the update derivative
    reaches dm through epsilon. Raw sqrt's infinite derivative instead creates
    0*infinity NaNs in autograd. The exact nonnegative zero branch below leaves
    every forward value unchanged and has a declared zero v-boundary tangent.
    This mode is local to the upstream functional update, never a global patch.
    """
    from torch.overrides import TorchFunctionMode
    class MomentRoot(TorchFunctionMode):
        def __torch_function__(self,func,types,args=(),kwargs=None):
            if func in (torch.sqrt,torch.Tensor.sqrt):
                value=args[0]
                positive=value>0
                safe=torch.where(positive,value,torch.ones_like(value))
                return torch.where(positive,safe.sqrt(),torch.zeros_like(value))
            return func(*args,**(kwargs or {}))
    with MomentRoot():yield
