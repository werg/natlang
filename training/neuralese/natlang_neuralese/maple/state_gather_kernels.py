"""Bounded native state gather: FP32 histories and FP64 source adjoints."""
import torch
import triton
import triton.language as tl


# Source/history lengths change between graphs. Specializing them recompiles
# both kernels during useful training; only feature geometry stays constant.
@triton.jit(do_not_specialize=['S','R','SB','N'],
            do_not_specialize_on_alignment=['S','R','SB','N'])
def _gather(source, index, out, S, C:tl.constexpr, R,
            SB, SS:tl.constexpr, SC:tl.constexpr, N, BLOCK:tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0,BLOCK)
    valid = offsets < N
    row = (offsets//C)%R
    batch = offsets//(R*C)
    column = offsets%C
    selected = tl.load(index+row,mask=valid,other=0)
    selected = tl.where(selected<0,selected+S,selected)
    value = tl.load(source+batch*SB+selected*SS+column*SC,
        mask=valid&(selected>=0)&(selected<S),other=0.).to(tl.float32)
    tl.store(out+offsets,value,mask=valid)


@triton.jit(do_not_specialize=['S','R','N'],
            do_not_specialize_on_alignment=['S','R','N'])
def _accumulate(grad,index,out,S,C:tl.constexpr,R,
                N,BLOCK:tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0,BLOCK)
    valid = offsets<N
    row = (offsets//C)%R
    batch = offsets//(R*C)
    column = offsets%C
    selected = tl.load(index+row,mask=valid,other=0)
    selected = tl.where(selected<0,selected+S,selected)
    value = tl.load(grad+offsets,mask=valid,other=0.).to(tl.float64)
    tl.atomic_add(out+(batch*S+selected)*C+column,value,
        mask=valid&(selected>=0)&(selected<S),sem='relaxed')


def gather_values(source,index):
    batch,sources,channels=source.shape
    index=index.contiguous()
    result=torch.empty((batch,*index.shape,channels),device=source.device,dtype=torch.float32)
    if result.numel():
        _gather[(triton.cdiv(result.numel(),1024),)](source,index,result,sources,channels,index.numel(),
            *source.stride(),result.numel(),BLOCK=1024)
    return result


def accumulate_sources(grad,index,shape):
    batch,sources,channels=shape
    result=torch.zeros(shape,device=grad.device,dtype=torch.float64)
    if grad.numel():
        _accumulate[(triton.cdiv(grad.numel(),1024),)](grad.contiguous(),index.contiguous(),result,
            sources,channels,index.numel(),grad.numel(),BLOCK=1024)
    return result
