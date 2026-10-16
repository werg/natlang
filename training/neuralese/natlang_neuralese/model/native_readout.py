"""Complete immutable native readout bindings for recorded conditional scores."""
from __future__ import annotations
import torch
import contextvars
from contextlib import contextmanager
from .native_artifact import NativeTensorArtifact
from .native_adapters import _Operation

SCHEMA='natlang.native-readout-snapshot/1'
_CACHE=contextvars.ContextVar('native_readout_operation_cache',default=None)


@contextmanager
def readout_scope():
    """One immutable device copy per readout per derivative operation."""
    if _CACHE.get() is not None:
        yield
        return
    token=_CACHE.set({})
    try:yield
    finally:_CACHE.reset(token)


def readout_layout(model):
    targets=[model.port.output_weight,model.port.final_norm_weight,
             model.input_rows if model.tied else model.output_rows]
    if model.token_override_ids:
        targets.append(model.extra_input_rows if model.tied else model.extra_output_rows)
    available={**dict(model.named_parameters()),**dict(model.named_buffers())}
    named={}
    for target in targets:
        names=[name for name,value in available.items() if value is target]
        if not names:raise ValueError('native readout tensor lacks its exact registered model name')
        name=names[0];named[name]=target
    layout={'schema':SCHEMA,'architecture':model.port.config.model_type,'tied':model.tied,
            'marker_ids':list(model.marker_ids),'token_override_ids':list(model.token_override_ids),
            'norm_eps_hex':float(model.port.norm_eps).hex(),
            'inventory':{name:{'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in sorted(named.items())}}
    return named,layout


def snapshot_readout(engine, *, values=None, owner=None):
    """Capture actual values, or an explicitly supplied complete readout value set."""
    named,layout=readout_layout(engine.shared_model)
    selected=named if values is None else values
    if (set(selected)!=set(named) or any(value.shape!=named[name].shape or value.dtype!=named[name].dtype
            for name,value in selected.items())):
        raise ValueError('native readout snapshot needs its complete exact named inventory')
    return engine.store.put_native_tensor_artifact(NativeTensorArtifact('parameter-set',dict(engine.native_identity),
        {name:value.detach().cpu().clone() for name,value in selected.items()},provenance={'readout':layout}),owner)


def bind_readout(engine, ident):
    model=engine.shared_model
    cache=_CACHE.get()
    key=(id(engine),id(model),ident,str(engine.device))
    if cache is not None and key in cache:return cache[key]
    record=engine.store.get_native_tensor_artifact(ident)
    named,layout=readout_layout(engine.shared_model)
    if record is None:raise ValueError('native query readout artifact is unavailable')
    artifact=record.artifact.validate()
    if (artifact.kind!='parameter-set' or artifact.native_identity!=engine.native_identity or
            artifact.provenance!={'readout':layout} or set(artifact.tensors)!=set(named) or
            any(value.shape!=named[name].shape or value.dtype!=named[name].dtype for name,value in artifact.tensors.items())):
        raise ValueError('native query readout changed its exact base/layout/tensor inventory')
    values={'model.'+name:value.detach().to(engine.device) for name,value in artifact.tensors.items()}
    identity=dict(engine.native_identity)
    def read(hidden):
        if engine.shared_model is not model or engine.native_identity!=identity:
            raise ValueError('native query readout executor changed its bound model')
        return torch.func.functional_call(_Operation(model,lambda:model.logits(hidden)),values,())
    if cache is not None:cache[key]=read
    return read
