"""Layer-major invocation actor scopes over one immutable native base.

This is exact execution plumbing, not an actor-selection policy. Callers declare
bindings; every physical token executes once per layer under its declared actor.
"""
from __future__ import annotations

import contextvars
import torch
from .native_adapters import (_ACTIVE, _Operation, NativeAdapterError, target_parameters,
    resolve_adapters, effective_identity, _effective_replacements)

_BANKS = contextvars.ContextVar('natlang_native_invocation_actors', default=())


def invocation_actor_bank(port):
    return next((bank for bank in reversed(_BANKS.get()) if bank.engine.shared_model.port is port), None)


class NativeInvocationActors:
    def __init__(self, engine, names, bindings, *, default=None, leaves=None, gradient_precision='storage'):
        if gradient_precision not in ('storage','fp32'):
            raise NativeAdapterError('native adapter gradient precision must be storage or fp32')
        if (not isinstance(bindings,dict) or any(name not in names for name in bindings)
                or len(set(names))!=len(names)):
            raise NativeAdapterError('invocation adapters must bind unique actual graph invocation names')
        if any(frame[0] is engine for frame in _ACTIVE.get()):
            raise NativeAdapterError('prepare invocation actors outside any active native functional substitution')
        self.engine,self.precision=engine,gradient_precision
        self.lifecycle={'active':True}
        self.actors={}
        self.parameters=target_parameters(engine.shared_model)
        # Capture immutable base parameters BEFORE entering any actor scope.
        cache={}
        for name in names:
            entries=bindings.get(name,default)
            resolved=resolve_adapters(engine,entries,leaves)
            identity=effective_identity(engine,resolved)
            key=identity['checkpoint_sha256']
            if key not in cache:
                binding=[{'id':ident,'scale_hex':scale.hex()} for ident,scale,*_ in resolved]
                replacement,dtypes=_effective_replacements(self.parameters,resolved,gradient_precision)
                frame=(engine,identity,binding,entries,set(map(id,replacement.values())),dtypes,gradient_precision,self.lifecycle)
                selected=[self.parameters[key.removeprefix('model.')] for key in replacement]
                cache[key]=(frame,replacement,selected)
            frame,replacement,selected=cache[key]
            # Matrix reuse follows exact effective identity; nested service calls
            # still retain each invocation's own declared request representation.
            frame=(*frame[:3],entries,*frame[4:])
            self.actors[name]=(frame,replacement,selected)

    def execute(self, name, callback):
        if not self.lifecycle['active']:
            raise NativeAdapterError('native invocation actor scope expired; complete backward within the graph operation')
        if any(frame[0] is self.engine and frame[2] for frame in _ACTIVE.get()):
            raise NativeAdapterError('native invocation actors cannot nest inside an adapted functional scope')
        if name not in self.actors:
            raise NativeAdapterError('native graph invocation has no declared actor')
        frame,replacement,selected=self.actors[name]
        token=_ACTIVE.set((*_ACTIVE.get(),frame))
        try:
            if not replacement:
                return callback()
            if self.precision=='fp32':
                from .native_weight_carrier import carrier_linear_modules
                with carrier_linear_modules(self.engine.shared_model,selected):
                    return torch.func.functional_call(_Operation(self.engine.shared_model,callback),replacement,())
            return torch.func.functional_call(_Operation(self.engine.shared_model,callback),replacement,())
        finally:
            _ACTIVE.reset(token)

    def row_groups(self,plan,device):
        if set(node.name for node in plan.invocations)!=set(self.actors):
            raise NativeAdapterError('native graph actor inventory differs from its actual invocation plan')
        groups={}
        owners=[]
        for node in plan.invocations:
            actor=self.actors[node.name]
            key=actor[0][1]['checkpoint_sha256']
            groups.setdefault(key,(node.name,[]))
            owners.append(key)
        for row,owner in enumerate(plan.node_ids):
            groups[owners[owner]][1].append(row)
        return [(name,torch.tensor(rows,device=device,dtype=torch.long)) for name,rows in groups.values() if rows]


def run_invocation_actors(engine,names,bindings,callback,*,default=None,leaves=None,gradient_precision='storage'):
    """One operation lease includes every actor's forward and checkpoint backward."""
    with engine.native_execution_lock:
        bank=NativeInvocationActors(engine,names,bindings,default=default,leaves=leaves,gradient_precision=gradient_precision)
        frame=(engine,dict(engine.native_identity),[],None,set(),{},gradient_precision,bank.lifecycle)
        active=_ACTIVE.set((*_ACTIVE.get(),frame));token=_BANKS.set((*_BANKS.get(),bank))
        try:
            return callback(bank)
        finally:
            bank.lifecycle['active']=False
            _BANKS.reset(token);_ACTIVE.reset(active)
