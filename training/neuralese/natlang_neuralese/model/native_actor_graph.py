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
    def __init__(self, engine, names, bindings, *, default=None, leaves=None, gradient_precision='storage', source_names=None):
        if gradient_precision not in ('storage','fp32'):
            raise NativeAdapterError('native adapter gradient precision must be storage or fp32')
        if (not isinstance(bindings,dict) or any(name not in names for name in bindings)
                or len(set(names))!=len(names)):
            raise NativeAdapterError('invocation adapters must bind unique actual graph invocation names')
        if any(frame[0] is engine for frame in _ACTIVE.get()):
            raise NativeAdapterError('prepare invocation actors outside any active native functional substitution')
        if source_names is None:source_names={name:name for name in names}
        if (not isinstance(source_names,dict) or len(source_names)!=len(names) or set(source_names.values())!=set(names)):
            raise NativeAdapterError('native invocation source names must bind exact actual packed ownership')
        self.source_names=dict(source_names)
        self.engine,self.precision=engine,gradient_precision
        from ..common.native_higher_order import active_reference_arithmetic
        reference=active_reference_arithmetic()
        self.reference_lifetime=reference is not None
        self.lifecycle=reference.lifetime if reference is not None else {'active':True}
        self.actors={}
        self.parameters=target_parameters(engine.shared_model)
        # Capture immutable base parameters BEFORE entering any actor scope.
        cache={}
        for name in names:
            entries=bindings.get(name,default)
            resolved=resolve_adapters(engine,entries,leaves)
            identity=effective_identity(engine,resolved)
            key=(identity['checkpoint_sha256'], tuple(isinstance(entry,dict) and entry.get('stop_gradient',False) for entry in entries or []))
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

    def receipt(self):
        import copy
        from .native_adapters import SCHEMA
        result={'schema':'natlang.native-invocation-actors/1','base_native_identity':dict(self.engine.native_identity),
                'invocations':{name:{'native_identity':dict(actor[0][1]),
                    'adapter_binding':({'schema':SCHEMA,'base_native_identity':dict(self.engine.native_identity),
                                        'adapters':copy.deepcopy(actor[0][2])} if actor[0][2] else None)}
                               for name,actor in self.actors.items()}}
        return validate_invocation_actor_receipt(result,self.engine.native_identity)

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
            from ..common.native_higher_order import active_reference_arithmetic
            if self.precision=='fp32' or active_reference_arithmetic() is not None:
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
            key=(actor[0][1]['checkpoint_sha256'], tuple(isinstance(entry,dict) and entry.get('stop_gradient',False) for entry in actor[0][3] or []))
            groups.setdefault(key,(node.name,[]))
            owners.append(key)
        for row,owner in enumerate(plan.node_ids):
            groups[owners[owner]][1].append(row)
        return [(name,torch.tensor(rows,device=device,dtype=torch.long)) for name,rows in groups.values() if rows]


def run_invocation_actors(engine,names,bindings,callback,*,default=None,leaves=None,gradient_precision='storage',source_names=None):
    """One operation lease includes every actor's forward and checkpoint backward."""
    with engine.native_execution_lock:
        bank=NativeInvocationActors(engine,names,bindings,default=default,leaves=leaves,gradient_precision=gradient_precision,source_names=source_names)
        frame=(engine,dict(engine.native_identity),[],None,set(),{},gradient_precision,bank.lifecycle)
        active=_ACTIVE.set((*_ACTIVE.get(),frame));token=_BANKS.set((*_BANKS.get(),bank))
        try:
            return callback(bank)
        finally:
            if not bank.reference_lifetime:bank.lifecycle['active']=False
            _BANKS.reset(token);_ACTIVE.reset(active)


def validate_invocation_actor_receipt(receipt,base):
    """Validate every actual invocation actor against one exact immutable base."""
    from types import SimpleNamespace
    from .native_adapters import validate_state_binding
    if (not isinstance(receipt,dict) or set(receipt)!={'schema','base_native_identity','invocations'}
            or receipt.get('schema')!='natlang.native-invocation-actors/1'
            or receipt.get('base_native_identity')!=base or not isinstance(receipt.get('invocations'),dict)
            or not receipt['invocations']):
        raise NativeAdapterError('native invocation actor receipt needs its exact base and complete invocation inventory')
    for name,actor in receipt['invocations'].items():
        if (not isinstance(name,str) or not name or not isinstance(actor,dict)
                or set(actor)!={'native_identity','adapter_binding'}
                or not isinstance(actor['native_identity'],dict) or set(actor['native_identity'])!=set(base)):
            raise NativeAdapterError('native invocation actor identity is malformed')
        binding=actor['adapter_binding']
        if binding is None:
            if actor['native_identity']!=base:
                raise NativeAdapterError('unadapted invocation actor differs from its immutable base')
        else:
            validate_state_binding(binding,SimpleNamespace(**actor['native_identity']))
            if binding['base_native_identity']!=base:
                raise NativeAdapterError('native invocation actor changed its immutable base')
    return receipt


def validate_invocation_actor_provenance(provenance,base):
    receipt=(provenance or {}).get('invocation_actors')
    if receipt is None:return None
    from ..common.hashing import canonical_json_sha256_hex
    validate_invocation_actor_receipt(receipt,base)
    names=provenance.get('invocation_source_names')
    if names is not None and (not isinstance(names,dict) or any(not isinstance(key,str) or not key for key in names)
            or len(names)!=len(receipt['invocations']) or set(names.values())!=set(receipt['invocations'])):
        raise NativeAdapterError('native invocation source names changed exact packed ownership')
    if provenance.get('actor_graph_sha256')!=canonical_json_sha256_hex(receipt):
        raise NativeAdapterError('native invocation actor graph digest differs from its exact receipt')
    return receipt


def validate_invocation_behavior_policy(policy,base):
    import re
    if (not isinstance(policy,dict) or set(policy)!={'schema','base_native_identity','collector_policy_role','episodes'}
            or policy.get('schema')!='natlang.native-invocation-policy/1'
            or policy.get('base_native_identity')!=base or policy.get('collector_policy_role')!='base-runtime-metadata'
            or not isinstance(policy.get('episodes'),dict) or not policy['episodes']):
        raise ValueError('native invocation behavior policy needs its exact base and immutable episode inventory')
    for episode,actors in policy['episodes'].items():
        if not isinstance(episode,str) or not re.fullmatch('[0-9a-f]{64}',episode):
            raise ValueError('native invocation behavior episode must bind exact original row bytes')
        validate_invocation_actor_receipt(actors,base)
    return policy


def validate_invocation_behavior_provenance(provenance,base):
    policy=(provenance or {}).get('behavior_policy')
    if policy is None:
        if 'behavior_policy_sha256' in (provenance or {}):
            raise NativeAdapterError('native invocation behavior digest lacks its actual descriptor')
        return None
    from ..common.hashing import canonical_json_sha256_hex
    validate_invocation_behavior_policy(policy,base)
    if provenance.get('behavior_policy_sha256')!=canonical_json_sha256_hex(policy):
        raise NativeAdapterError('native invocation behavior descriptor digest changed')
    return policy


from contextlib import contextmanager


@contextmanager
def invocation_graph_scope(port, names):
    """A graph view shares immutable actor weights and the enclosing AD lifetime."""
    bank=invocation_actor_bank(port)
    if bank is None:
        yield None
        return
    names=set(names)
    if not names or not names<=set(bank.actors):
        raise NativeAdapterError('native graph segment must retain actual declared invocation actors')
    import copy
    view=copy.copy(bank)
    view.actors={name:actor for name,actor in bank.actors.items() if name in names}
    view.source_names={source:target for source,target in bank.source_names.items() if target in names}
    token=_BANKS.set((*_BANKS.get(),view))
    try:yield view
    finally:_BANKS.reset(token)


@contextmanager
def suspend_invocation_graph(port):
    """A codec baseline uses its exact uniform actor, not episode row ownership.

    The caller enters that actor via bank.execute; only layer-major routing is
    suspended. Functional weights, gradient carriers and the AD lifetime stay.
    """
    token=_BANKS.set(tuple(bank for bank in _BANKS.get() if bank.engine.shared_model.port is not port))
    try:yield
    finally:_BANKS.reset(token)
