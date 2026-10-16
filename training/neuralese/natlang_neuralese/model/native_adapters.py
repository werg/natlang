"""Identity-bound tiny adapters for functional native ports.

Frozen bases and coefficients use the existing named tensor artifact lifecycle.
Application uses torch.func's exact parameter substitution, not legacy module
hooks. The operation includes any checkpoint recomputation and backward.
"""
from __future__ import annotations

import contextvars
import math
import torch
from torch import nn

from ..common.hashing import canonical_json_sha256_hex
from .native_artifact import NativeTensorArtifact
from .tiny_adapters import coefficient_core, singular_bases

SCHEMA = 'natlang.native-adapter/1'
_ACTIVE = contextvars.ContextVar('natlang_native_adapter_execution', default=())


class NativeAdapterError(ValueError):
    pass


def target_parameters(model):
    # Protect embeddings, output/interface rows and control transforms. Named
    # layer matrices include both 2-D projections and batched expert banks.
    return {name: value for name, value in [*model.named_parameters(), *model.named_buffers()]
            if name.startswith('port.hf.') and '.layers.' in name and value.is_floating_point()
            and (name.endswith('.weight') and value.ndim == 2 or
                 name.endswith(('.experts.gate_up', '.experts.down')) and value.ndim == 3)}


def create_adapter(engine, options, owner=None):
    with engine.native_execution_lock:
        if execution_binding(engine):
            raise ValueError('create native adapter bases from the unadapted live model')
        return _create_adapter(engine, options, owner)


def _create_adapter(engine, options, owner=None):
    kind = options.get('kind', 'xs')
    rank, dim, seed = options.get('rank', 8), options.get('u', 0), options.get('seed', 0)
    if (kind not in ('xs', 'tiny') or type(rank) is not int or rank < 1
            or type(dim) is not int or dim < 0 or (kind == 'tiny' and dim < 1)
            or type(seed) is not int):
        raise ValueError('native adapter requires kind xs/tiny, positive rank, tiny u and integer seed')
    if options.get('native_identity') not in (None, engine.native_identity):
        raise ValueError('native adapter creation is bound to another exact live model')
    names = options.get('parameter_names')
    available = target_parameters(engine.shared_model)
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or any(name not in available for name in names)):
        raise ValueError('parameter_names must uniquely select actual named native layer matrices')
    spec = {'schema': SCHEMA, 'kind': kind, 'rank': rank, 'u': dim, 'seed': seed,
            'parameter_names': names, 'bases': 'weight-svd-sign-canonical-v2'}
    tensors, coefficients = {}, {}
    with torch.inference_mode(False), torch.no_grad():
        for index, name in enumerate(names):
            weight = available[name]
            if rank > min(weight.shape[-2:]):
                raise ValueError(f'adapter rank exceeds target matrix dimensions: {name}')
            left, right = singular_bases(weight.detach().float(), rank, version=2)
            tensors['left.'+name], tensors['right.'+name] = left.cpu(), right.cpu()
            shape = (*weight.shape[:-2], rank * rank if kind == 'xs' else dim)
            coefficients[name] = torch.zeros(shape, dtype=torch.float32)
            if kind == 'tiny':
                from .tiny_adapters import AdapterBank, AdapterSpec
                legacy = AdapterSpec(base='native', kind=kind, rank=rank, dim=dim,
                                     layers=(0,), targets=('out',), seed=seed)
                tensors['projection.'+name] = AdapterBank.projections(engine.adapter_bank, legacy, index, 'cpu')
    basis = engine.store.put_native_tensor_artifact(NativeTensorArtifact(
        'adapter-basis', engine.native_identity, tensors, provenance={'adapter_spec': spec}), owner)
    return engine.store.put_native_tensor_artifact(NativeTensorArtifact(
        'adapter-coefficients', engine.native_identity, coefficients, basis.id,
        {'adapter_spec': spec}), owner)


def resolve_adapters(engine, entries, leaves=None):
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError('adapters must be a list of native coefficient artifact IDs or id/scale objects')
    resolved = []
    available = target_parameters(engine.shared_model)
    for entry in entries:
        if isinstance(entry, str):
            adapter_id, scale = entry, 1.
        elif isinstance(entry, dict) and set(entry) <= {'id', 'scale'}:
            adapter_id, scale = entry.get('id'), entry.get('scale', 1.)
        else:
            raise ValueError('native adapters require an exact id and optional finite scale; projected codes are unsupported')
        if not isinstance(adapter_id, str) or type(scale) not in (int, float) or not math.isfinite(scale):
            raise ValueError('native adapter ID and scale are invalid')
        record = engine.store.get_native_tensor_artifact(adapter_id)
        if record is None:
            raise ValueError(f'unknown native adapter {adapter_id}; restore its coefficients and basis first')
        adapter = record.artifact.validate()
        basis_record = engine.store.get_native_tensor_artifact(adapter.source_artifact_id or '')
        if basis_record is None:
            raise ValueError('native adapter is missing its exact frozen basis artifact')
        basis = basis_record.artifact.validate()
        spec = (adapter.provenance or {}).get('adapter_spec')
        if (adapter.kind != 'adapter-coefficients' or basis.kind != 'adapter-basis'
                or adapter.native_identity != engine.native_identity or basis.native_identity != engine.native_identity
                or not isinstance(spec, dict) or spec != (basis.provenance or {}).get('adapter_spec')
                or spec.get('schema') != SCHEMA or spec.get('bases') != 'weight-svd-sign-canonical-v2'):
            raise ValueError('native adapter/basis must bind the exact live checkpoint, architecture and declared specification')
        names, rank, dim, kind = (spec.get(key) for key in ('parameter_names', 'rank', 'u', 'kind'))
        if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names) or set(adapter.tensors) != set(names)
                or type(rank) is not int or rank < 1 or type(dim) is not int or dim < 0
                or kind not in ('xs', 'tiny') or (kind == 'tiny' and dim < 1)):
            raise ValueError('invalid native adapter specification/inventory')
        expected_basis = {prefix+name for name in names for prefix in
                          (('left.', 'right.', 'projection.') if kind == 'tiny' else ('left.', 'right.'))}
        if set(basis.tensors) != expected_basis:
            raise ValueError('native adapter basis inventory differs from its specification')
        values = leaves.get(adapter_id, adapter.tensors) if leaves else adapter.tensors
        if (any(value.dtype != torch.float32 for value in basis.tensors.values())
                or any(value.dtype != torch.float32 for value in values.values())):
            raise ValueError('native adapter bases and coefficient masters must be FP32')
        if set(values) != set(names):
            raise ValueError('native adapter coefficient inventory changed')
        for name in names:
            if name not in available:
                raise ValueError(f'native adapter target is not a supported layer matrix: {name}')
            shape = available[name].shape
            expected = (*shape[:-2], rank * rank if kind == 'xs' else dim)
            if (rank > min(shape[-2:]) or tuple(values[name].shape) != expected
                    or tuple(basis.tensors['left.'+name].shape) != (*shape[:-2], shape[-2], rank)
                    or tuple(basis.tensors['right.'+name].shape) != (*shape[:-2], shape[-1], rank)
                    or (kind == 'tiny' and tuple(basis.tensors['projection.'+name].shape) != (dim, rank, rank))):
                raise ValueError(f'native adapter tensor geometry differs for {name}')
        resolved.append((record.id, float(scale), spec, values, basis.tensors))
    return resolved


def effective_identity(engine, resolved):
    if not resolved:
        return dict(engine.native_identity)
    binding = {'schema': SCHEMA, 'base_native_identity': engine.native_identity,
               'adapters': [{'id': ident, 'scale_hex': scale.hex()} for ident, scale, *_ in resolved]}
    return {**engine.native_identity, 'checkpoint_sha256': canonical_json_sha256_hex(binding)}


def execution_identity(engine):
    active = next((frame for frame in reversed(_ACTIVE.get()) if frame[0] is engine), None)
    return active[1] if active is not None else engine.native_identity


def execution_binding(engine):
    active = next((frame for frame in reversed(_ACTIVE.get()) if frame[0] is engine), None)
    return active[2] if active is not None else []


def capture_native_execution(callback):
    """Carry exact actor metadata into CUDA checkpoint recomputation threads."""
    import functools
    frames = _ACTIVE.get()
    if not frames:
        return callback
    from ..common.native_higher_order import active_reference_arithmetic
    reference=active_reference_arithmetic()
    snapshots=[]
    if reference is not None:
        # A later derivative may recompute this earlier layer while another
        # functional update is active. Retain exact tensor references, never
        # detached copies, and re-enter their complete module bindings.
        seen=set()
        for frame in frames:
            engine=frame[0]
            if id(engine) in seen:continue
            seen.add(id(engine))
            model=engine.shared_model
            values={**dict(model.named_parameters(remove_duplicate=False)),
                    **dict(model.named_buffers(remove_duplicate=False))}
            snapshots.append((model,{'model.'+name:value for name,value in values.items()}))
    @functools.wraps(callback)
    def execute(*args, **kwargs):
        if any(not frame[7]['active'] for frame in frames):
            raise NativeAdapterError('native checkpoint actor scope expired; complete forward and backward within the same native operation')
        token = _ACTIVE.set(frames)
        try:
            def replay(index):
                if index==len(snapshots):return callback(*args,**kwargs)
                model,values=snapshots[index]
                return torch.func.functional_call(_Operation(model,lambda:replay(index+1)),values,())
            return replay(0)
        finally:
            _ACTIVE.reset(token)
    return execute


def native_fp32_state_cotangents(port=None):
    """The explicit FP32 adapter-gradient scope also retains shared KV fan-in."""
    frame = next((frame for frame in reversed(_ACTIVE.get()) if port is None or frame[0].shared_model.port is port), None)
    return torch.is_grad_enabled() and frame is not None and frame[6] == 'fp32'


def substituted_native_weight(weight):
    """Whether a live functional weight must bypass a frozen-value cache."""
    return any(id(weight) in frame[4] for frame in _ACTIVE.get())


class _Operation(nn.Module):
    def __init__(self, model, callback):
        super().__init__()
        self.model = model
        self.callback = callback

    def forward(self):
        return self.callback()


def _effective_replacements(parameters, resolved, gradient_precision):
    """Construct once from immutable base objects, never another actor's scope."""
    replacement = {}
    for _, scale, spec, coefficients, bases in resolved:
        for name, coefficient in coefficients.items():
            weight = parameters[name]
            left, right = (bases[prefix+name].to(device=weight.device, dtype=torch.float32)
                           for prefix in ('left.', 'right.'))
            projection = bases.get('projection.'+name)
            core = coefficient_core(spec['kind'], coefficient.to(device=weight.device), spec['rank'],
                projection.to(device=weight.device) if projection is not None else None)
            delta = ((left @ core) @ right.transpose(-2, -1)) * scale
            key = 'model.'+name
            from ..common.native_higher_order import active_reference_arithmetic
            if active_reference_arithmetic() is not None:
                # The declared mathematical reference differentiates actual
                # stored base values plus the unrounded FP32 low-rank delta.
                # It does not claim the rounded fast actor's VJP/Hessian.
                replacement[key] = replacement.get(key,weight).float() + delta.float()
            elif gradient_precision == 'fp32' and weight.dtype != torch.float32:
                from .native_weight_carrier import rounded_weight_carrier
                replacement[key] = rounded_weight_carrier(replacement.get(key, weight), delta, weight.dtype)
            else:
                replacement[key] = replacement.get(key, weight) + delta.to(weight.dtype)
    storage_dtypes = {}
    from ..common.native_higher_order import active_reference_arithmetic
    if gradient_precision == 'fp32' or active_reference_arithmetic() is not None:
        for key, value in replacement.items():
            dtype = parameters[key.removeprefix('model.')].dtype
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError('FP32 coefficient carriers require FP16, BF16 or FP32 native weights')
            if dtype != torch.float32:
                storage_dtypes[id(value)] = dtype
    return replacement, storage_dtypes


def run_native(engine, callback, entries=None, *, leaves=None, gradient_precision=None):
    """Run one atomic native operation with exact temporary parameter substitution.

    Unadapted operations acquire the same lock. torch.func restores original
    parameter objects even if the callback fails. The callback must include its
    backward so activation-checkpoint replay sees the same effective weights.
    """
    if gradient_precision not in (None, 'storage', 'fp32'):
        raise ValueError('native adapter gradient precision must be storage or fp32')
    with engine.native_execution_lock:
        active = next((frame for frame in reversed(_ACTIVE.get()) if frame[0] is engine), None)
        if active is not None:
            if entries is not None and entries != active[3]:
                raise ValueError('nested native execution cannot change its active adapter binding')
            if gradient_precision is not None and gradient_precision != active[6]:
                raise ValueError('nested native execution cannot change its coefficient gradient precision')
            return callback()
        try:
            resolved = resolve_adapters(engine, entries, leaves)
        except (ValueError, TypeError, KeyError) as error:
            raise NativeAdapterError(str(error)) from error
        identity = effective_identity(engine, resolved)
        binding = [{'id': ident, 'scale_hex': scale.hex()} for ident, scale, *_ in resolved]
        from ..common.native_higher_order import active_reference_arithmetic
        reference=active_reference_arithmetic()
        lifecycle = reference.lifetime if reference is not None else {'active': True}
        token = _ACTIVE.set((*_ACTIVE.get(), (engine, identity, binding, entries, set(), {}, gradient_precision or 'storage', lifecycle)))
        try:
            if not resolved:
                return callback()
            parameters = target_parameters(engine.shared_model)
            replacement, storage_dtypes = _effective_replacements(parameters, resolved, gradient_precision)
            _ACTIVE.get()[-1][5].update(storage_dtypes)
            _ACTIVE.get()[-1][4].update(id(value) for value in replacement.values())
            from ..common.native_higher_order import active_reference_arithmetic
            if gradient_precision == 'fp32' or active_reference_arithmetic() is not None:
                from .native_weight_carrier import carrier_linear_modules
                selected = [parameters[key.removeprefix('model.')] for key in replacement]
                with carrier_linear_modules(engine.shared_model, selected):
                    return torch.func.functional_call(_Operation(engine.shared_model, callback), replacement, ())
            return torch.func.functional_call(_Operation(engine.shared_model, callback), replacement, ())
        finally:
            if reference is None:lifecycle['active'] = False
            _ACTIVE.reset(token)


def optimize_adapter(engine, gradient, options, state_id, owner=None):
    """Immutable named coefficient update using the shared service optimizer."""
    from ..train.optim import named_service_optimizer
    if gradient.native_identity != engine.native_identity or gradient.kind != 'adapter-gradient':
        raise ValueError('adapter update requires a gradient bound to this exact live native base')
    adapter_id = gradient.source_artifact_id
    invocation_actors=(gradient.provenance or {}).get('invocation_actors')
    if invocation_actors is not None:
        from .native_actor_graph import validate_invocation_actor_receipt
        validate_invocation_actor_receipt(invocation_actors,engine.native_identity)
        for actor in invocation_actors['invocations'].values():
            resolve_adapters(engine,[{'id':entry['id'],'scale':float.fromhex(entry['scale_hex'])}
                for entry in (actor['adapter_binding'] or {}).get('adapters',[])])
    from .native_derivative_contract import validate_derivative_target
    derivative_target=validate_derivative_target(gradient)
    if derivative_target is not None:
        from .native_derivative_contract import verify_derivative_gradient_origin
        verify_derivative_gradient_origin(engine.store,gradient)
    if derivative_target is not None:
        # A meta-gradient differentiates an earlier coefficient value. Its
        # evaluated actors belong to the objective, not to the updated root.
        invocation_actors=None
    actor_binding = None if derivative_target is not None else (gradient.provenance or {}).get('adapter_binding')
    if actor_binding is not None:
        from types import SimpleNamespace
        validate_state_binding(actor_binding, SimpleNamespace(**gradient.provenance['effective_native_identity']))
        if (actor_binding['base_native_identity'] != engine.native_identity
                or adapter_id not in [entry['id'] for entry in actor_binding['adapters']]):
            raise ValueError('adapter policy gradient does not bind its exact coefficient parent and base')
        resolve_adapters(engine, [{'id':entry['id'], 'scale':float.fromhex(entry['scale_hex'])}
                                 for entry in actor_binding['adapters']])
    resolved = resolve_adapters(engine, [adapter_id])[0]
    _, _, spec, coefficients, _ = resolved
    if not set(gradient.tensors) <= coefficients.keys():
        raise ValueError('adapter gradient names differ from its exact coefficient parent')
    old_record = engine.store.get_native_tensor_artifact(state_id) if state_id else None
    if state_id and old_record is None:
        raise ValueError('adapter optimizer state is unavailable; restore its exact artifact')
    old = old_record.artifact if old_record else None
    if not isinstance(options, dict):
        raise ValueError('adapter optimizer options must be an object')
    algorithm = options.get('name')
    lr, decay = options.get('learning_rate', 1e-3), options.get('weight_decay', 0.)
    beta1, beta2, eps, momentum = (options.get(key, default) for key, default in
        (('beta1', .9), ('beta2', .999), ('eps', 1e-8), ('momentum', 0.)))
    numbers = (lr, decay, beta1, beta2, eps, momentum)
    if (algorithm not in ('adam', 'sgd') or any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers)
            or lr <= 0 or decay < 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or eps <= 0 or momentum < 0):
        raise ValueError('adapter optimizer options must declare valid finite Adam/SGD settings')
    if old is not None and (old.kind != 'optimizer-state' or old.native_identity != engine.native_identity
            or old.source_artifact_id != adapter_id or (old.provenance or {}).get('optimizer') != algorithm
            or (old.provenance or {}).get('scope') != SCHEMA):
        raise ValueError('adapter optimizer state must bind the exact coefficient parent and algorithm')
    state_tensors = old.tensors if old else {}
    if old is not None:
        prior_step = old.provenance.get('step')
        if (type(prior_step) is not int or prior_step < 1 or 'optimizer.step' not in state_tensors
                or state_tensors['optimizer.step'].numel() != 1
                or state_tensors['optimizer.step'].item() != prior_step):
            raise ValueError('adapter optimizer continuation lacks its exact step counter')
        prefixes = ('m.', 'v.', 'step.') if algorithm == 'adam' else ('momentum.',)
        for key, tensor in state_tensors.items():
            if key == 'optimizer.step':
                continue
            prefix = next((prefix for prefix in prefixes if key.startswith(prefix)), None)
            if prefix is None or key.removeprefix(prefix) not in coefficients:
                raise ValueError('adapter optimizer continuation has an unknown state tensor')
            parameter_name = key.removeprefix(prefix)
            if algorithm == 'adam':
                if not all(p+parameter_name in state_tensors for p in prefixes):
                    raise ValueError('adapter Adam continuation is incomplete')
                count = state_tensors['step.'+parameter_name]
                if (count.numel() != 1 or count.item() != int(count.item())
                        or not 1 <= count.item() <= prior_step):
                    raise ValueError('adapter Adam continuation has an invalid per-parameter step')
                if bool((state_tensors['v.'+parameter_name] < 0).any()):
                    raise ValueError('adapter Adam continuation has a negative second moment')
            if prefix != 'step.' and tensor.shape != coefficients[parameter_name].shape:
                raise ValueError('adapter optimizer continuation geometry differs')
    step = old.provenance['step']+1 if old else 1
    named = [(name, nn.Parameter(coefficients[name].detach().float().clone())) for name in sorted(gradient.tensors)]
    optimizer = named_service_optimizer(named, name=algorithm, learning_rate=lr, weight_decay=decay,
        beta1=beta1, beta2=beta2, eps=eps, momentum=momentum)
    for name, parameter in named:
        grad = gradient.tensors[name]
        if grad.shape != parameter.shape or not torch.isfinite(grad).all():
            raise ValueError(f'adapter gradient geometry/values differ for {name}')
        parameter.grad = grad.to(parameter)
        if algorithm == 'adam' and 'm.'+name in state_tensors:
            if not all(key+name in state_tensors for key in ('m.', 'v.', 'step.')):
                raise ValueError('adapter Adam continuation is incomplete')
            m, v, count = (state_tensors[key+name] for key in ('m.', 'v.', 'step.'))
            if m.shape != parameter.shape or v.shape != parameter.shape or count.numel() != 1:
                raise ValueError('adapter Adam continuation geometry differs')
            optimizer.state[parameter] = {'exp_avg': m.to(parameter).clone(), 'exp_avg_sq': v.to(parameter).clone(),
                                         'step': count.reshape(()).cpu().clone()}
        elif algorithm == 'sgd' and 'momentum.'+name in state_tensors:
            value = state_tensors['momentum.'+name]
            if value.shape != parameter.shape:
                raise ValueError('adapter SGD continuation geometry differs')
            optimizer.state[parameter] = {'momentum_buffer': value.to(parameter).clone()}
    optimizer.step()
    next_coefficients = {name: value.detach().cpu().clone() for name, value in coefficients.items()}
    next_state = {name: value.detach().cpu().clone() for name, value in state_tensors.items()}
    next_state['optimizer.step'] = torch.tensor([step], dtype=torch.float64)
    for name, parameter in named:
        next_coefficients[name] = parameter.detach().cpu()
        state = optimizer.state[parameter]
        if algorithm == 'adam':
            for prefix, key in (('m.', 'exp_avg'), ('v.', 'exp_avg_sq'), ('step.', 'step')):
                next_state[prefix+name] = state[key].detach().cpu().reshape(1) if key == 'step' else state[key].detach().cpu()
        elif 'momentum_buffer' in state:
            next_state['momentum.'+name] = state['momentum_buffer'].detach().cpu()
    parent = engine.store.get_native_tensor_artifact(adapter_id).artifact
    updated = engine.store.put_native_tensor_artifact(NativeTensorArtifact('adapter-coefficients', engine.native_identity,
        next_coefficients, parent.source_artifact_id, {'adapter_spec': spec,
            'coefficient_lineage':(parent.provenance or {}).get('coefficient_lineage',adapter_id),
            **({'derivative_program_id':gradient.provenance['derivative_program_id'],
                'gradient_id':engine.store.put_native_tensor_artifact(gradient,owner).id,'optimizer_state_parent_id':state_id,
                'arithmetic':gradient.provenance['arithmetic']} if derivative_target is not None else {})}), owner)
    next_entries = ([{'id': updated.id if entry['id'] == adapter_id else entry['id'],
                      'scale': float.fromhex(entry['scale_hex'])} for entry in actor_binding['adapters']]
                    if actor_binding else [updated.id])
    next_resolved = resolve_adapters(engine, next_entries)
    next_binding = {'schema': SCHEMA, 'base_native_identity': dict(engine.native_identity),
                    'adapters': [{'id': ident, 'scale_hex': scale.hex()} for ident, scale, *_ in next_resolved]}
    if invocation_actors is not None:
        import copy
        continued=copy.deepcopy(invocation_actors)
        for actor in continued['invocations'].values():
            binding=actor['adapter_binding']
            if binding is None:continue
            for entry in binding['adapters']:
                if entry['id']==adapter_id:entry['id']=updated.id
            resolved_actor=resolve_adapters(engine,[{'id':entry['id'],'scale':float.fromhex(entry['scale_hex'])} for entry in binding['adapters']])
            actor['native_identity']=effective_identity(engine,resolved_actor)
        validate_invocation_actor_receipt(continued,engine.native_identity)
    state = engine.store.put_native_tensor_artifact(NativeTensorArtifact('optimizer-state', engine.native_identity,
        next_state, updated.id, {'scope': SCHEMA, 'optimizer': algorithm, 'step': step,
          'gradient_id': engine.store.put_native_tensor_artifact(gradient, owner).id,
          'parent_adapter_id': adapter_id, 'optimizer_options': options,
          **({'invocation_actors':continued,'actor_graph_sha256':canonical_json_sha256_hex(continued),
              **({'invocation_source_names':gradient.provenance['invocation_source_names']} if 'invocation_source_names' in gradient.provenance else {})}
             if invocation_actors is not None else {})}), owner)
    if invocation_actors is not None:
        return {'adapter':updated.meta(),'optimizer_state_id':state.id,'step':step,'native_identity':engine.native_identity,
                'invocation_actors':continued,'actor_graph_sha256':canonical_json_sha256_hex(continued),
                **({'invocation_source_names':gradient.provenance['invocation_source_names']} if 'invocation_source_names' in gradient.provenance else {})}
    return {'adapter': updated.meta(), 'optimizer_state_id': state.id, 'step': step,
            'native_identity': engine.native_identity, 'adapter_binding': next_binding,
            'effective_native_identity': effective_identity(engine, next_resolved)}



def validate_state_binding(binding, block):
    if not all(hasattr(block, name) for name in ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')):
        raise ValueError('adapted native actor is missing its effective four-field identity')
    if (not isinstance(binding, dict) or set(binding) != {'schema', 'base_native_identity', 'adapters'}
            or binding.get('schema') != SCHEMA or not isinstance(binding.get('base_native_identity'), dict)
            or not isinstance(binding.get('adapters'), list) or not binding['adapters']):
        raise ValueError('adapted native state needs its exact base identity and ordered adapter binding')
    import re
    base = binding['base_native_identity']
    if set(base) != {'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'}:
        raise ValueError('adapted native state base identity is malformed')
    if not re.fullmatch('[0-9a-f]{64}', str(base.get('checkpoint_sha256', ''))):
        raise ValueError('adapted native state base checkpoint identity is malformed')
    for name in ('tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'):
        if base.get(name) != getattr(block, name):
            raise ValueError('adapted native state changed a protected base identity field')
    for entry in binding['adapters']:
        if (not isinstance(entry, dict) or set(entry) != {'id', 'scale_hex'}
                or not re.fullmatch('nnt1_[a-z2-7]{52}', str(entry.get('id', '')))
                or not isinstance(entry.get('scale_hex'), str)):
            raise ValueError('adapted native state binding has an invalid coefficient artifact or scale')
        try:
            scale = float.fromhex(entry['scale_hex'])
        except ValueError as error:
            raise ValueError('adapted native state scale is not a hexadecimal float') from error
        if not math.isfinite(scale) or scale.hex() != entry['scale_hex']:
            raise ValueError('adapted native state scale must be finite and canonical')
    if canonical_json_sha256_hex(binding) != block.checkpoint_sha256:
        raise ValueError('adapted native state effective identity differs from its exact adapter binding')


def require_model_execution_actor(model, identity, binding):
    """Reject adapted replay outside its exact functional forward/backward scope."""
    if binding is None:
        # Base replay cannot silently inherit adapter weights from an outer call.
        if any(frame[0].shared_model is model and frame[2] for frame in _ACTIVE.get()):
            raise ValueError('native replay base actor differs from the active adapted model')
        return
    frame = next((frame for frame in reversed(_ACTIVE.get()) if frame[0].shared_model is model), None)
    if (frame is None or frame[1] != identity or frame[2] != binding['adapters']
            or frame[0].native_identity != binding['base_native_identity']):
        raise ValueError('adapted native replay requires its exact effective actor scope through backward')


def native_forward_storage_dtype(weight):
    """Explicit forward storage dtype of an active carrier or its bank views."""
    current = weight
    while current is not None:
        for frame in reversed(_ACTIVE.get()):
            dtype = frame[5].get(id(current))
            if dtype is not None:
                return dtype
        current = getattr(current, '_base', None)
    return None


SAME_BASE_STATE_MODE = 'same-base-adapter-state'


def adapter_state_input_receipt(engine, block, state_id, *, target_position_start):
    """Authenticate an explicitly requested raw-state exchange, never relabel it.

    Geometry is checked by the existing splice/graph executor before committing
    the new cache. This verifier binds the source coefficient closure and both
    actors to the exact same immutable base, independent of matching widths.
    """
    from .state_block import NativeStateBlock
    if type(block) is not NativeStateBlock:
        raise NativeAdapterError('same-base adapter-state mode requires raw native state; use the learned semantic consumer for semantic messages')
    from .shared_state import validate_native_state_layout
    validate_native_state_layout(engine.shared_model.port, block)
    fields = ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')
    source = {key: getattr(block, key) for key in fields}
    source_binding = block.adapter_binding
    base = source_binding['base_native_identity'] if source_binding else source
    if base != engine.native_identity:
        raise NativeAdapterError('same-base adapter-state mode requires the exact same base checkpoint, tokenizer, embeddings and architecture')
    if source_binding:
        validate_state_binding(source_binding, block)
        resolved = resolve_adapters(engine, [{'id': item['id'], 'scale': float.fromhex(item['scale_hex'])}
                                            for item in source_binding['adapters']])
        if effective_identity(engine, resolved) != source:
            raise NativeAdapterError('source native state does not match its restored coefficient and basis closure')
    target_binding = execution_binding(engine)
    target_binding = ({'schema': SCHEMA, 'base_native_identity': dict(engine.native_identity),
                       'adapters': target_binding} if target_binding else None)
    if type(target_position_start) is not int or target_position_start < 0:
        raise NativeAdapterError('same-base adapter-state target position must be a nonnegative integer')
    return {'schema': 'natlang.native-adapter-state-consumption/1',
            'mode': SAME_BASE_STATE_MODE, 'source_state_id': state_id,
            'source_native_identity': source, 'target_native_identity': dict(execution_identity(engine)),
            'base_native_identity': dict(engine.native_identity),
            'source_adapter_binding': source_binding, 'target_adapter_binding': target_binding,
            'target_position_start': target_position_start, 'tokens': len(block.token_ids)}


def validate_adapter_state_receipt(block, receipt, *, target_identity, target_binding):
    """Exact immutable capture verifier; execution also resolves basis artifacts."""
    from ..serve.store import native_state_id
    source = {name: getattr(block, name) for name in
              ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')}
    base = block.adapter_binding['base_native_identity'] if block.adapter_binding else source
    target_base = target_binding['base_native_identity'] if target_binding else target_identity
    if (not isinstance(receipt, dict) or set(receipt) != {
            'schema', 'mode', 'source_state_id', 'source_native_identity', 'target_native_identity',
            'base_native_identity', 'source_adapter_binding', 'target_adapter_binding', 'target_position_start', 'tokens'}
            or receipt['schema'] != 'natlang.native-adapter-state-consumption/1'
            or receipt['mode'] != SAME_BASE_STATE_MODE or receipt['source_state_id'] != native_state_id(block)
            or receipt['source_native_identity'] != source or receipt['source_adapter_binding'] != block.adapter_binding
            or receipt['base_native_identity'] != base or base != target_base
            or receipt['target_native_identity'] != target_identity or receipt['target_adapter_binding'] != target_binding
            or type(receipt['target_position_start']) is not int or receipt['target_position_start'] < 0
            or type(receipt['tokens']) is not int or receipt['tokens'] != len(block.token_ids)):
        raise ValueError('same-base adapter-state receipt differs from exact source, target or positional contract')
    block.validate()
