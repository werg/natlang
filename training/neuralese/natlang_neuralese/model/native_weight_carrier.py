"""Separate exact forward weight storage from a coefficient's FP32 cotangent.

Only explicit native adapter scopes install these carriers. Ordinary model
execution keeps its original module objects, dtypes, kernels and autograd.
"""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
import os
import tempfile

import torch
from torch import nn
import torch.nn.functional as F


_COTANGENT_SCOPE = contextvars.ContextVar('natlang_bf16_parameter_cotangent_scope', default=None)
_RMS_NORM_TYPES = {'RMSNorm', 'Lfm2RMSNorm', 'Qwen2RMSNorm', 'Qwen3RMSNorm'}
_COTANGENT_TRANSFER_CHUNK_BYTES = 16 * 1024 * 1024


def _is_rms_norm_module(module):
    return type(module).__name__ in _RMS_NORM_TYPES and isinstance(getattr(module, 'weight', None), nn.Parameter)


def _conv1d_carrier_forward(scope, module, original_forward):
    def forward(x):
        if _reference_scope_active(scope):
            return _reference_module_forward(scope, module, original_forward, x)
        weight_anchors = scope.anchors_for_weight(module.weight)
        bias_anchors = scope.anchors_for_weight(module.bias) if module.bias is not None else ()
        count = len(weight_anchors)
        return _TorchStoredConv1d.apply(x, module.weight, module.bias, module, original_forward, count,
                                        *weight_anchors, *bias_anchors)
    return forward


def _rms_norm_carrier_forward(scope, module, original_forward):
    def forward(x):
        if _reference_scope_active(scope):
            return _reference_module_forward(scope, module, original_forward, x)
        anchors = scope.anchors_for_weight(module.weight)
        eps = next((getattr(module, name) for name in ('variance_epsilon', 'eps', 'epsilon')
                    if hasattr(module, name)), None)
        if eps is None:
            raise RuntimeError(f'RMSNorm carrier cannot identify epsilon for {type(module).__qualname__}')
        return _TorchStoredRMSNorm.apply(x, module.weight, float(eps), module, original_forward, *anchors)
    return forward


def _reference_scope_active(scope):
    if active_cotangent_scope() is not scope:
        return False
    from ..common.native_higher_order import active_reference_arithmetic
    return active_reference_arithmetic() is not None


def _reference_module_forward(scope, module, original_forward, *args, **kwargs):
    """Temporarily route selected low-precision module operands via shared casts."""
    replacements, originals = {}, {}
    for name in ('weight', 'bias'):
        parameter = getattr(module, name, None)
        if isinstance(parameter, torch.Tensor) and scope._entry_for(parameter) is not None:
            value = scope.reference_parameter(parameter).to(dtype=parameter.dtype,
                                                             device=parameter.device)
            replacements[name] = value
    try:
        for name, value in replacements.items():
            originals[name] = module._parameters[name]
            module._parameters[name] = value
        return original_forward(*args, **kwargs)
    finally:
        for name, value in originals.items():
            module._parameters[name] = value


def _embedding_carrier_forward(module):
    def forward(input):
        return stored_embedding(input, module.weight, padding_idx=module.padding_idx,
                                max_norm=module.max_norm, norm_type=module.norm_type,
                                scale_grad_by_freq=module.scale_grad_by_freq, sparse=module.sparse)
    return forward


def _lfm_causal_carrier_forward(scope, module, original_forward):
    function = getattr(original_forward, '__func__', original_forward)
    while getattr(function, '__wrapped__', None) is not None:
        function = function.__wrapped__
    namespace = function.__globals__
    function_name = 'causal_conv1d_fn'
    operation = namespace.get(function_name)
    if not callable(operation):
        raise RuntimeError('LFM short convolution has no callable causal_conv1d_fn')
    def dispatch(x, weight, bias=None, activation=None, **kwargs):
        if _reference_scope_active(scope):
            weight = scope.reference_parameter(weight).to(dtype=weight.dtype, device=weight.device)
            if bias is not None:
                bias = scope.reference_parameter(bias).to(dtype=bias.dtype, device=bias.device)
            return operation(x, weight, bias, activation=activation, **kwargs)
        anchors = scope.anchors_for_weight(weight)
        if not anchors:
            return operation(x, weight, bias, activation=activation, **kwargs)
        return _TorchStoredCausalConv1d.apply(x, weight, bias, operation, activation, kwargs,
                                             scope, len(anchors), *anchors)
    def forward(*args, **kwargs):
        previous = namespace[function_name]
        namespace[function_name] = dispatch
        try:
            return original_forward(*args, **kwargs)
        finally:
            namespace[function_name] = previous
    return forward


class ParameterCotangentScope:
    """Collect low-precision parameter cotangents in FP32 outside autograd fan-in.

    The per-parameter accumulator is a sparse, file-backed CPU tensor. CUDA
    backward paths therefore retain only their current FP32 contribution; the
    optimizer-facing gradient is materialized once, at finalization.
    """
    def __init__(self, parameters, *, mode='update', directory=None, track_auxiliary=True,
                 auxiliary_target_ids=None, parameter_target_ids=None):
        if mode not in ('update', 'vjp'):
            raise ValueError("cotangent scope mode must be 'update' or 'vjp'")
        self.mode = mode
        self.track_auxiliary = bool(track_auxiliary)
        self.auxiliary_target_ids = (None if auxiliary_target_ids is None
                                     else frozenset(auxiliary_target_ids))
        self.parameter_target_ids = (None if parameter_target_ids is None
                                     else frozenset(parameter_target_ids))
        unique = {}
        for name, parameter in parameters:
            if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16):
                unique.setdefault(id(parameter), (name, parameter))
        storages = {}
        for name, parameter in unique.values():
            key = (parameter.device.type, parameter.device.index, parameter.untyped_storage().data_ptr())
            previous = storages.get(key)
            if previous is not None and previous[1] is not parameter:
                raise RuntimeError(f'overlapping distinct low-precision optimizer parameters are unsupported: {name} and {previous[0]}')
            storages[key] = (name, parameter)
        for name, parameter in unique.values():
            if not parameter.is_contiguous():
                raise RuntimeError(f'non-contiguous low-precision optimizer parameter needs an explicit cotangent view binding: {name}')
        self.entries = {}
        self._auxiliary_entries = {}
        self._effective_routes = {}
        self.directory = tempfile.mkdtemp(prefix='natlang-cotangent-', dir=directory)
        self._token = None
        self._closed = False
        self._module_changes = []
        self._wrappers = {}
        self._gradient_hooks = []
        self.model = None
        self._reference_parameters = {}
        for index, (name, parameter) in enumerate(unique.values()):
            initial_grad=parameter.grad if parameter.is_leaf else None
            self.entries[id(parameter)] = {'name': name, 'parameter': parameter,
                'count': 0, 'path': os.path.join(self.directory, f'{index:06d}.f32'), 'accumulator': None,
                'anchor_value': None, 'anchor_backward': 0, 'initial_grad': initial_grad,
                'initial_grad_version': None if initial_grad is None else initial_grad._version}

    def __enter__(self):
        if _COTANGENT_SCOPE.get() is not None:
            self.close()
            raise RuntimeError('BF16 parameter cotangent scopes cannot nest')
        self._token = _COTANGENT_SCOPE.set(self)
        try:
            # Route ordinary transformer projections through the same custom edge.
            # Exact module and Parameter objects are restored on scope exit.
            if self.model is not None:
                selected = set(self.entries)
                for name, module in list(self.model.named_modules(remove_duplicate=False)):
                    weight = getattr(module, 'weight', None)
                    bias = getattr(module, 'bias', None)
                    if type(module) is nn.Embedding and id(weight) in selected:
                        original = module.forward
                        module.forward = _embedding_carrier_forward(module)
                        self._module_changes.append(('forward', module, original))
                    elif type(module) is nn.Linear and (id(weight) in selected or id(bias) in selected):
                        wrapper = self._wrappers.setdefault(id(module), NativeCarrierLinear(module))
                        if not name:
                            self._module_changes.append(('root', module, module.forward))
                            module.forward = wrapper.forward
                            continue
                        parent_name, _, child = name.rpartition('.')
                        parent = self.model.get_submodule(parent_name) if parent_name else self.model
                        self._module_changes.append((parent, child, module))
                        parent._modules[child] = wrapper
                    elif type(module) is nn.Conv1d and (id(weight) in selected or id(bias) in selected):
                        original = module.forward
                        module.forward = _conv1d_carrier_forward(self, module, original)
                        self._module_changes.append(('forward', module, original))
                    elif (type(module).__name__ == 'Lfm2ShortConv' and
                          id(getattr(getattr(module, 'conv', None), 'weight', None)) in selected):
                        original = module.forward
                        module.forward = _lfm_causal_carrier_forward(self, module, original)
                        self._module_changes.append(('forward', module, original))
                    elif (type(module).__name__ in _RMS_NORM_TYPES and isinstance(weight, torch.Tensor)
                          and id(weight) in selected):
                        original = module.forward
                        module.forward = _rms_norm_carrier_forward(self, module, original)
                        self._module_changes.append(('forward', module, original))
            if self.mode == 'vjp':
                for entry in self.entries.values():
                    self._gradient_hooks.append(entry['parameter'].register_hook(
                        lambda gradient, item=entry: self._audit_vjp_gradient(item, gradient)))
        except BaseException:
            self.close()
            raise
        return self

    def _audit_vjp_gradient(self, entry, gradient):
        # Auxiliary adapter values are normally FP32. Their gradients already
        # fan in without the low-precision rounding this scope protects, and
        # explicit high-precision reference arithmetic intentionally bypasses
        # the carrier edge. Audit only low-precision targets.
        if entry['parameter'].dtype not in (torch.float16, torch.bfloat16):
            return gradient
        reference = entry.get('reference_cotangent')
        if reference is not None:
            expected = reference.to(device=gradient.device, dtype=gradient.dtype)
            if not torch.equal(gradient, expected):
                raise RuntimeError(f"reference cotangent fan-in changed before cast: {entry['name']}")
            return gradient
        if not entry['count']:
            raise RuntimeError(
                f"low-precision parameter gradient arrived without a cotangent carrier: {entry['name']}"
            )
        expected = entry['accumulator'].to(device=gradient.device, dtype=gradient.dtype)
        if not torch.equal(gradient, expected):
            raise RuntimeError(f"low-precision parameter gradient included an uncarried path: {entry['name']}")
        return gradient

    @classmethod
    def for_model(cls, model, *, mode='update', directory=None, track_auxiliary=True,
                  auxiliary_target_ids=None, parameter_target_ids=None):
        parameter_target_ids = (None if parameter_target_ids is None
                                else frozenset(parameter_target_ids))
        selected, seen = [], set()
        for name, module in model.named_modules(remove_duplicate=False):
            weight = getattr(module, 'weight', None)
            bias = getattr(module, 'bias', None)
            if isinstance(module, nn.Embedding):
                if isinstance(weight, torch.Tensor) and weight.dtype in (torch.float16, torch.bfloat16) and id(weight) not in seen:
                    selected.append((name + '.weight', weight)); seen.add(id(weight))
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                for suffix, parameter in (('weight', weight), ('bias', bias)):
                    if (isinstance(parameter, torch.Tensor) and parameter.dtype in (torch.float16, torch.bfloat16)
                            and id(parameter) not in seen):
                        selected.append((name + '.' + suffix, parameter)); seen.add(id(parameter))
            elif type(module).__name__ in _RMS_NORM_TYPES and isinstance(weight, torch.Tensor):
                if weight.dtype in (torch.float16, torch.bfloat16) and id(weight) not in seen:
                    selected.append((name + '.weight', weight)); seen.add(id(weight))
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if ((name in ('experts.gate_up', 'experts.down') or
                 name.endswith(('.experts.gate_up', '.experts.down'))) and parameter.ndim == 3
                    and parameter.dtype in (torch.float16, torch.bfloat16) and id(parameter) not in seen):
                selected.append((name, parameter)); seen.add(id(parameter))
        # SharedStateModel's sparse role/override rows are registered directly
        # on the model rather than owned by nn.Embedding/nn.Linear modules.
        # They still participate in repeated embedding and readout uses, so
        # they need the same one-cast cotangent path when stored in BF16.
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if (name in {'input_rows', 'output_rows', 'extra_input_rows', 'extra_output_rows'}
                    and parameter.dtype in (torch.float16, torch.bfloat16)
                    and id(parameter) not in seen):
                selected.append((name, parameter)); seen.add(id(parameter))
        if parameter_target_ids is not None:
            selected = [(name, parameter) for name, parameter in selected
                        if id(parameter) in parameter_target_ids]
            seen.intersection_update(parameter_target_ids)
        uncovered = [(name, parameter) for name, parameter in model.named_parameters(remove_duplicate=False)
                     if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16)
                     and (parameter_target_ids is None or id(parameter) in parameter_target_ids)
                     and id(parameter) not in seen]
        if uncovered:
            names = ', '.join(name for name, _ in uncovered[:8])
            raise RuntimeError('trainable low-precision parameters lack a shared cotangent route: ' + names)
        scope = cls(selected, mode=mode, directory=directory, track_auxiliary=track_auxiliary,
                    auxiliary_target_ids=auxiliary_target_ids,
                    parameter_target_ids=parameter_target_ids)
        scope.model = model
        return scope

    def _make_entry(self, name, parameter, index):
        initial_grad=parameter.grad if parameter.is_leaf else None
        return {'name': name, 'parameter': parameter, 'count': 0,
            'path': os.path.join(self.directory, f'{index:06d}.f32'), 'accumulator': None,
            'anchor_value': None, 'anchor_backward': 0, 'initial_grad': initial_grad,
            'initial_grad_version': None if initial_grad is None else initial_grad._version}

    def _register_auxiliary(self, name, tensor):
        entry = self._auxiliary_entries.get(id(tensor))
        if entry is None:
            if not tensor.requires_grad:
                raise RuntimeError(f'cotangent auxiliary target does not require gradients: {name}')
            if not tensor.is_contiguous():
                raise RuntimeError(f'non-contiguous cotangent auxiliary target is unsupported: {name}')
            entry = self._make_entry(name, tensor, len(self.entries) + len(self._auxiliary_entries))
            self._auxiliary_entries[id(tensor)] = entry
            if self._token is not None and self.mode == 'vjp':
                self._gradient_hooks.append(tensor.register_hook(
                    lambda gradient, item=entry: self._audit_vjp_gradient(item, gradient)))
        return entry

    def register_effective_weight(self, effective, base_parameter, deltas):
        """Bind one adapter-composed weight to its shared base and delta cotangents.

        The forward remains the caller's exact effective tensor. Carrier backward
        sends the same unrounded dW to a shared base accumulator and this
        effective tensor's delta accumulator; their anchors route each derivative
        through the corresponding original autograd graph exactly once.
        """
        if self.mode != 'vjp':
            raise RuntimeError('functional effective weights require VJP cotangent mode')
        if base_parameter.dtype not in (torch.float16, torch.bfloat16):
            return
        base = self._entry_for(base_parameter)
        base_is_requested = (self.parameter_target_ids is None or
                             id(base_parameter) in self.parameter_target_ids)
        if (base is None and base_is_requested and base_parameter.requires_grad
                and base_parameter.dtype in (torch.float16, torch.bfloat16)):
            index = len(self.entries)
            base = self._make_entry('effective-base', base_parameter, index)
            self.entries[id(base_parameter)] = base
            if self._token is not None:
                self._gradient_hooks.append(base_parameter.register_hook(
                    lambda gradient, item=base: self._audit_vjp_gradient(item, gradient)))
        if not isinstance(deltas, (tuple, list)):
            raise RuntimeError('effective cotangent route requires its exact adapter-delta list')
        # Forward-only actors retain immutable coefficient tensors. They do not
        # need an auxiliary cotangent edge; a trainable base may still need one.
        delta_entries = []
        if self.track_auxiliary:
            for index, item in enumerate(deltas):
                if isinstance(item, tuple) and len(item) == 2:
                    delta, coefficient = item
                else:
                    delta, coefficient = item, None
                if (coefficient is not None and self.auxiliary_target_ids is not None
                        and id(coefficient) not in self.auxiliary_target_ids):
                    continue
                if delta.requires_grad:
                    delta_entries.append(self._register_auxiliary(f'effective-delta-{index}', delta))
        if base is None and not delta_entries:
            return
        previous = self._effective_routes.get(id(effective))
        route = (base, tuple(delta_entries))
        if previous is not None and (previous[0] is not base or
                                     len(previous[1]) != len(delta_entries) or
                                     any(left is not right for left, right in zip(previous[1], delta_entries))):
            raise RuntimeError('one effective tensor cannot bind to multiple cotangent routes')
        self._effective_routes[id(effective)] = route

    def anchor_for(self, parameter):
        entry = self._entry_for(parameter)
        if entry is None:
            return None
        if entry['anchor_value'] is None:
            entry['anchor_value'] = _ParameterCotangentAnchor.apply(entry['parameter'], self)
        return entry['anchor_value']

    def reference_parameter(self, parameter):
        """Share one differentiable FP32 cast for explicit higher-order references.

        Reference arithmetic must preserve Hessian/VJP semantics, so it cannot
        use the file-backed first-order custom edge. Reusing one ordinary cast
        node makes autograd sum every use in FP32 before the single cast back
        to a BF16 leaf.
        """
        entry = self._entry_for(parameter)
        if entry is None or parameter.dtype not in (torch.float16, torch.bfloat16):
            return parameter
        key = id(parameter)
        value = self._reference_parameters.get(key)
        if value is None:
            value = parameter.float()
            if value.requires_grad:
                def capture(gradient, item=entry):
                    item['reference_cotangent'] = gradient.detach()
                    return gradient
                value.register_hook(capture)
            self._reference_parameters[key] = value
        return value

    def widen_parameter(self, parameter):
        """Expose a direct low-precision parameter as an FP32 cotangent edge.

        This is for compact native sparse tables used directly by the model,
        outside a module-owned projection. Forward values are the exact FP32
        promotion of stored values. Backward accumulates in the shared scope,
        and the common anchor returns/commits one cast after all uses.
        """
        entry = self._entry_for(parameter)
        if entry is None:
            return parameter
        anchor = self.anchor_for(parameter)
        if anchor is None:
            # Frozen serving tables remain ordinary values even when another
            # scope happens to contain their storage.
            return parameter
        return _WidenedParameterCotangent.apply(parameter, anchor, self)

    def anchors_for_weight(self, weight):
        _, route = self._effective_route_for(weight)
        if route is None:
            anchor = self.anchor_for(weight)
            return () if anchor is None else (anchor,)
        base, deltas = route
        anchors = []
        if base is not None:
            anchors.append(self.anchor_for(base['parameter']))
        anchors.extend(self.anchor_for(delta['parameter']) for delta in deltas)
        return tuple(anchors)

    def _effective_route_for(self, weight):
        """Resolve an effective adapter bank through exact tensor view ancestry."""
        current = weight
        visited = set()
        while isinstance(current, torch.Tensor) and id(current) not in visited:
            visited.add(id(current))
            route = self._effective_routes.get(id(current))
            if route is not None:
                return current, route
            current = getattr(current, '_base', None)
        return None, None

    @staticmethod
    def _route_parameter_view(source, effective, view):
        """Map an effective-bank view to the matching base/delta storage view."""
        if tuple(source.shape) != tuple(effective.shape) or tuple(source.stride()) != tuple(effective.stride()):
            raise RuntimeError('effective cotangent source bank geometry differs from its exact replacement')
        relative_offset = view.storage_offset() - effective.storage_offset()
        target_offset = source.storage_offset() + relative_offset
        last = target_offset + sum((size - 1) * stride for size, stride in zip(view.shape, view.stride()))
        if target_offset < source.storage_offset() or last >= source.storage_offset() + source.numel():
            raise RuntimeError('effective cotangent view escapes its source bank')
        return source.as_strided(view.shape, view.stride(), target_offset)

    def _entry_for(self, parameter):
        entry = self.entries.get(id(parameter))
        if entry is not None:
            return entry
        entry = self._auxiliary_entries.get(id(parameter))
        if entry is not None:
            return entry
        storage = parameter.untyped_storage().data_ptr()
        matches = [candidate for entries in (self.entries.values(), self._auxiliary_entries.values())
                   for candidate in entries
                   if candidate['parameter'].untyped_storage().data_ptr() == storage]
        if len(matches) > 1:
            raise RuntimeError('cotangent view aliases multiple optimizer or auxiliary targets')
        return matches[0] if matches else None

    def new_backward(self):
        """Start another forward/backward microbatch in one optimizer update."""
        if self.mode != 'update' or self._closed:
            raise RuntimeError('new cotangent backward is only valid in an open optimizer-update scope')
        for entry in self.entries.values():
            entry['anchor_value'] = None
        for entry in self._auxiliary_entries.values():
            entry['anchor_value'] = None

    def add(self, parameter, contribution):
        entry = self._entry_for(parameter)
        if entry is None:
            raise RuntimeError('custom low-precision weight path has no registered cotangent anchor')
        if contribution.shape != parameter.shape:
            raise RuntimeError(f"cotangent shape mismatch for {entry['name']}")
        if contribution.dtype != torch.float32:
            contribution = contribution.float()
        if entry['accumulator'] is None:
            # A sparse file starts zero-filled; mapping does not reserve a second
            # full-model resident CPU allocation.
            entry['accumulator'] = torch.from_file(entry['path'], shared=True,
                size=entry['parameter'].numel(), dtype=torch.float32).reshape(entry['parameter'].shape)
        target = entry['accumulator']
        if parameter is not entry['parameter']:
            offset = parameter.storage_offset() - entry['parameter'].storage_offset()
            last = offset + sum((size - 1) * stride for size, stride in zip(parameter.shape, parameter.stride()))
            if offset < 0 or last >= entry['parameter'].numel():
                raise RuntimeError(f"cotangent view escapes registered optimizer parameter {entry['name']}")
            target = target.as_strided(parameter.shape, parameter.stride(), offset)
        # Avoid a second full-size CPU FP32 allocation while transferring a
        # device contribution into the file-backed accumulator. Chunk across
        # leading rows; native matrices and expert views keep each row bounded.
        row_bytes = max(4, contribution[0].numel() * 4)
        rows = max(1, _COTANGENT_TRANSFER_CHUNK_BYTES // row_bytes)
        for start in range(0, contribution.shape[0], rows):
            stop = min(contribution.shape[0], start + rows)
            piece = contribution[start:stop].detach().to(device='cpu', dtype=torch.float32)
            target[start:stop].add_(piece)
        entry['count'] += 1

    def add_embedding_rows(self, parameter, indices, contribution, *, padding_idx=None,
                           scale_grad_by_freq=False):
        """Add indexed embedding dW rows directly to the file-backed accumulator."""
        entry = self._entry_for(parameter)
        if entry is None:
            raise RuntimeError('embedding cotangent has no registered parameter anchor')
        if parameter.ndim != 2 or contribution.shape[:-1] != indices.shape or contribution.shape[-1] != parameter.shape[1]:
            raise RuntimeError(f"embedding cotangent shape mismatch for {entry['name']}")
        if entry['accumulator'] is None:
            entry['accumulator'] = torch.from_file(entry['path'], shared=True,
                size=entry['parameter'].numel(), dtype=torch.float32).reshape(entry['parameter'].shape)
        flat_indices = indices.detach().reshape(-1)
        flat_contribution = contribution.detach().reshape(-1, parameter.shape[1])
        row_bytes = max(4, parameter.shape[1] * 4)
        rows = max(1, _COTANGENT_TRANSFER_CHUNK_BYTES // row_bytes)
        target = entry['accumulator']
        if scale_grad_by_freq:
            fd, temporary_path = tempfile.mkstemp(prefix='.embedding-frequency-', dir=self.directory)
            os.close(fd)
            temporary = torch.from_file(temporary_path, shared=True, size=parameter.numel(),
                                       dtype=torch.float32).reshape(parameter.shape)
            counts = torch.zeros(parameter.shape[0], dtype=torch.int64)
            try:
                for start in range(0, flat_indices.numel(), rows):
                    stop = min(flat_indices.numel(), start + rows)
                    ids = flat_indices[start:stop].to(device='cpu', dtype=torch.long)
                    values = flat_contribution[start:stop].to(device='cpu', dtype=torch.float32)
                    if padding_idx is not None:
                        keep = ids != int(padding_idx)
                        ids, values = ids[keep], values[keep]
                    if ids.numel():
                        temporary.index_add_(0, ids, values)
                        counts.index_add_(0, ids, torch.ones_like(ids))
                used_rows = torch.nonzero(counts, as_tuple=False).flatten()
                for start in range(0, used_rows.numel(), rows):
                    ids = used_rows[start:start + rows]
                    values = temporary.index_select(0, ids) / counts[ids, None].to(torch.float32)
                    target.index_add_(0, ids, values)
            finally:
                del temporary
                os.remove(temporary_path)
        else:
            for start in range(0, flat_indices.numel(), rows):
                stop = min(flat_indices.numel(), start + rows)
                ids = flat_indices[start:stop].to(device='cpu', dtype=torch.long)
                values = flat_contribution[start:stop].to(device='cpu', dtype=torch.float32)
                if padding_idx is not None:
                    keep = ids != int(padding_idx)
                    ids, values = ids[keep], values[keep]
                if ids.numel():
                    target.index_add_(0, ids, values)
        entry['count'] += 1

    def finish(self):
        if self._closed:
            raise RuntimeError('BF16 parameter cotangent scope was already finalized')
        # A regular autograd edge is evidence that a use bypassed the shared
        # accumulator and would otherwise be rounded during BF16 fan-in.
        all_entries = [*self.entries.values(), *self._auxiliary_entries.values()]
        bypasses = []
        for entry in self.entries.values():
            parameter = entry['parameter']
            gradient = parameter.grad
            changed = (gradient is not entry['initial_grad'] or
                       gradient is not None and gradient._version != entry['initial_grad_version'])
            if not changed:
                continue
            reference = entry.get('reference_cotangent')
            if self.mode == 'vjp' and entry['anchor_backward'] > 0:
                continue
            if self.mode == 'update' and reference is not None:
                expected = reference.to(device=parameter.device, dtype=parameter.dtype)
                if entry['initial_grad'] is not None:
                    expected = entry['initial_grad'] + expected
                if gradient is not None and torch.equal(gradient, expected):
                    continue
            bypasses.append(entry['name'])
        if bypasses:
            counts = {entry['name']: entry['count'] for entry in self.entries.values()
                      if entry['name'] in bypasses}
            raise RuntimeError('low-precision parameter gradient bypassed cotangent carrier: '
                               + ', '.join(f'{name} (carried={counts[name]})' for name in bypasses[:8]))
        mixed_reference = [entry['name'] for entry in self.entries.values()
                           if self.mode == 'vjp' and entry.get('reference_cotangent') is not None
                           and entry['count']]
        if mixed_reference:
            raise RuntimeError('reference VJP mixed differentiable and file-backed cotangents: '
                               + ', '.join(mixed_reference[:8]))
        missing = [entry['name'] for entry in all_entries
                   if entry['count'] and entry['anchor_backward'] == 0 and self.mode == 'vjp']
        if missing:
            raise RuntimeError('cotangent carrier anchor did not complete backward: ' + ', '.join(missing[:8])
                               )
        if self.mode == 'update':
            for entry in self.entries.values():
                reference = entry.get('reference_cotangent')
                if reference is not None:
                    combined = reference
                    if entry['count']:
                        combined = combined + entry['accumulator'].to(device=reference.device,
                                                                      dtype=torch.float32)
                    entry['parameter'].grad = combined.to(device=entry['parameter'].device,
                                                          dtype=entry['parameter'].dtype)
                elif entry['count']:
                    entry['parameter'].grad = entry['accumulator'].to(device=entry['parameter'].device,
                                                                       dtype=entry['parameter'].dtype)
                else:
                    continue
        self._closed = True

    def close(self):
        if self._closed and self._token is None and not self.entries and not self._auxiliary_entries:
            return
        if self._token is not None:
            _COTANGENT_SCOPE.reset(self._token)
            self._token = None
        for parent, child, original in reversed(self._module_changes):
            if parent == 'forward':
                child.forward = original
            elif parent == 'root':
                child.forward = original
            else:
                parent._modules[child] = original
        self._module_changes.clear()
        for hook in self._gradient_hooks:
            hook.remove()
        self._gradient_hooks.clear()
        # Custom autograd contexts keep the scope alive until their graph is
        # collected. Drop file-backed tensors and route/anchor references now
        # so unlinking the files also releases their mmap mappings promptly.
        for entry in (*self.entries.values(), *self._auxiliary_entries.values()):
            entry['accumulator'] = None
            entry['anchor_value'] = None
        self._effective_routes.clear()
        self._reference_parameters.clear()
        self._auxiliary_entries.clear()
        self.entries.clear()
        self._wrappers.clear()
        self._closed = True
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _ParameterCotangentAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, scope):
        ctx.parameter, ctx.scope = parameter, scope
        return torch.zeros((), device=parameter.device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, sentinel_gradient):
        entry = ctx.scope._entry_for(ctx.parameter)
        if not entry['count'] or entry['accumulator'] is None:
            raise RuntimeError(f"cotangent anchor for {entry['name']} has no accumulated weight derivative")
        entry['anchor_backward'] += 1
        if ctx.scope.mode == 'update':
            return None, None
        return entry['accumulator'].to(device=ctx.parameter.device, dtype=torch.float32), None


class _WidenedParameterCotangent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, anchor, scope):
        ctx.parameter, ctx.scope = parameter, scope
        # The scalar is exactly zero; retaining it only supplies the common
        # once-cast optimizer/VJP edge. Do not preserve an autograd edge to the
        # original low-precision table here.
        return parameter.detach().float() + anchor

    @staticmethod
    def backward(ctx, gradient):
        ctx.scope.add(ctx.parameter, gradient)
        return None, gradient.new_ones((), dtype=torch.float32), None


class _TorchStoredRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, module, original_forward, *anchors):
        ctx.save_for_backward(x, weight)
        ctx.eps, ctx.anchors = float(eps), anchors
        ctx.parameter, ctx.scope = weight, active_cotangent_scope() if anchors else None
        # Run the source module implementation, preserving its exact backend
        # dispatch and forward casts. The custom Function owns only dW fan-in.
        return original_forward(x)

    @staticmethod
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        xf = x.float()
        inverse = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + ctx.eps)
        normalized = (xf * inverse).to(x.dtype)
        dw = (gradient.float() * normalized.float()).sum(dim=tuple(range(gradient.ndim - 1)))
        dy = (gradient * weight).to(x.dtype).float()
        dx = (dy - xf * inverse.square() * (dy * xf).mean(dim=-1, keepdim=True)) * inverse
        anchor_grads = (None,) * len(ctx.anchors)
        if ctx.anchors:
            _scope_add(ctx.parameter, dw, ctx.anchors, ctx.scope)
            dw = None
            anchor_grads = tuple(gradient.new_ones((), dtype=torch.float32) for _ in ctx.anchors)
        else:
            dw = dw.to(weight.dtype)
        return dx.to(x.dtype), dw, None, None, None, *anchor_grads


class _TorchStoredConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, module, original_forward, weight_anchor_count, *anchors):
        if module.padding_mode != 'zeros':
            raise RuntimeError('low-precision Conv1d cotangent carrier currently requires zero padding mode')
        ctx.save_for_backward(x, weight)
        ctx.bias, ctx.module = bias is not None, module
        ctx.weight_anchor_count = int(weight_anchor_count)
        ctx.weight_anchors = anchors[:ctx.weight_anchor_count]
        ctx.bias_anchors = anchors[ctx.weight_anchor_count:]
        ctx.weight, ctx.bias = weight, bias
        ctx.scope = active_cotangent_scope() if anchors else None
        return original_forward(x)

    @staticmethod
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        module = ctx.module
        dx = (torch.nn.grad.conv1d_input(x.shape, weight, gradient, module.stride,
                                         module.padding, module.dilation, module.groups)
              if ctx.needs_input_grad[0] else None)
        dw = (torch.nn.grad.conv1d_weight(x.float(), weight.shape, gradient.float(), module.stride,
                                          module.padding, module.dilation, module.groups)
              if ctx.needs_input_grad[1] else None)
        db = gradient.float().sum(dim=(0, 2)) if ctx.bias and ctx.needs_input_grad[2] else None
        anchor_grads = [None] * len(ctx.weight_anchors)
        if ctx.weight_anchors and dw is not None:
            _scope_add(ctx.weight, dw, ctx.weight_anchors, ctx.scope)
            dw = None
            anchor_grads = [gradient.new_ones((), dtype=torch.float32) for _ in ctx.weight_anchors]
        elif dw is not None:
            dw = dw.to(weight.dtype)
        bias_grads = [None] * len(ctx.bias_anchors)
        if ctx.bias_anchors and db is not None:
            _scope_add(ctx.bias, db, ctx.bias_anchors, ctx.scope)
            db = None
            bias_grads = [gradient.new_ones((), dtype=torch.float32) for _ in ctx.bias_anchors]
        elif db is not None:
            db = db.to(ctx.bias.dtype)
        return dx, dw, db, None, None, None, *anchor_grads, *bias_grads


class _TorchStoredFunctionalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, weight_anchor_count, *anchors):
        ctx.save_for_backward(x, weight)
        ctx.bias_value = bias
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.weight_anchor_count = int(weight_anchor_count)
        ctx.weight_anchors = anchors[:ctx.weight_anchor_count]
        ctx.bias_anchors = anchors[ctx.weight_anchor_count:]
        ctx.scope_weight = weight if ctx.weight_anchors else None
        ctx.scope_bias = bias if ctx.bias_anchors else None
        ctx.scope = active_cotangent_scope() if anchors else None
        return F.conv1d(x, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        dx = (torch.nn.grad.conv1d_input(x.shape, weight, gradient, ctx.stride,
                                         ctx.padding, ctx.dilation, ctx.groups)
              if ctx.needs_input_grad[0] else None)
        dw = (torch.nn.grad.conv1d_weight(x.float(), weight.shape, gradient.float(),
                                          ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
              if ctx.needs_input_grad[1] else None)
        db = (gradient.float().sum(dim=(0, 2))
              if ctx.bias_value is not None and ctx.needs_input_grad[2] else None)
        weight_grads = [None] * len(ctx.weight_anchors)
        if ctx.weight_anchors and dw is not None:
            _scope_add(ctx.scope_weight, dw, ctx.weight_anchors, ctx.scope)
            dw = None
            weight_grads = [gradient.new_ones((), dtype=torch.float32) for _ in ctx.weight_anchors]
        elif dw is not None:
            dw = dw.to(weight.dtype)
        bias_grads = [None] * len(ctx.bias_anchors)
        if ctx.bias_anchors and db is not None:
            _scope_add(ctx.scope_bias, db, ctx.bias_anchors, ctx.scope)
            db = None
            bias_grads = [gradient.new_ones((), dtype=torch.float32) for _ in ctx.bias_anchors]
        elif db is not None:
            db = db.to(ctx.bias_value.dtype)
        return dx, dw, db, None, None, None, None, None, *weight_grads, *bias_grads


class _TorchStoredCausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, operation, activation, kwargs, scope, anchor_count, *anchors):
        ctx.save_for_backward(x, weight, *(() if bias is None else (bias,)))
        ctx.has_bias = bias is not None
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.operation, ctx.activation, ctx.kwargs, ctx.scope = operation, activation, kwargs, scope
        ctx.anchor_count, ctx.anchors = int(anchor_count), anchors
        return operation(x, weight, bias, activation=activation, **kwargs)

    @staticmethod
    def backward(ctx, gradient):
        saved = ctx.saved_tensors
        x, weight = saved[:2]
        bias = saved[2] if ctx.has_bias else None
        input_values = [x.detach().requires_grad_(ctx.needs_input_grad[0]),
                        weight.detach().requires_grad_(ctx.needs_input_grad[1])]
        if ctx.has_bias:
            input_values.append(bias.detach().requires_grad_(ctx.needs_input_grad[2]))
        with torch.enable_grad():
            output = ctx.operation(input_values[0], input_values[1],
                                   input_values[2] if ctx.has_bias else None,
                                   activation=ctx.activation, **ctx.kwargs)
            requested = [value for value, needed in zip(input_values,
                         [ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]])
                         if needed]
            gradients = torch.autograd.grad(output, requested, gradient, allow_unused=True)
        iterator = iter(gradients)
        dx = next(iterator) if ctx.needs_input_grad[0] else None
        dw_native = next(iterator) if ctx.needs_input_grad[1] else None
        db_native = (next(iterator) if ctx.needs_input_grad[2] else None) if ctx.has_bias else None
        dw = dw_native.float() if dw_native is not None else None
        db = db_native.float() if db_native is not None else None
        if dw_native is not None and ctx.anchors:
            # Causal conv kernels may return BF16 parameter gradients. Recompute
            # just this linear-in-weight contribution with FP32 operands before
            # the common cross-invocation fan-in.
            x32 = x.detach().float().requires_grad_(True)
            w32 = weight.detach().float().requires_grad_(True)
            b32 = bias.detach().float().requires_grad_(True) if ctx.has_bias else None
            with torch.enable_grad():
                output32 = ctx.operation(x32, w32, b32, activation=ctx.activation, **ctx.kwargs)
                variables = (x32, w32, b32) if ctx.has_bias else (x32, w32)
                gradients32 = torch.autograd.grad(output32, variables, gradient.float(), allow_unused=True)
            dw = gradients32[1]
            db = gradients32[2] if ctx.has_bias else db
        weight_anchors = ctx.anchors[:ctx.anchor_count]
        bias_anchors = ctx.anchors[ctx.anchor_count:]
        weight_anchor_grads = [None] * len(weight_anchors)
        if weight_anchors and dw is not None:
            _scope_add(weight, dw, weight_anchors, ctx.scope)
            dw = None
            weight_anchor_grads = [gradient.new_ones((), dtype=torch.float32) for _ in weight_anchors]
        elif dw is not None:
            dw = dw.to(weight.dtype)
        bias_anchor_grads = [None] * len(bias_anchors)
        if bias_anchors and db is not None:
            _scope_add(bias, db, bias_anchors, ctx.scope)
            db = None
            bias_anchor_grads = [gradient.new_ones((), dtype=torch.float32) for _ in bias_anchors]
        elif db is not None:
            db = db.to(bias.dtype)
        return dx, dw, db, None, None, None, None, None, *weight_anchor_grads, *bias_anchor_grads


def active_cotangent_scope():
    return _COTANGENT_SCOPE.get()


def _scope_anchors(weight):
    scope = active_cotangent_scope()
    return () if scope is None else scope.anchors_for_weight(weight)


def widen_direct_parameter(parameter):
    """Route a direct sparse-table use through the active FP32 accumulator."""
    scope = active_cotangent_scope()
    if scope is None or parameter.dtype not in (torch.float16, torch.bfloat16):
        return parameter
    from ..common.native_higher_order import active_reference_arithmetic
    if active_reference_arithmetic() is not None:
        return scope.reference_parameter(parameter)
    return scope.widen_parameter(parameter)


def _scope_anchor(weight):
    """Compatibility helper for code that has only one unparametrized weight edge."""
    anchors = _scope_anchors(weight)
    return anchors[0] if anchors else None


def _scope_add(weight, contribution, anchors, scope=None):
    scope = scope or active_cotangent_scope()
    if scope is None or not anchors:
        return False
    route_owner, route = scope._effective_route_for(weight)
    if route is None:
        scope.add(weight, contribution)
    else:
        base, deltas = route
        if base is not None:
            source_view = scope._route_parameter_view(base['parameter'], route_owner, weight)
            scope.add(source_view, contribution)
        for delta in deltas:
            source_view = scope._route_parameter_view(delta['parameter'], route_owner, weight)
            scope.add(source_view, contribution)
    return True


class _TorchStoredFiniteConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, histories, weight, bias, c, weight_anchor_count, *anchors):
        out = (histories.contiguous().float() * weight[:, 0].T.float()[None, None]).sum(2)
        if bias is not None:
            out = out + bias.float()
        stored_out = out.to(c.dtype)
        ctx.save_for_backward(histories, weight, c, stored_out)
        ctx.has_bias = bias is not None
        ctx.bias_dtype = None if bias is None else bias.dtype
        ctx.weight_anchor_count = int(weight_anchor_count)
        ctx.weight_anchors = anchors[:ctx.weight_anchor_count]
        ctx.bias_anchors = anchors[ctx.weight_anchor_count:]
        ctx.scope_weight = weight if ctx.weight_anchors else None
        ctx.scope_bias = bias if ctx.bias_anchors else None
        ctx.scope = active_cotangent_scope() if anchors else None
        return c * stored_out

    @staticmethod
    def backward(ctx, gradient):
        histories, weight, c, stored_out = ctx.saved_tensors
        effective_gradient = (gradient * c).float()
        dh = (effective_gradient.unsqueeze(2) * weight[:, 0].T.float()[None, None]).to(histories.dtype)
        dc = gradient * stored_out
        if ctx.weight_anchors:
            dw = (effective_gradient.unsqueeze(2) * histories.detach().float()).sum(dim=(0, 1)).T[:, None, :]
            _scope_add(ctx.scope_weight, dw, ctx.weight_anchors, ctx.scope)
            d_weight = None
            weight_grads = [gradient.new_ones((), dtype=torch.float32) for _ in ctx.weight_anchors]
        else:
            d_weight = (effective_gradient.unsqueeze(2) * histories.detach().float()).sum(dim=(0, 1)).T[:, None, :].to(weight.dtype)
            weight_grads = []
        if ctx.has_bias and ctx.bias_anchors:
            db = effective_gradient.sum(dim=(0, 1))
            _scope_add(ctx.scope_bias, db, ctx.bias_anchors, ctx.scope)
            d_bias = None
            bias_grads = [gradient.new_ones((), dtype=torch.float32) for _ in ctx.bias_anchors]
        else:
            d_bias = effective_gradient.sum(dim=(0, 1)).to(ctx.bias_dtype) if ctx.has_bias else None
            bias_grads = []
        return dh, d_weight, d_bias, dc, None, *weight_grads, *bias_grads


class _TorchStoredEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight, padding_idx, norm_type, scale_grad_by_freq, anchor_count, *anchors):
        ctx.save_for_backward(indices)
        ctx.weight, ctx.padding_idx = weight, padding_idx
        ctx.norm_type, ctx.scale_grad_by_freq = float(norm_type), bool(scale_grad_by_freq)
        ctx.anchor_count, ctx.anchors = int(anchor_count), anchors
        ctx.scope = active_cotangent_scope()
        return F.embedding(indices, weight, padding_idx, None, norm_type, scale_grad_by_freq, False)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        ctx.scope.add_embedding_rows(ctx.weight, indices, gradient,
                                     padding_idx=ctx.padding_idx,
                                     scale_grad_by_freq=ctx.scale_grad_by_freq)
        anchor_gradients = tuple(gradient.new_ones((), dtype=torch.float32) for _ in ctx.anchors)
        return None, None, None, None, None, None, *anchor_gradients


class _RoundedWeightCarrier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, previous, delta, storage_dtype):
        # Preserve the existing ordered BF16 cast/add semantics exactly.
        return (previous.to(storage_dtype) + delta.to(storage_dtype)).float()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient, None


def rounded_weight_carrier(previous, delta, storage_dtype):
    return _RoundedWeightCarrier.apply(previous, delta, storage_dtype)


class _TorchStoredLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, storage_dtype, weight_anchor_count, *anchors):
        stored = weight.to(storage_dtype)
        ctx.save_for_backward(x, stored)
        ctx.bias = bias is not None
        ctx.weight_dtype = weight.dtype
        ctx.weight_anchor_count = int(weight_anchor_count)
        ctx.weight_anchors = anchors[:ctx.weight_anchor_count]
        ctx.bias_anchors = anchors[ctx.weight_anchor_count:]
        ctx.scope_weight = weight if ctx.weight_anchors else None
        ctx.scope_bias = bias if ctx.bias_anchors else None
        ctx.scope = active_cotangent_scope() if anchors else None
        return F.linear(x, stored, bias)

    @staticmethod
    def backward(ctx, grad):
        x, stored = ctx.saved_tensors
        flat_grad, flat_x = grad.reshape(-1, grad.shape[-1]), x.reshape(-1, x.shape[-1])
        dx = (grad @ stored) if ctx.needs_input_grad[0] else None
        dw = None
        if ctx.needs_input_grad[1]:
            if grad.is_cuda:
                from ..maple.fused_moe import invariant_weight_gradient
                dw = invariant_weight_gradient(flat_x, flat_grad)
            else:
                dw = flat_grad.float().T @ flat_x.float()
        db = flat_grad.float().sum(0) if ctx.bias and ctx.needs_input_grad[2] else None
        weight_anchor_grads = (None,) * len(ctx.weight_anchors)
        if ctx.weight_anchors and dw is not None:
            _scope_add(ctx.scope_weight, dw, ctx.weight_anchors, ctx.scope)
            dw = None
            weight_anchor_grads = tuple(grad.new_ones((), dtype=torch.float32) for _ in ctx.weight_anchors)
        elif dw is not None:
            # An adapter's rounded effective weight is FP32 even though this
            # projection stores BF16 operands for its forward. Preserve the
            # FP32 derivative through that effective tensor so its coefficient
            # basis receives an unrounded adjoint; ordinary BF16 parameters
            # still return their native BF16 gradient here.
            dw = dw.to(ctx.weight_dtype)
        bias_anchor_grads = (None,) * len(ctx.bias_anchors)
        if ctx.bias_anchors and db is not None:
            _scope_add(ctx.scope_bias, db, ctx.bias_anchors, ctx.scope)
            db = None
            bias_anchor_grads = tuple(grad.new_ones((), dtype=torch.float32) for _ in ctx.bias_anchors)
        elif db is not None:
            db = db.to(grad.dtype)
        return dx, dw, db, None, None, *weight_anchor_grads, *bias_anchor_grads


def stored_linear(x, weight, bias=None, *, storage_dtype, fixed=False, precision='ieee'):
    from ..common.native_higher_order import active_reference_arithmetic,reference_linear
    if active_reference_arithmetic() is not None:
        return reference_linear(x,weight,bias)
    weight_anchors = _scope_anchors(weight)
    bias_anchors = _scope_anchors(bias) if bias is not None else ()
    if fixed:
        anchors = weight_anchors
        if x.is_cuda and x.dtype in (torch.float32, torch.bfloat16):
            from ..maple.fused_moe import invariant_linear
            shape = x.shape
            out = invariant_linear(x.reshape(-1, shape[-1]), weight, precision=precision,
                                   forward_storage_dtype=storage_dtype, cotangent_anchors=anchors).reshape(*shape[:-1], weight.shape[0])
        else:
            out = _TorchStoredLinear.apply(x, weight, None, storage_dtype, len(anchors), *anchors)
        return out if bias is None else out + bias
    return _TorchStoredLinear.apply(x, weight, bias, storage_dtype, len(weight_anchors),
                                    *weight_anchors, *bias_anchors)


def stored_conv1d(x, weight, bias=None, *, stride=1, padding=0, dilation=1, groups=1):
    """Run the exact torch Conv1d forward with shared FP32 weight/bias fan-in."""
    anchors = _scope_anchors(weight)
    bias_anchors = _scope_anchors(bias) if bias is not None else ()
    if not anchors and not bias_anchors:
        return F.conv1d(x, weight, bias, stride, padding, dilation, groups)
    return _TorchStoredFunctionalConv1d.apply(x, weight, bias, stride, padding, dilation,
                                               groups, len(anchors), *anchors, *bias_anchors)


def stored_causal_conv1d(operation, x, weight, bias=None, *, activation=None, **kwargs):
    """Use a fused causal-convolution forward while widening its parameter fan-in."""
    anchors = _scope_anchors(weight)
    bias_anchors = _scope_anchors(bias) if bias is not None else ()
    if not anchors and not bias_anchors:
        return operation(x, weight, bias, activation=activation, **kwargs)
    return _TorchStoredCausalConv1d.apply(x, weight, bias, operation, activation, kwargs,
                                          active_cotangent_scope(), len(anchors),
                                          *anchors, *bias_anchors)


def stored_finite_convolution(histories, weight, bias, c):
    """Keep the native finite-history forward and widen its convolution dW."""
    from ..common.native_higher_order import active_reference_arithmetic
    if active_reference_arithmetic() is not None:
        scope = active_cotangent_scope()
        if scope is not None:
            weight = scope.reference_parameter(weight)
            if bias is not None:
                bias = scope.reference_parameter(bias)
        out = (histories.contiguous().float() * weight[:, 0].T.float()[None, None]).sum(2)
        if bias is not None:
            out = out + bias.float()
        return c * out.to(c.dtype)
    anchors = _scope_anchors(weight)
    bias_anchors = _scope_anchors(bias) if bias is not None else ()
    if not anchors and not bias_anchors:
        out = (histories.contiguous().float() * weight[:, 0].T.float()[None, None]).sum(2)
        if bias is not None:
            out = out + bias.float()
        return c * out.to(c.dtype)
    return _TorchStoredFiniteConvolution.apply(histories, weight, bias, c, len(anchors),
                                               *anchors, *bias_anchors)


def stored_embedding(indices, weight, *, padding_idx=None, max_norm=None, norm_type=2.0,
                     scale_grad_by_freq=False, sparse=False):
    """Keep exact lookup values and share low-precision embedding row gradients."""
    from ..common.native_higher_order import active_reference_arithmetic
    scope = active_cotangent_scope()
    if active_reference_arithmetic() is not None and scope is not None:
        if max_norm is not None:
            raise RuntimeError('reference low-precision embedding does not support max_norm mutation')
        if sparse:
            raise RuntimeError('reference low-precision embedding requires the dense gradient contract')
        weight = scope.reference_parameter(weight)
        return F.embedding(indices, weight, padding_idx, None, norm_type, False, False)
    anchors = _scope_anchors(weight)
    if not anchors:
        return F.embedding(indices, weight, padding_idx, max_norm, norm_type,
                           scale_grad_by_freq, sparse)
    if max_norm is not None:
        raise RuntimeError('trainable low-precision embedding cotangents do not support max_norm mutation')
    if sparse:
        raise RuntimeError('trainable low-precision embedding cotangents require the dense optimizer gradient contract')
    return _TorchStoredEmbedding.apply(indices, weight, padding_idx, norm_type,
                                       scale_grad_by_freq, len(anchors), *anchors)



class NativeCarrierLinear(nn.Linear):
    """Preserve native fixed projections and ordinary HF Linear arithmetic."""
    def __init__(self, original):
        # Preserve hooks/config/ties without resetting or cloning parameters.
        self.__dict__ = original.__dict__.copy()
        self._parameters = original._parameters.copy()
        self._buffers = original._buffers.copy()
        self._modules = original._modules.copy()
        self.forward_storage_dtype = original.weight.dtype

    def forward(self, x):
        return stored_linear(x, self.weight, self.bias, storage_dtype=self.forward_storage_dtype)

    def native_fixed_forward(self, x):
        return stored_linear(x, self.weight, self.bias, storage_dtype=self.forward_storage_dtype,
                             fixed=True, precision=getattr(self, 'native_linear_precision', 'ieee'))


@contextmanager
def carrier_linear_modules(model, selected_weights):
    """Temporarily wrap every Linear alias of selected BF16/FP16 matrices."""
    selected = {id(weight) for weight in selected_weights if weight.dtype != torch.float32}
    changes, wrappers = [], {}
    try:
        for name, module in list(model.named_modules(remove_duplicate=False)):
            if not name or not hasattr(module, 'weight') or id(module.weight) not in selected:
                continue
            # Outer native functional scopes may already have installed the
            # carrier before an invocation actor enters. Reuse that wrapper;
            # nesting it would both lose the original module metadata and make
            # the exact-Linear guard reject our own wrapper type.
            if isinstance(module, NativeCarrierLinear):
                continue
            if type(module) is not nn.Linear:
                raise ValueError(f'FP32 native coefficient carriers require exact Linear layer matrices or dense expert banks; {name} is {type(module).__module__}.{type(module).__qualname__}')
            wrapper = wrappers.setdefault(id(module), NativeCarrierLinear(module))
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            changes.append((parent, child, module))
            parent._modules[child] = wrapper
        yield
    finally:
        for parent, child, original in reversed(changes):
            parent._modules[child] = original
