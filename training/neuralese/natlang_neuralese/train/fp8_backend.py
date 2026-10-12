"""Opt-in Transformer Engine FP8 linear backend for the shared direct SFT trainer.

This module changes only eligible ``nn.Linear`` modules for the duration of
direct training. It keeps the original BF16 Parameter objects, and can restore
the original HF modules for evaluation and serving-style checks.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
from typing import Any

import torch
from torch import nn


SCHEMA = 'natlang.neuralese-te-fp8-training/1'
BACKEND_VERSION = 'te-dense-lfm-v1'
ROW_MULTIPLE = 16


def _eligible_paths(model: nn.Module) -> list[str]:
    """Return exact LFM dense-module paths, excluding tied embeddings/readout."""
    model_type = getattr(getattr(model, 'config', None), 'model_type', None)
    if model_type != 'lfm2':
        raise ValueError(f'TE FP8 backend supports LFM2 direct training only; got {model_type!r}')
    paths = []
    for name, module in model.named_modules():
        parts = name.split('.')
        if len(parts) != 5 or parts[0] != 'model' or parts[1] != 'layers':
            continue
        family, layer = parts[3], parts[4]
        if family == 'feed_forward' and layer in {'w1', 'w2', 'w3'}:
            paths.append(name)
        elif family == 'self_attn' and layer in {'q_proj', 'k_proj', 'v_proj', 'out_proj'}:
            paths.append(name)
        elif family == 'conv' and layer in {'in_proj', 'out_proj'}:
            paths.append(name)
    return paths


def runtime_identity(device: str | torch.device) -> dict[str, Any]:
    """Read the runtime fields required to bind this local experimental backend."""
    if not torch.cuda.is_available():
        raise RuntimeError('TE FP8 training requires CUDA')
    device = torch.device(device)
    if device.type != 'cuda':
        raise RuntimeError(f'TE FP8 training requires a CUDA device, got {device}')
    properties = torch.cuda.get_device_properties(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability < (8, 9):
        raise RuntimeError(f'TE FP8 training requires SM 8.9 or newer, got SM {capability[0]}.{capability[1]}')
    try:
        te_version = importlib.metadata.version('transformer_engine')
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError('TE FP8 training requested but Transformer Engine is unavailable') from exc
    if not te_version.startswith('2.19.'):
        raise RuntimeError(f'this experimental backend currently supports TE 2.19.x, found {te_version}')
    return {
        'backend': BACKEND_VERSION,
        'transformer_engine': te_version,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device_name': properties.name,
        'compute_capability': list(capability),
        'device': str(device),
    }


class FP8LinearAdapter:
    """Replace eligible LFM dense linears by TE linears while aliasing parameters."""

    def __init__(self, model: nn.Module, *, device: str | torch.device, recipe: str = 'delayed',
                 amax_history_len: int = 16):
        if recipe not in {'delayed', 'current'}:
            raise ValueError('FP8 recipe must be delayed or current')
        if type(amax_history_len) is not int or amax_history_len < 1:
            raise ValueError('FP8 amax history length must be a positive integer')
        self.model = model
        self.recipe_name = recipe
        self.amax_history_len = amax_history_len
        self.runtime = runtime_identity(device)
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format
        self.te = te
        self.recipe = (DelayedScaling(amax_history_len=amax_history_len, fp8_format=Format.HYBRID)
                       if recipe == 'delayed' else Float8CurrentScaling(fp8_format=Format.HYBRID))
        self.originals: dict[str, tuple[nn.Module, str, nn.Module]] = {}
        self.linears: dict[str, nn.Module] = {}
        self.row_counts = {'calls': 0, 'rows': 0, 'padded_rows': 0}
        self.has_executed = False
        self._convert()

    def _convert(self):
        paths = _eligible_paths(self.model)
        if not paths:
            raise ValueError('no eligible LFM dense linears found for TE FP8')
        expected_parameters = {name: id(parameter) for name, parameter in self.model.named_parameters()}
        device_index = torch.device(self.runtime['device']).index
        rng_devices = [torch.cuda.current_device() if device_index is None else device_index]
        # TE Linear initialization is discarded below. Fork both CPU and CUDA RNG so conversion cannot shift
        # projection initialization, dropout, sampling, or the training RNG stream.
        with torch.random.fork_rng(devices=rng_devices):
            for path in paths:
                parent, attribute, old = self._parent(path)
                if not isinstance(old, nn.Linear):
                    raise TypeError(f'eligible LFM module {path} is {type(old).__name__}, not nn.Linear')
                if old.weight.dtype != torch.bfloat16:
                    raise TypeError(f'eligible LFM parameter {path}.weight must be BF16, got {old.weight.dtype}')
                if old.in_features % ROW_MULTIPLE or old.out_features % ROW_MULTIPLE:
                    raise ValueError(f'eligible TE dimensions must be multiples of {ROW_MULTIPLE}: {path} '
                                     f'{old.in_features}x{old.out_features}')
                replacement = make_te_linear_preserving_rng(old, self.te.Linear)
                replacement.train(old.training)
                setattr(parent, attribute, replacement)
                self.originals[path] = (parent, attribute, old)
                self.linears[path] = replacement
        actual_parameters = {name: id(parameter) for name, parameter in self.model.named_parameters()}
        if actual_parameters != expected_parameters:
            raise RuntimeError('TE conversion changed parameter names, identities, or duplicate registration')
        return paths

    def _parent(self, path: str):
        parts = path.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        return parent, parts[-1], getattr(parent, parts[-1])

    @contextlib.contextmanager
    def autocast(self):
        with self.te.autocast(enabled=True, recipe=self.recipe):
            yield

    def checkpoint(self, function, *args, **kwargs):
        # PortBackbone's regular torch checkpoint contract includes preserve_rng_state. TE's checkpoint owns RNG
        # tracking through its FP8 recompute contexts; accept only the existing supported preserving mode.
        preserve_rng_state = kwargs.pop('preserve_rng_state', True)
        if preserve_rng_state is not True:
            raise ValueError('TE FP8 checkpoint recomputation requires RNG preservation')
        return self.te.checkpoint(function, *args, **kwargs)

    @contextlib.contextmanager
    def native_modules(self):
        """Temporarily use ordinary HF linears (same Parameter objects) for BF16 evaluation."""
        try:
            for path, (parent, attribute, original) in self.originals.items():
                setattr(parent, attribute, original)
            yield
        finally:
            for path, (parent, attribute, original) in self.originals.items():
                setattr(parent, attribute, self.linears[path])

    def reset_row_counts(self):
        self.row_counts = {'calls': 0, 'rows': 0, 'padded_rows': 0}
        for module in self.linears.values():
            module._fp8_row_stats = {'calls': 0, 'rows': 0, 'padded_rows': 0}

    def extra_state(self) -> dict[str, Any]:
        return {
            'schema': SCHEMA,
            'backend': BACKEND_VERSION,
            'recipe': self.recipe_name,
            'amax_history_len': self.amax_history_len,
            'runtime': self.runtime,
            'module_paths': sorted(self.linears),
            'te_extra_state': {name: module.get_extra_state() for name, module in self.linears.items()},
        }

    def restore_extra_state(self, state: dict[str, Any]):
        if not isinstance(state, dict) or state.get('schema') != SCHEMA:
            raise ValueError('checkpoint has no supported TE FP8 training state')
        if (state.get('backend') != BACKEND_VERSION or state.get('recipe') != self.recipe_name or
                state.get('amax_history_len') != self.amax_history_len or
                state.get('module_paths') != sorted(self.linears)):
            raise ValueError('checkpoint TE FP8 module/recipe state differs')
        if state.get('runtime') != self.runtime:
            raise ValueError('checkpoint TE FP8 runtime differs')
        extras = state.get('te_extra_state') or {}
        if set(extras) != set(self.linears):
            raise ValueError('checkpoint TE FP8 scale-state module paths differ')
        for name, module in self.linears.items():
            module.set_extra_state(extras[name])

    def inventory(self) -> dict[str, Any]:
        parameters = list(self.model.named_parameters())
        eligible_parameter_ids = {id(parameter) for path, module in self.linears.items()
                                  for parameter in module.parameters(recurse=False)}
        fp8_extra = {}
        runtime_cache = {}
        for name, module in self.linears.items():
            state = module.get_extra_state()
            fp8_extra[name] = _tensor_inventory(state)
            runtime_cache[name] = _object_tensor_inventory(module)
        return {
            'schema': SCHEMA,
            'runtime': self.runtime,
            'recipe': self.recipe_name,
            'save_original_input': False,
            'row_padding_multiple': ROW_MULTIPLE,
            'eligible_linear_count': len(self.linears),
            'eligible_parameter_bytes': sum(parameter.numel() * parameter.element_size()
                                            for _, parameter in parameters
                                            if id(parameter) in eligible_parameter_ids),
            'named_parameter_count': len(parameters),
            'fp8_extra_state_tensors': fp8_extra,
            'runtime_module_tensor_inventory': runtime_cache,
            'row_counts': dict(self.row_counts),
        }

    def collect_row_counts(self) -> dict[str, int]:
        result = {key: sum(module._fp8_row_stats[key] for module in self.linears.values())
                  for key in ('calls', 'rows', 'padded_rows')}
        self.row_counts = result
        self.has_executed = self.has_executed or result['calls'] > 0
        return result

    def cached_weight_bytes_estimate(self) -> int:
        # On Ada TE may keep separate rowwise/columnwise FP8 copies for forward and wgrad.
        return sum(module.weight.numel() * 2 for module in self.linears.values())

    def configuration(self) -> dict[str, Any]:
        return {'enabled': True, 'backend': BACKEND_VERSION, 'recipe': self.recipe_name,
                'amax_history_len': self.amax_history_len, 'save_original_input': False,
                'runtime': self.runtime, 'eligible_linear_count': len(self.linears),
                'eligible_module_paths': sorted(self.linears)}


EXPERT_SCHEMA = 'natlang.neuralese-fp8-experts-training/1'
EXPERT_BACKEND_VERSION = 'triton-grouped-experts-v1'


class FP8ExpertAdapter:
    """The MoE-expert path of the FP8 backend (``--fp8-training triton-experts``; Maple family: Mellum): the dense
    experts' grouped projections run as one Triton FP8 launch each (maple/expert_gemm.py: E4M3 operands with per-row
    current scaling, FP32 accumulation; input gradient through FP8, weight gradient BF16 in the parameter layout).
    Same interface as ``FP8LinearAdapter``: FP8 inside ``autocast()`` (the training update, its recompute included),
    BF16 under ``native_modules()`` (evaluation). Current scaling keeps no state, so the checkpoint envelope holds only
    the configuration. Composes with the QAT precision points: FP8 computes the already fake-quantized weights."""

    def __init__(self, model: nn.Module, *, device: str | torch.device):
        from ..maple.model import DenseExperts
        if getattr(getattr(model, 'config', None), 'model_type', None) not in {'mellum', 'maple'}:
            raise ValueError('the FP8 expert backend supports Maple-family (Mellum) dense experts only')
        if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
            raise RuntimeError('FP8 expert training requires CUDA')
        self.model = model
        self.experts = [name for name, module in model.named_modules() if isinstance(module, DenseExperts)]
        if not self.experts:
            raise ValueError('no dense experts found for the FP8 expert backend')
        capability = tuple(torch.cuda.get_device_capability(torch.device(device)))
        if capability < (8, 9):
            raise RuntimeError(f'FP8 expert training requires SM 8.9 or newer, got SM {capability}')
        self.runtime = {'backend': EXPERT_BACKEND_VERSION, 'torch': torch.__version__,
                        'compute_capability': list(capability), 'device': str(device)}
        self.has_executed = False
        self.row_counts = {'calls': 0, 'rows': 0, 'padded_rows': 0}

    @contextlib.contextmanager
    def autocast(self):
        from ..maple.model import DENSE_MOE
        previous = DENSE_MOE.get('fp8')
        DENSE_MOE['fp8'] = 'grouped'
        try:
            yield
        finally:
            DENSE_MOE['fp8'] = previous
        self.has_executed = True

    def checkpoint(self, function, *args, **kwargs):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(function, *args, use_reentrant=False, **kwargs)

    @contextlib.contextmanager
    def native_modules(self):
        from ..maple.model import DENSE_MOE
        previous = DENSE_MOE.get('fp8')
        DENSE_MOE['fp8'] = None
        try:
            yield
        finally:
            DENSE_MOE['fp8'] = previous

    def reset_row_counts(self):
        pass

    def collect_row_counts(self) -> dict[str, int]:
        return dict(self.row_counts)

    def extra_state(self) -> dict[str, Any]:
        return {'schema': EXPERT_SCHEMA, 'backend': EXPERT_BACKEND_VERSION, 'module_paths': sorted(self.experts)}

    def restore_extra_state(self, state: dict[str, Any]):
        if not isinstance(state, dict) or state.get('schema') != EXPERT_SCHEMA or \
                state.get('module_paths') != sorted(self.experts):
            raise ValueError('checkpoint has no matching FP8 expert training state')

    def inventory(self) -> dict[str, Any]:
        return {'schema': EXPERT_SCHEMA, 'runtime': self.runtime, 'recipe': 'current-rowwise',
                'expert_modules': len(self.experts)}

    def cached_weight_bytes_estimate(self) -> int:
        return 0  # FP8 weights live only within one layer's projection (saved for its backward)

    def configuration(self) -> dict[str, Any]:
        return {'enabled': True, 'backend': EXPERT_BACKEND_VERSION, 'recipe': 'current-rowwise',
                'runtime': self.runtime, 'expert_modules': sorted(self.experts)}


def make_backend(kind: str, model: nn.Module, *, device, recipe: str = 'delayed', amax_history_len: int = 16):
    """The direct trainer's FP8 backend for ``--fp8-training``: ``te-fp8`` (TE dense linears, LFM) or
    ``triton-experts`` (Triton grouped FP8 expert GEMMs, Mellum)."""
    if kind == 'te-fp8':
        return FP8LinearAdapter(model, device=device, recipe=recipe, amax_history_len=amax_history_len)
    if kind == 'triton-experts':
        return FP8ExpertAdapter(model, device=device)
    raise ValueError(f'unknown FP8 training backend {kind!r}')


def _tensor_inventory(value: Any) -> dict[str, Any]:
    tensors = []

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    grouped = {}
    for tensor in tensors:
        key = str(tensor.dtype)
        grouped[key] = grouped.get(key, 0) + tensor.numel() * tensor.element_size()
    return {'tensors': len(tensors), 'bytes_by_dtype': grouped}


def _object_tensor_inventory(value: Any) -> dict[str, Any]:
    """Inspect TE module-owned tensors without retaining them or double-counting storage aliases."""
    tensors = {}
    seen = set()

    def visit(item, depth=0):
        if depth > 4:
            return
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            tensors.setdefault(key, (item.dtype, storage.nbytes(), tuple(item.shape)))
            return
        identity = id(item)
        if identity in seen:
            return
        if isinstance(item, dict):
            seen.add(identity)
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            seen.add(identity)
            for child in item:
                visit(child, depth + 1)
        elif hasattr(item, '__dict__') and not isinstance(item, (nn.Module, type)):
            seen.add(identity)
            for child in vars(item).values():
                visit(child, depth + 1)

    for name, item in vars(value).items():
        if name not in {'_parameters', '_buffers', '_modules'}:
            visit(item)
    by_dtype = {}
    for dtype, nbytes, _ in tensors.values():
        key = str(dtype)
        by_dtype[key] = by_dtype.get(key, 0) + nbytes
    return {'unique_tensor_storages': len(tensors), 'unique_storage_bytes_by_dtype': by_dtype,
            'shapes': [list(shape) for _, _, shape in tensors.values()]}


def make_te_linear(original: nn.Linear, te_linear_type):
    """Construct one padded TE layer while preserving the source Parameter objects exactly."""
    replacement = _PaddedTELinear(
        te_linear_type, original.in_features, original.out_features, bias=original.bias is not None,
        params_dtype=original.weight.dtype, device=original.weight.device,
        save_original_input=False, return_bias=False)
    replacement.weight = original.weight
    replacement.weight.requires_grad_(original.weight.requires_grad)
    if original.bias is not None:
        replacement.bias = original.bias
        replacement.bias.requires_grad_(original.bias.requires_grad)
    replacement.train(original.training)
    if replacement.weight is not original.weight or (original.bias is not None and replacement.bias is not original.bias):
        raise RuntimeError('TE layer did not preserve the original Parameter objects')
    return replacement


def make_te_linear_preserving_rng(original: nn.Linear, te_linear_type, *, cuda_devices=()):
    """Call TE's initializer without advancing the caller's CPU/CUDA random streams."""
    with torch.random.fork_rng(devices=list(cuda_devices)):
        return make_te_linear(original, te_linear_type)


class _PaddedTELinear:
    """Factory for an nn.Module subclass that pads only flattened M to TE's row multiple."""

    def __new__(cls, base, *args, **kwargs):
        class Padded(base):
            def __init__(self, *inner_args, **inner_kwargs):
                super().__init__(*inner_args, **inner_kwargs)
                self._fp8_row_stats = {'calls': 0, 'rows': 0, 'padded_rows': 0}

            def forward(self, value):
                if value.shape[-1] != self.in_features:
                    raise ValueError(f'TE linear expected last dimension {self.in_features}, got {value.shape[-1]}')
                shape = value.shape
                flat = value.reshape(-1, self.in_features)
                rows = flat.shape[0]
                padded = (-rows) % ROW_MULTIPLE
                if padded:
                    flat = torch.cat((flat, flat.new_zeros((padded, self.in_features))), dim=0)
                output = super().forward(flat)
                if padded:
                    output = output[:rows]
                self._fp8_row_stats['calls'] += 1
                self._fp8_row_stats['rows'] += rows
                self._fp8_row_stats['padded_rows'] += padded
                return output.reshape(*shape[:-1], self.out_features)

        return Padded(*args, **kwargs)


def saved_tensor_inventory(run_forward_backward) -> dict[str, Any]:
    """Measure tensors saved by autograd for one diagnostic update without retaining extra references.

    The hook returns each original tensor unchanged. Grouping by storage identity distinguishes an FP8 saved
    activation from a BF16 duplicate, while keeping the diagnostic's storage behavior intact.
    """
    storages = {}
    shapes = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storages.setdefault(key, (str(tensor.dtype), storage.nbytes()))
        shape_key = (str(tensor.dtype), tuple(tensor.shape))
        shapes[shape_key] = shapes.get(shape_key, 0) + 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = run_forward_backward()
    by_dtype = {}
    for dtype, nbytes in storages.values():
        by_dtype[dtype] = by_dtype.get(dtype, 0) + nbytes
    return {'unique_storage_count': len(storages), 'unique_storage_bytes_by_dtype': by_dtype,
            'saved_tensor_shape_counts': [
                {'dtype': dtype, 'shape': list(shape), 'count': count}
                for (dtype, shape), count in sorted(shapes.items(), key=lambda item: (item[0][0], item[0][1]))],
            'storage_semantics': 'identity-pack-hooks; storage aliases counted once'}


class SavedTensorAudit:
    """Non-retaining autograd saved-tensor audit for one explicitly selected diagnostic update."""
    def __init__(self):
        self.storages = {}
        self.shapes = {}
        self._scope = None

    def _pack(self, tensor):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        self.storages.setdefault(key, (str(tensor.dtype), storage.nbytes()))
        shape_key = (str(tensor.dtype), tuple(tensor.shape))
        self.shapes[shape_key] = self.shapes.get(shape_key, 0) + 1
        return tensor

    def __enter__(self):
        self._scope = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda tensor: tensor)
        self._scope.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._scope.__exit__(exc_type, exc, traceback)

    def report(self):
        by_dtype = {}
        for dtype, nbytes in self.storages.values():
            by_dtype[dtype] = by_dtype.get(dtype, 0) + nbytes
        return {'unique_storage_count': len(self.storages), 'unique_storage_bytes_by_dtype': by_dtype,
                'saved_tensor_shape_counts': [
                    {'dtype': dtype, 'shape': list(shape), 'count': count}
                    for (dtype, shape), count in sorted(self.shapes.items(),
                                                        key=lambda item: (item[0][0], item[0][1]))],
                'storage_semantics': 'identity-pack-hooks; storage aliases counted once'}
