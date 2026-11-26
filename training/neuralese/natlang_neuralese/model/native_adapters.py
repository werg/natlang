"""Identity-bound XS modelLoRA actors for functional native ports.

Frozen bases and coefficients use the existing named tensor artifact lifecycle.
Application uses torch.func's exact parameter substitution, not projected
codes or legacy module hooks. The operation includes any checkpoint
recomputation and backward.
"""
from __future__ import annotations

import contextvars
import math
import torch
from torch import nn

from ..common.hashing import canonical_json_sha256_hex
from .native_artifact import NativeTensorArtifact
from .native_adapter_math import singular_bases

SCHEMA = 'natlang.native-adapter/1'
_ACTIVE = contextvars.ContextVar('natlang_native_adapter_execution', default=())
# Whether the running native operation substitutes live (trainable) adapter or bank-basis leaves: their values can change
# under unchanged artifact IDs, so value caches keyed by IDs (inference weights, prompt prefixes) must not serve them.
_LIVE_LEAVES = contextvars.ContextVar('natlang_native_live_adapter_leaves', default=False)


def live_adapter_leaves() -> bool:
    return _LIVE_LEAVES.get()


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


def _target_names(engine, names):
    available = target_parameters(engine.shared_model)
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or any(name not in available for name in names)):
        raise ValueError('parameter_names must uniquely select actual named native layer matrices')
    return available


def create_canonical_basis(engine, options, owner=None):
    """One maximum-rank basis per target; nested adapters slice its leading columns (§5.3)."""
    with engine.native_execution_lock:
        if execution_binding(engine):
            raise ValueError('create native adapter bases from the unadapted live model')
        max_rank, names = options.get('max_rank'), options.get('parameter_names')
        if type(max_rank) is not int or max_rank < 1:
            raise ValueError('canonical adapter basis requires a positive integer max_rank')
        available = _target_names(engine, names)
        spec = {'schema': SCHEMA, 'kind': 'canonical-basis', 'max_rank': max_rank,
                'parameter_names': names, 'bases': 'weight-svd-sign-canonical-v2'}
        tensors = {}
        with torch.inference_mode(False), torch.no_grad():
            for name in names:
                weight = available[name]
                if max_rank > min(weight.shape[-2:]):
                    raise ValueError(f'canonical basis rank exceeds target matrix dimensions: {name}')
                left, right = singular_bases(weight.detach().float(), max_rank)
                tensors['left.'+name], tensors['right.'+name] = left.cpu(), right.cpu()
        return engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'adapter-basis', engine.native_identity, tensors, provenance={'adapter_spec': spec}), owner)


def _create_adapter(engine, options, owner=None):
    kind = options.get('kind', 'xs')
    rank, seed = options.get('rank', 8), options.get('seed', 0)
    if kind not in ('xs', 'free-lora') or type(rank) is not int or rank < 1 or type(seed) is not int:
        raise ValueError('native modelLoRA requires kind xs or free-lora, a positive rank and integer seed')
    if options.get('native_identity') not in (None, engine.native_identity):
        raise ValueError('native adapter creation is bound to another exact live model')
    names = options.get('parameter_names')
    available = _target_names(engine, names)
    if kind == 'free-lora':
        # Free factors: (alpha/r) B A with B zero and seeded A, so the delta starts at zero and B gets gradients.
        alpha = options.get('alpha', rank)
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('free-lora alpha must be positive and finite')
        spec = {'schema': SCHEMA, 'kind': kind, 'rank': rank, 'seed': seed, 'alpha': float(alpha),
                'parameter_names': names, 'bases': 'free-lora-v1'}
        generator, factors = torch.Generator().manual_seed(seed), {}
        for name in names:
            *leading, rows, cols = available[name].shape
            if rank > min(rows, cols):
                raise ValueError(f'adapter rank exceeds target matrix dimensions: {name}')
            factors['lora_A.'+name] = torch.randn((*leading, rank, cols), generator=generator, dtype=torch.float32) / math.sqrt(cols)
            factors['lora_B.'+name] = torch.zeros((*leading, rows, rank), dtype=torch.float32)
        return engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'adapter-coefficients', engine.native_identity, factors, None, {'adapter_spec': spec}), owner)
    if options.get('basis_id') is not None:
        # A nested slice of one shared canonical basis: stack members then sum in that basis exactly.
        record = engine.store.get_native_tensor_artifact(options['basis_id'])
        basis_spec = (record.artifact.provenance or {}).get('adapter_spec') if record else None
        # A learned bank basis (tune/learned/bank.py) is sliced the same way under its own declared method, so
        # tuner nodes and learned-optimizer mixtures share one coordinate system.
        if (record is None or record.artifact.kind != 'adapter-basis' or record.artifact.native_identity != engine.native_identity
                or not isinstance(basis_spec, dict) or basis_spec.get('kind') not in ('canonical-basis', 'bank-basis')):
            raise ValueError('nested adapters require an exact canonical or bank basis artifact of this live model')
        if rank > basis_spec['max_rank'] or not set(names) <= set(basis_spec['parameter_names']):
            raise ValueError('nested adapter rank/targets must fit its canonical basis')
        spec = {'schema': SCHEMA, 'kind': kind, 'rank': rank, 'seed': seed, 'parameter_names': names,
                'bases': 'bank-basis-slice-v1' if basis_spec['kind'] == 'bank-basis' else 'nested-canonical-slice-v1',
                'basis_spec': basis_spec}
        coefficients = {name: torch.zeros((*available[name].shape[:-2], rank * rank), dtype=torch.float32) for name in names}
        return engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'adapter-coefficients', engine.native_identity, coefficients, record.id, {'adapter_spec': spec}), owner)
    spec = {'schema': SCHEMA, 'kind': kind, 'rank': rank, 'seed': seed,
            'parameter_names': names, 'bases': 'weight-svd-sign-canonical-v2'}
    tensors, coefficients = {}, {}
    with torch.inference_mode(False), torch.no_grad():
        for index, name in enumerate(names):
            weight = available[name]
            if rank > min(weight.shape[-2:]):
                raise ValueError(f'adapter rank exceeds target matrix dimensions: {name}')
            left, right = singular_bases(weight.detach().float(), rank)
            tensors['left.'+name], tensors['right.'+name] = left.cpu(), right.cpu()
            shape = (*weight.shape[:-2], rank * rank)
            coefficients[name] = torch.zeros(shape, dtype=torch.float32)
    basis = engine.store.put_native_tensor_artifact(NativeTensorArtifact(
        'adapter-basis', engine.native_identity, tensors, provenance={'adapter_spec': spec}), owner)
    return engine.store.put_native_tensor_artifact(NativeTensorArtifact(
        'adapter-coefficients', engine.native_identity, coefficients, basis.id,
        {'adapter_spec': spec}), owner)


def adapter_tensor_keys(spec):
    """Coefficient tensor names of a spec: one core per target, or B/A factors for free LoRA."""
    names = spec['parameter_names']
    return {prefix+name for name in names for prefix in ('lora_A.', 'lora_B.')} if spec['kind'] == 'free-lora' else set(names)


def resolve_adapters(engine, entries, leaves=None):
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError('adapters must be a list of native coefficient artifact IDs or id/scale objects')
    resolved = []
    available = target_shapes(engine)
    for entry in entries:
        if isinstance(entry, str):
            adapter_id, scale = entry, 1.
        elif isinstance(entry, dict) and set(entry) <= {'id', 'scale', 'stop_gradient'}:
            adapter_id, scale = entry.get('id'), entry.get('scale', 1.)
        else:
            raise ValueError('native adapters require an exact id and optional finite scale; projected codes are unsupported')
        if not isinstance(adapter_id, str) or type(scale) not in (int, float) or not math.isfinite(scale):
            raise ValueError('native adapter ID and scale are invalid')
        if isinstance(entry, dict) and 'stop_gradient' in entry and entry['stop_gradient'] is not True:
            raise ValueError('native adapter stop_gradient must be true or absent')
        record = engine.store.get_native_tensor_artifact(adapter_id)
        if record is None:
            raise ValueError(f'unknown native adapter {adapter_id}; restore its coefficients and basis first')
        adapter = record.artifact.validate()
        spec = (adapter.provenance or {}).get('adapter_spec')
        if (adapter.kind != 'adapter-coefficients' or adapter.native_identity != engine.native_identity
                or not isinstance(spec, dict) or spec.get('schema') != SCHEMA):
            raise ValueError('native adapter/basis must bind the exact live checkpoint, architecture and declared specification')
        names, rank, kind, method = (spec.get(key) for key in ('parameter_names', 'rank', 'kind', 'bases'))
        free = kind == 'free-lora'
        if free:
            alpha = spec.get('alpha')
            if (method != 'free-lora-v1' or adapter.source_artifact_id is not None
                    or type(alpha) is not float or not math.isfinite(alpha) or alpha <= 0):
                raise ValueError('free-lora adapters carry their own factors, a declared alpha and no basis')
            basis_tensors = {}
        else:
            basis_record = engine.store.get_native_tensor_artifact(adapter.source_artifact_id or '')
            if basis_record is None:
                raise ValueError('native adapter is missing its exact frozen basis artifact')
            basis = basis_record.artifact.validate()
            # A learned bank basis (tune/learned/bank.py) is its own declared method, never a weight-SVD basis.
            bank = method == 'bank-basis-slice-v1'
            nested = method == 'nested-canonical-slice-v1' or bank
            expected_spec = spec.get('basis_spec') if nested else spec
            if (basis.kind != 'adapter-basis' or basis.native_identity != engine.native_identity
                    or expected_spec != (basis.provenance or {}).get('adapter_spec')
                    or method not in ('weight-svd-sign-canonical-v2', 'nested-canonical-slice-v1', 'bank-basis-slice-v1') or kind != 'xs'
                    or (nested and not bank and (expected_spec.get('kind') != 'canonical-basis'
                                    or expected_spec.get('bases') != 'weight-svd-sign-canonical-v2'))
                    or (bank and (expected_spec.get('kind') != 'bank-basis'
                                  or expected_spec.get('bases') != 'bank-shared-basis-v1'))):
                raise ValueError('native adapter/basis must bind the exact live checkpoint, architecture and declared specification')
        if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names) or set(adapter.tensors) != adapter_tensor_keys(spec)
                or type(rank) is not int or rank < 1 or kind not in ('xs', 'free-lora')):
            raise ValueError('invalid native adapter specification/inventory')
        if not free:
            basis_names = expected_spec['parameter_names'] if nested else names
            if nested and (not set(names) <= set(basis_names) or rank > expected_spec['max_rank']):
                raise ValueError('nested adapter exceeds its canonical basis')
            expected_basis = {prefix+name for name in basis_names for prefix in ('left.', 'right.')}
            if set(basis.tensors) != expected_basis:
                raise ValueError('native adapter basis inventory differs from its specification')
            # Exact leading-slice views: nested members share one basis and sum in it.
            source_tensors = basis.tensors
            if bank and leaves and adapter.source_artifact_id in leaves:
                # Live trainable bank bases: the same geometry, checked below like the stored tensors.
                source_tensors = leaves[adapter.source_artifact_id]
                if set(source_tensors) != expected_basis:
                    raise ValueError('live bank basis leaves differ from the basis inventory')
            basis_tensors = {prefix+name: source_tensors[prefix+name][..., :rank] for name in names for prefix in ('left.', 'right.')}
        values = leaves.get(adapter_id, adapter.tensors) if leaves else adapter.tensors
        if isinstance(entry, dict) and entry.get('stop_gradient'):
            values = {name: value.detach() for name, value in values.items()}
        if (any(value.dtype != torch.float32 for value in basis_tensors.values())
                or any(value.dtype != torch.float32 for value in values.values())):
            raise ValueError('native adapter bases and coefficient masters must be FP32')
        if set(values) != adapter_tensor_keys(spec):
            raise ValueError('native adapter coefficient inventory changed')
        for name in names:
            if name not in available:
                raise ValueError(f'native adapter target is not a supported layer matrix: {name}')
            shape = available[name]
            if free:
                ok = (tuple(values['lora_B.'+name].shape) == (*shape[:-1], rank)
                      and tuple(values['lora_A.'+name].shape) == (*shape[:-2], rank, shape[-1]))
            else:
                ok = (tuple(values[name].shape) == (*shape[:-2], rank * rank)
                      and tuple(basis_tensors['left.'+name].shape) == (*shape[:-2], shape[-2], rank)
                      and tuple(basis_tensors['right.'+name].shape) == (*shape[:-2], shape[-1], rank))
            if rank > min(shape[-2:]) or not ok:
                raise ValueError(f'native adapter tensor geometry differs for {name}')
        resolved.append((record.id, float(scale), spec, values, basis_tensors))
    return resolved


def target_shapes(engine):
    """Adaptable matrix name -> shape: the live model's, or a serving front's declared ones (no resident model)."""
    declared = getattr(engine, 'native_target_shapes', None)
    if declared is not None:
        return declared
    return {name: tuple(value.shape) for name, value in target_parameters(engine.shared_model).items()}


def lowrank_factors(resolved):
    """Each adapted matrix's summed delta as one FP32 low-rank pair (B [out, r], A [r, in]), delta = B @ A: the
    deltas `_effective_replacements` adds, with ranks of stacked adapters concatenated."""
    factors = {}
    for _, scale, spec, coefficients, bases in resolved:
        for name in spec['parameter_names']:
            if spec['kind'] == 'free-lora':
                factor_b = coefficients['lora_B.'+name].float() * (spec['alpha'] / spec['rank'] * scale)
                factor_a = coefficients['lora_A.'+name].float()
            else:
                left, right = bases['left.'+name].float(), bases['right.'+name].float()
                core = coefficients[name].float().reshape(*coefficients[name].shape[:-1], spec['rank'], spec['rank'])
                factor_b, factor_a = (left @ core) * scale, right.transpose(-2, -1)
            if factor_b.ndim != 2:
                raise ValueError(f'low-rank serving supports 2-D layer matrices, not {name}')
            if name in factors:
                previous_b, previous_a = factors[name]
                factor_b, factor_a = torch.cat([previous_b, factor_b], 1), torch.cat([previous_a, factor_a], 0)
            factors[name] = (factor_b, factor_a)
    return factors


def factored_training_enabled():
    """``NATLANG_FACTORED_LORA`` (default on; ``0`` opts out): a differentiated adapted operation applies each target's
    low-rank delta as ``x A^T B^T`` beside the unchanged base projection instead of substituting a materialized
    ``W + B A`` (an FP32 delta per target plus its gradient: 21.6 GB CUDA on one 13k-token LFM3 row over every
    target). The serving engines (SGLang slots, llama.cpp LoRA) apply adapters factored too."""
    import os
    return os.environ.get('NATLANG_FACTORED_LORA', '1') != '0'


def factored_factors(resolved):
    """``{target: (B, A)}`` (differentiable in the adapter leaves) when a grad-enabled operation can run factored,
    else None (opted out, reference arithmetic, nothing adapted, or a 3-D expert bank target)."""
    if not resolved or not torch.is_grad_enabled() or not factored_training_enabled():
        return None
    from ..common.native_higher_order import active_reference_arithmetic
    if active_reference_arithmetic() is not None:
        return None
    try:
        return lowrank_factors(resolved)
    except ValueError:
        return None


def inference_factors(resolved, parameters):
    """``{target: (B, A)}`` (FP32, on each weight's device) for forward-only serving of an adapter stack, or None
    (``NATLANG_FACTORED_LORA=0`` opts out, or a 3-D expert bank target), in which case serving merges ``W + delta`` in
    the weight's storage dtype. Merging rounded the delta onto the BF16 weight grid while training applied it factored
    in FP32 (``factored_factors``), so served greedy decodes of a trained stack left the trained function (2026-11-25,
    compiled sites: the fork's FP32 LoRA, which applies factored, agreed with training longer than the merged native
    reference did)."""
    if not factored_training_enabled():
        return None
    try:
        factors = lowrank_factors(resolved)
    except ValueError:
        return None
    out = {}
    for name, (factor_b, factor_a) in factors.items():
        device = parameters[name].device
        out[name] = (factor_b.to(device=device, dtype=torch.float32), factor_a.to(device=device, dtype=torch.float32))
    return out


_FP8_E4M3_MAX = 448.0


def fp8_base_supported(device=None) -> bool:
    """Whether ``device`` runs the row-wise scaled FP8 E4M3 GEMMs of ``fp8_frozen_linear`` (Blackwell, compute capability
    >= 12.0: G4 sm_120 and GB10 sm_121; row-wise ``torch._scaled_mm`` with BF16 output)."""
    if not torch.cuda.is_available() or not hasattr(torch, '_scaled_mm'):
        return False
    try:
        dev = torch.device(device) if device is not None else torch.device('cuda', torch.cuda.current_device())
        if dev.type != 'cuda':
            return False
        major, _minor = torch.cuda.get_device_capability(dev)
        return major >= 12
    except Exception:
        return False


def fp8_base_default(device=None) -> bool:
    """The FP8 frozen-base default: OFF. Opt in with ``NATLANG_FP8_BASE=1`` (or ``fp8_base=True``).

    Measured 2026-11-22 on the GB10 (DECISIONS "LoRA core fixes"): E4M3 carries 3 mantissa bits, so even with per-token
    activation/gradient scales and per-channel weight scales the 1.7B model's FP8-vs-BF16 log-probabilities differ by
    far more than the BF16-vs-FP32 envelope the plan requires (lfm3opti.md §6 item 3). The BF16 base is the default
    until an FP8 recipe meets that bar."""
    import os
    return os.environ.get('NATLANG_FP8_BASE', '0') == '1' and fp8_base_supported(device)


# Frozen weights are quantized once: (data_ptr, shape, stride, version) -> (W q (N, K), per-row scale (1, N) for the
# forward; W^T q (K, N), per-row scale (1, K) for the input gradient). Looked up inside the custom ops, which Dynamo
# treats as opaque, so the cache never enters a traced graph.
_FP8_WEIGHTS: dict = {}


def _rowwise_fp8(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row E4M3 quantization: one scale per token (activations, output gradients) or per output channel
    (weights). (M, K) -> (q (M, K) float8_e4m3fn, scale (M, 1) fp32)."""
    scale = x2d.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / _FP8_E4M3_MAX
    q = (x2d.float() / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def fp8_weight(weight: torch.Tensor):
    """The cached per-channel FP8 quantization of a frozen base ``weight`` (computed once per weight)."""
    key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()), weight._version, weight.device)
    hit = _FP8_WEIGHTS.get(key)
    if hit is None:
        with torch.no_grad():
            w = weight.detach()
            wq, ws = _rowwise_fp8(w.contiguous())                 # (N, K), (N, 1): per output channel
            wtq, wts = _rowwise_fp8(w.t().contiguous())           # (K, N), (K, 1): per input channel
        hit = _FP8_WEIGHTS[key] = (wq, ws.reshape(1, -1).contiguous(), wtq, wts.reshape(1, -1).contiguous())
    return hit


def clear_fp8_weights() -> None:
    _FP8_WEIGHTS.clear()


def _scaled_mm_rows(a_q, a_scale, b_q_rowmajor, b_scale_row, out_dtype):
    """``a (M, K) @ b (N, K)^T`` with row-wise scales, rows padded to a multiple of 16 (``_scaled_mm`` alignment)."""
    rows = a_q.shape[0]
    pad = (-rows) % 16
    if pad:
        a_q = torch.nn.functional.pad(a_q.view(torch.uint8), (0, 0, 0, pad)).view(torch.float8_e4m3fn)
        a_scale = torch.nn.functional.pad(a_scale, (0, 0, 0, pad), value=1.)
    out = torch._scaled_mm(a_q, b_q_rowmajor.t(), scale_a=a_scale, scale_b=b_scale_row, out_dtype=out_dtype)
    return out[:rows] if pad else out


@torch.library.custom_op('natlang::fp8_frozen_linear', mutates_args=())
def _fp8_frozen_linear_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    wq, ws, _wtq, _wts = fp8_weight(weight)
    x2d = x.reshape(-1, x.shape[-1])
    xq, xs = _rowwise_fp8(x2d)
    return _scaled_mm_rows(xq, xs, wq, ws, x.dtype).reshape(*x.shape[:-1], weight.shape[0])


@_fp8_frozen_linear_op.register_fake
def _(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.custom_op('natlang::fp8_frozen_linear_dx', mutates_args=())
def _fp8_frozen_linear_dx_op(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _wq, _ws, wtq, wts = fp8_weight(weight)
    g2d = grad.reshape(-1, grad.shape[-1])
    gq, gs = _rowwise_fp8(g2d)                                  # per token: packed documents stay independent
    return _scaled_mm_rows(gq, gs, wtq, wts, grad.dtype).reshape(*grad.shape[:-1], weight.shape[1])


@_fp8_frozen_linear_dx_op.register_fake
def _(grad, weight):
    return grad.new_empty((*grad.shape[:-1], weight.shape[1]))


def _fp8_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[1])


def _fp8_backward(ctx, grad):
    (weight,) = ctx.saved_tensors
    return _fp8_frozen_linear_dx_op(grad.to(weight.dtype).contiguous(), weight), None


torch.library.register_autograd('natlang::fp8_frozen_linear', _fp8_backward, setup_context=_fp8_setup)


def fp8_frozen_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Frozen base projection ``x W^T`` in FP8 E4M3 (per-token activation and output-gradient scales, per-channel weight
    scales, the weight quantized once; a compile-safe ``torch.library`` op) when eligible, else ``F.linear``. The weight
    gets no gradient: only frozen bases take this path."""
    if (x.is_cuda and weight.is_cuda and x.dtype == torch.bfloat16 and weight.dtype == x.dtype
            and not weight.requires_grad and weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
            and fp8_base_supported(x.device)):
        out = _fp8_frozen_linear_op(x, weight)
        return out if bias is None else out + bias.to(out.dtype)
    return torch.nn.functional.linear(x, weight, bias)


@torch.library.custom_op('natlang::frozen_linear_f32', mutates_args=())
def _frozen_linear_f32_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1])
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        out = torch.mm(x2d, weight.t(), out_dtype=torch.float32)     # tensor cores, FP32 accumulator kept
    else:
        out = torch.mm(x2d.float(), weight.float().t())
    return out.reshape(*x.shape[:-1], weight.shape[0])


@_frozen_linear_f32_op.register_fake
def _(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.float32)


def _f32_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[1])
    ctx.x_dtype = inputs[0].dtype


def _f32_backward(ctx, grad):
    (weight,) = ctx.saved_tensors
    # the input gradient in the compute dtype, as F.linear's backward of a BF16 output computes it
    dx = torch.nn.functional.linear(grad.to(ctx.x_dtype), weight.t().to(ctx.x_dtype))
    return dx, None


torch.library.register_autograd('natlang::frozen_linear_f32', _f32_backward, setup_context=_f32_setup)


# Frozen-base GEMM modes of ``factored_linear`` (``fp8_base`` takes a bool or one of these): 'bf16' (F.linear, BF16
# output), 'f32acc' (FP32 accumulator kept, rounded once after the low-rank delta; the lora+rows default), 'fp8'.
BASE_MODES = ('bf16', 'f32acc', 'fp8')


def factored_linear(x: torch.Tensor, base_w: torch.Tensor, lora_a: torch.Tensor | None = None,
                    lora_b: torch.Tensor | None = None, scale: float = 1.0, bias: torch.Tensor | None = None,
                    *, fp8_base=False, adapter_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Pure factored linear projection ``y = W0 x + s * B (A x)`` with canonical unscaled FP32 leaves ``A, B``.

    - ``scale`` (``alpha / rank``) is applied once as a Python float; leaves remain unscaled so multi-backward prefix
      trees never cache derived non-leaf tensors.
    - Without factors (or with ``A, B`` None) the base projection is ``F.linear`` in the input dtype, as before.
    - With factors and a frozen base, ``W0 x`` keeps its FP32 accumulator (``natlang::frozen_linear_f32``), the low-rank
      branch runs in the input's tensor-core dtype (BF16 on CUDA) and the two are added in FP32 and rounded once. A base
      weight that trains (full fine-tuning) keeps ``F.linear``. ``adapter_dtype=torch.float32`` (or
      ``NATLANG_LORA_FP32_BRANCH=1``) runs the strict FP32 low-rank reference branch.
    - ``fp8_base=True`` runs the frozen base GEMM in FP8 (``fp8_frozen_linear``; BF16 output, then the FP32 add).
    """
    import os
    mode = 'fp8' if fp8_base is True else 'bf16' if fp8_base is False or fp8_base is None else fp8_base
    if mode not in BASE_MODES:
        raise ValueError(f'factored_linear base mode {fp8_base!r}: one of {BASE_MODES} or a bool')
    factored = lora_a is not None and lora_b is not None
    f32acc = (mode == 'f32acc' or factored) and not base_w.requires_grad and x.dtype != torch.float32
    if mode == 'fp8':
        out = fp8_frozen_linear(x, base_w, bias)
    elif f32acc:
        out = _frozen_linear_f32_op(x, base_w)
        if bias is not None:
            out = out + bias.float()
        if not factored:
            return out.to(x.dtype)
    else:
        out = torch.nn.functional.linear(x, base_w, bias)
    if not factored:
        return out
    out = out.float()
    branch_dtype = adapter_dtype
    if branch_dtype is None:
        if x.dtype == torch.float32 or os.environ.get('NATLANG_LORA_FP32_BRANCH') == '1':
            branch_dtype = torch.float32
        else:
            branch_dtype = x.dtype
    if branch_dtype == torch.float32:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            low = torch.nn.functional.linear(
                torch.nn.functional.linear(x.float(), lora_a.float()),
                lora_b.float(),
            )
            return (out + float(scale) * low).to(x.dtype)
    h_low = torch.nn.functional.linear(x.to(branch_dtype), lora_a.to(branch_dtype))
    delta = torch.nn.functional.linear(h_low, lora_b.to(branch_dtype))
    return (out + float(scale) * delta.float()).to(x.dtype)


class FactoredLoRALinear(nn.Module):
    """A target Linear plus its low-rank delta: the base projection keeps its own native arithmetic
    (``shared_state._linear``), the delta ``(x A^T) B^T`` runs in FP32 and the sum is rounded once to the output dtype."""

    def __init__(self, original, factor_b, factor_a):
        super().__init__()
        # The original is not a registered submodule; its own parameter objects are registered under their own names,
        # so the model's parameter inventory (names and identities) is unchanged while the wrapper is installed.
        object.__setattr__(self, 'original', original)
        self.register_parameter('weight', original.weight)
        self.register_parameter('bias', original.bias)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.factor_b, self.factor_a = factor_b, factor_a

    def forward(self, x):
        from .shared_state import _linear, _linear_weight
        out = _linear(self.original, x)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            if invariant_lora_enabled():
                # The row-local fixed-tile kernel of the base projections (exact FP32 on CUDA): a row's delta is the
                # same whether it runs in a one-token decode, a short chunk or a whole prompt.
                low = _linear_weight(_linear_weight(x.to(self.factor_a.dtype), self.factor_a, force_invariant=True),
                                     self.factor_b.to(x.device), force_invariant=True)
            else:
                low = torch.nn.functional.linear(
                    torch.nn.functional.linear(x.to(self.factor_a.dtype), self.factor_a),
                    self.factor_b.to(x.device),
                )
            return (out.float() + low.float()).to(out.dtype)


def invariant_lora_enabled():
    """``NATLANG_INVARIANT_LORA`` (default on; ``0`` opts out): ``FactoredLoRALinear`` computes its low-rank delta with
    the batch-invariant FP32 kernel (``maple.fused_moe.invariant_linear``, IEEE FP32) instead of cuBLAS ``F.linear``.
    cuBLAS ran the FP32 factors in TF32 (``NATLANG_STOCK_ARITHMETIC`` defaults to tf32) with a reduction chosen by the
    row count, so the Python native engine's chunked prefill and one-token decode under an adapter stack drifted from
    the graph executor's whole-prompt pass (2026-11-26, compiled-site parity: 0.15 nats at the prefill's last position,
    greedy agreement 6/32; bit-equal with this kernel)."""
    import os
    return os.environ.get('NATLANG_INVARIANT_LORA', '1') != '0'


from contextlib import contextmanager


@contextmanager
def factored_lora_modules(model, factors):
    """Temporarily replace every Linear alias of each adapted target by its ``FactoredLoRALinear``."""
    parameters = dict(model.named_parameters())
    by_weight = {}
    for name, pair in factors.items():
        if name not in parameters:
            raise NativeAdapterError(f'factored adapter target {name} is not a model parameter')
        by_weight[id(parameters[name])] = (name, pair)
    changes, found, wrappers = [], set(), {}
    try:
        for name, module in list(model.named_modules(remove_duplicate=False)):
            if not name or id(getattr(module, 'weight', None)) not in by_weight or isinstance(module, FactoredLoRALinear):
                continue
            if not isinstance(module, nn.Linear):
                raise NativeAdapterError(f'factored adapter target {name} is not a Linear layer')
            target, (factor_b, factor_a) = by_weight[id(module.weight)]
            wrapper = wrappers.get(id(module))
            if wrapper is None:
                wrapper = wrappers[id(module)] = FactoredLoRALinear(module, factor_b, factor_a.to(module.weight.device))
            parent_name, _, child = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            changes.append((parent, child, module))
            found.add(target)
            parent._modules[child] = wrapper
        if found != set(factors):
            raise NativeAdapterError(f'factored adapter targets without a Linear: {sorted(set(factors) - found)[:3]}')
        yield
    finally:
        for parent, child, original in reversed(changes):
            parent._modules[child] = original


class bound_execution:
    """The native execution binding of `run_native` (effective identity, adapter binding) for work that runs the
    adapters elsewhere (the SGLang actor): resolves the entries and exposes them; substitutes no parameters."""

    def __init__(self, engine, entries):
        self.engine, self.entries = engine, entries

    def __enter__(self):
        try:
            self.resolved = resolve_adapters(self.engine, self.entries)
        except (ValueError, TypeError, KeyError) as error:
            raise NativeAdapterError(str(error)) from error
        identity = effective_identity(self.engine, self.resolved)
        binding = [{'id': ident, 'scale_hex': scale.hex()} for ident, scale, *_ in self.resolved]
        self.token = _ACTIVE.set((*_ACTIVE.get(), (self.engine, identity, binding, self.entries, set(), {}, 'storage',
                                                   {'active': True})))
        return self.resolved

    def __exit__(self, *exc):
        _ACTIVE.reset(self.token)
        return False


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


def capture_native_checkpoint_execution(callback):
    """Preserve actor, arithmetic and gradient-carrier context on recompute."""
    from ..common.native_higher_order import capture_reference_execution
    from .native_gradient_partitions import capture_gradient_partitions
    from .native_weight_carrier import capture_cotangent_scope
    return capture_reference_execution(capture_native_execution(
        capture_cotangent_scope(capture_gradient_partitions(callback))))


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
    deltas_by_key = {}
    for _, scale, spec, coefficients, bases in resolved:
        for name in spec['parameter_names']:
            weight = parameters[name]
            if spec['kind'] == 'free-lora':
                coefficient = (coefficients['lora_B.'+name], coefficients['lora_A.'+name])
                factor_b, factor_a = (coefficients[prefix+name].to(device=weight.device, dtype=torch.float32)
                                      for prefix in ('lora_B.', 'lora_A.'))
                delta = (factor_b @ factor_a) * (spec['alpha'] / spec['rank'] * scale)
            else:
                coefficient = coefficients[name]
                left, right = (bases[prefix+name].to(device=weight.device, dtype=torch.float32)
                               for prefix in ('left.', 'right.'))
                core = coefficient.to(device=weight.device, dtype=torch.float32).reshape(
                    *coefficient.shape[:-1], spec['rank'], spec['rank'])
                delta = ((left @ core) @ right.transpose(-2, -1)) * scale
            key = 'model.'+name
            deltas_by_key.setdefault(key, []).append((delta, coefficient))
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
    from .native_weight_carrier import active_cotangent_scope
    cotangents = active_cotangent_scope()
    if cotangents is not None:
        for key, value in replacement.items():
            cotangents.register_effective_weight(value, parameters[key.removeprefix('model.')],
                                                 deltas_by_key[key])
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


class InferenceReplacementCache:
    """A stack's served form for no-grad inference, reused across the tokens and calls of one adapter stack: its FP32
    low-rank factors (``inference_factors``, the default: applied beside the base projections as training applies them)
    or, when it cannot run factored, its effective merged weights.

    Adapters and bases are immutable, content-addressed artifacts, so a stack's resolved adapters and effective
    weights are the same on every decode round; rebuilding them per token cost about 26x a plain decode step
    (LFM2.5, 92 matrices, three stacked adapters). Entries are keyed by the exact adapter entries and the version
    counters of the base parameters, and bounded by the bytes of the effective weights they hold.
    """
    def __init__(self, budget_bytes: int):
        from collections import OrderedDict
        self.budget, self.bytes, self.hits, self.misses = int(budget_bytes), 0, 0, 0
        self.entries = OrderedDict()

    @staticmethod
    def key(entries, parameters):
        return (canonical_json_sha256_hex(entries), tuple(value._version for value in parameters.values()))

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return value[0]

    def put(self, key, value, size):
        if size > self.budget:
            return
        self.entries[key] = (value, size)
        self.bytes += size
        while self.bytes > self.budget:
            _, (_, old) = self.entries.popitem(last=False)
            self.bytes -= old


def inference_replacement_cache(engine):
    """The engine's cache (`native_adapter_cache_gb`, default 4; 0 disables it)."""
    cache = getattr(engine, 'native_adapter_cache', None)
    if cache is None:
        import os
        gb = float(getattr(engine, 'native_adapter_cache_gb', os.environ.get('NATLANG_ADAPTER_CACHE_GB', 4)))
        if gb <= 0:
            return None
        cache = engine.native_adapter_cache = InferenceReplacementCache(int(gb * 2**30))
    return cache


def run_native(engine, callback, entries=None, *, leaves=None, gradient_precision=None,
               track_adapter_cotangents=True, adapter_cotangent_target_ids=None,
               cotangent_parameter_ids=None):
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
        from ..common.native_higher_order import active_reference_arithmetic
        reference=active_reference_arithmetic()
        # Forward-only inference of an adapted actor: nothing is differentiated, so the stack's effective weights
        # are computed once and reused; no cotangent routing is needed (anchors only route gradients).
        cache = (inference_replacement_cache(engine) if entries and leaves is None and gradient_precision is None
                 and reference is None and not torch.is_grad_enabled() else None)
        cached = cache_key = None
        if cache is not None:
            parameters = target_parameters(engine.shared_model)
            cache_key = InferenceReplacementCache.key(entries, parameters)
            cached = cache.get(cache_key)
        if cached is not None:
            resolved, identity, binding, cached_replacement, cached_storage = cached
        else:
            try:
                resolved = resolve_adapters(engine, entries, leaves)
            except (ValueError, TypeError, KeyError) as error:
                raise NativeAdapterError(str(error)) from error
            identity = effective_identity(engine, resolved)
            binding = [{'id': ident, 'scale_hex': scale.hex()} for ident, scale, *_ in resolved]
        lifecycle = reference.lifetime if reference is not None else {'active': True}
        token = _ACTIVE.set((*_ACTIVE.get(), (engine, identity, binding, entries, set(), {}, gradient_precision or 'storage', lifecycle)))
        leaves_token = _LIVE_LEAVES.set(bool(leaves))
        try:
            if not resolved:
                # No autograd carrier is consumed in inference. Avoid scanning
                # every layer parameter/buffer on each unadapted decode round.
                # Grad-enabled VJPs and functional adapter substitution retain
                # their complete original routing below.
                if not torch.is_grad_enabled():
                    return callback()
                # A serving model is normally frozen, so retain the exact old
                # no-adapter fast path. A model-parameter VJP temporarily marks
                # its selected BF16 leaves trainable; those uses still need the
                # shared carrier even though no functional actor was installed.
                parameters=target_parameters(engine.shared_model)
                selected=[value for value in parameters.values()
                          if value.requires_grad and value.dtype in (torch.float16,torch.bfloat16)]
                if not selected:
                    return callback()
                from .native_weight_carrier import ParameterCotangentScope
                cotangent_scope=ParameterCotangentScope.for_model(engine.shared_model,mode='vjp',
                    track_auxiliary=track_adapter_cotangents,
                    auxiliary_target_ids=adapter_cotangent_target_ids,
                    parameter_target_ids=cotangent_parameter_ids)
                with cotangent_scope:
                    result=callback()
                    cotangent_scope.finish()
                    return result
            parameters = target_parameters(engine.shared_model)
            from .native_weight_carrier import ParameterCotangentScope, carrier_linear_modules
            if cache is not None:
                if cached is None:
                    served = inference_factors(resolved, parameters)
                    if served is not None:
                        cached_replacement, cached_storage = None, served
                        size = sum(value.numel() * value.element_size() for pair in served.values() for value in pair)
                    else:
                        cached_replacement, cached_storage = _effective_replacements(parameters, resolved, gradient_precision)
                        size = sum(value.numel() * value.element_size() for value in cached_replacement.values())
                    cache.put(cache_key, (resolved, identity, binding, cached_replacement, cached_storage), size)
                if cached_replacement is None:
                    # Factored serving: the base projections stay the model's own and each low-rank delta runs in FP32
                    # beside them, the arithmetic the training executor differentiates (`factored_factors`).
                    with factored_lora_modules(engine.shared_model, cached_storage):
                        return callback()
                _ACTIVE.get()[-1][5].update(cached_storage)
                _ACTIVE.get()[-1][4].update(id(value) for value in cached_replacement.values())
                selected = [parameters[key.removeprefix('model.')] for key in cached_replacement]
                with carrier_linear_modules(engine.shared_model, selected):
                    return torch.func.functional_call(_Operation(engine.shared_model, callback), cached_replacement, ())
            factors = factored_factors(resolved)
            if factors is None and not torch.is_grad_enabled() and reference is None:
                factors = inference_factors(resolved, parameters)
            if factors is not None:
                # The base weights stay the model's own: ordinary autograd carries the delta's gradient to the
                # adapter leaves and the base projections keep their cotangent routing.
                with factored_lora_modules(engine.shared_model, factors):
                    return callback()
            scope_parameters = [(name, parameter) for name, parameter in parameters.items()
                                if cotangent_parameter_ids is None or id(parameter) in cotangent_parameter_ids]
            from .native_weight_carrier import carrier_backend
            cotangent_scope = ParameterCotangentScope(scope_parameters, mode='vjp',
                track_auxiliary=track_adapter_cotangents,
                auxiliary_target_ids=adapter_cotangent_target_ids,
                parameter_target_ids=cotangent_parameter_ids,
                accumulator_backend=carrier_backend(engine.shared_model))
            cotangent_scope.model=engine.shared_model
            with cotangent_scope:
                replacement, storage_dtypes = _effective_replacements(parameters, resolved, gradient_precision)
                _ACTIVE.get()[-1][5].update(storage_dtypes)
                _ACTIVE.get()[-1][4].update(id(value) for value in replacement.values())
                selected = [parameters[key.removeprefix('model.')] for key in replacement]
                from ..common.native_higher_order import active_reference_arithmetic
                if active_reference_arithmetic() is not None:
                    result = torch.func.functional_call(_Operation(engine.shared_model, callback), replacement, ())
                else:
                    with carrier_linear_modules(engine.shared_model, selected):
                        result = torch.func.functional_call(_Operation(engine.shared_model, callback), replacement, ())
                cotangent_scope.finish()
                return result
        finally:
            if reference is None:lifecycle['active'] = False
            _LIVE_LEAVES.reset(leaves_token)
            _ACTIVE.reset(token)
            if resolved and cache is None:
                from .native_actor_graph import release_operation_cycles
                release_operation_cycles()


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
    from .native_weight_carrier import active_cotangent_scope
    scope = active_cotangent_scope()
    current = weight
    while current is not None:
        for frame in reversed(_ACTIVE.get()):
            dtype = frame[5].get(id(current))
            if dtype is not None:
                return dtype
        current = getattr(current, '_base', None)
    if scope is not None and scope.anchors_for_weight(weight):
        return weight.dtype
    return None


SAME_BASE_STATE_MODE = 'same-base-adapter-state'


def adapter_state_input_receipt(engine, block, state_id, *, target_position_start, layout=None):
    """Authenticate an explicitly requested raw-state exchange, never relabel it.

    Geometry is checked by the existing splice/graph executor before committing
    the new cache. This verifier binds the source coefficient closure and both
    actors to the exact same immutable base, independent of matching widths.
    `layout(block)` checks the raw profile against the consumer (default: the
    native model's port; the SGLang actor checks it against its export).
    """
    from .state_block import NativeStateBlock
    if type(block) is not NativeStateBlock:
        raise NativeAdapterError('same-base adapter-state mode requires raw native state')
    if layout is None:
        from .shared_state import validate_native_state_layout
        validate_native_state_layout(engine.shared_model.port, block)
    else:
        layout(block)
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
