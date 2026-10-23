"""Functional token-local Qwen3.5 Gated DeltaNet transitions.

Native state stores only the producer projections before causal convolution.
Consumers replay those projections through their own convolution and recurrent
state, matching the existing native finite-convolution import contract. The
projection payload preserves producer hidden-state influence; imports transfer
no producer convolution window or terminal recurrent snapshot. Consumer tokens
advance consumer-local convolution and recurrence in logical order. Producer
private prefixes therefore remain absent as explicit state transfer, while
their influence on the producer's projected token values is retained.

For imported rows, the current format still carries every projection field.
Consumer recurrence uses imported keys, values, beta, and decay; imported
queries are not used for consumer-owned outputs, and imported ``z`` gates only
the readout of its own token. Native replay projects readouts only for
consumer-owned rows, so imported query and ``z`` have no direct consumer-loss
gradient. Their producer projections can still affect later producer-layer
exports through the producer's own hidden states. A smaller payload that drops
these unused imported-readout channels would need a separately versioned
schema.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


PAYLOAD_FIELDS = ('mixed_qkv', 'z', 'b', 'a')


def payload_widths(module):
    """Projected-channel widths in the one canonical token-local payload order."""
    return dict(zip(PAYLOAD_FIELDS, (module.conv_dim,
        module.num_v_heads * module.head_v_dim, module.num_v_heads, module.num_v_heads)))


def project_inputs(module, hidden, linear):
    """Project a token span into the canonical, pre-convolution payload."""
    return tuple(linear(getattr(module, name), hidden) for name in
                 ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a'))


def replay(module, payload, *, conv_prefix=None, recurrent_state=None, linear=None, output_positions=None,
           output_gradient_partitions=None):
    """Run a logical sequence through consumer-local convolution and recurrence.

    ``payload`` has one tensor per :data:`PAYLOAD_FIELDS`, each token first.
    Returns the normalized/output-projected mixer output and the final
    convolution/recurrent states. The upstream chunk rule supplies the same
    q/k normalization and gated-delta recurrence used by Qwen3.5.
    """
    if set(payload) != set(PAYLOAD_FIELDS):
        raise ValueError('gated delta payload has an invalid projected-input inventory')
    values = tuple(payload[name] for name in PAYLOAD_FIELDS)
    if any(value.ndim not in (2, 3) for value in values):
        raise ValueError('gated delta projected inputs must be [T,C] or [B,T,C]')
    if any(value.ndim != values[0].ndim for value in values):
        raise ValueError('gated delta projected inputs must share a batch layout')
    mixed, z, b, a = values if values[0].ndim == 3 else tuple(value[None] for value in values)
    if mixed.shape[1] == 0:
        return mixed.new_empty((mixed.shape[0], 0, module.hidden_size)), conv_prefix, recurrent_state
    conv = module.conv1d
    channels = module.conv_dim
    if mixed.shape[-1] != channels:
        raise ValueError('gated delta projected qkv width differs from its convolution')
    prefix = (mixed.new_zeros((mixed.shape[0], channels, module.conv_kernel_size - 1))
              if conv_prefix is None else conv_prefix)
    if prefix.shape != (mixed.shape[0], channels, module.conv_kernel_size - 1):
        raise ValueError('gated delta convolution prefix has an incompatible shape')
    joined = torch.cat((prefix, mixed.transpose(1, 2)), dim=2)
    from .native_weight_carrier import stored_conv1d
    convolved = stored_conv1d(joined, conv.weight, conv.bias, groups=channels)
    convolved = F.silu(convolved).transpose(1, 2)
    tail = module.conv_kernel_size - 1
    conv_state = joined[:, :, -tail:] if tail else joined[:, :, :0]

    query, key, value = torch.split(convolved, (module.key_dim, module.key_dim, module.value_dim), dim=-1)
    batch, steps, _ = query.shape
    query = query.view(batch, steps, -1, module.head_k_dim)
    key = key.view(batch, steps, -1, module.head_k_dim)
    value = value.view(batch, steps, -1, module.head_v_dim)
    beta = b.sigmoid()
    from .native_weight_carrier import widen_direct_parameter
    A_log = widen_direct_parameter(module.A_log)
    dt_bias = widen_direct_parameter(module.dt_bias)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    if module.num_v_heads // module.num_k_heads > 1:
        repeat = module.num_v_heads // module.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
    # FLA's fused recurrent path is inference-only: it does not provide a
    # backward. HF's torch recurrent wrapper may select that kernel when GPU
    # training kernels are enabled, so a one-token differentiable replay must
    # still use the chunk rule. Keep the recurrent path for no-grad decode.
    recurrence = (torch_recurrent_gated_delta_rule
                  if steps == 1 and not torch.is_grad_enabled()
                  else torch_chunk_gated_delta_rule)
    core, final = recurrence(
        query, key, value, g=g, beta=beta, initial_state=recurrent_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    core = core.reshape(batch, steps, -1, module.head_v_dim)
    z = z.reshape(batch, steps, -1, module.head_v_dim)
    if output_positions is not None:
        selected = torch.as_tensor(output_positions, dtype=torch.long, device=core.device)
        if selected.numel() == 0:
            return core.new_empty((batch, 0, module.hidden_size)), conv_state, final
        core = core.index_select(1, selected)
        z = z.index_select(1, selected)
        steps = selected.numel()
    if output_gradient_partitions is None:
        core = module.norm(core, z).reshape(batch, steps, -1)
        output = module.out_proj(core) if linear is None else linear(module.out_proj, core)
    else:
        from .native_gradient_partitions import gradient_partitions
        with gradient_partitions(output_gradient_partitions):
            core = module.norm(core, z).reshape(batch, steps, -1)
            output = module.out_proj(core) if linear is None else linear(module.out_proj, core)
    return output, conv_state, final
