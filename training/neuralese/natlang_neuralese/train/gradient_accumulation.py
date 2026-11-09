"""FP32 gradient accumulation across the micro-batches of one native optimizer update.

A BF16 parameter's ``.grad`` is BF16 by default, so an update of ``accumulate`` micro-batches rounded the running sum
to BF16 after every backward (16 roundings in production RL, each of the size of the update signal at fine-tuning
rates; plans/neuralese/RL_REGRESSION_DIAGNOSIS_20261109.md, ops/diag-sanity-1109). Torch 2.11's ``Tensor.grad_dtype``
makes autograd accumulate into an FP32 ``.grad`` instead: each micro-batch's gradient arrives in the dtype its
backward produced (a vendor GEMM's single BF16 rounding in the ``plain`` backend; the carrier's unrounded FP32
cotangent in the ``vjp`` backends) and is added to the FP32 sum without further rounding. The final consumer (the
gated MuonSR hook, AdamWKahan/AdamWSR) reads the FP32 total; a parameter whose optimizer needs a parameter-dtype
gradient gets the total cast once, here, before the step.

Used only when an update has more than one micro-batch: the FP32 sums cost 4 bytes per trainable low-precision
parameter (``allocation_report``) where the BF16 sums cost 2."""
from __future__ import annotations

import torch

# Recorded in checkpoint options (``gradient_accumulation_dtype``); BF16 before 2026-11-09.
GRADIENT_ACCUMULATION_DTYPE = 'float32'
LOW_PRECISION = (torch.bfloat16, torch.float16)


def float32_gradient_parameters(model):
    """Trainable low-precision parameters (each once) whose accumulated gradient is held in FP32."""
    seen, result = set(), []
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype in LOW_PRECISION and id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def float32_gradient_bytes(model):
    """Peak bytes of the FP32 accumulators of a multi-micro-batch update (every parameter's sum is live at the
    final micro-batch's backward)."""
    return sum(parameter.numel() * 4 for parameter in float32_gradient_parameters(model))


def _consumers(optimizer):
    children = ((getattr(optimizer, 'muon', None), getattr(optimizer, 'auxiliary', None),
                 getattr(optimizer, 'latent', None)) if hasattr(optimizer, 'muon') else (optimizer,))
    accepted = set()
    for child in children:
        if child is not None and getattr(child, 'accepts_float32_gradients', False):
            accepted.update(id(p) for group in child.param_groups for p in group['params'])
    return accepted


class Float32GradientAccumulation:
    """Context of one multi-micro-batch update: trainable low-precision parameters accumulate ``.grad`` in FP32.

    ``finish()`` (after the last backward) casts the FP32 total once for parameters whose optimizer needs a
    parameter-dtype gradient. Leaving the context (success or failure) drops every remaining FP32 gradient and
    restores the parameters' default gradient dtype; an update starts and ends at a gradient-free boundary."""

    def __init__(self, model, optimizer):
        self.parameters = float32_gradient_parameters(model)
        self.accepted = _consumers(optimizer)
        self._entered = []

    def __enter__(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                raise RuntimeError('FP32 gradient accumulation needs an empty gradient boundary')
        for parameter in self.parameters:
            parameter.grad_dtype = torch.float32
            self._entered.append(parameter)
        return self

    def finish(self):
        """Cast the exact FP32 total once, for parameters whose optimizer needs a parameter-dtype gradient."""
        for parameter in self.parameters:
            total = parameter.grad
            if total is None or id(parameter) in self.accepted:
                continue
            parameter.grad = None
            parameter.grad_dtype = parameter.dtype
            parameter.grad = total.to(parameter.dtype)

    def allocation_report(self):
        held = sum(p.grad.numel() * 4 for p in self.parameters if p.grad is not None and p.grad.dtype == torch.float32)
        return {'schema': 'natlang.fp32-gradient-accumulation/1', 'dtype': GRADIENT_ACCUMULATION_DTYPE,
                'parameters': len(self.parameters), 'peak_bytes': sum(p.numel() * 4 for p in self.parameters),
                'live_bytes': held}

    def __exit__(self, exc_type, exc, traceback):
        for parameter in self._entered:
            if parameter.grad is not None and parameter.grad.dtype != parameter.dtype:
                parameter.grad = None
            parameter.grad_dtype = parameter.dtype
        self._entered.clear()
        return False
