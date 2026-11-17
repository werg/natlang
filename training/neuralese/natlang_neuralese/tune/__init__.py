"""Hierarchical program-adapter math (plans/HIERARCHICAL_LORA_TUNING.md §3.2, §5, §7).

Exact tensor mechanism only: shared nested bases, the two delta
parameterizations, lexical stack composition/export, family gates, the joint
objective's normalization and gauge-invariant delta similarity. Policy
(placement, family proposals, teacher routing) lives in natlang programs.
"""


class TuneMathError(ValueError):
    pass


def engine_dtype(config):
    """The frozen base's precision of a fit or apply engine: the plan's ``dtype``, else the checkpoint's saved dtype
    (``load_native_engine(dtype=None)``). LFM3 checkpoints are BF16; an FP32 read of one is a diagnostic cast. Trained
    adapter parameters are FP32 either way (``tune.parameterization``)."""
    import torch
    name = (config or {}).get('dtype')
    return getattr(torch, name) if name else None
