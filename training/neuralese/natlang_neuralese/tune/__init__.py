"""Hierarchical program-adapter math (plans/HIERARCHICAL_LORA_TUNING.md §3.2, §5, §7).

Exact tensor mechanism only: shared nested bases, the two delta
parameterizations, lexical stack composition/export, family gates, the joint
objective's normalization and gauge-invariant delta similarity. Policy
(placement, family proposals, teacher routing) lives in natlang programs.
"""


class TuneMathError(ValueError):
    pass
