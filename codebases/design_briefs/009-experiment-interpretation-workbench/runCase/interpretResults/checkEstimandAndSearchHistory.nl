---
args: { output: ApprovedAnalysisOutput, design: EffectiveDesignTimeline[], measurements: MetricMeasurementAudit[], approvals: EvidencePage[], searches: EvidencePage[] }
returns: EstimandCheck
---
Verify the output's design version, population, assignment unit, outcome definition, and window against the effective plan. Confirm approval reference and snapshot against supplied approval evidence. Compare its outcome window with metric audit windows and flag outputs relying on invalid or limited instrumentation. Preserve the engine-produced estimate and uncertainty without recalculation. Record multiplicity handling and all subgroup searches; a post-hoc subgroup is exploratory even when the estimate looks unusual.
