---
args: { output: ApprovedAnalysisOutput, design: EffectiveDesignTimeline[], measurements: MetricMeasurementAudit[] }
returns: AssignmentOutcomeReview
---
Bind units = compareUnitDeclarations(output, design). Use the child comparison to check randomization/assignment unit separately from observation unit. Match the output population, outcome definition and observation window to the approved timeline and its metric windows. Carry invalid or limited measurement audits into this result. Return exact mismatches and the window identities that qualify the estimate.
