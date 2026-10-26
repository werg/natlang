---
args: { output: ApprovedAnalysisOutput, design: EffectiveDesignTimeline[], measurements: MetricMeasurementAudit[], approvals: EvidencePage[], searches: EvidencePage[] }
returns: EstimandCheck
---
Bind provenance = verifyOutputProvenance(output, design, approvals), assignmentOutcome = compareAssignmentOutcome(output, design, measurements), and subgroupSearch = auditSubgroupSearchHistory(output, searches). Construct issues from the actual child mismatches, including design/version approval, assignment and observation units, population, outcome window, limited/invalid measurement windows, uncertainty and missing subgroup searches. Preserve the engine-produced estimate and uncertainty without recalculation. Return one EstimandCheck carrying all three child reviews and their exact evidence IDs; a passed numerical result cannot cure a provenance or assignment mismatch.
