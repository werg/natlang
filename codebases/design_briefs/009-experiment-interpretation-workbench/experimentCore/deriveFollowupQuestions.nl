---
args: { outputs: ApprovedAnalysisOutput[], analyses: EvidencePage[], schemas: EvidencePage[], checks: EstimandCheck[], measurements: MetricMeasurementAudit[] }
returns: string[]
---
For each decision-sensitive uncertainty left by an approved, estimand-aligned output whose measurement window is valid, propose a follow-up only when the supplied analysis record and follow-up schema define an eligible population, time window, operation and completion test. Separate confirmatory new-data work from exploratory search. For a blocked or limited output, report the specific missing or invalid prerequisite; do not offer a confirmatory follow-up that would inherit the defect. If the source bundle omits any required design element, return the missing element as a question rather than inventing a runnable analysis.
