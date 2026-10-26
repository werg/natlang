---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ReportingBoundary[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for mapMeasurements. Reconcile measurements, invoices, contractor statements, and field records with their actual methods and periods. Preserve estimates versus direct measurements, identify missing coverage, and retain original units and uncertainty for exact conversion services. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
