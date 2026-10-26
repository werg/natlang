---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: FacilityObservation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for linkRelatedIssues. Compare reports by location, asset dependencies, timing, and symptom mechanism. Group likely shared issues for coordinated investigation while retaining unrelated complaints that use similar language; do not close a new report solely because a nearby task was completed. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
