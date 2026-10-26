---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CategoryRevision[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for simulateReclassification. Reclassify only potentially affected records under baseline and candidate definitions, compare changes, and inspect boundary cases. Build explicit bridges for historical reporting and retain cases that cannot be mapped uniquely after a split. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
