---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DatasetRecordAudit[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for measureCoverage. Compute exact slice coverage and overlap using the approved population definitions. Interpret whether apparent balance masks duplicated sources or missing mechanisms, and identify the most consequential acquisition gaps rather than optimizing superficial counts. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
