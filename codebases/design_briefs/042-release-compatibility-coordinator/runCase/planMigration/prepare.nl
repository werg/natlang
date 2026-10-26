---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: CompatibilityTest[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for planMigration. Construct staged migration plans with prerequisites, owners, compatibility shims where approved, and rollback conditions. Group consumers by meaningful risk and avoid advancing a wave before its required operational readiness evidence exists. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
