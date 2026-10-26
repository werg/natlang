---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RecoveryStep[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for verifyRestoration. Recheck the affected user-visible functions and representative cohorts after recovery actions. Distinguish reduced alerts from actual restoration, retain residual degradation, and schedule observation long enough to detect recurrence under the supplied incident policy. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
