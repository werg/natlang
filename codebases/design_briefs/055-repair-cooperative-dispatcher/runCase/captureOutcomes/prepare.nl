---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: RepairWorkPlan[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for captureOutcomes. Compare post-work checks and customer observations with the original symptoms. Record partial success, recurrence, and specialist referrals, then extract scoped lessons that improve future routing without treating one repair as universal proof of a cause. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
