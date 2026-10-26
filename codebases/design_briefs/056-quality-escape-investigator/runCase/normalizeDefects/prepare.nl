---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for normalizeDefects. Reconcile customer reports, operator notes, and inspection descriptions into comparable observed defects. Preserve measurement conditions and terminology differences, distinguish symptoms from suspected causes, and retain reports that may describe a different mechanism. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
