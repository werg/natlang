---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ChangeOption[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for assembleChange. Prepare a coordinated scope, cost, timing, and responsibility package with the relevant evidence. Separate agreed changes from disputed claims, bind approvals to the exact revision, and update downstream plans only after the appropriate authorities accept the change. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
