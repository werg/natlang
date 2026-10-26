---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IdentityCluster[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructCanonical. Assemble a canonical view without overwriting source records or selecting one source globally for every field. Resolve attributes using field-specific authority and effective time, preserve aliases and conflicting claims, and separate historical from current values. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
