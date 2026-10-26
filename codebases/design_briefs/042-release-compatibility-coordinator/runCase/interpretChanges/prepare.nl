---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretChanges. Read release diffs, requirements, and observed behavior to identify changed semantics, defaults, timing, and error handling. Distinguish internal refactoring from externally meaningful changes even when schemas and type signatures remain compatible. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
