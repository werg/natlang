---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for recoverProcedure. Parse the master instruction into actions, guards, prerequisites, and verification obligations. Distinguish explanatory advice from mandatory controls, preserve warnings in their operative context, and identify cross-references needed to execute the procedure correctly. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
