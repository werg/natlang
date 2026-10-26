---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for recoverSourceMeaning. Reconcile dictionaries, sample records, application notes, and domain-owner explanations. Identify fields whose meaning depends on record type or history, and distinguish null, not applicable, unknown, and deliberately withheld values before proposing transformations. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
