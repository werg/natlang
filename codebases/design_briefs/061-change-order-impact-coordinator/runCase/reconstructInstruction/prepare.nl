---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for reconstructInstruction. Reconcile site notes, messages, and drawing annotations to identify the actual requested action and its issuer. Distinguish clarification, proposed change, and authorized instruction; preserve conflicting versions and do not infer commercial authority from site presence. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
