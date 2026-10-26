---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ChangeWorkPackage[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for verifyConfiguration. Compare actual implemented configurations with approved applicability and acceptance criteria. Preserve authorized exceptions and unresolved field cases, then close the change only when required consumers and records reflect the correct revision state. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
