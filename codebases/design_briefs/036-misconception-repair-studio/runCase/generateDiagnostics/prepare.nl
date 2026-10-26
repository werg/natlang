---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MisconceptionHypothesis[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for generateDiagnostics. Construct a short diagnostic task varying the relevant conceptual feature while controlling unrelated difficulty. Include an independently checked answer space and accessible wording; avoid hints that reveal which hypothesis the system expects to confirm. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
