---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DebtSymptom[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for formMechanisms. Group symptoms by plausible shared mechanisms such as unstable interfaces or manual coordination. Distinguish architectural causes from staffing or process issues, and formulate checks that could disconfirm the proposed technical explanation. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
