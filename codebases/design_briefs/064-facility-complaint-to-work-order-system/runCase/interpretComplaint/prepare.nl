---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for interpretComplaint. Read occupant descriptions to identify location, symptom, timing, and practical impact. Distinguish observed behavior from guessed causes, resolve ambiguous room names, and preserve access or communication preferences without inferring sensitive personal circumstances. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
