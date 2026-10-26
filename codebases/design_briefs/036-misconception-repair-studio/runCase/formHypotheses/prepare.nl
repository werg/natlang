---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ResponseObservation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for formHypotheses. Generate a small set of plausible explanations for the observed pattern, including task ambiguity and accidental error. Identify a question whose answers would distinguish these explanations instead of treating one wrong response as a stable misconception. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
