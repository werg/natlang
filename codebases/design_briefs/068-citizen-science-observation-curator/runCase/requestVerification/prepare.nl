---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ObservationLink[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for requestVerification. Form targeted follow-up requests that could distinguish plausible interpretations or correct metadata. Use respectful wording, avoid leading the observer toward the expected label, and route specialist-sensitive cases for expert review. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
