---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ObservationInterpretation[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for detectDuplicates. Compare observer, time, location, media hashes, and narrative context to identify repeated submissions of one event. Preserve independent observations of the same phenomenon and avoid merging nearby events solely because they share a likely identity. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
