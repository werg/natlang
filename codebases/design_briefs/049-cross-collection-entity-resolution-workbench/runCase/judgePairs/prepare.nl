---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IdentityCandidate[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for judgePairs. Compare names, attributes, chronology, and context while distinguishing identity from affiliation or succession. Treat missing attributes as unknown rather than matching, and record both positive support and hard cannot-link evidence. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
