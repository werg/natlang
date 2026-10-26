---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: PortfolioEvidence[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for mapCompetencies. Compare evidence with the actual competency definition and performance level. Recognize alternative demonstrations, avoid matching by keywords alone, and identify which facets are supported versus merely mentioned in a reflection. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
