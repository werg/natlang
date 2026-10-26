---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: HandoverEvidence[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for planClosure. Group gaps into actionable requests for contractors, inspectors, or facilities staff. Prioritize dependencies affecting safe or practical use under supplied rules, and preserve accepted temporary arrangements with owners and expiry rather than hiding them as complete. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
