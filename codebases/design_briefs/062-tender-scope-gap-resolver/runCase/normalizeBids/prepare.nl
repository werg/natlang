---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for normalizeBids. Read each bid, qualification, schedule, and addendum as a package. Extract functional scope rather than relying on trade headings, distinguish allowances from firm inclusions, and retain conditional exclusions that affect interfaces with other bidders. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
