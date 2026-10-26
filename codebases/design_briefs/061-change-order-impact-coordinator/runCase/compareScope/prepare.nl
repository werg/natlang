---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: SiteInstruction[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for compareScope. Compare the instruction with the agreed baseline and approved prior changes. Identify additions, omissions, substitutions, or clarifications, preserve exclusions, and avoid treating a contractor’s preferred method as contractual scope unless the baseline supports it. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
