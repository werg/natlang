---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for indexEvidence. Read artifacts, reflections, and supervisor comments to identify observable work and its context. Separate the learner’s own contribution from group output, distinguish direct evidence from self-report, and preserve uncertainty about authorship or assistance. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
