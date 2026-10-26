---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Read artifacts, reflections, and supervisor comments to identify observable work and its context. Separate the learner’s own contribution from group output, distinguish direct evidence from self-report, and preserve uncertainty about authorship or assistance.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
