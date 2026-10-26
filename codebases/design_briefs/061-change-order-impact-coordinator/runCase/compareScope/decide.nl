---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare the instruction with the agreed baseline and approved prior changes. Identify additions, omissions, substitutions, or clarifications, preserve exclusions, and avoid treating a contractor’s preferred method as contractual scope unless the baseline supports it.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
