---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Inspect the learner’s response against the task and accepted solution space. Distinguish conceptual errors, arithmetic slips, misunderstood wording, and incomplete work; preserve alternative correct strategies rather than forcing one canonical explanation.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
