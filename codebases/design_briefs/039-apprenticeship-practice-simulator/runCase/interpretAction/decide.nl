---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Interpret the learner’s natural-language response against the available action set. Distinguish a proposed action from an executed one, ask for clarification when ambiguity affects consequences, and avoid rewarding jargon without the corresponding operational behavior.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
