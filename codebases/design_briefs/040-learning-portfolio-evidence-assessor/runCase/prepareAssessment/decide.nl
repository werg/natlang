---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Assemble a reviewable recommendation with evidence, gaps, and proposed next work. Allow the learner to correct factual attribution, preserve assessor disagreements, and record the authorized decision separately from the program’s recommendation.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
