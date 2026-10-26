---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Generate a bounded set of causal hypotheses from deployment history, dependencies, and symptoms. Track evidence for and against each, choose checks that distinguish mechanisms, and avoid repeatedly confirming the most recent or most fluent explanation.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
