---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare scorer disagreements and appeals against the released rubric and actual response. Identify systematic rubric defects, propose scoped corrections, and apply approved changes consistently to all affected learners rather than only those who appealed.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
