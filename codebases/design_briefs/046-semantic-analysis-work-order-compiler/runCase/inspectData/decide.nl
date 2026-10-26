---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Inspect schemas, dictionaries, representative records, and access constraints to determine whether the question is answerable. Distinguish absent data from inaccessible data and identify proxy fields that would change the meaning rather than silently substituting them.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
