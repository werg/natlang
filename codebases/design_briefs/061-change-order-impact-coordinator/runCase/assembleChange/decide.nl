---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Prepare a coordinated scope, cost, timing, and responsibility package with the relevant evidence. Separate agreed changes from disputed claims, bind approvals to the exact revision, and update downstream plans only after the appropriate authorities accept the change.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
