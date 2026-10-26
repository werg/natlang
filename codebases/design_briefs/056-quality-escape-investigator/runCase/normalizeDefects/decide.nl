---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile customer reports, operator notes, and inspection descriptions into comparable observed defects. Preserve measurement conditions and terminology differences, distinguish symptoms from suspected causes, and retain reports that may describe a different mechanism.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
