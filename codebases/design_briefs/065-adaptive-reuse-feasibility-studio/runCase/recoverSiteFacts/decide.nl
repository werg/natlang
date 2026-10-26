---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Reconcile inventories, surveys, historical plans, and stakeholder descriptions into verified facts and unknowns. Distinguish current observations from outdated drawings and preserve uncertainties that could materially change feasibility or cost.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
