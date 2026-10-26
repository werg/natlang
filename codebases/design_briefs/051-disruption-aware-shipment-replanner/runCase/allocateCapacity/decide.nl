---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Submit candidate plans to exact capacity and timing services, then compare feasible allocations under supplied priorities. Reserve selected capacity before promising it, retain unfulfilled demand explicitly, and detect conflicts across shipments sharing scarce resources.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
