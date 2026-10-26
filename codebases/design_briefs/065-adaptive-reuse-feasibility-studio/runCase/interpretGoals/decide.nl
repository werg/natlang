---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Translate stakeholder objectives into required functions and explicit tradeoffs. Separate mandatory supplied constraints from preferences, identify incompatible goals, and avoid choosing political or commercial priorities not authorized by the project brief.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
