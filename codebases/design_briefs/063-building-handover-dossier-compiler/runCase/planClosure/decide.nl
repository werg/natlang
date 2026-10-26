---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Group gaps into actionable requests for contractors, inspectors, or facilities staff. Prioritize dependencies affecting safe or practical use under supplied rules, and preserve accepted temporary arrangements with owners and expiry rather than hiding them as complete.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
