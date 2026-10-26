---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare descriptions, packaging, serials, lot information, and movement history to identify candidate stock. Distinguish a mislabeled item from a misplaced one, and request a discriminating scan or inspection rather than choosing the closest description.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
