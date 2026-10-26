---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Compare incoming records with category definitions and examples, not names alone. Distinguish a genuinely new concept from a poorly worded existing category, and retain ambiguous membership where the taxonomy permits multiple dimensions.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
