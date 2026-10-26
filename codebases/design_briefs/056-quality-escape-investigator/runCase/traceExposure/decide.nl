---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Traverse exact lot, material, and process lineage to identify confirmed and precautionary exposure. Interpret process changes and rework histories, and avoid narrowing the scope merely because records are missing for some potentially affected units.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
