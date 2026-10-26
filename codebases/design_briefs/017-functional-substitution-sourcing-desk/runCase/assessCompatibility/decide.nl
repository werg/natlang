---
args: { prepared: PreparedCompatibilityFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsCompatibilityFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Compare every essential requirement against documented candidate evidence and approved conversion rules. Check interacting properties and interface tolerances; preserve unknowns explicitly and reject candidates whose apparent equivalence depends on an unsupported assumption.
2. Return field-level atoms for the actual CompatibilityFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
