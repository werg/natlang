---
args: { prepared: PreparedAmendedObligation, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAmendedObligation
---
1. Apply this domain algorithm to each relevant evidence unit: Construct the amended obligation by applying the specific operation to its target. Preserve untouched exceptions, prerequisites, definitions, and cross-references; compare the result against counterexamples that would expose accidental broadening or deletion.
2. Return field-level atoms for the actual AmendedObligation record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
