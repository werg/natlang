---
args: { prepared: PreparedConceptAlignment, goal: GoalContract, policy: PolicySelection }
returns: DecisionsConceptAlignment
---
1. Apply this domain algorithm to each relevant evidence unit: Compare operational meanings and counterexamples for candidate terms. Establish equivalence only when obligations and consequences align; represent partial overlap explicitly and retain organization-specific variants where unification would lose behavior.
2. Return field-level atoms for the actual ConceptAlignment record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
