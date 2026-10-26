---
args: { prepared: PreparedOpportunityEstimate, goal: GoalContract, policy: PolicySelection }
returns: DecisionsOpportunityEstimate
---
1. Apply this domain algorithm to each relevant evidence unit: Translate each intervention into bounded work packages, request estimates from authorized sources, and identify shared enabling work. Preserve cost ranges and unknowns rather than inventing precision; inspect whether dependencies make apparently independent projects coupled.
2. Return field-level atoms for the actual OpportunityEstimate record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
