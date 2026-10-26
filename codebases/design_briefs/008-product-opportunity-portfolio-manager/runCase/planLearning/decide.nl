---
args: { prepared: PreparedOpportunityExperiment, goal: GoalContract, policy: PolicySelection }
returns: DecisionsOpportunityExperiment
---
1. Apply this domain algorithm to each relevant evidence unit: Choose the cheapest credible experiment capable of changing a portfolio decision. State how positive, negative, and inconclusive results would alter the selected work, and schedule learning before irreversible dependent commitments wherever feasible.
2. Return field-level atoms for the actual OpportunityExperiment record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
