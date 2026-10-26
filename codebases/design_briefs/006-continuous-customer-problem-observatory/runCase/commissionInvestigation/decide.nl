---
args: { prepared: PreparedInvestigationBrief, goal: GoalContract, policy: PolicySelection }
returns: DecisionsInvestigationBrief
---
1. Apply this domain algorithm to each relevant evidence unit: Select unresolved explanations that would change a product decision. Formulate a bounded investigation with relevant cohorts, contradictory cases, and evidence needed to distinguish mechanisms rather than recommending the loudest requested feature.
2. Return field-level atoms for the actual InvestigationBrief record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
