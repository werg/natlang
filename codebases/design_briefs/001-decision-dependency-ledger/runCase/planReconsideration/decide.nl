---
args: { prepared: PreparedReviewPlan, goal: GoalContract, policy: PolicySelection }
returns: DecisionsReviewPlan
---
1. Apply this domain algorithm to each relevant evidence unit: Traverse only necessary dependencies first, then inspect supporting edges for materiality. Group reviews sharing an unresolved premise, formulate a decision-ready question, and avoid automatically cancelling downstream work that remains independently justified.
2. Return field-level atoms for the actual ReviewPlan record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
