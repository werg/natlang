---
args: { prepared: PreparedExpenseQuestion, goal: GoalContract, policy: PolicySelection }
returns: DecisionsExpenseQuestion
---
1. Apply this domain algorithm to each relevant evidence unit: Group unresolved questions to minimize claimant burden while keeping each decision testable. Ask only for evidence necessary to determine eligibility, explain the applicable rule, and route privacy-sensitive material through restricted upload channels.
2. Return field-level atoms for the actual ExpenseQuestion record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
