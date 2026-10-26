---
args: { prepared: PreparedExpenseFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsExpenseFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Apply the policy edition effective for the expense and claimant context. Preserve category-specific limits, meal inclusions, and approval conditions; distinguish a documented exception from a persuasive but unauthorized justification.
2. Return field-level atoms for the actual ExpenseFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
